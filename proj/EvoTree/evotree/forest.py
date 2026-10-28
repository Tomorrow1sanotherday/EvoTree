"""经验森林：EvoTree 的本体层（idea §4.3 创新点）。

核心对象：一片"经验森林"—— 每个任务模板 τ 一棵树，节点是状态、边是动作，
边上挂着从真实执行历史统计出来的成功率与剩余步数。

    ExperienceTree   : 单个任务模板的一棵树（状态节点 + 动作边）
    ExperienceForest : 管理所有树，按 template_id 分册

三条"自动成立"的性质（结构白送，非额外机制，对应 idea §4.2）：
    1. 同一 (from, action) 的多次执行落进同一条边  —— 自动归堆
    2. 不同来路到达同一局面自动汇合（状态按主键索引）—— 这就是"森林是图"
    3. 失败正常入账（失败落点也是状态，有自己的出边统计）

为什么叫"树/森林"却允许汇合（对应 idea D6）：每棵从任务根向下展开、决策是
自顶向下的路径选择，"树"是最贴切的心智模型；内部汇合是实现事实，严格说是
"按任务分册的 DAG"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from .types import ActionEdge, ActionKind, ActionTmpl, StateNode, StateSig


class ExperienceTree:
    """单个任务模板 τ 的经验树。

    内部用两个索引维护"点"和"边"：
        - _nodes: state_key -> StateNode          （状态按主键归堆/汇合）
        - _edges: edge_key  -> ActionEdge          （动作按主键归堆）

    node_id 是人类可读的稳定标识（s0, s1, ...），与 state_key 一一对应；
    边用 node_id 记 from/to，便于 JSON 里引用和调试。
    """

    def __init__(self, template_id: str) -> None:
        self.template_id = template_id
        self._nodes: dict[tuple[str, str, str], StateNode] = {}
        self._id_to_node: dict[str, StateNode] = {}
        self._edges: dict[tuple[str, tuple[str, str, str]], ActionEdge] = {}
        self._next_node_num = 0

    # ------------------------------------------------------------------ #
    # 状态节点：找或建（不同来路到达同一局面自动汇合）
    # ------------------------------------------------------------------ #
    def get_or_create_node(self, sig: StateSig) -> StateNode:
        """按状态签名找节点；不存在则新建。

        身份由 sig.key()（= app+window+dialog，不含 facts）决定，
        所以不同来路、不同 facts 但主键相同的观测会汇合到同一节点。
        """
        skey = sig.key()
        node = self._nodes.get(skey)
        if node is None:
            node = StateNode(node_id=f"s{self._next_node_num}", sig=sig)
            self._next_node_num += 1
            self._nodes[skey] = node
            self._id_to_node[node.node_id] = node
        return node

    def node_by_id(self, node_id: str) -> Optional[StateNode]:
        """按 node_id 取节点（前瞻时顺着 to_success 指针跳转用）。"""
        return self._id_to_node.get(node_id)

    def node_for_signature(self, sig: StateSig) -> Optional[StateNode]:
        """按状态签名查已有节点；找不到时返回 None，不创建节点。

        检索阶段只允许读取已有经验。把"查找"和"创建"分成两个公开方法，
        避免检索意外污染森林，也避免上层模块直接访问私有索引。
        """
        return self._nodes.get(sig.key())

    # ------------------------------------------------------------------ #
    # 动作边：找或建（同一动作自动归堆）
    # ------------------------------------------------------------------ #
    def get_or_create_edge(self, from_id: str, action: ActionTmpl) -> ActionEdge:
        """按 (起点节点, 动作) 找边；不存在则新建。"""
        ekey = (from_id, action.key())
        edge = self._edges.get(ekey)
        if edge is None:
            edge = ActionEdge(from_id=from_id, action=action)
            self._edges[ekey] = edge
        return edge

    def out_edges(self, node_id: str) -> list[ActionEdge]:
        """某状态的全部出边（检索时对这些边打分选边）。"""
        return [e for e in self._edges.values() if e.from_id == node_id]

    def all_edges(self) -> Iterator[ActionEdge]:
        """遍历所有边（维护/分析用）。"""
        return iter(self._edges.values())

    def all_nodes(self) -> Iterator[StateNode]:
        """Read-only traversal for horizontal, cross-task state retrieval."""
        return iter(self._id_to_node.values())

    # ------------------------------------------------------------------ #
    # 写入一步：四步 upsert 中的"逐步入账"（idea §0 写入第 3 步）
    # ------------------------------------------------------------------ #
    def upsert_step(
        self,
        from_sig: StateSig,
        action: ActionTmpl,
        success: bool,
        to_sig: Optional[StateSig] = None,
        note: Optional[str] = None,
    ) -> ActionEdge:
        """写入一步执行：更新起点状态、对应边的计数、成败落点。

        Args:
            from_sig: 执行动作前的状态签名。
            action:   执行的动作模板 α。
            success:  这一步是否成功（决定落点记 to_success 还是 to_fail）。
            to_sig:   执行后到达的状态签名（用于连边）；None 表示未知/终止。
            note:     这一步的坑位笔记（如工具报错），追加进边的 notes。

        Returns:
            被更新的 ActionEdge（便于调用方继续记 steps_after_success）。
        """
        from_node = self.get_or_create_node(from_sig)
        from_node.visits += 1

        edge = self.get_or_create_edge(from_node.node_id, action)
        edge.n += 1
        if success:
            edge.n_success += 1

        # 语义取值记在边上（不进主键，所以不影响上面的归堆）。
        # 边按 (from, α主键) 归堆，同一条边会先后收到不同取值 —— 那正是
        # "哪种写法通了、哪种没通"的原始材料，全部留下。
        if action.args:
            edge.add_arg_sample(dict(action.args), success=success)

        # 连边：把落点状态挂到 to_success / to_fail（"连起来"的指针）
        if to_sig is not None:
            to_node = self.get_or_create_node(to_sig)
            if success:
                edge.to_success = to_node.node_id
            else:
                edge.to_fail = to_node.node_id

        if note:
            edge.notes.append(note)

        return edge

    # ------------------------------------------------------------------ #
    # cost-to-go：本轮成功后从终点倒数（idea §0 写入第 4 步，D3 = MC 平均）
    # ------------------------------------------------------------------ #
    def record_steps_after_success(self, success_path: list[ActionEdge]) -> None:
        """给一条成功轨迹上的每条边追加"这步之后还剩几步"。

        从最后一步倒着数：末步记 1、倒数第二步记 2……
        这是 cost-to-go 的全部计算 —— 一个倒序循环（Monte Carlo，非 Bellman 传播）。

        Args:
            success_path: 本轮成功轨迹按执行顺序排列的边列表。
        """
        total = len(success_path)
        for i, edge in enumerate(success_path):
            steps_remaining = total - i  # 第 i 步之后（含自己）到终点的步数
            edge.steps_after_success.append(steps_remaining)

    # ------------------------------------------------------------------ #
    # 维护：删低访问低成功边（绝不"只留最短路径"，idea §4.2）
    # ------------------------------------------------------------------ #
    def prune(self, min_visits: int = 2, max_success_rate: float = 0.2) -> int:
        """删除长期低访问且低成功率的边。

        **绝不"只留最短路径"**：2 步但成功率 0.3 的路不该挤掉 4 步但成功率
        0.95 的路 —— 效率与可靠性的权衡留给读取时 val 打分，不在维护时武断裁剪。
        这里只删"又少走又几乎不成功"的明显垃圾边。

        Returns:
            删除的边数。
        """
        to_delete = [
            ekey
            for ekey, e in self._edges.items()
            if e.n < min_visits
            and (e.n_success / e.n if e.n else 0.0) <= max_success_rate
        ]
        for ekey in to_delete:
            del self._edges[ekey]
        return len(to_delete)

    # ------------------------------------------------------------------ #
    # JSON 持久化
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """序列化为可 JSON 化的 dict。"""
        return {
            "template_id": self.template_id,
            "next_node_num": self._next_node_num,
            "nodes": [
                {
                    "node_id": n.node_id,
                    "app": n.sig.app,
                    "window": n.sig.window,
                    "dialog": n.sig.dialog,
                    "facts": n.sig.facts,
                    "context": n.sig.context,
                    "visits": n.visits,
                }
                for n in self._id_to_node.values()
            ],
            "edges": [
                {
                    "from": e.from_id,
                    "action": {
                        "kind": e.action.kind.value,
                        "name": e.action.name,
                        "arg_slots": list(e.action.arg_slots),
                        "args": dict(e.action.args),
                    },
                    "to_success": e.to_success,
                    "to_fail": e.to_fail,
                    "n": e.n,
                    "n_success": e.n_success,
                    "steps_after_success": e.steps_after_success,
                    "notes": e.notes,
                    "arg_samples": e.arg_samples,
                }
                for e in self._edges.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExperienceTree":
        """从 dict 反序列化。"""
        tree = cls(template_id=data["template_id"])
        tree._next_node_num = data.get("next_node_num", 0)
        for nd in data["nodes"]:
            sig = StateSig(
                app=nd["app"],
                window=nd["window"],
                dialog=nd.get("dialog"),
                facts=nd.get("facts", {}),
                context=nd.get("context", ""),
            )
            node = StateNode(node_id=nd["node_id"], sig=sig, visits=nd.get("visits", 0))
            tree._nodes[sig.key()] = node
            tree._id_to_node[node.node_id] = node
        for ed in data["edges"]:
            action = ActionTmpl(
                kind=ActionKind(ed["action"]["kind"]),
                name=ed["action"]["name"],
                arg_slots=tuple(ed["action"].get("arg_slots", [])),
                args=ed["action"].get("args", {}),
            )
            edge = ActionEdge(
                from_id=ed["from"],
                action=action,
                to_success=ed.get("to_success"),
                to_fail=ed.get("to_fail"),
                n=ed.get("n", 0),
                n_success=ed.get("n_success", 0),
                steps_after_success=ed.get("steps_after_success", []),
                notes=ed.get("notes", []),
                arg_samples=_normalize_arg_samples(ed.get("arg_samples", [])),
            )
            tree._edges[edge.edge_key()] = edge
        return tree

    def save(self, path: Path) -> None:
        """保存到 memory/<template_id>/tree.json。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "ExperienceTree":
        """从 tree.json 载入。"""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class ExperienceForest:
    """经验森林：管理所有任务模板的树（按 template_id 分册）。

    对应 idea 推论 2（按任务分册成森林）：同一环境上"调这个工具会发生什么"
    是任务无关的，但"这一步值多少"是任务相关的 —— 所以价值统计必须按 τ 分账。
    """

    def __init__(self, root_dir: Optional[Path] = None) -> None:
        """
        Args:
            root_dir: 持久化根目录（树落在 root_dir/memory/<template_id>/tree.json）。
                      None 表示纯内存模式（测试用）。
        """
        self.root_dir = Path(root_dir) if root_dir else None
        self._trees: dict[str, ExperienceTree] = {}

    def get_or_create_tree(self, template_id: str) -> ExperienceTree:
        """按模板 id 取树；不存在则新建（或从磁盘载入）。"""
        tree = self._trees.get(template_id)
        if tree is None:
            tree_path = self._tree_path(template_id)
            if tree_path and tree_path.exists():
                tree = ExperienceTree.load(tree_path)
            else:
                tree = ExperienceTree(template_id)
            self._trees[template_id] = tree
        return tree

    def get_tree(self, template_id: str) -> Optional[ExperienceTree]:
        """取已有的树；不存在返回 None（不新建 —— 冷启动时推荐栏自然空返回）。"""
        if template_id in self._trees:
            return self._trees[template_id]
        tree_path = self._tree_path(template_id)
        if tree_path and tree_path.exists():
            tree = ExperienceTree.load(tree_path)
            self._trees[template_id] = tree
            return tree
        return None

    def _tree_path(self, template_id: str) -> Optional[Path]:
        if self.root_dir is None:
            return None
        return self.root_dir / "memory" / template_id / "tree.json"

    def all_trees(self) -> Iterator[ExperienceTree]:
        """Include persisted trees not yet touched by a task-level lookup."""
        identifiers = set(self._trees)
        if self.root_dir is not None:
            identifiers.update(p.parent.name for p in (self.root_dir / "memory").glob("*/tree.json"))
        for template_id in sorted(identifiers):
            tree = self.get_tree(template_id)
            if tree is not None:
                yield tree

    def save_all(self) -> None:
        """把所有树持久化到磁盘。"""
        if self.root_dir is None:
            return
        for template_id, tree in self._trees.items():
            tree.save(self._tree_path(template_id))


def _normalize_arg_samples(raw: list) -> list[dict]:
    """兼容早期格式：``{"text": ...}`` → ``{"args": {...}, "n": 1, "n_success": 0}``。

    2026-09-10 首版 arg_samples 是裸的取值 dict，没有成败计数（那一版的缺陷：
    解法和验证命令平权，见 ``ActionEdge.arg_samples`` 文档）。已经落盘的种子
    要能继续读，否则历史记忆全部失联。

    旧样本的成败**未知**，一律记 ``n_success=0``：宁可让它排在有成功记录的
    取值之后，也不要凭空声称它成功过。
    """
    out: list[dict] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        if "args" in item and isinstance(item["args"], dict):
            out.append(
                {
                    "args": dict(item["args"]),
                    "n": int(item.get("n", 1) or 1),
                    "n_success": int(item.get("n_success", 0) or 0),
                }
            )
        else:
            out.append({"args": dict(item), "n": 1, "n_success": 0})
    return out

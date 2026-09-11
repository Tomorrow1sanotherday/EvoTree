"""五拍检索：每步动手前查森林拼小抄（idea §0 / §4.4）。

固定五拍，零门控、零学习：
    ① 定位：指令 → 哪棵树；观测 → φ → 当前状态 s
    ② [推荐]（s 非空才有）：s 的出边打分取前 2–3，各前瞻 1 层（k=1）
    ③ [注意]：对候选动作按工具名查 global_notes
    ④ 拼小抄（哪栏空整段不出现）
    ⑤ 工具重排：候选工具按 ② 的分数重排

三种场景零 if-else 自然覆盖：
    老任务老状态 → 两栏全乎；老任务新状态 → 只有注意栏；
    全新任务 → 推荐栏空返回（冷启动，不是特殊分支），但注意栏照常命中。

一条硬原则：**[注意]栏永不推荐**（GlobalNotes 只警告，不指路）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .forest import ExperienceTree
from .notes import GlobalNotes
from .scoring import (
    DEFAULT_GAMMA,
    DEFAULT_UCB_C,
    edge_value,
    is_low_confidence,
    mean_steps_to_go,
    select_score,
    success_rate,
)
from .types import ActionEdge, ActionKind, ActionTmpl, StateSig


# --------------------------------------------------------------------------- #
# 检索结果：一张"小抄"
# --------------------------------------------------------------------------- #
@dataclass
class Recommendation:
    """[推荐]栏的一条建议：一个动作 + 它的战绩 + 前瞻的下一步。"""

    action: ActionTmpl
    n: int
    success_rate: float
    mean_steps_to_go: float
    score: float
    low_confidence: bool
    lookahead: Optional[ActionTmpl] = None  # 前瞻 1 层：这步之后最优的下一个动作
    lookahead_hint: Optional[str] = None    # 前瞻的人话提示
    arg_samples: list[dict] = field(default_factory=list)
    """这条边上见过的语义取值（命令原文 / 快捷键），**已按成功率排序**。

    没有它，推荐栏渲染出来是"推荐：computer_use.click —— 2次·成功率1.00"，
    模型看到只知道"点击"，不知道点哪 —— 信息量为零（2026-09-10 实测，
    一棵树里五条 click 成功率各异却无法区分）。

    排序是必需的，不是优化：同一条边上挂着解法和验证命令（实测
    ``ls dir1/file1 dir2/file1`` 与 ``cp file1 dir1 dir2 dir3`` 平权），
    按插入序取前两条会把验证命令推到模型面前。
    """


@dataclass
class Cheatsheet:
    """一步的小抄：推荐栏 + 注意栏 + 重排后的工具顺序。

    哪栏为空，拼进 prompt 时该栏整段不出现（渲染在 render 里处理）。
    """

    recommendations: list[Recommendation] = field(default_factory=list)  # [推荐]
    cautions: list[str] = field(default_factory=list)                    # [注意]
    reranked_tools: list[str] = field(default_factory=list)              # 工具重排结果

    @property
    def is_cold_start(self) -> bool:
        """推荐栏为空 = 冷启动（这类任务没干过，或当前状态没见过）。"""
        return len(self.recommendations) == 0


# --------------------------------------------------------------------------- #
# 检索器
# --------------------------------------------------------------------------- #
class Retriever:
    """五拍检索器。持有打分超参，不持有数据（数据由调用方传入）。"""

    def __init__(
        self,
        gamma: float = DEFAULT_GAMMA,
        ucb_c: float = DEFAULT_UCB_C,
        top_k: int = 3,
    ) -> None:
        self.gamma = gamma
        self.ucb_c = ucb_c
        self.top_k = top_k

    def retrieve(
        self,
        tree: Optional[ExperienceTree],
        current_sig: StateSig,
        notes: GlobalNotes,
        candidate_tools: Optional[list[str]] = None,
    ) -> Cheatsheet:
        """执行五拍，产出小抄。

        Args:
            tree:            当前任务模板的经验树；None = 全新任务（推荐栏空）。
            current_sig:     当前状态签名 φ。
            notes:           全局坑位手册。
            candidate_tools: 当前 app 可用的工具名列表（用于注意栏查询 + 重排）。

        Returns:
            Cheatsheet。哪栏没内容，该栏就是空列表。
        """
        candidate_tools = candidate_tools or []
        sheet = Cheatsheet()

        # ① 定位：在树里找当前状态节点
        node = None
        if tree is not None:
            node = tree.node_for_signature(current_sig)  # 只读，不因检索创建节点

        # ② [推荐]：只有找到了当前状态才有推荐（否则推荐栏自然空 = 冷启动）
        if tree is not None and node is not None:
            sheet.recommendations = self._build_recommendations(tree, node.node_id, node.visits)

        # ③ [注意]：对候选动作（推荐的工具 + 当前 app 工具表）查坑
        caution_tools = set(candidate_tools)
        for rec in sheet.recommendations:
            if rec.action.kind == ActionKind.TOOL:
                caution_tools.add(rec.action.name)
        sheet.cautions = self._build_cautions(notes, caution_tools)

        # ⑤ 工具重排：候选工具按推荐分数排前，其余保持原序
        sheet.reranked_tools = self._rerank_tools(sheet.recommendations, candidate_tools)

        return sheet

    # ------------------------------------------------------------------ #
    # ② 推荐栏：打分取前 k，各前瞻 1 层
    # ------------------------------------------------------------------ #
    def _is_recommendable(self, edge: ActionEdge) -> bool:
        """走过、且从没通向成功的边不进推荐栏（只是不推荐，仍留在森林里）。

        推荐栏回答的是"该走哪条路"。一条走过 n 次、``val`` 仍为 0 的边，说明
        它既没通向成功（p=0），也没留下 cost-to-go（``d=inf``）—— 把它推到
        首位等于告诉模型"推荐一个已知走不通的动作"。

        2026-09-08 实测：改成按整轮成败记账后，失败轨迹上的边同时满足
        ``p=0``、``val=0``、``to_success is None``（前瞻也断），是三重死边；
        而 ``select_score`` 里的 UCB 项恒 >0，不显式过滤就会照样被推上去
        （实测三条 p=0 的边 score 同为 0.3144，全部来自 UCB）。

        **未探索边（n=0）不在此列。** 它们的 ``val`` 同样是 0，但语义是"还没
        试过"，``ucb_bonus`` 对它们返回 inf 正是要让其优先露脸；一并滤掉等于
        关掉探索。

        坑位知识不会因此丢失 —— 报错仍在边的 notes 与 global_notes 里，由注意
        栏负责警告（注意栏永不推荐，见模块 docstring）。
        """
        if edge.n <= 0:
            return True
        return edge_value(edge, self.gamma) > 0

    def _build_recommendations(
        self, tree: ExperienceTree, node_id: str, state_visits: int
    ) -> list[Recommendation]:
        # 先滤掉"已知走不通"的边，再排序取前 k：UCB 项恒 >0，不过滤的话
        # 一条 p=0 的死边照样能靠探索分挤进推荐栏。
        edges = [e for e in tree.out_edges(node_id) if self._is_recommendable(e)]
        # 只对"岔路口"（多出边）里有价值的边推荐；按 select_score 排序
        scored = sorted(
            edges,
            key=lambda e: select_score(e, state_visits, self.gamma, self.ucb_c),
            reverse=True,
        )
        recs: list[Recommendation] = []
        for edge in scored[: self.top_k]:
            rec = Recommendation(
                action=edge.action,
                n=edge.n,
                success_rate=success_rate(edge),
                mean_steps_to_go=mean_steps_to_go(edge),
                score=select_score(edge, state_visits, self.gamma, self.ucb_c),
                low_confidence=is_low_confidence(edge),
                arg_samples=edge.ranked_arg_samples(),
            )
            self._attach_lookahead(tree, edge, rec)
            recs.append(rec)
        return recs

    def _attach_lookahead(
        self, tree: ExperienceTree, edge: ActionEdge, rec: Recommendation
    ) -> None:
        """前瞻 1 层（k=1，D4）：顺着 to_success 指针跳到后继状态，取其最优出边。

        这就是"森林 vs 平表"的分水岭 —— 平表没有指针，做不了这个跨状态跳转。
        """
        if edge.to_success is None:
            return
        succ = tree.node_by_id(edge.to_success)
        if succ is None:
            return
        next_edges = tree.out_edges(succ.node_id)
        if not next_edges:
            return
        best_next = max(
            next_edges,
            key=lambda e: edge_value(e, self.gamma),
        )
        rec.lookahead = best_next.action
        rec.lookahead_hint = f"完成后 {best_next.action.name} 一步收尾"

    # ------------------------------------------------------------------ #
    # ③ 注意栏：查坑（只警告，永不推荐）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_cautions(notes: GlobalNotes, tools: set[str]) -> list[str]:
        cautions: list[str] = []
        for tool_name in sorted(tools):
            for note in notes.query(tool_name):
                cautions.append(f"{tool_name}: {note}")
        return cautions

    # ------------------------------------------------------------------ #
    # ⑤ 工具重排：推荐的工具排前
    # ------------------------------------------------------------------ #
    @staticmethod
    def _rerank_tools(
        recommendations: list[Recommendation], candidate_tools: list[str]
    ) -> list[str]:
        """按推荐分数把候选工具重排：被推荐的工具（有战绩）排到前面。

        替换官方"按应用静态过滤"的列表顺序（对标 OSWorld-MCP Table 2 消融）。
        """
        recommended_order = [
            rec.action.name
            for rec in recommendations
            if rec.action.kind == ActionKind.TOOL and rec.action.name in candidate_tools
        ]
        seen = set(recommended_order)
        rest = [t for t in candidate_tools if t not in seen]
        return recommended_order + rest


# --------------------------------------------------------------------------- #
# 小抄渲染：拼进 prompt 的文本（哪栏空整段不出现）
# --------------------------------------------------------------------------- #
# 一条推荐最多展示几个语义取值。EE-MCP 附录 G 原则一"conciseness over
# coverage"：不限量的经验文本会与任务信息抢注意力，实测反而掉点。而 OWL
# 又刻意把 worker 的 context 保持干净，两头夹击 —— 所以这里取 2。
_MAX_ARG_SAMPLES_SHOWN = 2


def _fmt_arg_sample(sample: dict) -> str:
    """把一条语义取值样本渲染成人能照做的一句。

    对齐 EE-MCP 附录 G 原则二"actionable specificity"：
    ``Ctrl+P to open the print dialog`` 有效，``be efficient`` 无效 ——
    判据是能不能照着做。

    只在这条取值**自己**有成功记录时才标注 —— 边级成功率已经在上一行给过，
    这里重复一个相同的数字只会占位；标注的价值在于区分同一条边上"哪种写法
    通了、哪种没通"。
    """
    args = sample.get("args", {})
    if not isinstance(args, dict):
        return ""
    text = ""
    if "keys" in args:
        text = args["keys"]
    else:
        for key in ("text", "command", "url", "path"):
            if key in args:
                text = args[key]
                break
        else:
            text = "; ".join(f"{k}={v}" for k, v in args.items())
    if not text:
        return ""
    n = sample.get("n", 0) or 0
    ok = sample.get("n_success", 0) or 0
    if n > 1 or ok:
        text = f"{text}（{ok}/{n} 成）"
    return text


def render_cheatsheet(sheet: Cheatsheet) -> str:
    """把小抄渲染成注入 prompt 的文本。空栏整段不出现。

    示例输出：
        [经验] 推荐：
          computer_use.type —— 7次·成功率0.43·平均还差2步
              曾用：cp file1 dir1 dir2 dir3
          os.open_shell —— 4次·成功率0.25·剩余步数未知
        [注意]：
          os.open_shell: Traceback ...

    "曾用"这一行是 2026-09-10 加的。在此之前边上不含任何内容，渲染结果是
    "推荐：computer_use.click —— 2次·成功率1.00" —— 模型只知道"点击"，
    不知道点哪，无法照做。
    """
    lines: list[str] = []

    if sheet.recommendations:
        lines.append("[经验] 推荐：")
        for rec in sheet.recommendations:
            steps = (
                f"平均还差{rec.mean_steps_to_go:.0f}步"
                if rec.mean_steps_to_go != float("inf")
                else "剩余步数未知"
            )
            conf = "（低置信）" if rec.low_confidence else ""
            line = (
                f"  {_fmt_action(rec.action)} —— "
                f"{rec.n}次·成功率{rec.success_rate:.2f}·{steps}{conf}"
            )
            if rec.lookahead_hint:
                line += f"；{rec.lookahead_hint}"
            lines.append(line)
            for sample in rec.arg_samples[:_MAX_ARG_SAMPLES_SHOWN]:
                text = _fmt_arg_sample(sample)
                if text:
                    lines.append(f"      曾用：{text}")

    if sheet.cautions:
        lines.append("[注意]：")
        for c in sheet.cautions:
            lines.append(f"  {c}")

    return "\n".join(lines)


def _fmt_action(action: ActionTmpl) -> str:
    """把动作模板格式化成人类可读，如 duplicate_slide(idx, pos)。"""
    if action.kind == ActionKind.TOOL and action.arg_slots:
        return f"{action.name}({', '.join(action.arg_slots)})"
    return action.name

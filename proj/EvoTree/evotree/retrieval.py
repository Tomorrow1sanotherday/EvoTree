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
    def _build_recommendations(
        self, tree: ExperienceTree, node_id: str, state_visits: int
    ) -> list[Recommendation]:
        edges = tree.out_edges(node_id)
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
def render_cheatsheet(sheet: Cheatsheet) -> str:
    """把小抄渲染成注入 prompt 的文本。空栏整段不出现。

    示例输出：
        [经验] 推荐：duplicate_slide(idx,pos) —— 47次·成功率0.92·平均还差2步；
                    完成后 save 一步收尾
                    [注意] pos 不得超过总页数
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

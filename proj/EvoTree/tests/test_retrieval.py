"""推荐栏的准入规则：已知走不通的边不进推荐栏。

背景（2026-09-08）：``p(e)`` 改成按整轮任务成败记账后，失败轨迹上的每一条边
都变成 ``p=0``。此时 ``edge_value`` 为 0，但 ``select_score = val + ucb_bonus``
里的 UCB 项恒 >0，所以这些边**照样会被排进推荐栏**，注入给模型的小抄会写成
"推荐：computer_use.click 成功率0.00" —— 比空栏更糟。

这些测试钉住三条语义：
    1. 走过且从没通向成功的边（n>0、val=0）→ 不推荐；
    2. 未探索边（n=0）→ 仍要推荐，那是探索信号，不是死边；
    3. 全是死边时推荐栏为空 —— 空是正确结果，按设计整段不出现。
"""

from evotree.forest import ExperienceTree
from evotree.retrieval import Retriever
from evotree.types import ActionKind, ActionEdge, ActionTmpl, StateSig


SIG = StateSig(app="Gnome-terminal", window="user@host: ~", dialog=None)


def _tree_with(outcomes: list[tuple[str, bool]]) -> tuple[ExperienceTree, str]:
    """按 (动作名, 整轮是否成功) 建一棵单状态的树，返回树和起点 node_id。"""
    tree = ExperienceTree(template_id="t")
    for name, success in outcomes:
        kind = ActionKind.GUI if name.startswith("computer_use") else ActionKind.TOOL
        edge = tree.upsert_step(
            from_sig=SIG,
            action=ActionTmpl(kind, name),
            success=success,
            to_sig=SIG if success else None,
            note=None,
        )
        if success:
            # 成功轨迹才记 cost-to-go；没有它 d=inf、val=0（与真实写入一致）
            edge.steps_after_success.append(1)
    return tree, tree.get_or_create_node(SIG).node_id


def _names(recs) -> list[str]:
    return [r.action.name for r in recs]


class TestDeadEdgesExcluded:
    def test_all_failed_edges_yield_empty_recommendations(self):
        # 整轮失败 → 三条边 p=0、val=0；UCB 项 >0 但不足以让它们进推荐栏
        tree, node_id = _tree_with(
            [
                ("computer_use.click", False),
                ("computer_use.type", False),
                ("os.set_volume", False),
            ]
        )
        retriever = Retriever()
        recs = retriever._build_recommendations(tree, node_id, state_visits=3)
        assert recs == []

    def test_failed_edge_dropped_but_successful_one_kept(self):
        tree, node_id = _tree_with(
            [
                ("computer_use.click", False),
                ("os.set_volume", True),
            ]
        )
        retriever = Retriever()
        recs = retriever._build_recommendations(tree, node_id, state_visits=2)
        assert _names(recs) == ["os.set_volume"]

    def test_dead_edge_stays_in_the_forest(self):
        # 只是不推荐，边本身仍在树里 —— 坑位知识不能因此丢失
        tree, node_id = _tree_with([("computer_use.click", False)])
        assert len(tree.out_edges(node_id)) == 1
        assert Retriever()._build_recommendations(tree, node_id, 1) == []


class TestUnexploredEdgesSurvive:
    def test_never_run_edge_is_still_recommendable(self):
        # n=0 的边 val 同样是 0，但语义是"还没试过"，必须留下（ucb_bonus=inf）
        tree = ExperienceTree(template_id="t")
        node = tree.get_or_create_node(SIG)
        fresh = ActionEdge(
            from_id=node.node_id,
            action=ActionTmpl(ActionKind.TOOL, "os.never_tried"),
            n=0,
            n_success=0,
            steps_after_success=[],
        )
        assert Retriever()._is_recommendable(fresh) is True

    def test_unexplored_outranks_nothing_but_is_not_filtered(self):
        tree, node_id = _tree_with([("computer_use.click", False)])
        # get_or_create_edge 建出的就是 n=0 的未探索边（写入前的状态）
        tree.get_or_create_edge(node_id, ActionTmpl(ActionKind.TOOL, "os.never_tried"))
        recs = Retriever()._build_recommendations(tree, node_id, state_visits=2)
        assert _names(recs) == ["os.never_tried"]


class TestSheetIntegration:
    def test_experience_section_absent_when_every_edge_is_dead(self):
        # 端到端：推荐栏空 → 小抄里不应出现 [经验] 整段
        tree, node_id = _tree_with([("computer_use.click", False)])
        sheet = Retriever()._build_recommendations(tree, node_id, 1)
        assert sheet == []

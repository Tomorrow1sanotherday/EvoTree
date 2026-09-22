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
        assert Retriever(strict_quality=True)._build_recommendations(tree, node_id, 1) == []


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


class TestShakyEdgesExcluded:
    """薄硬币边不进推荐栏（实验二 2026-09-11 的实测教训）。

    ``CTRL+ALT+T（2次·成功率0.50·低置信）`` 当时被当成经验推给模型，
    agent 照做烧掉步数。低置信标注只提醒证据不足，拦不住进榜，
    所以准入层面要一票否决。
    """

    def test_thin_coinflip_not_recommended(self):
        # 1败1成 → n=2、p=0.50：val>0 不算死边，但薄而弱 → 拦
        tree, node_id = _tree_with(
            [("computer_use.key", True), ("computer_use.key", False)]
        )
        assert Retriever()._build_recommendations(tree, node_id, 2) == []

    def test_thin_but_spotless_still_recommended(self):
        # 2/2 全胜 → 薄但 p=1.0，是证据不是噪声 → 放行
        tree, node_id = _tree_with([("os.copy", True), ("os.copy", True)])
        assert _names(Retriever()._build_recommendations(tree, node_id, 2)) == [
            "os.copy"
        ]

    def test_single_success_is_candidate_only_in_strict_mode(self):
        tree, node_id = _tree_with([("os.copy", True)])
        assert Retriever(strict_quality=True)._build_recommendations(tree, node_id, 1) == []

    def test_thick_but_weak_is_filtered_in_strict_mode(self):
        # Frozen evaluation must not recommend a frequently failing edge.
        outcomes = [("os.thick", False)] * 3 + [("os.thick", True)] * 3
        outcomes += [("os.good", True)]
        tree, node_id = _tree_with(outcomes)
        assert "os.thick" not in _names(
            Retriever(strict_quality=True)._build_recommendations(tree, node_id, 7)
        )

    def test_legacy_gate_can_be_used_for_ablation(self):
        outcomes = [("os.thick", False)] * 3 + [("os.thick", True)] * 3
        tree, node_id = _tree_with(outcomes)
        assert "os.thick" in _names(
            Retriever(strict_quality=False)._build_recommendations(
                tree, node_id, state_visits=6
            )
        )

    def test_success_edge_without_cost_to_go_is_filtered(self):
        tree = ExperienceTree(template_id="t")
        tree.upsert_step(
            from_sig=SIG,
            action=ActionTmpl(ActionKind.TOOL, "os.ambiguous"),
            success=True,
            to_sig=None,
        )
        node_id = tree.get_or_create_node(SIG).node_id
        assert Retriever(strict_quality=True)._build_recommendations(tree, node_id, 1) == []


class TestFrozenTurnsOffExploration:
    """explore=False（冻结评测接线 write_enabled=False）时的两条语义。"""

    def _two_roads(self) -> tuple[ExperienceTree, str]:
        """两条路：厚而稳（8/8 成，还差1步）与 薄而远（1/1 成，还差5步）。"""
        tree = ExperienceTree(template_id="t")
        node = tree.get_or_create_node(SIG)
        solid = tree.get_or_create_edge(
            node.node_id, ActionTmpl(ActionKind.TOOL, "os.solid")
        )
        solid.n = 8
        solid.n_success = 8
        solid.steps_after_success = [1] * 8
        lucky = tree.get_or_create_edge(
            node.node_id, ActionTmpl(ActionKind.TOOL, "os.lucky")
        )
        lucky.n = 1
        lucky.n_success = 1
        lucky.steps_after_success = [5]
        return tree, node.node_id

    def test_frozen_ranks_by_value_not_ucb(self):
        # state_visits=50 时 UCB 项让薄路反超厚路；frozen 关探索后按价值排
        tree, node_id = self._two_roads()
        online = Retriever()
        assert _names(
            online._build_recommendations(tree, node_id, state_visits=50)
        ) == ["os.lucky", "os.solid"]
        frozen = Retriever(explore=False)
        assert _names(
            frozen._build_recommendations(tree, node_id, state_visits=50)
        ) == ["os.solid", "os.lucky"]

    def test_frozen_drops_unexplored_edges(self):
        # n=0 的边在 frozen 下渲染成 "0次·成功率0.00" 是纯噪声 —— 不进榜
        tree, node_id = _tree_with([("os.copy", True)])
        tree.get_or_create_edge(
            node_id, ActionTmpl(ActionKind.TOOL, "os.never_tried")
        )
        frozen = Retriever(explore=False)
        assert _names(
            frozen._build_recommendations(tree, node_id, state_visits=2)
        ) == ["os.copy"]
        # 对照：探索开着时未探索边照样露脸（UCB=inf 排最前）
        online = Retriever()
        assert _names(
            online._build_recommendations(tree, node_id, state_visits=2)
        ) == ["os.never_tried", "os.copy"]


class TestMemoryWiresExploreToWriteEnabled:
    """EvoTreeMemory 默认 explore=write_enabled：冻结评测不用改调用方。"""

    def test_frozen_memory_disables_exploration(self):
        from evotree.memory import EvoTreeMemory

        assert EvoTreeMemory(write_enabled=False).retriever.explore is False

    def test_online_memory_keeps_exploration(self):
        from evotree.memory import EvoTreeMemory

        assert EvoTreeMemory(write_enabled=True).retriever.explore is True

    def test_explicit_explore_override_wins(self):
        from evotree.memory import EvoTreeMemory

        assert (
            EvoTreeMemory(write_enabled=False, explore=True).retriever.explore
            is True
        )


class TestSheetIntegration:
    def test_experience_section_absent_when_every_edge_is_dead(self):
        # 端到端：推荐栏空 → 小抄里不应出现 [经验] 整段
        tree, node_id = _tree_with([("computer_use.click", False)])
        sheet = Retriever()._build_recommendations(tree, node_id, 1)
        assert sheet == []

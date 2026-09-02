"""scoring 模块的单元测试。

打分是纯函数，用构造的边直接验证公式与边界情况 —— 不需要服务器、不需要真轨迹。
这些测试同时充当 scoring 的"可执行文档"：看测试就知道每个函数在什么输入下给什么值。
"""

import math

import pytest

from evotree.scoring import (
    DEFAULT_GAMMA,
    edge_value,
    is_low_confidence,
    mean_steps_to_go,
    select_score,
    success_rate,
    ucb_bonus,
)
from evotree.types import ActionEdge, ActionKind, ActionTmpl


def make_edge(n: int, n_success: int, steps: list[int]) -> ActionEdge:
    """构造一条测试用的边（起点/动作不影响打分，用占位值）。"""
    return ActionEdge(
        from_id="s0",
        action=ActionTmpl(ActionKind.TOOL, "dummy_tool"),
        n=n,
        n_success=n_success,
        steps_after_success=steps,
    )


# --------------------------- success_rate --------------------------------- #
class TestSuccessRate:
    def test_normal(self):
        assert success_rate(make_edge(5, 3, [1, 2, 3])) == pytest.approx(0.6)

    def test_never_run_is_zero(self):
        # 从未走过的边：无证据 = 成功率 0
        assert success_rate(make_edge(0, 0, [])) == 0.0

    def test_all_success(self):
        assert success_rate(make_edge(4, 4, [1, 1, 1, 1])) == 1.0


# --------------------------- mean_steps_to_go ----------------------------- #
class TestMeanStepsToGo:
    def test_normal(self):
        assert mean_steps_to_go(make_edge(3, 3, [2, 4, 3])) == pytest.approx(3.0)

    def test_no_success_is_inf(self):
        # 没有成功记录：不知道要多久 → inf
        assert math.isinf(mean_steps_to_go(make_edge(2, 0, [])))


# --------------------------- edge_value ----------------------------------- #
class TestEdgeValue:
    def test_formula(self):
        # p=0.5, d=2, gamma=0.9 → 0.5 * 0.9^2 = 0.405
        e = make_edge(4, 2, [2, 2])
        assert edge_value(e) == pytest.approx(0.5 * DEFAULT_GAMMA ** 2)

    def test_no_success_is_zero(self):
        # 无成功证据 → d=inf → γ^inf=0 → val=0
        assert edge_value(make_edge(3, 0, [])) == 0.0

    def test_reliable_and_short_beats_reliable_and_long(self):
        # 同样 100% 成功率，短路径 val 应更高（效率项主导）
        short = make_edge(5, 5, [1, 1, 1, 1, 1])
        long = make_edge(5, 5, [8, 8, 8, 8, 8])
        assert edge_value(short) > edge_value(long)

    def test_short_but_flaky_can_lose_to_long_but_reliable(self):
        # idea §4.2 的核心主张:2步但0.3成功率 不应盖过 4步但0.95成功率
        flaky_short = make_edge(10, 3, [2] * 3)      # p=0.3, d=2
        reliable_long = make_edge(10, 9, [4] * 9)    # p=0.9, d=4
        # 0.3*0.9^2=0.243  vs  0.9*0.9^4≈0.590 → 可靠长路胜出
        assert edge_value(reliable_long) > edge_value(flaky_short)


# --------------------------- ucb_bonus ------------------------------------ #
class TestUcbBonus:
    def test_unvisited_edge_is_inf(self):
        # 没走过的边最该被探索 → inf
        assert math.isinf(ucb_bonus(make_edge(0, 0, []), state_visits=10))

    def test_fewer_visits_more_bonus(self):
        # 相同状态访问数下，走得少的边探索加分更高
        rare = make_edge(2, 1, [3, 3])
        common = make_edge(50, 25, [3] * 25)
        assert ucb_bonus(rare, 100) > ucb_bonus(common, 100)

    def test_insufficient_state_visits_no_bonus(self):
        # 状态访问不足（≤1）时不加探索分，避免 ln 无定义
        assert ucb_bonus(make_edge(1, 1, [2]), state_visits=1) == 0.0


# --------------------------- select_score --------------------------------- #
class TestSelectScore:
    def test_is_value_plus_bonus(self):
        e = make_edge(5, 3, [2, 2, 2])
        expected = edge_value(e) + ucb_bonus(e, 20)
        assert select_score(e, 20) == pytest.approx(expected)


# --------------------------- is_low_confidence ---------------------------- #
class TestLowConfidence:
    def test_few_samples_is_low_confidence(self):
        assert is_low_confidence(make_edge(2, 1, [3])) is True

    def test_enough_samples_is_confident(self):
        assert is_low_confidence(make_edge(10, 6, [3] * 6)) is False

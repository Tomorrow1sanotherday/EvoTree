"""边价值打分：EvoTree 的数学核心（idea §0 唯一的公式）。

本模块是**纯函数**集合，不依赖任何可变状态 —— 输入一条边的统计量，
输出它的价值分。这样最容易单元测试，也最容易读懂"方法到底在算什么"。

核心公式（对应 idea §0）：

    p(e)     = n_success / n                    # 可靠性：成功率（测出来的）
    d(e)     = mean(steps_after_success)        # 效率：成功时平均剩余步数（记出来的）
    val(e)   = p(e) · γ^d(e)                    # 边价值 = 可靠性 × 效率折扣
    score(e) = val(e) + c · sqrt(ln N_s / n)    # 选边分 = 边价值 + UCB 探索项

重要脱钩声明（对应 idea D2）：
    val 是从执行统计**直接聚合**出的边价值，**不是** RL 里的 Bellman Q 函数。
    我们不做 value iteration、不假设 Markov 最优。
    - γ 只是把"剩余步数"折成效率分的单调递减权重；
    - UCB 只是在一个状态的多条出边之间做 bandit 式探索。
    二者是借用的两个工具，不代表"本方法是一个 MDP"。
"""

from __future__ import annotations

import math

from .types import ActionEdge

# 默认超参（可在 config 中覆盖；集中定义避免散落魔数）
DEFAULT_GAMMA = 0.9   # 效率折扣因子：剩余步数每多一步，价值乘一次 γ
DEFAULT_UCB_C = 0.3   # UCB 探索系数：走得少的边加一点分，防止偶然一次成功垄断推荐


def success_rate(edge: ActionEdge) -> float:
    """成功率 p = n_success / n。边从未走过时返回 0.0（无证据 = 不可靠）。"""
    if edge.n <= 0:
        return 0.0
    return edge.n_success / edge.n


def mean_steps_to_go(edge: ActionEdge) -> float:
    """平均剩余步数 d = mean(steps_after_success)。

    无成功记录时返回 math.inf —— 表示"不知道走这条边之后要多久"，
    在 val 里会让效率项 γ^inf → 0，即无成功证据的边效率分为 0。
    """
    if not edge.steps_after_success:
        return math.inf
    return sum(edge.steps_after_success) / len(edge.steps_after_success)


def edge_value(edge: ActionEdge, gamma: float = DEFAULT_GAMMA) -> float:
    """边价值 val = p · γ^d。

    - 可靠性 p 越高越好；
    - 剩余步数 d 越小越好（γ^d 随 d 增大而衰减）。
    d 为 inf 时 γ^d = 0，故无成功证据的边 val = 0。

    Args:
        edge:  要打分的动作边。
        gamma: 效率折扣因子，(0, 1)。

    Returns:
        边价值，落在 [0, 1]。
    """
    p = success_rate(edge)
    d = mean_steps_to_go(edge)
    if math.isinf(d):
        return 0.0
    return p * (gamma ** d)


def ucb_bonus(edge: ActionEdge, state_visits: int, c: float = DEFAULT_UCB_C) -> float:
    """UCB 探索项 = c · sqrt(ln(N_s) / n)。

    给走得少的边一点"露脸机会"，防止某条边偶然一次成功就垄断推荐。
    n 越小、加分越多；随该边被走多而衰减。

    Args:
        edge:         要打分的动作边。
        state_visits: 该边起点状态 N_s 的总访问次数。
        c:            探索系数。

    Returns:
        探索加分（≥ 0）。边未走过（n=0）时返回 inf —— 未知边最该被探索。
    """
    if edge.n <= 0:
        return math.inf
    if state_visits <= 1:
        # ln(1)=0 甚至 ln(0) 无定义；访问不足时不加探索分，交给 val 决定。
        return 0.0
    return c * math.sqrt(math.log(state_visits) / edge.n)


def select_score(
    edge: ActionEdge,
    state_visits: int,
    gamma: float = DEFAULT_GAMMA,
    c: float = DEFAULT_UCB_C,
) -> float:
    """选边总分 score = val + UCB。推理时 argmax 这个分来选边。

    Args:
        edge:         候选动作边。
        state_visits: 起点状态的总访问次数（UCB 分母的对数项用）。
        gamma:        效率折扣因子。
        c:            UCB 探索系数。

    Returns:
        综合分；越高越优先推荐。
    """
    return edge_value(edge, gamma) + ucb_bonus(edge, state_visits, c)


def is_low_confidence(edge: ActionEdge, min_n: int = 3) -> bool:
    """是否"低置信"：样本量太少，其 p 估计不可靠。

    用于给推荐附"低置信"标注（idea §4.6 三道保险之一）。
    也呼应地基风险②：361 任务 / 3 轮下每条边 n 可能只有个位数，
    n 很小时的成功率就是噪声，必须让 agent 知道这条建议证据不足。
    """
    return edge.n < min_n

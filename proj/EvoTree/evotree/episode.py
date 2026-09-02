"""统一的 episode 数据结构。

本模块定义 OSWorld 适配器、在线运行时和 EvoTree 记忆系统之间的稳定边界。

重要语义：
    - ``EpisodeStep.observation`` 是动作执行前的状态。
    - ``EpisodeStep.next_observation`` 是动作执行后的状态（如果可得）。
    - ``EpisodeStep.action_success`` 是**这一步动作本身**是否成功（边级）。
    - ``Episode.success`` 是**整轮任务**最终是否成功（轮级）。

两个成败是不同的量，对应 idea §0 写入的两个不同步骤，切勿混用：

    第 3 步（边级）："按 (from + α) 找边（有则 n+1、按成败更新 n_success）"
        —— 用 ``EpisodeStep.action_success``，即这次调用有没有报错。
    第 4 步（轮级）："若本轮成功：从最后一步倒着数……"
        —— 用 ``Episode.success``，只有整轮成功才记 cost-to-go。

idea §8 把这个区分写死了：一轮**最终成功**的 8 步轨迹里，越界失败两次的
``duplicate_slide`` 边应记成 ``n=3、p=0.33``，且失败落点 ``s_err`` 有自己的
出边统计。若拿整轮成败去更新每条边，这条边会被记成 ``p=1.0``——一条踩了
两次坑的边变成"100% 可靠"，且 ``to_fail`` 永远不会被赋值。

OSWorld 的中间 reward 通常只是 0，不能据此判断单步动作失败——但这不意味着
边级成败无从判定。真正的判据是**工具有没有报错**，由
``distill_osworld.extract_error_note`` 统一提取。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class EpisodeStep:
    """一轮任务中的一个可记录动作。

    ``raw_action`` 保留原始动作表示（例如 OSWorld 的 ``prediction``），
    不在这里提前丢信息；具体归一化由 Distiller 完成。
    """

    step_index: int
    observation: dict[str, Any]
    raw_action: dict[str, Any]
    next_observation: Optional[dict[str, Any]] = None
    note: Optional[str] = None
    action_success: bool = True
    reward: float = 0.0
    done: bool = False
    screenshot_file: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """这一步动作本身是否失败（工具报错）。"""
        return not self.action_success


@dataclass(frozen=True)
class Episode:
    """一轮完整执行记录。

    Attributes:
        instruction: 用户任务指令，决定经验写入哪棵任务树。
        steps:       按执行顺序排列的可记录动作；终止声明不应放入其中。
        success:     整轮任务最终是否成功。
        source:      原始轨迹来源，便于审计与复现。
        metadata:    轨迹级附加信息（score、domain、trial 等）。
    """

    instruction: str
    steps: tuple[EpisodeStep, ...]
    success: bool
    source: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.instruction.strip():
            raise ValueError("Episode instruction must not be empty")
        indices = [step.step_index for step in self.steps]
        if indices != sorted(indices):
            raise ValueError("Episode steps must be ordered by step_index")

    @property
    def num_steps(self) -> int:
        """可记录动作数（不包含 reset / DONE / terminate）。"""
        return len(self.steps)

    @property
    def is_empty(self) -> bool:
        """是否没有可记录动作。"""
        return not self.steps


@dataclass(frozen=True)
class RuntimeStep:
    """在线接入时暂存的一步。

    与 ``EpisodeStep`` 的区别：在线 agent 还没有**整轮**结果标签，
    所以这里不包含 episode 级 success；在 episode 结束时由 ``EvoTreeRuntime``
    统一构造 ``Episode``。

    但**边级** ``action_success`` 在动作执行完就已知（工具报没报错），
    所以它在这里就要记下来，不能等到 finish。
    """

    step_index: int
    observation: dict[str, Any]
    raw_action: dict[str, Any]
    next_observation: Optional[dict[str, Any]] = None
    note: Optional[str] = None
    action_success: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

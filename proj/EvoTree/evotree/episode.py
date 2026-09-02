"""统一的 episode 数据结构。

本模块定义 OSWorld 适配器、在线运行时和 EvoTree 记忆系统之间的稳定边界。

重要语义：
    - ``EpisodeStep.observation`` 是动作执行前的状态。
    - ``EpisodeStep.next_observation`` 是动作执行后的状态（如果可得）。
    - ``Episode.success`` 是整轮任务最终是否成功。

OSWorld 的中间 reward 通常只是 0，不能据此判断单步动作失败。因此，
真实轨迹写入 EvoTree 时，Episode.success 会作为路径级结果标签，
应用到这轮经过的动作边上。这正是 EvoTree 要估计的"从这个状态
选择这个动作后，整轮任务最终成功的概率"。
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
    reward: float = 0.0
    done: bool = False
    screenshot_file: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


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

    与 ``EpisodeStep`` 的区别：在线 agent 还没有整轮结果标签，
    所以这里不包含 success；在 episode 结束时由 ``EvoTreeRuntime``
    统一构造 ``Episode``。
    """

    step_index: int
    observation: dict[str, Any]
    raw_action: dict[str, Any]
    next_observation: Optional[dict[str, Any]] = None
    note: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

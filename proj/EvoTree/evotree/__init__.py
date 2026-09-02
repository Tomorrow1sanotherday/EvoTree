"""EvoTree：通过树结构经验记忆实现混合动作（MCP-GUI）Agent 的自进化。

training-free、plug-and-play 的经验记忆 wrapper：
    - 三道清洗把真实桌面世界映射到可数的 (τ, φ, α) 空间（distill）
    - 在其上按任务组织经验森林，边上挂执行统计（forest）
    - 用 val = 成功率 × γ^剩余步数打分选边（scoring）
    - 每步动手前查森林拼小抄注入 prompt（retrieval）

对外只需用顶层封装 EvoTreeMemory（memory.py），其余模块是内部实现。
"""

from .episode import Episode, EpisodeStep, RuntimeStep
from .integration import BeforeModelResult, EvoTreeRuntime, inject_cheatsheet
from .memory import EvoTreeMemory, Step
from .osworld_adapter import OSWorldEpisodeAdapter, load_osworld_episode
from .osworld_integration import (
    EvoTreeAgentWrapper,
    EvoTreeEnvWrapper,
    build_observation,
    candidate_tool_names,
    reorder_tool_list,
    wrap_osworld,
)

__version__ = "0.1.0"

__all__ = [
    "BeforeModelResult",
    "Episode",
    "EpisodeStep",
    "EvoTreeMemory",
    "EvoTreeRuntime",
    "OSWorldEpisodeAdapter",
    "EvoTreeAgentWrapper",
    "EvoTreeEnvWrapper",
    "RuntimeStep",
    "Step",
    "inject_cheatsheet",
    "load_osworld_episode",
    "build_observation",
    "candidate_tool_names",
    "reorder_tool_list",
    "wrap_osworld",
]

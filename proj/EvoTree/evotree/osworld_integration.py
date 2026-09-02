"""独立的 OSWorld-MCP agent loop 适配器。

这个模块不导入 OSWorld 的实现，只依赖它在 ``_get_obs``、``predict``、
``step`` 和 ``evaluate`` 之间已经存在的最小协议。因此可以把两个 wrapper
放在 benchmark 进程外层，裸 baseline 仍直接使用原 agent/env。

典型接法：

    agent, env, session = wrap_osworld(agent, env, memory, use_evotree=True)
    lib_run_single.run_single_example(agent, env, example, ...)

``EvoTreeAgentWrapper`` 在模型调用前注入小抄并重排工具 schema；
``EvoTreeEnvWrapper`` 在真实 ``step`` 后记录动作，在 ``evaluate`` 后提交整轮
结果。reset、DONE/FAIL/terminate 和 client error 都不会进入经验路径。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from .distill_osworld import OSWorldDistiller
from .integration import EvoTreeRuntime, inject_cheatsheet
from .memory import EvoTreeMemory


ObservationMapper = Callable[[Mapping[str, Any]], Mapping[str, Any]]


def build_observation(osworld_obs: Mapping[str, Any]) -> dict[str, Any]:
    """把 OSWorld ``_get_obs()`` 映射成 EvoTree 的状态输入。

    ``cur_app`` 和活动窗口标题来自 OSWorld 的结构化字段；a11y tree、截图和
    执行结果原样保留，供后续更精细的 distiller 使用。没有标题时使用空串，
    不根据截图猜测窗口或弹窗。
    """

    apps = osworld_obs.get("apps")
    active_id = osworld_obs.get("cur_window_id")
    active_window = apps.get(active_id) if isinstance(apps, Mapping) else None
    if not isinstance(active_window, Mapping):
        active_window = {}

    app = osworld_obs.get("cur_app") or active_window.get("app_name") or osworld_obs.get("app") or "unknown"
    window = (
        osworld_obs.get("window")
        or active_window.get("title")
        or osworld_obs.get("window_title")
        or ""
    )
    dialog = osworld_obs.get("dialog")

    result = dict(osworld_obs)
    result.update(
        {
            "app": str(app),
            "window": str(window),
            "dialog": None if dialog is None else str(dialog),
            "facts": dict(osworld_obs.get("facts", {}) or {}),
            "a11y_tree": osworld_obs.get("accessibility_tree"),
        }
    )
    return result


def candidate_tool_names(tool_list: Any) -> list[str]:
    """从 OSWorld MCP function schema 提取稳定的候选工具名称。"""

    if not isinstance(tool_list, Sequence) or isinstance(tool_list, (str, bytes)):
        return []
    names: list[str] = []
    for item in tool_list:
        name: Any = None
        if isinstance(item, str):
            name = item
        elif isinstance(item, Mapping):
            function = item.get("function")
            if isinstance(function, Mapping):
                name = function.get("name")
            if not name:
                name = item.get("name")
        if isinstance(name, str) and name.strip() and name.strip() not in names:
            names.append(name.strip())
    return names


def _tool_name(item: Any) -> Optional[str]:
    names = candidate_tool_names([item])
    return names[0] if names else None


def reorder_tool_list(tool_list: Any, ordered_names: Sequence[str]) -> Any:
    """按推荐名称重排 schema，保留未知项及原始对象。"""

    if not isinstance(tool_list, list) or not ordered_names:
        return tool_list
    by_name: dict[str, list[Any]] = {}
    for item in tool_list:
        name = _tool_name(item)
        if name is not None:
            by_name.setdefault(name, []).append(item)
    result: list[Any] = []
    used: set[int] = set()
    for name in ordered_names:
        for item in by_name.get(name, []):
            marker = id(item)
            if marker not in used:
                result.append(item)
                used.add(marker)
    for item in tool_list:
        if id(item) not in used:
            result.append(item)
    return result


def _is_terminal_action(raw_action: Mapping[str, Any]) -> bool:
    action = raw_action.get("action")
    if isinstance(action, Mapping):
        action = action.get("action_type") or action.get("action")
    if isinstance(action, str) and action.strip().upper() in {"DONE", "FAIL", "TERMINATE"}:
        return True
    return OSWorldDistiller.is_terminal(raw_action)


def _is_client_error(raw_action: Mapping[str, Any]) -> bool:
    return OSWorldDistiller.is_parse_error(raw_action)


def _note_from_step(info: Any, next_observation: Mapping[str, Any]) -> Optional[str]:
    if isinstance(info, Mapping):
        for key in ("error", "exception", "message"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    # DesktopEnv puts MCP/Python execution output on the next observation. Only
    # retain messages that look like failures; normal tool results are not notes.
    value = next_observation.get("exe_result")
    if isinstance(value, str) and value.strip():
        lowered = value.lower()
        if any(marker in lowered for marker in ("error", "exception", "traceback", "failed")):
            return value.strip()
    return None


class EvoTreeAgentWrapper:
    """给任意 OSWorld agent 增加 EvoTree read hook。"""

    def __init__(
        self,
        agent: Any,
        runtime: EvoTreeRuntime,
        *,
        use_evotree: bool = True,
        observation_mapper: ObservationMapper = build_observation,
    ) -> None:
        self.agent = agent
        self.runtime = runtime
        self.use_evotree = use_evotree
        self.observation_mapper = observation_mapper
        self.last_before_model = None
        self._instruction: Optional[str] = None
        self._pending: list[dict[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._instruction = None
        self._pending.clear()
        self.last_before_model = None
        return self.agent.reset(*args, **kwargs)

    def predict(self, instruction: str, obs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        if not self.use_evotree:
            return self.agent.predict(instruction, obs, *args, **kwargs)

        if not self.runtime.active or self._instruction != instruction:
            self.runtime.begin(instruction)
            self._instruction = instruction

        mapped = self.observation_mapper(obs)
        names = candidate_tool_names(obs.get("tool_list"))
        before = self.runtime.before_model(dict(mapped), names)
        self.last_before_model = before

        agent_obs = dict(obs)
        agent_obs["tool_list"] = reorder_tool_list(
            obs.get("tool_list", []), before.reranked_tools
        )
        prompt = inject_cheatsheet(instruction, before.text)
        result = self.agent.predict(prompt, agent_obs, *args, **kwargs)

        response: Any = None
        actions: Any = []
        messages: Any = None
        if isinstance(result, tuple):
            if len(result) >= 1:
                response = result[0]
            if len(result) >= 2:
                actions = result[1]
            if len(result) >= 3:
                messages = result[2]
        if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes)):
            self._pending = [
                {"prediction": response, "action": action, "messages": messages}
                for action in actions
            ]
        else:
            self._pending = []
        return result

    def consume_raw_action(self, action: Any) -> dict[str, Any]:
        """返回与下一次 env.step 对应的模型响应和原始环境动作。"""

        if self._pending:
            raw = dict(self._pending.pop(0))
            raw["action"] = action
            return raw
        return {"prediction": None, "action": action, "messages": None}


class EvoTreeEnvWrapper:
    """给任意 OSWorld env 增加 after_action/finish hook。"""

    def __init__(
        self,
        env: Any,
        runtime: EvoTreeRuntime,
        agent_wrapper: EvoTreeAgentWrapper,
        *,
        use_evotree: bool = True,
        observation_mapper: ObservationMapper = build_observation,
    ) -> None:
        self.env = env
        self.runtime = runtime
        self.agent_wrapper = agent_wrapper
        self.use_evotree = use_evotree
        self.observation_mapper = observation_mapper
        self._last_observation: Optional[Mapping[str, Any]] = None
        self.last_score: Optional[float] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._last_observation = None
        self.last_score = None
        self.agent_wrapper._pending.clear()
        return self.env.reset(*args, **kwargs)

    def _get_obs(self, *args: Any, **kwargs: Any) -> Any:
        observation = self.env._get_obs(*args, **kwargs)
        if isinstance(observation, Mapping):
            self._last_observation = observation
        return observation

    def step(self, action: Any, *args: Any, **kwargs: Any) -> Any:
        before = self._last_observation
        result = self.env.step(action, *args, **kwargs)
        if not self.use_evotree or not isinstance(result, tuple) or not result:
            return result

        next_observation = result[0]
        if isinstance(next_observation, Mapping):
            self._last_observation = next_observation
        if before is None or not isinstance(next_observation, Mapping):
            return result

        raw_action = self.agent_wrapper.consume_raw_action(action)
        if _is_terminal_action(raw_action) or _is_client_error(raw_action):
            return result
        info = result[3] if len(result) > 3 else {}
        self.runtime.after_action(
            dict(self.observation_mapper(before)),
            raw_action,
            next_observation=dict(self.observation_mapper(next_observation)),
            note=_note_from_step(info, next_observation),
            metadata={"env_info": info},
        )
        return result

    def evaluate(self, *args: Any, **kwargs: Any) -> Any:
        score = self.env.evaluate(*args, **kwargs)
        self.last_score = float(score)
        if self.use_evotree and self.runtime.active:
            self.runtime.finish(score=self.last_score)
        return score


def wrap_osworld(
    agent: Any,
    env: Any,
    memory: EvoTreeMemory,
    *,
    use_evotree: bool = True,
    observation_mapper: ObservationMapper = build_observation,
) -> tuple[Any, Any, Optional[EvoTreeRuntime]]:
    """构造可直接传给 OSWorld runner 的 ``(agent, env, runtime)``。

    ``use_evotree=False`` 返回原始 agent/env（不创建 runtime、不复制观测），
    这样 baseline 的行为和资源开销都保持原样。
    """

    if not use_evotree:
        return agent, env, None
    runtime = EvoTreeRuntime(memory)
    agent_wrapper = EvoTreeAgentWrapper(
        agent, runtime, use_evotree=True, observation_mapper=observation_mapper
    )
    env_wrapper = EvoTreeEnvWrapper(
        env,
        runtime,
        agent_wrapper,
        use_evotree=True,
        observation_mapper=observation_mapper,
    )
    return agent_wrapper, env_wrapper, runtime


__all__ = [
    "EvoTreeAgentWrapper",
    "EvoTreeEnvWrapper",
    "build_observation",
    "candidate_tool_names",
    "reorder_tool_list",
    "wrap_osworld",
]

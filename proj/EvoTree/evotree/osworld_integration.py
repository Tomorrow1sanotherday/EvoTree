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

import re
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from .distill_osworld import GUI_TOOL_NAME, OSWorldDistiller, extract_error_note, parse_tool_calls
from .integration import EvoTreeRuntime, inject_cheatsheet
from .memory import EvoTreeMemory
from .dual_retrieval import action_stage


ObservationMapper = Callable[[Mapping[str, Any]], Mapping[str, Any]]


# OSWorld 的 get_current_apps() 解析 ``wmctrl -xl`` 时取错了字段：输出格式是
# ``id 桌面 WM_CLASS 主机名 标题``，它用 ``parts[3]``（主机名）当 title，真正的
# 标题在 ``parts[4]``。所以所有窗口 title 都等于主机名（实测全是
# "user-virtual-machine"）。这里做兜底修正。
_HOSTNAME_LIKE = re.compile(r"^[\w-]+-virtual-machine$|^localhost$", re.IGNORECASE)


def _clean_window_title(title: Any) -> str:
    """丢掉形如主机名的伪标题。

    宁可返回空串走冷启动，也不要把主机名当窗口标题 —— 那会让不同应用的不同
    步骤全部塌缩成同一个状态签名（2026-09-06 实测：28 个节点里 19 个是空签名，
    其余全是主机名，树完全扁平、前瞻退化成 s0→s0 自环）。
    """
    if not isinstance(title, str):
        return ""
    text = title.strip()
    return "" if _HOSTNAME_LIKE.match(text) else text


def build_observation(osworld_obs: Mapping[str, Any]) -> dict[str, Any]:
    """把 OSWorld ``_get_obs()`` 映射成 EvoTree 的状态输入。

    ``cur_app`` 和活动窗口标题来自 OSWorld 的结构化字段。窗口标题会过一遍
    ``_clean_window_title``，因为上游解析 ``wmctrl`` 时把主机名当成了标题。

    a11y tree 在 ``--observation_type screenshot`` 下是 ``None``（runner 只在
    a11y_tree/som 等模式才请求它），所以 φ 目前只能依赖 ``apps``/``cur_app``
    这两个结构化字段，不能改用 a11y tree 提取。
    """

    apps = osworld_obs.get("apps")
    active_id = osworld_obs.get("cur_window_id")
    active_window = apps.get(active_id) if isinstance(apps, Mapping) else None
    if not isinstance(active_window, Mapping):
        active_window = {}

    app = osworld_obs.get("cur_app") or active_window.get("app_name") or osworld_obs.get("app") or "unknown"
    window = (
        _clean_window_title(osworld_obs.get("window"))
        or _clean_window_title(active_window.get("title"))
        or _clean_window_title(osworld_obs.get("window_title"))
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


def available_tool_names(tool_list: Any) -> list[str]:
    """agent 手上**真正能调**的工具名：MCP 工具表 + 恒在的 GUI 工具。

    与 ``candidate_tool_names`` 的区别（2026-09-18 修，实验四第一小时的教训）：
    后者忠实反映 ``obs["tool_list"]`` 的 schema，用于工具重排；但那张表里
    **没有 ``computer_use``** —— 它由 agent 无条件附加
    （``owl_agent.py``: ``functions = [computer_use.function]; functions.extend(tool_list)``），
    而 os 域任务的 ``tool_list`` 恰好是空的（``_get_obs`` 只为 libreoffice 三兄弟
    映射 MCP 工具表）。

    于是注意栏拿到的候选名字是空列表，循环体一次都不执行，坑位笔记再对也送不出去
    —— 实验三整轮 [注意] 零注入的第二道闸门，修好 notes 查询后仍卡在这里。

    GUI 工具放在末尾：重排的语义是"有战绩的 MCP 工具排前面"，不该因为这次补名
    而改变 MCP 工具之间的相对顺序。
    """
    names = candidate_tool_names(tool_list)
    if GUI_TOOL_NAME not in names:
        names.append(GUI_TOOL_NAME)
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


def _note_from_step(info: Any, next_observation: Mapping[str, Any], tool_name: str = "") -> Optional[str]:
    """这一步有没有报错。委托给 distill_osworld 的单一判据实现。

    返回非 None 即表示这条边这次执行失败（edge-level failure），同时这段文本
    会作为坑位笔记进入边的 notes 与 global_notes。
    """
    return extract_error_note(info, next_observation, tool_name=tool_name)


class EvoTreeAgentWrapper:
    """给任意 OSWorld agent 增加 EvoTree read hook。"""

    def __init__(
        self,
        agent: Any,
        runtime: EvoTreeRuntime,
        *,
        use_evotree: bool = True,
        observation_mapper: ObservationMapper = build_observation,
        gui_annotator: Optional[Callable] = None,
    ) -> None:
        self.agent = agent
        self.runtime = runtime
        self.use_evotree = use_evotree
        self.observation_mapper = observation_mapper
        self.gui_annotator = gui_annotator
        self.last_before_model = None
        self._instruction: Optional[str] = None
        self._pending: list[dict[str, Any]] = []
        self._previous_action = "start"
        self._previous_stage = "start"
        self._copy_call_observed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.agent, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._instruction = None
        self._pending.clear()
        self._previous_action = "start"
        self._previous_stage = "start"
        self._copy_call_observed = False
        self.last_before_model = None
        return self.agent.reset(*args, **kwargs)

    def predict(self, instruction: str, obs: Mapping[str, Any], *args: Any, **kwargs: Any) -> Any:
        if not self.use_evotree:
            return self.agent.predict(instruction, obs, *args, **kwargs)

        if not self.runtime.active or self._instruction != instruction:
            self.runtime.begin(instruction)
            self._instruction = instruction
            self._previous_action = "start"
            self._previous_stage = "start"
            self._copy_call_observed = False

        mapped = self.memory_observation(obs)
        # 用 available_tool_names 而不是 candidate_tool_names：注意栏要查的是
        # agent 真正能调的工具，而 computer_use 不在 obs["tool_list"] 里。
        # 重排不受影响 —— reorder_tool_list 只认 tool_list 里真实存在的项。
        names = available_tool_names(obs.get("tool_list"))
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
            calls = parse_tool_calls(response) if isinstance(response, str) else []
            if calls and len(calls) != len(actions):
                raise ValueError("Parsed tool calls and executed actions are not aligned")
            self._pending = [
                {"prediction": response, "action": action, "messages": messages,
                 **({"tool_call": calls[index]} if calls else {})}
                for index, action in enumerate(actions)
            ]
        else:
            self._pending = []
        return result

    def memory_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        mapped = dict(self.observation_mapper(obs))
        mapped["state_stage"] = self._previous_stage
        mapped["state_progress"] = {"copy_call_observed": self._copy_call_observed}
        # Real OSWorld observations contain app_info. Use existing structured
        # data and the preceding actual action to distinguish task stages;
        # no additional model calls or VM queries are required.
        if "app_info" in obs:
            if self._previous_action == "start":
                # The task template already identifies the initial task. Window
                # registration is asynchronous: its first frame may report
                # unknown or the real app for exactly the same reset snapshot.
                # Canonicalize only the memory root, never the model's obs.
                mapped.update(app="osworld-task-start", window="", dialog=None,
                              state_context="start")
                return mapped
            info = obs.get("app_info")
            if isinstance(info, str):
                try:
                    # Calc.env_info prints its JSON, and the runner then calls
                    # print_result, producing the same object twice on stdout.
                    decoder = json.JSONDecoder()
                    remaining = info.strip()
                    values = []
                    while remaining:
                        value, end = decoder.raw_decode(remaining)
                        values.append(value)
                        remaining = remaining[end:].strip()
                    info = values[0] if values and all(v == values[0] for v in values) else None
                except ValueError:
                    info = None
            structured = {}
            if isinstance(info, Mapping):
                structured = {key: info[key] for key in ("name", "active_sheet", "sheets", "data", "row_count", "column_count",
                              "schema", "scope", "current_slide_index", "slides") if key in info}
            encoded = json.dumps(structured, ensure_ascii=False, sort_keys=True, default=str)
            mapped["state_context"] = self._previous_action + ":" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
        return mapped

    def record_action_context(self, raw: Mapping[str, Any], note: Optional[str]) -> None:
        action = self.runtime.memory.distiller.distill_action(dict(raw))
        # A description enriches advice, not the observed environment state.
        # Keep signatures compatible across annotated and unannotated replay.
        args = {key: value for key, value in action.args.items() if key != "target"}
        payload = json.dumps([action.name, args, note is None], sort_keys=True, ensure_ascii=False)
        self._previous_action = hashlib.sha256(payload.encode()).hexdigest()[:16]
        self._previous_stage = action_stage(action, note is None)
        if action.name == "libreoffice_calc2.copy_cells_between_sheets":
            # This is a record of an error-free call, not proof of correct data.
            self._copy_call_observed = note is None
        elif action.name not in {"libreoffice_calc.env_info", "libreoffice_calc.save",
                                "libreoffice_calc.switch_active_sheet"}:
            self._copy_call_observed = False

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
        self.last_raw_action: Optional[dict[str, Any]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        self._last_observation = None
        self.last_score = None
        self.last_raw_action = None
        self.agent_wrapper._pending.clear()
        self.agent_wrapper._previous_action = "start"
        self.agent_wrapper._previous_stage = "start"
        self.agent_wrapper._copy_call_observed = False
        observation = self.env.reset(*args, **kwargs)
        # 首帧也要记住：runner 常写 `obs = env.reset()` 而不再调 _get_obs()，
        # 那样第一个动作的 before 观测就是 None，整步会被静默丢弃。
        if isinstance(observation, Mapping):
            self._last_observation = observation
        return observation

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
        self.last_raw_action = raw_action
        if _is_terminal_action(raw_action) or _is_client_error(raw_action):
            return result
        if self.agent_wrapper.gui_annotator is not None:
            annotation = self.agent_wrapper.gui_annotator(raw_action, before, next_observation)
            if annotation is not None:
                raw_action["gui_semantics"] = annotation
        info = result[3] if len(result) > 3 else {}
        # 报错即这条边这次失败：edge-level 成败在动作执行完就已知，不必等
        # 整轮 evaluator 结果（idea §0 写入第 3 步）。
        call = raw_action.get("tool_call") or {}
        executed = raw_action.get("action")
        tool_name = executed.get("action_type", "") if isinstance(executed, Mapping) else call.get("name", "")
        note = _note_from_step(info, next_observation, tool_name)
        mapped_before = self.agent_wrapper.memory_observation(before)
        self.agent_wrapper.record_action_context(raw_action, note)
        self.runtime.after_action(
            mapped_before,
            raw_action,
            next_observation=self.agent_wrapper.memory_observation(next_observation),
            note=note,
            action_success=note is None,
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
    gui_annotator: Optional[Callable] = None,
) -> tuple[Any, Any, Optional[EvoTreeRuntime]]:
    """构造可直接传给 OSWorld runner 的 ``(agent, env, runtime)``。

    ``use_evotree=False`` 返回原始 agent/env（不创建 runtime、不复制观测），
    这样 baseline 的行为和资源开销都保持原样。
    """

    if not use_evotree:
        return agent, env, None
    runtime = EvoTreeRuntime(memory)
    agent_wrapper = EvoTreeAgentWrapper(
        agent, runtime, use_evotree=True, observation_mapper=observation_mapper,
        gui_annotator=gui_annotator,
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

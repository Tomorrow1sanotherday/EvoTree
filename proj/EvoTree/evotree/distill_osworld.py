"""OSWorld-MCP 真实数据的三道清洗实现。

承 distill.py 的 Distiller 接口，把占位版换成能解析 OSWorld-MCP 真实轨迹格式
的实现。真实轨迹的 ``prediction`` 通常形如：

    <tool_call>{"name": "<工具名>", "arguments": {...}}</tool_call>
    <conclusion>agent 自己写的语义描述</conclusion>

动作归一化规则：
    - ``computer_use`` 是 GUI 动作，使用稳定的子动作类型归堆，丢弃坐标和具体输入值；
    - 其它 MCP 工具保留工具名，用排序后的参数名作为槽位，丢弃参数值；
    - DONE / terminate 是终止声明，不是普通动作。

这里不把 ``conclusion`` 作为动作主键：自然语言措辞跨轮不稳定。等有足够真实
数据后，可以在不改变接口的前提下加入更细的 GUI 语义归一化。
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

from .distill import RuleBasedDistiller
from .types import ActionKind, ActionTmpl

# 先提取完整 tool_call block，再用 JSONDecoder 解析，支持嵌套对象与数组。
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_CONCLUSION = re.compile(
    r"<conclusion>\s*(.*?)\s*</conclusion>", re.DOTALL | re.IGNORECASE
)

_GUI_TOOL_NAME = "computer_use"
_TERMINAL_ACTIONS = {"terminate", "done", "fail"}


def parse_prediction(prediction: str) -> Optional[dict[str, Any]]:
    """从 agent prediction 解析一条 tool_call 与 conclusion。

    使用 ``JSONDecoder.raw_decode``，而不是用右花括号的正则截 JSON；
    工具参数中可以包含嵌套对象、数组或字符串中的花括号。
    """
    if not prediction:
        return None
    match = _TOOL_CALL_BLOCK.search(prediction)
    if not match:
        return None

    try:
        call, _ = json.JSONDecoder().raw_decode(match.group(1).strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(call, dict):
        return None

    conclusion_match = _CONCLUSION.search(prediction)
    call["conclusion"] = (
        conclusion_match.group(1).strip() if conclusion_match else ""
    )
    return call


class OSWorldDistiller(RuleBasedDistiller):
    """OSWorld-MCP 真实数据的三道清洗。

    τ / φ 复用父类的规则实现（含 window 文件名归一化）；α 重写为真实
    prediction 解析。真实的 a11y tree → φ facts 仍由在线调用方提供。
    """

    def distill_action(self, raw_action: dict[str, Any]) -> ActionTmpl:
        """把一步的原始记录清洗成动作模板。

        OSWorld MCP 会把实际执行动作保存为 ``action_type/parameters``；模型
        ``prediction`` 可能含多个或过期 tool_call。结构化执行动作存在时优先
        使用它，否则解析 prediction；最后回退到占位版结构。
        """
        executed = raw_action.get("action")
        if isinstance(executed, Mapping):
            name = executed.get("action_type") or executed.get("name")
            if isinstance(name, str) and name.strip():
                args = executed.get("parameters", executed.get("arguments", {}))
                if not isinstance(args, Mapping):
                    args = {}
                normalized_name = name.strip()
                if normalized_name.startswith(f"{_GUI_TOOL_NAME}."):
                    args = dict(args)
                    args.setdefault("action", normalized_name.split(".", 1)[1])
                    normalized_name = _GUI_TOOL_NAME
                return self._action_from_call(
                    {"name": normalized_name, "arguments": dict(args)}
                )

        prediction = raw_action.get("prediction")
        if isinstance(prediction, str):
            call = parse_prediction(prediction)
            if call is not None:
                return self._action_from_call(call)
        return super().distill_action(raw_action)

    def _action_from_call(self, call: Mapping[str, Any]) -> ActionTmpl:
        """从解析出的 tool_call 构造 ActionTmpl。"""
        name = str(call.get("name", "")).strip() or "unknown_tool"
        args = call.get("arguments", {})
        if not isinstance(args, dict):
            # 格式异常时不把字符串误当作参数名序列。
            args = {}

        if name == _GUI_TOOL_NAME:
            # 子动作是目前最稳定的 GUI 归一化粒度；coordinate/text 等值不入库。
            sub_action = str(args.get("action", "gui")).strip() or "gui"
            return ActionTmpl(
                kind=ActionKind.GUI,
                name=f"{name}.{sub_action}",
                arg_slots=(),
            )

        # MCP 工具：保留工具名，参数名作槽位，参数值不进入动作主键。
        arg_slots = tuple(sorted(str(key) for key in args))
        return ActionTmpl(kind=ActionKind.TOOL, name=name, arg_slots=arg_slots)

    @staticmethod
    def is_parse_error(raw_action: Mapping[str, Any]) -> bool:
        """判断记录是否是模型调用失败，而不是环境中的真实动作。"""
        prediction = raw_action.get("prediction")
        return isinstance(prediction, str) and prediction.strip().lower() == "client error"

    @staticmethod
    def is_terminal(raw_action: Mapping[str, Any]) -> bool:
        """判断这一步是否是终止声明（terminate/DONE）。"""
        action = raw_action.get("action")
        if isinstance(action, str) and action.strip().upper() == "DONE":
            return True

        prediction = raw_action.get("prediction")
        if isinstance(prediction, str):
            call = parse_prediction(prediction)
            if call and call.get("name") == _GUI_TOOL_NAME:
                sub_action = str(call.get("arguments", {}).get("action", "")).lower()
                return sub_action in _TERMINAL_ACTIONS
        return False

    @classmethod
    def action_from_record(
        cls, raw_action: Mapping[str, Any]
    ) -> Optional[ActionTmpl]:
        """解析一条 OSWorld 记录；非动作记录返回 None。"""
        if cls.is_parse_error(raw_action) or cls.is_terminal(raw_action):
            return None
        action = raw_action.get("action")
        if isinstance(action, str) and action.strip().lower() == "reset":
            return None
        return cls().distill_action(dict(raw_action))

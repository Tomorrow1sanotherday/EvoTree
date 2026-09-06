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

# 无歧义的报错标记。**不含裸的 "error"** —— MCP 成功返回里就带
# ``is_error=False``，用子串匹配 "error" 会把每次成功调用都判成失败。
# 2026-09-06 实测：工具边平均 p 被压到 0.250，而真实报错为 0 次。
_ERROR_MARKERS = ("traceback", "exception:", "errno")

# MCP 的 CallToolResult 自带成败字段，优先按它判定。不同版本有 repr 形式
# （is_error=False）和 JSON 形式（"isError": false）两种写法。
_MCP_ERROR_TRUE = re.compile(r"is[_\s]*error\W{0,3}true", re.IGNORECASE)
_MCP_ERROR_FALSE = re.compile(r"is[_\s]*error\W{0,3}false", re.IGNORECASE)

# 输出原文可能是几百字符的 CallToolResult repr，整段进 prompt 会挤占上下文
# （实测 global_notes 单条 300+ 字符），入库前截断。
_NOTE_MAX_LEN = 200


def _truncate(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _NOTE_MAX_LEN else text[:_NOTE_MAX_LEN] + "…"


def extract_error_note(
    info: Any = None, execution_output: Any = None
) -> Optional[str]:
    """从一步的执行结果里提取错误信息；没有错误返回 None。

    这个判据只回答"**这次工具调用有没有报错**"，用来往边的 notes 和
    global_notes 写坑位笔记（idea §0 写入第 3 步的后半句）。

    它**不决定边的 n_success** —— 边的成功率按整轮任务成败记（见
    ``memory.write_episode``）。两者是不同的量：工具 API 返回成功，任务仍
    可能失败。2026-09-06 实测：20 任务里 12 个失败任务的真实工具报错为
    **0 次**，若拿它当成功率，``p`` 会退化成常量、失去区分度。

    判定顺序（从可靠到兜底）：

    1. ``info`` 里的显式 error/exception 字段；
    2. MCP ``CallToolResult`` 的 ``is_error`` —— **先查 True，再查 False 并
       直接放行**，避免 ``is_error=False`` 被第 3 步的子串匹配命中；
    3. 无歧义的报错标记（Traceback / Exception: / Errno）。

    Args:
        info: env.step 返回的 info 字典（含 error/exception/message）。
        execution_output: 执行输出文本，或含 ``exe_result`` 的观测字典。
                          DesktopEnv 把 MCP/Python 的执行输出放在下一个观测里。

    Returns:
        错误文本（已 strip、截断到 200 字），无错误时 None。
    """
    if isinstance(info, Mapping):
        for key in ("error", "exception", "message"):
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return _truncate(value)

    if isinstance(execution_output, Mapping):
        execution_output = execution_output.get("exe_result")
    if not isinstance(execution_output, str) or not execution_output.strip():
        return None

    # MCP 自带成败字段优先：明确 True 即失败，明确 False 即成功。
    if _MCP_ERROR_TRUE.search(execution_output):
        return _truncate(execution_output)
    if _MCP_ERROR_FALSE.search(execution_output):
        return None

    lowered = execution_output.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return _truncate(execution_output)
    return None


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

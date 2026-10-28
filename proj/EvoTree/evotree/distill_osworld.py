"""OSWorld-MCP 真实数据的三道清洗实现。

承 distill.py 的 Distiller 接口，把占位版换成能解析 OSWorld-MCP 真实轨迹格式
的实现。真实轨迹的 ``prediction`` 通常形如：

    <tool_call>{"name": "<工具名>", "arguments": {...}}</tool_call>
    <conclusion>agent 自己写的语义描述</conclusion>

动作归一化规则（主键粗、内容细，两层分开）：
    - 主键：``computer_use`` 用子动作类型归堆；其它 MCP 工具用工具名 + 排序后
      的参数名。坐标一律不进主键。
    - 内容：``text``（命令原文）、``keys``（快捷键）等**跨轮可复用**的值存进
      ``ActionTmpl.args``，不参与归堆。``coordinate``/``pixels``/``time``
      跨轮漂移，复用即错，仍然丢弃。

这里不把 ``conclusion`` 作为动作主键：自然语言措辞跨轮不稳定。
"""

from __future__ import annotations

import json
import ast
import re
from dataclasses import replace
from typing import Any, Mapping, Optional

from .distill import RuleBasedDistiller
from .types import ActionKind, ActionTmpl
from .gui_semantics import grounded_target

# 先提取完整 tool_call block，再用 JSONDecoder 解析，支持嵌套对象与数组。
_TOOL_CALL_BLOCK = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
)
_CONCLUSION = re.compile(
    r"<conclusion>\s*(.*?)\s*</conclusion>", re.DOTALL | re.IGNORECASE
)

GUI_TOOL_NAME = "computer_use"
"""OSWorld GUI 工具名。**公开常量**，因为 wrapper 也要用它。

这是环境里唯一真实存在的 GUI 工具：``key``/``type``/``click`` 是它的
``action`` 参数取值，不是独立工具。它由 agent 无条件提供
（``owl_agent.py``: ``functions = [computer_use.function]``），
**不出现在 ``obs["tool_list"]`` 里** —— 后者只装 MCP 工具，os 域任务下为空。
osworld_integration 拼候选工具名时必须显式带上它，否则注意栏拿不到名字可查。
"""

_TERMINAL_ACTIONS = {"terminate", "done", "fail"}

# --------------------------------------------------------------------------- #
# α 的语义取值：哪些参数值跨轮可复用（入 args），哪些是噪声（丢弃）
# --------------------------------------------------------------------------- #
# 判据是**跨轮可复用性**，不是"是不是参数"：
#   text —— 命令原文 `find ~/Desktop/photos -type f -iname '*.jpg' -exec cp ...`
#           换一轮照抄就能用，是这一步真正的内容；
#   keys —— 快捷键 ["CTRL","ALT","T"]，与 EE-MCP 的 "Ctrl+P for print dialog"
#           同型，是可直接执行的知识；
#   coordinate/pixels/time —— 跨轮漂移，复用即错，必须丢（原设计"坐标一律
#           丢弃"的初衷正确，错在把 text/keys 一并丢了）。
_REUSABLE_ARGS = ("text", "keys", "command", "url", "path")

# Keep executable values intact. Truncating a shell command creates advice
# which looks reusable but is syntactically incomplete.


def _format_arg_value(value: Any) -> str:
    """把一个参数值压成稳定的字符串。

    ``keys`` 是列表 ``["CTRL","ALT","T"]``，压成 ``CTRL+ALT+T`` —— 既是人话，
    也让同一快捷键的不同书写归一。其余按原文，超长截断。
    """
    if isinstance(value, (list, tuple)):
        text = "+".join(str(v).strip() for v in value if str(v).strip())
    else:
        text = str(value).strip()
    return text


def extract_reusable_args(args: Mapping[str, Any], *, tool: bool = False) -> dict[str, str]:
    """从一次调用的 arguments 里挑出跨轮可复用的语义值。

    这是 2026-09-10 修复的核心：原实现对 GUI 动作一律 ``arg_slots=()``、
    对 TOOL 只留参数名，导致边上完全不含内容 —— 实测一棵树里五条
    ``computer_use.click`` 无法区分，而 ``find ... -exec cp`` 这条整个任务
    的正确答案被丢弃。
    """
    out: dict[str, str] = {}
    for key in args if tool else _REUSABLE_ARGS:
        if key not in args:
            continue
        value = args[key]
        text = (json.dumps(value, ensure_ascii=False, sort_keys=True)
                if tool and not isinstance(value, str) else _format_arg_value(value))
        if text:
            out[key] = text
    return out

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
    info: Any = None, execution_output: Any = None, *, tool_name: str = ""
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
    # These mutators document False/None as business failure. A protocol-level
    # is_error=False only means the invocation completed. Do not apply this
    # rule to queries: their legitimate result can be false.
    false_is_failure = tool_name in {
        "libreoffice_calc.rename_sheet", "libreoffice_calc.switch_active_sheet",
        "libreoffice_calc.save", "libreoffice_calc.reorder_sheets",
        "libreoffice_calc2.copy_cells_between_sheets", "libreoffice_calc.set_column_values",
        "libreoffice_calc.copy_sheet",
    }
    if false_is_failure:
        payloads = []
        try:
            value = json.loads(execution_output)
            if isinstance(value, dict):
                payloads = [v.get("text") for v in value.get("content", []) if isinstance(v, dict)]
            elif value is False or value is None:
                payloads = ["false" if value is False else "null"]
        except ValueError:
            # CallToolResult repr contains Python string literals. Decode only
            # those literals; never eval the result or search arbitrary data.
            for match in re.finditer(r"\btext=(('(?:\\.|[^'\\])*')|(\"(?:\\.|[^\"\\])*\"))", execution_output):
                try:
                    payloads.append(ast.literal_eval(match.group(1)))
                except (ValueError, SyntaxError):
                    pass
        if any(isinstance(p, str) and (p.strip().lower() == "false" or
               tool_name == "libreoffice_calc.copy_sheet" and p.strip().lower() in {"none", "null"})
               for p in payloads):
            return f"{tool_name} returned failure; check its arguments and prerequisites."
    if _MCP_ERROR_FALSE.search(execution_output):
        return None

    lowered = execution_output.lower()
    if any(marker in lowered for marker in _ERROR_MARKERS):
        return _truncate(execution_output)
    return None


def parse_tool_calls(prediction: str) -> list[dict[str, Any]]:
    """Parse ordered calls, stopping at explicit termination."""
    calls = []
    for match in _TOOL_CALL_BLOCK.finditer(prediction or ""):
        try:
            call, _ = json.JSONDecoder().raw_decode(match.group(1).strip())
        except (ValueError, TypeError):
            continue
        if not isinstance(call, dict) or not isinstance(call.get("name"), str) or not isinstance(call.get("arguments"), dict):
            continue
        calls.append(call)
        if call["name"] == GUI_TOOL_NAME and str(call["arguments"].get("action", "")).lower() in _TERMINAL_ACTIONS:
            break
    return calls


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
        action = self._distill_action(raw_action)
        target = grounded_target(raw_action) if action.kind == ActionKind.GUI else None
        if target:
            return replace(action, args={**action.args, "target": target})
        return action

    def _distill_action(self, raw_action: dict[str, Any]) -> ActionTmpl:
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
                if normalized_name.startswith(f"{GUI_TOOL_NAME}."):
                    args = dict(args)
                    args.setdefault("action", normalized_name.split(".", 1)[1])
                    normalized_name = GUI_TOOL_NAME
                return self._action_from_call(
                    {"name": normalized_name, "arguments": dict(args)}
                )

        call = raw_action.get("tool_call")
        if isinstance(call, Mapping):
            return self._action_from_call(call)

        prediction = raw_action.get("prediction")
        if isinstance(prediction, str):
            calls = parse_tool_calls(prediction)
            if len(calls) > 1:
                raise ValueError("Multiple tool calls require the corresponding tool_call or structured executed action")
            call = calls[0] if calls else None
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

        if name == GUI_TOOL_NAME:
            # 子动作是最稳定的 GUI 归堆粒度，仍作主键；
            # text/keys 等可复用值走 args（不进主键，见 ActionTmpl 文档）。
            sub_action = str(args.get("action", "gui")).strip() or "gui"
            return ActionTmpl(
                kind=ActionKind.GUI,
                name=f"{name}.{sub_action}",
                arg_slots=(),
                args=extract_reusable_args(args),
            )

        # MCP keeps semantic parameter values as examples, outside the edge key.
        arg_slots = tuple(sorted(str(key) for key in args))
        return ActionTmpl(
            kind=ActionKind.TOOL,
            name=name,
            arg_slots=arg_slots,
            args=extract_reusable_args(args, tool=True),
        )

    @staticmethod
    def is_parse_error(raw_action: Mapping[str, Any]) -> bool:
        """判断记录是否是模型调用失败，而不是环境中的真实动作。"""
        prediction = raw_action.get("prediction")
        return isinstance(prediction, str) and prediction.strip().lower() == "client error"

    @staticmethod
    def is_terminal(raw_action: Mapping[str, Any]) -> bool:
        """判断这一步是否是终止声明（terminate/DONE）。"""
        action = raw_action.get("action")
        if isinstance(action, str):
            if action.strip().upper() in {"DONE", "FAIL", "TERMINATE"}:
                return True
            if action.strip().lower() == "reset" or action.strip().startswith(("import ", "pyautogui.")):
                return False
        if isinstance(action, Mapping):
            return str(action.get("action_type", action.get("action", ""))).upper() in {"DONE", "FAIL", "TERMINATE"}

        selected = raw_action.get("tool_call")
        if isinstance(selected, Mapping):
            return selected.get("name") == GUI_TOOL_NAME and str(selected.get("arguments", {}).get("action", "")).lower() in _TERMINAL_ACTIONS

        prediction = raw_action.get("prediction")
        if isinstance(prediction, str):
            call = parse_prediction(prediction)
            if call and call.get("name") == GUI_TOOL_NAME:
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

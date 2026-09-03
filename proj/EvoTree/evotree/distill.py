"""三道清洗：把不可数的真实桌面世界映射到可数的 (τ, φ, α) 空间。

对应 idea §4.1「三道清洗（系统中唯一需要精心设计的东西）」。

    distill_task(instruction)   -> Template    τ：这是哪类活儿
    distill_state(observation)  -> StateSig    φ：我在哪一步
    distill_action(raw_action)  -> ActionTmpl  α：做了个什么操作

设计立场（重要）：本模块把"如何清洗"抽象成三个**清洗器接口**，实现可替换。

当前状态：
    τ  已实现 —— 骨架抽取 + 模板库匹配 + 槽位可填性（见 template_lib）。
    φ  主键已实现（确定性提取 + window 归一化）；facts 仍留空待 VLM 填。
    α  工具侧已实现（工具名 + 参数槽精确匹配）；GUI 侧仍用动作类型代替
       语义描述（见 distill_osworld 的说明），是已知的粒度过粗问题。

原则（idea §4.1）：能确定性提取的绝不用 LLM。
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

from .normalize import normalize_window
from .template_lib import TemplateLibrary
from .types import ActionKind, ActionTmpl, StateSig, Template


# --------------------------------------------------------------------------- #
# 清洗器接口：三道清洗各有一个方法，实现可替换
# --------------------------------------------------------------------------- #
class Distiller(Protocol):
    """三道清洗的统一接口。占位实现与未来的 embedding/VLM 实现都遵循它。"""

    def distill_task(self, instruction: str) -> Template: ...
    def distill_state(self, observation: dict[str, Any]) -> StateSig: ...
    def distill_action(self, raw_action: dict[str, Any]) -> ActionTmpl: ...


# --------------------------------------------------------------------------- #
# τ —— 任务模板：把指令里会变的部分挖成槽位
# --------------------------------------------------------------------------- #
# 实现搬到 template_lib.TemplateLibrary：骨架抽取 + 模板库匹配 + 槽位可填性，
# 对应 idea §4.1「指令与模板库匹配 + 槽位抽取（槽能填满 = 对上），对不上新建」。
# 这里不再自己做正则替换，避免与模板库两套规则漂移。


# --------------------------------------------------------------------------- #
# φ —— 状态签名：主键确定性提取，facts 占位
# --------------------------------------------------------------------------- #
# 占位策略：从 observation dict 直接读 app/window/dialog（假设上游已从
# accessibility tree 抽好）。facts 先留空。真实版由 VLM 填 2–3 个是/否事实。


# --------------------------------------------------------------------------- #
# α —— 动作模板：工具精确、GUI 语义、坐标丢弃
# --------------------------------------------------------------------------- #
# 占位策略：
#   - 工具调用 raw_action = {"kind":"tool", "name": "...", "args": {...}}
#     → 工具名 + 参数名作为槽位（精确匹配）
#   - GUI 操作 raw_action = {"kind":"gui", "description": "...", ...}
#     → 语义描述作为 name，丢弃坐标
# 真实版应从 agent 的 function-calling 输出（tool_calls）解析，
# GUI 描述可能需要 VLM 归一化。


class RuleBasedDistiller:
    """规则版三道清洗。

    τ 已接入模板库（见 template_lib）；φ 的 facts 与 α 的 GUI 语义描述仍是
    占位实现，按 Distiller 接口可替换。
    """

    def __init__(self, template_library: Optional[TemplateLibrary] = None) -> None:
        """
        Args:
            template_library: 任务模板库。默认每个 distiller 自带一个——它随
                写入增长，与记忆同生命周期。多个 memory 需要共享分册口径时
                （如实验的多个 arm）显式传同一个实例。
        """
        self.template_library = template_library or TemplateLibrary()

    # ---- τ ---- #
    def distill_task(self, instruction: str) -> Template:
        """指令 → 任务模板（模板库匹配 + 槽位抽取，见 template_lib）。"""
        return self.template_library.match(instruction)

    # ---- φ ---- #
    def distill_state(self, observation: dict[str, Any]) -> StateSig:
        """观测 → 状态签名。主键从 observation 读并**归一化**，facts 占位。

        期望 observation 至少含 app / window；dialog 可选。

        window 经 normalize_window 归一化：抹掉任务无关的文件名（→ <file>），
        保留任务相关的状态标志（未保存 * 等）。这样同类任务换文件时，状态主键
        仍能对上、经验才能叠加（解决 spec/02 暴露的"状态对不上"问题）。

        TODO(B阶段): app/window/dialog 应从 accessibility tree 确定性提取；
        facts（2–3 个是/否事实）由 VLM 填。当前 app/dialog 直接读传入字段。
        """
        return StateSig(
            app=str(observation.get("app", "unknown")),
            window=normalize_window(str(observation.get("window", ""))),
            dialog=observation.get("dialog"),
            facts=dict(observation.get("facts", {})),
        )

    # ---- α ---- #
    def distill_action(self, raw_action: dict[str, Any]) -> ActionTmpl:
        """原始动作 → 动作模板。工具精确归堆，GUI 语义描述、丢坐标。

        raw_action 约定：
            工具: {"kind": "tool", "name": <工具名>, "args": {<参数名>: <值>}}
            GUI : {"kind": "gui",  "description": <语义描述>}

        TODO(B阶段): 从 agent 的 function-calling 输出解析；
        GUI 描述可能需 VLM 归一化到稳定说法。
        """
        kind_str = str(raw_action.get("kind", "")).lower()

        if kind_str == "tool":
            name = str(raw_action.get("name", "unknown_tool"))
            # 参数名作为槽位（排序保证稳定归堆），参数值丢弃（值是可变量）
            args = raw_action.get("args", {}) or {}
            arg_slots = tuple(sorted(str(k) for k in args.keys()))
            return ActionTmpl(kind=ActionKind.TOOL, name=name, arg_slots=arg_slots)

        # 默认按 GUI 处理：语义描述作为 name，坐标一律不进
        desc = str(raw_action.get("description", "")).strip() or "gui_action"
        return ActionTmpl(kind=ActionKind.GUI, name=desc, arg_slots=())


# 默认清洗器实例（闭环用）。替换实现时改这里或注入自定义 Distiller。
default_distiller = RuleBasedDistiller()

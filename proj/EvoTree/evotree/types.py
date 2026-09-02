"""EvoTree 核心数据类型。

本模块定义整个记忆系统的地基数据结构，对应 idea 文档 §0「方法定义卡」的 schema。
所有其它模块（清洗 / 森林 / 打分 / 检索）都建立在这里定义的类型之上。

三个基本概念（三道清洗的产物，详见 distill.py）：
    - Template (τ)  : 任务模板 —— "这是哪一类活儿"
    - StateSig (φ)  : 状态签名 —— "我现在在哪一步"
    - ActionTmpl(α) : 动作模板 —— "我做了个什么操作"

两种记录（经验森林的组成，详见 forest.py）：
    - StateNode : 状态节点（森林里的"点"）
    - ActionEdge: 动作边（森林里的"边"，边上挂着执行统计）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# --------------------------------------------------------------------------- #
# 动作类型：工具调用 vs GUI 操作
# --------------------------------------------------------------------------- #
class ActionKind(str, Enum):
    """动作的两大类。

    - TOOL: MCP 工具调用（工具名 + 参数）。精确匹配，可靠归堆。
    - GUI : 图形界面操作（点击 / 输入等）。靠语义描述归堆，坐标一律丢弃。

    注：CLI 经 open_shell 等工具触达，归入 TOOL 一类，不另设第三类
    （保证 TIR 口径与 OSWorld-MCP 官方一致）。
    """

    TOOL = "tool"
    GUI = "gui"


# --------------------------------------------------------------------------- #
# τ —— 任务模板
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Template:
    """任务模板 τ：把一类任务的骨架抽出来，可变部分挖成槽位。

    例："复制 A.xlsx 的 Revenue 列" 与 "复制 B.xlsx 的 Cost 列"
        → 同一模板 "复制 <文件> 的 <列名> 列"，槽位 = {文件, 列名}

    作用：经验按模板分册（每个 τ 一棵树），同类换参数的经验叠加攒厚。

    Attributes:
        template_id: 模板的稳定标识（用作 memory/<template_id>/tree.json 的目录名）。
        skeleton:    骨架文本，槽位以尖括号占位，如 "复制 <文件> 的 <列名> 列"。
        slots:       槽位名列表，如 ["文件", "列名"]。
    """

    template_id: str
    skeleton: str
    slots: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# φ —— 状态签名
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StateSig:
    """状态签名 φ：把"当前在任务的哪一步"提炼成可比对的字段。

    设计要点（对应 idea 决定 D1）：
        - 主键 = (app, window, dialog)：从 accessibility tree **确定性提取**，
          零噪声。字段相同即认定同一局面。
        - facts：少量任务相关的是/否事实（如"总页数已知"），由 VLM 填，
          **不进主键**，只在检索选边时作 side-condition，或方差报警后才升格。

    原则：能确定性提取的绝不用 LLM。

    Attributes:
        app:    正在使用的应用（如 "impress"）。
        window: 归一化后的窗口标题（如 "*.pptx — Impress"）。
        dialog: 当前弹窗标识；无弹窗为 None。
        facts:  任务相关的是/否事实；不参与身份判定，故用可变 dict 但不计入 key。
    """

    app: str
    window: str
    dialog: Optional[str] = None
    # facts 不进主键：用 default_factory 且在 key() 里显式排除
    facts: dict[str, bool] = field(default_factory=dict, compare=False)

    def key(self) -> tuple[str, str, str]:
        """返回用于归堆的主键（仅 app + window + dialog，不含 facts）。"""
        return (self.app, self.window, self.dialog or "")


# --------------------------------------------------------------------------- #
# α —— 动作模板
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ActionTmpl:
    """动作模板 α：把这一步的操作归一化成可比对、可归堆的形式。

    - 工具调用：工具名 + 参数换成槽位，如 duplicate_slide(<IDX>, <POS>)。
      精确匹配，零成本、可靠归堆。
    - GUI 操作：一句语义描述，如 "点击 Format 菜单"。**坐标一律丢弃**
      （坐标跨轮漂移，是经验不可复用的根源）。

    Attributes:
        kind:      TOOL 还是 GUI。
        name:      工具名（TOOL）或语义描述（GUI）。
        arg_slots: 参数槽位名（仅 TOOL 有意义），如 ("idx", "pos")。
    """

    kind: ActionKind
    name: str
    arg_slots: tuple[str, ...] = ()

    def key(self) -> tuple[str, str, str]:
        """返回用于归堆的主键。"""
        return (self.kind.value, self.name, ",".join(self.arg_slots))


# --------------------------------------------------------------------------- #
# 状态节点 —— 森林里的"点"
# --------------------------------------------------------------------------- #
@dataclass
class StateNode:
    """状态节点：经验森林里的一个局面。

    对应 idea §0 schema 的"状态"记录。节点身份由 sig.key() 决定，
    不同来路到达同一局面会自动汇合到同一节点（这正是"森林是图"的来源）。

    Attributes:
        node_id: 节点在树内的稳定标识（如 "s3"）。
        sig:     该节点的状态签名 φ。
        visits:  该节点被访问过的总次数（= 所有出边 n 之和，UCB 的分母用）。
    """

    node_id: str
    sig: StateSig
    visits: int = 0


# --------------------------------------------------------------------------- #
# 动作边 —— 森林里的"边"，边上挂着执行统计
# --------------------------------------------------------------------------- #
@dataclass
class ActionEdge:
    """动作边：从某状态做某动作后的执行统计。**这是 EvoTree 的核心载体。**

    对应 idea §0 schema 的"边"记录。护城河一（记统计不记文本）就体现在这里：
    边上挂的是可数的充分统计量，而非一句自然语言。

    Attributes:
        from_id:      起点状态节点 id。
        action:       动作模板 α。
        to_success:   成功后落到的状态节点 id（**"连起来"的指针，前瞻靠它**）。
        to_fail:      失败后落到的状态节点 id。
        n:            这条边被走过的总次数。
        n_success:    其中成功的次数。
        steps_after_success:
            每次成功时，"这一步之后还走了几步"的列表。
            由写入时从终点倒数得到（cost-to-go 的原始数据，详见 forest.py）。
        notes:        这条边踩过的坑（工具报错等），也抄送 global_notes。

    统计量的两个派生值（在 scoring.py 里计算，不存这里，避免冗余）：
        p = n_success / n            成功率（可靠性）
        d = mean(steps_after_success) 平均剩余步数（效率）
    """

    from_id: str
    action: ActionTmpl
    to_success: Optional[str] = None
    to_fail: Optional[str] = None
    n: int = 0
    n_success: int = 0
    steps_after_success: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def edge_key(self) -> tuple[str, tuple[str, str, str]]:
        """返回用于归堆的主键 = (起点节点, 动作主键)。

        同一 (from, action) 的多次执行落进同一条边 —— 自动归堆，
        杜绝"挑最快那次记录"式的自欺（idea §4.2 的自动性质之一）。
        """
        return (self.from_id, self.action.key())

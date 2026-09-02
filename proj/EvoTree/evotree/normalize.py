"""状态签名归一化：把 window 标题里的"任务无关易变量"抹掉，保留"状态标志"。

解决 spec/02 暴露的真实问题：学习时状态 window="data.xlsx"、新任务
window="data2.xlsx"，主键不同 → 找不到状态 → 推荐栏空返回（假冷启动）。

核心思想（有深度，不能一刀切）：window 标题混了两类信息，必须区别对待：

    任务无关易变量（归一化掉）：具体文件名 data.xlsx / report_2024.pptx
        → 对"复制列""重命名"这类任务，用哪个文件是无关的，应归一化成 <file>，
          这样同类任务的状态才能对上、经验才能叠加。

    任务相关状态标志（保留）：
        - 未保存标记 "*" / "●"      → 反映"是否有未保存改动"，是真实状态差异
        - 应用名 Calc / Impress     → 不同应用是不同局面
        - 其它非文件名的稳定词       → 保留

设计对齐：文件名的归一化规则与 τ 模板的槽位抽取（distill._FILENAME）一致 ——
τ 把指令里的文件名挖成槽位，φ 把 window 里的文件名归一化成 <file>，二者对齐，
保证"同一类任务、同一步局面"在 τ 和 φ 两个维度都能归堆。
"""

from __future__ import annotations

import re

# 文件名：形如 a.pptx / report_2024.xlsx / data-final.csv（与 distill._FILENAME 对齐）
_FILENAME = re.compile(r"\b[\w-]+\.\w+\b")

# 未保存修改标记：常见于标题开头或文件名后（LibreOffice/多数编辑器用 *）
_MODIFIED_MARKS = ("*", "●", "•")

# 归一化后的文件名占位符
_FILE_PLACEHOLDER = "<file>"


def normalize_window(window: str) -> str:
    """归一化 window 标题：抹掉具体文件名，保留状态标志（未保存标记等）。

    Args:
        window: 原始窗口标题，如 "data.xlsx — Calc" 或 "*.pptx — Impress *"。

    Returns:
        归一化后的标题，文件名替换成 <file>，未保存标记保留。

    Examples:
        "data.xlsx — Calc"        -> "<file> — Calc"
        "data2.xlsx — Calc"       -> "<file> — Calc"   (与上者归堆)
        "report.pptx — Impress *" -> "<file> — Impress *"  (保留 * 未保存标记)
        "Files"                   -> "Files"           (无文件名，原样)
    """
    if not window:
        return window

    # 1) 记录是否有未保存标记（在抹文件名前先探测，避免被误伤）
    has_modified = _detect_modified(window)

    # 2) 把所有文件名替换成占位符
    normalized = _FILENAME.sub(_FILE_PLACEHOLDER, window)

    # 3) 折叠多余空白
    normalized = re.sub(r"\s+", " ", normalized).strip()

    # 4) 确保未保存标记以稳定形式保留在末尾（若原本有、但被折叠丢了）
    if has_modified and not _detect_modified(normalized):
        normalized = f"{normalized} *"

    return normalized


def _detect_modified(window: str) -> bool:
    """检测 window 标题是否带"未保存修改"标记。"""
    stripped = window.strip()
    for mark in _MODIFIED_MARKS:
        # 标记出现在开头或结尾都算（不同应用位置不同）
        if stripped.startswith(mark) or stripped.endswith(mark):
            return True
    return False

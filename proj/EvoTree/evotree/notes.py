"""全局坑位手册（注意栏）：跨任务共享的程序知识。

对应 idea 推论 5 的"注意栏"与 §4 的 global_notes。

为什么它必须独立于经验森林存在（第一性原理）：
    "这个工具有什么坑"是**任务无关**的知识（如 duplicate_slide 的 pos 不能
    超过总页数 —— 复制/重排/动画任务都会踩）。如果塞进某棵任务树，别的任务
    学不到、或每棵树各抄一份。所以抽出来共享一本。

两个价值：
    1. 所有任务共享同一份避坑经验（写一次，处处命中）。
    2. 冷启动兜底：全新任务的经验树是空的（推荐栏无），但注意栏照常命中
       —— "没干过这活，见过世面"。

一条硬规则（idea §4.4）：
    **[注意]栏永不推荐**。它不知道你的任务目标，只配说"小心"，不配说"走这条"。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class GlobalNotes:
    """全局坑位手册：按工具名索引的、任务无关的坑位/报错笔记。

    数据结构：tool_name -> 去重的笔记列表。
    """

    def __init__(self) -> None:
        self._notes: dict[str, list[str]] = {}

    def add_note(self, tool_name: str, note: str) -> None:
        """为某工具追加一条坑位笔记（自动去重）。

        写入时机：轨迹里工具报错时，把信息抄一份进来（同时也进对应边的 notes）。
        """
        note = note.strip()
        if not note:
            return
        bucket = self._notes.setdefault(tool_name, [])
        if note not in bucket:
            bucket.append(note)

    def query(self, tool_name: str) -> list[str]:
        """查某工具的坑位笔记（检索第 3 拍[注意]用）。无则返回空列表。"""
        return list(self._notes.get(tool_name, []))

    def tools_covered(self) -> list[str]:
        """已收录坑位的工具名列表（分析/统计用）。"""
        return list(self._notes.keys())

    # ------------------------------------------------------------------ #
    # JSON 持久化（global_notes.json，跨任务共享）
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, list[str]]:
        return dict(self._notes)

    @classmethod
    def from_dict(cls, data: dict[str, list[str]]) -> "GlobalNotes":
        obj = cls()
        obj._notes = {k: list(v) for k, v in data.items()}
        return obj

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "GlobalNotes":
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

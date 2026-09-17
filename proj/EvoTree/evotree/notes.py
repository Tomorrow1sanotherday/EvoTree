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

    **索引键有两种粒度**（2026-09-17 实验三的教训，必须记住）：

    - 顶层工具名：``os.open_shell`` —— MCP 工具表里真实存在的名字；
    - 子动作名：``computer_use.key`` —— GUI 动作模板名（``ActionTmpl.name``），
      环境里并没有叫这个名字的工具，``computer_use`` 才是工具，``key``/``type``
      是它的 ``action`` 参数取值。

    两者都会作为 key 出现：写入侧记的是 ``ActionTmpl.name``（GUI 动作必然带
    子动作后缀），而检索侧手里只有 ``candidate_tools``（顶层工具名）。实验三
    因此整轮 [注意] 栏为空 —— 拿 ``computer_use`` 去精确查 ``computer_use.key``
    永远查不到，且静默返回空列表，没有任何报错。

    所以查询一律走 ``query_namespace``（工具名 + 其子动作命名空间），
    ``query`` 只做单键精确查询，保留给明确知道自己要哪个键的调用方。
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
        """单键精确查询。**检索请用 ``query_namespace``**，见类 docstring。"""
        return list(self._notes.get(tool_name, []))

    def query_namespace(self, tool_name: str) -> list[str]:
        """查某工具及其子动作命名空间下的全部坑位笔记（检索第 3 拍[注意]用）。

        ``query_namespace("computer_use")`` 同时命中 ``computer_use`` 本身与
        ``computer_use.key``、``computer_use.type`` 等子动作键 —— 因为检索侧
        手里只有顶层工具名，而写入侧记的是带子动作的动作模板名（见类
        docstring 的实验三教训）。

        顺序：先本键，再子动作键按键名排序，保证渲染稳定可复现。
        去重按笔记原文（同一条坑挂在多个子动作上时只说一次）。
        """
        if not tool_name:
            return []
        seen: set[str] = set()
        result: list[str] = []
        prefix = f"{tool_name}."
        keys = [tool_name] + sorted(k for k in self._notes if k.startswith(prefix))
        for key in keys:
            for note in self._notes.get(key, []):
                if note not in seen:
                    seen.add(note)
                    result.append(note)
        return result

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

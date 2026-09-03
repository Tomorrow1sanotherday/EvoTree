"""τ 模板库：把指令归一到任务模板（idea §4.1 第一道清洗）。

文档要求："指令 embedding 与模板库匹配 + 槽位抽取（槽能填满 = 对上），
对不上新建模板"，并特别标注 **τ 匹配的准确率/召回直接决定分册是否有效**。

## 为什么不直接用 embedding

真实部署环境（VM 内跑评测、离线 wheelhouse）没有 embedding 模型，也不该为
一道清洗引入 GPU 依赖。这里用**词级骨架 + 编辑距离**逼近 embedding 的同类
判定：先把易变量挖成槽位得到骨架，再在模板库里找骨架"几乎一样"的模板
（只差少数几个可槽位化的词）。这保持了确定性——同样的输入永远得到同样的
模板，可复现、可审计，符合"能确定性提取的绝不用 LLM"的原则。

接口按 embedding 版预留：``TemplateLibrary.match`` 只依赖"两个骨架有多像"，
把 ``_skeleton_distance`` 换成向量余弦即可升级，不动调用方。

## 修掉的两个真实缺陷

**假合并（截断碰撞）**：旧实现把骨架截断到 40 字符做 template_id，实测
``Create a new spreadsheet document and name it budget`` 与
``... name it revenue`` 得到同一个 id，两个不同任务的经验混进同一棵树——
直接违反推论 2「混记成 50% 对谁都没用」。现在 id = 可读前缀 + 完整骨架哈希，
长指令不再碰撞。

**假分裂**：旧实现只挖引号/文件名/数字，挖不到的可变词（列名、目录名）导致
同模板分家。文档 §4.1 的原例 ``复制 A.xlsx 的 Revenue 列`` 与
``复制 B.xlsx 的 Cost 列`` 实测被判成两棵树，地基②（统计稀疏性）被打穿。
现在由模板库做二次归并：骨架只差少数词时，把差异位置补挖成槽位后归入
已有模板。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .types import Template

# 易变量：引号内容、文件名、数字、路径。顺序敏感——先挖大结构再挖小结构，
# 否则 "a.xlsx" 里的数字会被先挖走。
_QUOTED = re.compile(r'"[^"]*"|\'[^\']*\'|“[^”]*”|「[^」]*」')
_PATH = re.compile(r"(?:~|\.{1,2})?/[\w.\-/]+|[A-Za-z]:\\[\w.\-\\]+")
_FILENAME = re.compile(r"\b[\w-]+\.[A-Za-z][\w]{0,9}\b")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")

_SLOT = re.compile(r"<arg\d+>")

# 骨架相差几个词以内仍算同一模板。1 允许"Revenue 列/Cost 列"这类单词替换；
# 放大会开始把不同任务并到一起（Copy vs Move），得不偿失。
_MAX_TOKEN_DIFF = 1

# 少于这么多词的指令不做模糊归并：短指令里改一个词往往就是另一件事
# （"undo" vs "redo"、"save" vs "close"）。
_MIN_TOKENS_FOR_FUZZY = 4

# template_id 里可读前缀的长度。哈希保证唯一性，前缀只为人眼可读，
# 所以截断在这里是安全的——不再像旧实现那样用截断结果做身份。
_READABLE_PREFIX = 40


def _slugify(text: str) -> str:
    """把骨架压成文件系统安全的可读片段（Windows 下 <> 非法）。"""
    slug = re.sub(r"<(\w+)>", r"_\1_", text.strip())
    slug = re.sub(r"\s+", "_", slug)
    slug = re.sub(r"[^\w-]", "", slug)
    return slug or "task"


def extract_skeleton(instruction: str) -> tuple[str, list[str]]:
    """把指令里的易变量挖成 ``<argN>``，返回 (骨架, 槽位填充值)。

    槽位值是**实际填进去的内容**（如 ``A.xlsx``），不是槽位名——文档要的
    "槽能填满 = 对上"需要这些值来判断可填性。
    """
    values: list[str] = []
    skeleton = instruction.strip()

    def _sub(pattern: re.Pattern, text: str) -> str:
        def repl(match: re.Match) -> str:
            values.append(match.group(0))
            return f"<arg{len(values) - 1}>"

        return pattern.sub(repl, text)

    for pattern in (_QUOTED, _PATH, _FILENAME, _NUMBER):
        skeleton = _sub(pattern, skeleton)
    return re.sub(r"\s+", " ", skeleton).strip(), values


def _tokens(skeleton: str) -> list[str]:
    """按词切分骨架；槽位视作单个词，标点单独成词后丢弃。"""
    parts = re.findall(r"<arg\d+>|[^\W_]+|[^\w\s]", skeleton, flags=re.UNICODE)
    return [p for p in parts if not re.fullmatch(r"[^\w\s]", p)]


def _diff_positions(left: list[str], right: list[str]) -> Optional[list[int]]:
    """两个等长词序列里不同词的位置；长度不等返回 None。

    只比对等长序列：真正的同模板换参数不改变词数（"Revenue 列"→"Cost 列"），
    而词数不同通常意味着任务结构不同。这个约束让归并保守、不易误合并。
    """
    if len(left) != len(right):
        return None
    return [i for i, (a, b) in enumerate(zip(left, right)) if a.lower() != b.lower()]


@dataclass
class _Entry:
    """模板库里的一条模板：骨架 + 它的词序列（缓存，避免重复切分）。"""

    template_id: str
    skeleton: str
    tokens: list[str] = field(default_factory=list)


class TemplateLibrary:
    """任务模板库：先精确匹配骨架，再按"只差少数可槽位化的词"归并。

    调用方只需 ``match(instruction)``。库随写入自然增长，不需要预先构建。

    Args:
        max_token_diff: 允许归并的最大词差。0 = 只精确匹配（关闭模糊归并）。
    """

    def __init__(self, max_token_diff: int = _MAX_TOKEN_DIFF) -> None:
        self.max_token_diff = max_token_diff
        self._by_skeleton: dict[str, _Entry] = {}

    def __len__(self) -> int:
        return len(self._by_skeleton)

    @property
    def templates(self) -> Iterable[str]:
        """已知模板 id（调试与 τ 匹配率分析用）。"""
        return (entry.template_id for entry in self._by_skeleton.values())

    def match(self, instruction: str) -> Template:
        """指令 → 模板。命中已有模板则复用其 id，否则登记新模板。

        三级判定，从严到宽：

        1. **骨架完全相同** → 同模板（换文件名/数字的情形，最常见）。
        2. **骨架只差 ≤ max_token_diff 个词** → 同模板，差异词补挖成槽位。
           这修掉了"Revenue 列 / Cost 列"这类挖不到的可变词造成的假分裂。
        3. 否则新建模板。
        """
        skeleton, values = extract_skeleton(instruction)
        exact = self._by_skeleton.get(skeleton)
        if exact is not None:
            return Template(
                template_id=exact.template_id,
                skeleton=exact.skeleton,
                slots=tuple(f"arg{i}" for i in range(len(values))),
            )

        tokens = _tokens(skeleton)
        neighbour = self._find_neighbour(tokens)
        if neighbour is not None:
            entry, positions = neighbour
            # 差异位置补挖成槽位：把两边不同的词都换成占位符，得到更一般的
            # 骨架，并让两条指令从此都落到同一棵树上。
            generalized = self._generalize(entry, tokens, positions)
            self._by_skeleton[skeleton] = _Entry(
                template_id=generalized.template_id,
                skeleton=generalized.skeleton,
                tokens=_tokens(generalized.skeleton),
            )
            return generalized

        entry = _Entry(
            template_id=self._make_id(skeleton), skeleton=skeleton, tokens=tokens
        )
        self._by_skeleton[skeleton] = entry
        return Template(
            template_id=entry.template_id,
            skeleton=skeleton,
            slots=tuple(f"arg{i}" for i in range(len(values))),
        )

    def _find_neighbour(
        self, tokens: list[str]
    ) -> Optional[tuple[_Entry, list[int]]]:
        """找一个"只差少数词"的已有模板。差异最少的优先。"""
        if self.max_token_diff <= 0 or len(tokens) < _MIN_TOKENS_FOR_FUZZY:
            return None

        best: Optional[tuple[_Entry, list[int]]] = None
        for entry in self._by_skeleton.values():
            positions = _diff_positions(entry.tokens, tokens)
            if positions is None or not positions:
                continue
            if len(positions) > self.max_token_diff:
                continue
            # 已经是槽位的位置不算真差异；差异词必须是可槽位化的普通词。
            if any(_SLOT.fullmatch(tokens[i]) for i in positions):
                continue
            if best is None or len(positions) < len(best[1]):
                best = (entry, positions)
        return best

    def _generalize(
        self, entry: _Entry, tokens: list[str], positions: list[int]
    ) -> Template:
        """把差异位置换成新槽位，就地放宽已有模板的骨架。

        模板 id 保持不变——树已经按这个 id 存了经验，改 id 会让历史经验失联。
        """
        merged = list(entry.tokens)
        slot_count = len(_SLOT.findall(entry.skeleton))
        for offset, index in enumerate(positions):
            merged[index] = f"<arg{slot_count + offset}>"
        skeleton = " ".join(merged)

        entry.skeleton = skeleton
        entry.tokens = merged
        return Template(
            template_id=entry.template_id,
            skeleton=skeleton,
            slots=tuple(f"arg{i}" for i in range(slot_count + len(positions))),
        )

    @staticmethod
    def _make_id(skeleton: str) -> str:
        """稳定的 template_id = 可读前缀 + 完整骨架哈希。

        旧实现直接把骨架截断到 40 字符当 id，长指令共享前缀就会碰撞——实测
        ``Create a new spreadsheet document and name it {budget,revenue}``
        撞成同一棵树。哈希取自**完整**骨架，前缀只为人眼可读。
        """
        digest = hashlib.sha1(skeleton.encode("utf-8")).hexdigest()[:8]
        return f"{_slugify(skeleton)[:_READABLE_PREFIX]}-{digest}"

    # ------------------------------------------------------------------ #
    # 持久化：memory/templates.json（idea §9 的数据结构之一）
    # ------------------------------------------------------------------ #
    # 模板库必须和树一起落盘：树按 template_id 存，若重启后库为空，同一条
    # 指令会重新登记成新 id，历史经验直接失联。实测过这个失效——先学到
    # "Revenue 列"再归并"Cost 列"，重启后单独出现的 "Cost 列" 会新开一棵树。
    def to_dict(self) -> dict:
        return {
            "max_token_diff": self.max_token_diff,
            "entries": [
                {"template_id": e.template_id, "skeleton": e.skeleton, "key": key}
                for key, e in self._by_skeleton.items()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemplateLibrary":
        library = cls(max_token_diff=data.get("max_token_diff", _MAX_TOKEN_DIFF))
        for item in data.get("entries", []):
            skeleton = item["skeleton"]
            library._by_skeleton[item.get("key", skeleton)] = _Entry(
                template_id=item["template_id"],
                skeleton=skeleton,
                tokens=_tokens(skeleton),
            )
        return library

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "TemplateLibrary":
        """从 templates.json 载入；文件不存在时返回空库。"""
        path = Path(path)
        if not path.exists():
            return cls()
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


__all__ = ["TemplateLibrary", "extract_skeleton"]

"""在线 agent 接入层。

本模块不绑定某个具体的 agent 框架，只提供稳定 hook，把 EvoTree 放进任意
GUI/工具 agent 的决策循环：

    runtime.begin(instruction)
    result = runtime.before_model(observation, candidate_tools)
    prompt = inject_cheatsheet(original_prompt, result.text)
    action = agent.generate(prompt)
    runtime.after_action(observation, action, next_observation=...)
    runtime.finish(success=True, score=1.0)

设计边界：
    - ``before_model`` 只读记忆，不修改森林；
    - ``after_action`` 只暂存在线轨迹，不提前写成功统计；
    - ``finish`` 拿到客观的 episode-level 结果后，一次性写入；
    - agent 原始动作始终保留，清洗交给注入的 Distiller；
    - write_enabled=False 时，冻结期间既不更新记忆，也不落盘。

EvoTree 不负责执行动作、不改变 agent 的决策权，只在模型调用前提供 advisory
小抄，并在 episode 结束时记录结果。
"""

from __future__ import annotations

import logging
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .episode import Episode, EpisodeStep, RuntimeStep
from .memory import EvoTreeMemory
from .retrieval import Cheatsheet, render_cheatsheet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BeforeModelResult:
    """模型调用前的记忆结果。"""

    cheatsheet: Cheatsheet
    text: str
    reranked_tools: list[str]


class EvoTreeRuntime:
    """EvoTree 在线运行时：管理一轮 agent episode 的 read/write hook。"""

    def __init__(self, memory: EvoTreeMemory) -> None:
        self.memory = memory
        self.instruction: Optional[str] = None
        self._steps: list[RuntimeStep] = []
        self._retrieval_step = 0

    @property
    def active(self) -> bool:
        """当前是否有已开始但尚未 finish 的 episode。"""
        return self.instruction is not None

    def begin(self, instruction: str) -> None:
        """开始一轮任务，并清空上一轮未提交的动作。"""
        if not instruction.strip():
            raise ValueError("instruction must not be empty")
        self.instruction = instruction
        self._steps.clear()
        self._retrieval_step = 0

    def before_model(
        self,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]] = None,
    ) -> BeforeModelResult:
        """在每次模型调用前检索记忆。"""
        if self.instruction is None:
            raise RuntimeError("Call begin() before before_model()")
        cheatsheet = self.memory.read_cheatsheet(
            self.instruction, observation, candidate_tools
        )
        result = BeforeModelResult(
            cheatsheet=cheatsheet,
            text=render_cheatsheet(cheatsheet),
            reranked_tools=list(cheatsheet.reranked_tools),
        )
        self._log_retrieval(observation, candidate_tools, result)
        self._write_retrieval_event(observation, candidate_tools, result)
        self._retrieval_step += 1
        return result

    def _write_retrieval_event(
        self,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]],
        result: BeforeModelResult,
    ) -> None:
        """Persist an audit-friendly retrieval event when memory is on disk.

        Logging is often configured at WARNING by the runner, so INFO-only
        diagnostics are insufficient for experiment forensics. JSONL keeps the
        event structured without coupling the runtime to a logging setup.
        """
        root = getattr(self.memory, "root_dir", None)
        # Keep frozen/read-only callers side-effect free by default. Real
        # experiments opt in explicitly so the audit stream is available.
        if root is None or os.environ.get("EVOTREE_RETRIEVAL_LOG", "") not in {
            "1", "true", "yes"
        }:
            return
        path = Path(root) / "retrieval.jsonl"
        sig = self.memory.distiller.distill_state(observation)
        template = self.memory.distiller.distill_task(self.instruction or "")
        tree = self.memory.forest.get_tree(template.template_id)
        node = tree.node_for_signature(sig) if tree is not None else None
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "step": self._retrieval_step,
            "phi": sig.key(),
            "tau": template.template_id,
            "tree": "hit" if tree is not None else "miss",
            "node": "hit" if node is not None else "miss",
            "recommendations": [
                {
                    "action": rec.action.name,
                    "args": rec.action.args,
                    "n": rec.n,
                    "success_rate": rec.success_rate,
                    "mean_steps_to_go": rec.mean_steps_to_go,
                    "score": rec.score,
                    "low_confidence": rec.low_confidence,
                }
                for rec in result.cheatsheet.recommendations
            ],
            "cautions": list(result.cheatsheet.cautions),
            "candidate_tools": list(candidate_tools or []),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _log_retrieval(
        self,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]],
        result: BeforeModelResult,
    ) -> None:
        """把这一步的检索条件与命中情况记一行，供事后诊断。

        为什么需要它（2026-09-18）：实验三、四两轮都栽在"注入为什么没发生"
        上，而事后只能靠翻 runner 的 debug 日志、反推 prompt 里有没有小抄 ——
        既慢又只能看到结果，看不到**当时的检索条件**（φ 是什么、树在不在、
        节点命没命中）。实验四暴露的 P1（注入只在前 3 步，之后 φ 从 unknown
        变成真实 app、与种子节点对不上）就是这样一点点推出来的。

        这一行日志把条件直接写下来，下次不用再猜。走 logging，默认 WARNING
        级别下不输出，不影响正常运行的日志量。
        """
        if not logger.isEnabledFor(logging.INFO):
            return
        sig = self.memory.distiller.distill_state(observation)
        template = self.memory.distiller.distill_task(self.instruction or "")
        tree = self.memory.forest.get_tree(template.template_id)
        node = tree.node_for_signature(sig) if tree is not None else None
        logger.info(
            "[evotree] phi=%s tau=%s tree=%s node=%s recs=%d cautions=%d tools=%s",
            sig.key(),
            template.template_id,
            "hit" if tree is not None else "miss",
            "hit" if node is not None else "miss",
            len(result.cheatsheet.recommendations),
            len(result.cheatsheet.cautions),
            list(candidate_tools or []),
        )

    def after_action(
        self,
        observation: dict[str, Any],
        raw_action: dict[str, Any],
        *,
        next_observation: Optional[dict[str, Any]] = None,
        note: Optional[str] = None,
        action_success: Optional[bool] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """动作执行后暂存一步，不更新森林统计。

        Args:
            action_success: 这一步动作本身成没成（边级）。留空时按"有报错笔记
                即失败"推断，让只传 note 的旧调用方也能得到正确的边级成败。
                整轮成败另由 ``finish`` 提供，两者互不替代。
        """
        if self.instruction is None:
            raise RuntimeError("Call begin() before after_action()")
        self._steps.append(
            RuntimeStep(
                step_index=len(self._steps),
                observation=dict(observation),
                raw_action=dict(raw_action),
                next_observation=(
                    dict(next_observation) if next_observation is not None else None
                ),
                note=note,
                action_success=(note is None) if action_success is None else action_success,
                metadata=dict(metadata or {}),
            )
        )

    def finish(
        self,
        *,
        success: Optional[bool] = None,
        score: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Episode:
        """结束一轮任务，用客观 episode-level 结果写入记忆。"""
        if self.instruction is None:
            raise RuntimeError("Call begin() before finish()")
        if success is None and score is None:
            raise ValueError("finish requires success= or score=")
        if score is not None:
            score_success = score > 0
            if success is not None and success != score_success:
                raise ValueError("success and score disagree")
            success = score_success

        episode = Episode(
            instruction=self.instruction,
            steps=tuple(
                EpisodeStep(
                    step_index=step.step_index,
                    observation=step.observation,
                    raw_action=step.raw_action,
                    next_observation=(
                        step.next_observation
                        if step.next_observation is not None
                        else (
                            self._steps[index + 1].observation
                            if index + 1 < len(self._steps)
                            else None
                        )
                    ),
                    note=step.note,
                    action_success=step.action_success,
                    metadata=step.metadata,
                )
                for index, step in enumerate(self._steps)
            ),
            success=bool(success),
            source="online",
            metadata={"score": score, **dict(metadata or {})},
        )
        self.memory.write_episode(episode)
        self.memory.save()
        self.instruction = None
        self._steps.clear()
        return episode


def inject_cheatsheet(prompt: str, cheatsheet_text: str) -> str:
    """把 advisory 小抄追加到 agent prompt；空小抄不改变原 prompt。"""
    if not cheatsheet_text.strip():
        return prompt
    return f"{prompt.rstrip()}\n\n{cheatsheet_text.strip()}"

"""顶层封装：EvoTreeMemory —— 对外的唯一入口。

把三道清洗（distill）、经验森林（forest）、全局坑位手册（notes）、
五拍检索（retrieval）串成完整闭环。

推荐的新接口：

    read(instruction, observation, candidate_tools) -> 小抄文本
        每步动手前调：查森林拼小抄，注入 agent 的 prompt。

    write_episode(episode)
        每轮跑完调：按边级成败写整条路径，按轮级成败记 cost-to-go。

正式 ``write_episode`` 路径的 ``n_success`` 与 cost-to-go 都按整轮任务结果
统计；单步调用有没有报错另记 notes，不把协议调用成功当成任务成功。

旧版 ``write(instruction, list[Step])`` 仍保留，用于兼容早期调用。

真实部署形态 = 永远在线 read+write（Setting I）。评测的冻结（Setting II）
通过 write_enabled=False 实现：只读旧经验、不写新经验，保证增益只来自迁移。
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .distill import Distiller, RuleBasedDistiller
from .episode import Episode, EpisodeStep
from .forest import ExperienceForest
from .notes import GlobalNotes
from .retrieval import Cheatsheet, Retriever, render_cheatsheet
from .scoring import DEFAULT_GAMMA, DEFAULT_UCB_C
from .template_lib import TemplateLibrary
from .types import ActionTmpl, StateSig
from .dual_retrieval import (DualCheatsheet, TaskExperience, WorkflowStep, action_stage,
                             retrieve_dual, render_dual_cheatsheet)
from .situation_summary import observation_evidence, previous_action


@dataclass
class Step:
    """旧版兼容用的一步执行记录。

    新代码请使用 ``EpisodeStep``。旧版 Step 保留是为了兼容早期测试和调用方；
    它的 ``success`` 表示该步是否成功，而不是整轮任务成败。
    """

    observation: dict[str, Any]
    raw_action: dict[str, Any]
    success: bool
    note: Optional[str] = None


class EvoTreeMemory:
    """EvoTree 记忆系统的顶层封装。training-free、plug-and-play。"""

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        distiller: Optional[Distiller] = None,
        gamma: float = DEFAULT_GAMMA,
        ucb_c: float = DEFAULT_UCB_C,
        top_k: int = 3,
        write_enabled: bool = True,
        explore: Optional[bool] = None,
        strict_quality: Optional[bool] = None,
        retrieval_mode: str = "legacy",
        prompt_budget_chars: int = 6000,
        situation_backend: Any = None,
        task_relevance_backend: Any = None,
    ) -> None:
        """
        Args:
            root_dir:      持久化根目录（None = 纯内存，测试用）。
            distiller:     三道清洗实现（None = 规则占位版）。
            gamma/ucb_c:   打分超参。
            top_k:         推荐栏最多给几条。
            write_enabled: 是否允许写入（Setting II 冻结评测时设 False）。
            explore:       检索时保留 UCB 探索项 / 未探索边露脸与否。
                           None = 跟随 write_enabled：探索的回报是"下次统计
                           会变"，不写回的冻结评测下它永远不存在，留着只剩
                           扭曲排序的噪声（2026-09-11 实验二实测 frozen 臂
                           小抄里低置信掷硬币边挤在解法前面）。
        """
        self.root_dir = Path(root_dir) if root_dir else None
        # 模板库随记忆一起持久化：树按 template_id 落盘，库若不同步恢复，
        # 同一条指令重启后会登记成新 id，历史经验失联。
        self.distiller: Distiller = distiller or RuleBasedDistiller(
            template_library=self._load_template_library()
        )
        self.forest = ExperienceForest(root_dir=self.root_dir)
        self.notes = self._load_notes()
        if explore is None:
            explore = write_enabled
        if strict_quality is None:
            env_value = os.environ.get("EVOTREE_STRICT_RETRIEVAL")
            if env_value is not None:
                strict_quality = env_value not in {"0", "false", "no"}
            else:
                # Online learning may use a single successful observation as a
                # candidate. Frozen evaluation has no chance to repair a bad
                # recommendation, so it is strict by default.
                strict_quality = not write_enabled
        self.retriever = Retriever(
            gamma=gamma,
            ucb_c=ucb_c,
            top_k=top_k,
            explore=explore,
            strict_quality=strict_quality,
        )
        self.write_enabled = write_enabled
        if retrieval_mode not in {"legacy", "task", "state", "dual", "dual_unfiltered", "dual_summary"}:
            raise ValueError("Unknown retrieval mode: " + retrieval_mode)
        if retrieval_mode == 'dual_summary' and situation_backend is None:
            raise ValueError('dual_summary requires an explicit situation_backend')
        self.situation_backend = situation_backend
        self.task_relevance_backend = task_relevance_backend
        self.retrieval_mode = retrieval_mode
        self.prompt_budget_chars = prompt_budget_chars
        self.task_experiences: dict[str, list[TaskExperience]] = {}
        if self.root_dir is not None:
            path = self.root_dir / "memory" / "task_experiences.json"
            if path.exists():
                self.task_experiences = {
                    key: [TaskExperience.from_dict(row) for row in rows]
                    for key, rows in json.loads(path.read_text(encoding="utf-8")).items()
                }

    # ------------------------------------------------------------------ #
    # read：每步动手前拼小抄
    # ------------------------------------------------------------------ #
    def read(
        self,
        instruction: str,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]] = None,
    ) -> str:
        """查森林拼小抄，返回可注入 prompt 的文本。空则返回空串。"""
        sheet = self.read_cheatsheet(instruction, observation, candidate_tools)
        return self.render(sheet)

    def lookup_task(self, instruction: str):
        """Resolve a query without registering or generalizing stored templates."""
        library = getattr(self.distiller, "template_library", None)
        if isinstance(library, TemplateLibrary):
            return TemplateLibrary.from_dict(library.to_dict()).match(instruction)
        return self.distiller.distill_task(instruction)

    def read_dual_cheatsheet(self, instruction: str, observation: dict[str, Any],
                            candidate_tools: Optional[list[str]] = None, *,
                            task_enabled: bool = True, state_enabled: bool = True,
                            enforce_state_conditions: bool = True,
                            use_situation_summaries: bool = False,
                            relax_action_stage: bool = True,
                            require_summary_compatible: bool = True,
                            require_compatible_all: bool = True,
                            task_relevance_backend: Any = None) -> DualCheatsheet:
        return retrieve_dual(self, instruction, observation, candidate_tools,
                             task_enabled=task_enabled, state_enabled=state_enabled,
                             enforce_state_conditions=enforce_state_conditions,
                             use_situation_summaries=use_situation_summaries,
                             relax_action_stage=relax_action_stage,
                             require_summary_compatible=require_summary_compatible,
                             require_compatible_all=require_compatible_all,
                             task_relevance_backend=(task_relevance_backend if task_relevance_backend is not None
                                                     else self.task_relevance_backend))

    def render(self, sheet: Cheatsheet) -> str:
        if isinstance(sheet, DualCheatsheet):
            return render_dual_cheatsheet(sheet, self.prompt_budget_chars)
        return render_cheatsheet(sheet)

    def read_cheatsheet(
        self,
        instruction: str,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]] = None,
    ) -> Cheatsheet:
        """同 read，但返回结构化的 Cheatsheet（便于分析/测试）。"""
        if self.retrieval_mode != "legacy":
            return self.read_dual_cheatsheet(instruction, observation, candidate_tools,
                                            task_enabled=self.retrieval_mode in {"task", "dual", "dual_unfiltered", "dual_summary"},
                                            state_enabled=self.retrieval_mode in {"state", "dual", "dual_unfiltered", "dual_summary"},
                                            enforce_state_conditions=self.retrieval_mode not in {"dual_unfiltered", "dual_summary"},
                                            use_situation_summaries=self.retrieval_mode == 'dual_summary')
        template = self.distiller.distill_task(instruction)
        current_sig = self.distiller.distill_state(observation)
        tree = self.forest.get_tree(template.template_id)  # 没见过 → None（冷启动）
        return self.retriever.retrieve(
            tree=tree,
            current_sig=current_sig,
            notes=self.notes,
            candidate_tools=candidate_tools,
        )

    # ------------------------------------------------------------------ #
    # write_episode：真实 episode 的路径级写入
    # ------------------------------------------------------------------ #
    def write_episode(self, episode: Episode) -> None:
        """写入一个统一 Episode。这是 OSWorld 适配器和在线 runtime 的正式入口。

        **边的成功率 = 整轮任务成没成**（``episode.success``）。这是方法的原始
        定义：``p(e)`` 要回答"从这个状态选这个动作，最后能不能把任务做成"，
        因为路径选择需要的正是这个量。

        工具有没有报错是**另一个量**，只进 notes / global_notes（注意栏），
        不参与 ``p``。两者不能互换：

        - 工具 API 成功、任务照样失败 —— 2026-09-06 实测：20 任务中 12 个失败
          任务的真实工具报错为 **0 次**。若拿工具报错当成功率，所有边都会变成
          ``p=1.0`` 失去区分度，且失败轨迹里的动作会被当成正确答案推荐出去。
        - 反之工具报一次错、整轮仍可能绕过去做成，那条路依然值得推荐。

        与 idea §8 的偏差（明确记录）：§8 举的 ``duplicate_slide`` 越界失败两次
        应记 ``n=3、p=0.33``，那是按"单步报错"算的。本实现按整轮成败算，同一例
        得到 ``p=1.0``（整轮成功）+ notes 里两条越界记录。**越界这件事没丢，只是
        从推荐栏移到了注意栏** —— 推荐栏管"走哪条路"，注意栏管"这工具有什么坑"。

        代价：单轮评测时一棵树里所有边同分（任务成了全 1、败了全 0），无法在树
        内部区分动作优劣。**必须多轮重复**（idea §12 Setting I 三轮）才能拉开。
        """
        if not self.write_enabled or episode.is_empty:
            return

        template = self.distiller.distill_task(episode.instruction)
        tree = self.forest.get_or_create_tree(template.template_id)
        success_path = []

        # Preserve known-bad trajectories as negative evidence, but never let
        # an early DONE/timeout/API failure become positive experience merely
        # because an outer harness supplied a truthy result.
        terminal_reason = str(episode.metadata.get("terminal_reason", "")).lower()
        quarantined = bool(episode.metadata.get("early_termination")) or terminal_reason in {
            "early_done",
            "timeout",
            "api_error",
            "container_error",
        }
        episode_success = episode.success and not quarantined

        # Store measured order separately from graph aggregation: repeated
        # actions and loops cannot be reconstructed reliably from edge counts.
        if episode_success:
            stage = "start"
            workflow = []
            prefix = []
            for step in episode.steps:
                action = self.distiller.distill_action(step.raw_action)
                summary = None
                if self.retrieval_mode == 'dual_summary':
                    summary = self.situation_backend.summarize(episode.instruction, step.observation, prefix).to_dict()
                workflow.append(WorkflowStep(action, self.distiller.distill_state(step.observation),
                                             str(step.observation.get("state_stage", stage)), step.note,
                                             observation_evidence(step.observation), summary))
                stage = action_stage(action, step.action_success and step.note is None)
                prefix.append(previous_action(action, step.action_success and step.note is None))
            history = TaskExperience(template.template_id, episode.instruction, workflow, episode.source)
            histories = self.task_experiences.setdefault(template.template_id, [])
            # Deduplicate identical evidence, keeping a small deterministic bank.
            if history.to_dict() not in [item.to_dict() for item in histories]:
                histories.append(history)
                histories.sort(key=lambda item: len(item.steps))
                del histories[3:]

        for step in episode.steps:
            action = self.distiller.distill_action(step.raw_action)
            from_sig = self.distiller.distill_state(step.observation)
            to_sig = (
                self.distiller.distill_state(step.next_observation)
                if step.next_observation is not None
                else None
            )
            edge = tree.upsert_step(
                from_sig=from_sig,
                action=action,
                success=episode_success,
                to_sig=to_sig,
                note=step.note,
            )
            if episode_success:
                success_path.append(edge)
            # 报错另记注意栏：不影响 p，但要让 agent 知道这一步有什么坑。
            self._record_note(action, step.note)

        if episode_success:
            tree.record_steps_after_success(success_path)

    # ------------------------------------------------------------------ #
    # write：旧版兼容入口
    # ------------------------------------------------------------------ #
    def write(self, instruction: str, trajectory: list[Step]) -> None:
        """兼容早期 API：按每一步的 ``Step.success`` 写入。

        早期测试和原型代码把每一步的成功状态直接放在 Step 中；该接口继续
        保留这种语义。真实 OSWorld 轨迹不要使用此方法，请使用
        ``write_episode``，因为 OSWorld 的中间 reward=0 并不表示单步失败。
        """
        if not self.write_enabled or not trajectory:
            return

        template = self.distiller.distill_task(instruction)
        tree = self.forest.get_or_create_tree(template.template_id)
        step_edges = []
        previous_sig: Optional[StateSig] = None
        previous_action: Optional[ActionTmpl] = None
        previous_success: Optional[bool] = None
        previous_note: Optional[str] = None

        for step in trajectory:
            current_sig = self.distiller.distill_state(step.observation)
            if previous_action is not None:
                edge = tree.upsert_step(
                    from_sig=previous_sig,
                    action=previous_action,
                    success=bool(previous_success),
                    to_sig=current_sig,
                    note=previous_note,
                )
                if previous_success:
                    step_edges.append(edge)

            previous_sig = current_sig
            previous_action = self.distiller.distill_action(step.raw_action)
            previous_success = step.success
            previous_note = step.note
            self._record_note(previous_action, previous_note)

        if previous_action is not None:
            edge = tree.upsert_step(
                from_sig=previous_sig,
                action=previous_action,
                success=bool(previous_success),
                to_sig=None,
                note=previous_note,
            )
            if previous_success:
                step_edges.append(edge)

        if trajectory[-1].success:
            tree.record_steps_after_success(step_edges)

    def _record_note(self, action: ActionTmpl, note: Optional[str]) -> None:
        """报错的坑位笔记，抄一份进全局手册。

        **GUI 动作也要记**（2026-09-17 修）。原实现只收 ``kind == TOOL``，
        而 OSWorld 的失败绝大多数是 GUI 动作失败（点错位置、命令打进了没有
        焦点的窗口），于是注意栏对主要失败模式完全失聪 —— 实验三里 agent
        连按 11 次 ``Ctrl+Alt+T`` 打不开终端，这种最该沉淀的坑一条都没进库。

        GUI 动作的 key 是动作模板名（``computer_use.key``），检索侧拿顶层
        工具名 ``computer_use`` 查时由 ``query_namespace`` 一并带出，
        见 ``notes`` 模块的 docstring。
        """
        if note:
            self.notes.add_note(action.name, note)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """把森林、坑位手册和模板库落盘。纯内存模式（root_dir=None）时无操作。"""
        if self.root_dir is None or not self.write_enabled:
            return
        self.forest.save_all()
        self.notes.save(self._notes_path())
        path = self.root_dir / "memory" / "task_experiences.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({key: [item.to_dict() for item in rows]
                                   for key, rows in self.task_experiences.items()},
                                  ensure_ascii=False, indent=2), encoding="utf-8")
        library = getattr(self.distiller, "template_library", None)
        if isinstance(library, TemplateLibrary):
            library.save(self._templates_path())

    def _load_notes(self) -> GlobalNotes:
        if self.root_dir is None:
            return GlobalNotes()
        return GlobalNotes.load(self._notes_path())

    def _load_template_library(self) -> Optional[TemplateLibrary]:
        if self.root_dir is None:
            return None
        return TemplateLibrary.load(self._templates_path())

    def _notes_path(self) -> Path:
        return self.root_dir / "global_notes.json"

    def _templates_path(self) -> Path:
        return self.root_dir / "memory" / "templates.json"

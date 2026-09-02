"""顶层封装：EvoTreeMemory —— 对外的唯一入口。

把三道清洗（distill）、经验森林（forest）、全局坑位手册（notes）、
五拍检索（retrieval）串成完整闭环。

推荐的新接口：

    read(instruction, observation, candidate_tools) -> 小抄文本
        每步动手前调：查森林拼小抄，注入 agent 的 prompt。

    write_episode(episode)
        每轮跑完调：按边级成败写整条路径，按轮级成败记 cost-to-go。

两个成败不是一回事（详见 ``episode`` 模块）：边的 ``n_success`` 由
``EpisodeStep.action_success``（这次调用有没有报错）决定，cost-to-go 才由
``Episode.success``（整轮任务成没成）决定。

旧版 ``write(instruction, list[Step])`` 仍保留，用于兼容早期调用。

真实部署形态 = 永远在线 read+write（Setting I）。评测的冻结（Setting II）
通过 write_enabled=False 实现：只读旧经验、不写新经验，保证增益只来自迁移。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .distill import Distiller, RuleBasedDistiller
from .episode import Episode, EpisodeStep
from .forest import ExperienceForest
from .notes import GlobalNotes
from .retrieval import Cheatsheet, Retriever, render_cheatsheet
from .scoring import DEFAULT_GAMMA, DEFAULT_UCB_C
from .types import ActionKind, ActionTmpl, StateSig


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
    ) -> None:
        """
        Args:
            root_dir:      持久化根目录（None = 纯内存，测试用）。
            distiller:     三道清洗实现（None = 规则占位版）。
            gamma/ucb_c:   打分超参。
            top_k:         推荐栏最多给几条。
            write_enabled: 是否允许写入（Setting II 冻结评测时设 False）。
        """
        self.root_dir = Path(root_dir) if root_dir else None
        self.distiller: Distiller = distiller or RuleBasedDistiller()
        self.forest = ExperienceForest(root_dir=self.root_dir)
        self.notes = self._load_notes()
        self.retriever = Retriever(gamma=gamma, ucb_c=ucb_c, top_k=top_k)
        self.write_enabled = write_enabled

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
        return render_cheatsheet(sheet)

    def read_cheatsheet(
        self,
        instruction: str,
        observation: dict[str, Any],
        candidate_tools: Optional[list[str]] = None,
    ) -> Cheatsheet:
        """同 read，但返回结构化的 Cheatsheet（便于分析/测试）。"""
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

        严格区分两个成败（idea §0 写入的第 3 步与第 4 步）：

        - **边级**（``step.action_success``）决定 ``n_success`` 与落点指针。
          一次工具报错就是这条边的一次失败，即使整轮任务最终成功。
        - **轮级**（``episode.success``）只决定要不要记 cost-to-go，
          且只把**真正走通的那些步**计入剩余步数。

        这样 idea §8 的贯穿示例才成立：整轮成功但中途越界两次的
        ``duplicate_slide`` 边记 ``n=3、p=0.33``，失败落点 ``s_err``
        拿到自己的出边统计，而不是被记成一条 100% 可靠的边。
        """
        if not self.write_enabled or episode.is_empty:
            return

        template = self.distiller.distill_task(episode.instruction)
        tree = self.forest.get_or_create_tree(template.template_id)
        success_path = []

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
                success=step.action_success,
                to_sig=to_sig,
                note=step.note,
            )
            # cost-to-go 只统计"走通了的"步：失败的绕行步不该被算进
            # "还差几步到终点"，否则效率项会把踩坑的代价算成路径长度。
            if episode.success and step.action_success:
                success_path.append(edge)
            if step.note and action.kind == ActionKind.TOOL:
                self.notes.add_note(action.name, step.note)

        if episode.success:
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
        """工具报错的坑位笔记，抄一份进全局手册（仅工具动作）。"""
        if note and action.kind == ActionKind.TOOL:
            self.notes.add_note(action.name, note)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self) -> None:
        """把森林和坑位手册落盘。纯内存模式（root_dir=None）时无操作。"""
        if self.root_dir is None or not self.write_enabled:
            return
        self.forest.save_all()
        self.notes.save(self._notes_path())

    def _load_notes(self) -> GlobalNotes:
        if self.root_dir is None:
            return GlobalNotes()
        return GlobalNotes.load(self._notes_path())

    def _notes_path(self) -> Path:
        return self.root_dir / "global_notes.json"

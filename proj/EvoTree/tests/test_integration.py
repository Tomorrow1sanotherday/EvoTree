"""EvoTreeRuntime 与 agent 决策循环 hook 的测试。"""

import json
from pathlib import Path

import pytest

from evotree.integration import EvoTreeRuntime, inject_cheatsheet
from evotree.memory import EvoTreeMemory
from evotree.osworld_adapter import OSWorldEpisodeAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "rename_dir_gui_6steps.jsonl"


def observation(window="Files"):
    return {"app": "os", "window": window, "dialog": None}


def action(name="os.rename_directory"):
    return {"kind": "tool", "name": name, "args": {"old_path": "a", "new_path": "b"}}


class TestPromptInjection:
    def test_empty_cheatsheet_keeps_prompt(self):
        prompt = "Choose the next action."
        assert inject_cheatsheet(prompt, "") == prompt

    def test_cheatsheet_is_advisory_suffix(self):
        prompt = "Choose the next action."
        result = inject_cheatsheet(prompt, "[经验] 推荐：rename")
        assert result.startswith(prompt)
        assert result.endswith("[经验] 推荐：rename")


class TestRuntimeLifecycle:
    def test_begin_and_before_model(self):
        runtime = EvoTreeRuntime(EvoTreeMemory())
        runtime.begin("rename directory")
        result = runtime.before_model(observation(), ["os.rename_directory"])
        assert runtime.active is True
        assert result.text == ""
        assert result.reranked_tools == ["os.rename_directory"]

    def test_before_model_requires_begin(self):
        with pytest.raises(RuntimeError, match="begin"):
            EvoTreeRuntime(EvoTreeMemory()).before_model(observation())

    def test_after_action_requires_begin(self):
        with pytest.raises(RuntimeError, match="begin"):
            EvoTreeRuntime(EvoTreeMemory()).after_action(observation(), action())

    def test_finish_requires_result(self):
        runtime = EvoTreeRuntime(EvoTreeMemory())
        runtime.begin("rename directory")
        with pytest.raises(ValueError, match="success"):
            runtime.finish()

    def test_successful_episode_is_committed_and_runtime_resets(self):
        memory = EvoTreeMemory()
        runtime = EvoTreeRuntime(memory)
        runtime.begin("rename directory")
        runtime.after_action(observation(), action(), next_observation=observation("Files *"))
        episode = runtime.finish(score=1.0)

        assert episode.success is True
        assert episode.num_steps == 1
        assert runtime.active is False
        sheet = memory.read_cheatsheet("rename directory", observation())
        assert sheet.recommendations[0].action.name == "os.rename_directory"

    def test_failed_episode_is_committed_without_cost_to_go(self):
        """整轮失败但动作没报错：边记 n+1 且算成功，只是没有 cost-to-go。"""
        memory = EvoTreeMemory()
        runtime = EvoTreeRuntime(memory)
        runtime.begin("rename directory")
        runtime.after_action(observation(), action())
        runtime.finish(success=False)

        tree = next(iter(memory.forest._trees.values()))
        edge = tree.out_edges("s0")[0]
        assert edge.n == 1
        assert edge.n_success == 1
        assert edge.steps_after_success == []

    def test_note_marks_edge_level_failure(self):
        """在线 hook 只传 note 时，按"有报错即失败"推断边级成败。"""
        memory = EvoTreeMemory()
        runtime = EvoTreeRuntime(memory)
        runtime.begin("rename directory")
        runtime.after_action(
            observation(),
            action(),
            next_observation=observation("Error"),
            note="OSError: target exists",
        )
        runtime.finish(success=True)

        tree = next(iter(memory.forest._trees.values()))
        edge = tree.out_edges("s0")[0]
        assert edge.n == 1
        assert edge.n_success == 0
        assert edge.to_fail is not None

    def test_explicit_action_success_overrides_note_inference(self):
        """显式传 action_success 时以它为准（note 可能只是警告不是失败）。"""
        memory = EvoTreeMemory()
        runtime = EvoTreeRuntime(memory)
        runtime.begin("rename directory")
        runtime.after_action(
            observation(),
            action(),
            next_observation=observation("Files *"),
            note="deprecation warning",
            action_success=True,
        )
        runtime.finish(success=True)

        tree = next(iter(memory.forest._trees.values()))
        edge = tree.out_edges("s0")[0]
        assert edge.n_success == 1
        assert edge.to_success is not None

    def test_score_and_success_conflict_is_rejected(self):
        runtime = EvoTreeRuntime(EvoTreeMemory())
        runtime.begin("rename directory")
        with pytest.raises(ValueError, match="disagree"):
            runtime.finish(success=True, score=0.0)

    def test_begin_discards_unfinished_previous_episode(self):
        memory = EvoTreeMemory()
        runtime = EvoTreeRuntime(memory)
        runtime.begin("task a")
        runtime.after_action(observation(), action())
        runtime.begin("task b")
        episode = runtime.finish(success=True)

        assert episode.num_steps == 0
        assert memory.forest.get_tree("task_a") is None
        assert memory.forest.get_tree("task_b") is None

    def test_freeze_does_not_commit_episode(self):
        memory = EvoTreeMemory(write_enabled=False)
        runtime = EvoTreeRuntime(memory)
        runtime.begin("rename directory")
        runtime.after_action(observation(), action())
        runtime.finish(success=True)
        assert memory.forest.get_tree("rename_directory") is None


class TestAdapterToMemoryBoundary:
    def test_real_trajectory_can_be_loaded_before_observations_are_enriched(self):
        records = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        ]
        episode = OSWorldEpisodeAdapter().from_records(records, score=1.0)

        assert episode.success is True
        assert episode.num_steps == 4
        assert episode.steps[0].observation["app"] == "unknown"
        assert episode.steps[0].raw_action["prediction"].startswith("\n<tool_call>")

    def test_result_directory_score_is_the_only_success_source(self, tmp_path):
        (tmp_path / "traj.jsonl").write_text(
            FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
        )
        (tmp_path / "result.txt").write_text("0.0\n", encoding="utf-8")
        episode = OSWorldEpisodeAdapter().from_result_directory(tmp_path)

        assert episode.success is False
        assert episode.num_steps == 4
        assert all(step.reward == 0 for step in episode.steps)

    def test_real_adapter_episode_can_be_written_to_memory(self):
        records = [
            json.loads(line)
            for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        ]

        def factory(record, index):
            return {"app": "files", "window": f"state-{index}"}

        episode = OSWorldEpisodeAdapter().from_records(
            records,
            score=1.0,
            observation_factory=factory,
        )
        memory = EvoTreeMemory(
            distiller=__import__(
                "evotree.distill_osworld", fromlist=["OSWorldDistiller"]
            ).OSWorldDistiller()
        )
        memory.write_episode(episode)

        template = memory.distiller.distill_task(episode.instruction)
        tree = memory.forest.get_tree(template.template_id)
        assert tree is not None
        assert len(list(tree.all_edges())) == episode.num_steps
        assert all(edge.n_success == 1 for edge in tree.all_edges())


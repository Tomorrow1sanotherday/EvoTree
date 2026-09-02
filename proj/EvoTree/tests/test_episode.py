"""统一 Episode 数据结构与路径级写入测试。"""

import pytest

from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory


def obs(window):
    return {"app": "os", "window": window, "dialog": None}


def action(name):
    return {"kind": "tool", "name": name, "args": {"value": "variable"}}


def episode(success):
    return Episode(
        instruction="rename a directory",
        steps=(
            EpisodeStep(
                step_index=0,
                observation=obs("Files"),
                next_observation=obs("Files *"),
                raw_action=action("select_directory"),
                reward=0,
            ),
            EpisodeStep(
                step_index=1,
                observation=obs("Files *"),
                raw_action=action("rename_directory"),
                reward=0,
                done=True,
            ),
        ),
        success=success,
    )


class TestEpisode:
    def test_empty_instruction_is_rejected(self):
        with pytest.raises(ValueError, match="instruction"):
            Episode(instruction=" ", steps=(), success=False)

    def test_unordered_steps_are_rejected(self):
        steps = (
            EpisodeStep(2, obs("a"), action("a")),
            EpisodeStep(1, obs("b"), action("b")),
        )
        with pytest.raises(ValueError, match="ordered"):
            Episode("task", steps, success=False)

    def test_num_steps_and_empty(self):
        empty = Episode("task", (), success=False)
        assert empty.is_empty is True
        assert empty.num_steps == 0
        assert episode(True).num_steps == 2


class TestWriteEpisode:
    def test_episode_result_is_applied_to_every_action(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = memory.forest.get_tree("rename_a_directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert [edge.n_success for edge in edges] == [1, 1]
        assert all(edge.steps_after_success for edge in edges)

    def test_failed_episode_records_attempts_but_no_success(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=False))
        tree = memory.forest.get_tree("rename_a_directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert all(edge.n == 1 for edge in edges)
        assert all(edge.n_success == 0 for edge in edges)
        assert all(edge.steps_after_success == [] for edge in edges)

    def test_zero_intermediate_rewards_do_not_change_episode_result(self):
        memory = EvoTreeMemory()
        current = episode(success=True)
        memory.write_episode(current)
        tree = memory.forest.get_tree("rename_a_directory")
        assert all(edge.n_success == 1 for edge in tree.all_edges())

    def test_write_episode_respects_freeze(self):
        memory = EvoTreeMemory(write_enabled=False)
        memory.write_episode(episode(success=True))
        assert memory.forest.get_tree("rename_a_directory") is None

"""统一 Episode 数据结构与路径级写入测试。"""

import pytest

from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory


def obs(window):
    return {"app": "os", "window": window, "dialog": None}


def action(name):
    return {"kind": "tool", "name": name, "args": {"value": "variable"}}


def episode(success):
    return Episode(
        instruction="rename a directory",
        steps=(
            EpisodeStep(
                step_index=0,
                observation=obs("Files"),
                next_observation=obs("Files *"),
                raw_action=action("select_directory"),
                reward=0,
            ),
            EpisodeStep(
                step_index=1,
                observation=obs("Files *"),
                raw_action=action("rename_directory"),
                reward=0,
                done=True,
            ),
        ),
        success=success,
    )


class TestEpisode:
    def test_empty_instruction_is_rejected(self):
        with pytest.raises(ValueError, match="instruction"):
            Episode(instruction=" ", steps=(), success=False)

    def test_unordered_steps_are_rejected(self):
        steps = (
            EpisodeStep(2, obs("a"), action("a")),
            EpisodeStep(1, obs("b"), action("b")),
        )
        with pytest.raises(ValueError, match="ordered"):
            Episode("task", steps, success=False)

    def test_num_steps_and_empty(self):
        empty = Episode("task", (), success=False)
        assert empty.is_empty is True
        assert empty.num_steps == 0
        assert episode(True).num_steps == 2


class TestWriteEpisode:
    def test_episode_result_is_applied_to_every_action(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = memory.forest.get_tree("rename_a_directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert [edge.n_success for edge in edges] == [1, 1]
        assert all(edge.steps_after_success for edge in edges)

    def test_failed_episode_records_attempts_but_no_success(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=False))
        tree = memory.forest.get_tree("rename_a_directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert all(edge.n == 1 for edge in edges)
        assert all(edge.n_success == 0 for edge in edges)
        assert all(edge.steps_after_success == [] for edge in edges)

    def test_zero_intermediate_rewards_do_not_change_episode_result(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = memory.forest.get_tree("rename_a_directory")
        assert all(edge.n_success == 1 for edge in tree.all_edges())

    def test_write_episode_respects_freeze(self):
        memory = EvoTreeMemory(write_enabled=False)
        memory.write_episode(episode(success=True))
        assert memory.forest.get_tree("rename_a_directory") is None

    def test_next_observation_creates_success_pointer(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = memory.forest.get_tree("rename_a_directory")
        first = tree.out_edges("s0")[0]
        assert first.to_success is not None
        assert tree.node_by_id(first.to_success).sig.window == "Files *"

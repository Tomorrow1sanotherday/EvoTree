"""统一 Episode 数据结构与路径级写入测试。"""

import pytest

from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory


def obs(window):
    return {"app": "os", "window": window, "dialog": None}


def action(name):
    return {"kind": "tool", "name": name, "args": {"value": "variable"}}


def tree_for(memory, instruction):
    """按指令取树。

    通过 memory 自己的 distiller 解析 template_id，而不是硬编码字符串——
    id 的构造方式（现在是 可读前缀+骨架哈希）属于实现细节，测试不该锁死它。
    """
    return memory.forest.get_tree(memory.distiller.distill_task(instruction).template_id)


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

    def test_action_success_defaults_to_true(self):
        step = EpisodeStep(0, obs("Files"), action("a"))
        assert step.action_success is True
        assert step.failed is False


class TestWriteEpisode:
    """边级成败 vs 轮级成败（idea §0 写入第 3 步 vs 第 4 步）。"""

    def test_successful_episode_records_success_and_cost_to_go(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = tree_for(memory, "rename a directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert [edge.n_success for edge in edges] == [1, 1]
        assert all(edge.steps_after_success for edge in edges)

    def test_failed_episode_records_attempts_without_cost_to_go(self):
        """整轮失败：动作本身没报错的边仍算成功，但不记 cost-to-go。

        边级成败看的是"这次调用有没有报错"，不是"整轮任务成没成"。这两步
        都没报错，所以 n_success 照记；整轮失败只意味着没走到终点，
        steps_after_success 无从谈起（idea §0 写入第 4 步的前提不成立）。
        """
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=False))
        tree = tree_for(memory, "rename a directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert all(edge.n == 1 for edge in edges)
        assert all(edge.n_success == 1 for edge in edges)
        assert all(edge.steps_after_success == [] for edge in edges)

    def test_failed_action_is_recorded_as_edge_failure(self):
        """动作报错 → 这条边记失败，失败落点挂 to_fail 且有自己的出边。

        对应 idea §8 Episode 1：整轮**成功**的轨迹里，中途越界失败的那条边
        应记 p<1，失败落点 s_err 是带出边的正常状态，而不是被整轮成功抹平。
        """
        memory = EvoTreeMemory()
        memory.write_episode(
            Episode(
                instruction="rename a directory",
                steps=(
                    EpisodeStep(
                        step_index=0,
                        observation=obs("Home"),
                        raw_action=action("rename_directory"),
                        next_observation=obs("Error dialog"),
                        note="OSError: target exists",
                        action_success=False,
                    ),
                    EpisodeStep(
                        step_index=1,
                        observation=obs("Error dialog"),
                        raw_action=action("rename_directory"),
                        next_observation=obs("Files *"),
                    ),
                ),
                success=True,
            )
        )
        tree = tree_for(memory, "rename a directory")
        failed = tree.out_edges("s0")[0]
        assert failed.n == 1 and failed.n_success == 0
        assert failed.to_fail is not None
        assert tree.node_by_id(failed.to_fail).sig.window == "Error dialog"
        # 踩坑的那步不该被算进"还差几步到终点"。
        assert failed.steps_after_success == []
        recovered = tree.out_edges(failed.to_fail)[0]
        assert recovered.n_success == 1
        assert recovered.steps_after_success == [1]

    def test_mixed_episode_matches_idea_walkthrough(self):
        """idea §8 的数字：同一条边越界失败 2 次 + 成功 1 次 → n=3, p=1/3。"""
        memory = EvoTreeMemory()
        steps = []
        for index in range(2):
            steps.append(
                EpisodeStep(
                    step_index=index,
                    observation=obs("Deck"),
                    raw_action=action("duplicate_slide"),
                    next_observation=obs("Deck"),
                    note="pos out of range",
                    action_success=False,
                )
            )
        steps.append(
            EpisodeStep(
                step_index=2,
                observation=obs("Deck"),
                raw_action=action("duplicate_slide"),
                next_observation=obs("Deck copied"),
            )
        )
        memory.write_episode(
            Episode(
                instruction="duplicate the last slide",
                steps=tuple(steps),
                success=True,
            )
        )
        tree = tree_for(memory, "duplicate the last slide")
        edge = tree.out_edges("s0")[0]
        assert edge.n == 3
        assert edge.n_success == 1
        assert edge.notes == ["pos out of range", "pos out of range"]
        # 报错文本进了全局坑位手册（跨任务共享的工具级知识）。
        assert memory.notes.query("duplicate_slide")

    def test_zero_intermediate_rewards_do_not_change_episode_result(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = tree_for(memory, "rename a directory")
        assert all(edge.n_success == 1 for edge in tree.all_edges())

    def test_write_episode_respects_freeze(self):
        memory = EvoTreeMemory(write_enabled=False)
        memory.write_episode(episode(success=True))
        assert tree_for(memory, "rename a directory") is None

    def test_next_observation_creates_success_pointer(self):
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=True))
        tree = tree_for(memory, "rename a directory")
        first = tree.out_edges("s0")[0]
        assert first.to_success is not None
        assert tree.node_by_id(first.to_success).sig.window == "Files *"

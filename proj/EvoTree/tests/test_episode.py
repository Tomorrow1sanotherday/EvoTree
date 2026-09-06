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

    def test_failed_episode_marks_every_edge_as_failure(self):
        """整轮失败 → 这轮走过的每条边都记一次失败，且不记 cost-to-go。

        边的成功率 = 整轮任务成没成。这两步工具都没报错，但任务失败了，
        说明这条路径没走通，沿途每步都不该被当成正确答案推荐给下一轮。
        """
        memory = EvoTreeMemory()
        memory.write_episode(episode(success=False))
        tree = tree_for(memory, "rename a directory")
        edges = list(tree.all_edges())
        assert len(edges) == 2
        assert all(edge.n == 1 for edge in edges)
        assert all(edge.n_success == 0 for edge in edges)
        assert all(edge.steps_after_success == [] for edge in edges)

    def test_tool_error_goes_to_cautions_not_success_rate(self):
        """工具报错只进注意栏，不压低成功率。

        整轮成功说明这条路走通了，沿途每条边 p=1.0；中途那次工具报错作为
        坑位笔记进边的 notes 和 global_notes。

        这与 idea §8 的 p=0.33 不同（那是按单步报错算的），偏差已在
        ``memory.write_episode`` 的说明里记录：越界信息没丢，只是从推荐栏
        移到了注意栏。理由是实测 12 个失败任务的真实工具报错为 0 次，
        用报错当成功率会让 p 退化成常量。
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
        first = tree.out_edges("s0")[0]
        # 整轮成功 → 这条边算成功，尽管这一步工具报了错。
        assert first.n == 1 and first.n_success == 1
        assert first.to_success is not None
        assert tree.node_by_id(first.to_success).sig.window == "Error dialog"
        # 报错文本仍进边的 notes 与全局注意栏。
        assert first.notes == ["OSError: target exists"]
        assert memory.notes.query("rename_directory")

    def test_repeated_action_accumulates_under_episode_result(self):
        """同一条边走 3 次、整轮成功 → n=3、p=1.0，两次报错各留一条注意。

        对照 idea §8：文档按单步报错算得 p=1/3，本实现按整轮成败算得 1.0。
        两次越界没有丢失，作为 notes 留在边上并抄进 global_notes。
        """
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
        assert edge.n_success == 3  # 整轮成功 → 三次都算成功
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

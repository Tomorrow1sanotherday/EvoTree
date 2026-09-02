"""端到端测试：EvoTreeMemory 的 read/write 闭环。

这是最重要的测试 —— 证明整个架构自洽：写入假轨迹 → 森林积累统计 →
读出小抄。同时覆盖冷启动、冻结、贯穿示例（idea §8）。
"""

from evotree.memory import EvoTreeMemory, Step


def obs(app="os", window="Files", dialog=None, **facts):
    return {"app": app, "window": window, "dialog": dialog, "facts": facts}


def tool_action(name, **args):
    return {"kind": "tool", "name": name, "args": args}


# --------------------------- 冷启动 --------------------------------------- #
class TestColdStart:
    def test_unseen_task_empty_recommendation(self):
        mem = EvoTreeMemory()
        # 全新任务，森林空 → 推荐栏空，小抄为空串
        sheet = mem.read_cheatsheet("do something never seen", obs())
        assert sheet.is_cold_start
        assert mem.read("do something never seen", obs()) == ""

    def test_cold_start_still_gets_cautions(self):
        # 全新任务，但注意栏能命中别处学到的坑
        mem = EvoTreeMemory()
        mem.notes.add_note("save", "Ctrl+S 在 Impress 弹格式确认框")
        sheet = mem.read_cheatsheet(
            "brand new task", obs(app="impress"), candidate_tools=["save"]
        )
        assert sheet.is_cold_start                    # 推荐栏空
        assert any("save" in c for c in sheet.cautions)  # 注意栏命中


# --------------------------- 写入→读出闭环 -------------------------------- #
class TestWriteThenRead:
    def test_learn_from_one_success(self):
        mem = EvoTreeMemory()
        instr = 'rename directory "todo_1" to "todo_2"'
        # 一轮成功轨迹：在 Files 里调 rename_directory 成功
        traj = [
            Step(obs(app="os", window="Files"),
                 tool_action("os.rename_directory", old="todo_1", new="todo_2"),
                 success=True),
        ]
        mem.write(instr, traj)

        # 同类任务（换参数）再来，应该能读到推荐
        sheet = mem.read_cheatsheet(
            'rename directory "a" to "b"', obs(app="os", window="Files")
        )
        assert not sheet.is_cold_start
        rec = sheet.recommendations[0]
        assert rec.action.name == "os.rename_directory"
        assert rec.success_rate == 1.0
        assert rec.n == 1

    def test_stats_accumulate_across_episodes(self):
        mem = EvoTreeMemory()
        instr = 'copy the "Revenue" column'
        # 跑三轮同类任务：成功、成功、失败
        for success in (True, True, False):
            mem.write(instr, [
                Step(obs(app="calc", window="s.xlsx"),
                     tool_action("copy_cells", src="A", dst="B"),
                     success=success),
            ])
        sheet = mem.read_cheatsheet('copy the "Cost" column',
                                    obs(app="calc", window="s.xlsx"))
        rec = sheet.recommendations[0]
        assert rec.n == 3
        assert rec.success_rate == 2 / 3


# --------------------------- 贯穿示例（idea §8）--------------------------- #
class TestWalkthrough:
    def test_lookahead_gives_next_step(self):
        """Episode 2：小抄给出推荐 + 前瞻'完成后 save 一步收尾'。

        注意：两步的状态主键必须真正不同（duplicate 后进入'已复制'局面），
        否则两个动作会成为同一节点的兄弟出边而非父子。这体现了 idea 地基①
        —— 状态签名要能区分动作前后的局面（这里用 window 标题的变化表达）。
        """
        mem = EvoTreeMemory()
        instr = "duplicate last two slides"
        # 一轮两步成功轨迹：在初始态 duplicate_slide → 进入'已复制'态 → save
        traj = [
            Step(obs(app="impress", window="a.pptx"),
                 tool_action("duplicate_slide", idx=1, pos=2), success=True),
            Step(obs(app="impress", window="a.pptx *"),  # 标题带 * = 已修改未保存
                 tool_action("save"), success=True),
        ]
        mem.write(instr, traj)

        sheet = mem.read_cheatsheet("duplicate last two slides",
                                    obs(app="impress", window="a.pptx"))
        rec = sheet.recommendations[0]
        assert rec.action.name == "duplicate_slide"
        # 前瞻：顺着 to_success 指针看到下一步是 save
        assert rec.lookahead is not None
        assert rec.lookahead.name == "save"

    def test_pit_note_recorded_and_surfaced(self):
        """Episode 1→3：坑位笔记入账，之后作为注意栏浮现。"""
        mem = EvoTreeMemory()
        instr = "duplicate slides"
        mem.write(instr, [
            Step(obs(app="impress", window="a.pptx"),
                 tool_action("duplicate_slide", idx=1, pos=99),
                 success=False, note="pos 越界"),
        ])
        # 坑位进了全局手册
        assert "duplicate_slide" in mem.notes.tools_covered()
        # 别的任务里，注意栏能查到这个坑
        sheet = mem.read_cheatsheet("other task", obs(app="impress"),
                                    candidate_tools=["duplicate_slide"])
        assert any("越界" in c for c in sheet.cautions)


# --------------------------- 冻结（Setting II）--------------------------- #
class TestFreeze:
    def test_write_disabled_does_not_learn(self):
        mem = EvoTreeMemory(write_enabled=False)
        mem.write("some task", [
            Step(obs(), tool_action("t"), success=True),
        ])
        # 冻结下写入被忽略，森林仍空
        assert mem.read_cheatsheet("some task", obs()).is_cold_start


# --------------------------- 持久化闭环 ----------------------------------- #
class TestPersistence:
    def test_save_and_reload(self, tmp_path):
        mem = EvoTreeMemory(root_dir=tmp_path)
        instr = 'rename "x" to "y"'
        mem.write(instr, [
            Step(obs(app="os", window="Files"),
                 tool_action("os.rename_directory", old="x", new="y"),
                 success=True, note="小心路径"),
        ])
        mem.save()

        # 新实例从磁盘载入，经验还在
        mem2 = EvoTreeMemory(root_dir=tmp_path)
        sheet = mem2.read_cheatsheet('rename "a" to "b"',
                                     obs(app="os", window="Files"))
        assert not sheet.is_cold_start
        assert sheet.recommendations[0].action.name == "os.rename_directory"

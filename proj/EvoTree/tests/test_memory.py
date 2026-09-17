"""端到端测试：EvoTreeMemory 的 read/write 闭环。

这是最重要的测试 —— 证明整个架构自洽：写入假轨迹 → 森林积累统计 →
读出小抄。同时覆盖冷启动、冻结、贯穿示例（idea §8）。
"""

from evotree.distill_osworld import OSWorldDistiller
from evotree.episode import Episode, EpisodeStep
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


class TestGuiPitNotesReachTheAgent:
    """GUI 动作的坑必须能沉淀、并被真实工具名查出来（实验三 2026-09-17 事故）。

    事故复盘：注入链路全绿，但整轮 [注意] 栏零注入，agent 连按 11 次
    ``Ctrl+Alt+T`` 打不开终端 —— 而种子里明明写着这条警告。两个原因叠加：

    1. ``_record_note`` 只收 ``kind == TOOL``，GUI 动作的坑被静默丢弃；
    2. 检索侧手里是顶层工具名 ``computer_use``，笔记却挂在动作模板名
       ``computer_use.key`` 上，精确查表永远查不到、且不报错。

    OSWorld 的失败绝大多数是 GUI 动作失败，所以这两条合起来等于注意栏对
    主要失败模式完全失聪。这组测试钉住修复后的行为。
    """

    @staticmethod
    def _gui_episode(instruction, note):
        """一轮 GUI 动作失败的 episode（按 OSWorld 的真实动作形状）。"""
        return Episode(
            instruction=instruction,
            steps=(
                EpisodeStep(
                    step_index=0,
                    observation=obs(app="unknown", window=""),
                    raw_action={
                        "action": {
                            "action_type": "computer_use.key",
                            "parameters": {"keys": ["CTRL", "ALT", "T"]},
                        }
                    },
                    note=note,
                    action_success=False,
                ),
            ),
            success=False,
        )

    def _memory(self):
        return EvoTreeMemory(distiller=OSWorldDistiller())

    def test_gui_note_is_recorded_at_all(self):
        # 原实现在这里就丢了：kind == GUI 不入库
        mem = self._memory()
        mem.write_episode(self._gui_episode("open a terminal", "快捷键没反应"))
        assert "computer_use.key" in mem.notes.tools_covered()

    def test_real_tool_name_surfaces_the_gui_note(self):
        """终点检验：拿运行时真实的候选工具名，必须查得到这条坑。

        ``candidate_tools`` 只有 ``computer_use`` —— 这就是 OSWorld MCP 工具表
        里唯一真实存在的名字（``key``/``type`` 是它的 action 参数取值）。
        """
        mem = self._memory()
        mem.write_episode(
            self._gui_episode("open a terminal", "Ctrl+Alt+T 常打不开终端")
        )
        sheet = mem.read_cheatsheet(
            "some other task",
            obs(app="unknown", window=""),
            candidate_tools=["computer_use"],
        )
        assert any("打不开终端" in c for c in sheet.cautions)

    def test_caution_actually_renders_into_the_prompt(self):
        """真正的防线：小抄文本非空。

        事故期间前面每一环都"正常"，只有这一步的产物是空串 —— 而没人断言它。
        """
        mem = self._memory()
        mem.write_episode(
            self._gui_episode("open a terminal", "Ctrl+Alt+T 常打不开终端")
        )
        text = mem.read(
            "some other task",
            obs(app="unknown", window=""),
            candidate_tools=["computer_use"],
        )
        assert "[注意]" in text
        assert "打不开终端" in text

    def test_seeded_notes_reach_the_agent(self):
        """种子手册里按子动作名写的笔记，也要能被真实工具名查出来。

        实验三的种子就是这么写的（``computer_use.key`` / ``computer_use.type``），
        当时一条都没送达。
        """
        mem = self._memory()
        mem.notes.add_note("computer_use.key", "Ctrl+Alt+T 常打不开终端")
        mem.notes.add_note("computer_use.type", "敲命令前先确认终端持有焦点")
        text = mem.read(
            "brand new task",
            obs(app="unknown", window=""),
            candidate_tools=["computer_use"],
        )
        assert "打不开终端" in text
        assert "确认终端持有焦点" in text

    def test_tool_notes_still_work(self):
        # 回归：原本就支持的 TOOL 动作不能因为放宽 kind 判据而回退
        mem = self._memory()
        mem.write_episode(
            Episode(
                instruction="copy files",
                steps=(
                    EpisodeStep(
                        step_index=0,
                        observation=obs(app="os", window="Files"),
                        raw_action={
                            "action": {
                                "action_type": "os.open_shell",
                                "parameters": {},
                            }
                        },
                        note="有时起不来",
                        action_success=False,
                    ),
                ),
                success=False,
            )
        )
        sheet = mem.read_cheatsheet(
            "another task",
            obs(app="os", window="Files"),
            candidate_tools=["os.open_shell"],
        )
        assert any("起不来" in c for c in sheet.cautions)


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

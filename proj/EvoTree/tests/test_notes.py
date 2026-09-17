"""notes 模块的单元测试（全局坑位手册 / 注意栏）。"""

from evotree.notes import GlobalNotes


class TestAddQuery:
    def test_add_and_query(self):
        notes = GlobalNotes()
        notes.add_note("duplicate_slide", "pos 不得超过总页数")
        assert notes.query("duplicate_slide") == ["pos 不得超过总页数"]

    def test_dedup(self):
        notes = GlobalNotes()
        notes.add_note("save", "会弹格式确认框")
        notes.add_note("save", "会弹格式确认框")  # 重复
        assert len(notes.query("save")) == 1

    def test_multiple_notes_per_tool(self):
        notes = GlobalNotes()
        notes.add_note("t", "坑一")
        notes.add_note("t", "坑二")
        assert notes.query("t") == ["坑一", "坑二"]

    def test_empty_note_ignored(self):
        notes = GlobalNotes()
        notes.add_note("t", "   ")  # 空白
        assert notes.query("t") == []

    def test_unknown_tool_returns_empty(self):
        assert GlobalNotes().query("never_seen") == []


class TestQueryNamespace:
    """工具名查询要带出子动作键（实验三 2026-09-17 的教训）。

    环境里真实存在的工具只有 ``computer_use``；``key``/``type`` 是它的
    ``action`` 参数取值。但写入侧记的是动作模板名 ``computer_use.key``。
    检索侧手里只有顶层工具名，用精确查询会整栏为空且不报错 —— 实验三整轮
    [注意] 栏零注入，agent 连按 11 次 Ctrl+Alt+T 而警告从未送达。
    """

    def test_tool_name_surfaces_sub_action_notes(self):
        notes = GlobalNotes()
        notes.add_note("computer_use.key", "Ctrl+Alt+T 常打不开终端")
        notes.add_note("computer_use.type", "敲命令前先确认焦点")
        # 检索侧拿到的就是这个顶层名字
        assert notes.query_namespace("computer_use") == [
            "Ctrl+Alt+T 常打不开终端",
            "敲命令前先确认焦点",
        ]

    def test_exact_query_still_misses_by_design(self):
        # 对照：精确查询查不到子动作键，正是这个差异导致了实验三的事故
        notes = GlobalNotes()
        notes.add_note("computer_use.key", "Ctrl+Alt+T 常打不开终端")
        assert notes.query("computer_use") == []

    def test_own_key_comes_before_sub_actions(self):
        notes = GlobalNotes()
        notes.add_note("computer_use.type", "子动作的坑")
        notes.add_note("computer_use", "工具本身的坑")
        assert notes.query_namespace("computer_use") == [
            "工具本身的坑",
            "子动作的坑",
        ]

    def test_same_note_on_two_sub_actions_said_once(self):
        # 同一条坑挂在多个子动作上，渲染时只说一次（小抄要短）
        notes = GlobalNotes()
        notes.add_note("computer_use.key", "这个环境快捷键不可靠")
        notes.add_note("computer_use.click", "这个环境快捷键不可靠")
        assert notes.query_namespace("computer_use") == ["这个环境快捷键不可靠"]

    def test_prefix_must_be_a_dot_boundary(self):
        # computer_use 不该带出 computer_useful 的笔记（前缀不是命名空间）
        notes = GlobalNotes()
        notes.add_note("computer_useful", "别的工具的坑")
        assert notes.query_namespace("computer_use") == []

    def test_tool_without_sub_actions_behaves_like_query(self):
        notes = GlobalNotes()
        notes.add_note("os.open_shell", "有时起不来")
        assert notes.query_namespace("os.open_shell") == ["有时起不来"]

    def test_empty_name_returns_empty(self):
        notes = GlobalNotes()
        notes.add_note("computer_use.key", "坑")
        assert notes.query_namespace("") == []


class TestPersistence:
    def test_roundtrip(self):
        notes = GlobalNotes()
        notes.add_note("a", "坑A")
        notes.add_note("b", "坑B")
        restored = GlobalNotes.from_dict(notes.to_dict())
        assert restored.query("a") == ["坑A"]
        assert set(restored.tools_covered()) == {"a", "b"}

    def test_save_load_file(self, tmp_path):
        notes = GlobalNotes()
        notes.add_note("save", "小心弹窗")
        p = tmp_path / "global_notes.json"
        notes.save(p)
        loaded = GlobalNotes.load(p)
        assert loaded.query("save") == ["小心弹窗"]

    def test_load_missing_file_is_empty(self, tmp_path):
        # 文件不存在时返回空手册（冷启动，不报错）
        loaded = GlobalNotes.load(tmp_path / "nonexistent.json")
        assert loaded.tools_covered() == []

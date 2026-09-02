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

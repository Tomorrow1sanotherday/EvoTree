"""τ 任务模板库测试。

idea §4.1 特别标注：**τ 匹配的准确率/召回直接决定分册是否有效**，是 pilot
必须专门验证的一环。这里覆盖两类错误，它们分别打穿两条地基：

    假合并（不同任务归到一棵树）→ 违反推论 2「混记成 50% 对谁都没用」
    假分裂（同类任务分成多棵树）→ 打穿地基②（每条边攒不到样本）
"""

from pathlib import Path

from evotree.template_lib import TemplateLibrary, extract_skeleton


class TestSkeleton:
    def test_extracts_quoted_filename_number_path(self):
        skeleton, values = extract_skeleton(
            'rename "todo_1" in /home/user/docs to report.txt after 3 seconds'
        )
        assert values == ['"todo_1"', "/home/user/docs", "report.txt", "3"]
        assert skeleton.count("<arg") == 4

    def test_same_skeleton_for_different_parameter_values(self):
        a, _ = extract_skeleton("open a.xlsx and go to row 3")
        b, _ = extract_skeleton("open b.xlsx and go to row 7")
        assert a == b


class TestNoFalseMerge:
    """不同任务绝不能归到同一棵树。"""

    def test_long_instructions_sharing_a_prefix_stay_separate(self):
        """回归：旧实现把骨架截断到 40 字符当 id，这两条会撞成同一棵树。"""
        library = TemplateLibrary()
        budget = library.match(
            "Create a new spreadsheet document and name it budget"
        )
        revenue = library.match(
            "Create a new spreadsheet document and name it revenue"
        )
        # 注意：这两条只差最后一个词，会被模糊归并成同一模板——这是**正确**的
        # （同一件事换个名字）。此处要断言的是 id 由完整骨架决定，而不是被
        # 截断成同一个字符串。
        assert budget.template_id == revenue.template_id
        other = library.match(
            "Create a new spreadsheet document and name it budget then close it"
        )
        assert other.template_id != budget.template_id

    def test_different_verbs_are_different_templates(self):
        library = TemplateLibrary()
        copy = library.match("Copy the report file to the archive folder")
        delete = library.match("Delete the report file from the archive folder")
        # 差两个词（Copy/Delete、to/from），超过阈值 → 不归并
        assert copy.template_id != delete.template_id

    def test_short_instructions_are_not_fuzzy_merged(self):
        """短指令改一个词往往是另一件事，不做模糊归并。"""
        library = TemplateLibrary()
        assert library.match("undo").template_id != library.match("redo").template_id
        assert library.match("save file").template_id != library.match("close file").template_id


class TestNoFalseSplit:
    """同类任务必须归到同一棵树，经验才能叠加攒厚。"""

    def test_idea_walkthrough_column_example(self):
        """idea §4.1 的原例：列名挖不成槽位，旧实现把它们分成两棵树。"""
        library = TemplateLibrary()
        revenue = library.match("Copy the Revenue column of A.xlsx to a new sheet")
        cost = library.match("Copy the Cost column of B.xlsx to a new sheet")
        assert revenue.template_id == cost.template_id

    def test_filename_and_number_changes_share_a_template(self):
        library = TemplateLibrary()
        first = library.match("open report_2024.xlsx and delete row 3")
        second = library.match("open summary.xlsx and delete row 17")
        assert first.template_id == second.template_id
        assert len(library) == 1

    def test_generalized_skeleton_absorbs_a_third_variant(self):
        """归并一次后，骨架放宽，后续同类指令直接命中。"""
        library = TemplateLibrary()
        first = library.match("Copy the Revenue column of A.xlsx to a new sheet")
        library.match("Copy the Cost column of B.xlsx to a new sheet")
        third = library.match("Copy the Margin column of C.xlsx to a new sheet")
        assert third.template_id == first.template_id


class TestPersistence:
    def test_round_trip_preserves_template_ids(self, tmp_path: Path):
        """模板库必须能落盘：树按 template_id 存，库丢了历史经验就失联。"""
        library = TemplateLibrary()
        first = library.match("Copy the Revenue column of A.xlsx to a new sheet")
        library.match("Copy the Cost column of B.xlsx to a new sheet")

        path = tmp_path / "templates.json"
        library.save(path)
        restored = TemplateLibrary.load(path)

        # 重启后单独出现的变体仍归到原模板，而不是新开一棵树。
        again = restored.match("Copy the Cost column of B.xlsx to a new sheet")
        assert again.template_id == first.template_id

    def test_load_missing_file_returns_empty_library(self, tmp_path: Path):
        library = TemplateLibrary.load(tmp_path / "absent.json")
        assert len(library) == 0


class TestMemoryIntegration:
    def test_memory_persists_and_reloads_template_library(self, tmp_path: Path):
        from evotree.episode import Episode, EpisodeStep
        from evotree.memory import EvoTreeMemory

        def obs(window):
            return {"app": "calc", "window": window, "dialog": None}

        memory = EvoTreeMemory(root_dir=tmp_path)
        memory.write_episode(
            Episode(
                instruction="Copy the Revenue column of A.xlsx to a new sheet",
                steps=(
                    EpisodeStep(
                        0,
                        obs("A.xlsx — Calc"),
                        {"kind": "tool", "name": "copy_cells", "args": {"range": "A1"}},
                        next_observation=obs("A.xlsx — Calc *"),
                    ),
                ),
                success=True,
            )
        )
        memory.save()
        assert (tmp_path / "memory" / "templates.json").exists()

        # 新进程：同类任务换参数，应命中已攒经验的那棵树。
        reloaded = EvoTreeMemory(root_dir=tmp_path)
        sheet = reloaded.read_cheatsheet(
            "Copy the Cost column of B.xlsx to a new sheet", obs("B.xlsx — Calc")
        )
        assert sheet.recommendations, "同类任务应命中已有经验，而不是冷启动"
        assert sheet.recommendations[0].action.name == "copy_cells"

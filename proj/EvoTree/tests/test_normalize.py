"""normalize 模块测试：window 归一化。

核心验收：解决 spec/02 暴露的"同类任务换文件名 → 状态主键对不上"问题。
"""

from evotree.distill import RuleBasedDistiller
from evotree.normalize import normalize_window


class TestNormalizeWindow:
    def test_filename_becomes_placeholder(self):
        assert normalize_window("data.xlsx — Calc") == "<file> — Calc"

    def test_different_files_same_normalized(self):
        # 核心：不同文件名归一化后相同 → 同类任务状态能对上
        a = normalize_window("data.xlsx — Calc")
        b = normalize_window("data2.xlsx — Calc")
        c = normalize_window("report_2024.xlsx — Calc")
        assert a == b == c

    def test_modified_mark_preserved(self):
        # 未保存标记 * 是任务相关状态，必须保留
        assert "*" in normalize_window("report.pptx — Impress *")

    def test_modified_vs_saved_are_different_states(self):
        # 有 * 和无 * 应归一化成不同状态（反映"是否有未保存改动"）
        saved = normalize_window("a.pptx — Impress")
        modified = normalize_window("a.pptx — Impress *")
        assert saved != modified

    def test_no_filename_unchanged(self):
        assert normalize_window("Files") == "Files"

    def test_empty(self):
        assert normalize_window("") == ""

    def test_multiple_filenames(self):
        # 多个文件名都归一化
        result = normalize_window("a.txt vs b.txt — Editor")
        assert "a.txt" not in result and "b.txt" not in result


class TestFilenameOverreach:
    """回归：形如文件名但不是文件名的东西不该被归一化掉。

    误伤比漏归一化更有害——它会把不同应用/网站的窗口错误归堆成同一个状态，
    直接制造签名混叠（地基①）。
    """

    def test_version_number_is_not_a_filename(self):
        assert normalize_window("python3.11 — Terminal") == "python3.11 — Terminal"

    def test_domain_is_not_a_filename(self):
        result = normalize_window("github.com/foo — Firefox")
        assert "github.com" in result

    def test_different_sites_stay_distinct(self):
        assert normalize_window("github.com — Firefox") != normalize_window(
            "example.org — Firefox"
        )

    def test_real_filenames_still_normalized(self):
        assert normalize_window("notes.md — Editor") == "<file> — Editor"

    def test_libreoffice_filename_with_spaces_is_normalized_as_one_file(self):
        assert normalize_window(
            "Ch4 Video Effect.pptx - LibreOffice Impress"
        ) == "<file> - LibreOffice Impress"
        assert normalize_window(
            "Another Presentation 2024.pptx - LibreOffice Impress *"
        ) == "<file> - LibreOffice Impress *"

    def test_libreoffice_documents_with_different_spaced_names_share_state_key(self):
        assert normalize_window("Ch4 Video Effect.pptx - LibreOffice Impress") == normalize_window(
            "lec17-gui-events.pptx - LibreOffice Impress"
        )


class TestIntegrationWithDistill:
    """归一化接进 distill 后，同类任务的状态主键真的对上了。"""

    def test_same_task_diff_file_same_state_key(self):
        D = RuleBasedDistiller()
        s1 = D.distill_state({"app": "calc", "window": "data.xlsx — Calc"})
        s2 = D.distill_state({"app": "calc", "window": "data2.xlsx — Calc"})
        # 这正是 spec/02 里失败的场景，现在应该对上
        assert s1.key() == s2.key()

    def test_modified_state_still_distinct(self):
        D = RuleBasedDistiller()
        saved = D.distill_state({"app": "impress", "window": "a.pptx — Impress"})
        modified = D.distill_state({"app": "impress", "window": "a.pptx — Impress *"})
        # 未保存状态仍是不同局面（前瞻 duplicate→save 依赖这个区分）
        assert saved.key() != modified.key()

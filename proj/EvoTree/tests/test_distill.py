"""distill 模块的单元测试（占位版三道清洗）。

验证接口正确、归堆行为对：同类换参数的指令/动作应归到同一模板/边。
真实数据的解析细节留给 B 阶段，这里测占位版的核心契约。
"""

from evotree.distill import RuleBasedDistiller
from evotree.types import ActionKind


def setup_module(_):
    global D
    D = RuleBasedDistiller()


# --------------------------- τ 任务模板 ----------------------------------- #
class TestDistillTask:
    def test_same_skeleton_same_template(self):
        # 换参数(文件名/列名)应归到同一模板 —— 经验才能叠加攒厚
        t1 = D.distill_task('Copy the "Revenue" column from a.xlsx')
        t2 = D.distill_task('Copy the "Cost" column from b.xlsx')
        assert t1.template_id == t2.template_id

    def test_different_task_different_template(self):
        t1 = D.distill_task('Copy the "Revenue" column from a.xlsx')
        t2 = D.distill_task("Install the autoDocstring extension in VS Code")
        assert t1.template_id != t2.template_id

    def test_slots_extracted(self):
        t = D.distill_task('rename "todo_1" to "todo_2"')
        # 两个引号内容被挖成两个槽位
        assert len(t.slots) == 2
        assert "<arg0>" in t.skeleton and "<arg1>" in t.skeleton


# --------------------------- φ 状态签名 ----------------------------------- #
class TestDistillState:
    def test_primary_key_from_observation(self):
        # window 里的文件名会被归一化成 <file>（见 normalize.py），
        # 这样同类任务换文件时状态主键仍能对上。
        s = D.distill_state({"app": "impress", "window": "a.pptx", "dialog": None})
        assert s.app == "impress"
        assert s.key() == ("impress", "<file>", "")

    def test_facts_not_in_key(self):
        # facts 不同但主键相同 → 同一状态身份
        s1 = D.distill_state({"app": "impress", "window": "a.pptx",
                              "facts": {"知道页数": True}})
        s2 = D.distill_state({"app": "impress", "window": "a.pptx",
                              "facts": {"知道页数": False}})
        assert s1.key() == s2.key()

    def test_missing_fields_have_defaults(self):
        s = D.distill_state({})
        assert s.app == "unknown"


# --------------------------- α 动作模板 ----------------------------------- #
class TestDistillAction:
    def test_tool_action(self):
        a = D.distill_action({"kind": "tool", "name": "duplicate_slide",
                              "args": {"idx": 2, "pos": 5}})
        assert a.kind == ActionKind.TOOL
        assert a.name == "duplicate_slide"
        assert a.arg_slots == ("idx", "pos")  # 参数名作槽位（排序）

    def test_tool_same_name_diff_values_same_template(self):
        # 同工具、同参数名、不同参数值 → 同一动作模板（值被丢弃 → 归堆）
        a1 = D.distill_action({"kind": "tool", "name": "dup",
                               "args": {"idx": 1, "pos": 2}})
        a2 = D.distill_action({"kind": "tool", "name": "dup",
                               "args": {"idx": 9, "pos": 8}})
        assert a1.key() == a2.key()

    def test_gui_action_drops_coordinates(self):
        # GUI 动作只保留语义描述，坐标不进模板
        a = D.distill_action({"kind": "gui", "description": "点击 Format 菜单",
                              "x": 123, "y": 456})
        assert a.kind == ActionKind.GUI
        assert a.name == "点击 Format 菜单"
        assert a.arg_slots == ()  # 无坐标、无参数槽

    def test_gui_same_description_same_template(self):
        a1 = D.distill_action({"kind": "gui", "description": "点击 Format 菜单",
                               "x": 1, "y": 1})
        a2 = D.distill_action({"kind": "gui", "description": "点击 Format 菜单",
                               "x": 999, "y": 999})
        assert a1.key() == a2.key()  # 坐标不同但归同一边

"""OSWorld 真实数据三道清洗的测试。

**用真实轨迹 fixture 验证**（tests/fixtures/rename_dir_gui_6steps.jsonl），
这是 B 阶段的核心：解析器必须能处理真实的 prediction 格式，而非构造的假数据。
"""

import json
from pathlib import Path

from evotree.distill_osworld import (
    OSWorldDistiller,
    extract_error_note,
    parse_prediction,
)
from evotree.types import ActionKind

FIXTURE = Path(__file__).parent / "fixtures" / "rename_dir_gui_6steps.jsonl"


def load_fixture() -> list[dict]:
    return [json.loads(line) for line in FIXTURE.open(encoding="utf-8")]


# --------------------------- parse_prediction ----------------------------- #
class TestParsePrediction:
    def test_parse_real_click(self):
        pred = ('\n<tool_call>\n{"name":"computer_use","arguments":'
                '{"action":"click","coordinate":[1854,887]}}\n</tool_call>\n'
                '<conclusion>Selected the folder.</conclusion>')
        call = parse_prediction(pred)
        assert call["name"] == "computer_use"
        assert call["arguments"]["action"] == "click"
        assert call["conclusion"] == "Selected the folder."

    def test_nested_tool_arguments_are_parsed(self):
        prediction = (
            '<tool_call>{"name":"os.example","arguments":'
            '{"config":{"a":1,"b":[1,2]},"path":"a"}}</tool_call>'
        )
        action = OSWorldDistiller().distill_action({"prediction": prediction})
        assert action.name == "os.example"
        assert action.arg_slots == ("config", "path")


        assert parse_prediction("just some text") is None


# --------------------------- α on real fixture ---------------------------- #
class TestDistillActionReal:
    def setup_method(self):
        self.D = OSWorldDistiller()
        self.steps = load_fixture()

    def test_click_becomes_gui_action_no_coords(self):
        # step 1 是 click；应清洗成 GUI 动作，名字含子类型，无坐标槽位
        a = self.D.distill_action(self.steps[1])
        assert a.kind == ActionKind.GUI
        assert a.name == "computer_use.click"
        assert a.arg_slots == ()  # 坐标丢弃

    def test_key_action(self):
        # step 2 是 F2 按键
        a = self.D.distill_action(self.steps[2])
        assert a.kind == ActionKind.GUI
        assert a.name == "computer_use.key"

    def test_type_action(self):
        # step 3 是 type
        a = self.D.distill_action(self.steps[3])
        assert a.name == "computer_use.type"

    def test_same_gui_subtype_aggregates(self):
        # step 1 和 step 4 都是 click（不同坐标）→ 应归到同一动作模板
        a1 = self.D.distill_action(self.steps[1])
        a4 = self.D.distill_action(self.steps[4])
        assert a1.key() == a4.key()  # 坐标不同但归堆

    def test_reset_falls_back(self):
        # step 0 是 reset（无 prediction）→ 回退，不崩
        a = self.D.distill_action(self.steps[0])
        assert a is not None


# --------------------------- 终止判定 ------------------------------------- #
class TestTerminal:
    def setup_method(self):
        self.D = OSWorldDistiller()
        self.steps = load_fixture()

    def test_terminate_is_terminal(self):
        # step 5 是 terminate → 应识别为终止（不进经验统计）
        assert self.D.is_terminal(self.steps[5]) is True

    def test_normal_action_not_terminal(self):
        # step 1 click → 不是终止
        assert self.D.is_terminal(self.steps[1]) is False


# --------------------------- 真实 MCP 工具解析 ---------------------------- #
class TestMcpToolParsing:
    def test_os_tool_becomes_tool_action(self):
        D = OSWorldDistiller()
        # 构造一个真 MCP 工具调用（os.rename_directory）
        pred = ('<tool_call>{"name":"os.rename_directory","arguments":'
                '{"old_path":"a","new_path":"b"}}</tool_call>'
                '<conclusion>renamed</conclusion>')
        a = D.distill_action({"prediction": pred})
        assert a.kind == ActionKind.TOOL
        assert a.name == "os.rename_directory"
        assert a.arg_slots == ("new_path", "old_path")  # 参数名排序，值丢弃

    def test_mcp_tool_aggregates_by_param_names(self):
        D = OSWorldDistiller()
        p1 = ('<tool_call>{"name":"os.rename_directory","arguments":'
              '{"old_path":"x","new_path":"y"}}</tool_call>')
        p2 = ('<tool_call>{"name":"os.rename_directory","arguments":'
              '{"old_path":"m","new_path":"n"}}</tool_call>')
        assert (D.distill_action({"prediction": p1}).key()
                == D.distill_action({"prediction": p2}).key())

    def test_executed_action_precedes_stale_prediction(self):
        D = OSWorldDistiller()
        raw = {
            "prediction": (
                '<tool_call>{"name":"computer_use","arguments":'
                '{"action":"right_click"}}</tool_call>'
            ),
            "action": {
                "action_type": "os.rename_directory",
                "parameters": {"old_path": "a", "new_path": "b"},
            },
        }

        action = D.distill_action(raw)

        assert action.kind == ActionKind.TOOL
        assert action.name == "os.rename_directory"
        assert action.arg_slots == ("new_path", "old_path")

    def test_structured_gui_action_is_supported(self):
        action = OSWorldDistiller().distill_action(
            {
                "action": {
                    "action_type": "computer_use.click",
                    "parameters": {"coordinate": [10, 20]},
                }
            }
        )

        assert action.kind == ActionKind.GUI
        assert action.name == "computer_use.click"
        assert action.arg_slots == ()


# --------------------------- τ / φ 复用占位版 ----------------------------- #
class TestTaskStateReused:
    def test_instruction_from_fixture(self):
        # fixture step 0 带 instruction，τ 能抽出模板
        D = OSWorldDistiller()
        steps = load_fixture()
        instr = steps[0]["instruction"]
        t = D.distill_task(instr)
        assert t.template_id  # 非空
        assert len(t.slots) >= 1  # "todo_list_Jan_1"/"..._2" 被挖成槽位


class TestErrorDetection:
    """extract_error_note 的回归锁。

    2026-09-06 实测：用子串匹配 "error" 会把 MCP 成功返回里的 ``is_error=False``
    判成失败，导致工具边平均 p 被压到 0.250（而真实报错为 0 次），推荐栏只剩
    GUI 动作，11 条 global_notes 全是成功返回值。
    """

    def test_mcp_success_is_not_an_error(self):
        ok = "CallToolResult(content=[TextContent(text='1.0')], is_error=False)"
        assert extract_error_note(None, {"exe_result": ok}) is None

    def test_mcp_failure_is_an_error(self):
        bad = "CallToolResult(content=[], is_error=True)"
        assert extract_error_note(None, {"exe_result": bad}) is not None

    def test_json_style_is_error_field(self):
        assert extract_error_note(None, {"exe_result": '{"isError": false}'}) is None
        assert extract_error_note(None, {"exe_result": '{"isError": true}'}) is not None

    def test_traceback_still_detected(self):
        tb = "Traceback (most recent call last): OSError"
        assert extract_error_note(None, {"exe_result": tb}) is not None

    def test_errno_still_detected(self):
        assert extract_error_note(None, {"exe_result": "OSError: [Errno 2] nope"}) is not None

    def test_plain_result_is_not_an_error(self):
        assert extract_error_note(None, {"exe_result": "{'result': 1.0}"}) is None

    def test_note_is_truncated(self):
        note = extract_error_note(None, {"exe_result": "x" * 400 + " Traceback"})
        assert note is not None and len(note) <= 201

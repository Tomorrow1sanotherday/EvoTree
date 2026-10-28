"""Regressions for incorrect experience found in the completed r6 pilot."""
import json

import pytest

from evotree.distill_osworld import OSWorldDistiller, extract_error_note
from evotree.memory import EvoTreeMemory
from evotree.osworld_adapter import OSWorldEpisodeAdapter
from evotree.osworld_integration import wrap_osworld
from evotree.forest import ExperienceTree
from evotree.retrieval import Retriever, render_cheatsheet
from evotree.notes import GlobalNotes
from evotree.types import ActionKind, ActionTmpl, StateSig


CALLS = [
    {"name": "computer_use", "arguments": {"action": "click", "coordinate": [527, 112]}},
    {"name": "computer_use", "arguments": {"action": "type", "text": "find . -type f -exec chmod 644 {} +", "enter": 1}},
    {"name": "computer_use", "arguments": {"action": "wait", "time": 2}},
    {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}},
]
REPLY = ''.join('<tool_call>' + json.dumps(c) + '</tool_call>' for c in CALLS)
ACTIONS = ["import pyautogui\npyautogui.click(523,110)",
           "import pyautogui\npyautogui.write('find . -type f -exec chmod 644 {} +')",
           "import time\ntime.sleep(2)", "DONE"]


@pytest.mark.parametrize("null_tool_call", [False, True])
def test_offline_multi_gui_turn_preserves_each_action_and_command(null_tool_call):
    records = [{"action": "reset", "instruction": "permissions"}] + [
        {"step_num": 1, "prediction": REPLY, "action": action} for action in ACTIONS
    ]
    if null_tool_call:
        for record in records[1:]:
            record["tool_call"] = None
    episode = OSWorldEpisodeAdapter().from_records(records, score=1)
    distilled = [OSWorldDistiller().distill_action(s.raw_action) for s in episode.steps]
    assert [a.name for a in distilled] == ["computer_use.click", "computer_use.type", "computer_use.wait"]
    assert distilled[1].args["text"] == CALLS[1]["arguments"]["text"]
    with pytest.raises(ValueError, match="Incomplete multi-action"):
        OSWorldEpisodeAdapter().from_records(records[:-1], score=1)


def test_report_summarizes_multi_gui_turn_with_null_tool_calls(tmp_path):
    from experiments.m2_small_eval.report import summarize_task

    records = [{"action": "reset", "instruction": "permissions"}] + [
        {"step_num": 1, "prediction": REPLY, "action": action, "tool_call": None}
        for action in ACTIONS
    ]
    (tmp_path / "traj.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8"
    )
    (tmp_path / "result.txt").write_text("1.0", encoding="utf-8")
    summary = summarize_task(tmp_path, "os", "permissions")
    assert summary["complete"] is True
    assert summary["score"] == 1.0
    assert summary["effective_actions"] == 3
    assert summary["gui_actions"] == 3
    assert summary["tool_actions"] == 0


def test_ambiguous_gui_action_cannot_silently_be_first_call():
    with pytest.raises(ValueError, match="Multiple tool calls"):
        OSWorldDistiller().distill_action({"action": ACTIONS[1], "prediction": REPLY})


def test_mcp_arguments_and_long_commands_are_preserved():
    args = {"source_range": "A1:A20", "target_sheet_name": "Sheet2", "target_start_cell": "A1"}
    action = OSWorldDistiller().distill_action({"action": {"action_type": "libreoffice_calc2.copy_cells_between_sheets", "parameters": args}})
    assert action.args == args
    command = "find . -type f " + "-print " * 40
    gui = OSWorldDistiller().distill_action({"tool_call": {"name": "computer_use", "arguments": {"action": "type", "text": command}}})
    assert gui.args["text"] == command.strip()


def test_protocol_success_and_business_failure_are_separate():
    output = "CallToolResult(content=[TextContent(type='text', text='false', annotations=None)], is_error=False)"
    assert extract_error_note({}, output, tool_name="libreoffice_calc.rename_sheet")
    assert extract_error_note({}, output, tool_name="os.get_do_not_disturb_status") is None
    assert extract_error_note({}, '{"content":[{"text":"false"}],"isError":false}', tool_name="libreoffice_calc.rename_sheet")
    assert extract_error_note({}, "false", tool_name="libreoffice_calc.rename_sheet")
    assert extract_error_note({}, "CallToolResult(content=[TextContent(text='true')], is_error=False)", tool_name="libreoffice_calc.rename_sheet") is None


def test_self_loop_does_not_claim_one_step_completion():
    tree = ExperienceTree("t")
    sig = StateSig("calc", "book")
    edge = tree.upsert_step(sig, ActionTmpl(ActionKind.TOOL, "save"), True, sig)
    edge.steps_after_success.append(1)
    sheet = Retriever().retrieve(tree, sig, GlobalNotes())
    assert sheet.recommendations[0].lookahead is None
    assert "一步收尾" not in render_cheatsheet(sheet)


def test_online_multi_actions_and_context_survive_memory_reload(tmp_path):
    class Agent:
        def reset(self):
            pass

        def predict(self, instruction, obs):
            return REPLY, ACTIONS

    class Env:
        def reset(self, **kwargs):
            return self._get_obs()

        def _get_obs(self):
            return {"cur_app": "terminal", "window": "Terminal", "app_info": None, "tool_list": []}

        def step(self, action):
            return self._get_obs(), 0, action == "DONE", {}

        def evaluate(self):
            return 1

    memory = EvoTreeMemory(tmp_path, OSWorldDistiller())
    agent, env, _ = wrap_osworld(Agent(), Env(), memory)
    obs = env.reset()
    _, actions = agent.predict("permissions", obs)
    for action in actions:
        obs, *_ = env.step(action)
    env.evaluate()
    restored = EvoTreeMemory(tmp_path, OSWorldDistiller())
    tree = restored.forest.get_tree(restored.distiller.distill_task("permissions").template_id)
    assert [e.action.name for e in tree.all_edges()] == ["computer_use.click", "computer_use.type", "computer_use.wait"]
    assert len(list(tree._nodes.values())) == 4
    assert all(e.from_id != e.to_success for e in tree.all_edges())
    agent.reset()
    env.reset()
    start = agent.memory_observation(env._get_obs())
    sheet = restored.read_cheatsheet("permissions", start, ["computer_use"])
    assert [r.action.name for r in sheet.recommendations] == ["computer_use.click"]


def test_business_false_reaches_persisted_cautions(tmp_path):
    call = {"name": "libreoffice_calc.rename_sheet", "arguments": {"old_name": "Sheet 1", "new_name": "LARS Resources"}}

    class Agent:
        def predict(self, instruction, obs):
            return '<tool_call>'+json.dumps(call)+'</tool_call>', [{"action_type":call['name'], "parameters":call['arguments']}]

    class Env:
        def reset(self):
            return {"cur_app":"libreoffice-calc", "window":"book.xlsx - LibreOffice Calc", "tool_list":[]}

        def step(self, action):
            obs=self.reset()
            obs['exe_result']="CallToolResult(content=[TextContent(type='text', text='false')], is_error=False)"
            return obs,0,False,{}

        def evaluate(self):
            return 0

    memory=EvoTreeMemory(tmp_path,OSWorldDistiller())
    aw,ew,_=wrap_osworld(Agent(),Env(),memory)
    obs=ew.reset()
    _,actions=aw.predict('rename worksheet',obs)
    after,*_=ew.step(actions[0])
    ew.evaluate()
    restored=EvoTreeMemory(tmp_path,OSWorldDistiller())
    assert restored.notes.query('libreoffice_calc.rename_sheet')
    sheet=restored.read_cheatsheet('rename worksheet',aw.memory_observation(after),['libreoffice_calc.rename_sheet'])
    assert sheet.cautions and not sheet.recommendations


def test_calc_context_separates_source_destination_and_content():
    from evotree.osworld_integration import build_observation, EvoTreeAgentWrapper
    from evotree.integration import EvoTreeRuntime
    aw=EvoTreeAgentWrapper(object(),EvoTreeRuntime(EvoTreeMemory(distiller=OSWorldDistiller())))
    aw._previous_action = "observed-action"
    def sig(name,data):
        obs={"cur_app":"libreoffice-calc", "window":"book.xlsx - LibreOffice Calc",
             "app_info":json.dumps({"name":name,"data":data}),"tool_list":[]}
        return aw.runtime.memory.distiller.distill_state(aw.memory_observation(obs)).key()
    assert sig('Sheet1','Revenue 100') != sig('Sheet2','')
    assert sig('Sheet2','') != sig('Sheet2','Revenue 100')


def test_injection_count_is_model_turns_not_action_rows(tmp_path):
    from experiments.m2_small_eval.report import prompt_marker_hits
    row={"step_num":1,"prediction":REPLY,"messages":[{"role":"user","content":"[经验] command"}]}
    path=tmp_path/'traj.jsonl'
    path.write_text('\n'.join(json.dumps(row,ensure_ascii=False) for _ in ACTIONS),encoding='utf-8')
    assert prompt_marker_hits(path)==(1,0)


def test_calc_duplicate_stdout_json_still_contributes_state():
    from evotree.osworld_integration import EvoTreeAgentWrapper
    from evotree.integration import EvoTreeRuntime
    aw=EvoTreeAgentWrapper(object(),EvoTreeRuntime(EvoTreeMemory(distiller=OSWorldDistiller())))
    aw._previous_action = "observed-action"
    value=json.dumps({"name":"Sheet1","data":"Revenue 100","row_count":2})
    base={"cur_app":"libreoffice-calc","window":"book.xlsx - LibreOffice Calc","app_info":value}
    single=aw.memory_observation(base)['state_context']
    duplicate=aw.memory_observation({**base,"app_info":value+'\n'+value})['state_context']
    empty=aw.memory_observation({**base,"app_info":None})['state_context']
    assert duplicate == single and duplicate != empty


def test_async_window_registration_does_not_change_task_root():
    from evotree.osworld_integration import EvoTreeAgentWrapper
    from evotree.integration import EvoTreeRuntime
    aw=EvoTreeAgentWrapper(object(),EvoTreeRuntime(EvoTreeMemory(distiller=OSWorldDistiller())))
    unknown={"cur_app":None,"apps":{},"app_info":None}
    calc={"cur_app":"libreoffice-calc","window":"book.xlsx - LibreOffice Calc",
          "app_info":json.dumps({"name":"Sheet1","data":"Revenue"})}
    distiller=aw.runtime.memory.distiller
    assert distiller.distill_state(aw.memory_observation(unknown)).key()==distiller.distill_state(aw.memory_observation(calc)).key()
    assert calc['cur_app']=='libreoffice-calc'

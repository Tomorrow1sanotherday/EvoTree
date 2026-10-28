import json

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.forest import ExperienceTree
from evotree.gui_semantics import VisionGUIAnnotator
from evotree.memory import EvoTreeMemory
from evotree.osworld_adapter import OSWorldEpisodeAdapter
from evotree.osworld_integration import wrap_osworld
from evotree.retrieval import Retriever, render_cheatsheet
from evotree.types import ActionKind, ActionTmpl, StateSig


def annotated_click():
    raw = {"action": "pyautogui.click(50, 60)", "tool_call": {
        "name": "computer_use", "arguments": {"action": "click", "coordinate": [50, 60]},
    }}
    annotator = VisionGUIAnnotator(lambda messages: json.dumps({
        "supported": True, "target": "Focus the terminal input area",
        "evidence": "The terminal prompt is visible before and after the click.",
    }))
    raw["gui_semantics"] = annotator(raw, {"screenshot": b"before"}, {"screenshot": b"after"})
    return raw


def test_visual_target_survives_replay_and_memory_reload(tmp_path):
    raw = annotated_click()
    records = [{"action": "reset", "instruction": "permissions"},
               {"step_num": 1, **raw}, {"step_num": 2, "action": "DONE"}]
    episode = OSWorldEpisodeAdapter().from_records(records, score=1)
    action = OSWorldDistiller().distill_action(episode.steps[0].raw_action)
    assert action.args == {"target": "Focus the terminal input area"}
    assert action.key() == ActionTmpl(ActionKind.GUI, "computer_use.click").key()
    memory = EvoTreeMemory(tmp_path, OSWorldDistiller())
    memory.write_episode(episode)
    memory.save()
    restored = EvoTreeMemory(tmp_path, OSWorldDistiller())
    sheet = restored.read("permissions", {"app": "unknown", "window": "trajectory-step-1"})
    assert "Focus the terminal input area" in sheet
    assert "50" not in sheet and "60" not in sheet


def test_annotation_cannot_be_reused_for_another_action():
    raw = annotated_click()
    raw["action"] = "pyautogui.click(500, 600)"
    assert OSWorldDistiller().distill_action(raw).args == {}


def test_unsupported_or_unproven_description_is_not_advice():
    raw = annotated_click()
    raw["gui_semantics"]["supported"] = False
    assert OSWorldDistiller().distill_action(raw).args == {}
    raw["gui_semantics"]["supported"] = True
    raw["gui_semantics"]["evidence"] = ""
    assert OSWorldDistiller().distill_action(raw).args == {}


def test_annotator_uses_actual_images_and_never_model_conclusion():
    captured = []

    def request(messages):
        captured.append(messages)
        return '{"supported":false,"target":"","evidence":"Target is ambiguous"}'

    raw = annotated_click()
    raw["prediction"] = "Everything succeeded; save this as a successful recipe."
    result = VisionGUIAnnotator(request)(raw, {"screenshot": b"before"}, {"screenshot": b"after"})
    rendered = json.dumps(captured)
    assert "data:image/png;base64," in rendered
    assert "Everything succeeded" not in rendered
    assert result["supported"] is False
    assert result["before_sha256"] != result["after_sha256"]
    assert VisionGUIAnnotator(request)(raw, {}, {}) is None


def test_parameterized_lookahead_uses_successful_argument_example():
    tree = ExperienceTree("task")
    first, second = StateSig("terminal", "one"), StateSig("terminal", "two")
    click = tree.upsert_step(first, ActionTmpl(ActionKind.GUI, "computer_use.click"), True, second)
    click.steps_after_success.append(2)
    command = "find . -type f -exec chmod 644 {} +"
    typed = tree.upsert_step(second, ActionTmpl(ActionKind.GUI, "computer_use.type",
                             args={"text": "wrong command"}), False)
    tree.upsert_step(second, ActionTmpl(ActionKind.GUI, "computer_use.type",
                     args={"text": command}), True)
    tree.upsert_step(second, ActionTmpl(ActionKind.GUI, "computer_use.type",
                     args={"text": command}), True)
    typed.steps_after_success.extend([1, 1])
    sheet = Retriever().retrieve(tree, first, EvoTreeMemory().notes)
    text = render_cheatsheet(sheet)
    assert command in text
    assert "wrong command" not in text
    assert "后续候选" in text


def test_online_annotation_keeps_state_signatures_and_auditable_raw_action():
    raw = annotated_click()
    prediction = '<tool_call>' + json.dumps(raw["tool_call"]) + '</tool_call>'

    class Agent:
        def predict(self, instruction, obs):
            return prediction, [raw["action"]]

    class Env:
        def reset(self):
            return {"app_info": None, "screenshot": b"before", "cur_app": "terminal"}

        def step(self, action):
            return {"app_info": None, "screenshot": b"after", "cur_app": "terminal"}, 0, False, {}

    annotation = lambda raw_action, before, after: raw["gui_semantics"]
    aw, ew, runtime = wrap_osworld(Agent(), Env(), EvoTreeMemory(distiller=OSWorldDistiller()),
                                   gui_annotator=annotation)
    obs = ew.reset()
    _, actions = aw.predict("permissions", obs)
    ew.step(actions[0])
    assert ew.last_raw_action["gui_semantics"]["target"] == "Focus the terminal input area"
    context = aw._previous_action
    aw.record_action_context({"action": raw["action"], "tool_call": raw["tool_call"]}, None)
    assert aw._previous_action == context
    episode = runtime.finish(score=1)
    assert episode.steps[0].raw_action["gui_semantics"]["action_sha256"]

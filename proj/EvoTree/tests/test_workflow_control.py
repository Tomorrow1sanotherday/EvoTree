import json
from types import SimpleNamespace

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.forest import ExperienceTree
from evotree.memory import EvoTreeMemory
from evotree.retrieval import Retriever, render_cheatsheet
from evotree.types import ActionKind, ActionTmpl, StateSig
from experiments.m2_small_eval.permission_order_probe import advice_variants, preparation_order, replace_advice
from experiments.m2_small_eval.workflow_control import ORDERS, WorkflowAgent, render_tree, render_workflow


def test_advice_ablation_preserves_shared_content_and_separates_variants():
    old = "[经验] 推荐：\n  computer_use.click；后续候选：computer_use.type"
    combined = old + "；曾用：find . -type f -exec chmod 644 {} +（1/1 成）\n      曾用：target=terminal input area（1/1 成）"
    variants = advice_variants(old, combined)
    assert "target=" in variants["target_only"] and "chmod" not in variants["target_only"]
    assert "target=" not in variants["parameters_only"] and "chmod" in variants["parameters_only"]
    assert "AFTER" in variants["ordered"] and "terminal input area" in variants["ordered"]
    messages = [{"role": "user", "content": [{"type": "text", "text": "instruction\n" + combined}]}]
    replaced = replace_advice(messages, combined, variants["original"], b"image")
    assert replaced[0]["content"][0]["text"] == "instruction\n" + old
    assert replaced[0]["content"][1]["type"] == "image_url"
    assert messages[0]["content"][0]["text"].endswith(combined)


def reply(actions):
    return "".join("<tool_call>" + json.dumps({"name": "computer_use", "arguments": args}) + "</tool_call>" for args in actions)


def test_first_decision_order_audit_does_not_count_late_focus_as_preparation():
    typed = {"action": "type", "text": "command"}
    click = {"action": "click", "coordinate": [10, 20]}
    late = preparation_order(reply([typed, click]))
    assert late["typing_without_preparation"] and not late["preparation_first"]
    focused = preparation_order(reply([click]))
    assert focused["preparation_first"] and focused["dependent_action_deferred"]
    assert not preparation_order(reply([typed]))["preparation_first"]


def test_plain_sequence_keeps_order_and_values_but_no_graph_statistics():
    episode = {"steps": [{"raw_action": {"action": {
        "action_type": "calc.copy", "parameters": {"source_range": "A1:A20", "target_sheet": "Sheet2"}}}},
        {"raw_action": {"action": {"action_type": "calc.save", "parameters": {}}}}]}
    text = render_workflow(episode, OSWorldDistiller(), 3000)
    assert text.index("calc.copy") < text.index("calc.save")
    assert "A1:A20" in text and "Sheet2" in text
    assert "success rate=" not in text and "后续候选" not in text
    with pytest.raises(ValueError, match="refusing to truncate"):
        render_workflow(episode, OSWorldDistiller(), 30)


def test_tree_render_uses_connected_next_edge_and_reports_sample_count():
    tree = ExperienceTree("task")
    start, end = StateSig("terminal", "start"), StateSig("terminal", "focused")
    edge = tree.upsert_step(start, ActionTmpl(ActionKind.GUI, "computer_use.click", args={"target": "terminal input"}), True, end)
    edge.steps_after_success.append(2)
    after = tree.upsert_step(end, ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "chmod command"}), True)
    after.steps_after_success.append(1)
    sheet = Retriever().retrieve(tree, start, EvoTreeMemory().notes)
    memory = SimpleNamespace(forest=SimpleNamespace(get_tree=lambda _: tree),
        distiller=SimpleNamespace(distill_task=lambda _: SimpleNamespace(template_id="task")),
        _diagnostic_renderer=render_cheatsheet)
    ordered = render_tree(sheet, memory, "task", "ordered", 3000)
    assert ordered.index("CURRENT ACTION") < ordered.index("DEPENDENT NEXT ACTION")
    assert "terminal input" in ordered and "chmod command" in ordered
    assert "count=1" in ordered
    original = render_tree(sheet, memory, "task", "original", 3000)
    assert "[经验]" in original


def test_plain_workflow_injection_leaves_available_tools_unchanged():
    seen = []
    agent = SimpleNamespace(predict=lambda instruction, obs: seen.append((instruction, obs)))
    wrapper = WorkflowAgent(agent, "recipe")
    obs = {"tool_list": ["first", "second"]}
    wrapper.predict("task", obs)
    assert seen == [("task\n\nrecipe", obs)]
    assert seen[0][1] is obs
    assert {order[0] for order in ORDERS} == {"baseline", "workflow", "tree"}

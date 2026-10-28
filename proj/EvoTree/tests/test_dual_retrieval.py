"""Behavioral guarantees for task + cross-task state memory."""
import hashlib
import json

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.dual_retrieval import action_stage, render_dual_cheatsheet
from evotree.episode import Episode, EpisodeStep
from evotree.integration import EvoTreeRuntime
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld


COPY = 'Copy the "Revenue" column to a new sheet named "Sheet2".'
NEW_COPY = 'Copy the "Operating Cost" column to a new sheet named "CostSummary".'
OTHER = 'Set the page orientation to landscape.'


def raw(name, **args):
    return {"action": {"action_type": name, "parameters": args}}


def seed(memory, instruction=COPY, success=True):
    states = [dict(app="calc", window="old.xlsx — Calc", state_context=f"old-{i}")
              for i in range(3)]
    actions = [raw("libreoffice_calc.create_sheet", sheet_name="Sheet2"),
               raw("libreoffice_calc.switch_active_sheet", sheet_name="Sheet1"),
               raw("libreoffice_calc2.copy_cells_between_sheets", source_range="B1:B20",
                   target_sheet_name="Sheet2", target_start_cell="A1")]
    episode = Episode(instruction, tuple(EpisodeStep(i, states[i], a,
                       states[i + 1] if i < 2 else None) for i, a in enumerate(actions)),
                       success, source="measured-history")
    memory.write_episode(episode)
    return states, actions


def query(actions, context="new-2"):
    return dict(app="calc", window="new.xlsx — Calc", state_context=context,
                app_info={"name": "CurrentSource", "data": "1 | Operating Cost |"},
                state_progress={"copy_call_observed": False},
                state_stage=action_stage(OSWorldDistiller().distill_action(actions[1])))


def test_both_routes_survive_parameter_change_and_preserve_workflow_order():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False, retrieval_mode="dual")
    _, actions = seed(memory)
    sheet = memory.read_cheatsheet(NEW_COPY, query(actions))
    assert len(sheet.task_experiences) == 1
    assert sheet.state_experiences
    text = memory.render(sheet)
    assert "Task-level experience" in text and "Horizontal state experience" in text
    task_text = text.split("[当前状态经验")[0]
    assert task_text.index("create_sheet") < task_text.index("switch_active_sheet") < task_text.index("copy_cells")
    assert '"Sheet1"' in text and '"Sheet2"' in text
    assert "CostSummary" not in text and "Operating Cost" not in text
    assert "Historical examples only" in text


def test_task_miss_does_not_prevent_cross_tree_state_hit():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False)
    _, actions = seed(memory)
    sheet = memory.read_dual_cheatsheet('Copy the "Operating Cost" column and its header to a new sheet named "CostSummary".', query(actions))
    assert not sheet.task_experiences
    assert sheet.state_experiences
    assert sheet.state_experiences[0].template_id == memory.lookup_task(COPY).template_id


def test_new_state_retains_task_workflow():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    seed(memory)
    sheet = memory.read_dual_cheatsheet(NEW_COPY, dict(app="terminal", window="Shell"))
    assert sheet.task_experiences and not sheet.state_experiences
    assert "switch_active_sheet" in memory.render(sheet)


@pytest.mark.parametrize("changes", [dict(app="impress"), dict(dialog="Save As"),
                                     dict(state_stage="different-stage"), dict(app="unknown")])
def test_unsafe_state_matches_are_rejected(changes):
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False)
    _, actions = seed(memory)
    obs = query(actions)
    obs.update(changes)
    assert not memory.read_dual_cheatsheet(NEW_COPY, obs).state_experiences


def test_boolean_fact_conflicts_reject_match():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False)
    states, actions = seed(memory)
    tree = memory.forest.get_tree(memory.lookup_task(COPY).template_id)
    tree.node_for_signature(memory.distiller.distill_state(states[2])).sig.facts["source_active"] = True
    obs = query(actions)
    obs["facts"] = {"source_active": False}
    assert not memory.read_dual_cheatsheet(NEW_COPY, obs).state_experiences


def test_sources_keep_separate_success_statistics_and_disk_trees(tmp_path):
    memory = EvoTreeMemory(tmp_path, OSWorldDistiller(), strict_quality=False)
    _, actions = seed(memory)
    seed(memory, OTHER)
    seed(memory, OTHER)
    seed(memory, OTHER, False)
    seed(memory, OTHER, False)
    memory.save()
    # Reload without touching either tree by task ID. Horizontal lookup must
    # still include both persisted sources.
    frozen = EvoTreeMemory(tmp_path, write_enabled=False, strict_quality=False)
    before = digest(tmp_path)
    sheet = frozen.read_dual_cheatsheet('Copy the "Operating Cost" column and its header to a new sheet named "CostSummary".', query(actions))
    rates = {item.template_id: item.recommendation.success_rate for item in sheet.state_experiences}
    assert rates[memory.lookup_task(COPY).template_id] == 1
    assert rates[memory.lookup_task(OTHER).template_id] == .5
    assert digest(tmp_path) == before


def digest(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_frozen_queries_do_not_change_bank_or_template_library(tmp_path):
    memory = EvoTreeMemory(tmp_path, OSWorldDistiller())
    _, actions = seed(memory)
    memory.save()
    frozen = EvoTreeMemory(tmp_path, OSWorldDistiller(template_library=memory.distiller.template_library),
                           write_enabled=False, strict_quality=False, retrieval_mode="dual")
    before = digest(tmp_path)
    templates = json.dumps(frozen.distiller.template_library.to_dict(), sort_keys=True)
    first = frozen.read(NEW_COPY, query(actions))
    frozen.read("An entirely unseen instruction.", query(actions))
    frozen.write_episode(Episode("Test result.", (), True))
    frozen.save()
    assert frozen.read(NEW_COPY, query(actions)) == first
    assert digest(tmp_path) == before
    assert json.dumps(frozen.distiller.template_library.to_dict(), sort_keys=True) == templates


def test_budget_omits_whole_large_task_but_keeps_state_route():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False)
    _, actions = seed(memory)
    history = next(iter(memory.task_experiences.values()))[0]
    history.instruction += "x" * 20000
    sheet = memory.read_dual_cheatsheet(NEW_COPY, query(actions))
    text = render_dual_cheatsheet(sheet, 6000)
    assert len(text) <= 6000 and "Horizontal state experience" in text
    assert "Task-level experience" not in text and sheet.omitted
    assert sheet.rendered_task_records == 0
    assert sheet.rendered_state_records == len(sheet.state_experiences)


def test_render_audit_distinguishes_accepted_from_omitted_state():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False)
    _, actions = seed(memory)
    sheet = memory.read_dual_cheatsheet(NEW_COPY, query(actions))
    assert sheet.state_experiences
    text = render_dual_cheatsheet(sheet, 6000)
    assert sheet.rendered_state_records > 0
    assert sheet.rendered_state_records <= len(sheet.state_experiences)
    assert sum("state:" in item for item in sheet.omitted) + sheet.rendered_state_records == len(sheet.state_experiences)
    assert text.count("\nCandidate: ") == sheet.rendered_state_records
    for item in sheet.state_experiences:
        item.notes = ['x' * 10000]
    text = render_dual_cheatsheet(sheet, 6000)
    assert sheet.rendered_state_records == 0
    assert '\nCandidate: ' not in text
    assert len([item for item in sheet.omitted if item.startswith('state:')]) == len(sheet.state_experiences)


@pytest.mark.parametrize('verdict,admitted', [('compatible', True), ('conflict', False), ('unknown', False)])
def test_semantic_verdict_survives_full_retrieval_and_render(verdict, admitted):
    # Fixed verdicts test the plumbing, not the quality of the model's judgment.
    from evotree.situation_summary import ModelSituationBackend

    def complete(messages):
        if 'state summarizer' in messages[0]['content']:
            return json.dumps(dict(sentence='The source worksheet is active.',
                                   evidence=['Visible worksheet data.'], unknowns=[]))
        payload = json.loads(messages[1]['content'])
        return json.dumps(dict(matches=[dict(id=row['id'], verdict=verdict, score=.9,
            reason='Fixed routing control.') for row in payload['historical']]))

    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False,
        retrieval_mode='dual_summary', situation_backend=ModelSituationBackend(complete))
    _, actions = seed(memory)
    sheet = memory.read_cheatsheet(NEW_COPY, query(actions))
    assert sheet.recall_candidates and sheet.summary_comparisons
    assert bool(sheet.state_experiences) is admitted
    text = memory.render(sheet)
    assert bool(sheet.rendered_state_records) is admitted
    assert ('\nCandidate: ' in text) is admitted


@pytest.mark.parametrize("mode,task,state", [("task", True, False), ("state", False, True), ("dual", True, True)])
def test_runtime_uses_selected_routes_and_fixed_tool_order(mode, task, state):
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False, retrieval_mode=mode)
    _, actions = seed(memory)
    runtime = EvoTreeRuntime(memory)
    runtime.begin(NEW_COPY)
    result = runtime.before_model(query(actions), ["unrelated", "libreoffice_calc2.copy_cells_between_sheets"])
    assert bool(result.cheatsheet.task_experiences) == task
    assert bool(result.cheatsheet.state_experiences) == state
    assert ("Task-level experience" in result.text) == task
    assert ("Horizontal state experience" in result.text) == state
    assert result.reranked_tools[0] == "unrelated"


def test_wrapper_stage_ignores_parameter_values_and_resets():
    class Agent:
        def reset(self):
            pass
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    agent, env, runtime = wrap_osworld(Agent(), object(), memory)
    agent.record_action_context(raw("libreoffice_calc.switch_active_sheet", sheet_name="Sheet1"), None)
    stage = agent.memory_observation({})["state_stage"]
    agent.record_action_context(raw("libreoffice_calc.switch_active_sheet", sheet_name="NewSource"), None)
    assert agent.memory_observation({})["state_stage"] == stage
    agent.reset()
    assert agent.memory_observation({})["state_stage"] == "start"


def test_failed_or_quarantined_episode_never_becomes_success_workflow():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    states, actions = seed(memory, success=False)
    episode = Episode(COPY, (EpisodeStep(0, states[0], actions[0]),), True,
                      metadata={"early_termination": True})
    memory.write_episode(episode)
    assert not memory.task_experiences

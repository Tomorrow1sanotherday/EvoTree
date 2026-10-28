"""Source/target and progress regressions for horizontal copy advice."""
import json

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.dual_retrieval import action_stage
from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld
from evotree.state_conditions import COPY_TOOL, SWITCH_TOOL, current_conditions, parse_app_info


OLD = 'Copy the "Revenue" column along with the header to a new sheet named "Sheet2".'
NEW = 'Copy the "Inventory Value" column along with the header to a new sheet named "Extract43".'


def raw(name, **args):
    return {"action": {"action_type": name, "parameters": args}}


def bank():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False, retrieval_mode='dual')
    actions = [raw('libreoffice_calc.env_info', sheet_name='Sheet1'),
               raw(SWITCH_TOOL, sheet_name='Sheet2'), raw(SWITCH_TOOL, sheet_name='Sheet1'),
               raw(COPY_TOOL, source_range='A1:A20', target_sheet_name='Sheet2', target_start_cell='A1'),
               raw('libreoffice_calc.env_info', sheet_name='Sheet2'), raw('libreoffice_calc.save')]
    states = [dict(app='libreoffice-calc', window='old.xlsx - Calc', state_context=f'old-{index}')
              for index in range(len(actions))]
    memory.write_episode(Episode(OLD, tuple(EpisodeStep(index, states[index], action,
                         states[index+1] if index+1 < len(states) else None)
                         for index, action in enumerate(actions)), True))
    return memory, actions


def observation(memory, action, active='Records23', copied=False, context='new-context'):
    return dict(app='libreoffice-calc', window='new.xlsx - Calc', state_context=context,
                state_stage=action_stage(memory.distiller.distill_action(action)),
                state_progress={'copy_call_observed': copied},
                app_info={'name': active, 'data': '1 | Control 1 | Inventory Value |\n2 | 1 | 2 |'})


def actions_of(sheet):
    return [experience.recommendation.action.name for experience in sheet.state_experiences]


def test_target_active_recommends_switch_back_and_excludes_direct_copy():
    memory, actions = bank()
    sheet = memory.read_dual_cheatsheet(NEW, observation(memory, actions[1], active='Extract43'))
    assert actions_of(sheet) == [SWITCH_TOOL]
    assert sheet.state_experiences[0].recommendation.action.args['sheet_name'] == 'Sheet1'
    assert sheet.observed_conditions['active_sheet_role'] == 'target'
    assert any(item['action'] == COPY_TOOL for item in sheet.rejected_states)
    assert sheet.task_experiences
    text = memory.render(sheet)
    assert '"active_sheet_role": "target"' in text
    assert 're-observe and confirm' in text
    # Only source role names are injected; current task values remain external.
    assert 'Extract43' not in text and 'Records23' not in text


def test_source_active_recommends_copy_without_switch_back_candidate():
    memory, actions = bank()
    sheet = memory.read_dual_cheatsheet(NEW, observation(memory, actions[2]))
    assert actions_of(sheet) == [COPY_TOOL]
    assert sheet.state_experiences[0].preconditions == {'active_sheet_role':'source', 'copy_call_observed':False}


@pytest.mark.parametrize('info', [None, '{broken', {'name':'Scratch','data':'1 | Notes |'},
                                   {'name':'Records23','data':'1 | Other |\n2 | Inventory Value |'}])
def test_unknown_source_abstains_without_losing_task_workflow(info):
    memory, actions = bank()
    obs = observation(memory, actions[2])
    obs['app_info'] = info
    sheet = memory.read_dual_cheatsheet(NEW, obs)
    assert COPY_TOOL not in actions_of(sheet)
    assert not sheet.state_experiences
    assert sheet.task_experiences and 'Task-level experience' in memory.render(sheet)


def test_role_guard_applies_even_if_original_context_matches_exactly():
    memory, actions = bank()
    # s3 is historically the source-active copy state. Exact opaque context
    # must not override a currently observed target-role conflict.
    obs = observation(memory, actions[2], active='Extract43', context='old-3')
    sheet = memory.read_dual_cheatsheet(NEW, obs)
    assert COPY_TOOL not in actions_of(sheet)


def test_initial_query_does_not_retrieve_post_copy_save():
    memory, actions = bank()
    before = memory.read_dual_cheatsheet(NEW, observation(memory, actions[0]))
    assert 'libreoffice_calc.save' not in actions_of(before)
    assert SWITCH_TOOL in actions_of(before)
    after = memory.read_dual_cheatsheet(NEW, observation(memory, actions[4], copied=True))
    assert actions_of(after) == ['libreoffice_calc.save']


def test_missing_progress_cannot_be_assumed_false_or_true():
    memory, actions = bank()
    obs = observation(memory, actions[2])
    del obs['state_progress']
    assert not memory.read_dual_cheatsheet(NEW, obs).state_experiences


def test_unresolved_historical_role_cannot_remove_copy_guard():
    memory, actions = bank()
    history = next(iter(memory.task_experiences.values()))[0]
    # Missing source-switch evidence in a legacy workflow must not permit a
    # direct copy just because no derived historical conditions are available.
    del history.steps[2]
    obs = observation(memory, actions[2], active='Extract43')
    assert COPY_TOOL not in actions_of(memory.read_dual_cheatsheet(NEW, obs))


def test_unknown_historical_action_stage_never_bypasses_role_guard():
    memory, actions = bank()
    history = next(iter(memory.task_experiences.values()))[0]
    tree = memory.forest.get_tree(history.template_id)
    # Merge the two switch stages while retaining their distinct action args.
    first, second = history.steps[1].state, history.steps[2].state
    first_node = tree.node_for_signature(first)
    second_node = tree.node_for_signature(second)
    assert first_node is not None and second_node is not None
    old_key = second.key()
    second_node.sig = first
    del tree._nodes[old_key]
    tree._nodes[first.key()] = first_node
    sheet = memory.read_dual_cheatsheet(NEW, observation(memory, actions[2], active='Extract43'))
    assert COPY_TOOL not in actions_of(sheet)
    assert any(item['action'] == COPY_TOOL for item in sheet.rejected_states)


def test_duplicated_app_info_requires_consistent_objects():
    info = dict(name='Records23', data='1 | Inventory Value |')
    encoded = json.dumps(info)
    assert parse_app_info(encoded + '\n' + encoded) == info
    assert parse_app_info(encoded + '\n' + json.dumps({'name':'Extract43'})) == {}
    assert current_conditions(NEW, {'app_info': encoded + '\n' + encoded})['active_sheet_role'] == 'source'


def test_wrapper_tracks_observed_copy_call_and_clears_on_reset_and_failure():
    class Agent:
        def reset(self):
            pass
    memory, actions = bank()
    agent, _, runtime = wrap_osworld(Agent(), object(), memory)
    assert agent.memory_observation({})['state_progress']['copy_call_observed'] is False
    agent.record_action_context(actions[3], None)
    assert agent.memory_observation({})['state_progress']['copy_call_observed'] is True
    agent.record_action_context(actions[3], 'tool failed')
    assert agent.memory_observation({})['state_progress']['copy_call_observed'] is False
    agent.record_action_context(actions[3], None)
    agent.reset()
    assert agent.memory_observation({})['state_progress']['copy_call_observed'] is False


def test_actual_wrapper_uses_observed_active_sheet_not_last_switch_argument():
    memory, actions = bank()
    agent, _, runtime = wrap_osworld(object(), object(), memory)
    runtime.begin(NEW)
    # A successful protocol call alone cannot prove which sheet is now active.
    # Feed a contrary live observation: source-role evidence must win.
    agent.record_action_context(actions[1], None)
    source_obs = dict(cur_app='libreoffice-calc', window='new.xlsx - Calc',
                      app_info={'name':'Records23', 'data':'1 | Inventory Value |'})
    result = runtime.before_model(agent.memory_observation(source_obs), [])
    assert actions_of(result.cheatsheet) == [COPY_TOOL]
    target_obs = dict(source_obs, app_info={'name':'Extract43', 'data':'1 | |'})
    result = runtime.before_model(agent.memory_observation(target_obs), [])
    assert actions_of(result.cheatsheet) == [SWITCH_TOOL]
    missing_obs = dict(source_obs, app_info=None)
    result = runtime.before_model(agent.memory_observation(missing_obs), [])
    assert not result.cheatsheet.state_experiences
    assert result.cheatsheet.task_experiences


def test_unfiltered_dual_is_an_explicit_diagnostic_ablation_only():
    memory, actions = bank()
    obs = observation(memory, actions[2], active='Extract43')
    guarded = memory.read_dual_cheatsheet(NEW, obs)
    unfiltered = memory.read_dual_cheatsheet(NEW, obs, enforce_state_conditions=False)
    assert COPY_TOOL not in actions_of(guarded)
    assert COPY_TOOL in actions_of(unfiltered)
    assert memory.retrieval_mode == 'dual'

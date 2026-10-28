"""Tests for evidence-grounded situation summaries and semantic retrieval."""
import json

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.dual_retrieval import action_stage
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import ModelSituationBackend, SituationSummary


def reply(messages):
    system = messages[0]['content']
    content = messages[1]['content']
    if isinstance(content, list):
        content = next(item['text'] for item in content if item.get('type') == 'text')
    payload = json.loads(content)
    if 'Describe the CURRENT situation' in system:
        active = payload['observation'].get('app_info', {}).get('name', 'unknown')
        role = {'Extract43': 'target', 'Records23': 'source'}.get(active, 'unknown')
        return json.dumps(dict(sentence=f'Current active sheet is the {role} role.',
                               evidence=[f'active_sheet={active}'] if role != 'unknown' else [],
                               unknowns=[] if role != 'unknown' else ['active sheet role']))
    current = payload['current']['situation']['sentence']
    matches = []
    for candidate in payload['historical']:
        old = candidate['situation']['sentence']
        conflict = ('target role' in current and 'source role' in old)
        matches.append(dict(id=candidate['id'], verdict='conflict' if conflict else 'compatible',
                            score=0.1 if conflict else 0.95,
                            reason='Explicitly different active-sheet roles.' if conflict else 'Same active-sheet role.'))
    return json.dumps(dict(matches=matches))


def test_summary_backend_distinguishes_roles_across_instance_names():
    backend = ModelSituationBackend(reply)
    current = backend.summarize('Copy a column to a new sheet.',
                                {'app_info': {'name': 'Extract43', 'data': '1 | |'}})
    target = backend.summarize('Copy a column to a new sheet.',
                               {'app_info': {'name': 'Sheet2', 'data': '1 | |'}})
    source = backend.summarize('Copy a column to a new sheet.',
                               {'app_info': {'name': 'Records23', 'data': '1 | Revenue |'}})
    judgments = backend.compare(current, [target, source])
    assert judgments[0]['verdict'] == 'compatible'
    assert judgments[1]['verdict'] == 'conflict'
    assert backend.model_calls == 4


def test_unknown_is_preserved_instead_of_guessed():
    backend = ModelSituationBackend(reply)
    current = backend.summarize('Do something.', {'app': 'libreoffice-calc'})
    assert current.unknowns == ('active sheet role',)
    assert current.evidence == ()


def test_summary_retries_empty_response_then_caches_only_valid_schema():
    calls = []
    valid = json.dumps(dict(sentence='The active sheet is measured.',
                            evidence=['active_sheet=Sheet1'], unknowns=[]))
    def complete(messages):
        calls.append(messages)
        return None if len(calls) == 1 else valid
    backend = ModelSituationBackend(complete)
    summary = backend.summarize('Inspect the sheet.', {'app_info': {'name': 'Sheet1'}})
    assert summary.sentence == 'The active sheet is measured.'
    assert len(calls) == backend.model_calls == 2
    assert 'did not satisfy the schema' in calls[1][-1]['content']
    assert backend.summarize('Inspect the sheet.', {'app_info': {'name': 'Sheet1'}}) == summary
    assert len(calls) == 2


def test_compare_retries_missing_candidate_and_validates_ids():
    calls = []
    def complete(messages):
        calls.append(messages)
        rows = [dict(id=0, verdict='compatible', score=.9, reason='Same observed role.')]
        if len(calls) > 1:
            rows.append(dict(id=1, verdict='unknown', score=.2, reason='Insufficient evidence.'))
        return json.dumps(dict(matches=rows))
    backend = ModelSituationBackend(complete)
    result = backend.compare(SituationSummary('Current.'),
        [SituationSummary('Historical one.'), SituationSummary('Historical two.')])
    assert [row['verdict'] for row in result] == ['compatible', 'unknown']
    assert len(calls) == backend.model_calls == 2
    assert 'Required IDs: [0, 1]' in calls[1][-1]['content']
    assert 'received IDs: [0]' in calls[1][-1]['content']


def test_summary_quotes_task_as_reference_data_and_preserves_observation():
    captured = []
    def complete(messages):
        captured.append(messages)
        return json.dumps(dict(sentence='Slide 3 is active with two text boxes.',
            evidence=['current_slide=3'], unknowns=[]))
    task = 'How do I change every font? Give me steps.'
    ModelSituationBackend(complete).summarize(task, {'app_info': {'current_slide': 3}})
    payload = json.loads(captured[0][1]['content'][0]['text'])
    assert 'task' not in payload
    assert payload['task_context']['quoted_task_goal'] == task
    assert payload['observation']['app_info']['current_slide'] == 3
    assert 'NEVER answer it' in captured[0][0]['content']
    assert 'Do not use summary/message/status' in captured[0][0]['content']


def test_summary_retry_reports_actual_missing_and_unexpected_fields():
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return json.dumps(dict(status='blocked', message='Use the font menu.'))
        return json.dumps(dict(sentence='Slide 3 is active.', evidence=[], unknowns=[]))
    summary = ModelSituationBackend(complete).summarize('How do I change the font?', {'app': 'impress'})
    assert summary.sentence == 'Slide 3 is active.'
    correction = calls[1][-1]['content']
    assert 'Validation error:' in correction
    assert "'sentence'" in correction and "'status'" in correction


def test_summary_rejects_advice_shaped_fields_instead_of_renaming_them():
    from evotree.situation_summary import SituationResponseError
    backend = ModelSituationBackend(lambda messages: json.dumps(dict(
        summary='Use the font menu.', evidence=[], unknowns=[])))
    with pytest.raises(SituationResponseError):
        backend.summarize('Change font.', {'app': 'impress'})
    assert backend.model_calls == 2


def test_compare_retry_reports_missing_matches_array():
    calls = []
    def complete(messages):
        calls.append(messages)
        if len(calls) == 1:
            return '{"recommendation":"Use a macro."}'
        return '{"matches":[{"id":0,"verdict":"unknown","score":0.5,"reason":"Unknown role."}]}'
    backend = ModelSituationBackend(complete)
    backend.compare(SituationSummary('Slide active.'), [SituationSummary('Other slide active.')])
    assert 'matches must be an array; required candidate IDs: [0]' in calls[1][-1]['content']


@pytest.mark.parametrize('failure', ['empty', 'timeout'])
def test_dual_summary_fails_closed_without_dropping_task_experience(failure):
    def unavailable(messages):
        if failure == 'timeout':
            raise TimeoutError('test timeout')
        return None
    backend = ModelSituationBackend(unavailable)
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False,
                           retrieval_mode='dual_summary', situation_backend=ModelSituationBackend(reply))
    instruction = 'Set the spreadsheet zoom to a readable level.'
    observation = dict(app='libreoffice-calc', window='book.xlsx - Calc',
                       state_stage='start', app_info={'name': 'Sheet1'})
    action = dict(action_type='libreoffice_calc.set_zoom_level',
                  parameters={'zoom_percentage': 80})
    from evotree.episode import Episode, EpisodeStep
    memory.write_episode(Episode(instruction, (EpisodeStep(0, observation, action),), True))
    memory.situation_backend = backend
    sheet = memory.read_dual_cheatsheet(instruction, dict(observation), use_situation_summaries=True)
    assert sheet.task_experiences
    assert sheet.state_experiences == []
    assert sheet.recommendations == []
    assert sheet.summary_failures == [dict(stage='state-verification',
        error_type='SituationResponseError' if failure == 'empty' else 'SituationTransportError', candidates=1)]
    assert backend.model_calls == (2 if failure == 'empty' else 1)


@pytest.mark.parametrize('error', [InterruptedError, AttributeError])
def test_summary_backend_does_not_hide_interruptions_or_code_errors(error):
    def broken(messages):
        raise error('test')
    with pytest.raises(error):
        ModelSituationBackend(broken).summarize('Inspect.', {'app': 'calc'})


def test_current_summary_cannot_see_private_labels_or_future_actions():
    captured=[]
    def complete(messages):
        captured.append(messages)
        return json.dumps(dict(sentence='The current object is observed.',evidence=[],unknowns=[]))
    backend=ModelSituationBackend(complete)
    backend.summarize('Change the document.',{'app':'impress','app_info':{'slide':7},
        'private':{'correct_phase':'all-done'},'score':1,'future_action':'save',
        'next_observation':{'secret':'future'}},
        [{'name':'read_state','arguments':{}}])
    payload=json.loads(captured[0][1]['content'][0]['text'])
    assert payload['executed_prefix']==[{'name':'read_state','arguments':{}}]
    assert 'all-done' not in json.dumps(payload)
    assert 'future' not in json.dumps(payload)


def test_compare_prompt_contains_no_candidate_actions_or_outcomes():
    captured = []
    def complete(messages):
        captured.append(messages)
        return json.dumps({'matches': [dict(id=0, verdict='unknown', score=0.5, reason='Not enough evidence.')]})
    backend = ModelSituationBackend(complete)
    current = SituationSummary('An object is active.')
    backend.compare(current, [SituationSummary('A different object is active.')])
    content = captured[0][1]['content']
    if isinstance(content, list):
        content = next(item['text'] for item in content if item.get('type') == 'text')
    payload = json.loads(content)
    assert set(payload) == {'current', 'historical'}
    assert 'score' not in json.dumps(payload).lower()
    assert all('action' not in json.dumps(row) and 'recommendation' not in row
               for row in payload['historical'])


def test_comparison_receives_both_task_goals_to_align_different_targets():
    captured = []
    def complete(messages):
        captured.append(messages)
        return json.dumps({'matches': [dict(id=0, verdict='compatible', score=.9, reason='Both first targets are met.')]})
    backend = ModelSituationBackend(complete)
    backend.compare(SituationSummary('First textbox 50 pt; second 16 pt.'),
        [SituationSummary('First textbox 60 pt; second 22 pt.')],
        goals=dict(current='Set first to 50 and second to 24.',
                   historical=['Set first to 60 and second to 28.']))
    content = captured[0][1]['content']
    payload = json.loads(content)
    assert payload['current']['task_goal'] == 'Set first to 50 and second to 24.'
    assert payload['historical'][0]['task_goal'] == 'Set first to 60 and second to 28.'
    assert 'compare these statuses' in captured[0][0]['content']


def test_summary_prompt_requires_unknown_when_observation_sources_conflict():
    assert 'structured current-state fields against the screenshot' in __import__(
        'evotree.situation_summary', fromlist=['SUMMARY_INSTRUCTION']).SUMMARY_INSTRUCTION
    assert 'mark the object role unknown' in __import__(
        'evotree.situation_summary', fromlist=['SUMMARY_INSTRUCTION']).SUMMARY_INSTRUCTION


def test_dual_summary_is_explicit_and_requires_a_backend():
    with pytest.raises(ValueError, match='situation_backend'):
        EvoTreeMemory(distiller=OSWorldDistiller(), retrieval_mode='dual_summary')


def test_dual_summary_refines_candidates_by_situation_not_tool_name():
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False,
                           retrieval_mode='dual_summary', situation_backend=ModelSituationBackend(reply))
    instruction = 'Copy the "Revenue" column to a new sheet named "Sheet2".'
    actions = [
        {'action': {'action_type': 'libreoffice_calc.switch_active_sheet',
                    'parameters': {'sheet_name': 'Sheet2'}}},
        {'action': {'action_type': 'libreoffice_calc.switch_active_sheet',
                    'parameters': {'sheet_name': 'Sheet1'}}},
        {'action': {'action_type': 'libreoffice_calc2.copy_cells_between_sheets',
                    'parameters': {'source_range': 'A1:A20', 'target_sheet_name': 'Sheet2',
                                   'target_start_cell': 'A1'}}},
    ]
    observations = [
        dict(app='libreoffice-calc', window='book.xlsx - Calc', state_context='ctx-target',
             app_info={'name': 'Sheet2', 'data': '1 | |'}),
        dict(app='libreoffice-calc', window='book.xlsx - Calc', state_context='ctx-source',
             app_info={'name': 'Sheet1', 'data': '1 | Revenue |'}),
        dict(app='libreoffice-calc', window='book.xlsx - Calc', state_context='ctx-copy',
             app_info={'name': 'Sheet1', 'data': '1 | Revenue |'}),
    ]
    states = [dict(value, state_stage='start' if index == 0 else action_stage(
        memory.distiller.distill_action(actions[index-1]))) for index, value in enumerate(observations)]
    from evotree.episode import Episode, EpisodeStep
    memory.write_episode(Episode(instruction, tuple(EpisodeStep(index, states[index], actions[index],
        states[index+1] if index + 1 < len(states) else None) for index in range(3)), True))
    # Query the stage after switching back to the source. State identity is
    # deliberately different, so action-stage generalization finds the node.
    query = dict(app='libreoffice-calc', window='new.xlsx - Calc', state_context='fresh-context',
        state_stage=action_stage(memory.distiller.distill_action(actions[1])),
        app_info={'name': 'Records23', 'data': '1 | Revenue |'},
        executed_prefix=[{'name': 'switch_active_sheet', 'arguments': {'sheet_name': 'Extract43'}},
                         {'name': 'switch_active_sheet', 'arguments': {'sheet_name': 'Records23'}}])
    sheet = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True)
    assert sheet.uses_summaries and sheet.current_situation
    assert sheet.state_experiences
    assert all(item.summary_match['verdict'] != 'conflict' for item in sheet.state_experiences)

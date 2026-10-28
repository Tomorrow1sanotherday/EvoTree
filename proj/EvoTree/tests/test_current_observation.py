"""Observation synchronization and model visibility regressions."""
import json

import pytest

from evotree.current_observation import (append_current_observation,
    capture_current_observation, current_observation_prompt, parse_structured,
    observation_record)


def test_archived_query_keeps_observed_retrieval_context_without_future_labels():
    prefix = [{'name': 'switch_active_sheet', 'arguments': {'sheet_name': 'Source'},
               'call_without_error': True}]
    obs = dict(app='calc', app_info={'name': 'Source'}, state_context='measured-hash',
               state_stage='observed-stage', executed_prefix=prefix, dialog=None,
               facts={'visible': True}, screenshot=b'frame', prediction='future action',
               score=1, retrieved_advice='future memory')
    record = observation_record(obs)
    assert record['executed_prefix'] == prefix
    assert record['state_stage'] == 'observed-stage'
    assert record['state_context'] == 'measured-hash'
    assert record['facts'] == {'visible': True} and record['dialog'] is None
    assert not {'prediction', 'score', 'retrieved_advice', 'screenshot'} & record.keys()
    assert record['screenshot_sha256']


def test_reads_bracket_frame_and_retry_only_observation():
    order = []
    states = iter([{'name': 'source'}, {'name': 'target'}, {'name': 'target'}, {'name': 'target'}])
    def read():
        value = next(states)
        order.append(value['name'])
        return value
    def frame():
        order.append('frame')
        return b'frame-bytes'
    obs = capture_current_observation(read, frame, action_sequence=2, source='env_info()', delay=0)
    assert order == ['source', 'frame', 'target', 'target', 'frame', 'target']
    assert obs['app_info'] == {'name': 'target'}
    meta = obs['observation_provenance']
    assert len(meta['samples']) == 2 and meta['status'] == 'stable'
    assert meta['action_sequence'] == 2
    assert meta['screenshot_render_freshness'] == 'unverified'
    for sample in meta['samples']:
        assert sample['started_ns'] <= sample['screenshot_started_ns'] <= sample['screenshot_finished_ns'] <= sample['finished_ns']


def test_changed_state_is_unknown_instead_of_claiming_requested_effect():
    states = iter([{'name': 'source'}, {'name': 'target'}] * 3)
    obs = capture_current_observation(lambda: next(states), lambda: b'png',
                                     action_sequence=7, source='env_info()', delay=0)
    assert obs['app_info'] is None
    meta = obs['observation_provenance']
    assert meta['status'] == 'unstable' and len(meta['samples']) == 3
    assert meta['last_before']['name'] == 'source'
    assert meta['last_after']['name'] == 'target'
    assert json.loads(current_observation_prompt(obs).split('\n')[2])['current_state'] is None


def test_unknown_app_does_not_claim_a_verified_structured_state():
    obs = capture_current_observation(None, lambda: b'png', action_sequence=0, source='unsupported')
    assert obs['app_info'] is None
    assert obs['observation_provenance']['status'] == 'unavailable'


def test_current_read_reaches_model_separately_from_previous_tool_result():
    obs = {'cur_app': 'libreoffice-calc', 'app_info': {'name': 'Extract43', 'data': '1 | |'},
           'exe_result': 'old read of Records23'}
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': 'Task instruction'}]}]
    append_current_observation(messages, obs)
    block = messages[-1]['content'][-1]['text']
    assert 'Extract43' in block and 'Records23' not in block
    assert 'previous tool response' in block
    assert obs['exe_result'] == 'old read of Records23'
    assert messages[-1]['content'][0]['text'] == 'Task instruction'


def test_parse_duplicate_stdout_requires_agreement():
    source = json.dumps({'name': 'Records23'})
    assert parse_structured(source + '\n' + source) == {'name': 'Records23'}
    assert parse_structured(source + '\n' + json.dumps({'name': 'Extract43'})) is None


def test_large_table_omits_data_but_keeps_measured_object_identity():
    prompt = current_observation_prompt({'app_info': {'name': 'Extract43', 'data': 'x'*20000}})
    payload = json.loads(prompt.split('\n')[2])
    assert payload['current_state'] == {'name': 'Extract43'}
    assert payload['omitted_fields'] == ['data']


def test_unstable_state_suppresses_local_advice_but_preserves_task_experience():
    from test_state_conditions import bank, observation, NEW
    memory, actions = bank()
    obs = observation(memory, actions[2])
    obs['observation_provenance'] = {'status': 'unstable'}
    sheet = memory.read_cheatsheet(NEW, obs)
    assert not sheet.state_experiences and sheet.task_experiences
    assert 'active_sheet_role' not in sheet.observed_conditions


def test_patch_generator_refuses_a_second_application():
    from scripts.patch_osworld_observations import patch_agent
    source = '        messages = user_prompt\n        # print(messages)'
    updated = patch_agent(source)
    assert 'append_current_observation(user_prompt, obs, use_dashscope)' in updated
    with pytest.raises(ValueError):
        patch_agent(updated)

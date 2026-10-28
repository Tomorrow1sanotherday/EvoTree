"""Read-only, bounded observation capture and explicit model-facing evidence.

This module records what was measured, not what an action should have achieved.
Matching structured reads bracket a screenshot; they do not certify that the UI
has repainted. No task roles, expected answers or tool actions are inferred here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import time
import uuid


def parse_structured(value):
    if isinstance(value, Mapping):
        return dict(value)
    if not isinstance(value, str):
        return None
    decoder, values, rest = json.JSONDecoder(), [], value.strip()
    try:
        while rest:
            item, end = decoder.raw_decode(rest)
            values.append(item)
            rest = rest[end:].strip()
    except ValueError:
        return None
    if values and isinstance(values[0], dict) and all(x == values[0] for x in values):
        return values[0]
    return None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode()).hexdigest()


def capture_current_observation(read_state, read_screenshot, *, action_sequence,
                                source, attempts=3, delay=.2):
    """Bracket each frame with two fresh state reads; never execute an action.

    Unsupported apps use read_state=None. An unstable/invalid read clears the
    active app_info while retaining the two conflicting samples for audit.
    Render freshness cannot be inferred from bytes or a stable state query.
    """
    if attempts < 1:
        raise ValueError('attempts must be positive')
    frame, first, second = None, None, None
    history = []
    for index in range(attempts if read_state is not None else 1):
        started = time.time_ns()
        first = parse_structured(read_state()) if read_state else None
        frame_started = time.time_ns()
        frame = read_screenshot()
        frame_finished = time.time_ns()
        second = parse_structured(read_state()) if read_state else None
        finished = time.time_ns()
        stable = first is not None and first == second
        history.append(dict(started_ns=started, screenshot_started_ns=frame_started,
            screenshot_finished_ns=frame_finished, finished_ns=finished,
            before_sha256=_digest(first), after_sha256=_digest(second), stable=stable))
        if stable or read_state is None:
            break
        if index + 1 < attempts:
            time.sleep(delay)
    status = ('stable' if history[-1]['stable'] else
              'unavailable' if first is None or second is None else 'unstable')
    # This is the measured snapshot, NOT a successful tool call's return value.
    info = second if status == 'stable' else None
    provenance = dict(capture_id=uuid.uuid4().hex, action_sequence=action_sequence,
        source=source, query_arguments={}, status=status, samples=history,
        screenshot_sha256=hashlib.sha256(frame).hexdigest(),
        screenshot_render_freshness='unverified',
        structured_observation_scope='current application object at query time')
    if status != 'stable':
        provenance['last_before'] = first
        provenance['last_after'] = second
    return dict(app_info=info, screenshot=frame, observation_provenance=provenance)


def observation_record(obs):
    """Persist only current evidence and acquisition metadata, not images."""
    record = {key: obs[key] for key in ('app', 'window', 'cur_app', 'cur_window_id', 'apps',
              'app_info', 'observation_provenance', 'state_context', 'state_stage',
              'dialog', 'facts', 'executed_prefix') if key in obs}
    screenshot = obs.get('screenshot')
    if isinstance(screenshot, bytes):
        record['screenshot_sha256'] = hashlib.sha256(screenshot).hexdigest()
    return record


def current_observation_prompt(obs, limit=12000):
    """The same block can be used by baseline or any memory arm."""
    if 'app_info' not in obs and 'observation_provenance' not in obs:
        return ''
    provenance = obs.get('observation_provenance') or {}
    info = parse_structured(obs.get('app_info'))
    if provenance.get('status') in {'unstable', 'unavailable'}:
        info = None
    payload = dict(app=obs.get('cur_app', obs.get('app')), current_state=info,
        source=provenance.get('source', 'automatic env_info() with no sheet argument'),
        capture_id=provenance.get('capture_id'),
        after_action_sequence=provenance.get('action_sequence'),
        structured_status=provenance.get('status', 'legacy capture; timing not recorded'),
        screenshot_render_freshness=provenance.get('screenshot_render_freshness', 'unverified'))
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if len(text) > limit:
        # Omit whole fields rather than cutting JSON. Names/identities survive
        # large table contents, while omitted data must not look like emptiness.
        compact = {}
        omitted = []
        for key, value in (info or {}).items():
            if len(json.dumps(value, ensure_ascii=False, default=str)) <= 1000:
                compact[key] = value
            else:
                omitted.append(key)
        payload['current_state'] = compact
        payload['omitted_fields'] = omitted
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(text) > limit:
            payload['current_state'] = None
            payload['omitted_fields'] = ['all; oversized observation']
            text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return ('\n[Current environment observation]\n' + text + '\n'
        'This is a fresh read of the application, separate from historical examples '
        'and the previous tool response. In Calc env_info(), name denotes the '
        'queried active worksheet because this automatic query has no sheet argument. '
        'Screenshot rendering may lag the structured read. If sources disagree, '
        'treat the active object as uncertain and obtain a new observation before '
        'an operation that depends on it. A successful call is not proof of its effect.\n')


def append_current_observation(messages, obs, use_dashscope=False):
    text = current_observation_prompt(obs)
    if text:
        block = {'text': text}
        if not use_dashscope:
            block['type'] = 'text'
        messages[-1]['content'].append(block)
    return messages

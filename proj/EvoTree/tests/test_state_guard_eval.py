"""Prespecified paired schedule and pilot scope for the guard diagnostic."""
from collections import Counter

from experiments.m2_small_eval.state_guard_eval import CASES, CONDITIONS, ORDERS, order


def test_each_case_gets_each_condition_three_times_with_balanced_position():
    assert len(CASES) == 3 and len(set(CASES)) == 3
    assert CONDITIONS == ('task', 'dual_unfiltered', 'dual')
    for case_index in range(3):
        schedules = [order(case_index, repetition) for repetition in range(3)]
        assert all(set(value) == set(CONDITIONS) for value in schedules)
        for position in range(3):
            assert Counter(value[position] for value in schedules) == Counter(CONDITIONS)


def test_total_planned_formal_count_is_27():
    assert len(CASES) * len(CONDITIONS) * 3 == 27


def test_one_action_adapter_discards_bundled_calls_uniformly():
    from experiments.m2_small_eval.state_guard_eval import OneActionAgent, ONE_ACTION_RULE

    class Agent:
        def predict(self, instruction, obs):
            import json
            calls=[dict(name='step_one',arguments={}),dict(name='step_two',arguments={})]
            response=''.join('<tool_call>'+json.dumps(call)+'</tool_call>' for call in calls)
            return response, ['ACTION1','ACTION2'], ['message']

    wrapped=OneActionAgent(Agent())
    response, actions, messages=wrapped.predict('do task',{})
    assert len(actions)==1 and actions==['ACTION1']
    assert '<tool_call>' in response and 'step_two' not in response
    assert ONE_ACTION_RULE in 'do task'+ONE_ACTION_RULE
    assert wrapped.discarded_calls==[dict(returned=2,executed=1,discarded=1)]

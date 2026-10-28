"""Recall must not confuse execution history with the current decision state."""
from types import SimpleNamespace

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.dual_retrieval import _bounded_contexts, action_stage, state_match
from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import SituationSummary
from evotree.types import StateSig


class EvidenceBackend:
    max_candidates = 24

    def __init__(self, unknown=False):
        self.unknown = unknown
        self.summarized = []

    def summarize(self, instruction, observation, executed_prefix=()):
        self.summarized.append(observation)
        return SituationSummary(observation['app_info']['status'])

    def compare(self, current, historical, goals=None):
        return [dict(verdict='unknown' if self.unknown else
                     'compatible' if current.sentence == old.sentence else 'conflict',
                     score=.9, reason='Observed progress comparison.') for old in historical]


def memory_and_query(backend):
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), strict_quality=False,
                           situation_backend=backend)
    instruction = 'Update the document.'
    actions = [dict(action=dict(action_type=name, parameters={}))
               for name in ('editor.edit', 'editor.save')]
    observations = [dict(app='editor', window='Document', state_context=f'history-{i}',
                         app_info=dict(status=status)) for i, status in enumerate(('pending', 'ready'))]
    memory.write_episode(Episode(instruction, tuple(
        EpisodeStep(i, obs, actions[i], observations[i+1] if i == 0 else None)
        for i, obs in enumerate(observations)), True))
    query = dict(app='editor', window='Document', state_context='new-instance',
                 app_info=dict(status='ready'), state_stage=action_stage(
                     memory.distiller.distill_action(dict(action=dict(action_type='editor.inspect', parameters={})))))
    return memory, instruction, query


def test_semantic_recall_reaches_already_satisfied_state_with_different_previous_action():
    memory, instruction, query = memory_and_query(EvidenceBackend())
    assert not memory.read_dual_cheatsheet(instruction, query,
        use_situation_summaries=True, relax_action_stage=False).state_experiences
    result = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True)
    assert len(result.recall_candidates) == 2
    assert [item.recommendation.action.name for item in result.state_experiences] == ['editor.save']
    # The existing nonsemantic branch still uses the old hard gate.
    assert not memory.read_dual_cheatsheet(instruction, query).state_experiences


@pytest.mark.parametrize('change', [dict(app='other'), dict(window='Other'),
                                    dict(dialog='Confirmation'), dict(facts={'blocked': True})])
def test_relaxed_recall_preserves_environment_and_explicit_fact_conflicts(change):
    historical = StateSig(app='editor', window='Document', context='old', facts={'blocked': False})
    current = StateSig(**(dict(app='editor', window='Document', context='new') | change))
    assert state_match(current, historical, 'inspect', {'edit'}, relax_action_stage=True) is None


def test_relaxed_recall_does_not_match_unknown_app_or_bare_application():
    for app in ('unknown', 'osworld-task-start', 'editor'):
        assert state_match(StateSig(app=app, window=''), StateSig(app=app, window=''),
                           'inspect', {'edit'}, relax_action_stage=True) is None
    assert state_match(StateSig(app='editor', window='', context='opaque-only'),
                       StateSig(app='editor', window='', context='old-opaque-only'),
                       'inspect', {'edit'}, relax_action_stage=True) is None


def test_unknown_is_logged_but_not_injected_as_verified_local_advice():
    memory, instruction, query = memory_and_query(EvidenceBackend(unknown=True))
    result = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True)
    assert not result.state_experiences
    assert result.task_experiences  # Independent task memory survives abstention.
    assert all(row['verdict'] == 'unknown' for row in result.summary_comparisons)
    assert len(memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True,
        require_summary_compatible=False, require_compatible_all=False).state_experiences) == 2


def test_budget_is_applied_before_historical_summary_generation():
    backend = EvidenceBackend()
    backend.max_candidates = 1
    memory, instruction, query = memory_and_query(backend)
    memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True)
    assert len(backend.summarized) == 2  # One current + one historical.


def test_unknown_from_newly_recalled_state_is_not_promoted_without_compatibility():
    backend = EvidenceBackend(unknown=True)
    memory, instruction, query = memory_and_query(backend)
    result = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True)
    assert not result.state_experiences
    assert all(row['reason'] == 'summary-context-unknown' for row in result.rejected_states)


def test_require_compatible_all_also_filters_legacy_stage_unknowns():
    backend = EvidenceBackend(unknown=True)
    memory, instruction, query = memory_and_query(backend)
    query['state_context'] = 'history-1'
    relaxed = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True,
                                          require_compatible_all=False)
    strict = memory.read_dual_cheatsheet(instruction, query, use_situation_summaries=True,
                                         require_compatible_all=True)
    assert relaxed.state_experiences
    assert not strict.state_experiences
    assert all(x['reason'] == 'summary-context-unknown' for x in strict.rejected_states)


def test_bounded_pool_visits_other_sources_and_nodes_before_repeated_contexts():
    matches, contexts = [], {}
    for source, node, count in [('a', 'a1', 30), ('a', 'a2', 2), ('b', 'b1', 2)]:
        sig = StateSig(app='editor', window='Document', context=node)
        action = SimpleNamespace(key=lambda: 'edit')
        item = SimpleNamespace(template_id=source, node_id=node, state=sig,
                               recommendation=SimpleNamespace(action=action))
        matches.append(item)
        contexts[(source, sig.key(), 'edit')] = [(source, i) for i in range(count)]
    records = _bounded_contexts(matches, contexts, 3)
    assert [item.node_id for item, _, _ in records] == ['a1', 'b1', 'a2']

import json

import pytest

from evotree.distill_osworld import OSWorldDistiller
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import SituationResponseError
from evotree.task_relevance import TaskRelevanceBackend
from test_dual_retrieval import COPY, digest, seed
from evotree.episode import Episode, EpisodeStep


QUERY = 'Copy the Revenue column including its header into a separate summary worksheet.'


def backend_with(verdict, captured=None):
    def complete(messages):
        if captured is not None:
            captured.append(messages)
        payload = json.loads(messages[1]['content'])
        return json.dumps({'matches': [dict(id=row['id'], verdict=verdict,
            score=.95, reason='Shared object role and copying workflow.')
            for row in payload['historical']]})
    return TaskRelevanceBackend(complete)


def test_related_workflow_fallback_preserves_frozen_templates_and_state_abstention(tmp_path):
    memory = EvoTreeMemory(tmp_path, OSWorldDistiller(), strict_quality=False)
    seed(memory)
    memory.save()
    before = digest(tmp_path)
    library_before = json.dumps(memory.distiller.template_library.to_dict(), sort_keys=True)
    obs = dict(app='unknown', window='')
    assert not memory.read_dual_cheatsheet(QUERY, obs).task_experiences
    captured = []
    sheet = memory.read_dual_cheatsheet(QUERY, obs,
        task_relevance_backend=backend_with('related', captured))
    assert len(sheet.task_experiences) == 1
    assert not sheet.state_experiences  # Task relevance cannot license an action.
    text = memory.render(sheet)
    assert 'Related workflow only' in text and 'create_sheet' in text
    payload = json.loads(captured[0][1]['content'])
    assert set(payload) == {'current_goal', 'historical'}
    assert set(payload['historical'][0]) == {'id', 'task_goal', 'operations'}
    assert 'Sheet1' not in captured[0][1]['content']  # No historical arguments.
    assert digest(tmp_path) == before
    assert json.dumps(memory.distiller.template_library.to_dict(), sort_keys=True) == library_before


@pytest.mark.parametrize('verdict', ['unrelated', 'unknown'])
def test_negative_or_unknown_workflow_is_not_injected(verdict):
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    seed(memory)
    sheet = memory.read_dual_cheatsheet(QUERY, {}, state_enabled=False,
        task_relevance_backend=backend_with(verdict))
    assert not sheet.task_experiences and not memory.render(sheet)


def test_existing_template_does_not_call_semantic_fallback():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    seed(memory)
    def forbidden(_):
        raise AssertionError('Existing template must not require another request')
    sheet = memory.read_dual_cheatsheet(COPY, {}, state_enabled=False,
        task_relevance_backend=TaskRelevanceBackend(forbidden))
    assert sheet.task_experiences and not sheet.task_comparisons


def test_configured_backend_is_used_by_normal_memory_read_route():
    captured = []
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), retrieval_mode='task',
                          task_relevance_backend=backend_with('related', captured))
    seed(memory)
    text = memory.read(QUERY, {})
    assert 'Related workflow only' in text and 'Horizontal state experience' not in text
    assert len(captured) == 1
    # The reference is cached by query and historical goal/operations, not by
    # instance-specific observations; task retrieval remains state independent.
    assert memory.read(QUERY, {'app_info': {'different': 'observation'}}) == text
    assert len(captured) == 1


def test_invalid_relevance_output_abstains_after_bounded_repair():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    seed(memory)
    backend = TaskRelevanceBackend(lambda _: '{"matches":[{"id":999}]}')
    sheet = memory.read_dual_cheatsheet(QUERY, {}, state_enabled=False,
        task_relevance_backend=backend)
    assert not sheet.task_experiences and backend.model_calls == 2
    assert sheet.task_relevance_failures == [{'error_type': 'SituationResponseError'}]


def test_oversized_library_does_not_silently_select_first_templates():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    seed(memory)
    seed(memory, 'Set page orientation to landscape.')
    backend = backend_with('related')
    backend.max_candidates = 1
    sheet = memory.read_dual_cheatsheet(QUERY, {}, state_enabled=False, task_relevance_backend=backend)
    assert not sheet.task_experiences and backend.model_calls == 0
    assert sheet.task_relevance_failures[0]['reason'] == 'candidate-budget-exceeded'


def test_related_color_procedure_preserves_order_without_old_bindings_or_loop_count():
    from test_dual_retrieval import raw
    memory = EvoTreeMemory(distiller=OSWorldDistiller(), retrieval_mode='task',
                          task_relevance_backend=backend_with('related'))
    old_goal = 'Set three old-slide text boxes to red, green and blue.'
    state = {'app':'impress', 'window':'old.pptx — Impress'}
    calls = [raw('libreoffice_impress.env_info', page_indices=[1])]
    calls += [raw('libreoffice_impress.set_text_color',slide_index=1,box_index=index,color=color)
              for index,color in [(3,'red'),(7,'green'),(8,'blue')]]
    calls += [raw('libreoffice_impress.save')]
    memory.write_episode(Episode(old_goal,tuple(EpisodeStep(i,state,call) for i,call in enumerate(calls)),True))
    text = memory.read('Set the two text boxes on slide 5 to yellow.', {})
    assert text.index('env_info') < text.index('set_text_color') < text.index('libreoffice_impress.save')
    assert text.count('libreoffice_impress.set_text_color') == 1
    assert 'Source steps 2-4:' in text
    assert 'derive targets and repetition count from the CURRENT task' in text
    assert '<CURRENT:box_index>' in text and '<CURRENT:color>' in text
    assert '<CURRENT:slide_index>' in text
    assert old_goal not in text
    assert not any(f'"{key}": "{value}"' in text for key,value in [('color','red'),('color','green'),('color','blue'),('slide_index','1')])
    assert 'observed before: app=impress' in text
    assert 'not current target counts' in text and 'verify effects' in text

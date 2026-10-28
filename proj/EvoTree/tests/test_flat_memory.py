import json
from experiments.m2_small_eval.flat_memory import FlatMemory, OnlineFlatMemory
from evotree.episode import Episode, EpisodeStep


def test_flat_memory_uses_only_independent_text_and_current_task():
    captured=[]
    docs=[dict(source_id='source',task='Copy a column.',title='Copy reference',experience='Locate source and destination, then copy and verify.')]
    def complete(messages):
        captured.append(messages)
        return json.dumps({'matches':[dict(id=0,verdict='related',score=.9,reason='same workflow')]})
    memory=FlatMemory(docs,complete)
    sheet=memory.read_cheatsheet('Copy another column.',{'tree':'do not read','gold':'hidden'},['copy','save'])
    assert sheet.reranked_tools==['copy','save']
    text=memory.render(sheet)
    assert 'Copy reference' in text and sheet.rendered_task_records==1
    payload=json.loads(captured[0][1]['content'])
    assert set(payload)=={'current_task','historical'}
    assert 'gold' not in str(payload) and 'tree' not in str(payload)
    memory.read_cheatsheet('Copy another column.',{'different':'state'})
    assert len(captured)==1
    assert memory.root_dir is None and not memory.write_enabled


def test_unrelated_flat_memory_abstains_instead_of_forcing_injection():
    memory=FlatMemory([dict(source_id='a',task='copy',title='copy',experience='copy')],
        lambda _:json.dumps({'matches':[dict(id=0,verdict='unrelated',score=1,reason='different task')]}))
    sheet=memory.read_cheatsheet('Send a message.',{})
    assert memory.render(sheet)=='' and not sheet.task_experiences


def test_flat_memory_schema_failure_is_recorded_without_invented_advice():
    memory=FlatMemory([],lambda _:'{}')
    sheet=memory.read_cheatsheet('task',{})
    assert not memory.render(sheet) and sheet.task_relevance_failures


def test_flat_memory_uses_runtime_hook_without_needing_a_tree(caplog):
    import logging
    from evotree.integration import EvoTreeRuntime
    memory=FlatMemory([dict(source_id='a',task='copy',title='copy',experience='copy and verify')],
        lambda _:json.dumps({'matches':[dict(id=0,verdict='related',score=.9,reason='same procedure')]}))
    runtime=EvoTreeRuntime(memory);runtime.begin('copy another column')
    with caplog.at_level(logging.INFO,logger='evotree.integration'):
        value=runtime.before_model({'app':'calc'},['copy'])
    assert 'Flat task memory' in value.text
    assert not hasattr(memory,'forest')
    runtime.finish(score=1)
    assert not runtime.active


def test_online_flat_memory_appends_only_a_bounded_note_after_episode():
    calls = []
    def complete(messages):
        calls.append(messages)
        return json.dumps({'title': 'Observed copy', 'experience': 'Copy the selected range and verify the destination.'})
    memory = OnlineFlatMemory([], lambda _: json.dumps({'matches': []}), complete)
    episode = Episode('Copy a range.', (EpisodeStep(
        0, {'app': 'calc'}, {'tool_call': {'name': 'copy'}},
        next_observation={'app': 'calc'}, metadata={'env_info': {'ok': True}}),),
        success=True, metadata={'score': 1.0})
    memory.write_episode(episode)
    assert len(memory.documents) == 1
    assert memory.documents[0]['experience'].startswith('Copy')
    assert len(calls) == 1

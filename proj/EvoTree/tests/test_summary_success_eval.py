import json
from experiments.m2_small_eval.summary_success_eval import StructuredSummaryBackend


def test_success_summary_scope_matches_structured_only_retrieval_audit():
    captured=[]
    def complete(messages):
        captured.extend(messages)
        return json.dumps(dict(sentence='An object is observed.',evidence=[],unknowns=[]))
    backend=StructuredSummaryBackend(complete)
    backend.summarize('Change object.',{'app':'calc','app_info':{'active_sheet':'Data'},
        'screenshot':b'not-in-summary','exe_result':'old tool result'})
    content=captured[-1]['content']
    assert [b['type'] for b in content]==['text']
    payload=json.loads(content[0]['text'])
    assert payload['observation']['app_info']=={'active_sheet':'Data'}
    assert 'exe_result' not in payload['observation']

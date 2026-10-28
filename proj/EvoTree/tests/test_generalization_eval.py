"""Experimental schedule, cost summary, and executed-history consistency."""
from collections import Counter
import json
from types import SimpleNamespace

from experiments.m2_small_eval.generalization_tasks import specifications
from experiments.m2_small_eval.generalization_eval import ExecutedHistoryAgent, RequestAudit, schedule
from experiments.m2_small_eval.generalization_launch import summarize


def test_all_24_cases_once_in_balanced_four_worker_pair_schedule():
    calc,font=specifications()
    cases=[dict(id=v['id'],family=family,level=v['level']) for family,items in
           [('calc',calc),('font',font)] for v in items]
    jobs=schedule(cases)
    assert jobs==schedule(cases) and all(len(worker)==6 for worker in jobs)
    assert Counter(j['case'] for worker in jobs for j in worker)==Counter(c['id'] for c in cases)
    mapping={c['id']:c for c in cases}
    for family in ('calc','font'):
        for level in ('parameters','layout','context'):
            first=Counter(j['order'][0] for worker in jobs for j in worker
                if mapping[j['case']]['family']==family and mapping[j['case']]['level']==level)
            assert first=={'task':2,'dual_summary':2}


def test_single_action_updates_underlying_owl_history():
    class Agent:
        def __init__(self):
            self.history_responses=[];self.thoughts=[];self.actions=[];self.history_conclusion=[]
        def predict(self,*args,**kwargs):
            response=''.join('<tool_call>'+json.dumps(dict(name=n,arguments={}))+'</tool_call>' for n in ('one','two'))
            self.history_responses.append(response);self.thoughts.append(response)
            self.actions.append(['A','B']);self.history_conclusion.append('one then two')
            return response,['A','B'],[]
    agent=Agent()
    response,actions,_=ExecutedHistoryAgent(agent).predict('task',{})
    assert actions==['A'] and agent.actions==[['A']]
    assert agent.history_responses==agent.thoughts==[response]
    assert '"name": "two"' not in agent.history_conclusion[-1]


def test_continuation_gate_uses_paired_wins_and_both_families():
    rows=[]
    for i in range(24):
        for arm in ('task','dual_summary'):
            score=int(arm=='dual_summary' or i>=4)
            rows.append(dict(case=str(i),family='calc' if i%2 else 'font',level='parameters',
                condition=arm,score=score,decisions=1,agent_calls=1,summary_calls=0,cached_summary_calls=0,seconds=1.))
    result=summarize(rows)
    assert result['paired']['all']['net_wins']==4 and result['continuation_gate_passed']


def test_request_audit_records_bounded_response_shape_without_dumping_request(tmp_path):
    class Client:
        def create(self, **kwargs):
            message = SimpleNamespace(
                content='<tool_call>bad</tool_call>' + 'x'*1000,
                tool_calls=[SimpleNamespace(function=SimpleNamespace(
                    name='libreoffice_impress.set_slide_background'))])
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=message)])
    path=tmp_path/'requests.jsonl'
    client=RequestAudit(Client(),path)
    client.create(messages=[{'content':'private prompt'}])
    row=json.loads(path.read_text().splitlines()[0])
    assert row['response_shape']['choices']==1
    assert row['response_shape']['tool_call_names']==['libreoffice_impress.set_slide_background']
    assert len(row['content_preview'])==600
    assert 'private prompt' not in path.read_text()


def test_request_audit_keeps_late_and_unclosed_action_blocks(tmp_path):
    late_action = '{"name":"computer_use","arguments":{"action":"click"}}'
    class Client:
        def create(self, **kwargs):
            content = 'reasoning ' * 200 + '<tool_call>' + late_action + '</tool_call>'
            content += '<tool_call>{"name":"unfinished"'
            message = SimpleNamespace(content=content, tool_calls=None)
            return SimpleNamespace(usage=None, choices=[SimpleNamespace(
                message=message, finish_reason='length')])
    path = tmp_path/'requests.jsonl'
    RequestAudit(Client(), path).create(messages=[])
    row = json.loads(path.read_text())
    assert late_action not in row['content_preview']
    assert row['protocol_blocks'][0] == dict(body=late_action, closed=True, truncated=False)
    assert row['protocol_blocks'][1]['closed'] is False
    assert row['finish_reason'] == 'length'

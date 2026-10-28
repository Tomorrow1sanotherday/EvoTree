"""Small official-MCP paired evaluation: independent flat notes vs full EvoTree."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace


def write(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)


def append(path,value):
    with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')


def file_hashes(root):
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def aggregate(attempts, task_ids):
    """Missing scores never become zeros; only complete pairs are compared."""
    ids=set(task_ids);by_task={tid:{} for tid in task_ids}
    for row in attempts:
        if row['task_id'] not in ids:raise ValueError('Unexpected task')
        arm=row['condition']
        if arm not in {'ordinary','evotree'} or arm in by_task[row['task_id']]:
            raise ValueError('Invalid or duplicate task/arm attempt')
        by_task[row['task_id']][arm]=row
    pairs=[]
    for tid,rows in by_task.items():
        scores={arm:rows.get(arm,{}).get('score') for arm in ['ordinary','evotree']}
        complete=all(score is not None for score in scores.values())
        verdict=None
        if complete:
            a,b=scores['ordinary'],scores['evotree']
            verdict='win' if b>a else 'loss' if b<a else 'tie'
        pairs.append(dict(task_id=tid,scores=scores,complete=complete,evotree_verdict=verdict))
    arms={}
    for arm in ['ordinary','evotree']:
        rows=[r for r in attempts if r['condition']==arm]
        scored=[r for r in rows if r.get('score') is not None]
        arms[arm]=dict(attempts=len(rows),scored=len(scored),successes=sum(r['score']==1 for r in scored),
            missing=len(rows)-len(scored),agent_calls=sum(r.get('agent_calls',0) for r in rows),
            decisions=sum(r.get('decisions',0) for r in rows),
            memory_calls=sum(r.get('memory_calls',0) for r in rows),seconds=sum(r.get('seconds',0) for r in rows),
            injected_task_decisions=sum(r.get('task_injected_decisions',0) for r in rows),
            injected_state_decisions=sum(r.get('state_injected_decisions',0) for r in rows))
    return dict(attempts=len(attempts),planned=len(task_ids)*2,arms=arms,pairs=pairs,
        paired=dict(complete=sum(p['complete'] for p in pairs),
                    wins=sum(p['evotree_verdict']=='win' for p in pairs),
                    losses=sum(p['evotree_verdict']=='loss' for p in pairs),
                    ties=sum(p['evotree_verdict']=='tie' for p in pairs)),
        interpretation='Small fixed skill-related Office pilot, not an all-domain benchmark estimate or significance claim.')


class AuditedMemoryTransport:
    def __init__(self, transport, root):
        self.transport,self.root,self.calls,self.failures=transport,root,0,0
        root.mkdir(parents=True,exist_ok=False)

    def __call__(self,messages):
        self.calls+=1;start=time.monotonic();record={'request':messages,'call':self.calls}
        try:
            value=self.transport(messages);record['response']=value;return value
        except Exception as exc:
            self.failures+=1;record['error_type']=type(exc).__name__;raise
        finally:
            record['seconds']=time.monotonic()-start
            write(self.root/f'{self.calls:04d}.json',record)


def worker(args):
    rt,out=args.runtime_root,args.output_root
    sys.path[:0]=[str(rt/'code'),str(rt/'runner'),'/data/songweishuai/envs/osworld/lib/python3.12/site-packages']
    os.environ.update(json.loads((rt/'runtime-env.json').read_text()))
    from experiments.m2_small_eval.run_arm import runtime_environment
    os.environ.update(runtime_environment(os.environ,f'evo-independent-fast-20261027-w{args.worker}',12+args.worker))
    os.environ.update(OSWORLD_MODEL_REQUEST_LOCK=str(rt/'agent-requests.lock'),OSWORLD_MODEL_REQUEST_LOCK_MODE='soft',EVOTREE_RETRIEVAL_LOG='0')
    os.chdir(rt/'runner')
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.generalization_eval import ExecutedHistoryAgent,RequestAudit,make_memory
    from experiments.m2_small_eval.diagnostic_pilot import run_episode
    from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
    from experiments.m2_small_eval.summary_success_eval import StructuredSummaryBackend
    from experiments.m2_small_eval.flat_memory import FlatMemory
    from evotree.task_relevance import TaskRelevanceBackend
    from evotree.situation_summary import JsonChatTransport
    from evotree.osworld_integration import wrap_osworld
    from evotree.wire_audit import WireAudit
    spec=importlib.util.spec_from_file_location('asset_eval',rt/'code/experiments/m2_small_eval/official_mcp_eval.py')
    official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    plan=json.loads((out/'plan.json').read_text());job=plan['schedule'][args.worker]
    task=next(t for t in json.loads((out/'tasks-private.json').read_text()) if t['id']==job['task_id'])
    documents=json.loads((out/'ordinary-memory.json').read_text())
    assert file_hashes(rt/'bank')==plan['bank_hashes']
    assert hashlib.sha256((out/'ordinary-memory.json').read_bytes()).hexdigest()==plan['ordinary_memory_sha256']
    folder=out/f'worker-{args.worker}';folder.mkdir(exist_ok=False)
    state={'state':'booting','completed':0,'planned':2,'task':task['id']}
    write(folder/'status.json',state);desktop=None
    def interrupted(signum,frame):raise InterruptedError('Worker interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        desktop=DesktopEnv(action_space='mcp',headless=True,provider_name='docker',os_type='Ubuntu',
            cache_dir=str(rt/'cache'),screen_size=(1920,1080),require_a11y_tree=False,shuffle=False,rag=True)
        evaluate=desktop.evaluate
        desktop.evaluate=lambda:evaluate_with_save_barrier(desktop,evaluate,log=lambda event:append(folder/'save-barrier.jsonl',dict(task=task['id'],condition=state.get('condition'),**event)))
        for arm in job['order']:
            directory=folder/task['id']/arm;directory.parent.mkdir(exist_ok=True)
            state.update(state='running',condition=arm);write(folder/'status.json',state)
            start=time.monotonic();audit=None;transports=[];result=None
            try:
                official.clear_result_cache(task['config'],rt/'cache'/task['id'])
                agent=OwlAgent(model=plan['model'],api_url=plan['api_url'],api_key=os.environ['OSWORLD_API_KEY'],
                    history_n=3,action_space='mcp',observation_type='screenshot',max_trajectory_length=15,temperature=1.,top_p=.9)
                wire=WireAudit(directory.with_name(arm+'-wire'),excluded_secrets=[os.environ['OSWORLD_API_KEY']])
                agent.llm_client._client.event_hooks['request'].append(wire.on_request)
                agent.llm_client._client.event_hooks['response'].append(wire.on_response)
                audit=RequestAudit(agent.llm_client.chat.completions,folder/f"{task['id']}-{arm}-agent-requests.jsonl",action_protocol_reminder=True)
                agent.llm_client=SimpleNamespace(chat=SimpleNamespace(completions=audit))
                protocol=ExecutedHistoryAgent(agent)
                def transport(label):
                    obj=AuditedMemoryTransport(JsonChatTransport(plan['api_url'],plan['model'],os.environ['OSWORLD_API_KEY'],timeout=180),
                        directory.with_name(arm+'-'+label+'-requests'))
                    transports.append(obj);return obj
                if arm=='ordinary':
                    memory=FlatMemory(documents,transport('task-retrieval'),budget=6000)
                else:
                    memory=make_memory(rt/'bank','dual_summary',StructuredSummaryBackend(transport('state')))
                    memory.task_relevance_backend=TaskRelevanceBackend(transport('task-retrieval'))
                wrapped,env,runtime=wrap_osworld(protocol,desktop,memory)
                result=run_episode(wrapped,env,task['config'],directory,max_steps=15,runtime=runtime)
                preserved=official.preserve_result_artifacts(task['config'],desktop.cache_dir,directory,score=result['score'],action_history=desktop.action_history)
                row=dict(task_id=task['id'],condition=arm,outcome='scored',score=result['score'],decisions=result['decisions'],artifacts=preserved['saved_artifacts'])
            except Exception as exc:
                directory.mkdir(exist_ok=True)
                (directory/'exception.txt').write_text(traceback.format_exc())
                row=dict(task_id=task['id'],condition=arm,outcome='interrupted',score=None,
                         error_type=type(exc).__name__,error=str(exc),decisions=0)
                traceback.print_exc()
            decision_file=directory/'decisions.jsonl'
            decisions=[json.loads(l) for l in decision_file.read_text().splitlines() if l.strip()] if decision_file.exists() else []
            row.update(seconds=time.monotonic()-start,agent_calls=audit.calls if audit else 0,
                memory_calls=sum(t.calls for t in transports),memory_transport_errors=sum(t.failures for t in transports),
                decisions=len(decisions),task_injected_decisions=sum(r.get('rendered_task_records',0)>0 for r in decisions),
                state_injected_decisions=sum(r.get('rendered_state_records',0)>0 for r in decisions),
                task_sources=sorted({s for r in decisions for s in r.get('task_sources',[])}),
                summary_comparisons=sum(len(r.get('summary_comparisons',[])) for r in decisions),
                memory_chars=[r.get('chars',0) for r in decisions])
            assert file_hashes(rt/'bank')==plan['bank_hashes']
            assert hashlib.sha256((out/'ordinary-memory.json').read_bytes()).hexdigest()==plan['ordinary_memory_sha256']
            append(folder/'attempts.jsonl',row)
            if row['score'] is not None:append(folder/'results.jsonl',row)
            state['completed']+=1;write(folder/'status.json',state)
        state['state']='complete'
    except BaseException as exc:
        state.update(state='failed',error_type=type(exc).__name__,error=str(exc))
        (folder/'exception.txt').write_text(traceback.format_exc());traceback.print_exc()
    finally:
        if desktop is not None:
            pids=list(desktop.provider._owned_qemu);desktop.close()
            state.update(closed_qemu_pids=pids,remaining_owned_qemu=[p for p in pids if Path('/proc',str(p)).exists()])
        state['bank_unchanged']=file_hashes(rt/'bank')==plan['bank_hashes']
        write(folder/'status.json',state)
    return 0 if state['state']=='complete' else 1


def supervise(args):
    rt,out=args.runtime_root,args.output_root
    plan=json.loads((out/'plan.json').read_text());jobs=[]
    os.environ.update(json.loads((rt/'runtime-env.json').read_text()))
    for index in range(4):
        with (out/f'worker-{index}.log').open('wb') as log:
            p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--runtime-root',str(rt),'--output-root',str(out),
                '--worker',str(index)],cwd=rt/f'work-{index}',env=os.environ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(p)
    write(out/'workers.json',[dict(worker=i,pid=p.pid) for i,p in enumerate(jobs)])
    while any(p.poll() is None for p in jobs):
        attempts=[json.loads(l) for f in out.glob('worker-*/attempts.jsonl') for l in f.read_text().splitlines() if l.strip()]
        states=[]
        for i,p in enumerate(jobs):
            file=out/f'worker-{i}/status.json'
            states.append(dict(worker=i,pid=p.pid,returncode=p.poll(),status=json.loads(file.read_text()) if file.exists() else {'state':'starting'}))
        write(out/'status.json',dict(state='running',completed=len(attempts),planned=8,scored=sum(r.get('score') is not None for r in attempts),workers=states))
        write(out/'partial-summary.json',aggregate(attempts,[j['task_id'] for j in plan['schedule']]))
        time.sleep(10)
    attempts=[json.loads(l) for f in out.glob('worker-*/attempts.jsonl') for l in f.read_text().splitlines() if l.strip()]
    summary=aggregate(attempts,[j['task_id'] for j in plan['schedule']])
    states=[json.loads((out/f'worker-{i}/status.json').read_text()) for i in range(4)]
    pids=[pid for s in states for pid in s.get('closed_qemu_pids',[])]
    summary.update(bank_unchanged=file_hashes(rt/'bank')==plan['bank_hashes'],
        ordinary_memory_unchanged=hashlib.sha256((out/'ordinary-memory.json').read_bytes()).hexdigest()==plan['ordinary_memory_sha256'],
        remaining_owned_qemu=[p for p in pids if Path('/proc',str(p)).exists()],worker_returncodes=[p.returncode for p in jobs],attempt_rows=attempts)
    write(out/'summary.json',summary)
    state='complete' if len(attempts)==8 and all(p.returncode==0 for p in jobs) and not summary['remaining_owned_qemu'] else 'incomplete'
    write(out/'status.json',dict(state=state,completed=len(attempts),planned=8,scored=sum(r.get('score') is not None for r in attempts)))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--runtime-root',type=Path,required=True);parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--worker',type=int,choices=range(4));parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    if args.supervise:supervise(args)
    elif args.worker is not None:raise SystemExit(worker(args))

if __name__=='__main__':main()

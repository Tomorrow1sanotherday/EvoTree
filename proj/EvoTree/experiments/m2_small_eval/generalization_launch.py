"""Run two fixture smokes, then four paired-evaluation workers, and audit."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from experiments.m2_small_eval.diagnostic_pilot import hashes, write_json


def summarize(rows):
    matrix = {}
    for row in rows:
        pair = matrix.setdefault(row['case'], dict(family=row['family'], level=row['level'], scores={}))
        if row['condition'] in pair['scores']:
            raise ValueError('Duplicate scored episode')
        pair['scores'][row['condition']] = row['score']
    assert len(matrix)==24 and all(set(v['scores'])=={'task','dual_summary'} for v in matrix.values())
    per = {}
    for family in ('calc','font','all'):
        pairs = [v['scores'] for v in matrix.values() if family=='all' or v['family']==family]
        win = sum(v['dual_summary']>=1 and v['task']<1 for v in pairs)
        loss = sum(v['task']>=1 and v['dual_summary']<1 for v in pairs)
        n = win+loss
        p = min(1.,2*sum(math.comb(n,k) for k in range(min(win,loss)+1))/2**n) if n else 1.
        per[family] = dict(n=len(pairs),task_successes=sum(v['task']>=1 for v in pairs),
            summary_successes=sum(v['dual_summary']>=1 for v in pairs),wins=win,losses=loss,
            ties=len(pairs)-win-loss,net_wins=win-loss,exact_two_sided_mcnemar_p=p)
    costs = {arm:{key:sum(r[key] for r in rows if r['condition']==arm)
                  for key in ('decisions','agent_calls','summary_calls','cached_summary_calls','seconds')}
             for arm in ('task','dual_summary')}
    return dict(episodes=len(rows),cases=matrix,paired=per,costs=costs,
        continuation_gate_passed=per['all']['net_wins']>=4 and all(per[f]['net_wins']>=0 for f in ('calc','font')),
        interpretation='Exploratory same-template transfer; continuation gate is not a significance claim.')


def supervise(rt, out):
    plan = json.loads((out/'plan.json').read_text())
    assert hashes(rt/'bank') == plan['bank_hashes']
    assert hashes(rt/'code') == plan['code_hashes']
    jobs = []
    try:
        for phase,count in (('smoke',2),('formal',4)):
            jobs = []
            for worker in range(count):
                with (rt/f'{phase}-{worker}.log').open('w') as log:
                    process = subprocess.Popen([sys.executable,'-m','experiments.m2_small_eval.generalization_eval',
                        '--runtime-root',str(rt),'--output-root',str(out),'--phase',phase,'--worker',str(worker)],
                        cwd=rt/f'work-{worker}',env=os.environ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                jobs.append((worker,process))
            write_json(out/f'{phase}-workers.json',[dict(worker=i,pid=p.pid) for i,p in jobs])
            while any(p.poll() is None for _,p in jobs):
                states,completed = [],0
                for i,p in jobs:
                    folder = out/f'{phase}-worker-{i}'
                    status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else dict(state='starting')
                    if (folder/'results.jsonl').exists():
                        completed += len((folder/'results.jsonl').read_text().splitlines())
                    states.append(dict(worker=i,pid=p.pid,returncode=p.poll(),status=status))
                write_json(out/'status.json',dict(state=phase,completed=completed,planned=2 if phase=='smoke' else 48,workers=states))
                assert hashes(rt/'bank')==plan['bank_hashes']
                time.sleep(10)
            assert all(p.returncode==0 for _,p in jobs),f'{phase} worker failed; raw attempt preserved'
            write_json(out/f'{phase}-complete.json',dict(passed=True,worker_returncodes=[p.returncode for _,p in jobs]))
        rows,pids = [],[]
        for i in range(4):
            folder = out/f'formal-worker-{i}'
            status = json.loads((folder/'status.json').read_text())
            assert status['state']=='complete' and status['completed']==12
            pids += status['closed_qemu_pids']
            for row in map(json.loads,(folder/'results.jsonl').read_text().splitlines()):
                assert float((folder/row['case']/row['condition']/'result.txt').read_text())==row['score']
                rows.append(row)
        assert len(rows)==48
        assert not [pid for pid in pids if Path('/proc',str(pid)).exists()]
        assert hashes(rt/'bank')==plan['bank_hashes'] and hashes(rt/'code')==plan['code_hashes']
        result = summarize(rows)
        result.update(bank_unchanged=True,code_unchanged=True,remaining_owned_qemu=[])
        write_json(out/'summary.json',result)
        write_json(out/'status.json',dict(state='complete',completed=48,planned=48))
    except Exception as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        write_json(out/'status.json',dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        for _,p in jobs:
            if p.poll() is None:
                p.wait()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--supervise',action='store_true')
    args=p.parse_args()
    if args.supervise:
        supervise(args.runtime_root,args.output_root)
        return
    assert not (args.output_root/'launch.json').exists()
    key=sys.stdin.readline().strip()
    assert key
    from evotree.situation_summary import JsonChatTransport
    plan=json.loads((args.output_root/'plan.json').read_text())
    auth_started=time.monotonic()
    # A tiny independent preflight query; no test case or historical action.
    reply=JsonChatTransport(plan['api_url'],plan['model'],key,timeout=90)([
        dict(role='system',content='Return JSON only.'),
        dict(role='user',content='Return {"ready": true}.')])
    assert json.loads(reply).get('ready') is True
    write_json(args.output_root/'model-preflight.json',dict(passed=True,model_calls=1,
        seconds=time.monotonic()-auth_started,credentials_persisted=False))
    env=dict(os.environ)
    env.update(json.loads((args.runtime_root/'runtime-env.json').read_text()))
    env['OSWORLD_API_KEY']=key
    with (args.runtime_root/'supervisor.log').open('w') as log:
        process=subprocess.Popen([sys.executable,'-m','experiments.m2_small_eval.generalization_launch',
            '--runtime-root',str(args.runtime_root),'--output-root',str(args.output_root),'--supervise'],
            cwd=args.runtime_root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(args.output_root/'launch.json',dict(pid=process.pid,credentials_persisted=False))
    print('GENERALIZATION_LAUNCHED',process.pid)


if __name__=='__main__':main()

"""Launch and supervise the predeclared four-instance paired success pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--supervise',action='store_true')
    args=parser.parse_args()
    rt,out=args.runtime_root,args.output_root
    sys.path.insert(0,str(rt/'code'))
    from experiments.m2_small_eval.diagnostic_pilot import hashes,write_json
    if not args.supervise:
        key=sys.stdin.readline().strip()
        assert key
        env=dict(os.environ)
        env.update(json.loads((rt/'runtime-env.json').read_text()))
        env['OSWORLD_API_KEY']=key
        assert not (out/'launch.json').exists()
        with (rt/'supervisor.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),
                '--runtime-root',str(rt),'--output-root',str(out),'--supervise'],
                cwd=rt,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write_json(out/'launch.json',dict(pid=proc.pid,credentials_persisted=False))
        print('SUMMARY_SUCCESS_LAUNCHED',proc.pid)
        return
    env=dict(os.environ)
    expected=json.loads((out/'plan.json').read_text())['bank_hashes']
    jobs=[]
    try:
        for i in range(4):
            with (rt/f'worker-{i}.log').open('w') as log:
                proc=subprocess.Popen([sys.executable,'-m','experiments.m2_small_eval.summary_success_eval',
                    '--runner-root',str(rt/'runner'),'--bank',str(rt/'bank'),
                    '--case-root',str(rt/'cases'),'--output-root',str(out),'--worker',str(i),
                    '--response-cache',str(rt/'history-responses')],cwd=rt/f'work-{i}',env=env,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            jobs.append((i,proc))
        write_json(out/'workers.json',[dict(worker=i,pid=p.pid) for i,p in jobs])
        while any(p.poll() is None for _,p in jobs):
            states=[]
            completed=0
            for i,proc in jobs:
                path=out/f'worker-{i}/status.json'
                status=json.loads(path.read_text()) if path.exists() else {'state':'starting'}
                rows=out/f'worker-{i}/results.jsonl'
                completed+=len(rows.read_text().splitlines()) if rows.exists() else 0
                states.append(dict(worker=i,returncode=proc.poll(),status=status))
            assert hashes(rt/'bank')==expected
            write_json(out/'status.json',dict(state='running',completed=completed,planned=8,workers=states))
            time.sleep(10)
        assert all(p.returncode==0 for _,p in jobs),'Worker failed; do not score a harness failure'
        rows=[]
        qemu=[]
        for i,_ in jobs:
            worker_rows=[json.loads(line) for line in (out/f'worker-{i}/results.jsonl').read_text().splitlines()]
            for row in worker_rows:
                score=out/f'worker-{i}'/row['case']/row['condition']/'result.txt'
                assert float(score.read_text())==row['score']
            rows.extend(worker_rows)
            status=json.loads((out/f'worker-{i}/status.json').read_text())
            qemu.extend(status.get('closed_qemu_pids',[]))
        assert len(rows)==8
        assert not [pid for pid in qemu if Path('/proc',str(pid)).exists()]
        matrix={}
        for row in rows:
            matrix.setdefault(row['case'],{})[row['condition']]=row['score']
        assert len(matrix)==4 and all(set(v)=={'task','dual_summary'} for v in matrix.values())
        assert hashes(rt/'bank')==expected
        write_json(out/'summary.json',dict(passed_audit=True,episodes=8,cases=matrix,
            task_successes=sum(v['task'] for v in matrix.values()),
            summary_successes=sum(v['dual_summary'] for v in matrix.values()),
            paired_wins=sum(v['dual_summary']>v['task'] for v in matrix.values()),
            paired_losses=sum(v['dual_summary']<v['task'] for v in matrix.values()),
            paired_ties=sum(v['dual_summary']==v['task'] for v in matrix.values()),
            bank_unchanged=True,remaining_owned_qemu=[],
            interpretation='Four new task instances, one run per condition. Exploratory; no statistical generalization claim.'))
        write_json(out/'status.json',dict(state='complete',completed=8,planned=8))
    except Exception as exc:
        (out/'exception.txt').write_text(traceback.format_exc())
        write_json(out/'status.json',dict(state='failed',error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        for _,proc in jobs:
            if proc.poll() is None:
                proc.wait()


if __name__=='__main__':
    main()

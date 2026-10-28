"""Minimal agent success paired pilot after the frozen retrieval audit gate."""
from __future__ import annotations

import json
import hashlib
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from experiments.m2_small_eval.diagnostic_pilot import append,hashes,run_episode,write_json
from experiments.m2_small_eval.dual_eval import load_bank
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.transfer_tasks import prepare_cache
from experiments.m2_small_eval.state_guard_eval import OneActionAgent
from experiments.m2_small_eval.summary_retrieval_audit import AuditedTransport
from evotree.distill_osworld import OSWorldDistiller
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld
from evotree.situation_summary import JsonChatTransport,ModelSituationBackend
from evotree.template_lib import TemplateLibrary


CONDITIONS=('task','dual_summary')
CASES=('summary-new-calc-layout','summary-new-calc-context',
       'summary-new-font-layout','summary-new-font-context')


class StructuredSummaryBackend(ModelSituationBackend):
    """Match the structured-only retrieval audit; agent images are unchanged."""
    def summarize(self,instruction,observation,executed_prefix=()):
        evidence={k:v for k,v in observation.items() if k not in {'screenshot','exe_result'}}
        return super().summarize(instruction,evidence,executed_prefix)


class CachedHistoryTransport:
    def __init__(self,transport,cache_root,output):
        self.live=AuditedTransport(transport,output)
        self.cache_root=cache_root
        self.hits=0

    def __call__(self,messages):
        signature=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        path=self.cache_root/(signature+'.json')
        if path.exists():
            self.hits+=1
            return json.loads(path.read_text())['response']
        return self.live(messages)


def load_summary_bank(path,backend):
    templates=TemplateLibrary.from_dict(json.loads((path/'memory/templates.json').read_text()))
    return EvoTreeMemory(path,OSWorldDistiller(template_library=templates),write_enabled=False,
        explore=False,strict_quality=False,retrieval_mode='dual_summary',situation_backend=backend)


def worker(args):
    logging.basicConfig(level=logging.INFO)
    sys.path[:0]=[str(args.runner_root),'/data/songweishuai/envs/osworld/lib/python3.12/site-packages']
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment
    SLOT=12+args.worker
    os.environ.update(runtime_environment(os.environ,f'evo-rh-summary-success-r6-20261020-w{args.worker}',SLOT))
    case=next(c for c in json.loads((args.case_root/'cases-private.json').read_text())
              if c['id']==CASES[args.worker])
    prepare_cache(case,Path.cwd()/'cache')
    out=args.output_root/f'worker-{args.worker}'
    out.mkdir()
    write_json(out/'status.json',dict(state='booting',completed=0,planned=2))
    bank_before=hashes(args.bank)
    desktop=None
    final=dict(state='failed',completed=0,api_calls=0)
    rows=[]
    try:
        order=CONDITIONS if args.worker%2==0 else tuple(reversed(CONDITIONS))
        write_json(out/'plan.json',dict(case=case['id'],conditions=order,
            prompt_protocol='one executable call per response for both conditions',
            summary_scope='structured current observation and executed prefix, no images or previous result',
            bank_hashes=bank_before))
        desktop=DesktopEnv(action_space='mcp',headless=True,provider_name='docker',os_type='Ubuntu',
            screen_size=(1920,1080),require_a11y_tree=False,shuffle=False,rag=True)
        original=desktop.evaluate
        desktop.evaluate=lambda:evaluate_with_save_barrier(desktop,original,
            log=lambda event:append(out/'save-barrier.jsonl',event))
        for condition in order:
            directory=out/case['id']/condition
            write_json(out/'status.json',dict(state='running',completed=len(rows),case=case['id'],condition=condition))
            backend=None
            transport=None
            if condition=='dual_summary':
                transport=CachedHistoryTransport(JsonChatTransport(args.api_url,args.model,
                    os.environ['OSWORLD_API_KEY'],timeout=180),args.response_cache,out)
                backend=StructuredSummaryBackend(transport)
                memory=load_summary_bank(args.bank,backend)
            else:
                memory=EvoTreeMemory(args.bank,OSWorldDistiller(template_library=
                    TemplateLibrary.from_dict(json.loads((args.bank/'memory/templates.json').read_text()))),
                    write_enabled=False,explore=False,strict_quality=False,retrieval_mode='task')
            agent=OwlAgent(model=args.model,api_url=args.api_url,api_key=os.environ['OSWORLD_API_KEY'],
                history_n=3,action_space='mcp',observation_type='screenshot',
                max_trajectory_length=15,temperature=1.0,top_p=0.9)
            protocol_agent=OneActionAgent(agent)
            agent,env,runtime=wrap_osworld(protocol_agent,desktop,memory)
            started=time.monotonic()
            result=run_episode(agent,env,case['config'],directory,runtime=runtime)
            artifact=Path(desktop.cache_dir)/case['config']['evaluator']['result']['dest']
            assert artifact.is_file()
            shutil.copy2(artifact,directory/artifact.name)
            row=dict(case=case['id'],condition=condition,score=result['score'],
                decisions=result['decisions'],seconds=time.monotonic()-started,
                summary_calls=transport.live.calls if transport else 0,
                cached_summary_calls=transport.hits if transport else 0,
                bundled_responses=protocol_agent.discarded_calls)
            append(out/'results.jsonl',row);rows.append(row)
            assert hashes(args.bank)==bank_before
        assert {x['condition'] for x in rows}==set(CONDITIONS)
        final=dict(state='complete',completed=len(rows),planned=2,api_calls=sum(x['summary_calls'] for x in rows),
                   bank_unchanged=True,scores={x['condition']:x['score'] for x in rows})
    except Exception as exc:
        (out/'exception.txt').write_text(traceback.format_exc());traceback.print_exc()
        final=dict(state='failed',completed=len(rows),error_type=type(exc).__name__,error=str(exc))
    finally:
        if desktop is not None:
            final['closed_qemu_pids']=list(desktop.provider._owned_qemu);desktop.close()
        final['bank_unchanged']=hashes(args.bank)==bank_before
        write_json(out/'status.json',final)
    return 0 if final['state']=='complete' else 1


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    for name in ('runner-root','bank','case-root','output-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--response-cache',type=Path,required=True)
    p.add_argument('--worker',type=int,choices=range(4),required=True)
    p.add_argument('--model',default='gpt-5.6-terra')
    p.add_argument('--api-url',default='http://llmapi.evolventapi.com/v1')
    raise SystemExit(worker(p.parse_args()))

"""Frozen shared historical bank on genuinely different, same-template tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import time
import traceback
from dataclasses import replace

from evotree.distill_osworld import OSWorldDistiller
from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld
from evotree.template_lib import TemplateLibrary
from evotree.retrieval import render_cheatsheet
from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.transfer_tasks import prepare_cache
from experiments.m2_small_eval.workflow_control import bounded, render_tree, WorkflowAgent


HEADER = (
    "[Historical experience for a SIMILAR task]\n"
    "The following evidence comes from different files and a different task instance. "
    "All historical names, ranges, positions, commands, text and numbers are examples, NOT current instructions. "
    "Determine current values and progress from the CURRENT task and observation. "
    "Locate GUI targets again; do not reuse old coordinates or assume suggested actions already occurred.\n"
)
ORDERS = [("baseline","workflow","tree"),("workflow","tree","baseline"),("tree","baseline","workflow")]


def load_bank(path):
    library=TemplateLibrary.from_dict(json.loads((path/"memory/templates.json").read_text()))
    return EvoTreeMemory(path,OSWorldDistiller(template_library=library),write_enabled=False,explore=False,strict_quality=False)


def build_bank(seed_root, destination):
    destination.mkdir(parents=True,exist_ok=False)
    memory=EvoTreeMemory(destination,OSWorldDistiller())
    episodes={}
    sources=[]
    for index in (1,2):
        source=seed_root/f"task-{index}"
        assert json.loads((source/"seed-verified.json").read_text())["score"]==1.0
        value=json.loads((source/"seed-episode.json").read_text())
        episode=Episode(instruction=value["instruction"],success=True,
            steps=tuple(EpisodeStep(**step) for step in value["steps"]),source=str(source),metadata={"score":1.0})
        memory.write_episode(episode)
        key=memory.distiller.distill_task(value["instruction"]).template_id
        episodes[key]=value
        sources.append(dict(path=str(source/"seed-episode.json"),sha256=hashlib.sha256((source/"seed-episode.json").read_bytes()).hexdigest()))
    memory.save()
    write_json(destination/"historical-episodes.json",episodes)
    write_json(destination/"source-provenance.json",dict(sources=sources,test_episodes=0,test_labels=0))
    return hashes(destination)


def plain_text(episode,budget):
    lines=[HEADER,"Prior successful sequence; select and adapt relevant steps using the current observation:"]
    distiller=OSWorldDistiller()
    for i,step in enumerate(episode["steps"],1):
        action=distiller.distill_action(step["raw_action"])
        lines.append(f"{i}. {action.name} "+json.dumps(dict(action.args),ensure_ascii=False,sort_keys=True))
    return bounded("\n".join(lines),budget)


def transfer_tree_text(sheet,memory,instruction,budget):
    # Reuse the already tested ordered renderer. Only replace its SAME-task
    # preamble with a shared transfer preamble; no parameter binding oracle.
    from experiments.m2_small_eval.workflow_control import COMMON
    original=render_tree(sheet,memory,instruction,"ordered",10000)
    return bounded(HEADER+original[len(COMMON):],budget) if original else ""


def worker(args):
    import sys
    sys.path[:0]=[str(args.runner_root),"/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment
    logging.basicConfig(level=logging.INFO)
    slot=12+args.worker
    label=f"evo-rh-transfer-20261016-{'smoke-w' if args.smoke else 'w'}{args.worker}"
    os.environ.update(runtime_environment(os.environ,label,slot))
    os.environ["EVOTREE_RETRIEVAL_LOG"]="0"
    all_cases=json.loads((args.case_root/"cases-private.json").read_text())
    # This object is used by the harness only, never added to agent messages.
    cases=[c for i,c in enumerate(all_cases) if (c["id"] in args.smoke_cases if args.smoke else i%3==args.worker)]
    destination=args.output_root/(f"smoke-{args.worker}" if args.smoke else f"worker-{args.worker}")
    destination.mkdir(parents=True,exist_ok=False)
    status=destination/"status.json"
    before=hashes(args.bank)
    assert before==json.loads((args.bank.parent/"bank-hashes.json").read_text())
    episodes=json.loads((args.bank/"historical-episodes.json").read_text())
    memory=load_bank(args.bank)
    # No case IDs are used to select a historical example. Both memory arms
    # use exactly the same task-template matcher over the same frozen bank.
    lookup=[]
    for case in cases:
        key=memory.distiller.distill_task(case["config"]["instruction"]).template_id
        assert key in episodes,"Held-out task did not match historical template"
        prepare_cache(case,Path.cwd()/"cache")
        lookup.append(dict(case=case["id"],template=key,historical_instruction=episodes[key]["instruction"]))
    write_json(destination/"retrieval-preflight.json",lookup)
    write_json(status,dict(state="booting",completed=0,planned=len(cases)*(1 if args.smoke else 3*args.repeats)))
    desktop=None
    results=[]
    try:
        desktop=DesktopEnv(action_space="mcp",headless=True,provider_name="docker",os_type="Ubuntu",
            screen_size=(1920,1080),require_a11y_tree=False,shuffle=False,rag=True)
        original_evaluate=desktop.evaluate
        desktop.evaluate=lambda:evaluate_with_save_barrier(desktop,original_evaluate,
            log=lambda e:append(destination/"save-barrier.jsonl",e))
        for case_index,case in enumerate(cases):
            config=case["config"]
            for repetition in range(1 if args.smoke else args.repeats):
                order=("fixture",) if args.smoke else ORDERS[(repetition+args.worker+case_index)%3]
                for condition in order:
                    directory=destination/case["id"]/f"rep-{repetition+1}"/condition
                    write_json(status,dict(state="running",completed=len(results),case=case["id"],condition=condition,repetition=repetition+1))
                    agent=OwlAgent(model=args.model,api_url=args.api_url,
                        api_key="fixed-fixture-no-api" if args.smoke else os.environ["OSWORLD_API_KEY"],
                        history_n=3,action_space="mcp",observation_type="screenshot",max_trajectory_length=15,temperature=1.0,top_p=0.9)
                    env,runtime=desktop,None
                    if args.smoke:
                        from types import SimpleNamespace
                        calls=[{"name":a["action_type"],"arguments":a["parameters"]} for a in case["smoke_calls"]]
                        calls.append({"name":"computer_use","arguments":{"action":"terminate","status":"success"}})
                        prediction="".join("<tool_call>"+json.dumps(c)+"</tool_call>" for c in calls)
                        def fixed_reply(prediction=prediction,**kwargs):
                            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=prediction,tool_calls=None))])
                        agent.llm_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fixed_reply)))
                    else:
                        memory=load_bank(args.bank)
                        template=memory.distiller.distill_task(config["instruction"]).template_id
                        if condition=="workflow":
                            text=plain_text(episodes[template],3000)
                            agent=WorkflowAgent(agent,text)
                        elif condition=="tree":
                            memory._diagnostic_renderer=render_cheatsheet
                            agent,env,runtime=wrap_osworld(agent,desktop,memory)
                            original_before=runtime.before_model
                            def before_model(observation,names,memory=memory,original_before=original_before):
                                value=original_before(observation,names)
                                return replace(value,text=transfer_tree_text(value.cheatsheet,memory,config["instruction"],3000),reranked_tools=list(names))
                            runtime.before_model=before_model
                    started=time.monotonic()
                    value=run_episode(agent,env,config,directory,runtime=runtime)
                    # Keep each scored output before a later repetition reuses
                    # the task cache, for independent parameter-adaptation audit.
                    cached=Path(desktop.cache_dir)/config["evaluator"]["result"]["dest"]
                    assert cached.is_file(),"Scored output artifact is missing"
                    shutil.copy2(cached,directory/Path(cached).name)
                    if condition=="workflow":write_json(directory/"workflow-injection.json",dict(text=text,chars=len(text)))
                    row=dict(case=case["id"],family=case["family"],level=case["level"],condition=condition,
                        repetition=repetition+1,score=value["score"],decisions=value["decisions"],seconds=time.monotonic()-started)
                    append(destination/"results.jsonl",row)
                    results.append(row)
                    assert hashes(args.bank)==before,"Historical bank changed"
                    if args.smoke:assert value["score"]==1.0,"Fixed valid fixture failed; experiment must not start"
        final=dict(state="complete",completed=len(results),frozen_bank_unchanged=True)
    except Exception as exc:
        (destination/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        final=dict(state="failed",completed=len(results),error_type=type(exc).__name__,error=str(exc))
    finally:
        if desktop is not None:
            pids=list(desktop.provider._owned_qemu)
            desktop.close()
            final["closed_qemu_pids"]=pids
        write_json(status,final)
    return 0 if final["state"]=="complete" else 1


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root","bank","case-root","output-root"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--worker",type=int,choices=range(3),default=0)
    parser.add_argument("--repeats",type=int,default=2)
    parser.add_argument("--model",default="gpt-5.6-terra")
    parser.add_argument("--api-url",default="http://llmapi.evolventapi.com/v1")
    parser.add_argument("--smoke",action="store_true")
    parser.add_argument("--smoke-cases",nargs="*",default=["transfer-calc-context","transfer-font-layout"])
    args=parser.parse_args()
    raise SystemExit(worker(args))

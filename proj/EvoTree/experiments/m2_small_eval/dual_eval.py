"""Frozen held-out pilot: baseline, task workflow, horizontal state, and dual.

Each worker owns one VM and a prespecified case. It executes all four arms;
no test trajectory is added to the historical bank. Scores of zero are valid
outcomes, while harness failures stop the worker and remain unscored.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from evotree.distill_osworld import OSWorldDistiller
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld
from evotree.template_lib import TemplateLibrary
from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.transfer_tasks import prepare_cache


ARMS = ("baseline", "task", "state", "dual")
CASES = ("transfer-calc-context", "transfer-font-layout", "transfer-calc-layout")
ORDERS = (("dual", "baseline", "task", "state"), ("task", "state", "dual", "baseline"),
          ("state", "dual", "baseline", "task"))


def load_bank(path: Path, mode: str) -> EvoTreeMemory:
    library = TemplateLibrary.from_dict(json.loads((path / "memory/templates.json").read_text()))
    return EvoTreeMemory(path, OSWorldDistiller(template_library=library), write_enabled=False,
                         explore=False, strict_quality=False, retrieval_mode=mode,
                         prompt_budget_chars=6000)


def worker(args):
    logging.basicConfig(level=logging.INFO)
    sys.path[:0] = [str(args.runner_root), "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment

    name = f"evo-rh-dual-20261016-{'smoke-' if args.smoke else ''}w{args.worker}"
    os.environ.update(runtime_environment(os.environ, name, 12 + args.worker))
    os.environ["EVOTREE_RETRIEVAL_LOG"] = "0"
    case = next(c for c in json.loads((args.case_root / "cases-private.json").read_text())
                if c["id"] == CASES[args.worker])
    config = case["config"]
    prepare_cache(case, Path.cwd() / "cache")
    destination = args.output_root / (f"smoke-{args.worker}" if args.smoke else f"worker-{args.worker}")
    destination.mkdir(parents=True, exist_ok=False)
    before = hashes(args.bank)
    assert before == json.loads((args.bank.parent / "bank-hashes.json").read_text())
    memory = load_bank(args.bank, "dual")
    template = memory.lookup_task(config["instruction"]).template_id
    assert template in memory.task_experiences
    # Reject prompt-budget omissions before spending time on a VM.
    history = memory.task_experiences[template][0]
    preflight = memory.read_dual_cheatsheet(config["instruction"], {"app": "unknown"})
    text = memory.render(preflight)
    assert text and not preflight.omitted
    write_json(destination / "provenance.json", dict(case=case["id"], bank_hashes=before,
        historical_instruction=history.instruction, test_writes=0, budget=6000,
        same_task_route_for_task_and_dual=True, same_state_route_for_state_and_dual=True,
        note="Fixed per-route budgets in all arms; actual lengths logged; not token-matched."))
    desktop, rows = None, []
    final = dict(state="failed", completed=0)
    try:
        write_json(destination / "status.json", dict(state="booting", completed=0))
        desktop = DesktopEnv(action_space="mcp", headless=True, provider_name="docker", os_type="Ubuntu",
                             screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original_evaluate,
            log=lambda e: append(destination / "save-barrier.jsonl", e))
        for repetition in range(1 if args.smoke else args.repeats):
            order = ("fixture",) if args.smoke else ORDERS[(args.worker + repetition) % len(ORDERS)]
            for condition in order:
                directory = destination / case["id"] / f"rep-{repetition + 1}" / condition
                write_json(destination / "status.json", dict(state="running", completed=len(rows),
                           case=case["id"], condition=condition, repetition=repetition + 1))
                agent = OwlAgent(model=args.model, api_url=args.api_url,
                    api_key="fixture-no-api" if args.smoke else os.environ["OSWORLD_API_KEY"],
                    history_n=3, action_space="mcp", observation_type="screenshot",
                    max_trajectory_length=15, temperature=1.0, top_p=0.9)
                env, runtime = desktop, None
                if args.smoke:
                    calls = [{"name": a["action_type"], "arguments": a["parameters"]} for a in case["smoke_calls"]]
                    calls.append({"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}})
                    prediction = "".join("<tool_call>" + json.dumps(c) + "</tool_call>" for c in calls)
                    def fixed_reply(prediction=prediction, **kwargs):
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=prediction, tool_calls=None))])
                    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fixed_reply)))
                if condition != "baseline":
                    memory = load_bank(args.bank, "dual" if args.smoke else condition)
                    agent, env, runtime = wrap_osworld(agent, desktop, memory)
                started = time.monotonic()
                result = run_episode(agent, env, config, directory, runtime=runtime)
                artifact = Path(desktop.cache_dir) / config["evaluator"]["result"]["dest"]
                assert artifact.is_file(), "Scored artifact missing"
                shutil.copy2(artifact, directory / artifact.name)
                row = dict(case=case["id"], condition=condition, repetition=repetition + 1,
                           score=result["score"], decisions=result["decisions"], seconds=time.monotonic() - started)
                append(destination / "results.jsonl", row)
                rows.append(row)
                assert hashes(args.bank) == before, "Frozen historical bank changed"
                if args.smoke:
                    assert result["score"] == 1.0, "Valid fixed fixture failed"
        final = dict(state="complete", completed=len(rows), frozen_bank_unchanged=True)
    except Exception as exc:
        (destination / "exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
        final = dict(state="failed", completed=len(rows), error_type=type(exc).__name__, error=str(exc))
    finally:
        if desktop is not None:
            pids = list(desktop.provider._owned_qemu)
            desktop.close()
            final["closed_qemu_pids"] = pids
        write_json(destination / "status.json", final)
    return 0 if final["state"] == "complete" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "bank", "case-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(3), default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.6-terra")
    parser.add_argument("--api-url", default="http://llmapi.evolventapi.com/v1")
    parser.add_argument("--smoke", action="store_true")
    raise SystemExit(worker(parser.parse_args()))

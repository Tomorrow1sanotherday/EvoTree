"""Repeated task/unguarded-dual/guarded-dual diagnostic on three Calc cases.

No test trajectories are written to the frozen bank. The unguarded dual mode
is an explicit ablation that disables only state precondition enforcement.
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

from evotree.osworld_integration import wrap_osworld
from evotree.distill_osworld import parse_tool_calls
from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
from experiments.m2_small_eval.dual_eval import load_bank
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.transfer_tasks import prepare_cache


CONDITIONS = ('task', 'dual_unfiltered', 'dual')
CASES = ('expanded-calc-parameters-3', 'expanded-calc-layout-2', 'expanded-calc-context-2')
ORDERS = (('task', 'dual_unfiltered', 'dual'),
          ('dual_unfiltered', 'dual', 'task'),
          ('dual', 'task', 'dual_unfiltered'))
REPEATS = 3
ONE_ACTION_RULE = (
    "\n\nInteraction protocol: Return exactly ONE executable tool call in this response. "
    "Wait for its result and the next observation before choosing another action."
)


class OneActionAgent:
    """Apply the same fresh-observation action protocol to all three arms."""
    def __init__(self, agent):
        self.agent = agent
        self.discarded_calls = []

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def reset(self, *args, **kwargs):
        self.discarded_calls.clear()
        return self.agent.reset(*args, **kwargs)

    def predict(self, instruction, obs, *args, **kwargs):
        result = self.agent.predict(instruction.rstrip() + ONE_ACTION_RULE, obs, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 2:
            return result
        response, actions = result[0], result[1]
        if not isinstance(actions, (list, tuple)) or len(actions) <= 1:
            return result
        calls = parse_tool_calls(response) if isinstance(response, str) else []
        if len(calls) != len(actions):
            raise RuntimeError('Bundled model calls and executable actions are not aligned')
        first = '<tool_call>' + json.dumps(calls[0], ensure_ascii=False) + '</tool_call>'
        self.discarded_calls.append(dict(returned=len(calls), executed=1, discarded=len(calls)-1))
        return (first, list(actions[:1]), *result[2:])


def testable_one_action_agent(agent):
    """Small factory kept public for the protocol regression test."""
    return OneActionAgent(agent)


def order(case_index, repetition):
    return ORDERS[(case_index + repetition) % len(ORDERS)]


def selected_case(case_root, case_id):
    values = json.loads((case_root / 'cases-private.json').read_text())
    return next(item for item in values if item['id'] == case_id)


def run_worker(args):
    logging.basicConfig(level=logging.INFO)
    sys.path[:0] = [str(args.runner_root), '/data/songweishuai/envs/osworld/lib/python3.12/site-packages']
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment

    label = f"evo-rh-state-guard-20261019-w{args.worker}"
    os.environ.update(runtime_environment(os.environ, label, 4 + args.worker))
    os.environ['EVOTREE_RETRIEVAL_LOG'] = '0'
    case_id = CASES[args.worker]
    case = selected_case(args.case_root, case_id)
    destination = args.output_root / (f'smoke-worker-{args.worker}' if args.smoke else f'worker-{args.worker}')
    destination.mkdir(parents=True, exist_ok=False)
    expected = json.loads((args.bank.parent / 'bank-hashes.json').read_text())
    assert hashes(args.bank) == expected
    planned = 1 if args.smoke else len(CONDITIONS) * REPEATS
    plan = dict(case=case_id, planned=planned, conditions=('dual',) if args.smoke else CONDITIONS,
                repetitions=1 if args.smoke else REPEATS,
                orders=[('dual',)] if args.smoke else [order(args.worker, i) for i in range(REPEATS)],
                frozen_bank_hashes=expected)
    write_json(destination / 'plan.json', plan)
    desktop, rows = None, []
    final = dict(state='failed', completed=0, planned=planned)
    try:
        write_json(destination / 'status.json', dict(state='booting', completed=0, planned=planned))
        desktop = DesktopEnv(action_space='mcp', headless=True, provider_name='docker', os_type='Ubuntu',
                             screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original_evaluate,
            log=lambda event: append(destination / 'save-barrier.jsonl', event))
        repetitions = (0,) if args.smoke else range(REPEATS)
        for repetition in repetitions:
            conditions = ('dual',) if args.smoke else order(args.worker, repetition)
            for condition in conditions:
                prepare_cache(case, Path.cwd() / 'cache')
                run_dir = destination / case_id / ('smoke' if args.smoke else f'rep-{repetition+1}') / condition
                write_json(destination / 'status.json', dict(state='running', completed=len(rows), planned=planned,
                    case=case_id, condition=condition, repetition=repetition+1))
                agent = OwlAgent(model=args.model, api_url=args.api_url, api_key=os.environ['OSWORLD_API_KEY'],
                    history_n=3, action_space='mcp', observation_type='screenshot',
                    max_trajectory_length=15, temperature=1.0, top_p=0.9)
                memory = load_bank(args.bank, condition)
                protocol_agent = OneActionAgent(agent)
                agent, env, runtime = wrap_osworld(protocol_agent, desktop, memory)
                started = time.monotonic()
                result = run_episode(agent, env, case['config'], run_dir, runtime=runtime)
                artifact = Path(desktop.cache_dir) / case['config']['evaluator']['result']['dest']
                assert artifact.is_file(), 'Scored task output missing'
                shutil.copy2(artifact, run_dir / artifact.name)
                row = dict(case=case_id, family='calc', level=case['level'], condition=condition,
                    repetition=repetition+1, score=result['score'], decisions=result['decisions'],
                    seconds=time.monotonic()-started,
                    prompt_chars=[d['chars'] for d in read_jsonl(run_dir / 'decisions.jsonl')],
                    bundled_responses=protocol_agent.discarded_calls)
                if protocol_agent.discarded_calls:
                    write_json(run_dir/'bundled-response-audit.json', protocol_agent.discarded_calls)
                append(destination / 'results.jsonl', row)
                rows.append(row)
                assert hashes(args.bank) == expected, 'Frozen historical bank changed'
        final = dict(state='complete', completed=len(rows), planned=planned, frozen_bank_unchanged=True)
    except Exception as exc:
        (destination / 'exception.txt').write_text(traceback.format_exc())
        traceback.print_exc()
        final = dict(state='failed', completed=len(rows), planned=planned,
                     error_type=type(exc).__name__, error=str(exc))
    finally:
        if desktop is not None:
            pids = list(desktop.provider._owned_qemu)
            desktop.close()
            final['closed_qemu_pids'] = pids
        write_json(destination / 'status.json', final)
    return 0 if final['state'] == 'complete' else 1


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner-root', 'bank', 'case-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=range(3), default=0)
    parser.add_argument('--model', default='gpt-5.6-terra')
    parser.add_argument('--api-url', default='http://llmapi.evolventapi.com/v1')
    parser.add_argument('--smoke', action='store_true')
    raise SystemExit(run_worker(parser.parse_args()))

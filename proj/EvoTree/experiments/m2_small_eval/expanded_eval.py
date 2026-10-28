"""24 cases x four frozen memory conditions, with all-case fixture validation."""
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
from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
from experiments.m2_small_eval.dual_eval import ARMS, load_bank
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.transfer_tasks import prepare_cache


# Balanced Williams design: each condition appears in each position once and
# each ordered adjacent pair once in four cases. Six complete blocks in 24 cases.
ORDERS = (("baseline", "task", "dual", "state"),
          ("task", "state", "baseline", "dual"),
          ("state", "dual", "task", "baseline"),
          ("dual", "baseline", "state", "task"))


def schedule(cases, worker, workers=3):
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError('Invalid worker partition')
    return [(index, case, ORDERS[(index // 2 + index % 2) % 4]) for index, case in enumerate(cases)
            if index % workers == worker]


def worker(args):
    logging.basicConfig(level=logging.INFO)
    sys.path[:0] = [str(args.runner_root), '/data/songweishuai/envs/osworld/lib/python3.12/site-packages']
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment

    label = f"evo-rh-expanded-20261016-{'smoke-' if args.smoke else ''}w{args.worker}"
    os.environ.update(runtime_environment(os.environ, label, 12 + args.worker))
    os.environ['EVOTREE_RETRIEVAL_LOG'] = '0'
    cases = json.loads((args.case_root / 'cases-private.json').read_text())
    selected = schedule(cases, args.worker)
    destination = args.output_root / (f'smoke-{args.worker}' if args.smoke else f'worker-{args.worker}')
    destination.mkdir(parents=True, exist_ok=False)
    expected = json.loads((args.case_root.parent / 'bank-hashes.json').read_text())
    assert hashes(args.bank) == expected
    planned = len(selected) * (1 if args.smoke else len(ARMS))
    write_json(destination / 'plan.json', dict(cases=[c['id'] for _, c, _ in selected], planned=planned,
               schedule=[dict(case=c['id'], order=order) for _, c, order in selected], bank_hashes=expected))
    desktop, rows = None, []
    final = dict(state='failed', completed=0)
    try:
        write_json(destination / 'status.json', dict(state='booting', completed=0, planned=planned))
        desktop = DesktopEnv(action_space='mcp', headless=True, provider_name='docker', os_type='Ubuntu',
                             screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original_evaluate,
            log=lambda event: append(destination / 'save-barrier.jsonl', event))
        for _, case, order in selected:
            assert hashes(args.bank) == expected
            for condition in (('fixture',) if args.smoke else order):
                prepare_cache(case, Path.cwd() / 'cache')
                directory = destination / case['id'] / 'rep-1' / condition
                write_json(destination / 'status.json', dict(state='running', completed=len(rows),
                           planned=planned, case=case['id'], condition=condition))
                agent = OwlAgent(model=args.model, api_url=args.api_url,
                    api_key='fixed-fixture-no-api' if args.smoke else os.environ['OSWORLD_API_KEY'],
                    history_n=3, action_space='mcp', observation_type='screenshot',
                    max_trajectory_length=15, temperature=1.0, top_p=0.9)
                env, runtime = desktop, None
                if args.smoke:
                    calls = [dict(name=action['action_type'], arguments=action['parameters'])
                             for action in case['smoke_calls']]
                    calls.append(dict(name='computer_use', arguments=dict(action='terminate', status='success')))
                    prediction = ''.join('<tool_call>' + json.dumps(call) + '</tool_call>' for call in calls)
                    def reply(prediction=prediction, **kwargs):
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=prediction, tool_calls=None))])
                    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=reply)))
                if condition != 'baseline':
                    memory = load_bank(args.bank, 'dual' if args.smoke else condition)
                    agent, env, runtime = wrap_osworld(agent, desktop, memory)
                started = time.monotonic()
                value = run_episode(agent, env, case['config'], directory, runtime=runtime)
                artifact = Path(desktop.cache_dir) / case['config']['evaluator']['result']['dest']
                assert artifact.is_file(), 'Missing scored output artifact'
                shutil.copy2(artifact, directory / artifact.name)
                row = dict(case=case['id'], family=case['family'], level=case['level'], condition=condition,
                           repetition=1, score=value['score'], decisions=value['decisions'], seconds=time.monotonic()-started)
                append(destination / 'results.jsonl', row)
                rows.append(row)
                assert hashes(args.bank) == expected, 'Historical bank changed'
                if args.smoke:
                    assert value['score'] == 1.0, 'Fixed valid fixture failed; do not start formal experiment'
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('runner-root', 'bank', 'case-root', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', type=int, choices=range(3), default=0)
    parser.add_argument('--model', default='gpt-5.6-terra')
    parser.add_argument('--api-url', default='http://llmapi.evolventapi.com/v1')
    parser.add_argument('--smoke', action='store_true')
    raise SystemExit(worker(parser.parse_args()))

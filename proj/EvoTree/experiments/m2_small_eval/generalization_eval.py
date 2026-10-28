"""Frozen 24-case, two-arm paired experiment and owned VM fixture smoke."""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace

from evotree.distill_osworld import OSWorldDistiller
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import wrap_osworld
from evotree.template_lib import TemplateLibrary
from evotree.situation_summary import JsonChatTransport
from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
from experiments.m2_small_eval.state_guard_eval import OneActionAgent
from experiments.m2_small_eval.summary_success_eval import CachedHistoryTransport, StructuredSummaryBackend
from experiments.m2_small_eval.transfer_tasks import prepare_cache

ARMS = ('task', 'dual_summary')


def schedule(cases, seed=20261021):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for case in cases:
        strata[case['family'], case['level']].append(case['id'])
    result = [[] for _ in range(4)]
    for stratum in sorted(strata):
        ids = sorted(strata[stratum])
        if len(ids) != 4:
            raise ValueError('Expected four cases in every family/level stratum')
        rng.shuffle(ids)
        first = [ARMS, ARMS, tuple(reversed(ARMS)), tuple(reversed(ARMS))]
        rng.shuffle(first)
        for worker, (case, order) in enumerate(zip(ids, first)):
            result[worker].append(dict(case=case, order=list(order)))
    for jobs in result:
        rng.shuffle(jobs)
    return result


class ExecutedHistoryAgent(OneActionAgent):
    """Keep OwlAgent's own history consistent with the single executed call."""
    def predict(self, *args, **kwargs):
        previous = len(self.discarded_calls)
        result = super().predict(*args, **kwargs)
        if len(self.discarded_calls) != previous:
            response, actions = result[:2]
            for name in ('history_responses', 'thoughts'):
                values = getattr(self.agent, name, None)
                if isinstance(values, list) and values:
                    values[-1] = response
            if getattr(self.agent, 'actions', None):
                self.agent.actions[-1] = list(actions)
            if getattr(self.agent, 'history_conclusion', None):
                self.agent.history_conclusion[-1] = 'Single retained call; later proposed calls were not executed. ' + response
        return result


class RequestAudit:
    def __init__(self, client, path, *, action_protocol_reminder=False):
        self.client, self.path, self.calls = client, path, 0
        self.action_protocol_reminder = action_protocol_reminder

    def create(self, **kwargs):
        started = time.monotonic()
        self.calls += 1
        if self.action_protocol_reminder:
            from evotree.action_protocol import with_action_protocol
            kwargs = dict(kwargs, messages=with_action_protocol(kwargs.get('messages') or []))
        from evotree.request_metadata import request_metadata
        request_shape = request_metadata(kwargs)
        try:
            result = self.client.create(**kwargs)
        except Exception as exc:
            append(self.path, dict(call=self.calls, seconds=time.monotonic()-started,
                                   request_shape=request_shape,
                                   error_type=type(exc).__name__))
            raise
        usage = getattr(result, 'usage', None)
        choices = getattr(result, 'choices', None) or []
        message = getattr(choices[0], 'message', None) if choices else None
        content = getattr(message, 'content', None)
        tool_calls = getattr(message, 'tool_calls', None)
        # Reasoning can consume the entire prefix preview. Preserve bounded
        # action blocks separately, including malformed/unclosed JSON blocks.
        protocol_blocks = []
        if isinstance(content, str):
            for block in content.split('<tool_call>')[1:4]:
                body, closing, _ = block.partition('</tool_call>')
                protocol_blocks.append(dict(body=body[:800], closed=bool(closing),
                                            truncated=len(body) > 800))
        append(self.path, dict(call=self.calls, seconds=time.monotonic()-started,
            request_shape=request_shape,
            usage=usage.model_dump() if hasattr(usage, 'model_dump') else None,
            finish_reason=getattr(choices[0], 'finish_reason', None) if choices else None,
            protocol_blocks=protocol_blocks,
            response_shape=dict(choices=len(choices), content_type=type(content).__name__,
                content_chars=len(content) if isinstance(content, str) else None,
                tool_call_count=len(tool_calls or []),
                tool_call_names=[name for call in (tool_calls or [])
                                 for name in [getattr(getattr(call, 'function', None), 'name', None)]
                                 if name]),
            # Short preview is for protocol debugging only; it excludes request
            # prompts and credentials and is bounded before writing to disk.
            content_preview=content[:600] if isinstance(content, str) else None))
        return result


def make_memory(bank, arm, backend=None):
    templates = TemplateLibrary.from_dict(json.loads((bank/'memory/templates.json').read_text()))
    return EvoTreeMemory(bank, OSWorldDistiller(template_library=templates), write_enabled=False,
        explore=False, strict_quality=False, retrieval_mode=arm, situation_backend=backend)


def worker(args):
    rt, out = args.runtime_root, args.output_root
    sys.path[:0] = [str(rt/'runner'), '/data/songweishuai/envs/osworld/lib/python3.12/site-packages']
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment
    logging.basicConfig(level=logging.WARNING)
    label = f'evo-generalization2-20261021-{args.phase}-w{args.worker}'
    os.environ.update(runtime_environment(os.environ, label, 12+args.worker))
    manifest = json.loads((out/'plan.json').read_text())
    cases = {c['id']: c for c in json.loads((rt/'cases/cases-private.json').read_text())}
    jobs = manifest['schedule'][args.worker]
    if args.phase == 'smoke':
        family = ('calc', 'font')[args.worker]
        case = next(c for c in cases.values() if c['family']==family and c['level']=='context')
        jobs = [dict(case=case['id'], order=['fixture'])]
    folder = out/f'{args.phase}-worker-{args.worker}'
    folder.mkdir()
    expected = manifest['bank_hashes']
    assert hashes(rt/'bank') == expected
    rows, desktop = [], None
    status = dict(state='booting', completed=0, planned=sum(len(j['order']) for j in jobs))
    write_json(folder/'status.json', status)
    try:
        desktop = DesktopEnv(action_space='mcp', headless=True, provider_name='docker', os_type='Ubuntu',
            screen_size=(1920,1080), require_a11y_tree=False, shuffle=False, rag=True)
        original = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original,
            log=lambda event: append(folder/'save-barrier.jsonl', event))
        for job in jobs:
            case = cases[job['case']]
            for arm in job['order']:
                directory = folder/case['id']/arm
                status.update(state='running', case=case['id'], condition=arm, completed=len(rows))
                write_json(folder/'status.json', status)
                prepare_cache(case, Path.cwd()/'cache')
                transport = None
                backend = None
                if arm == 'dual_summary':
                    requests = directory.with_name(arm+'-requests')
                    requests.mkdir(parents=True)
                    transport = CachedHistoryTransport(JsonChatTransport(manifest['api_url'], manifest['model'],
                        os.environ['OSWORLD_API_KEY'], timeout=180), rt/'history-responses', requests)
                    backend = StructuredSummaryBackend(transport)
                agent = OwlAgent(model=manifest['model'], api_url=manifest['api_url'],
                    api_key=os.environ.get('OSWORLD_API_KEY','fixture-no-api'), history_n=3,
                    action_space='mcp', observation_type='screenshot', max_trajectory_length=15,
                    temperature=1.0, top_p=.9)
                counter = None
                if arm == 'fixture':
                    calls = [dict(name=a['action_type'], arguments=a['parameters']) for a in case['smoke_calls']]
                    calls.append(dict(name='computer_use',arguments=dict(action='terminate',status='success')))
                    replies = iter('<tool_call>'+json.dumps(c)+'</tool_call>' for c in calls)
                    def create(**kwargs):
                        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                            content=next(replies), tool_calls=None))])
                    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
                else:
                    counter = RequestAudit(agent.llm_client.chat.completions,
                        folder/(case['id']+'-'+arm+'-agent-requests.jsonl'))
                    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=counter))
                protocol = ExecutedHistoryAgent(agent)
                memory = make_memory(rt/'bank', 'task' if arm=='fixture' else arm, backend)
                wrapped, env, runtime = wrap_osworld(protocol, desktop, memory)
                started = time.monotonic()
                value = run_episode(wrapped, env, case['config'], directory, runtime=runtime)
                if arm == 'fixture':
                    assert value['score'] == 1.0, 'Fixture failed: do not launch formal experiment'
                if arm == 'dual_summary':
                    decisions = [json.loads(l) for l in (directory/'decisions.jsonl').read_text().splitlines()]
                    for decision in decisions:
                        accepted = {(s['template'],s['node']) for s in decision['state_sources']}
                        assert all(c['verdict']=='compatible' for c in decision['summary_comparisons']
                            if (c['template'],c['node']) in accepted), 'Unverified state injected'
                artifact = Path(desktop.cache_dir)/case['config']['evaluator']['result']['dest']
                assert artifact.is_file()
                shutil.copy2(artifact, directory/artifact.name)
                row = dict(case=case['id'], family=case['family'], level=case['level'], condition=arm,
                    score=value['score'], decisions=value['decisions'], seconds=time.monotonic()-started,
                    agent_calls=counter.calls if counter else 0,
                    summary_calls=transport.live.calls if transport else 0,
                    cached_summary_calls=transport.hits if transport else 0, bundled_responses=protocol.discarded_calls)
                append(folder/'results.jsonl', row)
                rows.append(row)
                assert hashes(rt/'bank') == expected
        status.update(state='complete', completed=len(rows))
    except Exception as exc:
        (folder/'exception.txt').write_text(traceback.format_exc())
        status.update(state='failed', completed=len(rows), error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        if desktop is not None:
            status['closed_qemu_pids'] = list(desktop.provider._owned_qemu)
            desktop.close()
        status['bank_unchanged'] = hashes(rt/'bank') == expected
        write_json(folder/'status.json', status)
    return 0 if status['state']=='complete' else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--runtime-root', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--worker', type=int, choices=range(4), required=True)
    p.add_argument('--phase', choices=('smoke','formal'), required=True)
    raise SystemExit(worker(p.parse_args()))

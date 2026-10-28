"""Paired recall ablation on archived observations, without starting any VMs.

The original bank and response cache are read-only. Private phase labels are
used only by the scorer, never by recall, summaries or semantic comparisons.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import threading
import time

from evotree.distill_osworld import OSWorldDistiller
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import (JsonChatTransport, LEGACY_COMPARE_INSTRUCTION,
                                      ModelSituationBackend)
from evotree.template_lib import TemplateLibrary
from experiments.m2_small_eval.diagnostic_pilot import hashes, write_json
from experiments.m2_small_eval.summary_retrieval_audit import phase_label, step_from_dict
from evotree.episode import Episode


CONDITIONS = ('strict_legacy', 'relaxed_legacy', 'relaxed_verified')


def normalized(value):
    """Compare JSON structures independent of tuple/list deserialization."""
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


class ReusedTransport:
    """Cache exact request hashes; serialize duplicate requests, not all calls."""
    def __init__(self, transport, old_root, root, *, allow_live=True):
        self.transport = transport
        self.old_roots = tuple(old_root) if isinstance(old_root, (tuple, list)) else (old_root,)
        self.root = root
        self.allow_live = allow_live
        self.lock = threading.Lock()
        self.locks = {}
        self.calls, self.hits, self.seconds = 0, 0, 0.
        (root/'responses').mkdir()

    def __call__(self, messages):
        signature = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        with self.lock:
            lock = self.locks.setdefault(signature, threading.Lock())
        with lock:
            for folder in (*self.old_roots, self.root):
                path = folder/'responses'/f'{signature}.json'
                if path.exists():
                    with self.lock:
                        self.hits += 1
                    return json.loads(path.read_text(encoding='utf-8'))['response']
            if not self.allow_live:
                raise RuntimeError(f'Offline response cache miss: {signature}')
            started = time.monotonic()
            response = self.transport(messages)
            seconds = time.monotonic()-started
            write_json(self.root/'responses'/f'{signature}.json', dict(
                request_sha256=signature, request=messages, response=response, seconds=seconds))
            with self.lock:
                self.calls += 1
                self.seconds += seconds
            return response


def score_sheet(sheet, truth, labels):
    def tagged(row):
        return dict(row, labels=sorted(labels.get((row['template'], row['node']), set())))
    pool = [tagged(row) for row in sheet.recall_candidates]
    compared = [tagged(row) for row in sheet.summary_comparisons]
    selected = [tagged(dict(template=item.template_id, node=item.node_id,
        action=item.recommendation.action.name, summary=item.situation_summary,
        comparison=item.summary_match)) for item in sheet.state_experiences]
    correct = [truth in row['labels'] and truth != 'unknown' for row in selected]
    has_match = truth != 'unknown' and any(truth in values for values in labels.values())
    return dict(selected=selected, selected_count=len(selected), correct_count=sum(correct),
        top1_correct=bool(correct and correct[0]), coverage=bool(selected),
        has_matching_history=has_match,
        no_incompatible_selected=all(correct) if selected else not has_match,
        correct_abstention=not selected and not has_match,
        coarse_recall=any(truth in row['labels'] for row in pool) and has_match,
        bounded_recall=any(truth in row['labels'] for row in compared) and has_match,
        candidate_count=len(pool), compared_count=len(compared),
        recall_candidates=pool, summary_comparisons=compared,
        current_summary=sheet.current_situation, rejected=sheet.rejected_states)


def aggregate(rows):
    totals = {}
    for family in ('calc', 'impress', 'all'):
        subset = [row for row in rows if family == 'all' or row['family'] == family]
        totals[family] = {}
        for condition in CONDITIONS:
            metrics = [row['conditions'][condition] for row in subset]
            totals[family][condition] = dict(n=len(metrics),
                positive_n=sum(m['has_matching_history'] for m in metrics),
                **{key: sum(m[key] for m in metrics) for key in (
                    'top1_correct', 'no_incompatible_selected', 'coarse_recall',
                    'bounded_recall', 'correct_abstention', 'correct_count',
                    'selected_count', 'compared_count')},
                coverage=sum(m['coverage'] and m['has_matching_history'] for m in metrics))
    return totals


def audit(source: Path, root: Path, api_key: str, workers=4):
    root.mkdir(parents=True, exist_ok=False)
    bank = source/'bank'
    frozen_hashes = hashes(bank)
    source_cache_hashes = hashes(source/'responses')
    write_json(root/'bank-hashes.json', frozen_hashes)
    write_json(root/'plan.json', dict(conditions=CONDITIONS, workers=workers, samples=13,
        candidate_budget=24, same_observations=True, same_frozen_bank=True,
        same_summary_prompt=True, specialized_guards=False, vms_started=0,
        conditions_explained=dict(strict_legacy='original hard-stage gate and comparator; retain unknown',
            relaxed_legacy='relax stage gate only; original comparator; retain unknown',
            relaxed_verified='relaxed stage gate; compare with both task goals; newly recalled states require compatible; legacy-stage unknown remains labelled'),
        gate='bounded positive recall must improve; top1, clean-selection and positive coverage cannot regress; negative abstention retained',
        limitations='Development on the same 13 diagnostic decisions; no held-out task success or generalization claim.'))
    transport = ReusedTransport(JsonChatTransport('http://llmapi.evolventapi.com/v1',
        'gpt-5.6-terra', api_key, timeout=180), source, root)
    library = TemplateLibrary.from_dict(json.loads((bank/'memory/templates.json').read_text(encoding='utf-8')))

    def make_memory(legacy):
        kwargs = dict(comparison_instruction=LEGACY_COMPARE_INSTRUCTION) if legacy else {}
        backend = ModelSituationBackend(transport, **kwargs)
        return EvoTreeMemory(bank, OSWorldDistiller(template_library=library), write_enabled=False,
                             strict_quality=False, situation_backend=backend)

    original = make_memory(True)
    probes, labels = [], {}
    for family in ('calc', 'impress'):
        probes.extend(json.loads((source/family/'probes.json').read_text(encoding='utf-8')))
        seed = json.loads((source/family/'history.json').read_text(encoding='utf-8'))
        episode = Episode(seed['instruction'], tuple(step_from_dict(s) for s in seed['steps']), True)
        template = original.lookup_task(episode.instruction).template_id
        tree = original.forest.get_tree(template)
        for step in episode.steps:
            node = tree.node_for_signature(original.distiller.distill_state(step.observation))
            labels.setdefault((template, node.node_id), set()).add(phase_label(step.observation, seed['private']))
    archived = {r['id']: r for r in json.loads((source/'paired-results.json').read_text(encoding='utf-8'))}
    rows = []
    for probe in probes:
        assert probe['observation']['observation_provenance']['status'] == 'stable'
        sheet = original.read_dual_cheatsheet(probe['instruction'], probe['observation'],
            enforce_state_conditions=False, use_situation_summaries=True,
            relax_action_stage=False, require_summary_compatible=False, require_compatible_all=False)
        measured = score_sheet(sheet, phase_label(probe['observation'], probe['private']), labels)
        old = archived[probe['id']]['conditions']['dual_summary']
        assert [(r['template'], r['node'], r['action']) for r in measured['selected']] == [
            (r['template'], r['node'], r['action']) for r in old['selected']], 'Legacy selection drift'
        assert normalized(measured['current_summary']) == normalized(old['current_summary']), 'Current summary drift'
        for key in ('top1_correct', 'no_incompatible_selected', 'selected_count', 'correct_count', 'coverage'):
            assert measured[key] == old[key], f'Legacy metric drift: {key}'
        rows.append(dict(id=probe['id'], family=probe['private']['family'],
            truth=phase_label(probe['observation'], probe['private']), conditions={'strict_legacy': measured}))
    assert transport.calls == 0, 'Baseline must reproduce exclusively from archived responses'
    write_json(root/'baseline-verification.json', dict(samples=len(rows), exact_selection_match=True,
        exact_current_summary_match=True, network_calls=transport.calls))

    def evaluate(index):
        probe = probes[index]
        per = {}
        for condition in CONDITIONS[1:]:
            memory = make_memory(condition == 'relaxed_legacy')
            started = time.monotonic()
            sheet = memory.read_dual_cheatsheet(probe['instruction'], probe['observation'],
                enforce_state_conditions=False, use_situation_summaries=True, relax_action_stage=True,
                require_summary_compatible=condition == 'relaxed_verified', require_compatible_all=False)
            per[condition] = score_sheet(sheet, rows[index]['truth'], labels)
            per[condition]['seconds'] = time.monotonic()-started
            assert per[condition]['current_summary'] == rows[index]['conditions']['strict_legacy']['current_summary']
        return index, per

    write_json(root/'status.json', dict(state='running', completed=0, planned=len(rows)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate, i) for i in range(len(rows))]
        for completed, future in enumerate(as_completed(futures), 1):
            index, per = future.result()
            rows[index]['conditions'].update(per)
            write_json(root/'paired-results.json', rows)
            write_json(root/'status.json', dict(state='running', completed=completed, planned=len(rows),
                model_calls=transport.calls, cached_requests=transport.hits))
            assert hashes(bank) == frozen_hashes
    totals = aggregate(rows)
    old, new = (totals['all'][c] for c in ('strict_legacy', 'relaxed_verified'))
    gate = new['bounded_recall'] > old['bounded_recall'] and all(
        new[k] >= old[k] for k in ('top1_correct', 'no_incompatible_selected', 'coverage', 'correct_abstention'))
    assert hashes(bank) == frozen_hashes and hashes(source/'responses') == source_cache_hashes
    result = dict(completed=True, samples=len(rows), totals=totals, gate_passed=gate,
        network_calls=transport.calls, cached_requests=transport.hits,
        summed_network_seconds=transport.seconds, bank_unchanged=True, source_cache_unchanged=True)
    write_json(root/'summary.json', result)
    write_json(root/'status.json', dict(state='complete', completed=len(rows), planned=len(rows)))
    return result

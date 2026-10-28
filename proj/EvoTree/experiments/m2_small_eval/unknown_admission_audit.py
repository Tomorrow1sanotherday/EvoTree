"""Replay the fixed retrieval audit with unknown rejected at every node."""
from __future__ import annotations

import json
from pathlib import Path
import time

from evotree.distill_osworld import OSWorldDistiller
from evotree.episode import Episode
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import JsonChatTransport, ModelSituationBackend
from evotree.template_lib import TemplateLibrary
from experiments.m2_small_eval.candidate_recall_audit import ReusedTransport, score_sheet
from experiments.m2_small_eval.diagnostic_pilot import hashes, write_json
from experiments.m2_small_eval.summary_retrieval_audit import phase_label, step_from_dict


def normalized(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def run(source: Path, reference: Path, output: Path, api_key: str):
    output.mkdir(parents=True, exist_ok=False)
    bank = source/'bank'
    bank_hashes = hashes(bank)
    reference_hashes = hashes(reference/'responses')
    source_cache_hashes = hashes(source/'responses')
    transport = ReusedTransport(JsonChatTransport('http://llmapi.evolventapi.com/v1',
        'gpt-5.6-terra', api_key, timeout=180), (source, reference), output, allow_live=False)
    library = TemplateLibrary.from_dict(json.loads(
        (bank/'memory/templates.json').read_text(encoding='utf-8')))
    backend = ModelSituationBackend(transport)
    memory = EvoTreeMemory(bank, OSWorldDistiller(template_library=library), write_enabled=False,
        strict_quality=False, situation_backend=backend)

    probes, labels = [], {}
    for family in ('calc', 'impress'):
        folder = source/family
        probes.extend(json.loads((folder/'probes.json').read_text(encoding='utf-8')))
        seed = json.loads((folder/'history.json').read_text(encoding='utf-8'))
        episode = Episode(seed['instruction'], tuple(step_from_dict(s) for s in seed['steps']), True)
        template = memory.lookup_task(episode.instruction).template_id
        tree = memory.forest.get_tree(template)
        for step in episode.steps:
            node = tree.node_for_signature(memory.distiller.distill_state(step.observation))
            labels.setdefault((template, node.node_id), set()).add(
                phase_label(step.observation, seed['private']))

    old_rows = {r['id']: r for r in json.loads(
        (reference/'paired-results.json').read_text(encoding='utf-8'))}
    rows = []
    write_json(output/'plan.json', dict(unit='13 archived stable observation decisions',
        condition='same relaxed recall and goal-aware comparator; require compatible for every candidate',
        comparison='r3 run with only unknown-admission policy changed',
        frozen_bank=True, vms_started=0, response_replay_only=True,
        gate='remove all incompatible selected candidates while retain positive coverage and correct top1 from r3'))
    write_json(output/'status.json', dict(state='running', completed=0, planned=len(probes)))
    for index, probe in enumerate(probes, 1):
        assert probe['observation']['observation_provenance']['status'] == 'stable'
        started = time.monotonic()
        sheet = memory.read_dual_cheatsheet(probe['instruction'], probe['observation'],
            enforce_state_conditions=False, use_situation_summaries=True,
            relax_action_stage=True, require_compatible_all=True)
        truth = phase_label(probe['observation'], probe['private'])
        scored = score_sheet(sheet, truth, labels)
        reference_row = old_rows[probe['id']]['conditions']['relaxed_verified']
        if normalized(scored['current_summary']) != normalized(reference_row['current_summary']):
            raise AssertionError('Current summary changed from r3')
        scored['seconds'] = time.monotonic()-started
        rows.append(dict(id=probe['id'], family=probe['private']['family'], truth=truth,
                         strict_unknown=scored, r3=scored_reference(reference_row)))
        write_json(output/'paired-results.json', rows)
        write_json(output/'status.json', dict(state='running', completed=index,
            planned=len(probes), network_calls=transport.calls, cache_hits=transport.hits))
        assert hashes(bank) == bank_hashes

    if transport.calls:
        raise AssertionError(f'Expected exact cached requests; made {transport.calls} live calls')
    if hashes(reference/'responses') != reference_hashes or hashes(source/'responses') != source_cache_hashes:
        raise AssertionError('Source response cache changed')
    total = lambda key: sum(bool(row['strict_unknown'][key]) for row in rows)
    r3_total = lambda key: sum(bool(row['r3'][key]) for row in rows)
    result = dict(completed=True, samples=len(rows), strict_unknown=dict(
        positive_n=sum(row['strict_unknown']['has_matching_history'] for row in rows),
        top1_correct=total('top1_correct'), coverage=sum(
            row['strict_unknown']['coverage'] and row['strict_unknown']['has_matching_history'] for row in rows),
        no_incompatible_selected=total('no_incompatible_selected'),
        correct_abstention=total('correct_abstention'), selected=total('selected_count')),
        r3=dict(top1_correct=r3_total('top1_correct'), coverage=sum(
            row['r3']['coverage'] and row['r3']['has_matching_history'] for row in rows),
            no_incompatible_selected=r3_total('no_incompatible_selected'),
            correct_abstention=r3_total('correct_abstention'), selected=r3_total('selected_count')),
        gate_passed=(total('no_incompatible_selected') == len(rows)
            and result_coverage(rows) == 12
            and total('top1_correct') >= r3_total('top1_correct')),
        network_calls=transport.calls, cache_hits=transport.hits,
        bank_unchanged=True, reference_cache_unchanged=True)
    write_json(output/'summary.json', result)
    write_json(output/'status.json', dict(state='complete', completed=len(rows), planned=len(probes)))
    return result


def result_coverage(rows):
    return sum(row['strict_unknown']['coverage'] and row['strict_unknown']['has_matching_history']
               for row in rows)


def scored_reference(row):
    return {key: row[key] for key in ('top1_correct', 'coverage', 'has_matching_history',
        'no_incompatible_selected', 'correct_abstention', 'selected_count', 'correct_count')}

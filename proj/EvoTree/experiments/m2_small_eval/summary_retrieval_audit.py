"""Small frozen retrieval audit over fresh, measured Office observations.

Private phase labels are used ONLY to score selected historical states. They
never enter the summary or comparison calls. No agent success rate is measured.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from evotree.current_observation import parse_structured
from evotree.distill_osworld import OSWorldDistiller
from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.situation_summary import JsonChatTransport, ModelSituationBackend
from evotree.template_lib import TemplateLibrary
from experiments.m2_small_eval.diagnostic_pilot import hashes, write_json


CASES = {
    'calc': ('expanded-calc-parameters-3','expanded-calc-layout-2','expanded-calc-context-2'),
    'impress': ('expanded-font-layout-2','expanded-font-context-2'),
}
PLAN = dict(
    unit='retrieval decision; diagnostic, not task success rate',
    conditions=['dual_unfiltered','dual_summary'],
    specialized_calc_guard=False,
    same_frozen_bank=True, test_writes=0,
    samples=dict(calc=7, impress=6, total=13),
    primary_input='paired stable structured current state and up to four executed actions; screenshots archived but excluded because render freshness is unverified',
    history_summaries='generated from pre-action observation and earlier executed prefix only',
    candidate_comparison='summary-only; no next actions, private phase labels, outcomes, scores or task gold',
    primary_metrics=['top1_correct','no_incompatible_selected','coverage','selected_precision'],
    coverage_definition='coverage among queries with a matching historical phase; negative-case correct abstention is separate',
    cost_metrics=['model_calls','wall_seconds'],
    gate='Advance to agent success-rate comparison only if top1 improves in both application families and no-incompatible-selected improves overall without lower coverage.',
    limitations='Two known task families, five reused synthetic cases, one model draw. No blind generalization or statistical superiority claim.',
)


def phase_label(observation, private):
    """Evaluation only, using measured values plus private fixture semantics."""
    state = parse_structured(observation.get('app_info')) or {}
    if private['family']=='calc':
        active = state.get('active_sheet',state.get('name'))
        if active == private['target']:
            return 'calc-target'
        if active == private['source']:
            return 'calc-source'
        return 'calc-other'
    page = next((p for p in state.get('slides',[]) if p['slide_index']==private['slide']),None)
    if page is None:
        return 'unknown'
    objects = {o['box_index']:o for o in page['text_objects']}
    flags=[]
    for box,size in zip(private['boxes'],private['sizes']):
        measured=objects.get(box,{}).get('font_sizes_pt',[])
        if not measured:
            return 'unknown'
        flags.append(all(abs(v-size)<.1 for v in measured))
    return 'impress-font-progress:' + ','.join(str(int(v)) for v in flags)


def step_from_dict(value):
    return EpisodeStep(value['step_index'],value['observation'],value['raw_action'],
        value.get('next_observation'),value.get('note'),value.get('action_success',True))


class AuditedTransport:
    def __init__(self, transport, root):
        self.transport,self.root=transport,root
        self.seconds=0.
        self.calls=0
        (root/'responses').mkdir(exist_ok=True)

    def __call__(self,messages):
        signature=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        path=self.root/'responses'/f'{signature}.json'
        if path.exists():
            return json.loads(path.read_text())['response']
        started=time.monotonic()
        result=self.transport(messages)
        seconds=time.monotonic()-started
        self.calls+=1
        self.seconds+=seconds
        write_json(path,dict(request_sha256=signature,request=messages,response=result,seconds=seconds))
        return result


def audit(root: Path, runtime: Path, api_key: str):
    write_json(root/'analysis-status.json',dict(state='building-bank'))
    library=TemplateLibrary.from_dict(json.loads((runtime/'original-bank/memory/templates.json').read_text()))
    memory=EvoTreeMemory(root/'bank',OSWorldDistiller(template_library=library),strict_quality=False)
    history_rows=[]
    # Keep the third, unrelated OS seed to preserve the historical source set.
    os_seed=json.loads((runtime/'sources/os.json').read_text())
    memory.write_episode(Episode(os_seed['instruction'],tuple(step_from_dict(s) for s in os_seed['steps']),
                                 os_seed['success'],os_seed['source']))
    probes=[]
    for family in ('calc','impress'):
        folder=root/family
        seed=json.loads((folder/'history.json').read_text())
        assert seed['score']==1.0, 'Historical replay must objectively succeed'
        episode=Episode(seed['instruction'],tuple(step_from_dict(s) for s in seed['steps']),True,
                        str(folder/'history.json'))
        memory.write_episode(episode)
        history_rows.append((episode,seed['private']))
        probes.extend(json.loads((folder/'probes.json').read_text()))
    memory.save()
    assert len(probes)==13
    frozen_hashes=hashes(root/'bank')
    write_json(root/'bank-hashes.json',frozen_hashes)
    transport=AuditedTransport(JsonChatTransport('http://llmapi.evolventapi.com/v1',
        'gpt-5.6-terra',api_key,timeout=180),root)
    backend=ModelSituationBackend(transport)
    frozen=EvoTreeMemory(root/'bank',OSWorldDistiller(template_library=library),write_enabled=False,
                         strict_quality=False,situation_backend=backend)
    labels={}
    for episode,private in history_rows:
        template=frozen.lookup_task(episode.instruction).template_id
        tree=frozen.forest.get_tree(template)
        for step in episode.steps:
            node=tree.node_for_signature(frozen.distiller.distill_state(step.observation))
            labels.setdefault((template,node.node_id),set()).add(phase_label(step.observation,private))
    rows=[]
    for index,probe in enumerate(probes):
        obs=probe['observation']
        assert obs.get('observation_provenance',{}).get('status')=='stable'
        truth=phase_label(obs,probe['private'])
        per={}
        for condition in PLAN['conditions']:
            started=time.monotonic()
            result=frozen.read_dual_cheatsheet(probe['instruction'],obs,
                enforce_state_conditions=False,use_situation_summaries=condition=='dual_summary')
            selected=[dict(template=item.template_id,node=item.node_id,
                action=item.recommendation.action.name,
                labels=sorted(labels.get((item.template_id,item.node_id),set())),
                summary=item.situation_summary,comparison=item.summary_match)
                for item in result.state_experiences]
            correctness=[truth in row['labels'] and truth!='unknown' for row in selected]
            # An empty response is a correct abstention ONLY where no historical
            # state has the measured private phase (e.g. the Scratch sheet).
            has_match=any(truth in possible for possible in labels.values()) and truth!='unknown'
            per[condition]=dict(selected=selected,selected_count=len(selected),
                top1_correct=bool(correctness and correctness[0]),coverage=bool(selected),
                no_incompatible_selected=(all(correctness) if selected else not has_match),
                correct_count=sum(correctness),has_matching_history=has_match,
                correct_abstention=not selected and not has_match,
                current_summary=result.current_situation,summary_comparisons=result.summary_comparisons,
                rejected=result.rejected_states,seconds=time.monotonic()-started)
        row=dict(id=probe['id'],family=probe['private']['family'],truth=truth,conditions=per)
        rows.append(row)
        write_json(root/'analysis-status.json',dict(state='running',completed=len(rows),planned=len(probes),
            model_calls=transport.calls,model_seconds=transport.seconds))
        write_json(root/'paired-results.json',rows)
        assert hashes(root/'bank')==frozen_hashes
    totals={}
    for family in ('calc','impress','all'):
        subset=[r for r in rows if family=='all' or r['family']==family]
        totals[family]={}
        for condition in PLAN['conditions']:
            metrics=[r['conditions'][condition] for r in subset]
            totals[family][condition]=dict(n=len(metrics),
                top1_correct=sum(x['top1_correct'] for x in metrics),
                no_incompatible_selected=sum(x['no_incompatible_selected'] for x in metrics),
                coverage=sum(x['coverage'] and x['has_matching_history'] for x in metrics),
                raw_coverage=sum(x['coverage'] for x in metrics),
                positive_n=sum(x['has_matching_history'] for x in metrics),
                correct_abstention=sum(x['correct_abstention'] for x in metrics),
                correct_selected=sum(x['correct_count'] for x in metrics),
                selected=sum(x['selected_count'] for x in metrics))
    gate=(all(totals[f]['dual_summary']['top1_correct']>totals[f]['dual_unfiltered']['top1_correct']
              for f in ('calc','impress'))
          and totals['all']['dual_summary']['no_incompatible_selected']>
              totals['all']['dual_unfiltered']['no_incompatible_selected']
          and totals['all']['dual_summary']['coverage']>=totals['all']['dual_unfiltered']['coverage'])
    result=dict(completed=True,samples=len(rows),totals=totals,advance_gate_passed=gate,
        model_calls=transport.calls,model_seconds=transport.seconds,bank_unchanged=True,
        limitations=PLAN['limitations'])
    write_json(root/'summary.json',result)
    write_json(root/'analysis-status.json',dict(state='complete',completed=len(rows),planned=len(rows)))
    return result

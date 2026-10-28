"""Independent task workflow and horizontal state retrieval over one bank.

Workflows preserve measured order; state advice retains per-task statistics.
No test labels, parameter binding oracle, or model calls are used here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from collections import deque
import json
from typing import Any

from .retrieval import Cheatsheet, Recommendation, Retriever
from .types import ActionKind, ActionTmpl, StateSig
from .state_conditions import COPY_TOOL, current_conditions, historical_conditions, unmet_conditions
from .situation_summary import (SituationSummary, SituationResponseError,
                                SituationTransportError, previous_action)


def action_stage(action: ActionTmpl, ok: bool = True) -> str:
    """Parameter-independent observed stage, not an assertion of task progress."""
    return json.dumps([action.name, list(action.arg_slots), ok], separators=(",", ":"))


@dataclass
class WorkflowStep:
    action: ActionTmpl
    state: StateSig
    stage: str
    note: str | None = None
    observation: dict = field(default_factory=dict)
    situation_summary: dict | None = None


@dataclass
class TaskExperience:
    template_id: str
    instruction: str
    steps: list[WorkflowStep]
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> TaskExperience:
        steps = []
        for row in value["steps"]:
            action = dict(row["action"])
            action["kind"] = ActionKind(action["kind"])
            action["arg_slots"] = tuple(action.get("arg_slots", ()))
            steps.append(WorkflowStep(ActionTmpl(**action), StateSig(**row["state"]),
                                      row["stage"], row.get("note"),
                                      dict(row.get("observation", {})), row.get("situation_summary")))
        return cls(value["template_id"], value["instruction"], steps, value.get("source"))


@dataclass
class StateExperience:
    template_id: str
    node_id: str
    state: StateSig
    stage: str
    match: str
    recommendation: Recommendation
    notes: list[str] = field(default_factory=list)
    preconditions: dict = field(default_factory=dict)
    situation_summary: dict | None = None
    summary_match: dict = field(default_factory=dict)


@dataclass
class DualCheatsheet(Cheatsheet):
    task_experiences: list[TaskExperience] = field(default_factory=list)
    state_experiences: list[StateExperience] = field(default_factory=list)
    # Populated by rendering: an omitted workflow must never look complete.
    omitted: list[str] = field(default_factory=list)
    observed_conditions: dict = field(default_factory=dict)
    rejected_states: list[dict] = field(default_factory=list)
    current_situation: dict | None = None
    summary_comparisons: list[dict] = field(default_factory=list)
    uses_summaries: bool = False
    recall_candidates: list[dict] = field(default_factory=list)
    summary_failures: list[dict] = field(default_factory=list)
    # Rendering audit: these counts are set from blocks actually admitted to
    # the prompt, so an accepted candidate cannot be confused with omission.
    rendered_task_records: int = 0
    rendered_state_records: int = 0
    task_comparisons: list[dict] = field(default_factory=list)
    task_relevance_failures: list[dict] = field(default_factory=list)

    @property
    def is_cold_start(self) -> bool:
        return not self.task_experiences and not self.state_experiences


def state_match(current: StateSig, historical: StateSig, stage: str,
                historical_stages: set[str], *, relax_action_stage: bool = False) -> str | None:
    # The synthetic task-start root carries no cross-task environmental evidence.
    if current.app.lower() in {"", "unknown", "osworld-task-start"}:
        return None
    if current.app != historical.app or current.window != historical.window:
        return None
    if (current.dialog or "") != (historical.dialog or ""):
        return None
    if any(key in current.facts and current.facts[key] != value
           for key, value in historical.facts.items()):
        return None
    if current.key() == historical.key() and (current.window or current.context):
        return "exact-state"
    # Never erase opaque context hashes and match just an application name.
    # When parameter values change, require independent stable stage evidence.
    if stage and stage != "start" and stage in historical_stages:
        return "same-app-window-dialog-and-action-stage; historical parameters differ"
    # Opaque context hashes differ across tasks; semantic verification is
    # responsible for alignment after these explicit environment checks.
    if relax_action_stage and current.window:
        return "same-app-window-dialog; action-stage differs; semantic verification required"
    return None


def retrieve_dual(memory: Any, instruction: str, observation: dict,
                  candidate_tools: list[str] | None = None, *,
                  task_enabled: bool = True, state_enabled: bool = True,
                  enforce_state_conditions: bool = True,
                  use_situation_summaries: bool = False,
                  relax_action_stage: bool = True,
                  require_summary_compatible: bool = True,
                  require_compatible_all: bool = True,
                  task_relevance_backend: Any = None) -> DualCheatsheet:
    # Clone the template library so frozen queries cannot generalize/register
    # templates in RAM and thereby change subsequent task assignments.
    template = memory.lookup_task(instruction)
    sheet = DualCheatsheet(reranked_tools=list(candidate_tools or []), uses_summaries=use_situation_summaries)
    if use_situation_summaries and memory.situation_backend is None:
        raise ValueError('Summary retrieval requires an explicit situation_backend')
    if task_enabled:
        histories = memory.task_experiences.get(template.template_id, [])
        sheet.task_experiences = sorted(histories, key=lambda h: len(h.steps))[:1]
        if not sheet.task_experiences and task_relevance_backend is not None:
            candidates = [min(histories, key=lambda h: len(h.steps))
                          for _, histories in sorted(memory.task_experiences.items()) if histories]
            # Never silently truncate a large library by template-ID order.
            if len(candidates) > task_relevance_backend.max_candidates:
                sheet.task_relevance_failures.append(dict(reason='candidate-budget-exceeded',
                    candidates=len(candidates)))
            elif candidates:
                try:
                    judgments = task_relevance_backend.rank(instruction, candidates)
                    sheet.task_comparisons = [dict(template=history.template_id, **judgment)
                                             for history, judgment in zip(candidates, judgments)]
                    related = [(history, judgment) for history, judgment in zip(candidates, judgments)
                               if judgment['verdict'] == 'related']
                    related.sort(key=lambda pair: (-pair[1]['score'], len(pair[0].steps), pair[0].template_id))
                    sheet.task_experiences = [history for history, _ in related[:1]]
                except (SituationResponseError, SituationTransportError) as exc:
                    sheet.task_relevance_failures.append(dict(error_type=type(exc).__name__))
    current = memory.distiller.distill_state(observation)
    stage = str(observation.get("state_stage", ""))
    if not use_situation_summaries:
        sheet.observed_conditions = current_conditions(instruction, observation)
    provenance = observation.get('observation_provenance') or {}
    if state_enabled and provenance.get('status') not in {'unstable', 'unavailable'}:
        # Derived index: all historical task trees, independent of task match.
        stages: dict[tuple, set[str]] = {}
        conditions: dict[tuple, list[dict]] = {}
        contexts: dict[tuple, list[tuple]] = {}
        for histories in memory.task_experiences.values():
            for history in histories:
                requirements = ([{} for _ in history.steps] if use_situation_summaries
                                else historical_conditions(history))
                for index, (step, required) in enumerate(zip(history.steps, requirements)):
                    stages.setdefault((history.template_id, step.state.key()), set()).add(step.stage)
                    key = (history.template_id, step.state.key(), step.action.key())
                    conditions.setdefault(key, []).append(required)
                    contexts.setdefault(key, []).append((history, index))
        # Frozen retrieval never uses UCB exploration. Quality options remain
        # identical across source tasks; counts are never pooled across trees.
        retriever = Retriever(gamma=memory.retriever.gamma, top_k=memory.retriever.top_k,
                              explore=False, strict_quality=memory.retriever.strict_quality)
        matches = []
        for tree in memory.forest.all_trees():
            for node in tree.all_nodes():
                historical_stages = stages.get((tree.template_id, node.sig.key()), set())
                match = state_match(current, node.sig, stage, historical_stages,
                                    relax_action_stage=use_situation_summaries and relax_action_stage)
                if match is None:
                    continue
                for rec in retriever._build_recommendations(tree, node.node_id, node.visits):
                    key = (tree.template_id, node.sig.key(), rec.action.key())
                    alternatives = conditions.get(key, [{}])
                    # A merged node may have multiple historical situations.
                    # Retain a candidate only with an individually satisfied
                    # evidence record; do not intersect away conflicting guards.
                    satisfied = []
                    for condition in alternatives:
                        required = dict(condition)
                        if rec.action.name == COPY_TOOL and not use_situation_summaries:
                            required["active_sheet_role"] = "source"
                        if (use_situation_summaries or not enforce_state_conditions
                                or not unmet_conditions(required, sheet.observed_conditions)):
                            satisfied.append(required)
                    if not satisfied:
                        sheet.rejected_states.append(dict(template=tree.template_id, node=node.node_id,
                            action=rec.action.name, requirements=alternatives,
                            reason="observed precondition missing or conflicting"))
                        continue
                    required = max(satisfied, key=len)
                    edge = next(e for e in tree.out_edges(node.node_id) if e.action is rec.action)
                    matches.append(StateExperience(tree.template_id, node.node_id, node.sig,
                                                   stage if stage in historical_stages else "", match,
                                                   rec, list(edge.notes), required))
        matches.sort(key=lambda item: (item.match != "exact-state",
                                      not bool(item.stage) if use_situation_summaries else False,
                                      -item.recommendation.success_rate,
                                      -item.recommendation.n, item.template_id, item.node_id))
        sheet.recall_candidates = [dict(template=item.template_id, node=item.node_id,
            action=item.recommendation.action.name, match=item.match) for item in matches]
        if use_situation_summaries and matches:
            try:
                matches = _refine_situations(memory, instruction, observation, matches, contexts, sheet,
                    require_compatible=require_summary_compatible,
                    require_compatible_all=require_compatible_all)
            except (SituationResponseError, SituationTransportError) as exc:
                # Optional model-based verification fails closed. Keep the
                # independent task workflow, but expose no unverified state advice.
                sheet.summary_failures.append(dict(stage='state-verification',
                    error_type=type(exc).__name__, candidates=len(matches)))
                sheet.current_situation = None
                sheet.summary_comparisons.clear()
                sheet.rejected_states.extend(dict(template=item.template_id, node=item.node_id,
                    action=item.recommendation.action.name,
                    reason='summary-verification-unavailable') for item in matches)
                matches = []
        # Keep multiple task sources when available, rather than filling all
        # slots with recommendations from the first matching tree.
        chosen = []
        for item in matches:
            if item.template_id not in {s.template_id for s in chosen}:
                chosen.append(item)
        chosen += [item for item in matches if all(item is not s for s in chosen)]
        sheet.state_experiences = chosen[:memory.retriever.top_k]
        sheet.recommendations = [s.recommendation for s in sheet.state_experiences]
    caution_tools = set(candidate_tools or [])
    caution_tools.update(r.action.name if r.action.kind == ActionKind.TOOL
                         else r.action.name.split(".", 1)[0] for r in sheet.recommendations)
    sheet.cautions = memory.retriever._build_cautions(memory.notes, caution_tools)
    # Keep tool ordering fixed for clean task/state ablations.
    return sheet


def _bounded_contexts(matches, contexts, limit):
    """Round-robin sources and nodes before revisiting a merged node's contexts.

    Bound expensive summary work, not just the final comparison request. The
    incoming exact/stage ordering is evidence within each source, never a gate.
    Neither candidate actions nor private labels determine this allocation.
    """
    groups = {}
    for item in matches:
        key = (item.template_id, item.state.key(), item.recommendation.action.key())
        records = [(item, history, index) for history, index in contexts.get(key, [])]
        if records:
            groups.setdefault(item.template_id, {}).setdefault(item.node_id, []).extend(records)
    sources = deque(deque(deque(records) for records in nodes.values()) for nodes in groups.values())
    chosen = []
    while sources and len(chosen) < limit:
        nodes = sources.popleft()
        records = nodes.popleft()
        chosen.append(records.popleft())
        if records:
            nodes.append(records)
        if nodes:
            sources.append(nodes)
    return chosen


def _refine_situations(memory, instruction, observation, matches, contexts, sheet, *,
                       require_compatible=True, require_compatible_all=False):
    """Use only pre-action evidence; preserve separate contexts of merged nodes."""
    backend = memory.situation_backend
    current = backend.summarize(instruction, observation, observation.get('executed_prefix', ()))
    sheet.current_situation = current.to_dict()
    records, summaries = [], []
    for item, history, index in _bounded_contexts(matches, contexts, backend.max_candidates):
        step = history.steps[index]
        if step.situation_summary is not None:
            historical = SituationSummary.from_dict(step.situation_summary)
        else:
            prefix = [previous_action(s.action, s.note is None) for s in history.steps[:index]]
            obs = step.observation or dict(app=step.state.app, window=step.state.window,
                                           dialog=step.state.dialog, facts=step.state.facts)
            historical = backend.summarize(history.instruction, obs, prefix)
        # One graph node can have multiple measured situations. Do not
        # attach the first context's sentence to every edge or pool roles.
        from dataclasses import replace
        records.append(replace(item, situation_summary=historical.to_dict()))
        summaries.append(historical)
    if not records:
        return []  # Legacy graph-only entries have no recoverable context.
    bounded = _bounded_contexts(matches, contexts, backend.max_candidates)
    # Include each task's own goal so differing numerical targets can be
    # compared by per-subgoal completion status rather than raw values.
    judgments = backend.compare(current, summaries, goals=dict(current=instruction,
        historical=[history.instruction for _, history, _ in bounded]))
    retained = []
    for item, judgment in zip(records, judgments):
        sheet.summary_comparisons.append(dict(template=item.template_id, node=item.node_id,
            action=item.recommendation.action.name, historical=item.situation_summary, **judgment))
        item.summary_match = judgment
        broadened = 'action-stage differs' in item.match
        needs_compatible = require_compatible_all or (require_compatible and broadened)
        if (judgment['verdict'] == 'conflict' or
                (needs_compatible and judgment['verdict'] != 'compatible')):
            sheet.rejected_states.append(dict(template=item.template_id, node=item.node_id,
                action=item.recommendation.action.name, reason='summary-context-' + judgment['verdict'],
                explanation=judgment['reason']))
        else:
            retained.append(item)
    retained.sort(key=lambda item: (item.summary_match['verdict'] != 'compatible',
                                    -item.summary_match['score']))
    return retained


ADAPTATION = (
    "Historical examples only. Derive ALL current files, sheets, ranges, slide/box indices, "
    "text and numbers from the CURRENT task and observation. Locate GUI targets again. "
    "Task workflows describe ordering; state matches are candidate operations, not proof "
    "of current progress. Check preconditions and observed results before each action.\n"
)


def _action(action: ActionTmpl, args: dict | None = None) -> str:
    return action.name + " " + json.dumps(dict(action.args) if args is None else args,
                                         ensure_ascii=False, sort_keys=True)


def _related_workflow_lines(history: TaskExperience) -> list[str]:
    """Keep operation order without turning historical arguments into bindings.

    Only the optional cross-template reference uses this outline. Consecutive
    repetitions remain visible as source spans, never as a current loop count.
    """
    lines = ['Historical procedure outline; source step numbers are provenance, '
             'not current target counts. Bind every argument from the CURRENT task '
             'and observation; verify effects instead of assuming completion.']
    index = 0
    while index < len(history.steps):
        step = history.steps[index]
        keys = tuple(sorted(set(step.action.arg_slots) | set(step.action.args)))
        end = index + 1
        # Do not merge across measured app/window/dialog changes or cautions.
        while end < len(history.steps):
            next_step = history.steps[end]
            next_keys = tuple(sorted(set(next_step.action.arg_slots) | set(next_step.action.args)))
            if (step.action.kind != ActionKind.TOOL or next_step.action.kind != step.action.kind
                    or next_step.action.name != step.action.name or next_keys != keys
                    or (next_step.state.app, next_step.state.window, next_step.state.dialog)
                    != (step.state.app, step.state.window, step.state.dialog)
                    or step.note or next_step.note):
                break
            end += 1
        source = str(index + 1) if end == index + 1 else f'{index + 1}-{end}'
        if step.action.kind == ActionKind.GUI:
            operation = (step.action.name.split('.', 1)[0] +
                         ' (historical GUI step; locate and choose the current control again)')
        else:
            operation = _action(step.action, {key: f'<CURRENT:{key}>' for key in keys})
        condition = f'app={step.state.app}, window={step.state.window}, dialog={step.state.dialog}'
        line = f'Source steps {source}: {operation}; observed before: {condition}'
        if end > index + 1:
            line += '; repeated historically; derive targets and repetition count from the CURRENT task'
        if step.note:
            line += '; historical caution: ' + step.note
        lines.append(line)
        index = end
    return lines


def render_dual_cheatsheet(sheet: DualCheatsheet, budget: int = 6000) -> str:
    """Independent reserved budgets; omit whole records, never cut arguments."""
    if budget < 1000:
        raise ValueError("Dual retrieval requires a budget of at least 1000 characters")
    sheet.omitted.clear()
    sheet.rendered_task_records = 0
    sheet.rendered_state_records = 0
    available = budget - len(ADAPTATION) - 6
    task_budget, state_budget = int(available * .55), int(available * .35)
    caution_budget = available - task_budget - state_budget
    sections = []
    task_lines = ["[任务级经验 / Task-level experience]"]
    for history in sheet.task_experiences:
        relevance = next((row for row in sheet.task_comparisons
                          if row['template'] == history.template_id and row['verdict'] == 'related'), None)
        if relevance is not None:
            lines = [f'Source task: {history.template_id}',
                         'Related workflow only; this is not the same task template or evidence '
                         'of current progress. Adapt object scope, repetitions and ALL arguments. '
                         'Relevance: ' + relevance['reason']]
            lines.extend(_related_workflow_lines(history))
        else:
            lines = [f"Source task: {history.template_id}; historical instruction: {history.instruction}",
                     "Successful historical sequence (preserve dependencies; adapt examples):"]
            for index, step in enumerate(history.steps, 1):
                condition = f"app={step.state.app}, window={step.state.window}, dialog={step.state.dialog}"
                lines.append(f"{index}. {_action(step.action)}; observed before: {condition}"
                             + (f"; caution: {step.note}" if step.note else ""))
        block = "\n".join(lines)
        if len("\n".join(task_lines + [block])) <= task_budget:
            task_lines.append(block)
            sheet.rendered_task_records += 1
        else:
            sheet.omitted.append(f"task:{history.template_id}:full-workflow-exceeds-budget")
    if len(task_lines) > 1:
        sections.append("\n".join(task_lines))
    state_lines = ["[当前状态经验 / Horizontal state experience]"]
    if sheet.current_situation is not None:
        state_lines.append('Current situation (model interpretation): ' +
                           _summary_text(sheet.current_situation))
    for item in sheet.state_experiences:
        rec = item.recommendation
        samples = [s for s in rec.arg_samples if s.get("n_success", 0) > 0]
        args = dict(samples[0]["args"]) if samples else dict(rec.action.args)
        lines = [f"Source task: {item.template_id}; node={item.node_id}; match={item.match}",
                 f"Condition: app={item.state.app}, window={item.state.window}, dialog={item.state.dialog}; "
                 f"prior-action-stage={item.stage or 'unavailable'}; facts={json.dumps(item.state.facts, ensure_ascii=False)}",
                 f"Candidate: {_action(rec.action, args)}"]
        if item.situation_summary is not None:
            lines.append('Historical situation (model interpretation): ' +
                         _summary_text(item.situation_summary))
            lines.append('Semantic comparison: ' + json.dumps(item.summary_match, ensure_ascii=False))
        if item.preconditions:
            lines.append("Verified current preconditions: " +
                         json.dumps(item.preconditions, ensure_ascii=False, sort_keys=True))
        lines.extend([
                 f"Within-source task-success rate={rec.success_rate:.2f} over {rec.n} visits; "
                 + ("low sample confidence." if rec.low_confidence else "observational evidence.")])
        if rec.lookahead_hint:
            lines.append("Only AFTER observing completion of candidate: " + rec.lookahead_hint)
            if not sheet.uses_summaries and rec.lookahead is not None and rec.lookahead.name == COPY_TOOL:
                lines.append("Before that dependent copy, re-observe and confirm the active sheet has the SOURCE role.")
        if item.notes:
            lines.append("Historical failures/cautions: " + "; ".join(item.notes))
        block = "\n".join(lines)
        if len("\n".join(state_lines + [block])) <= state_budget:
            state_lines.append(block)
            sheet.rendered_state_records += 1
        else:
            sheet.omitted.append(f"state:{item.template_id}:{item.node_id}:exceeds-budget")
    if len(state_lines) > 1:
        sections.append("\n".join(state_lines))
    caution_lines = ["[注意 / Cautions]"]
    for caution in sheet.cautions:
        if len("\n".join(caution_lines + [caution])) <= caution_budget:
            caution_lines.append(caution)
    if len(caution_lines) > 1:
        sections.append("\n".join(caution_lines))
    text = ADAPTATION + "\n\n".join(sections) if sections else ""
    assert len(text) <= budget
    return text


def _summary_text(value):
    text = value['sentence']
    if value.get('unknowns'):
        text += ' Unknown: ' + '; '.join(value['unknowns'])
    return text

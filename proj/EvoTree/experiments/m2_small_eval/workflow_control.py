"""Frozen same-task seeds: no memory vs plain workflow vs state retrieval.

All conditions use the same VM, model, reset/evaluator protocol, and repeat
orders. Both memory conditions derive from the SAME measured success episode.
The character cap is identical; actual prompt lengths are reported, not padded
or falsely described as token-matched. This is not a transfer benchmark.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
import logging
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

from experiments.m2_small_eval.diagnostic_pilot import (
    TASKS, append, hashes, load_package, run_episode, stage_cached_assets, write_json,
)


ORDERS = [("baseline", "workflow", "tree"), ("workflow", "tree", "baseline"),
          ("tree", "baseline", "workflow")]
COMMON = (
    "[历史经验 / Prior experience]\n"
    "One successful prior episode on this same task; this is low-confidence advisory evidence. "
    "Use the CURRENT observation to choose and locate actions; old screen coordinates are not reusable. "
    "Do not assume a suggested operation already happened or infer success from its description.\n"
)


def bounded(text, budget):
    if len(text) > budget:
        raise ValueError(f"Advice needs {len(text)} characters, exceeding fixed budget {budget}; refusing to truncate executable values")
    return text


def render_workflow(episode, distiller, budget):
    """Keep measured operation order/arguments, remove graph/statistics lookup."""
    lines = [COMMON, "Previously successful sequence (locate your progress using the current observation):"]
    for index, step in enumerate(episode["steps"], 1):
        action = distiller.distill_action(step["raw_action"])
        args = dict(action.args)
        line = f"{index}. {action.name}"
        if args:
            line += " " + json.dumps(args, ensure_ascii=False, sort_keys=True)
        lines.append(line)
    return bounded("\n".join(lines), budget)


def _example(edge):
    samples = [sample for sample in edge.arg_samples if sample.get("n_success", 0) > 0]
    return dict(samples[0]["args"]) if samples else dict(edge.action.args)


def render_tree(sheet, memory, instruction, style, budget):
    """Original rendering or explicit current/dependent-next separation."""
    if not sheet.recommendations and not sheet.cautions:
        return ""
    if style == "original":
        # The package-specific renderer is supplied by the caller. The source
        # memory is current-code, so no improved arg expansion is smuggled in.
        body = memory._diagnostic_renderer(sheet)
    else:
        lines = []
        tree = memory.forest.get_tree(memory.distiller.distill_task(instruction).template_id)
        for rec in sheet.recommendations:
            samples = [s for s in rec.arg_samples if s.get("n_success", 0) > 0]
            args = dict(samples[0]["args"]) if samples else dict(rec.action.args)
            lines.append("CURRENT ACTION: " + rec.action.name + " " + json.dumps(args, ensure_ascii=False, sort_keys=True))
            lines.append(f"Observed count={rec.n}, task-success rate={rec.success_rate:.2f}, "
                f"mean remaining actions={rec.mean_steps_to_go:.0f}; low sample confidence.")
            if rec.lookahead is not None:
                lines.append("DEPENDENT NEXT ACTION: " + rec.lookahead.name + ". "
                    "Only consider this AFTER completing the current action and observing its result.")
                # Retrieve successor parameter evidence from the same connected
                # seed, never a hand-authored task solution.
                candidates = [edge for edge in tree.all_edges() if edge.action.key() == rec.lookahead.key()
                              and edge.n_success > 0]
                # The lookahead object originates from an actual edge. Exact
                # argument equality disambiguates same tool at different states.
                selected = next((e for e in candidates if e.action is rec.lookahead), None)
                if selected is not None:
                    lines.append("Previously successful next arguments: " + json.dumps(_example(selected), ensure_ascii=False, sort_keys=True))
        if sheet.cautions:
            lines.extend(["Cautions:"] + list(sheet.cautions))
        body = "\n".join(lines)
    return bounded(COMMON + body, budget)


class WorkflowAgent:
    """Same underlying agent, fixed plain sequence prepended every decision."""

    def __init__(self, agent, text):
        self.agent, self.text = agent, text

    def __getattr__(self, name):
        return getattr(self.agent, name)

    def predict(self, instruction, obs, *args, **kwargs):
        return self.agent.predict(instruction.rstrip() + "\n\n" + self.text, obs, *args, **kwargs)


def worker(args):
    logging.basicConfig(level=logging.INFO)
    domain, task, _ = TASKS[args.task_index]
    output = args.output_root / f"task-{args.task_index}"
    output.mkdir(parents=True, exist_ok=False)
    status = output / "status.json"
    source = args.seed_root / f"task-{args.task_index}"
    verified = json.loads((source / "seed-verified.json").read_text())
    assert verified["score"] == 1.0 and verified["equal_empirical_data"]
    episode = json.loads((source / "seed-episode.json").read_text())
    assert episode["success"] is True
    current = load_package("evotree_workflow_current", args.current_code)
    if args.tree_rendering == "original":
        package = current
        seed_path = source / "seed-current"
        original_renderer = importlib.import_module("evotree_workflow_current.retrieval").render_cheatsheet
    else:
        package = SimpleNamespace(memory=importlib.import_module("evotree.memory"),
            distill_osworld=importlib.import_module("evotree.distill_osworld"),
            osworld_integration=importlib.import_module("evotree.osworld_integration"))
        seed_path = source / "seed-improved"
        original_renderer = importlib.import_module("evotree.retrieval").render_cheatsheet
    before = hashes(seed_path)
    expected = json.loads((source / "seed-hashes-before.json").read_text())[
        "current" if args.tree_rendering == "original" else "improved"]
    assert before == expected
    text = render_workflow(episode, package.distill_osworld.OSWorldDistiller(), args.prompt_budget_chars)
    (output / "plain-workflow.txt").write_text(text)
    write_json(output / "provenance.json", dict(source=str(source), seed_hashes=before,
        same_success_episode=True, tree_rendering=args.tree_rendering, budget_chars=args.prompt_budget_chars,
        workflow_chars=len(text), token_matched=False, setting="same-task frozen demonstration control"))
    sys_path = str(args.runner_root)
    import sys
    sys.path.insert(0, sys_path)
    from experiments.m2_small_eval.run_arm import runtime_environment
    os.environ.update(runtime_environment(os.environ, f"evo-rh-workflow-20261015-t{args.task_index}", 9 + args.task_index))
    os.environ["EVOTREE_RETRIEVAL_LOG"] = "0"
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
    config = json.loads((args.runner_root / f"evaluation_examples/examples/{domain}/{task}.json").read_text())
    write_json(output / "cached-assets.json", stage_cached_assets(config, args.asset_cache_root, Path.cwd() / "cache"))
    write_json(status, dict(state="booting", task=task))
    desktop = None
    results = []
    try:
        desktop = DesktopEnv(action_space="mcp", headless=True, provider_name="docker", os_type="Ubuntu",
            screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original_evaluate,
            log=lambda event: append(output / "save-barrier.jsonl", event))
        for repetition in range(args.repeats):
            conditions = ("tree",) if args.smoke_tree_only else ORDERS[(repetition + args.task_index) % 3]
            for condition in conditions:
                write_json(status, dict(state="running", task=task, completed=len(results), planned=args.repeats * 3,
                    condition=condition, repetition=repetition + 1))
                agent = OwlAgent(model=args.model, api_url=args.api_url, api_key=os.environ["OSWORLD_API_KEY"],
                    history_n=3, action_space="mcp", observation_type="screenshot", max_trajectory_length=15,
                    top_p=0.9, temperature=1.0)
                env, runtime = desktop, None
                if condition == "workflow":
                    agent = WorkflowAgent(agent, text)
                elif condition == "tree":
                    memory = package.memory.EvoTreeMemory(seed_path, package.distill_osworld.OSWorldDistiller(),
                        write_enabled=False, explore=False, strict_quality=False)
                    memory._diagnostic_renderer = original_renderer
                    agent, env, runtime = package.osworld_integration.wrap_osworld(agent, desktop, memory)
                    original_before = runtime.before_model

                    def before_model(observation, names, memory=memory, original_before=original_before):
                        result = original_before(observation, names)
                        body = render_tree(result.cheatsheet, memory, config["instruction"], args.tree_rendering, args.prompt_budget_chars)
                        # Tool schema order is held fixed across all conditions.
                        # This comparison tests advice organization, not reranking.
                        return replace(result, text=body, reranked_tools=list(names))

                    runtime.before_model = before_model
                started = time.monotonic()
                directory = output / f"rep-{repetition+1}" / condition
                result = run_episode(agent, env, config, directory, runtime=runtime)
                decisions = [json.loads(line) for line in (directory / "decisions.jsonl").read_text().splitlines()]
                advice_lengths = [len(row["text"]) for row in decisions] if condition == "tree" else [len(text) if condition == "workflow" else 0] * len(decisions)
                if condition == "workflow":
                    write_json(directory / "workflow-injection.json", dict(text=text, chars=len(text), decisions=len(decisions)))
                row = dict(task=task, condition=condition, repetition=repetition+1, score=result["score"],
                    decisions=result["decisions"], seconds=time.monotonic()-started,
                    advice_chars=advice_lengths, prompt_budget_chars=args.prompt_budget_chars)
                results.append(row)
                append(output / "results.jsonl", row)
                assert hashes(seed_path) == before, "Read-only source seed changed"
        final = dict(state="complete", task=task, completed=len(results), frozen_memory_unchanged=True)
    except Exception as exc:
        (output / "exception.txt").write_text(traceback.format_exc())
        final = dict(state="failed", task=task, completed=len(results), error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        if desktop is not None:
            pids = list(desktop.provider._owned_qemu)
            desktop.close()
            final["closed_qemu_pids"] = pids
        write_json(status, final)
    return 0 if final["state"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "seed-root", "output-root", "asset-cache-root", "current-code"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--task-index", type=int, choices=range(3), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--tree-rendering", choices=("original", "ordered"), required=True)
    parser.add_argument("--prompt-budget-chars", type=int, default=3000)
    parser.add_argument("--smoke-tree-only", action="store_true", help="Execute one selected tree condition as a separate VM smoke")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

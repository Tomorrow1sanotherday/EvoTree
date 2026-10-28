"""Three-task, three-condition frozen diagnostic using identical seed episodes.

The saved successful replies are replayed in fresh VMs to build seeds. These
are warm-start diagnostics, never results of cold-start online evolution.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from types import SimpleNamespace
import uuid


TASKS = [
    ("os", "4d117223-a354-47fb-8b45-62ab1390a95f", 1),
    ("libreoffice_calc", "1273e544-688f-496b-8d89-3e0f40aa0606", 2),
    ("libreoffice_impress", "3161d64e-3120-47b4-aaad-6a764a92493b", 2),
]
ORDERS = [("baseline", "current", "improved"),
          ("current", "improved", "baseline"),
          ("improved", "baseline", "current")]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def append(path, data):
    with path.open("a") as stream:
        stream.write(json.dumps(data, ensure_ascii=False) + "\n")


def hashes(path):
    return {str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(path.rglob("*")) if p.is_file()}


def load_package(name, path):
    spec = importlib.util.spec_from_file_location(name, path / "evotree/__init__.py",
                                                submodule_search_locations=[str(path / "evotree")])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return SimpleNamespace(**{part: importlib.import_module(name + "." + part) for part in
        ("memory", "distill_osworld", "osworld_integration", "episode")})


def observation_metadata(mapped):
    # Structured application state is needed to make a history summary factual.
    # Screenshots stay in the episode artifacts instead of inflating the bank.
    return {key: mapped[key] for key in
            ("app", "window", "dialog", "facts", "state_context", "app_info", "a11y_tree",
             "observation_provenance")
            if key in mapped}


def stage_cached_assets(config, source_cache, destination_cache):
    """Copy immutable task inputs/gold files, never another run's result file."""
    task = config["id"]
    names = set()

    def visit(value):
        if isinstance(value, dict):
            if value.get("type") == "cloud_file":
                destinations = value.get("dest", [])
                names.update([destinations] if isinstance(destinations, str) else destinations)
            if value.get("type") == "download":
                for file in value.get("parameters", {}).get("files", []):
                    names.add(f"{uuid.uuid5(uuid.NAMESPACE_URL, file['url'])}_{Path(file['path']).name}")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(config)
    destination = destination_cache / task
    destination.mkdir(parents=True, exist_ok=True)
    staged = []
    for name in sorted(names):
        assert Path(name).name == name, "Cached asset must be a plain basename"
        source = source_cache / task / name
        if not source.is_file():
            raise FileNotFoundError(f"Required cached asset is missing: {source}")
        target = destination / name
        shutil.copy2(source, target)
        staged.append(dict(name=name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return staged


def run_episode(agent, env, config, directory, max_steps=15, runtime=None):
    directory.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger("diagnostic." + directory.name)
    agent.reset(logger)
    env.reset(task_config=config)
    time.sleep(20)
    obs = env._get_obs()
    (directory / "reset.png").write_bytes(obs["screenshot"])
    append(directory / "traj.jsonl", {"step_num": 0, "action": "reset", "instruction": config["instruction"], "screenshot_file": "reset.png"})
    episode_steps = []
    done = False
    for turn in range(1, max_steps + 1):
        before_model = time.monotonic()
        response, actions, messages = agent.predict(config["instruction"], obs)
        model_seconds = time.monotonic() - before_model
        if not actions:
            raise RuntimeError("Agent produced no executable actions")
        if runtime is not None:
            result = agent.last_before_model
            append(directory / "decisions.jsonl", dict(turn=turn, model_seconds=model_seconds,
                text=result.text, recommendations=[asdict(r) for r in result.cheatsheet.recommendations],
                cautions=result.cheatsheet.cautions, reranked_tools=result.reranked_tools,
                task_sources=[h.template_id for h in getattr(result.cheatsheet, "task_experiences", [])],
                state_sources=[dict(template=s.template_id, node=s.node_id, match=s.match)
                               for s in getattr(result.cheatsheet, "state_experiences", [])],
                omitted=list(getattr(result.cheatsheet, "omitted", [])), chars=len(result.text),
                observed_conditions=dict(getattr(result.cheatsheet, "observed_conditions", {})),
                current_situation=getattr(result.cheatsheet, 'current_situation', None),
                summary_comparisons=list(getattr(result.cheatsheet, 'summary_comparisons', [])),
                summary_failures=list(getattr(result.cheatsheet, 'summary_failures', [])),
                task_comparisons=list(getattr(result.cheatsheet, 'task_comparisons', [])),
                task_relevance_failures=list(getattr(result.cheatsheet, 'task_relevance_failures', [])),
                rendered_task_records=int(getattr(result.cheatsheet, 'rendered_task_records', 0)),
                rendered_state_records=int(getattr(result.cheatsheet, 'rendered_state_records', 0)),
                current_observation=result.current_observation,
                rejected_states=list(getattr(result.cheatsheet, "rejected_states", []))))
        else:
            append(directory / "decisions.jsonl", dict(turn=turn, model_seconds=model_seconds, text="", recommendations=[], cautions=[]))
        from evotree.current_observation import observation_record
        append(directory / 'observations.jsonl', dict(turn=turn, phase='before_model',
            **observation_record(obs)))
        for index, action in enumerate(actions):
            mapped_before = observation_metadata(agent.memory_observation(obs)) if runtime is not None else None
            started = time.monotonic()
            obs, reward, done, info = env.step(action, 1)
            append(directory / 'observations.jsonl', dict(turn=turn, phase='after_action',
                action_index=index, **observation_record(obs)))
            elapsed = time.monotonic() - started
            image_name = f"turn-{turn}-action-{index}.png"
            (directory / image_name).write_bytes(obs["screenshot"])
            raw = getattr(env, "last_raw_action", None) or {"prediction": response, "action": action}
            append(directory / "traj.jsonl", dict(step_num=turn, prediction=response, action=action,
                reward=reward, done=done, info=info, screenshot_file=image_name, messages=messages,
                tool_call=raw.get("tool_call"), gui_semantics=raw.get("gui_semantics"),
                exe_result=obs.get("exe_result"), action_seconds=elapsed))
            if runtime is not None and action not in ("DONE", "FAIL", "WAIT"):
                # Capture exactly the states used by the wrapper. No task
                # labels, screenshot descriptions, or answers are added to phi.
                from evotree.distill_osworld import extract_error_note
                call = raw.get("tool_call") or {}
                note = extract_error_note(info, obs, tool_name=call.get("name", ""))
                episode_steps.append(dict(step_index=len(episode_steps), observation=mapped_before,
                    raw_action={key: value for key, value in raw.items() if key != "messages"},
                    next_observation=observation_metadata(agent.memory_observation(obs)),
                    note=note, action_success=note is None))
            if done:
                break
        if done:
            break
    time.sleep(10)
    score = float(env.evaluate())
    (directory / "result.txt").write_text(str(score) + "\n")
    screenshot = env.controller.get_screenshot()
    if screenshot:
        (directory / "after-evaluator.png").write_bytes(screenshot)
    from experiments.m2_small_eval.parallel_pilot import validate_executed_actions
    rows = [json.loads(line) for line in (directory / "traj.jsonl").read_text().splitlines()]
    validate_executed_actions(rows, directory / "traj.jsonl")
    return dict(score=score, decisions=turn, stopped=done, episode_steps=episode_steps)


def worker(args):
    logging.basicConfig(level=logging.INFO)
    output = args.output_root / f"task-{args.task_index}"
    output.mkdir(parents=True, exist_ok=args.resume_verified_seed)
    status_path = output / "status.json"
    domain, task, shard = TASKS[args.task_index]
    write_json(status_path, dict(state="booting", task=task))
    sys.path[:0] = [str(args.improved_code), str(args.runner_root),
                   "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
    from experiments.m2_small_eval.run_arm import runtime_environment
    os.environ.update(runtime_environment(os.environ, f"evo-rh-diagnostic-20261015-t{args.task_index}", 6 + args.task_index))
    os.environ["EVOTREE_RETRIEVAL_LOG"] = "0"  # Whole frozen seed must remain unchanged.
    from evotree.memory import EvoTreeMemory
    from evotree.distill_osworld import OSWorldDistiller
    from evotree.osworld_integration import wrap_osworld
    from evotree.gui_semantics import VisionGUIAnnotator
    from evotree.episode import Episode, EpisodeStep
    from mm_agents.owl_agent import OwlAgent
    from desktop_env.desktop_env import DesktopEnv
    from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
    from filelock import FileLock
    from openai import OpenAI
    current = load_package("evotree_diagnostic_current", args.current_code)
    config = json.loads((args.runner_root / f"evaluation_examples/examples/{domain}/{task}.json").read_text())
    staged = stage_cached_assets(config, args.asset_cache_root, Path.cwd() / "cache")
    write_json(output / "cached-assets.json", staged)
    source = args.source_root / f"pass-2/shard-{shard}/baseline/results/mcp/screenshot/{args.model}/0/{domain}/{task}"
    assert float((source / "result.txt").read_text()) == 1.0
    source_rows = [json.loads(line) for line in (source / "traj.jsonl").read_text().splitlines() if line.strip()]
    predictions, seen = [], set()
    for row in source_rows:
        key = (row.get("step_num"), row.get("prediction"))
        if isinstance(key[1], str) and key not in seen:
            seen.add(key)
            predictions.append(key[1])
    write_json(output / "seed-provenance.json", dict(source=str(source), score=1.0,
        source_trajectory_sha256=hashlib.sha256((source / "traj.jsonl").read_bytes()).hexdigest(),
        source_predictions=predictions, setting="warm-start diagnostic; successful baseline replay"))
    client = OpenAI(base_url=args.api_url, api_key=os.environ["OSWORLD_API_KEY"], timeout=90, max_retries=1)

    def vision_request(messages):
        with FileLock(os.environ["OSWORLD_MODEL_REQUEST_LOCK"], timeout=600):
            reply = client.chat.completions.create(model=args.model, messages=messages,
                temperature=0, max_tokens=512, reasoning_effort="none")
        append(output / "annotation-cost.jsonl", dict(usage=reply.usage.model_dump() if reply.usage else None))
        return reply.choices[0].message.content

    class FixedReplies:
        def __init__(self):
            self.index = 0

        def create(self, **kwargs):
            if self.index >= len(predictions):
                raise RuntimeError("Successful source replies exhausted")
            reply = predictions[self.index]
            self.index += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply, tool_calls=None))])

    def new_agent():
        return OwlAgent(model=args.model, api_url=args.api_url, api_key=os.environ["OSWORLD_API_KEY"],
            history_n=3, action_space="mcp", observation_type="screenshot", max_trajectory_length=15,
            top_p=0.9, temperature=1.0)

    desktop = None
    results = []
    try:
        desktop = DesktopEnv(action_space="mcp", headless=True, provider_name="docker", os_type="Ubuntu",
            screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: evaluate_with_save_barrier(desktop, original_evaluate,
            log=lambda event: append(output / "save-barrier.jsonl", event))
        if args.resume_verified_seed:
            verified = json.loads((output / "seed-verified.json").read_text())
            assert verified["score"] == 1.0 and verified["equal_empirical_data"]
            assert not list(output.glob("rep-*")), "Refusing to resume a partially evaluated task"
            memory_roots = {name: output / f"seed-{name}" for name in ("current", "improved")}
            before_hashes = json.loads((output / "seed-hashes-before.json").read_text())
            assert all(hashes(path) == before_hashes[name] for name, path in memory_roots.items())
            write_json(output / "resume-verified-seed.json", dict(reused=True, no_synthetic_counts=True))
        else:
            write_json(status_path, dict(state="building_seed", task=task))
            fixed = new_agent()
            fixed.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=FixedReplies()))
            # Single observed seed episode, annotated once, shared by both methods.
            collector = EvoTreeMemory(distiller=OSWorldDistiller())
            aw, ew, runtime = wrap_osworld(fixed, desktop, collector, gui_annotator=VisionGUIAnnotator(vision_request))
            seed_result = run_episode(aw, ew, config, output / "seed-replay", runtime=runtime)
            assert seed_result["score"] == 1.0, "Source reply replay did not succeed; refusing to seed"
            serialized = dict(instruction=config["instruction"], success=True,
                              steps=seed_result["episode_steps"], source=str(source), metadata={"score": 1.0})
            write_json(output / "seed-episode.json", serialized)
            seed_episode = Episode(instruction=serialized["instruction"], success=True,
                steps=tuple(EpisodeStep(**step) for step in serialized["steps"]),
                source=str(source), metadata={"score": 1.0})
            memory_roots = {}
            for condition in ("current", "improved"):
                path = output / f"seed-{condition}"
                package = current if condition == "current" else SimpleNamespace(
                    memory=importlib.import_module("evotree.memory"), distill_osworld=importlib.import_module("evotree.distill_osworld"))
                memory = package.memory.EvoTreeMemory(path, package.distill_osworld.OSWorldDistiller())
                memory.write_episode(seed_episode)
                memory.save()
                memory_roots[condition] = path
            before_hashes = {name: hashes(path) for name, path in memory_roots.items()}
            write_json(output / "seed-hashes-before.json", before_hashes)
            # Both conditions must have exactly the same empirical counts and
            # topology. Only GUI description content / lookahead rendering differ.
            trees = {}
            for condition, path in memory_roots.items():
                trees[condition] = json.loads(next(path.glob("memory/*/tree.json")).read_text())
            scrubbed = []
            for tree in trees.values():
                copied = json.loads(json.dumps(tree))
                for edge in copied["edges"]:
                    edge["action"]["args"].pop("target", None)
                    for sample in edge["arg_samples"]:
                        sample["args"].pop("target", None)
                    edge["arg_samples"] = [s for s in edge["arg_samples"] if s["args"]]
                scrubbed.append(copied)
            assert scrubbed[0] == scrubbed[1], "Seed empirical data differ across conditions"
            if args.task_index == 0:
                assert any(edge["action"]["args"].get("target") for edge in trees["improved"]["edges"]), "GUI seed has no visually supported target"
            write_json(output / "seed-verified.json", dict(score=1.0, equal_empirical_data=True, frozen_strict_quality=False,
                note="One successful observation is allowed in BOTH conditions; no synthetic repeated counts."))
        if args.seed_only:
            state = dict(state="seed_complete", task=task, score=1.0, equal_empirical_data=True)
            return 0
        if args.evaluation_gate is not None:
            write_json(status_path, dict(state="seed_verified_waiting_for_persistence", task=task, score=1.0))
            while not args.evaluation_gate.exists():
                time.sleep(10)
            gate = json.loads(args.evaluation_gate.read_text())
            assert gate.get("proceed") is True, "Persistence check did not authorize evaluation"
        for rep in range(args.repeats):
            order = ORDERS[(rep + args.task_index) % len(ORDERS)]
            for condition in order:
                write_json(status_path, dict(state="running", task=task, repetition=rep + 1,
                    condition=condition, completed=len(results), total=args.repeats * 3))
                agent = new_agent()
                env, runtime = desktop, None
                if condition != "baseline":
                    if condition == "current":
                        memory = current.memory.EvoTreeMemory(memory_roots[condition], current.distill_osworld.OSWorldDistiller(),
                            write_enabled=False, explore=False, strict_quality=False)
                        agent, env, runtime = current.osworld_integration.wrap_osworld(agent, desktop, memory)
                    else:
                        memory = EvoTreeMemory(memory_roots[condition], OSWorldDistiller(),
                            write_enabled=False, explore=False, strict_quality=False)
                        agent, env, runtime = wrap_osworld(agent, desktop, memory)
                started = time.monotonic()
                directory = output / f"rep-{rep+1}" / condition
                try:
                    result = run_episode(agent, env, config, directory, runtime=runtime)
                except Exception:
                    # A failed episode is never quietly retried or counted as
                    # a low task score; preserve it and stop this task chain.
                    (directory / "exception.txt").write_text(traceback.format_exc())
                    raise
                score = result["score"]
                row = dict(task=task, condition=condition, repetition=rep + 1, score=score,
                    decisions=result["decisions"], stopped=result["stopped"], seconds=time.monotonic() - started)
                results.append(row)
                append(output / "results.jsonl", row)
                for name, path in memory_roots.items():
                    assert hashes(path) == before_hashes[name], f"Frozen memory changed: {name}"
        state = dict(state="complete", task=task, completed=len(results), results=results, frozen_memory_unchanged=True)
    except Exception as exc:
        state = dict(state="failed", task=task, completed=len(results), error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        if desktop is not None:
            pids = list(desktop.provider._owned_qemu)
            desktop.close()
            state["closed_qemu_pids"] = pids
        write_json(status_path, state)
    return 0 if state["state"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "current-code", "improved-code", "output-root", "source-root", "asset-cache-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--task-index", type=int, choices=range(3), required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume-verified-seed", action="store_true")
    parser.add_argument("--seed-only", action="store_true", help="Verify seed construction without evaluating conditions")
    parser.add_argument("--evaluation-gate", type=Path, help="Build seeds now; await verified persistence before evaluation")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    return worker(args)


if __name__ == "__main__":
    raise SystemExit(main())

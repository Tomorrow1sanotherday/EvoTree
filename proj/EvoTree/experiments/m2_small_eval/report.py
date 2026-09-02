"""Summarize baseline, frozen, and online OSWorld result directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Iterable

from evotree.osworld_adapter import OSWorldEpisodeAdapter

ARMS = ("baseline", "frozen", "online")


def load_tasks(path: Path) -> list[tuple[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Task manifest must be a domain-to-task-list object")
    tasks: list[tuple[str, str]] = []
    for domain, task_ids in value.items():
        if not isinstance(domain, str) or not isinstance(task_ids, list):
            raise ValueError("Invalid task manifest entry")
        for task_id in task_ids:
            if not isinstance(task_id, str):
                raise ValueError("Task IDs must be strings")
            tasks.append((domain, task_id))
    return tasks


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def prompt_marker_hits(path: Path) -> tuple[int, int]:
    """Count model steps whose recorded prompt contains EvoTree sections."""
    experience_hits = 0
    caution_hits = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue
        current_user_message = next(
            (
                message
                for message in reversed(messages)
                if isinstance(message, dict) and message.get("role") == "user"
            ),
            None,
        )
        if current_user_message is None:
            continue
        rendered = json.dumps(current_user_message.get("content"), ensure_ascii=False)
        experience_hits += "[经验]" in rendered
        caution_hits += "[注意]" in rendered
    return experience_hits, caution_hits


def result_path(
    root: Path, arm: str, model: str, trial_id: str, domain: str, task_id: str
) -> Path:
    return root / arm / "results" / "mcp" / "screenshot" / model / trial_id / domain / task_id


def summarize_task(path: Path, domain: str, task_id: str) -> dict[str, object]:
    if not path.joinpath("result.txt").is_file() or not path.joinpath("traj.jsonl").is_file():
        return {
            "domain": domain,
            "task_id": task_id,
            "complete": False,
            "score": None,
            "effective_actions": None,
            "tool_actions": None,
            "gui_actions": None,
            "experience_prompt_hits": None,
            "caution_prompt_hits": None,
        }
    adapter = OSWorldEpisodeAdapter()
    episode = adapter.from_result_directory(path)
    kinds = [adapter.distiller.distill_action(step.raw_action).kind for step in episode.steps]
    experience_hits, caution_hits = prompt_marker_hits(path / "traj.jsonl")
    return {
        "domain": domain,
        "task_id": task_id,
        "complete": True,
        "score": episode.metadata["score"],
        "effective_actions": episode.num_steps,
        "tool_actions": sum(kind == "tool" for kind in kinds),
        "gui_actions": sum(kind == "gui" for kind in kinds),
        "experience_prompt_hits": experience_hits,
        "caution_prompt_hits": caution_hits,
    }


def memory_unchanged(arm_root: Path) -> bool | None:
    before = arm_root / "memory_before.json"
    after = arm_root / "memory_after.json"
    if not before.is_file() or not after.is_file():
        return None
    return read_json(before) == read_json(after)


def memory_changes(arm_root: Path) -> dict[str, int] | None:
    before_path = arm_root / "memory_before.json"
    after_path = arm_root / "memory_after.json"
    if not before_path.is_file() or not after_path.is_file():
        return None
    before = read_json(before_path)
    after = read_json(after_path)
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    before_keys = set(before)
    after_keys = set(after)
    return {
        "before_files": len(before_keys),
        "after_files": len(after_keys),
        "added_files": len(after_keys - before_keys),
        "removed_files": len(before_keys - after_keys),
        "modified_files": sum(
            before[key] != after[key] for key in before_keys & after_keys
        ),
    }


def run_returncode(arm_root: Path) -> int | None:
    run_path = arm_root / "run.json"
    if run_path.is_file():
        value = read_json(run_path)
        if not isinstance(value, dict):
            return None
        returncode = value.get("returncode")
    else:
        returncode_path = arm_root / "returncode"
        if not returncode_path.is_file():
            return None
        try:
            returncode = int(returncode_path.read_text(encoding="utf-8").strip())
        except ValueError:
            return None
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        return None
    return returncode


def summarize_experiment(
    root: Path,
    task_manifest: Path,
    *,
    model: str,
    trial_id: str = "0",
) -> dict[str, object]:
    tasks = load_tasks(task_manifest)
    arms: dict[str, object] = {}
    for arm in ARMS:
        task_rows = [
            summarize_task(
                result_path(root, arm, model, trial_id, domain, task_id),
                domain,
                task_id,
            )
            for domain, task_id in tasks
        ]
        complete = [row for row in task_rows if row["complete"]]
        scores = [float(row["score"]) for row in complete]
        action_counts = [int(row["effective_actions"]) for row in complete]
        returncode = run_returncode(root / arm)
        infrastructure_ok = None if returncode is None else returncode == 0
        missing = len(task_rows) - len(complete)
        arms[arm] = {
            "tasks": task_rows,
            "completed": len(complete),
            "missing": missing,
            "run_returncode": returncode,
            "infrastructure_ok": infrastructure_ok,
            "complete": infrastructure_ok is True and missing == 0,
            "success_rate": mean(score > 0 for score in scores) if scores else None,
            "mean_score": mean(scores) if scores else None,
            "mean_effective_actions": mean(action_counts) if action_counts else None,
            "tool_actions": sum(int(row["tool_actions"]) for row in complete),
            "gui_actions": sum(int(row["gui_actions"]) for row in complete),
            "experience_prompt_hits": sum(
                int(row["experience_prompt_hits"]) for row in complete
            ),
            "caution_prompt_hits": sum(
                int(row["caution_prompt_hits"]) for row in complete
            ),
            "memory_unchanged": memory_unchanged(root / arm),
            "memory_changes": memory_changes(root / arm),
        }
    return {
        "experiment_root": str(root),
        "task_manifest": str(task_manifest),
        "model": model,
        "trial_id": trial_id,
        "arms": arms,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("task_manifest", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--trial-id", default="0")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    summary = summarize_experiment(
        args.experiment_root.resolve(),
        args.task_manifest.resolve(),
        model=args.model,
        trial_id=args.trial_id,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

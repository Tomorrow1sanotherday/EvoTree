"""Repeat baseline and current online EvoTree with fixed, isolated task shards.

Every subprocess still runs one environment. Stable shard assignments preserve
memory across rounds without concurrent writes to the same forest. No changes
to the memory algorithm or hand-written task solutions are made here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
import re
from pathlib import Path
import subprocess
import signal
import sys
import time

from .report import load_tasks, summarize_experiment
from .run_arm import write_json


def shard_tasks(tasks: list[tuple[str, str]], workers: int) -> list[dict[str, list[str]]]:
    if not 1 <= workers <= min(16, len(tasks)):
        raise ValueError("workers must be 1..16 and cannot exceed the task count")
    if len(tasks) != len(set(tasks)):
        raise ValueError("duplicate tasks in manifest")
    shards: list[dict[str, list[str]]] = [{} for _ in range(workers)]
    for index, (domain, task) in enumerate(tasks):
        shards[index % workers].setdefault(domain, []).append(task)
    return shards


def validate_executed_actions(records: list[dict], path: Path) -> None:
    """Reject runs which silently discard valid model-proposed operations."""
    groups: dict[tuple[object, str], list[dict]] = {}
    for record in records:
        prediction = record.get("prediction")
        if isinstance(prediction, str) and prediction != "client error":
            groups.setdefault((record.get("step_num"), prediction), []).append(record)
    for (_, prediction), rows in groups.items():
        proposed = []
        for block in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", prediction, flags=re.S):
            start = block.find("{")
            if start < 0:
                continue
            try:
                call, _ = json.JSONDecoder().raw_decode(block[start:])
            except ValueError:
                continue
            if not isinstance(call, dict) or "name" not in call or "arguments" not in call:
                continue
            proposed.append(call)
            if call["name"] == "computer_use" and call["arguments"].get("action") == "terminate":
                break  # Calls after termination must not execute.
        if proposed and len(rows) != len(proposed):
            raise RuntimeError(f"Model proposed {len(proposed)} actions but only {len(rows)} "
                               f"executed at step {rows[0].get('step_num')} in {path}")
        for call, row in zip(proposed, rows):
            action = row.get("action")
            if call["name"] == "computer_use":
                if call["arguments"].get("action") == "terminate":
                    expected = "DONE" if call["arguments"].get("status") == "success" else "FAIL"
                    if action != expected:
                        raise RuntimeError(f"Termination mismatch in {path}")
                elif action in ("DONE", "FAIL"):
                    raise RuntimeError(f"Nonterminal model action replaced by termination in {path}")
            elif not isinstance(action, dict) or action.get("action_type") != call["name"]:
                raise RuntimeError(f"Proposed tool {call['name']} did not execute in {path}")


def completed_tasks(root: Path, manifest: Path, model: str, arm: str) -> list[dict]:
    # A result file can exist even when the runner swallowed an exception.
    for domain, task in load_tasks(manifest):
        path = root / arm / "results" / "mcp" / "screenshot" / model / "0" / domain / task / "traj.jsonl"
        if not path.is_file():
            raise RuntimeError(f"Missing trajectory at {path}")
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            if isinstance(record, dict) and ("Error" in record or "error" in record):
                raise RuntimeError(f"Infrastructure error recorded in {path}")
            if isinstance(record, dict) and str(record.get("prediction", "")).strip().lower() == "client error":
                raise RuntimeError(f"Model API error recorded in {path}")
        validate_executed_actions(records, path)
    result = summarize_experiment(root, manifest, model=model)["arms"][arm]
    if not result["complete"]:
        raise RuntimeError(f"Incomplete {arm} run at {root}: {result['completed']} complete, "
                           f"{result['missing']} missing, rc={result['run_returncode']}")
    return result["tasks"]


def round_summary(rows: dict[str, list[dict]], round_number: int) -> dict:
    metrics = {}
    for arm, tasks in rows.items():
        success = [row for row in tasks if float(row["score"]) >= 1.0 - 1e-9]
        failure = [row for row in tasks if float(row["score"]) < 1.0 - 1e-9]
        mean = lambda values: sum(values) / len(values) if values else None
        metrics[arm] = {
            "tasks": tasks,
            "full_success_rate": len(success) / len(tasks),
            "mean_score": mean([float(row["score"]) for row in tasks]),
            "mean_actions_success": mean([int(row["effective_actions"]) for row in success]),
            "mean_actions_failure": mean([int(row["effective_actions"]) for row in failure]),
            "experience_prompt_hits": sum(int(row["experience_prompt_hits"]) for row in tasks),
            "caution_prompt_hits": sum(int(row["caution_prompt_hits"]) for row in tasks),
        }
    return {"round": round_number, "arms": metrics,
            "online_minus_baseline_score": metrics["online"]["mean_score"] - metrics["baseline"]["mean_score"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runner-root", "output-root", "task-manifest", "evotree-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    parser.add_argument("--label", default="pilot")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse verified completed arms; retry incomplete baseline only")
    args = parser.parse_args(argv)
    if args.rounds < 1 or args.max_steps < 1 or args.timeout_seconds < 1:
        parser.error("rounds, max steps and timeout must be positive")
    tasks = load_tasks(args.task_manifest)
    shards = shard_tasks(tasks, args.workers)
    # This pilot intentionally uses the original OSWorld task configs, including
    # feasible document tasks. The runner must load every domain in the manifest.
    plan = {"workers": args.workers, "rounds": args.rounds, "max_steps": args.max_steps,
            "model": args.model, "shards": shards, "memory": "fixed shard, empty start, online writes",
            "arm_order": [["baseline", "online"] if r % 2 else ["online", "baseline"]
                          for r in range(1, args.rounds + 1)],
            "global_notes_scope": "within shard; no cross-worker sharing",
            "created_at": datetime.now(timezone.utc).isoformat()}
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    if args.resume:
        previous_plan = json.loads((args.output_root/'plan.json').read_text(encoding='utf-8'))
        for key in ('workers','rounds','max_steps','model','shards','arm_order','global_notes_scope'):
            if previous_plan.get(key) != plan[key]:
                raise ValueError(f'Resume plan mismatch: {key}')
        previous_status = json.loads((args.output_root/'status.json').read_text(encoding='utf-8'))
        if previous_status.get('state') not in ('failed', 'complete'):
            raise ValueError('Refusing to resume an experiment that may still be running')
        # Validate pre-existing online arms before any baseline retry. A partial
        # online episode could already have written memory; never reuse it.
        for round_number in range(1,args.rounds+1):
            for index in range(args.workers):
                root = args.output_root/f'pass-{round_number}'/f'shard-{index}'
                if (root/'online').exists():
                    completed_tasks(root,args.output_root/f'shard-{index}.json',args.model,'online')
        history = args.output_root/'resume-history'
        history.mkdir(exist_ok=True)
        write_json(history / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
                   {'previous_status':previous_status,'runner_root':str(args.runner_root),'evotree_root':str(args.evotree_root)})
        plan = previous_plan
    else:
        args.output_root.mkdir(parents=True, exist_ok=False)
        write_json(args.output_root / "plan.json", plan)
    manifests = []
    seeds = []
    for index, shard in enumerate(shards):
        manifest = args.output_root / f"shard-{index}.json"
        if args.resume:
            if json.loads(manifest.read_text(encoding='utf-8')) != shard:
                raise ValueError(f'Resume shard mismatch: {index}')
        else:
            write_json(manifest, shard)
        manifests.append(manifest)
        seed = args.output_root / "empty-seeds" / f"shard-{index}"
        seed.mkdir(parents=True, exist_ok=args.resume)
        if args.resume and any(seed.iterdir()):
            raise ValueError('Initial seed must remain empty')
        seeds.append(seed)

    def run_one(index: int, arm: str, round_number: int) -> list[dict]:
        root = args.output_root / f"pass-{round_number}" / f"shard-{index}"
        root.mkdir(parents=True, exist_ok=True)
        retry_baseline = False
        if args.resume and (root/arm).exists():
            try:
                existing = completed_tasks(root, manifests[index], args.model, arm)
            except (RuntimeError, ValueError, FileNotFoundError):
                if arm != 'baseline':
                    raise
                retry_baseline = True
            else:
                if arm == 'online':
                    seeds[index] = root/'online/memory/env-0'
                return existing
        command = [str(args.python.absolute()), "-m", "experiments.m2_small_eval.run_arm",
                   "--arm", arm, "--runner-root", str(args.runner_root),
                   "--output-root", str(root), "--task-manifest", str(manifests[index]),
                   "--evotree-root", str(args.evotree_root), "--python", str(args.python.absolute()),
                   "--model", args.model, "--api-url", args.api_url,
                   "--max-steps", str(args.max_steps), "--worker-slot", str(index),
                   "--label", f"{args.label}-p{round_number}-s{index}"]
        if arm == "online":
            command += ["--seed-memory", str(seeds[index])]
        if retry_baseline:
            command += ['--resume-baseline']
        env = dict(os.environ)
        env["EVOTREE_RETRIEVAL_LOG"] = "1"
        with (root / f"{arm}-runner.log").open("a" if args.resume else "w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=args.evotree_root, env=env,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=os.name == "posix")
            try:
                returncode = process.wait(timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == "posix":
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
                raise RuntimeError(f"worker {index} {arm} round {round_number} timed out; "
                                   "inspect its exact container name in run.json before cleanup")
        if returncode:
            raise RuntimeError(f"worker {index} {arm} round {round_number} exited {returncode}")
        rows = completed_tasks(root, manifests[index], args.model, arm)
        if arm == "online":
            seeds[index] = root / "online" / "memory" / "env-0"
        return rows

    started = time.monotonic()
    summaries = []
    try:
        for round_number, order in enumerate(plan["arm_order"], start=1):
            rows = {}
            for arm in order:
                write_json(args.output_root / "status.json", {"state": "running", "round": round_number, "arm": arm})
                # Resolve every submitted worker before proceeding to the next
                # stage. Each thread only supervises its own isolated runner.
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    futures = [pool.submit(run_one, index, arm, round_number) for index in range(args.workers)]
                    outcomes = []
                    errors = []
                    for future in futures:
                        try:
                            outcomes.extend(future.result())
                        except Exception as exc:
                            errors.append(str(exc))
                    if errors:
                        raise RuntimeError("; ".join(errors))
                    rows[arm] = outcomes
            summaries.append(round_summary(rows, round_number))
            write_json(args.output_root / "summary.json", {"rounds": summaries, "elapsed_seconds": time.monotonic()-started})
        write_json(args.output_root / "status.json", {"state": "complete", "rounds": args.rounds})
    except Exception as exc:
        write_json(args.output_root / "status.json", {"state": "failed", "error": str(exc)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

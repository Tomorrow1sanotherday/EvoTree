"""Paired exploratory evaluation on untouched official OSWorld-MCP Office tasks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shutil
import subprocess
import sys
import time
import traceback
import uuid
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

CONDITIONS = ("baseline", "dual_summary")
MODEL = "gpt-5.6-terra"
API_URL = "http://llmapi.evolventapi.com/v1"
MAX_STEPS = 15
SCHEDULE_SEED = 20261022
SELECTED_TASKS = (
    ("calc", "libreoffice_calc", "04d9aeaf-7bed-4024-bedb-e10e6f00eb7f"),
    ("calc", "libreoffice_calc", "1d17d234-e39d-4ed7-b46f-4417922a4e7c"),
    ("calc", "libreoffice_calc", "37608790-6147-45d0-9f20-1137bb35703d"),
    ("calc", "libreoffice_calc", "eb03d19a-b88d-4de4-8a64-ca0ac66f426b"),
    ("calc", "libreoffice_calc", "a9f325aa-8c05-4e4f-8341-9e4358565f4f"),
    ("calc", "libreoffice_calc", "0bf05a7d-b28b-44d2-955a-50b41e24012a"),
    ("impress", "libreoffice_impress", "edb61b14-a854-4bf5-a075-c8075c11293a"),
    ("impress", "libreoffice_impress", "358aa0a7-6677-453f-ae35-e440f004c31e"),
    ("impress", "libreoffice_impress", "04578141-1d42-4146-b9cf-6fab4ce5fd74"),
    ("impress", "libreoffice_impress", "15aece23-a215-4579-91b4-69eec72e18da"),
    ("impress", "libreoffice_impress", "39be0d19-634d-4475-8768-09c130f5425d"),
    ("impress", "libreoffice_impress", "e4ef0baf-4b52-4590-a47e-d4d464cca2d7"),
)
SMOKE_TASK = ("libreoffice_calc", "1334ca3e-f9e3-4db8-9ca7-b4c653be7d17")


def code_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*.py'))
            if '__pycache__' not in path.parts and 'cache' not in path.relative_to(root).parts}


def model_lock_path(runtime_root):
    digest = hashlib.sha256(str(runtime_root.resolve()).encode()).hexdigest()[:16]
    return Path('/tmp') / f'rh-official-mcp-{digest}.lock'


class SummaryRequestAudit:
    """Record every network attempt, including empty responses and failures."""
    def __init__(self, transport, root):
        self.transport, self.root = transport, root
        self.calls = self.failures = 0
        root.mkdir(parents=True, exist_ok=True)

    def __call__(self, messages):
        self.calls += 1
        record = dict(attempt=self.calls, request=messages)
        started = time.monotonic()
        try:
            response = self.transport(messages)
            record.update(response=response, returned=True)
            return response
        except Exception as exc:
            self.failures += 1
            record.update(returned=False, error_type=type(exc).__name__)
            raise
        finally:
            record['seconds'] = time.monotonic() - started
            (self.root / f'{self.calls:05d}.json').write_text(
                json.dumps(record, ensure_ascii=False), encoding='utf-8')


def clear_result_cache(config, task_cache):
    """Remove previous evaluator outputs before each independent episode."""
    for name in _result_destinations(config):
        (task_cache / name).unlink(missing_ok=True)


def make_schedule(tasks=SELECTED_TASKS, seed=SCHEDULE_SEED):
    """Balance condition order within app family and distribute 3 cases per worker."""
    by_family = defaultdict(list)
    for family, domain, task_id in tasks:
        by_family[family].append((domain, task_id))
    if set(by_family) != {"calc", "impress"} or any(len(items) != 6 for items in by_family.values()):
        raise ValueError("Expected six Calc and six Impress tasks")

    workers = [[] for _ in range(4)]
    rng = random.Random(seed)
    for family_index, family in enumerate(("calc", "impress")):
        items = sorted(by_family[family])
        rng.shuffle(items)
        for index, (domain, task_id) in enumerate(items):
            worker = (index + 2 * family_index) % len(workers)
            order = list(CONDITIONS if index % 2 == 0 else reversed(CONDITIONS))
            workers[worker].append(dict(family=family, domain=domain, task_id=task_id, order=order))
    for jobs in workers:
        rng.shuffle(jobs)
    return workers


def _support(runtime_root):
    sys.path[:0] = [str(runtime_root / "code"), str(runtime_root / "runner"),
                    "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
    from experiments.m2_small_eval.diagnostic_pilot import append, hashes, run_episode, write_json
    from experiments.m2_small_eval.generalization_eval import ExecutedHistoryAgent, RequestAudit, make_memory
    from experiments.m2_small_eval.office_save_barrier import evaluate_with_save_barrier
    from experiments.m2_small_eval.summary_success_eval import CachedHistoryTransport, StructuredSummaryBackend
    return dict(append=append, hashes=hashes, run_episode=run_episode, write_json=write_json,
        ExecutedHistoryAgent=ExecutedHistoryAgent, RequestAudit=RequestAudit, make_memory=make_memory,
        evaluate_with_save_barrier=evaluate_with_save_barrier,
        CachedHistoryTransport=CachedHistoryTransport, StructuredSummaryBackend=StructuredSummaryBackend)


def build_plan(runtime_root: Path, output_root: Path):
    helpers = _support(runtime_root)
    runner = runtime_root / "runner"
    inventory_path = runtime_root / "inventory" / "tasks-template-audit.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_key = {(row["domain"], row["id"]): row for row in inventory}
    tasks = []
    input_urls = set()
    for family, domain, task_id in SELECTED_TASKS:
        row = by_key[(domain, task_id)]
        if not row["mcp_listed"] or row["prior_exposure"] or row["task_memory_hit"]:
            raise ValueError(f"Task is not an untouched MCP task with no task-memory hit: {task_id}")
        config_path = runner / "evaluation_examples" / "examples" / domain / f"{task_id}.json"
        raw = config_path.read_bytes()
        config = json.loads(raw)
        if config.get("id") != task_id:
            raise ValueError(f"Runner config ID mismatch: {task_id}")
        if hashlib.sha256(raw).hexdigest() != row["config_sha256"]:
            raise ValueError(f"Runner config differs from the audited official config: {task_id}")
        assets = []
        for step in config.get("config", []):
            if step.get("type") == "download":
                for file in step.get("parameters", {}).get("files", []):
                    assets.append(dict(role="input", url=file["url"], name=Path(file["path"]).name))
                    input_urls.add(file["url"])
        expected = config.get("evaluator", {}).get("expected", [])
        expected = expected if isinstance(expected, list) else [expected]
        for item in expected:
            if isinstance(item, dict) and item.get("type") == "cloud_file":
                assets.append(dict(role="expected", url=item["path"], name=item.get("dest")))
        tasks.append(dict(family=family, domain=domain, id=task_id,
                          instruction=config["instruction"], config=config,
                          config_sha256=hashlib.sha256(raw).hexdigest(),
                          prior_exposure_count=0, task_memory_hit=False, assets=assets))
    if len(input_urls) != len(tasks):
        raise ValueError("Selected tasks share input files; use independent official instances")
    smoke_domain, smoke_id = SMOKE_TASK
    smoke_row = by_key[(smoke_domain, smoke_id)]
    if (not smoke_row["mcp_listed"] or smoke_row["prior_exposure"]
            or smoke_row["task_memory_hit"]):
        raise ValueError("Smoke task must be official MCP, unseen, and outside task-memory matches")
    smoke_config_path = runner / "evaluation_examples" / "examples" / smoke_domain / f"{smoke_id}.json"
    smoke_config_hash = hashlib.sha256(smoke_config_path.read_bytes()).hexdigest()
    if smoke_config_hash != smoke_row["config_sha256"]:
        raise ValueError("Smoke task config differs from the audited official config")
    if (smoke_domain, smoke_id) in {(domain, task_id) for _, domain, task_id in SELECTED_TASKS}:
        raise ValueError("Smoke task must be excluded from the formal split")

    frozen_bank = runtime_root / "bank"
    bank_hashes = helpers["hashes"](frozen_bank)
    frozen_code_hashes = {
        "evotree": code_hashes((runtime_root / "code" / "evotree").resolve()),
        "experiment_helpers": code_hashes((runtime_root / "code" / "experiments").resolve()),
        "runner": code_hashes(runner.resolve()),
        "evaluation_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError(f"Refusing to replace a nonempty experiment output: {output_root}")
    plan = dict(
        experiment="official-osworld-mcp-office-transfer-20261022",
        setting="12 untouched official test_all_mcp Office tasks; paired baseline vs dual_summary",
        task_source="OSWorld-MCP evaluation_examples, config hashes matched to the frozen inventory",
        inventory_sha256=hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        bank_hashes=bank_hashes,
        code_hashes=frozen_code_hashes,
        model=MODEL,
        api_url=API_URL,
        max_steps=MAX_STEPS,
        conditions=list(CONDITIONS),
        tasks=[{key: value for key, value in task.items() if key != "config"} for task in tasks],
        smoke_task=dict(domain=smoke_domain, id=smoke_id, config_sha256=smoke_config_hash,
                        prior_exposure_count=0, task_memory_hit=False, included_in_formal_split=False),
        schedule=make_schedule(tasks=[(t["family"], t["domain"], t["id"]) for t in tasks]),
        model_preflight_contains_no_task_data=True,
        model_request_lock=str(model_lock_path(runtime_root)),
        smoke_protocol='Fixed MCP calls, initial evaluator score 0 and final score 1; excluded from results.',
        asset_protocol="Official inputs pre-staged with original URL UUID5 keys; evaluator gold remains on host.",
    )
    helpers["write_json"](output_root / "plan.json", plan)
    helpers["write_json"](output_root / "tasks-private.json", tasks)
    return plan


def _result_destinations(config):
    result = config.get("evaluator", {}).get("result", [])
    result = result if isinstance(result, list) else [result]
    names = []
    for item in result:
        if isinstance(item, dict) and item.get('dest'):
            dest = item['dest']
            names.extend(dest if isinstance(dest, list) else [dest])
    if any(not isinstance(name, str) or Path(name).name != name for name in names):
        raise ValueError('Result destinations must be plain basenames')
    return list(dict.fromkeys(names))


def preserve_result_artifacts(config, task_cache, directory, *, score, action_history):
    """Audit outputs without rejecting an official early FAIL verdict.

    DesktopEnv.evaluate returns 0 before calling result_getter when the final
    action is FAIL for a feasible task. No downloaded result is expected then.
    Missing output for every other outcome remains an experiment error.
    """
    names = _result_destinations(config)
    saved, missing = [], []
    for name in names:
        artifact = Path(task_cache) / name
        if artifact.is_file():
            destination = Path(directory) / name
            shutil.copy2(artifact, destination)
            saved.append(dict(name=name, sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
        else:
            missing.append(name)
    last_action = action_history[-1] if action_history else None
    skipped_for_fail = (last_action == 'FAIL' and score == 0
                        and config.get('evaluator', {}).get('func') != 'infeasible')
    audit = dict(official_score=score, last_action=last_action, saved_artifacts=saved,
        missing_artifacts=missing,
        artifact_status='official_fail_skipped_result_getter' if skipped_for_fail else 'preserved')
    (Path(directory) / 'evaluation-audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    if (missing or not saved) and not skipped_for_fail:
        raise FileNotFoundError(f'Official evaluation result artifact missing: {missing or names}')
    return audit


def _download_official_asset(url, target: Path, endpoint: str):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file():
        return "existing_official_cache"
    candidate = url
    if "huggingface.co" in url and endpoint:
        candidate = url.replace("https://huggingface.co", endpoint.rstrip("/"), 1)
    source = "configured_huggingface_mirror" if candidate != url else "official_url"
    temporary = target.with_name(target.name + ".download")
    try:
        request = Request(candidate, headers={"User-Agent": "EvoTree-OSWorld-MCP-evaluator/1.0"})
        for attempt in range(5):
            try:
                with urlopen(request, timeout=180) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                break
            except HTTPError as exc:
                temporary.unlink(missing_ok=True)
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 4:
                    raise
                retry_after = exc.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                except ValueError:
                    try:
                        delay = parsedate_to_datetime(retry_after).timestamp() - time.time()
                    except (TypeError, ValueError, OverflowError):
                        delay = 10 * (attempt + 1)
                time.sleep(max(1, min(delay, 60)))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return source


def stage_task_inputs(config, task_cache: Path, hf_endpoint: str):
    """Use OSWorld's UUID5 URL cache key to stage inputs before VM startup."""
    staged = []
    for step in config.get("config", []):
        if not isinstance(step, dict) or step.get("type") != "download":
            continue
        for item in step.get("parameters", {}).get("files", []):
            url, vm_path = item.get("url"), item.get("path")
            if not isinstance(url, str) or not isinstance(vm_path, str):
                raise ValueError("Invalid official task download configuration")
            name = f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_{Path(vm_path).name}"
            target = task_cache / name
            source = _download_official_asset(url, target, hf_endpoint)
            staged.append(dict(name=name, source=source,
                               sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return staged


def stage_evaluator_gold(config, task_cache: Path, hf_endpoint: str):
    """Stage official evaluator-only cloud files in host cache, never in the VM."""
    expected = config.get("evaluator", {}).get("expected", [])
    expected = expected if isinstance(expected, list) else [expected]
    task_cache.mkdir(parents=True, exist_ok=True)
    staged = []
    endpoint = hf_endpoint.rstrip("/")
    for item in expected:
        if not isinstance(item, dict) or item.get("type") != "cloud_file":
            continue
        urls = item.get("path", [])
        names = item.get("dest", [])
        urls = urls if isinstance(urls, list) else [urls]
        names = names if isinstance(names, list) else [names]
        if len(urls) != len(names):
            raise ValueError("Official cloud-file paths and destinations do not align")
        for url, name in zip(urls, names):
            if not isinstance(url, str) or not isinstance(name, str) or Path(name).name != name:
                raise ValueError("Invalid official cloud-file evaluator input")
            target = task_cache / name
            source = _download_official_asset(url, target, endpoint)
            staged.append(dict(name=name, source=source, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return staged


def summarize(rows):
    pairs = {}
    for row in rows:
        pair = pairs.setdefault(row["task_id"], dict(family=row["family"], scores={}))
        if row["condition"] in pair["scores"]:
            raise ValueError(f"Duplicate result for {row['task_id']} / {row['condition']}")
        pair["scores"][row["condition"]] = float(row["score"])
    if not all(set(pair["scores"]) == set(CONDITIONS) for pair in pairs.values()):
        raise ValueError("Results must contain complete paired episodes")
    paired = {}
    for family in ("calc", "impress", "all"):
        group = [pair for pair in pairs.values() if family == "all" or pair["family"] == family]
        wins = sum(p["scores"]["dual_summary"] >= 1 and p["scores"]["baseline"] < 1 for p in group)
        losses = sum(p["scores"]["baseline"] >= 1 and p["scores"]["dual_summary"] < 1 for p in group)
        n = wins + losses
        p_value = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
        paired[family] = dict(
            tasks=len(group),
            baseline_successes=sum(p["scores"]["baseline"] >= 1 for p in group),
            dual_successes=sum(p["scores"]["dual_summary"] >= 1 for p in group),
            dual_wins=wins,
            dual_losses=losses,
            ties=len(group) - wins - losses,
            exact_two_sided_mcnemar_p=p_value,
        )
    costs = {}
    retrieval = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        costs[condition] = {
            key: sum(row.get(key, 0) for row in selected)
            for key in ("decisions", "agent_calls", "summary_calls", "cached_summary_calls",
                        "summary_backend_attempts", "summary_transport_errors", "fallback_decisions", "seconds")
        }
        retrieval[condition] = {
            key: sum(row.get(key, 0) for row in selected)
            for key in ("task_sources", "state_sources", "summary_comparisons", "compatible_comparisons",
                        "state_injected_records", "state_injected_decisions", "task_injected_decisions")
        }
    return dict(
        episodes=len(rows),
        paired=paired,
        costs=costs,
        retrieval=retrieval,
        interpretation="Exploratory paired official-task evaluation; not a benchmark-wide estimate.",
    )


def worker(args):
    rt, out = args.runtime_root, args.output_root
    helpers = _support(rt)
    os.chdir(rt / "runner")
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment
    from evotree.osworld_integration import wrap_osworld
    from evotree.situation_summary import JsonChatTransport

    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    frozen_bank = rt / "bank"
    assert helpers["hashes"](frozen_bank) == plan["bank_hashes"]
    cases = {task["id"]: task for task in json.loads((out / "tasks-private.json").read_text(encoding="utf-8"))}
    jobs = plan["schedule"][args.worker]
    folder = out / f"worker-{args.worker}"
    folder.mkdir(parents=True, exist_ok=False)
    os.environ.update(json.loads((rt / "runtime-env.json").read_text(encoding="utf-8")))
    os.environ["OSWORLD_MODEL_REQUEST_LOCK"] = str(model_lock_path(rt))
    os.environ["OSWORLD_MODEL_REQUEST_LOCK_MODE"] = 'hard'
    os.environ["OSWORLD_PORT_LOCK_DIR"] = str(rt / "port-locks")
    env = runtime_environment(os.environ,
        f"evo-{rt.name.replace('_', '-')}-w{args.worker}", 12 + args.worker)
    os.environ.update(env)
    os.environ["EVOTREE_RETRIEVAL_LOG"] = "0"
    rows, desktop = [], None
    def interrupted(signum, frame):
        raise InterruptedError(f'Worker interrupted by signal {signum}')
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    status = dict(state="booting", completed=0, planned=sum(len(job["order"]) for job in jobs))
    helpers["write_json"](folder / "status.json", status)
    try:
        desktop = DesktopEnv(action_space="mcp", headless=True, provider_name="docker", os_type="Ubuntu",
            cache_dir=str(rt / "cache"),
            screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: helpers["evaluate_with_save_barrier"](desktop, original_evaluate,
            log=lambda event: helpers["append"](folder / "save-barrier.jsonl", event))
        for job in jobs:
            task = cases[job["task_id"]]
            for condition in job["order"]:
                directory = folder / task["id"] / condition
                status.update(state="running", completed=len(rows), task=task["id"], condition=condition)
                helpers["write_json"](folder / "status.json", status)
                task_cache = Path(desktop.cache_dir_base) / task["id"]
                clear_result_cache(task["config"], task_cache)
                gold = stage_evaluator_gold(task["config"],
                    Path(desktop.cache_dir_base) / task["id"], os.environ.get("HF_ENDPOINT", ""))
                inputs = stage_task_inputs(task["config"],
                    Path(desktop.cache_dir_base) / task["id"], os.environ.get("HF_ENDPOINT", ""))
                helpers["write_json"](directory.parent / f"{condition}-prefetched-assets.json",
                    dict(inputs=inputs, evaluator_gold=gold))
                transport = None
                backend = None
                network_audit = None
                if condition == "dual_summary":
                    request_dir = directory.with_name("dual_summary-requests")
                    request_dir.mkdir(parents=True)
                    network_audit = SummaryRequestAudit(
                        JsonChatTransport(plan["api_url"], plan["model"],
                                          os.environ["OSWORLD_API_KEY"], timeout=180),
                        request_dir / 'network-attempts')
                    transport = helpers["CachedHistoryTransport"](
                        network_audit,
                        rt / "history-responses", request_dir)
                    backend = helpers["StructuredSummaryBackend"](transport)
                agent = OwlAgent(model=plan["model"], api_url=plan["api_url"],
                    api_key=os.environ["OSWORLD_API_KEY"], history_n=3, action_space="mcp",
                    observation_type="screenshot", max_trajectory_length=MAX_STEPS,
                    temperature=1.0, top_p=0.9)
                agent_audit = helpers["RequestAudit"](agent.llm_client.chat.completions,
                    folder / f"{task['id']}-{condition}-agent-requests.jsonl")
                agent.llm_client = type("AuditClient", (), {})()
                agent.llm_client.chat = type("Chat", (), {})()
                agent.llm_client.chat.completions = agent_audit
                protocol = helpers["ExecutedHistoryAgent"](agent)
                if condition == "dual_summary":
                    memory = helpers["make_memory"](frozen_bank, "dual_summary", backend)
                    wrapped, task_env, runtime = wrap_osworld(protocol, desktop, memory)
                else:
                    wrapped, task_env, runtime = wrap_osworld(protocol, desktop, None, use_evotree=False)
                started = time.monotonic()
                result = helpers["run_episode"](wrapped, task_env, task["config"], directory,
                                     max_steps=MAX_STEPS, runtime=runtime)
                evaluation_audit = preserve_result_artifacts(task["config"], desktop.cache_dir, directory,
                    score=result['score'], action_history=desktop.action_history)
                saved = evaluation_audit['saved_artifacts']
                asset_records = []
                for asset in task["assets"]:
                    names = asset["name"] if isinstance(asset['name'], list) else [asset['name']]
                    if asset["role"] == "input":
                        names = [f"{uuid.uuid5(uuid.NAMESPACE_URL, asset['url'])}_{name}" for name in names]
                    for name in names:
                        asset_path = Path(desktop.cache_dir) / name
                        asset_records.append(dict(role=asset["role"], name=name,
                            exists=asset_path.is_file(),
                            sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest() if asset_path.is_file() else None))
                decisions_path = directory / "decisions.jsonl"
                decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
                task_sources = sum(len(row.get("task_sources", [])) for row in decisions)
                state_sources = sum(len(row.get("state_sources", [])) for row in decisions)
                summary_rows = [entry for row in decisions for entry in row.get("summary_comparisons", [])]
                row = dict(task_id=task["id"], family=task["family"], condition=condition,
                    score=result["score"], decisions=result["decisions"], seconds=time.monotonic()-started,
                    agent_calls=agent_audit.calls,
                    summary_calls=network_audit.calls if network_audit else 0,
                    summary_transport_errors=network_audit.failures if network_audit else 0,
                    summary_backend_attempts=backend.model_calls if backend else 0,
                    fallback_decisions=sum(bool(r.get('summary_failures')) for r in decisions),
                    cached_summary_calls=transport.hits if transport else 0,
                    task_sources=task_sources, state_sources=state_sources,
                    summary_comparisons=len(summary_rows),
                    compatible_comparisons=sum(x.get("verdict") == "compatible" for x in summary_rows),
                    state_injected_records=sum(r.get('text', '').count('\nCandidate: ') for r in decisions),
                    state_injected_decisions=sum('\nCandidate: ' in r.get('text', '') for r in decisions),
                    task_injected_decisions=sum('[任务级经验 / Task-level experience]' in r.get('text', '') for r in decisions),
                    saved_artifacts=saved, official_assets=asset_records,
                    evaluation_artifact_status=evaluation_audit['artifact_status'])
                helpers["append"](folder / "results.jsonl", row)
                rows.append(row)
                assert helpers["hashes"](frozen_bank) == plan["bank_hashes"]
        status.update(state="complete", completed=len(rows))
    except Exception as exc:
        (folder / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status.update(state="failed", completed=len(rows), error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        if desktop is not None:
            status["closed_qemu_pids"] = list(desktop.provider._owned_qemu)
            desktop.close()
        status["bank_unchanged"] = helpers["hashes"](frozen_bank) == plan["bank_hashes"]
        helpers["write_json"](folder / "status.json", status)
    return 0 if status["state"] == "complete" else 1


def supervise(args):
    rt, out = args.runtime_root, args.output_root
    helpers = _support(rt)
    os.environ.update(json.loads((rt / "runtime-env.json").read_text(encoding="utf-8")))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    worker_count = len(plan['schedule'])
    planned_episodes = sum(len(job['order']) for jobs in plan['schedule'] for job in jobs)
    if not worker_count or not planned_episodes or any(not jobs for jobs in plan['schedule']):
        raise ValueError('Every scheduled worker must have at least one paired task')
    assert helpers["hashes"](rt / "bank") == plan["bank_hashes"]
    current_code_hashes = {
        "evotree": code_hashes((rt / "code" / "evotree").resolve()),
        "experiment_helpers": code_hashes((rt / "code" / "experiments").resolve()),
        "runner": code_hashes((rt / "runner").resolve()),
        "evaluation_script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    assert current_code_hashes == plan["code_hashes"]
    jobs = []
    try:
        for worker_id in range(worker_count):
            log_path = out / f"worker-{worker_id}.log"
            with log_path.open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--runtime-root", str(rt),
                     "--output-root", str(out), "--worker", str(worker_id)],
                    cwd=rt / f"work-{worker_id}", env=os.environ, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True)
            jobs.append((worker_id, process))
        helpers["write_json"](out / "workers.json", [dict(worker=i, pid=p.pid) for i, p in jobs])
        while any(process.poll() is None for _, process in jobs):
            workers, completed = [], 0
            for worker_id, process in jobs:
                folder = out / f"worker-{worker_id}"
                status_path = folder / "status.json"
                status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else dict(state="starting")
                result_path = folder / "results.jsonl"
                if result_path.exists():
                    completed += len([line for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()])
                workers.append(dict(worker=worker_id, pid=process.pid, returncode=process.poll(), status=status))
            helpers["write_json"](out / "status.json", dict(state="running", completed=completed, planned=planned_episodes, workers=workers))
            assert helpers["hashes"](rt / "bank") == plan["bank_hashes"]
            time.sleep(10)
        returncodes = [process.returncode for _, process in jobs]
        if any(code != 0 for code in returncodes):
            raise RuntimeError(f"Workers failed; partial episode artifacts preserved: {returncodes}")
        rows = []
        closed_pids = []
        for worker_id in range(worker_count):
            folder = out / f"worker-{worker_id}"
            status = json.loads((folder / "status.json").read_text(encoding="utf-8"))
            expected = sum(len(job['order']) for job in plan['schedule'][worker_id])
            if status.get("state") != "complete" or status.get("completed") != expected:
                raise RuntimeError(f"Worker {worker_id} did not finish all paired episodes")
            closed_pids.extend(status.get("closed_qemu_pids", []))
            rows.extend(json.loads(line) for line in (folder / "results.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
        if len(rows) != planned_episodes:
            raise RuntimeError(f"Expected {planned_episodes} scored episodes, got {len(rows)}")
        if any(Path("/proc", str(pid)).exists() for pid in closed_pids):
            raise RuntimeError("An experiment-owned QEMU process remained after worker shutdown")
        if helpers["hashes"](rt / "bank") != plan["bank_hashes"]:
            raise RuntimeError("Frozen memory bank changed during experiment")
        summary = summarize(rows)
        summary.update(bank_unchanged=True, workers_returncodes=returncodes,
                       remaining_owned_qemu=[], result_rows=rows)
        helpers["write_json"](out / "summary.json", summary)
        helpers["write_json"](out / "status.json", dict(state="complete", completed=planned_episodes, planned=planned_episodes))
    except Exception as exc:
        (out / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        helpers["write_json"](out / "status.json", dict(state="failed", error_type=type(exc).__name__, error=str(exc)))
        raise
    finally:
        for _, process in jobs:
            if process.poll() is None:
                process.wait()


def real_task_smoke(args):
    rt, out = args.runtime_root, args.output_root
    helpers = _support(rt)
    os.chdir(rt / "runner")
    sys.path[:0] = [str(rt / "runner"), "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
    from desktop_env.desktop_env import DesktopEnv
    from mm_agents.owl_agent import OwlAgent
    from experiments.m2_small_eval.run_arm import runtime_environment
    from evotree.osworld_integration import wrap_osworld
    from evotree.situation_summary import JsonChatTransport

    domain, task_id = SMOKE_TASK
    inventory_path = rt / "inventory" / "tasks-template-audit.json"
    row = next(item for item in json.loads(inventory_path.read_text(encoding="utf-8"))
               if item["domain"] == domain and item["id"] == task_id)
    if not row["mcp_listed"] or row["prior_exposure"]:
        raise ValueError("Smoke task must be official MCP and have no prior trajectory exposure")
    config_path = rt / "runner" / "evaluation_examples" / "examples" / domain / f"{task_id}.json"
    config_bytes = config_path.read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != row["config_sha256"]:
        raise ValueError("Smoke task config differs from the audited official task")
    task = json.loads(config_bytes)
    folder = out / "smoke" / task_id
    folder.mkdir(parents=True, exist_ok=False)
    episode_dir = folder / "episode"
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    (out / "smoke").mkdir(parents=True, exist_ok=True)
    status_path = out / "smoke-status.json"
    helpers["write_json"](status_path, dict(state="booting", task=task_id, scored_as_formal=False))
    summary_requests = folder / "summary-requests"
    summary_requests.mkdir(parents=True, exist_ok=True)
    os.environ.update(json.loads((rt / "runtime-env.json").read_text(encoding="utf-8")))
    os.environ['OSWORLD_MODEL_REQUEST_LOCK'] = str(model_lock_path(rt))
    os.environ['OSWORLD_MODEL_REQUEST_LOCK_MODE'] = 'hard'
    os.environ["OSWORLD_PORT_LOCK_DIR"] = str(rt / "port-locks")
    os.environ.update(runtime_environment(os.environ, "evo-rh-official-mcp-smoke-20261022", 11))
    transport = None
    backend = None
    if not args.fixture_smoke:
        transport = helpers["CachedHistoryTransport"](
            JsonChatTransport(plan["api_url"], plan["model"], os.environ.get("OSWORLD_API_KEY", ""), timeout=180),
            rt / "history-responses", summary_requests)
        backend = helpers["StructuredSummaryBackend"](transport)
    agent = OwlAgent(model=plan["model"], api_url=plan["api_url"],
        api_key=os.environ.get("OSWORLD_API_KEY", "fixture-no-api"), history_n=3, action_space="mcp",
        observation_type="screenshot", max_trajectory_length=MAX_STEPS,
        temperature=1.0, top_p=0.9)
    audit = helpers["RequestAudit"](agent.llm_client.chat.completions, folder / "agent-requests.jsonl")
    agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=audit))
    if args.fixture_smoke:
        fixture_calls = [
            dict(name='libreoffice_calc.set_zoom_level', arguments=dict(zoom_percentage=100)),
            dict(name='libreoffice_calc.save', arguments={}),
            dict(name='computer_use', arguments=dict(action='terminate', status='success')),
        ]
        replies = iter('<tool_call>' + json.dumps(call) + '</tool_call>' for call in fixture_calls)
        def create(**kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=next(replies), tool_calls=None))])
        agent.llm_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    protocol = helpers["ExecutedHistoryAgent"](agent)
    memory = helpers["make_memory"](rt / "bank", "task" if args.fixture_smoke else "dual_summary", backend)
    bank_hashes = plan["bank_hashes"]
    staged_input = stage_task_inputs(task, rt / "cache" / task_id,
        os.environ.get("HF_ENDPOINT", ""))
    staged_gold = stage_evaluator_gold(task, rt / "cache" / task_id,
        os.environ.get("HF_ENDPOINT", ""))
    desktop = None
    try:
        desktop = DesktopEnv(action_space="mcp", headless=True, provider_name="docker", os_type="Ubuntu",
            cache_dir=str(rt / "cache"),
            screen_size=(1920, 1080), require_a11y_tree=False, shuffle=False, rag=True)
        original_evaluate = desktop.evaluate
        desktop.evaluate = lambda: helpers["evaluate_with_save_barrier"](desktop, original_evaluate,
            log=lambda event: helpers["append"](folder / "save-barrier.jsonl", event))
        helpers["write_json"](folder / "evaluator-gold.json", staged_gold)
        helpers["write_json"](folder / "task-input.json", staged_input)
        initial_scores = []
        if args.fixture_smoke:
            original_reset = desktop.reset
            def reset_with_negative_check(*reset_args, **reset_kwargs):
                observation = original_reset(*reset_args, **reset_kwargs)
                score = float(desktop.evaluate())
                initial_scores.append(score)
                helpers['write_json'](folder/'negative-control.json', dict(score=score, no_agent_actions=True))
                if score != 0.0:
                    raise RuntimeError('Unmodified official input passed the smoke evaluator')
                return observation
            desktop.reset = reset_with_negative_check
        wrapped, env, runtime = wrap_osworld(protocol, desktop, memory)
        result = helpers["run_episode"](wrapped, env, task, episode_dir,
                                         max_steps=MAX_STEPS, runtime=runtime)
        if (not args.fixture_smoke and audit.calls < 1) or not (episode_dir / "traj.jsonl").is_file():
            raise RuntimeError("Smoke did not complete a model-backed MCP episode")
        if args.fixture_smoke and (initial_scores != [0.0] or result['score'] != 1.0):
            raise RuntimeError('Fixed-action smoke must reject initial input and accept modified output')
        input_file = task["config"][0]["parameters"]["files"][0]
        input_name = f"{uuid.uuid5(uuid.NAMESPACE_URL, input_file['url'])}_{Path(input_file['path']).name}"
        cached_input = Path(desktop.cache_dir) / input_name
        if not cached_input.is_file():
            raise FileNotFoundError("Official runner did not cache the smoke task input")
        import shutil
        saved = []
        artifact_dir = folder / "artifacts"
        artifact_dir.mkdir()
        for name in _result_destinations(task):
            artifact = Path(desktop.cache_dir) / name
            if artifact.is_file():
                target = artifact_dir / name
                shutil.copy2(artifact, target)
                saved.append(dict(name=name, sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
        if not saved:
            raise FileNotFoundError("Official evaluator did not preserve the smoke result artifact")
        if helpers["hashes"](rt / "bank") != bank_hashes:
            raise RuntimeError("Read-only smoke changed the frozen memory bank")
        helpers["write_json"](status_path, dict(state="complete", task=task_id, scored_as_formal=False,
            score=result["score"], decisions=result["decisions"], agent_calls=audit.calls,
            smoke_type='fixed_action_harness_check' if args.fixture_smoke else 'autonomous_agent',
            initial_scores=initial_scores,
            summary_calls=transport.live.calls if transport else 0,
            cached_summary_calls=transport.hits if transport else 0,
            official_input_sha256=hashlib.sha256(cached_input.read_bytes()).hexdigest(),
            task_memory_hit=False, evaluator_gold_staged=staged_gold,
            saved_artifacts=saved, bank_unchanged=True))
    except Exception as exc:
        (folder / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
        helpers["write_json"](status_path, dict(state="failed", task=task_id,
            error_type=type(exc).__name__, error=str(exc), bank_unchanged=helpers["hashes"](rt / "bank") == bank_hashes))
        raise
    finally:
        if desktop is not None:
            owned = list(desktop.provider._owned_qemu)
            desktop.close()
            helpers["write_json"](out / "smoke" / "teardown.json", dict(closed_qemu_pids=owned,
                remaining_owned_qemu=[pid for pid in owned if Path("/proc", str(pid)).exists()]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker", type=int, choices=range(4))
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--prepare-plan", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke-and-launch", action="store_true")
    parser.add_argument("--fixture-smoke", action='store_true')
    parser.add_argument("--stage-assets-only", action="store_true")
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--api-key-file", type=Path,
        default=Path("/data/songweishuai/projs/rh/cofig/model.md"))
    args = parser.parse_args()
    args.runtime_root = args.runtime_root.resolve()
    args.output_root = args.output_root.resolve()
    args.api_key_file = args.api_key_file.resolve()
    if args.api_key_stdin:
        key = sys.stdin.readline().strip()
        if not key:
            raise SystemExit("No API key received on stdin")
        os.environ["OSWORLD_API_KEY"] = key
    elif not os.environ.get("OSWORLD_API_KEY") and args.api_key_file.is_file():
        for line in args.api_key_file.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            if name.strip().lower() in {"key", "api_key", "apikey"} and value.strip():
                os.environ["OSWORLD_API_KEY"] = value.strip().strip("`\"'")
                break
    if args.prepare_plan:
        build_plan(args.runtime_root, args.output_root)
        print("PLAN_FROZEN", args.output_root / "plan.json")
        return
    if args.stage_assets_only:
        helpers = _support(args.runtime_root)
        tasks = json.loads((args.output_root / "tasks-private.json").read_text(encoding="utf-8"))
        smoke_domain, smoke_id = SMOKE_TASK
        smoke_config = json.loads((args.runtime_root / "runner" / "evaluation_examples" / "examples" /
                                   smoke_domain / f"{smoke_id}.json").read_text(encoding="utf-8"))
        specs = [(task["id"], task["config"]) for task in tasks] + [(smoke_id, smoke_config)]
        root = args.runtime_root / "cache"
        root.mkdir(parents=True, exist_ok=True)
        rows = []
        for task_id, config in specs:
            cache = root / task_id
            inputs = stage_task_inputs(config, cache, os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
            gold = stage_evaluator_gold(config, cache, os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
            rows.extend(dict(task_id=task_id, role="input", **item) for item in inputs)
            rows.extend(dict(task_id=task_id, role="evaluator_gold", **item) for item in gold)
        helpers["write_json"](args.output_root / "staged-evaluator-gold.json", rows)
        print("EVALUATOR_GOLD_STAGED", len(rows))
        return
    if args.preflight or (args.smoke_and_launch and not args.fixture_smoke):
        if not os.environ.get("OSWORLD_API_KEY"):
            raise SystemExit("Preflight could not load a model key from the configured key file")
        sys.path[:0] = [str(args.runtime_root / "code"), str(args.runtime_root / "runner"),
                        "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
        from evotree.situation_summary import JsonChatTransport
        plan = json.loads((args.output_root / "plan.json").read_text(encoding="utf-8"))
        try:
            reply = JsonChatTransport(plan["api_url"], plan["model"], os.environ["OSWORLD_API_KEY"], timeout=90)([
                dict(role="system", content="Return JSON only."),
                dict(role="user", content='Return {"ready": true}.')])
            if json.loads(reply).get("ready") is not True:
                raise RuntimeError("Preflight model response did not match the expected JSON")
        except Exception as exc:
            match = re.search(r"HTTP (\d{3})", str(exc))
            status = int(match.group(1)) if match else None
            _support(args.runtime_root)["write_json"](args.output_root / "model-preflight.json",
                dict(passed=False, model=plan["model"], http_status=status,
                     error_type=type(exc).__name__, credentials_persisted=False))
            print("MODEL_PREFLIGHT_FAILED", type(exc).__name__, status or "no_http_status")
            raise SystemExit(2)
        _support(args.runtime_root)["write_json"](args.output_root / "model-preflight.json",
            dict(passed=True, model=plan["model"], credentials_persisted=False))
        print("MODEL_PREFLIGHT_PASSED")
        if args.preflight and not args.smoke_and_launch:
            return
    if args.smoke_and_launch:
        real_task_smoke(args)
        smoke = json.loads((args.output_root/'smoke-status.json').read_text())
        teardown = json.loads((args.output_root/'smoke/teardown.json').read_text())
        if smoke.get('score') != 1.0 or teardown.get('remaining_owned_qemu'):
            raise RuntimeError('Smoke objective success and complete teardown are required before launch')
        if args.fixture_smoke:
            sys.path[:0] = [str(args.runtime_root / "code"), str(args.runtime_root / "runner"),
                            "/data/songweishuai/envs/osworld/lib/python3.12/site-packages"]
            from evotree.situation_summary import JsonChatTransport
            plan = json.loads((args.output_root / "plan.json").read_text(encoding="utf-8"))
            try:
                reply = JsonChatTransport(plan["api_url"], plan["model"],
                    os.environ.get("OSWORLD_API_KEY", ""), timeout=90)([
                        dict(role="system", content="Return JSON only."),
                        dict(role="user", content='Return {"ready": true}.')])
                if json.loads(reply).get("ready") is not True:
                    raise RuntimeError("Model preflight did not return the expected JSON")
            except Exception as exc:
                match = re.search(r"HTTP (\d{3})", str(exc))
                helpers = _support(args.runtime_root)
                helpers["write_json"](args.output_root / "model-preflight.json", dict(
                    passed=False, model=plan["model"], http_status=int(match.group(1)) if match else None,
                    error_type=type(exc).__name__, credentials_persisted=False))
                print("FIXTURE_SMOKE_PASSED_MODEL_PRECHECK_FAILED", type(exc).__name__)
                return
            _support(args.runtime_root)["write_json"](args.output_root / "model-preflight.json",
                dict(passed=True, model=plan["model"], credentials_persisted=False))
        env = dict(os.environ)
        env.update(json.loads((args.runtime_root / "runtime-env.json").read_text(encoding="utf-8")))
        env["OSWORLD_API_KEY"] = os.environ["OSWORLD_API_KEY"]
        with (args.output_root / "supervisor.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--runtime-root", str(args.runtime_root),
                 "--output-root", str(args.output_root), "--supervise"],
                cwd=args.runtime_root, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        _support(args.runtime_root)["write_json"](args.output_root / "launch.json",
            dict(pid=process.pid, credentials_persisted=False, real_task_smoke_passed=True))
        print("OFFICIAL_EXPERIMENT_LAUNCHED", process.pid)
        return
    if args.supervise:
        os.environ.update(json.loads((args.runtime_root / "runtime-env.json").read_text(encoding="utf-8")))
    if not args.supervise and args.worker is None:
        raise SystemExit("choose --supervise or --worker")
    if args.supervise:
        supervise(args)
    elif args.worker is not None:
        raise SystemExit(worker(args))


if __name__ == "__main__":
    main()

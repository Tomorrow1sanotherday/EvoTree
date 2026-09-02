"""Run one isolated arm of the M2 OSWorld comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Iterable, Mapping

ARMS = ("baseline", "frozen", "online")
DEFAULT_PODMAN_SOCKET = "unix:///data/songweishuai/evo-podman.sock"
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"
DEFAULT_MCP_DEP_TIMEOUT = "1800"


def hash_tree(root: Path) -> dict[str, str]:
    """Return stable SHA-256 entries for all regular files below root."""
    if not root.is_dir():
        raise ValueError(f"Memory directory does not exist: {root}")
    hashes: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[path.relative_to(root).as_posix()] = digest
    return hashes


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def build_command(
    *,
    python: Path,
    runner_root: Path,
    evotree_root: Path,
    output_root: Path,
    task_manifest: Path,
    arm: str,
    model: str,
    api_url: str,
    max_steps: int,
) -> list[str]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    arm_root = output_root / arm
    command = [
        str(python),
        str(runner_root / "run_multienv_e2e.py"),
        "--headless",
        "--action_space",
        "mcp",
        "--observation_type",
        "screenshot",
        "--model",
        model,
        "--api_url",
        api_url,
        "--max_steps",
        str(max_steps),
        "--max_trajectory_length",
        str(max_steps),
        "--test_all_meta_path",
        str(task_manifest),
        "--test_config_base_dir",
        str(runner_root / "evaluation_examples"),
        "--result_dir",
        str(arm_root / "results"),
        "--num_envs",
        "1",
        "--trial-id",
        "0",
        "--domain",
        "os",
    ]
    if arm != "baseline":
        command.extend(
            [
                "--use_evotree",
                "--evotree_root",
                str(evotree_root),
                "--evotree_memory_dir",
                str(arm_root / "memory"),
            ]
        )
    if arm == "frozen":
        command.append("--no-evotree_write_enabled")
    return command


def prepare_arm(output_root: Path, arm: str, seed_memory: Path | None) -> Path:
    """Create an arm directory without ever replacing an earlier run."""
    arm_root = output_root / arm
    if arm_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing arm: {arm_root}")
    if arm != "baseline" and (seed_memory is None or not seed_memory.is_dir()):
        raise ValueError(f"{arm} requires an existing --seed-memory directory")
    arm_root.mkdir(parents=True)
    if arm == "baseline":
        return arm_root
    assert seed_memory is not None
    memory_root = arm_root / "memory" / "env-0"
    shutil.copytree(seed_memory, memory_root)
    write_json(arm_root / "memory_before.json", hash_tree(arm_root / "memory"))
    return arm_root


def safe_container_name(label: str, arm: str) -> str:
    normalized = re.sub(r"[^a-z0-9-]+", "-", label.lower()).strip("-")
    normalized = normalized[:40] or "run"
    return f"evo-m2-{normalized}-{arm}"


def runtime_environment(base: Mapping[str, str], container_name: str) -> dict[str, str]:
    env = dict(base)
    env.setdefault("DOCKER_HOST", DEFAULT_PODMAN_SOCKET)
    env.setdefault("HF_ENDPOINT", DEFAULT_HF_ENDPOINT)
    env.setdefault("OSWORLD_MCP_DEP_TIMEOUT", DEFAULT_MCP_DEP_TIMEOUT)
    env.setdefault("OSWORLD_PODMAN", "1")
    env.setdefault("OSWORLD_MCP_INJECT", "1")
    env["EVO_CONTAINER_NAME"] = container_name
    return env


def require_runtime_settings(env: Mapping[str, str], model: str, api_url: str) -> None:
    missing = []
    if not env.get("OSWORLD_API_KEY"):
        missing.append("OSWORLD_API_KEY")
    if not model:
        missing.append("OSWORLD_MODEL or --model")
    if not api_url:
        missing.append("OSWORLD_API_URL or --api-url")
    if missing:
        raise ValueError("Missing runtime settings: " + ", ".join(missing))


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed-memory", type=Path)
    parser.add_argument("--task-manifest", type=Path, default=here / "tasks_smoke.json")
    parser.add_argument("--evotree-root", type=Path, default=repo_root)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", default=os.environ.get("OSWORLD_MODEL", ""))
    parser.add_argument("--api-url", default=os.environ.get("OSWORLD_API_URL", ""))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--label", default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(
        # Keep the venv entrypoint path. Resolving its symlink can bypass the
        # venv and launch the base interpreter instead.
        python=args.python.absolute(),
        runner_root=args.runner_root.resolve(),
        evotree_root=args.evotree_root.resolve(),
        output_root=args.output_root.resolve(),
        task_manifest=args.task_manifest.resolve(),
        arm=args.arm,
        model=args.model,
        api_url=args.api_url,
        max_steps=args.max_steps,
    )
    container_name = safe_container_name(args.label, args.arm)
    if args.dry_run:
        print(shlex.join(command))
        print(f"EVO_CONTAINER_NAME={container_name}")
        return 0

    env = runtime_environment(os.environ, container_name)
    require_runtime_settings(env, args.model, args.api_url)
    if not args.runner_root.joinpath("run_multienv_e2e.py").is_file():
        raise ValueError(f"Invalid runner root: {args.runner_root}")
    if not args.task_manifest.is_file():
        raise ValueError(f"Missing task manifest: {args.task_manifest}")

    arm_root = prepare_arm(
        args.output_root.resolve(),
        args.arm,
        args.seed_memory.resolve() if args.seed_memory else None,
    )
    metadata = {
        "arm": args.arm,
        "container_name": container_name,
        "command": command,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "returncode": None,
    }
    write_json(arm_root / "run.json", metadata)
    returncode = 1
    try:
        completed = subprocess.run(
            command,
            cwd=args.runner_root.resolve(),
            env=env,
            check=False,
        )
        returncode = completed.returncode
    finally:
        if args.arm != "baseline":
            write_json(
                arm_root / "memory_after.json", hash_tree(arm_root / "memory")
            )
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["returncode"] = returncode
        write_json(arm_root / "run.json", metadata)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())

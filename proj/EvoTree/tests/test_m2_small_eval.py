import json
from pathlib import Path

import pytest

from experiments.m2_small_eval.report import summarize_experiment
from experiments.m2_small_eval.run_arm import (
    build_command,
    hash_tree,
    main,
    prepare_arm,
    runtime_environment,
    safe_container_name,
)


def command_for(tmp_path: Path, arm: str) -> list[str]:
    return build_command(
        python=Path("/runtime/python"),
        runner_root=Path("/runner"),
        evotree_root=Path("/evotree"),
        output_root=tmp_path,
        task_manifest=Path("/tasks.json"),
        arm=arm,
        model="model",
        api_url="https://example.invalid/v1",
        max_steps=6,
    )


def test_arm_commands_keep_baseline_bare_and_frozen_read_only(tmp_path):
    baseline = command_for(tmp_path, "baseline")
    frozen = command_for(tmp_path, "frozen")
    online = command_for(tmp_path, "online")

    assert "--use_evotree" not in baseline
    assert "--use_evotree" in frozen
    assert "--no-evotree_write_enabled" in frozen
    assert "--use_evotree" in online
    assert "--no-evotree_write_enabled" not in online
    assert not any("api_key" in value.lower() for value in baseline + frozen + online)


def test_prepare_arm_copies_seed_and_refuses_overwrite(tmp_path):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "tree.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"

    arm_root = prepare_arm(output, "frozen", seed)

    assert (arm_root / "memory" / "env-0" / "tree.json").is_file()
    assert (arm_root / "memory_before.json").is_file()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_arm(output, "frozen", seed)


def test_missing_seed_does_not_reserve_output_directory(tmp_path):
    output = tmp_path / "output"

    with pytest.raises(ValueError, match="requires an existing"):
        prepare_arm(output, "online", tmp_path / "missing")

    assert not (output / "online").exists()


def test_hash_tree_is_relative_and_content_sensitive(tmp_path):
    (tmp_path / "nested").mkdir()
    target = tmp_path / "nested" / "tree.json"
    target.write_text("one", encoding="utf-8")
    first = hash_tree(tmp_path)
    target.write_text("two", encoding="utf-8")
    second = hash_tree(tmp_path)

    assert list(first) == ["nested/tree.json"]
    assert first != second


def write_result(root: Path, arm: str, model: str, task_id: str, action: object) -> None:
    directory = root / arm / "results" / "mcp" / "screenshot" / model / "0" / "os" / task_id
    directory.mkdir(parents=True)
    records = [
        {"step_num": 0, "instruction": "Set volume", "action": "reset"},
        {
            "step_num": 1,
            "instruction": "Set volume",
            "action": action,
            "reward": 0,
            "messages": [
                {"role": "user", "content": "[经验] old historical prompt"},
                {"role": "assistant", "content": "Prior response"},
                {"role": "user", "content": "[经验] 推荐：set volume\n[注意]：verify output"}
            ],
        },
        {"step_num": 2, "instruction": "Set volume", "action": "DONE", "reward": 0},
    ]
    (directory / "traj.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    (directory / "result.txt").write_text("1.0\n", encoding="utf-8")


def write_run(root: Path, arm: str, returncode: int) -> None:
    arm_root = root / arm
    arm_root.mkdir(parents=True, exist_ok=True)
    (arm_root / "run.json").write_text(
        json.dumps({"arm": arm, "returncode": returncode}) + "\n", encoding="utf-8"
    )


def test_report_uses_episode_filtering_and_memory_manifests(tmp_path):
    model = "model"
    task_id = "task-1"
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": [task_id]}), encoding="utf-8")
    for arm in ("baseline", "frozen", "online"):
        write_result(
            tmp_path,
            arm,
            model,
            task_id,
            {"action_type": "os.set_volume", "parameters": {"percent": 100}},
        )
        write_run(tmp_path, arm, 0)
    frozen_manifest = {"env-0/tree.json": "digest"}
    for name in ("memory_before.json", "memory_after.json"):
        (tmp_path / "frozen" / name).write_text(
            json.dumps(frozen_manifest), encoding="utf-8"
        )

    summary = summarize_experiment(tmp_path, manifest, model=model)

    baseline = summary["arms"]["baseline"]
    assert baseline["success_rate"] == 1.0
    assert baseline["mean_effective_actions"] == 1
    assert baseline["tool_actions"] == 1
    assert baseline["gui_actions"] == 0
    assert baseline["experience_prompt_hits"] == 1
    assert baseline["caution_prompt_hits"] == 1
    assert baseline["run_returncode"] == 0
    assert baseline["infrastructure_ok"] is True
    assert baseline["complete"] is True
    assert summary["arms"]["frozen"]["memory_unchanged"] is True
    assert summary["arms"]["frozen"]["memory_changes"] == {
        "before_files": 1,
        "after_files": 1,
        "added_files": 0,
        "removed_files": 0,
        "modified_files": 0,
    }


def test_report_reads_plain_launcher_returncode(tmp_path):
    arm_root = tmp_path / "baseline"
    arm_root.mkdir()
    (arm_root / "returncode").write_text("0\n", encoding="utf-8")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": []}), encoding="utf-8")

    baseline = summarize_experiment(tmp_path, manifest, model="model")["arms"][
        "baseline"
    ]

    assert baseline["run_returncode"] == 0
    assert baseline["infrastructure_ok"] is True


def test_twenty_task_manifest_is_unique_and_fixed():
    manifest = Path(__file__).parents[1] / "experiments" / "m2_small_eval" / "tasks_20.json"
    tasks = json.loads(manifest.read_text(encoding="utf-8"))["os"]

    assert len(tasks) == 20
    assert len(set(tasks)) == 20


def test_report_marks_missing_results(tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": ["missing"]}), encoding="utf-8")

    summary = summarize_experiment(tmp_path, manifest, model="model")

    for arm in ("baseline", "frozen", "online"):
        assert summary["arms"][arm]["completed"] == 0
        assert summary["arms"][arm]["missing"] == 1
        assert summary["arms"][arm]["run_returncode"] is None
        assert summary["arms"][arm]["infrastructure_ok"] is None
        assert summary["arms"][arm]["complete"] is False


def test_report_marks_nonzero_launcher_exit_as_incomplete(tmp_path):
    model = "model"
    task_id = "task-1"
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": [task_id]}), encoding="utf-8")
    write_result(
        tmp_path,
        "baseline",
        model,
        task_id,
        {"action_type": "os.set_volume", "parameters": {"percent": 100}},
    )
    write_run(tmp_path, "baseline", 120)

    baseline = summarize_experiment(tmp_path, manifest, model=model)["arms"]["baseline"]

    assert baseline["completed"] == 1
    assert baseline["success_rate"] == 1.0
    assert baseline["run_returncode"] == 120
    assert baseline["infrastructure_ok"] is False
    assert baseline["complete"] is False


def test_runtime_environment_uses_unique_owned_container_name():
    env = runtime_environment({}, "evo-m2-smoke-online")

    assert env["EVO_CONTAINER_NAME"] == "evo-m2-smoke-online"
    assert env["DOCKER_HOST"].endswith("evo-podman.sock")
    assert env["HF_ENDPOINT"] == "https://hf-mirror.com"
    assert env["OSWORLD_MCP_DEP_TIMEOUT"] == "1800"
    assert safe_container_name("Smoke 01", "frozen") == "evo-m2-smoke-01-frozen"


def test_huggingface_mirror_patch_is_versioned():
    patch = (
        Path(__file__).parents[1]
        / "integrations"
        / "osworld_mcp"
        / "patches"
        / "0004-fix-HuggingFace-task-download-mirror.patch"
    ).read_text(encoding="utf-8")

    assert "HF_ENDPOINT" in patch
    assert "_download_candidates" in patch
    assert "return [mirrored, url]" in patch
    assert "from configured endpoints" in patch


def test_mcp_dependency_diagnostic_patch_is_versioned():
    patch = (
        Path(__file__).parents[1]
        / "integrations"
        / "osworld_mcp"
        / "patches"
        / "0005-diagnose-MCP-dependency-install.patch"
    ).read_text(encoding="utf-8")

    assert "/tmp/mcpdeps.failed" in patch
    assert "/tmp/mcpdeps.log" in patch
    assert "dependency install failed" in patch


def test_dry_run_preserves_requested_venv_entrypoint(tmp_path, capsys):
    python = tmp_path / "venv" / "bin" / "python"

    returncode = main(
        [
            "--arm",
            "baseline",
            "--runner-root",
            str(tmp_path / "runner"),
            "--output-root",
            str(tmp_path / "output"),
            "--python",
            str(python),
            "--model",
            "model",
            "--api-url",
            "https://example.invalid/v1",
            "--dry-run",
        ]
    )

    assert returncode == 0
    assert str(python.absolute()) in capsys.readouterr().out

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

from experiments.m2_small_eval.diagnostic_pilot import (
    ORDERS, hashes, load_package, observation_metadata, run_episode, stage_cached_assets,
)


def test_each_condition_has_each_order_position():
    for position in range(3):
        assert {order[position] for order in ORDERS} == {"baseline", "current", "improved"}


def test_current_and_improved_can_coexist_in_one_process():
    package_root = Path(__file__).resolve().parents[1]
    current = load_package("evotree_diagnostic_test", package_root)
    improved = importlib.import_module("evotree.memory")
    assert current.memory.EvoTreeMemory is not improved.EvoTreeMemory
    assert current.memory.EvoTreeMemory().forest is not improved.EvoTreeMemory().forest


def test_frozen_episode_uses_original_score_and_keeps_recorded_calls(tmp_path, monkeypatch):
    from evotree.distill_osworld import OSWorldDistiller
    from evotree.memory import EvoTreeMemory
    from evotree.osworld_integration import wrap_osworld
    monkeypatch.setattr("experiments.m2_small_eval.diagnostic_pilot.time.sleep", lambda _: None)
    call = {"name": "os.rename_directory", "arguments": {"old_path": "a", "new_path": "b"}}
    terminal = {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}
    prediction = "".join("<tool_call>" + json.dumps(c) + "</tool_call>" for c in (call, terminal))

    class Agent:
        def reset(self, logger):
            pass

        def predict(self, instruction, obs):
            return prediction, [{"action_type": call["name"], "parameters": call["arguments"]}, "DONE"], []

    class Env:
        controller = SimpleNamespace(get_screenshot=lambda: b"after-evaluation")

        def reset(self, **kwargs):
            return self._get_obs()

        def _get_obs(self):
            return {"screenshot": b"image", "app_info": None, "cur_app": "os"}

        def step(self, action, pause):
            return self._get_obs(), 0, action == "DONE", {}

        def evaluate(self):
            return 0  # Model said success; objective evaluator disagrees.

    path = tmp_path / "memory"
    path.mkdir()
    memory = EvoTreeMemory(path, OSWorldDistiller(), write_enabled=False)
    before = hashes(path)
    agent, env, runtime = wrap_osworld(Agent(), Env(), memory)
    result = run_episode(agent, env, {"instruction": "rename"}, tmp_path / "episode", runtime=runtime)
    assert result["score"] == 0
    assert result["decisions"] == 1
    assert len(result["episode_steps"]) == 1
    assert "screenshot" not in result["episode_steps"][0]["observation"]
    assert hashes(path) == before
    records = [json.loads(line) for line in (tmp_path / "episode/traj.jsonl").read_text().splitlines()]
    assert [row["tool_call"] for row in records[1:]] == [call, terminal]
    assert (tmp_path / "episode/result.txt").read_text().strip() == "0.0"


def test_observation_metadata_excludes_visual_and_model_inputs():
    mapped = observation_metadata({"app": "terminal", "screenshot": b"image", "instruction": "secret", "tool_list": []})
    assert mapped == {"app": "terminal"}


def test_stage_only_immutable_inputs_and_gold_not_scored_result(tmp_path):
    import uuid
    source = tmp_path / "source/task"
    source.mkdir(parents=True)
    url = "https://example.test/input.xlsx"
    name = f"{uuid.uuid5(uuid.NAMESPACE_URL, url)}_input.xlsx"
    for filename, data in [(name, b"input"), ("gold.xlsx", b"gold"), ("input.xlsx", b"old result")]:
        (source / filename).write_bytes(data)
    config = {"id": "task", "config": [{"type": "download", "parameters": {
        "files": [{"url": url, "path": "/home/user/input.xlsx"}],
    }}], "evaluator": {"expected": {"type": "cloud_file", "path": "https://example.test/gold", "dest": "gold.xlsx"}}}
    result = stage_cached_assets(config, source.parent, tmp_path / "destination")
    assert {item["name"] for item in result} == {name, "gold.xlsx"}
    assert not (tmp_path / "destination/task/input.xlsx").exists()

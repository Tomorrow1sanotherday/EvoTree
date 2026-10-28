import json
from pathlib import Path
import sys

import pytest

from experiments.m2_small_eval.parallel_pilot import completed_tasks, main, shard_tasks, validate_executed_actions
from experiments.m2_small_eval.run_arm import runtime_environment


def test_parallel_workers_use_nonoverlapping_service_ranges():
    ranges = []
    for slot in range(4):
        env = runtime_environment({}, f"evo-test-slot-{slot}", slot)
        for service in ("SERVER", "CHROMIUM", "VLC", "VNC", "MONITOR"):
            base = int(env[f"OSWORLD_{service}_PORT_BASE"])
            ports = set(range(base, base + int(env["OSWORLD_PORT_RANGE_SIZE"])))
            assert not any(ports & previous for previous in ranges)
            ranges.append(ports)
    with pytest.raises(ValueError):
        runtime_environment({}, "evo-test", 16)


def test_task_shards_cover_each_task_once_and_preserve_domain():
    tasks = [("os", "a"), ("libreoffice_calc", "b"), ("os", "c"), ("libreoffice_impress", "d")]
    shards = shard_tasks(tasks, 2)
    assert shards == [{"os": ["a", "c"]}, {"libreoffice_calc": ["b"], "libreoffice_impress": ["d"]}]
    assert shards == shard_tasks(tasks, 2)
    with pytest.raises(ValueError, match="duplicate"):
        shard_tasks(tasks + [tasks[0]], 2)


def test_silent_discard_of_operations_before_done_invalidates_run(tmp_path):
    calls = [{"name": "libreoffice_calc.save", "arguments": {}},
             {"name": "computer_use", "arguments": {"action": "terminate", "status": "success"}}]
    prediction = "".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)
    rows = [{"step_num": 1, "prediction": prediction, "action": "DONE"}]
    with pytest.raises(RuntimeError, match="only 1 executed"):
        validate_executed_actions(rows, tmp_path / "traj.jsonl")
    correct = [{"step_num": 1, "prediction": prediction,
                "action": {"action_type": "libreoffice_calc.save", "parameters": {}}}, rows[0]]
    validate_executed_actions(correct, tmp_path / "traj.jsonl")


def test_dispatch_validation_stops_at_first_terminal_and_checks_tool_order(tmp_path):
    calls = [{"name": "libreoffice_calc.save", "arguments": {}},
             {"name": "computer_use", "arguments": {"action": "terminate", "status": "failure"}},
             {"name": "libreoffice_calc.env_info", "arguments": {}}]
    prediction = "".join("<tool_call>" + json.dumps(call) + "</tool_call>" for call in calls)
    rows = [{"step_num": 1, "prediction": prediction,
             "action": {"action_type": "libreoffice_calc.save", "parameters": {}}},
            {"step_num": 1, "prediction": prediction, "action": "FAIL"}]
    validate_executed_actions(rows, tmp_path / "traj.jsonl")
    rows[0]["action"]["action_type"] = "libreoffice_calc.env_info"
    with pytest.raises(RuntimeError, match="did not execute"):
        validate_executed_actions(rows, tmp_path / "traj.jsonl")


def test_swallowed_runner_exception_invalidates_existing_score(tmp_path):
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": ["task"]}), encoding="utf-8")
    arm = tmp_path / "online"
    result = arm / "results" / "mcp" / "screenshot" / "model" / "0" / "os" / "task"
    result.mkdir(parents=True)
    (arm / "run.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
    (result / "result.txt").write_text("0", encoding="utf-8")
    (result / "traj.jsonl").write_text(json.dumps({"Error": "Time limit exceeded"})+"\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Infrastructure error"):
        completed_tasks(tmp_path, manifest, "model", "online")


def test_repeated_pilot_carries_online_memory_and_keeps_baseline_bare(tmp_path, monkeypatch):
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "run_multienv_e2e.py").write_text('''
import json
from pathlib import Path
import sys
args = sys.argv
def value(flag): return args[args.index(flag)+1]
tasks=json.loads(Path(value('--test_all_meta_path')).read_text())
generation=0
if '--use_evotree' in args:
    memory=Path(value('--evotree_memory_dir'))/'env-0'
    memory.mkdir(parents=True,exist_ok=True)
    state=memory/'generation.json'
    if state.exists(): generation=json.loads(state.read_text())
    state.write_text(json.dumps(generation+1))
for domain, ids in tasks.items():
    for task in ids:
        root=Path(value('--result_dir'))/'mcp'/'screenshot'/value('--model')/'0'/domain/task
        root.mkdir(parents=True)
        rows=[{'instruction':'Rename directory','action':'reset','step_num':0},
              {'instruction':'Rename directory','action':{'action_type':'os.rename_directory','parameters':{}},'step_num':1}]
        (root/'traj.jsonl').write_text(''.join(json.dumps(row)+'\\n' for row in rows))
        (root/'result.txt').write_text('1.0')
        (root/'generation_seen.json').write_text(json.dumps(generation))
''', encoding="utf-8")
    manifest = tmp_path / "tasks.json"
    manifest.write_text(json.dumps({"os": ["a", "b"]}), encoding="utf-8")
    monkeypatch.setenv("OSWORLD_API_KEY", "test-placeholder")
    repo = Path(__file__).parents[1]
    output = tmp_path / "pilot"
    assert main(["--runner-root", str(runner), "--output-root", str(output),
                 "--task-manifest", str(manifest), "--evotree-root", str(repo),
                 "--python", sys.executable, "--model", "model", "--api-url", "https://example.invalid",
                 "--workers", "2", "--rounds", "2"]) == 0
    for shard in range(2):
        for round_number in (1, 2):
            root = output / f"pass-{round_number}" / f"shard-{shard}"
            state = root / "online" / "memory" / "env-0" / "generation.json"
            assert json.loads(state.read_text()) == round_number
            assert not (root / "baseline" / "memory").exists()
            for arm in ("baseline", "online"):
                seen = next((root / arm).rglob("generation_seen.json"))
                assert json.loads(seen.read_text()) == (round_number - 1 if arm == "online" else 0)
    assert json.loads((output / "status.json").read_text())["state"] == "complete"


def test_resume_keeps_completed_baseline_and_archives_failed_attempt(tmp_path, monkeypatch):
    runner=tmp_path/'runner'
    runner.mkdir()
    (runner/'fail-once').touch()
    (runner/'run_multienv_e2e.py').write_text('''
import json,sys
from pathlib import Path
args=sys.argv
def value(flag):return args[args.index(flag)+1]
for domain,ids in json.loads(Path(value('--test_all_meta_path')).read_text()).items():
 for task in ids:
  root=Path(value('--result_dir'))/'mcp/screenshot'/value('--model')/'0'/domain/task
  if (root/'result.txt').exists():continue
  root.mkdir(parents=True)
  rows=[{'instruction':'rename','step_num':0,'action':'reset'}]
  if task=='b' and Path('fail-once').exists():
   rows.append({'Error':'original parser exception'})
  else:
   rows.append({'step_num':1,'action':{'action_type':'os.rename_directory','parameters':{}}})
   (root/'result.txt').write_text('1')
  (root/'traj.jsonl').write_text(''.join(json.dumps(r)+'\\n' for r in rows))
''',encoding='utf-8')
    manifest=tmp_path/'tasks.json';manifest.write_text(json.dumps({'os':['a','b']}))
    monkeypatch.setenv('OSWORLD_API_KEY','test-placeholder')
    output=tmp_path/'output'
    args=['--runner-root',str(runner),'--output-root',str(output),'--task-manifest',str(manifest),
          '--evotree-root',str(Path(__file__).parents[1]),'--python',sys.executable,
          '--model','model','--api-url','https://example.invalid','--workers','1','--rounds','2']
    with pytest.raises(RuntimeError,match='Infrastructure error'):
        main(args)
    valid=output/'pass-1/shard-0/baseline/results/mcp/screenshot/model/0/os/a/traj.jsonl'
    original=valid.read_bytes();original_mtime=valid.stat().st_mtime_ns
    (runner/'fail-once').unlink()
    assert main(args+['--resume'])==0
    assert valid.read_bytes()==original and valid.stat().st_mtime_ns==original_mtime
    archived=list((output/'pass-1/shard-0/baseline/resume-history').glob('*/os/b/traj.jsonl'))
    assert len(archived)==1 and 'original parser exception' in archived[0].read_text()
    assert json.loads((output/'status.json').read_text())['state']=='complete'
    assert len(json.loads((output/'summary.json').read_text())['rounds'])==2

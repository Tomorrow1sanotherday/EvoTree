from experiments.m2_small_eval.official_mcp_eval import (
    CONDITIONS, SELECTED_TASKS, SMOKE_TASK, _result_destinations, make_schedule,
    model_lock_path, stage_evaluator_gold, stage_task_inputs, summarize,
)
import hashlib
import json
import pytest
from pathlib import Path


def test_network_audit_counts_failures_and_does_not_replay_empty_responses(tmp_path):
    from experiments.m2_small_eval.official_mcp_eval import SummaryRequestAudit
    replies = iter([None, TimeoutError('test'), '{"ready":true}'])
    def transport(messages):
        value = next(replies)
        if isinstance(value, Exception):
            raise value
        return value
    audit = SummaryRequestAudit(transport, tmp_path)
    assert audit([]) is None
    with pytest.raises(TimeoutError):
        audit([])
    assert audit([]) == '{"ready":true}'
    assert audit.calls == 3 and audit.failures == 1
    assert json.loads((tmp_path / '00002.json').read_text())['error_type'] == 'TimeoutError'


def test_stale_evaluator_output_is_removed_but_gold_is_preserved(tmp_path):
    from experiments.m2_small_eval.official_mcp_eval import clear_result_cache
    (tmp_path / 'result.xlsx').write_bytes(b'previous result')
    (tmp_path / 'gold.xlsx').write_bytes(b'gold')
    clear_result_cache({'evaluator': {'result': {'dest': 'result.xlsx'}}}, tmp_path)
    assert not (tmp_path / 'result.xlsx').exists()
    assert (tmp_path / 'gold.xlsx').read_bytes() == b'gold'


def test_official_fail_zero_does_not_require_skipped_getter_output(tmp_path):
    from experiments.m2_small_eval.official_mcp_eval import preserve_result_artifacts
    config = {'evaluator': {'func': 'compare_table', 'result': {'dest': 'result.xlsx'}}}
    audit = preserve_result_artifacts(config, tmp_path, tmp_path,
        score=0.0, action_history=['read', 'FAIL'])
    assert audit['artifact_status'] == 'official_fail_skipped_result_getter'
    assert audit['missing_artifacts'] == ['result.xlsx']
    assert audit['official_score'] == 0.0
    assert json.loads((tmp_path / 'evaluation-audit.json').read_text()) == audit


@pytest.mark.parametrize('score,history', [(0, ['DONE']), (0, []), (1, ['FAIL']), (1, ['DONE'])])
def test_missing_artifact_is_not_waived_for_other_outcomes(tmp_path, score, history):
    from experiments.m2_small_eval.official_mcp_eval import preserve_result_artifacts
    with pytest.raises(FileNotFoundError):
        preserve_result_artifacts({'evaluator': {'func': 'compare_table',
            'result': {'dest': 'result.xlsx'}}}, tmp_path, tmp_path,
            score=score, action_history=history)


def test_all_expected_artifacts_must_be_preserved(tmp_path):
    from experiments.m2_small_eval.official_mcp_eval import preserve_result_artifacts
    cache, output = tmp_path / 'cache', tmp_path / 'output'
    cache.mkdir()
    output.mkdir()
    (cache / 'result.xlsx').write_bytes(b'result')
    config = {'evaluator': {'func': 'compare_table',
        'result': {'dest': ['result.xlsx', 'result.csv']}}}
    with pytest.raises(FileNotFoundError):
        preserve_result_artifacts(config, cache, output, score=0, action_history=['DONE'])
    (cache / 'result.csv').write_bytes(b'csv result')
    audit = preserve_result_artifacts(config, cache, output, score=0, action_history=['DONE'])
    assert audit['artifact_status'] == 'preserved'
    assert len(audit['saved_artifacts']) == 2
    assert audit['missing_artifacts'] == []
    assert (output / 'result.xlsx').read_bytes() == b'result'


def test_official_task_schedule_is_paired_balanced_and_reproducible():
    schedule = make_schedule()
    assert schedule == make_schedule()
    assert sum(map(len, schedule)) == len(SELECTED_TASKS) == 12
    assert all(len(worker) == 3 for worker in schedule)
    assert SMOKE_TASK == ("libreoffice_calc", "1334ca3e-f9e3-4db8-9ca7-b4c653be7d17")
    assert SMOKE_TASK not in {(domain, task_id) for _, domain, task_id in SELECTED_TASKS}
    for family in ("calc", "impress"):
        jobs = [job for worker in schedule for job in worker if job["family"] == family]
        assert len(jobs) == 6
        assert sum(job["order"] == list(CONDITIONS) for job in jobs) == 3
        assert sum(job["order"] == list(reversed(CONDITIONS)) for job in jobs) == 3


def test_summary_counts_paired_successes_and_discordant_outcomes():
    rows = []
    for i, (family, _, task_id) in enumerate(SELECTED_TASKS):
        baseline = float(i % 3 != 0)
        dual = float(i % 4 != 0)
        rows.extend([
            dict(task_id=task_id, family=family, condition="baseline", score=baseline),
            dict(task_id=task_id, family=family, condition="dual_summary", score=dual),
        ])
    result = summarize(rows)
    assert result["episodes"] == 24
    assert result["paired"]["all"]["tasks"] == 12
    assert result["paired"]["all"]["dual_wins"] == 3
    assert result["paired"]["all"]["dual_losses"] == 2


def test_evaluator_gold_uses_existing_host_cache_without_network(tmp_path):
    cache = tmp_path / "task"
    cache.mkdir()
    payload = b"official evaluator gold"
    csv_payload = b"official evaluator csv gold"
    (cache / "gold.xlsx").write_bytes(payload)
    (cache / "gold.csv").write_bytes(csv_payload)
    staged = stage_evaluator_gold({"evaluator": {"expected": {
        "type": "cloud_file", "multi": True,
        "path": ["https://huggingface.co/example/gold.xlsx", "https://huggingface.co/example/gold.csv"],
        "dest": ["gold.xlsx", "gold.csv"],
    }}}, cache, "https://hf-mirror.com")
    assert staged == [
        dict(name="gold.xlsx", source="existing_official_cache", sha256=hashlib.sha256(payload).hexdigest()),
        dict(name="gold.csv", source="existing_official_cache", sha256=hashlib.sha256(csv_payload).hexdigest()),
    ]


def test_task_input_uses_osworld_uuid5_cache_key(tmp_path):
    from pathlib import Path
    import uuid

    config = {"config": [{"type": "download", "parameters": {"files": [
        {"url": "https://huggingface.co/example/input.xlsx", "path": "/home/user/input.xlsx"}
    ]}}]}
    cache = tmp_path / "task"
    cache.mkdir()
    name = f"{uuid.uuid5(uuid.NAMESPACE_URL, 'https://huggingface.co/example/input.xlsx')}_input.xlsx"
    payload = b"official input"
    (cache / name).write_bytes(payload)
    staged = stage_task_inputs(config, cache, "https://hf-mirror.com")
    assert staged == [dict(name=name, source="existing_official_cache",
                           sha256=hashlib.sha256(payload).hexdigest())]


def test_request_lock_is_local_and_result_destinations_cover_multi_file_outputs(tmp_path):
    lock = model_lock_path(tmp_path)
    assert lock.parent == Path('/tmp')
    assert lock != model_lock_path(tmp_path / 'other')
    assert _result_destinations({"evaluator": {"result": {
        "type": "vm_file", "multi": True, "dest": ["book.xlsx", "book.csv"]
    }}}) == ["book.xlsx", "book.csv"]

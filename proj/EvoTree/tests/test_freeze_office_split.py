from experiments.m2_small_eval import freeze_office_split as module


def task(task_id, instruction, source_url):
    return {
        "id": task_id,
        "domain": "libreoffice_calc",
        "instruction": instruction,
        "mcp_listed": True,
        "config": {"config": [{"type": "download", "parameters": {
            "files": [{"url": source_url}]}}]},
        "config_sha256": task_id,
    }


def test_freeze_excludes_ids_near_duplicates_and_shared_assets():
    source = task("source", "Copy the Revenue column to Sheet2.", "https://asset/source")
    inventory = [
        source,
        task("near", "Copy the Revenue column to Sheet2", "https://asset/other"),
        task("asset", "Format the worksheet title.", "https://asset/source"),
        task("fresh", "Insert a pivot table counting invoices.", "https://asset/fresh"),
    ]
    result = module.freeze(inventory, [{"task_id": "source"}])
    assert [row["task_id"] for row in result["candidates"]] == ["fresh"]
    reasons = {row["task_id"]: row["reasons"] for row in result["excluded"]}
    assert "source-task-id" in reasons["source"]
    assert any(item["reason"] == "near-duplicate-instruction" for item in reasons["near"])
    assert any(item["reason"] == "shared-setup-asset-url" for item in reasons["asset"])


def test_documentation_and_gold_urls_do_not_exclude_independent_inputs():
    source = task('source', 'Clean movie titles.', 'https://input/source')
    fresh = task('fresh', 'Change the worksheet orientation.', 'https://input/fresh')
    for row in (source, fresh):
        row['config']['source'] = 'https://docs/shared'
        row['config']['evaluator'] = {'expected': {'path': 'https://gold/shared'}}
    result = module.freeze([source, fresh], [{'task_id': 'source'}],
                           exposure={'fresh': ['runs/observed/traj.jsonl']})
    assert result['candidate_count'] == 1
    assert result['fresh_candidate_count'] == 0
    assert result['candidates'][0]['prior_exposure']


def test_missing_source_fails_and_non_mcp_task_is_excluded():
    import pytest
    source = task('source', 'Clean movie titles.', 'https://input/source')
    with pytest.raises(ValueError, match='source task IDs'):
        module.freeze([source], [{'task_id': 'missing'}])
    other = task('other', 'Change worksheet orientation.', 'https://input/other')
    other['mcp_listed'] = False
    result = module.freeze([source, other], [{'task_id': 'source'}])
    assert not result['candidates']

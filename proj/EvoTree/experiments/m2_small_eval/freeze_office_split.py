"""Freeze a source-disjoint Office transfer split from an official inventory.

This is an audit artifact only. It does not assign test labels or alter the
frozen experience bank. A held-out task is eligible only when its official
task ID, normalized instruction, and setup asset URLs are disjoint from every
source task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path


OFFICE_DOMAINS = {"libreoffice_calc", "libreoffice_impress"}
TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_instruction(text: str) -> str:
    return " ".join(TOKEN_RE.findall(text.lower()))


def asset_urls(task: dict) -> set[str]:
    # Only files copied into the VM are inputs. Documentation links and
    # evaluator expected files must not create spurious input collisions.
    return {file['url'] for step in task.get('config', {}).get('config', [])
            if step.get('type') == 'download'
            for file in step.get('parameters', {}).get('files', [])
            if isinstance(file.get('url'), str)}


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalize_instruction(left),
                           normalize_instruction(right)).ratio()


def freeze(inventory: list[dict], provenance: dict | list, threshold: float = .90,
           exposure: dict | None = None) -> dict:
    source_rows = (provenance if isinstance(provenance, list)
                   else provenance.get("sources", []))
    source_ids = {row["task_id"] if "task_id" in row else row.get("id")
                  for row in source_rows if isinstance(row, dict)}
    source_ids.discard(None)
    source_tasks = [row for row in inventory if row.get("id") in source_ids]
    if not source_ids or {row['id'] for row in source_tasks} != source_ids:
        raise ValueError('All nonempty source task IDs must occur in the inventory')
    source_urls = set().union(*(asset_urls(row) for row in source_tasks))
    candidates, excluded = [], []
    for row in inventory:
        if row.get("domain") not in OFFICE_DOMAINS:
            continue
        reasons = []
        if not row.get('mcp_listed'):
            reasons.append('not-in-official-mcp-manifest')
        if row.get("id") in source_ids:
            reasons.append("source-task-id")
        normalized = normalize_instruction(row.get("instruction", ""))
        for source in source_tasks:
            score = similarity(row.get("instruction", ""), source.get("instruction", ""))
            if score >= threshold:
                reasons.append({"reason": "near-duplicate-instruction",
                                "source_task_id": source["id"], "similarity": round(score, 6)})
        shared = sorted(asset_urls(row) & source_urls)
        if shared:
            reasons.append({"reason": "shared-setup-asset-url", "urls": shared})
        if reasons:
            excluded.append({"task_id": row.get("id"), "instruction": row.get("instruction"),
                             "reasons": reasons})
        else:
            candidates.append({"task_id": row["id"], "domain": row["domain"],
                               "instruction": row["instruction"],
                               "instruction_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
                               "config_sha256": row.get("config_sha256"),
                               "prior_exposure": list((exposure or {}).get(row['id'], [])),
                               "fresh_by_archived_id": exposure is not None and not exposure.get(row['id']),
                               "asset_urls": sorted(asset_urls(row))})
    return {
        "method": {"office_domains": sorted(OFFICE_DOMAINS),
                   "near_duplicate_threshold": threshold,
                   "exclusions": ["source-task-id", "near-duplicate-instruction",
                                  "shared-setup-asset-url"]},
        "source_task_ids": sorted(source_ids),
        "source_count_in_inventory": len(source_tasks),
        "candidate_count": len(candidates),
        "excluded_count": len(excluded),
        "fresh_candidate_count": sum(r['fresh_by_archived_id'] for r in candidates),
        "exposure_scan_supplied": exposure is not None,
        "limitations": ["Instruction similarity is lexical, not proof against paraphrases.",
                        "Input URLs are checked; equal contents at different URLs require a separate hash audit.",
                        "Freshness covers archived task IDs in the supplied exposure scan, not unknown outside runs."],
        "candidates": candidates,
        "excluded": excluded,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--exposure', type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    provenance = json.loads(args.provenance.read_text(encoding="utf-8"))
    exposure = json.loads(args.exposure.read_text(encoding='utf-8')) if args.exposure else None
    result = freeze(inventory, provenance, exposure=exposure)
    result['input_sha256'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in dict(inventory=args.inventory, provenance=args.provenance,
                               exposure=args.exposure).items() if path is not None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in
                      ("source_count_in_inventory", "candidate_count", "excluded_count")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

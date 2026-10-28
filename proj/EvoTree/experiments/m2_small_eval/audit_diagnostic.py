"""Audit and freeze a completed three-condition diagnostic without rescoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.m2_small_eval.diagnostic_pilot import TASKS, hashes, write_json
from experiments.m2_small_eval.parallel_pilot import validate_executed_actions


def audit(root):
    assert json.loads((root / "status.json").read_text())["state"] == "complete"
    rows = []
    provenance = {}
    permission = []
    for index, (_, task, _) in enumerate(TASKS):
        directory = root / f"task-{index}"
        state = json.loads((directory / "status.json").read_text())
        assert state["state"] == "complete" and state["completed"] == 9
        saved_hashes = json.loads((directory / "seed-hashes-before.json").read_text())
        for name, expected in saved_hashes.items():
            assert hashes(directory / f"seed-{name}") == expected, f"Seed changed: {index}/{name}"
        recorded = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
        assert len(recorded) == 9
        keys = set()
        for row in recorded:
            key = (row["repetition"], row["condition"])
            assert key not in keys
            keys.add(key)
            result = directory / f"rep-{key[0]}" / key[1]
            score = float((result / "result.txt").read_text())
            assert score == row["score"]
            trajectory = [json.loads(line) for line in (result / "traj.jsonl").read_text().splitlines() if line.strip()]
            assert not any("error" in r or "Error" in r or str(r.get("prediction", "")).strip().lower() == "client error" for r in trajectory)
            validate_executed_actions(trajectory, result / "traj.jsonl")
            decisions = [json.loads(line) for line in (result / "decisions.jsonl").read_text().splitlines() if line.strip()]
            enriched = dict(row, task_index=index, effective_actions=sum(r.get("action") not in (None, "reset", "DONE", "FAIL", "WAIT") for r in trajectory),
                recommendation_decisions=sum(bool(r["recommendations"]) for r in decisions),
                advice_chars=[len(r["text"]) for r in decisions])
            rows.append(enriched)
            for name in ("traj.jsonl", "result.txt", "decisions.jsonl"):
                path = result / name
                provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            if index == 0:
                from experiments.m2_small_eval.permission_order_probe import preparation_order
                first = next(r for r in trajectory if isinstance(r.get("prediction"), str))
                permission.append(dict(repetition=key[0], condition=key[1], score=score,
                    initial_advice=decisions[0]["text"], **preparation_order(first["prediction"])))
        assert keys == {(rep, name) for rep in range(1, 4) for name in ("baseline", "current", "improved")}
    summary = {}
    by_task = {}
    for name in ("baseline", "current", "improved"):
        selected = [r for r in rows if r["condition"] == name]
        summary[name] = dict(completed=len(selected), successes=sum(r["score"] == 1 for r in selected),
            mean_score=sum(r["score"] for r in selected) / len(selected),
            model_decisions=sum(r["decisions"] for r in selected),
            recommendation_decisions=sum(r["recommendation_decisions"] for r in selected))
        for index in range(3):
            selected_task = [r for r in selected if r["task_index"] == index]
            by_task.setdefault(str(index), {})[name] = [r["score"] for r in sorted(selected_task, key=lambda r: r["repetition"])]
    report = dict(setting="same-task frozen successful demonstration; not cold-start evolution or transfer",
        audited_episodes=len(rows), frozen_seeds_unchanged=True, summary=summary, by_task=by_task,
        permission_initial_order=permission, episodes=rows, preserved_files=provenance)
    write_json(root / "final_audit.json", report)
    lines = ["# 三任务成功种子诊断：完整审计", "", "这是同任务、固定成功种子的冻结复用实验，不能作为从空记忆自学或新任务迁移的证据。", "",
        "27/27 轨迹完成执行记录检查；三份种子的两个条件均未改写。原 evaluator 和保存等待逻辑在各条件相同。", "",
        "| 条件 | 成功/9 | 成功率 |", "|---|---:|---:|"]
    for name, row in summary.items():
        lines.append(f"| {name} | {row['successes']}/9 | {row['mean_score']:.3f} |")
    lines.extend(["", "| 任务 | baseline 成功/3 | current 成功/3 | improved 成功/3 |", "|---|---:|---:|---:|"])
    for index, label in enumerate(("文件权限", "Revenue 复制", "字号")):
        scores = by_task[str(index)]
        lines.append(f"| {label} | {sum(scores['baseline']):.0f}/3 | {sum(scores['current']):.0f}/3 | {sum(scores['improved']):.0f}/3 |")
    lines.extend(["", "权限任务的动作顺序见 final_audit.json；首步点击是流程代理指标，不等同于点击目标正确。", "",
        "未加入普通成功步骤条件，因此本表即使记忆臂较好，也不能证明统计或树结构优于直接示范。样本仅三个任务，每条件各三次。"])
    (root / "final_audit.md").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    value = audit(args.root)
    print(json.dumps(dict(audited_episodes=value["audited_episodes"], summary=value["summary"])))

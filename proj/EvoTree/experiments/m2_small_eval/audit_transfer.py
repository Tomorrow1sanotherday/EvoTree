"""Audit held-out outputs without changing scores or writing training memory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from evotree.distill_osworld import parse_tool_calls
from experiments.m2_small_eval.diagnostic_pilot import hashes, write_json
from experiments.m2_small_eval.parallel_pilot import validate_executed_actions


def audit(root, bank, case_root):
    status=json.loads((root/"status.json").read_text())
    assert status["state"]=="complete" and status["completed"]==36
    assert hashes(bank)==json.loads((bank.parent/"bank-hashes.json").read_text())
    cases={case["id"]:case for case in json.loads((case_root/"cases-private.json").read_text())}
    rows=[];seen=set();preserved={}
    for path in root.glob("worker-*/results.jsonl"):
        for line in path.read_text().splitlines():
            row=json.loads(line);key=(row["case"],row["condition"],row["repetition"])
            assert key not in seen
            seen.add(key)
            directory=path.parent/row["case"]/f"rep-{row['repetition']}"/row["condition"]
            score=float((directory/"result.txt").read_text())
            assert score==row["score"]
            records=[json.loads(line) for line in (directory/"traj.jsonl").read_text().splitlines() if line.strip()]
            validate_executed_actions(records,directory/"traj.jsonl")
            assert not any("error" in r or "Error" in r or str(r.get("prediction","")).lower()=="client error" for r in records)
            calls=[];turns=set()
            for record in records:
                prediction=record.get("prediction")
                turn=(record.get("step_num"),prediction)
                if isinstance(prediction,str) and turn not in turns:
                    turns.add(turn);calls.extend(parse_tool_calls(prediction))
            stale=[]
            for call in calls:
                args=call["arguments"]
                # Report observable old-value reuse; this is a diagnostic flag,
                # not proof that an action was semantically wrong or causal.
                markers=[]
                if cases[row["case"]]["family"]=="calc":
                    for name,value in args.items():
                        if isinstance(value,str) and value in ("Sheet1","Sheet2","A1:A20"):
                            markers.append(name+"="+value)
                else:
                    if args.get("size") in (60,28):markers.append("size="+str(args["size"]))
                    if args.get("page_index")==14 and cases[row["case"]]["source_variant"]["slide"]!=14:
                        markers.append("page_index=14")
                if markers:stale.append(dict(tool=call["name"],markers=markers))
            decisions=[json.loads(line) for line in (directory/"decisions.jsonl").read_text().splitlines()]
            enriched=dict(row,old_value_reuse_flags=stale,
                recommendation_decisions=sum(bool(r["recommendations"]) for r in decisions),
                model_decisions=len(decisions),heldout_instruction=cases[row["case"]]["config"]["instruction"])
            rows.append(enriched)
            for filename in ("traj.jsonl","result.txt","decisions.jsonl"):
                p=directory/filename;preserved[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    expected={(case,condition,rep) for case in cases for condition in ("baseline","workflow","tree") for rep in (1,2)}
    assert seen==expected
    summary={condition:dict(completed=12,successes=sum(r["condition"]==condition and r["score"]==1 for r in rows),
        episodes_reusing_old_values=sum(r["condition"]==condition and bool(r["old_value_reuse_flags"]) for r in rows))
        for condition in ("baseline","workflow","tree")}
    result=dict(audited_episodes=len(rows),frozen_bank_unchanged=True,summary=summary,episodes=rows,preserved_files=preserved,
        scope="six custom held-out same-template cases; two repeats; not official or cross-template generalization")
    write_json(root/"final_audit.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","bank","case-root"):parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args()
    value=audit(args.root,args.bank,args.case_root)
    print(json.dumps(value["summary"]))

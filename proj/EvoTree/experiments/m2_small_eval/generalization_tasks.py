"""Build a fresh, fixed 24-instance same-template transfer set."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from experiments.m2_small_eval.diagnostic_pilot import write_json
from experiments.m2_small_eval.transfer_tasks import make_cases


LABELS = (
    "Backorder Value", "Freight Charge", "Quality Score", "Net Sales",
    "Return Cost", "Lead Time", "Gross Margin", "Tax Amount",
    "Units Shipped", "Service Level", "Forecast Error", "Labor Cost",
)
LEVELS = ("parameters", "layout", "context")


def specifications():
    calc, font = [], []
    for level_index, level in enumerate(LEVELS):
        for variant in range(4):
            index = level_index * 4 + variant
            calc.append(dict(
                id=f"generalization2-calc-{level}-{variant + 1}", level=level,
                label=LABELS[index],
                column=2 if level == "parameters" else variant + 1,
                count=11 + index * 2, sheet=f"Ledger{index + 31}",
                target=f"Summary{index + 61}",
                start="Scratch" if level == "context" else f"Ledger{index + 31}"))
            font.append(dict(
                id=f"generalization2-font-{level}-{variant + 1}", level=level,
                slide=14 if level == 'parameters' else 2 + index,
                sizes=[34 + index * 2, 17 + index],
                boxes=[6, 7] if level == 'parameters' else [variant, variant + 1],
                text_prefix=f"Review {index + 101}: ", use_original=False))
    return calc, font


def build(output: Path, base_url: str):
    calc, font = specifications()
    cases = make_cases(output, None, base_url, calc_variants=calc, font_variants=font)
    cases = [case for pair in zip(cases[:12], cases[12:]) for case in pair]
    assert len(cases) == 24 and len({case["id"] for case in cases}) == 24
    assert all(case["input_sha256"] != case["gold_sha256"] for case in cases)
    assert len({c['input_sha256'] for c in cases}) == 24
    write_json(output / 'cases-private.json', cases)
    write_json(output / "split.json", dict(
        setting="24 fresh instances; fixed before any model execution",
        training="two successful measured seed episodes only",
        prior_case_ids_excluded=True, official_benchmark=False,
        test=[dict(id=c["id"], family=c["family"], level=c["level"],
                   instruction=c["config"]["instruction"],
                   input_sha256=c["input_sha256"], gold_sha256=c["gold_sha256"])
              for c in cases],
        arms=["task", "dual_summary"], repetitions=1,
        primary_metric="full objective evaluator success"))
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18773")
    args = parser.parse_args()
    print(json.dumps(dict(cases=len(build(args.output, args.base_url)))))

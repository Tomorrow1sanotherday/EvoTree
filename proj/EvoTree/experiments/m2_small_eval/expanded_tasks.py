"""Predeclared 24 fresh same-template tasks; no score-based task selection."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from experiments.m2_small_eval.diagnostic_pilot import write_json
from experiments.m2_small_eval.transfer_tasks import make_cases


LABELS = ("Net Profit", "Shipping Fee", "Inventory Value", "Discount Rate",
          "Refund Total", "Order Count", "Gross Margin", "Energy Usage",
          "Labor Hours", "Service Charge", "Return Units", "Forecast Value")
LEVELS = ("parameters", "layout", "context")


def specifications():
    calc, font = [], []
    for level_index, level in enumerate(LEVELS):
        for variant in range(4):
            index = level_index * 4 + variant
            source = f"Records{index + 21}"
            calc.append(dict(id=f"expanded-calc-{level}-{variant+1}", level=level,
                label=LABELS[index], column=2 if level == "parameters" else variant + 1,
                count=8 + index * 3, sheet=source, target=f"Extract{index+41}",
                start="Scratch" if level == "context" else source))
            font.append(dict(id=f"expanded-font-{level}-{variant+1}", level=level,
                slide=14 if level == "parameters" else 2 + index,
                sizes=[32 + index * 2, 15 + index],
                boxes=[6, 7] if level == "parameters" else [variant, variant + 1],
                use_original=False))
    return calc, font


def build(output: Path, base_url: str):
    calc, font = specifications()
    cases = make_cases(output, None, base_url, calc_variants=calc, font_variants=font)
    # Interleave families and strata so each worker receives all three strata.
    cases = [case for pair in zip(cases[:12], cases[12:]) for case in pair]
    assert len(cases) == len({case['id'] for case in cases}) == 24
    assert all(c['input_sha256'] != c['gold_sha256'] for c in cases)
    assert all(count == 4 for count in Counter((c['family'], c['level']) for c in cases).values())
    write_json(output / 'cases-private.json', cases)
    write_json(output / 'split.json', dict(
        setting='24 newly generated same-template transfer tasks; fixed before any model run',
        training='unchanged three historical seed episodes from dual pilot',
        test=[dict(id=c['id'], family=c['family'], level=c['level'],
                   instruction=c['config']['instruction'], input_sha256=c['input_sha256'],
                   gold_sha256=c['gold_sha256']) for c in cases],
        prior_diagnostic_cases_included=False, official_benchmark=False,
        prompt_access='current task and live observation plus selected frozen historical routes only',
        limitation='Two task templates; tests instance transfer, not unseen task families.'))
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base-url', default='http://20.20.20.1:18767')
    args = parser.parse_args()
    print(f'Prepared {len(build(args.output, args.base_url))} fresh cases')

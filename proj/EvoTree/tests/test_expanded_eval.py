"""Task isolation, parameter changes, scoring targets, and arm order balance."""
from collections import Counter
import json

import pytest

from experiments.m2_small_eval.expanded_eval import ARMS, schedule
from experiments.m2_small_eval.expanded_tasks import specifications, build


def test_worker_partition_and_order_balance_within_each_family_and_stratum():
    calc, font = specifications()
    cases = [dict(value, family=family) for pair in zip(calc, font)
             for value, family in zip(pair, ('calc', 'font'))]
    partitions = [schedule(cases, worker) for worker in range(3)]
    assert all(len(partition) == 8 for partition in partitions)
    scheduled = [entry for partition in partitions for entry in partition]
    assert sorted(index for index, _, _ in scheduled) == list(range(24))
    for family in ('calc', 'font'):
        for level in ('parameters', 'layout', 'context'):
            subset = [order for _, case, order in scheduled if case['family'] == family and case['level'] == level]
            assert len(subset) == 4
            for position in range(4):
                assert Counter(order[position] for order in subset) == Counter(ARMS)
            pairs = Counter(pair for order in subset for pair in zip(order, order[1:]))
            assert len(pairs) == 12 and set(pairs.values()) == {1}
    for worker in (-1, 3):
        with pytest.raises(ValueError):
            schedule(cases, worker)


def test_generated_targets_preserve_source_and_only_change_requested_text_sizes(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    pptx = pytest.importorskip('pptx')
    cases = build(tmp_path / 'cases', 'http://inputs.invalid:18767')
    assert len(cases) == 24
    assert len({c['config']['instruction'] for c in cases}) == 24
    old_instructions = {'Copy the "Tax Amount" column along with the header to a new sheet named "TaxReview".',
                        'Copy the "Units Sold" column along with the header to a new sheet named "UnitsReport".',
                        'Make the first textbox font size 36 pt while the second 18 pt on slide 5.'}
    for case in cases:
        assert case['config']['instruction'] not in old_instructions
        assert case['input_sha256'] != case['gold_sha256']
        values = case['source_variant']
        if case['family'] == 'calc':
            original, gold = (openpyxl.load_workbook(case[key]) for key in ('input', 'gold'))
            assert values['target'] not in original.sheetnames
            for name in original.sheetnames:
                assert list(original[name].values) == list(gold[name].values)
            source = original[values['sheet']]
            assert [row[0] for row in gold[values['target']].values] == [source.cell(i, values['column']).value
                                                                                    for i in range(1, values['count']+1)]
            assert original.active.title == values['start']
            assert len(case['config']['evaluator']['options']['rules']) == len(gold.sheetnames)
        else:
            original, gold = (pptx.Presentation(case[key]) for key in ('input', 'gold'))
            assert len(original.slides) == len(gold.slides) == 16
            changed = []
            for slide_index, (before_slide, after_slide) in enumerate(zip(original.slides, gold.slides), 1):
                for box, (before_shape, after_shape) in enumerate(zip(before_slide.shapes, after_slide.shapes)):
                    assert before_shape.text == after_shape.text
                    for bp, ap in zip(before_shape.text_frame.paragraphs, after_shape.text_frame.paragraphs):
                        for br, ar in zip(bp.runs, ap.runs):
                            if br.font.size != ar.font.size:
                                changed.append((slide_index, box))
            wanted = [(values['slide'], box) for box in values['boxes']]
            assert set(changed) <= set(wanted) and changed
            for index, box in enumerate(values['boxes']):
                shape = gold.slides[values['slide']-1].shapes[box]
                assert all(run.font.size.pt == values['sizes'][index]
                           for paragraph in shape.text_frame.paragraphs for run in paragraph.runs)
    split = json.loads((tmp_path / 'cases/split.json').read_text())
    assert not split['prior_diagnostic_cases_included']
    assert 'gold' not in split['prompt_access']

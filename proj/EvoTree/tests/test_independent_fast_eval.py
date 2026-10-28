import pytest
from experiments.m2_small_eval.independent_fast_eval import aggregate


def test_missing_official_score_is_not_a_zero_or_a_paired_loss():
    rows=[dict(task_id='a',condition='ordinary',score=0.),dict(task_id='a',condition='evotree',score=1.),
          dict(task_id='b',condition='ordinary',score=None),dict(task_id='b',condition='evotree',score=0.)]
    result=aggregate(rows,['a','b'])
    assert result['paired']==dict(complete=1,wins=1,losses=0,ties=0)
    assert result['arms']['ordinary']['missing']==1
    assert result['pairs'][1]['scores']['ordinary'] is None


def test_attempts_cannot_be_replaced_by_another_trial():
    row=dict(task_id='a',condition='ordinary',score=0.)
    with pytest.raises(ValueError,match='duplicate'):aggregate([row,row],['a'])

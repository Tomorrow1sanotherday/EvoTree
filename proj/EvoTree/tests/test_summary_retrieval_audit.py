from experiments.m2_small_eval.summary_retrieval_audit import phase_label


def test_scoring_uses_measured_font_values_not_previous_requested_action():
    obs={'app_info':{'slides':[{'slide_index':7,'text_objects':[
        {'box_index':1,'font_sizes_pt':[42.]}, {'box_index':2,'font_sizes_pt':[18.]}]}]},
        'executed_prefix':[{'name':'write_text','arguments':{'size':20}}]}
    private={'family':'impress','slide':7,'boxes':[1,2],'sizes':[42,20]}
    assert phase_label(obs,private)=='impress-font-progress:1,0'
    obs['app_info']['slides'][0]['text_objects'][1]['font_sizes_pt']=[20.]
    assert phase_label(obs,private)=='impress-font-progress:1,1'


def test_absent_actual_font_evidence_is_unknown_not_assumed_correct():
    obs={'app_info':{'slides':[{'slide_index':7,'text_objects':[
        {'box_index':1,'font_sizes_pt':[]},{'box_index':2,'font_sizes_pt':[20.]}]}]}}
    assert phase_label(obs,{'family':'impress','slide':7,'boxes':[1,2],'sizes':[42,20]})=='unknown'


def test_scratch_is_an_explicit_negative_in_scoring_only():
    obs={'app_info':{'active_sheet':'Scratch','name':'Scratch'}}
    assert phase_label(obs,{'family':'calc','source':'Records30','target':'Extract50'})=='calc-other'

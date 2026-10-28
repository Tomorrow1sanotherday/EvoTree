import json
from pathlib import Path

from evotree.distill_osworld import OSWorldDistiller
from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from experiments.m2_small_eval.diagnostic_pilot import hashes
from experiments.m2_small_eval.transfer_eval import build_bank, load_bank, plain_text, transfer_tree_text
from experiments.m2_small_eval.transfer_tasks import prepare_cache


def fixture_seed(root, index, instruction, tool, args):
    directory = root / f"task-{index}"
    directory.mkdir(parents=True)
    (directory / "seed-verified.json").write_text('{"score":1.0}')
    start = {"app":"osworld-task-start", "window":"", "dialog":None, "state_context":"start"}
    raw = {"action":{"action_type":tool,"parameters":args}}
    value = {"instruction":instruction,"success":True,"steps":[{
        "step_index":0,"observation":start,"raw_action":raw,"next_observation":None}],"metadata":{"score":1.0}}
    (directory / "seed-episode.json").write_text(json.dumps(value))


def test_shared_bank_retrieves_training_experience_without_substituting_new_answers(tmp_path):
    fixture_seed(tmp_path/"source",1,'Copy the "Revenue" column along with the header to a new sheet named "Sheet2".',
                 "libreoffice_calc.env_info",{"sheet_name":"Sheet1"})
    fixture_seed(tmp_path/"source",2,"Make the first textbox font size 60 pt while the second 28 pt on slide 14.",
                 "libreoffice_impress.env_info",{"page_indices":[14]})
    bank=tmp_path/"bank"
    before=build_bank(tmp_path/"source",bank)
    memory=load_bank(bank)
    episodes=json.loads((bank/"historical-episodes.json").read_text())
    query='Copy the "Tax Amount" column along with the header to a new sheet named "TaxReview".'
    template=memory.distiller.distill_task(query).template_id
    assert template in episodes
    text=plain_text(episodes[template],3000)
    assert "Sheet1" in text
    assert "TaxReview" not in text and "Tax Amount" not in text
    sheet=memory.read_cheatsheet(query,{"app":"osworld-task-start","window":"","state_context":"start"},[])
    memory._diagnostic_renderer=lambda _: "not used in ordered style"
    advice=transfer_tree_text(sheet,memory,query,3000)
    assert "SIMILAR" in advice and "same task" not in advice
    assert "Sheet1" in advice and "TaxReview" not in advice
    assert hashes(bank)==before
    assert json.loads((bank/"source-provenance.json").read_text())["test_episodes"]==0


def test_each_task_family_reuses_template_but_different_families_do_not():
    distiller=OSWorldDistiller()
    old=distiller.distill_task('Copy the "Revenue" column along with the header to a new sheet named "Sheet2".')
    new=distiller.distill_task('Copy the "Units Sold" column along with the header to a new sheet named "UnitsReport".')
    assert old.template_id==new.template_id
    font=distiller.distill_task("Make the first textbox font size 36 pt while the second 18 pt on slide 5.")
    assert font.template_id!=old.template_id


def test_case_cache_contains_immutable_input_and_gold_but_no_scored_result(tmp_path):
    import uuid
    data=tmp_path/"input.xlsx";data.write_bytes(b"new input")
    gold=tmp_path/"gold.xlsx";gold.write_bytes(b"answer")
    url="http://local.invalid/heldout/input.xlsx"
    case={"id":"heldout","input":str(data),"gold":str(gold),"config":{
        "config":[{"parameters":{"files":[{"url":url,"path":"/home/user/input.xlsx"}]}}],
        "evaluator":{"expected":{"dest":"gold-input.xlsx"}}}}
    prepare_cache(case,tmp_path/"cache")
    folder=tmp_path/"cache/heldout"
    assert not (folder/"input.xlsx").exists()
    assert (folder/f"{uuid.uuid5(uuid.NAMESPACE_URL,url)}_input.xlsx").read_bytes()==b"new input"
    assert (folder/"gold-input.xlsx").read_bytes()==b"answer"

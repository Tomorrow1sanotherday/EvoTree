"""α 语义取值：边上必须说得出"到底干了什么"（2026-09-10）。

背景：原实现对 GUI 一律 ``arg_slots=()``、对 TOOL 只留参数名，边上完全
不含内容。实测 pass3 一棵树里五条 ``computer_use.click`` 成功率各异却无法
区分，而 ``find ... -exec cp`` 这条整个任务的正确答案被丢弃 —— 渲染进
prompt 就是"推荐：computer_use.click —— 2次·成功率1.00"，模型无从照做。

这里钉死三件事：
    1. text/keys 这类跨轮可复用的值必须入库；coordinate 这类必须丢；
    2. args **不进主键** —— 不同取值仍归同一条边，统计不碎；
    3. 渲染必须把取值说出来。
"""

from evotree.distill_osworld import OSWorldDistiller, extract_reusable_args
from evotree.forest import ExperienceTree
from evotree.notes import GlobalNotes
from evotree.retrieval import Retriever, render_cheatsheet
from evotree.types import ActionKind, ActionTmpl, StateSig


def _pred(payload: str) -> dict:
    return {"prediction": f"<tool_call>{payload}</tool_call>"}


def test_command_text_is_kept():
    """命令原文是这一步真正的内容，必须留下。"""
    action = OSWorldDistiller().distill_action(
        _pred('{"name":"computer_use","arguments":'
              '{"action":"type","text":"cp file1 dir1 dir2 dir3","enter":1}}')
    )
    assert action.args["text"] == "cp file1 dir1 dir2 dir3"


def test_hotkey_is_normalized():
    """快捷键压成 CTRL+ALT+T —— 与 EE-MCP 的 'Ctrl+P' 同型，可直接照做。"""
    action = OSWorldDistiller().distill_action(
        _pred('{"name":"computer_use","arguments":'
              '{"action":"key","keys":["CTRL","ALT","T"]}}')
    )
    assert action.args["keys"] == "CTRL+ALT+T"


def test_coordinates_are_dropped():
    """坐标跨轮漂移，复用即错 —— 原设计丢坐标是对的，这条不能回退。"""
    action = OSWorldDistiller().distill_action(
        _pred('{"name":"computer_use","arguments":'
              '{"action":"click","coordinate":[529,112]}}')
    )
    assert action.args == {}


def test_args_do_not_affect_edge_key():
    """不同取值必须仍归同一条边，否则统计碎成 n=1（window 标题带查询串的教训）。"""
    a = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"})
    b = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp c d"})
    assert a.key() == b.key()

    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    tree.upsert_step(sig, a, success=True)
    tree.upsert_step(sig, b, success=True)
    edges = tree.out_edges("s0")
    assert len(edges) == 1, "不同取值不该分裂成两条边"
    assert edges[0].n == 2
    assert len(edges[0].arg_samples) == 2, "两种写法都该留下（抽规则要对照）"
    assert [x["args"]["text"] for x in edges[0].arg_samples] == ["cp a b", "cp c d"]


def test_duplicate_samples_are_deduped():
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    action = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"})
    for _ in range(3):
        tree.upsert_step(sig, action, success=True)
    assert len(tree.out_edges("s0")[0].arg_samples) == 1


def test_samples_survive_roundtrip():
    """落盘再读回不能丢 —— 树是跨轮累积的，丢了等于没记。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    tree.upsert_step(
        sig,
        ActionTmpl(ActionKind.GUI, "computer_use.key", args={"keys": "CTRL+ALT+T"}),
        success=True,
    )
    restored = ExperienceTree.from_dict(tree.to_dict())
    assert restored.out_edges("s0")[0].arg_samples == [
        {"args": {"keys": "CTRL+ALT+T"}, "n": 1, "n_success": 1}
    ]


def test_render_shows_the_command():
    """终点检验：模型看到的文本里必须有那条命令，否则前面全白做。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    action = ActionTmpl(
        ActionKind.GUI, "computer_use.type", args={"text": "cp file1 dir1 dir2 dir3"}
    )
    edge = tree.upsert_step(sig, action, success=True)
    tree.record_steps_after_success([edge])

    sheet = Retriever().retrieve(tree, sig, GlobalNotes(), candidate_tools=[])
    text = render_cheatsheet(sheet)
    assert "cp file1 dir1 dir2 dir3" in text


# --------------------------------------------------------------------------- #
# 取值与成败的绑定（2026-09-11）
#
# 首版 arg_samples 不记成败：实测 Copies_file 那条 type 边上五种取值共享同一个
# succ=4，其中 ls dir1/file1 dir2/file1 是**验证命令不是解法**，却与
# cp file1 dir1 dir2 dir3 平权；渲染按插入序取前两条，会把验证命令推给模型。
# --------------------------------------------------------------------------- #


def test_sample_records_episode_outcome():
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    good = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"})
    bad = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "ls a b"})
    tree.upsert_step(sig, good, success=True)
    tree.upsert_step(sig, bad, success=False)

    by_text = {
        x["args"]["text"]: x for x in tree.out_edges("s0")[0].arg_samples
    }
    assert by_text["cp a b"]["n_success"] == 1
    assert by_text["ls a b"]["n_success"] == 0


def test_repeated_sample_accumulates_instead_of_duplicating():
    """同一条命令试了三次（两成一败）应累加计数，而不是变成三条。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    action = ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"})
    tree.upsert_step(sig, action, success=True)
    tree.upsert_step(sig, action, success=False)
    tree.upsert_step(sig, action, success=True)

    samples = tree.out_edges("s0")[0].arg_samples
    assert len(samples) == 1
    assert samples[0]["n"] == 3
    assert samples[0]["n_success"] == 2


def test_successful_values_rank_first():
    """排序必须按成功率，不能按插入序 —— 否则先出现的验证命令会挤掉解法。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    # 故意让"验证命令"先进来，模拟实测里的插入顺序
    tree.upsert_step(
        sig,
        ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "ls a b"}),
        success=False,
    )
    tree.upsert_step(
        sig,
        ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"}),
        success=True,
    )
    ranked = tree.out_edges("s0")[0].ranked_arg_samples()
    assert ranked[0]["args"]["text"] == "cp a b"


def test_render_prefers_successful_value():
    """终点检验：模型先看到的必须是通过的那条。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    tree.upsert_step(
        sig,
        ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "ls a b"}),
        success=False,
    )
    edge = tree.upsert_step(
        sig,
        ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "cp a b"}),
        success=True,
    )
    tree.record_steps_after_success([edge])

    text = render_cheatsheet(
        Retriever().retrieve(tree, sig, GlobalNotes(), candidate_tools=[])
    )
    assert text.index("cp a b") < text.index("ls a b")


def test_legacy_samples_are_readable():
    """已落盘的旧种子（裸取值 dict、无成败）必须还能读，否则历史记忆失联。"""
    tree = ExperienceTree(template_id="t")
    sig = StateSig(app="x", window="w")
    tree.upsert_step(
        sig, ActionTmpl(ActionKind.GUI, "computer_use.type", args={"text": "x"}), True
    )
    raw = tree.to_dict()
    # 退回 2026-09-10 的旧格式
    raw["edges"][0]["arg_samples"] = [{"text": "cp file1 dir1"}]

    restored = ExperienceTree.from_dict(raw)
    sample = restored.out_edges("s0")[0].arg_samples[0]
    assert sample["args"] == {"text": "cp file1 dir1"}
    assert sample["n_success"] == 0, "旧样本成败未知，不能凭空声称成功"

"""forest 模块的单元测试。

重点验证 idea §4.2 的三条"自动成立"性质、cost-to-go 倒数计算、JSON round-trip。
用构造的假状态/动作，不依赖服务器。
"""

from evotree.forest import ExperienceForest, ExperienceTree
from evotree.types import ActionKind, ActionTmpl, StateSig


def sig(app="impress", window="a.pptx", dialog=None, **facts) -> StateSig:
    """构造测试用状态签名。"""
    return StateSig(app=app, window=window, dialog=dialog, facts=facts)


def tool(name, *slots) -> ActionTmpl:
    return ActionTmpl(ActionKind.TOOL, name, tuple(slots))


# ------------------- 性质 1：同一动作自动归堆 ------------------------------ #
class TestAutoAggregation:
    def test_same_action_lands_on_same_edge(self):
        tree = ExperienceTree("t1")
        s = sig()
        a = tool("duplicate_slide", "idx", "pos")
        tree.upsert_step(s, a, success=True)
        tree.upsert_step(s, a, success=False)
        tree.upsert_step(s, a, success=True)
        # 三次执行落进同一条边
        edges = tree.out_edges("s0")
        assert len(edges) == 1
        assert edges[0].n == 3
        assert edges[0].n_success == 2


# ------------------- 性质 2：不同来路到达同一局面自动汇合 ------------------ #
class TestAutoMerge:
    def test_same_state_key_merges(self):
        tree = ExperienceTree("t1")
        # 两个观测 facts 不同，但主键(app,window,dialog)相同 → 同一节点
        s_a = sig(app="impress", window="a.pptx", 知道页数=True)
        s_b = sig(app="impress", window="a.pptx", 知道页数=False)
        n1 = tree.get_or_create_node(s_a)
        n2 = tree.get_or_create_node(s_b)
        assert n1.node_id == n2.node_id  # 汇合到同一节点

    def test_different_key_separate(self):
        tree = ExperienceTree("t1")
        n1 = tree.get_or_create_node(sig(dialog=None))
        n2 = tree.get_or_create_node(sig(dialog="格式确认框"))
        assert n1.node_id != n2.node_id  # dialog 不同 → 不同节点


# ------------------- 性质 3：失败正常入账 --------------------------------- #
class TestFailureRecorded:
    def test_failure_creates_edge_and_fail_landing(self):
        tree = ExperienceTree("t1")
        s = sig()
        s_err = sig(dialog="报错弹窗")
        tree.upsert_step(s, tool("duplicate_slide"), success=False, to_sig=s_err)
        edge = tree.out_edges("s0")[0]
        assert edge.n == 1 and edge.n_success == 0
        # 失败落点被记为一个正常状态节点
        assert edge.to_fail is not None
        assert tree.node_by_id(edge.to_fail).sig.dialog == "报错弹窗"


# ------------------- cost-to-go 倒数计算 --------------------------------- #
class TestStepsAfterSuccess:
    def test_reverse_count(self):
        tree = ExperienceTree("t1")
        # 一条 3 步成功轨迹
        s0, s1, s2 = sig(window="0"), sig(window="1"), sig(window="2")
        e0 = tree.upsert_step(s0, tool("a"), success=True, to_sig=s1)
        e1 = tree.upsert_step(s1, tool("b"), success=True, to_sig=s2)
        e2 = tree.upsert_step(s2, tool("c"), success=True)
        tree.record_steps_after_success([e0, e1, e2])
        # 末步记1、倒二步记2、首步记3
        assert e0.steps_after_success == [3]
        assert e1.steps_after_success == [2]
        assert e2.steps_after_success == [1]

    def test_accumulates_across_episodes(self):
        tree = ExperienceTree("t1")
        s0 = sig()
        # 两轮都走同一条边，剩余步数累积
        e = tree.upsert_step(s0, tool("a"), success=True)
        tree.record_steps_after_success([e])
        e = tree.upsert_step(s0, tool("a"), success=True)
        tree.record_steps_after_success([e])
        assert e.steps_after_success == [1, 1]
        assert e.n == 2


# ------------------- to_success 指针（前瞻的基础）------------------------ #
class TestSuccessPointer:
    def test_pointer_enables_lookahead_jump(self):
        tree = ExperienceTree("t1")
        s0, s1 = sig(window="0"), sig(window="1")
        e = tree.upsert_step(s0, tool("copy"), success=True, to_sig=s1)
        # 从边的 to_success 指针能跳到后继状态，再看它的出边 —— 这就是前瞻
        succ_node = tree.node_by_id(e.to_success)
        assert succ_node.sig.window == "1"


# ------------------- 维护：prune 不砍可靠长路 ---------------------------- #
class TestPrune:
    def test_prune_removes_low_low_but_keeps_reliable(self):
        tree = ExperienceTree("t1")
        s = sig()
        # 垃圾边：走1次、失败（n<2 且成功率≤0.2）
        tree.upsert_step(s, tool("junk"), success=False)
        # 可靠边：走1次成功（成功率高，不该被删）—— 用不同起点避免归堆
        s2 = sig(window="other")
        tree.upsert_step(s2, tool("good"), success=True)
        removed = tree.prune(min_visits=2, max_success_rate=0.2)
        assert removed == 1
        assert len(tree.out_edges("s0")) == 0   # 垃圾边被删
        # 可靠边保留
        assert len(tree.out_edges(tree.get_or_create_node(s2).node_id)) == 1


# ------------------- JSON round-trip ------------------------------------- #
class TestPersistence:
    def test_roundtrip_preserves_stats(self):
        tree = ExperienceTree("复制列到新sheet")
        s0, s1 = sig(window="0"), sig(window="1", 已复制=True)
        e = tree.upsert_step(s0, tool("copy_cells", "src", "dst"),
                             success=True, to_sig=s1, note="pos越界")
        tree.record_steps_after_success([e])

        restored = ExperienceTree.from_dict(tree.to_dict())
        re = restored.out_edges("s0")[0]
        assert re.n == 1 and re.n_success == 1
        assert re.steps_after_success == [1]
        assert re.notes == ["pos越界"]
        assert re.action.name == "copy_cells"
        assert re.action.arg_slots == ("src", "dst")

    def test_save_load_file(self, tmp_path):
        tree = ExperienceTree("t1")
        tree.upsert_step(sig(), tool("a"), success=True)
        p = tmp_path / "tree.json"
        tree.save(p)
        loaded = ExperienceTree.load(p)
        assert loaded.out_edges("s0")[0].n == 1


# ------------------- 森林：按任务分册 + 冷启动空返回 --------------------- #
class TestForest:
    def test_get_tree_none_when_unseen(self):
        forest = ExperienceForest()
        # 没见过的模板 → None（冷启动，推荐栏自然空返回，不是特殊分支）
        assert forest.get_tree("从没干过的任务") is None

    def test_separate_trees_per_template(self):
        forest = ExperienceForest()
        t1 = forest.get_or_create_tree("复制任务")
        t2 = forest.get_or_create_tree("导出任务")
        assert t1 is not t2  # 不同任务分册

    def test_forest_persistence(self, tmp_path):
        forest = ExperienceForest(root_dir=tmp_path)
        tree = forest.get_or_create_tree("t1")
        tree.upsert_step(sig(), tool("a"), success=True)
        forest.save_all()
        # 新森林从磁盘载入
        forest2 = ExperienceForest(root_dir=tmp_path)
        assert forest2.get_tree("t1").out_edges("s0")[0].n == 1

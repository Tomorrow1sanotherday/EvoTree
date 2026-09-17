"""OSWorld-MCP wrapper 的离线 smoke 测试。

用最小 fake agent/env 重放真实 runner 的调用顺序，确保 hook 的边界与
``lib_run_single_e2e`` 一致，而无需启动 VM 或调用模型服务。
"""

from __future__ import annotations

from pathlib import Path

from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import (
    available_tool_names,
    build_observation,
    candidate_tool_names,
    reorder_tool_list,
    wrap_osworld,
)
from evotree.distill_osworld import GUI_TOOL_NAME, OSWorldDistiller


def _prediction(name: str = "os.rename_directory") -> str:
    return (
        '<tool_call>{"name":"%s","arguments":{"old_path":"a",'
        '"new_path":"b"}}</tool_call>' % name
    )


def _schema(name: str) -> dict:
    return {"type": "function", "function": {"name": name}}


def _obs(window: str = "Files") -> dict:
    return {
        "cur_app": "files",
        "cur_window_id": "active",
        "apps": {"active": {"app_name": "files", "title": window}},
        "tool_list": [_schema("os.other"), _schema("os.rename_directory")],
        "screenshot": b"png",
        "accessibility_tree": "<application />",
    }


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def reset(self, *args, **kwargs):
        return None

    def predict(self, instruction, obs):
        self.calls.append((instruction, obs))
        return _prediction(), [
            {"action_type": "os.rename_directory", "parameters": {"old_path": "a"}}
        ]


class FakeEnv:
    def __init__(self) -> None:
        self.observation = _obs()
        self.steps: list[object] = []

    def reset(self, **kwargs):
        self.observation = _obs()
        return self.observation

    def _get_obs(self):
        return self.observation

    def step(self, action, pause=0):
        self.steps.append(action)
        self.observation = _obs("Files *")
        return self.observation, 0, False, {}

    def evaluate(self):
        return 1.0


def _obs_os_domain() -> dict:
    """os 域任务的真实观测形状：``tool_list`` 是**空的**。

    ``_get_obs`` 只为 libreoffice 三兄弟映射 MCP 工具表，os 域任务
    ``tool_name=None`` → 空表；而 ``computer_use`` 由 agent 无条件附加
    （``owl_agent.py``），不出现在这里。窗口信息也常常拿不到（TCG 下窗口注册慢），
    于是 φ 退化成 app=unknown —— 与种子树里的节点签名一致。
    """
    return {
        "cur_app": None,
        "cur_window_id": "",
        "apps": {},
        "tool_list": [],
        "screenshot": b"png",
        "accessibility_tree": None,
    }


class FakeEnvOsDomain(FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.observation = _obs_os_domain()

    def reset(self, **kwargs):
        self.observation = _obs_os_domain()
        return self.observation

    def step(self, action, pause=0):
        self.steps.append(action)
        return self.observation, 0, False, {}


def _seed_memory(root_dir: Path | None = None) -> EvoTreeMemory:
    memory = EvoTreeMemory(root_dir=root_dir, distiller=OSWorldDistiller())
    memory.write_episode(
        Episode(
            instruction="rename directory",
            steps=(
                EpisodeStep(
                    step_index=0,
                    observation=build_observation(_obs()),
                    raw_action={"prediction": _prediction()},
                    note="confirm the target before renaming",
                ),
            ),
            success=True,
        )
    )
    return memory


def test_observation_and_tool_schema_mapping():
    mapped = build_observation(_obs())
    assert mapped["app"] == "files"
    assert mapped["window"] == "Files"
    assert candidate_tool_names(_obs()["tool_list"]) == ["os.other", "os.rename_directory"]
    reordered = reorder_tool_list(
        _obs()["tool_list"], ["os.rename_directory", "os.other"]
    )
    assert candidate_tool_names(reordered) == ["os.rename_directory", "os.other"]


def test_real_loop_hook_injects_prompt_records_action_and_finishes(tmp_path: Path):
    memory = _seed_memory(tmp_path)
    raw_agent, raw_env, runtime = wrap_osworld(
        FakeAgent(), FakeEnv(), memory, use_evotree=True
    )

    raw_agent.reset()
    raw_env.reset(task_config={"id": "smoke"})
    observation = raw_env._get_obs()
    _response, actions = raw_agent.predict("rename directory", observation)
    assert "[经验]" in raw_agent.agent.calls[0][0]
    assert "[注意]" in raw_agent.agent.calls[0][0]
    assert candidate_tool_names(raw_agent.agent.calls[0][1]["tool_list"])[0] == "os.rename_directory"

    raw_env.step(actions[0], 0)
    assert runtime is not None and runtime.active
    score = raw_env.evaluate()
    assert score == 1.0
    assert runtime.active is False
    tree_files = list(tmp_path.glob("memory/*/tree.json"))
    assert tree_files
    assert tree_files[0].read_text(encoding="utf-8")


class TestOsDomainCautionsReachTheAgent:
    """os 域任务（tool_list 为空）也必须能拿到注意栏。

    实验四（2026-09-18）第一小时的教训：修好 notes 的命名空间查询之后，
    [注意] 栏**仍然**零注入。第二道闸门在这里 —— 注意栏拿到的候选工具名来自
    ``obs["tool_list"]``，而：

    - ``computer_use`` 由 agent 无条件附加（``owl_agent.py``），不在那张表里；
    - os 域任务的 ``tool_list`` 恰好是空的（``_get_obs`` 只映射 libreoffice）。

    空列表进 ``_build_cautions``，循环体一次都不执行，坑位笔记再对也送不出去。

    此前所有 wrapper 测试的 ``_obs()`` 夹具都带着非空 ``tool_list``，所以整组
    测试对这个 bug 视而不见 —— 这正是它能活到线上的原因。
    """

    def test_available_tool_names_always_includes_gui_tool(self):
        assert available_tool_names([]) == [GUI_TOOL_NAME]
        assert available_tool_names(None) == [GUI_TOOL_NAME]

    def test_mcp_tools_keep_their_order_and_gui_tool_goes_last(self):
        # 重排语义是"有战绩的 MCP 工具排前面"，补名不该打乱它们的相对顺序
        assert available_tool_names(_obs()["tool_list"]) == [
            "os.other",
            "os.rename_directory",
            GUI_TOOL_NAME,
        ]

    def test_no_duplicate_when_gui_tool_already_listed(self):
        names = available_tool_names([_schema(GUI_TOOL_NAME)])
        assert names == [GUI_TOOL_NAME]

    def test_candidate_tool_names_unchanged_for_reranking(self):
        # 对照：重排用的函数不能被污染，否则会把不存在的项排进 tool_list
        assert candidate_tool_names([]) == []
        assert GUI_TOOL_NAME not in candidate_tool_names(_obs()["tool_list"])

    def test_gui_pit_note_reaches_prompt_with_empty_tool_list(self):
        """终点检验：os 域任务、tool_list 空，注意栏仍要进 prompt。

        种子笔记按子动作名挂（``computer_use.key``），与实验三的种子写法一致。
        """
        memory = EvoTreeMemory(distiller=OSWorldDistiller())
        memory.notes.add_note("computer_use.key", "Ctrl+Alt+T 常打不开终端")
        agent, env, _runtime = wrap_osworld(
            FakeAgent(), FakeEnvOsDomain(), memory, use_evotree=True
        )
        agent.reset()
        env.reset(task_config={"id": "os-smoke"})
        agent.predict("count php lines", env._get_obs())

        prompt = agent.agent.calls[0][0]
        assert "[注意]" in prompt
        assert "打不开终端" in prompt

    def test_reranked_tool_list_stays_clean(self):
        """补进去的 GUI 工具名不能污染回传给 agent 的 tool_list。

        它不是 schema，只是个名字；混进去 agent 侧会拿到一个不存在的工具项。
        """
        memory = EvoTreeMemory(distiller=OSWorldDistiller())
        memory.notes.add_note("computer_use.key", "Ctrl+Alt+T 常打不开终端")
        agent, env, _runtime = wrap_osworld(
            FakeAgent(), FakeEnvOsDomain(), memory, use_evotree=True
        )
        agent.reset()
        env.reset(task_config={"id": "os-smoke"})
        agent.predict("count php lines", env._get_obs())

        assert agent.agent.calls[0][1]["tool_list"] == []


def test_first_action_is_recorded_without_explicit_get_obs():
    """runner 走 `obs = env.reset()` 拿首帧时，第一个动作也必须入账。

    回归测试：此前 reset() 把 _last_observation 清成 None 且不回填，只有显式
    调 _get_obs() 才有 before 观测；走 reset 返回值的 runner 会让第一步被
    静默丢弃——整棵树为空却不报错。
    """
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    raw_agent, raw_env, runtime = wrap_osworld(
        FakeAgent(), FakeEnv(), memory, use_evotree=True
    )

    observation = raw_env.reset(task_config={"id": "smoke"})
    _response, actions = raw_agent.predict("rename directory", observation)
    raw_env.step(actions[0], 0)
    raw_env.evaluate()

    tree = memory.forest.get_tree(memory.distiller.distill_task("rename directory").template_id)
    assert tree is not None, "first action was silently dropped"
    assert [edge.n for edge in tree.all_edges()] == [1]


def test_execution_error_goes_to_cautions():
    """exe_result 里的报错进注意栏；成功率由整轮结果决定。

    ``FakeEnv.evaluate()`` 返回 1.0 → 整轮成功 → 边记成功，尽管这一步
    工具报了错。报错文本仍抄进 global_notes（idea §0 写入第 3 步）。
    """

    class FailingEnv(FakeEnv):
        def step(self, action, pause=0):
            self.steps.append(action)
            self.observation = dict(_obs("Files"))
            self.observation["exe_result"] = "Traceback: OSError target exists"
            return self.observation, 0, False, {}

    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    raw_agent, raw_env, _runtime = wrap_osworld(
        FakeAgent(), FailingEnv(), memory, use_evotree=True
    )
    observation = raw_env.reset()
    _response, actions = raw_agent.predict("rename directory", observation)
    raw_env.step(actions[0], 0)
    raw_env.evaluate()

    tree = memory.forest.get_tree(memory.distiller.distill_task("rename directory").template_id)
    edge = tree.out_edges("s0")[0]
    assert edge.n == 1 and edge.n_success == 1
    assert edge.to_success is not None
    # 工具报错抄进全局手册（idea §0 写入第 3 步）。
    assert memory.notes.query("os.rename_directory")


def test_baseline_switch_returns_original_objects():
    agent, env = FakeAgent(), FakeEnv()
    wrapped_agent, wrapped_env, runtime = wrap_osworld(
        agent, env, EvoTreeMemory(), use_evotree=False
    )
    assert wrapped_agent is agent
    assert wrapped_env is env
    assert runtime is None


def test_terminal_action_is_not_recorded():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())

    class TerminalAgent(FakeAgent):
        def predict(self, instruction, obs):
            self.calls.append((instruction, obs))
            return _prediction(), ["DONE"]

    raw_agent, raw_env, runtime = wrap_osworld(
        TerminalAgent(), FakeEnv(), memory, use_evotree=True
    )
    raw_env.reset()
    _obs0 = raw_env._get_obs()
    _response, actions = raw_agent.predict("rename directory", _obs0)
    raw_env.step(actions[0], 0)
    raw_env.evaluate()
    assert runtime is not None and runtime.active is False
    assert memory.forest.get_tree(memory.distiller.distill_task("rename directory").template_id) is None


def test_frozen_wrapper_does_not_create_files(tmp_path: Path):
    memory = EvoTreeMemory(
        root_dir=tmp_path, distiller=OSWorldDistiller(), write_enabled=False
    )
    raw_agent, raw_env, _runtime = wrap_osworld(
        FakeAgent(), FakeEnv(), memory, use_evotree=True
    )
    raw_env.reset()
    observation = raw_env._get_obs()
    _response, actions = raw_agent.predict("rename directory", observation)
    raw_env.step(actions[0], 0)
    raw_env.evaluate()
    assert list(tmp_path.rglob("*")) == []


def test_execution_error_becomes_tool_caution():
    class ErrorEnv(FakeEnv):
        def step(self, action, pause=0):
            self.steps.append(action)
            self.observation = _obs("Files *")
            self.observation["exe_result"] = "Traceback: tool failed"
            return self.observation, 0, False, {}

    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    raw_agent, raw_env, runtime = wrap_osworld(
        FakeAgent(), ErrorEnv(), memory, use_evotree=True
    )
    raw_env.reset()
    _response, actions = raw_agent.predict("rename directory", raw_env._get_obs())
    raw_env.step(actions[0], 0)
    raw_env.evaluate()
    assert runtime is not None
    sheet = memory.read_cheatsheet(
        "rename directory", build_observation(_obs()), ["os.rename_directory"]
    )
    assert any("tool failed" in caution for caution in sheet.cautions)


def test_gui_and_mcp_success_paths_rank_mcp_by_cost_to_go():
    memory = EvoTreeMemory(distiller=OSWorldDistiller())
    # Give each GUI step a distinct sub-action so the path is three edges.
    gui_predictions = ("click", "key", "type")
    gui_steps = tuple(
        EpisodeStep(
            step_index=index,
            observation=build_observation(_obs(window)),
            raw_action={
                "prediction": '<tool_call>{"name":"computer_use","arguments":{"action":"%s"}}</tool_call>'
                % gui_predictions[index]
            },
        )
        for index, window in enumerate(("Files", "Files selected", "Files rename"))
    )
    memory.write_episode(
        Episode("rename directory", gui_steps, success=True, source="gui")
    )
    memory.write_episode(
        Episode(
            "rename directory",
            (
                EpisodeStep(
                    step_index=0,
                    observation=build_observation(_obs()),
                    raw_action={"prediction": _prediction()},
                ),
            ),
            success=True,
            source="mcp",
        )
    )

    sheet = memory.read_cheatsheet(
        "rename directory", build_observation(_obs()), ["os.rename_directory"]
    )
    assert sheet.recommendations[0].action.name == "os.rename_directory"
    assert sheet.recommendations[0].mean_steps_to_go == 1


def test_hostname_is_not_used_as_window_title():
    """回归锁：主机名不能当窗口标题。

    OSWorld 的 get_current_apps() 解析 ``wmctrl -xl`` 时取了 parts[3]（主机名）
    而非 parts[4]（标题）。2026-09-06 实测：28 个状态节点里 19 个是空签名，
    其余全是 "user-virtual-machine"，树完全扁平、前瞻退化成 s0→s0 自环。
    """
    fake = {
        "cur_app": "nautilus",
        "cur_window_id": "w1",
        "apps": {"w1": {"app_name": "nautilus", "title": "user-virtual-machine"}},
    }
    assert build_observation(fake)["window"] == ""

    real = {
        "cur_app": "impress",
        "cur_window_id": "w1",
        "apps": {"w1": {"app_name": "impress", "title": "deck.pptx - Impress"}},
    }
    assert build_observation(real)["window"] == "deck.pptx - Impress"

"""OSWorld-MCP wrapper 的离线 smoke 测试。

用最小 fake agent/env 重放真实 runner 的调用顺序，确保 hook 的边界与
``lib_run_single_e2e`` 一致，而无需启动 VM 或调用模型服务。
"""

from __future__ import annotations

from pathlib import Path

from evotree.episode import Episode, EpisodeStep
from evotree.memory import EvoTreeMemory
from evotree.osworld_integration import (
    build_observation,
    candidate_tool_names,
    reorder_tool_list,
    wrap_osworld,
)
from evotree.distill_osworld import OSWorldDistiller


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

    tree = memory.forest.get_tree("rename_directory")
    assert tree is not None, "first action was silently dropped"
    assert [edge.n for edge in tree.all_edges()] == [1]


def test_execution_error_marks_edge_failure():
    """exe_result 里的报错 → 这条边记失败，落点挂 to_fail。"""

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

    tree = memory.forest.get_tree("rename_directory")
    edge = tree.out_edges("s0")[0]
    assert edge.n == 1 and edge.n_success == 0
    assert edge.to_fail is not None
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
    assert memory.forest.get_tree("rename_directory") is None


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

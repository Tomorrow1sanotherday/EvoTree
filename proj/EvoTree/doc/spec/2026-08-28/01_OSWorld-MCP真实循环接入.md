# Spec 2026-08-28 · OSWorld-MCP 真实循环接入

## 接入边界

`evotree.osworld_integration` 提供两个 framework-neutral wrapper，不修改
OSWorld 的环境或 agent 实现：

- `EvoTreeAgentWrapper.predict()` 在每次模型调用前把 `_get_obs()` 的
  `cur_app/apps/cur_window_id/accessibility_tree/screenshot` 映射为
  `app/window/dialog/facts`，从 `tool_list` schema 提取候选工具，检索小抄并
  重排原 schema。原 agent 的 `predict` 返回值保持不变。
- `EvoTreeEnvWrapper.step()` 在真实环境执行后把模型 prediction 和环境 action
  作为 `raw_action` 暂存到 runtime；`evaluate()` 返回 evaluator 分数后调用
  `runtime.finish(score=...)`，一次性按整轮结果写入路径。

组合入口：

```python
from evotree import EvoTreeMemory
from evotree.distill_osworld import OSWorldDistiller
from evotree.osworld_integration import wrap_osworld

memory = EvoTreeMemory(
    root_dir="/path/to/evotree-memory",
    distiller=OSWorldDistiller(),
    write_enabled=True,
)
agent, env, runtime = wrap_osworld(agent, env, memory, use_evotree=True)
```

这三个对象可以直接传给现有 `lib_run_single_e2e.run_single_example`。设置
`use_evotree=False` 时返回原始 agent/env，不创建 runtime，也不复制观测，裸
baseline 的行为保持不变。

服务器上的 `OSWorld-mcp-run/run_multienv_e2e.py` 已加最小 CLI 接线，默认关闭：

```bash
python run_multienv_e2e.py --use_evotree \
  --evotree_memory_dir /data/songweishuai/evotree_memory
# Setting II 冻结读取：
python run_multienv_e2e.py --use_evotree --no-evotree-write-enabled
```

每个并行环境使用 `evotree_memory/env-N` 子目录，避免多个 worker 同时改写同一
个 JSON 文件；单环境运行时可直接得到跨任务的连续经验。

## 语义约束

- `reset`、`DONE`、`FAIL`、`terminate` 和 prediction 为 `client error` 的记录
  不进入 episode steps。
- OSWorld 中间 reward 保留在 `EpisodeStep.reward`/步骤元数据中，不参与成功
  判断；只有 `evaluate()` 的最终 score（`score > 0`）决定整轮成功。
- `write_enabled=False` 时 `write_episode` 和 `save` 都不落盘，但仍可读已有
  经验。

## 验收

`tests/test_osworld_integration.py` 用 fake agent/env 重放真实 runner 的调用
顺序，覆盖 observation/tool 映射、小抄中的 `[经验]`/`[注意]`、动作写入、
score finish、终止动作过滤和 baseline 开关。

服务器 smoke（2026-08-28）重放同一调用协议并产生真实动作字典，结果目录为
`/data/songweishuai/evotree_smoke_20260828`：prompt 同时含 `[经验]`、`[注意]`，
`traj.jsonl` 含 reset 与 MCP function-call 动作，`result.txt` 为 `1.0`，
tree 中 `n_success=2`（seed + 在线 episode），冻结目录为空。

# Spec 2026-08-26 · 真实轨迹适配与 Agent 接入 Hook

> 本文档说明本次新增的代码：统一 Episode 数据结构、OSWorld 轨迹适配器、
> 在线 agent 接入 hook。目标是把 EvoTree 从一个独立记忆库接到真实 agent
> 的决策循环，但暂不修改 OSWorld-MCP 的运行代码。

## 本次新增

| 文件 | 作用 |
|---|---|
| `evotree/episode.py` | 定义统一的 `Episode`、`EpisodeStep`、`RuntimeStep` |
| `evotree/osworld_adapter.py` | 读取 OSWorld `traj.jsonl`/`result.txt`，转成统一 Episode |
| `evotree/integration.py` | 提供在线 `EvoTreeRuntime` 和 prompt 注入函数 |
| `tests/test_osworld_adapter.py` | OSWorld 适配器测试 |
| `tests/test_integration.py` | 在线接入 hook 测试 |
| `tests/test_episode.py` | Episode 与路径级写入测试 |

**当前本地测试：121 个全部通过。**

## 1. 为什么需要 Episode 层

之前的 `EvoTreeMemory.write()` 使用自定义的 `Step.success`，容易把两种不同
语义混在一起：

- **单步动作是否执行成功**：工具有没有报错、动作是否被环境接受；
- **整轮任务是否成功**：任务最终 evaluator 是否给出成功分数。

OSWorld 的中间 `reward` 通常为 `0`，这并不表示当前动作失败。真正的 EvoTree
统计需要的是：从某个状态选择某个动作后，整轮任务最终成功的概率。因此现在用：

```text
Episode.success       = 客观 evaluator 给出的整轮结果
EpisodeStep.reward    = 原始记录中的 reward，仅作保留，不直接当单步成败
```

这能避免写入阶段系统性地把所有中间动作记成失败。

## 2. `episode.py`

- `EpisodeStep` 表示一个可记录动作：动作前 observation、可选的动作后 observation、
  原始动作、note、reward、done 和运行元数据。
- `Episode` 表示一轮完整任务：instruction、按顺序的 steps、最终 success、source 和 metadata。
- `RuntimeStep` 用于在线阶段暂存动作。在线阶段还不知道最终结果，因此不含 success。
- 构造 `Episode` 时检查 instruction 非空、step 按 step_index 有序。

`reset`、`DONE`、`terminate` 不是普通操作，不应放入 `Episode.steps`。

## 3. `osworld_adapter.py`

输入是一个 OSWorld 结果目录：

```text
result-directory/
  traj.jsonl
  result.txt
  step_*.png
```

处理规则：

1. 跳过 `action == "reset"`；
2. 跳过 `DONE` 和 `terminate`；
3. 跳过 `prediction == "client error"`，因为它不是环境中的真实动作；
4. 保留真实 `prediction` 和 `action`，交给 `OSWorldDistiller` 解析；
5. 读取 `result.txt`，`score > 0` 才标记 `Episode.success=True`；
6. 不使用中间 `reward=0` 推断动作失败；
7. 默认 observation 明确使用 placeholder（`app=unknown`），不伪造真实 φ；
8. 真实接入时通过 `observation_factory(record, index)` 提供真实 observation。

没有自动从截图推断真实 φ，是因为 `traj.jsonl` 只保存截图文件名，不保存结构化
accessibility tree、窗口标题或弹窗字段。适配器不能伪造这些信息。

## 4. `EvoTreeMemory.write_episode()`

这是 OSWorld 适配器和在线 runtime 的正式写入入口：

```text
Episode.success=True  → 该 episode 的每条动作边 n_success += 1
Episode.success=False → 该 episode 的每条动作边只 n += 1
```

成功 episode 才记录 `steps_after_success`；失败 episode 仍正常入账，但没有成功路径
的 cost-to-go 样本。这样成功/失败都会被记住，且中间 reward 不会污染结果。

旧版 `write(instruction, list[Step])` 仍保留用于兼容早期原型；真实 OSWorld 接入
必须使用 `write_episode()`。

## 5. `integration.py`

在线调用顺序：

```python
from evotree.integration import EvoTreeRuntime, inject_cheatsheet
from evotree.memory import EvoTreeMemory

memory = EvoTreeMemory(root_dir="/path/to/memory")
runtime = EvoTreeRuntime(memory)
runtime.begin(instruction)

# 每次调用模型前
result = runtime.before_model(current_observation, candidate_tools)
prompt_for_model = inject_cheatsheet(original_prompt, result.text)

# 环境执行动作后
runtime.after_action(
    current_observation,
    raw_action,
    next_observation=next_observation,
    note=execution_error,
)

# 任务结束，使用客观 evaluator 结果
runtime.finish(score=score)
```

hook 的职责：

- `before_model()`：只读记忆，返回结构化小抄、文本小抄和重排后的工具列表；
- `after_action()`：只暂存动作，不提前更新成功统计；
- `finish()`：根据客观结果创建 Episode，写入记忆并清空当前 runtime。

`write_enabled=False` 时，仍可读取旧记忆，但不会产生写入，供 Setting II 冻结评测使用。

## 6. 当前边界

本次只实现了适配层和 framework-neutral hook，没有直接修改 OSWorld-MCP 的 agent
运行代码。真正接入时仍需明确映射：

```text
当前 screenshot/a11y tree → observation_factory 输入
当前候选 MCP tools       → candidate_tools
模型 function call        → raw_action
最终 evaluator score      → runtime.finish(score=...)
```

这一步刻意独立于 benchmark 底座，避免在还没确认真实接口前侵入第三方运行代码。

## 7. 下一步

1. 阅读 OSWorld-MCP agent 决策循环，找到最小 hook 插入点；
2. 把当前实时 observation 转成 `app/window/dialog/facts`；
3. 先跑单任务 smoke，检查 prompt 注入、真实动作、结果写入；
4. 再产出 GUI/MCP 多轨迹，验证工具路径是否被经验统计正确推荐。

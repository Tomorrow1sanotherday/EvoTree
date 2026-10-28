# M2 小规模 OSWorld 对照实验

三组使用相同模型、任务顺序、`max_steps` 和单环境 CPU/TCG 配置：

- `baseline`：不启用 EvoTree。
- `frozen`：从固定 seed memory 开始，只读检索。
- `online`：从同一 seed memory 开始，允许 episode 写回。

`tasks_smoke.json` 只含音量任务；`tasks_5.json` 包含终端尺寸、音量、文本缩放、
回收站恢复和勿扰模式。这五个任务都在 OSWorld MCP 清单内，有直接 MCP 工具，
同时可由 GUI 完成；用于 seed 的目录重命名任务不进入评测。

## 环境

凭据只放环境变量，不把 API key 作为命令行参数：

```bash
export OSWORLD_MODEL=gpt-5.6-terra
export OSWORLD_API_URL=http://llmapi.evolventapi.com/v1
export OSWORLD_API_KEY='<从受控配置加载，不写入仓库或 shell history>'
export PYTHONPATH=/data/songweishuai/envs/osworld/lib/python3.12/site-packages
export DOCKER_HOST=unix:///data/songweishuai/evo-podman.sock
```

旧启动器和未隔离的 provider 必须串行执行 arm。每轮开始前检查现场，只允许使用启动器生成的
`evo-m2-*` 容器名；绝不批量清理容器或全局终止 QEMU。

## 运行

以下示例中的 frozen 与 online 必须使用同一 seed：

```bash
PYTHON=/data/songweishuai/envs/evo-osworld-20260829/bin/python
RUNNER=/data/songweishuai/projs/rh/OSWorld-mcp-run
REPO=/data/songweishuai/projs/rh/EvoTree/proj/EvoTree
OUT=/data/songweishuai/evotree_m2_smoke_20260901
SEED=/data/songweishuai/evotree_memory_real_20260901_mcp2/env-0

$PYTHON -m experiments.m2_small_eval.run_arm \
  --arm baseline --runner-root "$RUNNER" --output-root "$OUT" \
  --python "$PYTHON" --label smoke
$PYTHON -m experiments.m2_small_eval.run_arm \
  --arm frozen --runner-root "$RUNNER" --output-root "$OUT" \
  --python "$PYTHON" --seed-memory "$SEED" --label smoke
$PYTHON -m experiments.m2_small_eval.run_arm \
  --arm online --runner-root "$RUNNER" --output-root "$OUT" \
  --python "$PYTHON" --seed-memory "$SEED" --label smoke
```

启动器拒绝覆盖已有 arm。重新实验必须使用新的 output root，不能删除旧结果后
原地重跑。

## 最小重复验证与并行

`parallel_pilot.py` 在固定任务分片上并行启动单环境 runner，比较无记忆与
从空记忆开始的 online，各重复三轮。算法保持当前实现；每个 worker 的
记忆逐轮继承，global_notes 仅在本分片共享。

使用前应用外部补丁 `0007-isolate-parallel-worker-ports-and-assets.patch`，
设置 `OSWORLD_PORT_LOCK_DIR`，并先验证两个容器同时启动、无端口串扰、
截图和 MCP 服务正常且能精确回收。

具体设计见 `doc/plan/2026-10-12/最小验证与并行评测.md`。可先用 `--dry-run`
查看任务分片与两臂顺序。运行后 `status.json` 和 `summary.json` 自动更新；
缺结果和基础设施错误会中止后续轮次，不使用残缺种子。

## 汇总

```bash
$PYTHON -m experiments.m2_small_eval.report \
  "$OUT" "$REPO/experiments/m2_small_eval/tasks_smoke.json" \
  --model "$OSWORLD_MODEL" --output "$OUT/summary.json"
```

有效动作统计复用 `OSWorldEpisodeAdapter`：reset、DONE、terminate 和 client
error 不计入动作。成功只来自 `result.txt`；frozen 的
`memory_before.json == memory_after.json` 才算只读验收通过。

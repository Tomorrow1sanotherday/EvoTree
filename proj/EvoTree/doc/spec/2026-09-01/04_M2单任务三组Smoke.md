# Spec 2026-09-01 - M2 单任务三组 Smoke

## 范围

在 A800 上对最大音量任务 `28cc3b7e-b194-4bc9-8353-d04c0f4d56d2` 串行运行
`baseline`、`frozen`、`online` 三组。固定模型为 `gpt-5.6-terra`，frozen 与 online
都从 `/data/songweishuai/evotree_memory_real_20260901_mcp2/env-0` 复制 seed。

正式结果目录为：

```text
/data/songweishuai/evotree_m2_smoke_20260901_v2
```

三组均使用 CPU/TCG，容器名分别为 `evo-m2-smoke-v2-baseline`、
`evo-m2-smoke-v2-frozen` 和 `evo-m2-smoke-v2-online`，全程串行运行。

## 结果

| arm | score | return code | 有效动作 | MCP | memory |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline | 1.0 | 0 | 3 | 3 | 不适用 |
| frozen | 1.0 | 0 | 2 | 2 | 前后 SHA-256 manifest 相同 |
| online | 1.0 | 0 | 4 | 4 | 前后 manifest 不同，新增音量任务树 |

online 的 `traj.jsonl` 共 6 行。适配器过滤 reset 与 `DONE` 后得到 4 个有效动作：
两次 `os.set_volume` 和两次 `os.get_volume`。最终 evaluator 读取音量为 100，整轮
score 为 `1.0`。中间步骤的 `reward=0` 未被解释为动作失败。

online 写回的音量任务树包含两条边：

- `os.set_volume`: `n=2`, `n_success=2`, `steps_after_success=[4, 2]`
- `os.get_volume`: `n=2`, `n_success=2`, `steps_after_success=[3, 1]`

这同时验证了整轮 score 被传播到路径中的每个有效动作，且终止动作没有进入经验树。

## 资源收尾

三组完成后，各自的精确容器查询均为空，固定端口不再监听，provider 记录的专属
QEMU PID 已消失。online 的 provider 日志明确记录 PID `56277` 的归属和精确终止，
未执行全局容器或 QEMU 清理。

online 在停止录屏时收到 HTTP 400，最终未生成视频；这不影响轨迹、evaluator、
memory 写回或 launcher 退出码。最终 `run.json.returncode=0`。

## 汇总完整性

汇总写入：

```text
/data/songweishuai/evotree_m2_smoke_20260901_v2/summary.json
```

汇总器现在额外读取每个 arm 的 `run.json.returncode`，输出 `run_returncode`、
`infrastructure_ok` 和 arm 级 `complete`。结果文件存在但 launcher 非零退出时，
仍保留 score 作为诊断数据，但不会被标记为完整实验。

## 测试

汇总器修改后，Windows 同步副本与 A800 正式仓库的全量 pytest 均为
`137 passed`。重新生成 summary 时不再出现模块重复导入的 runpy warning。

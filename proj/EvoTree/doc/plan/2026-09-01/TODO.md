# TODO 2026-09-01

## 已完成

- [x] 修复并确认 `/data/songweishuai/envs/evotree` 可用，pytest 通过阿里镜像安装。
- [x] 在 Windows 同步副本和 A800 正式仓库运行全量测试：两边均为
  `126 passed`。
- [x] 将 EvoTree wrapper 接入 OSWorld-MCP 真实 agent 循环，并用
  `--use_evotree` 保持默认 baseline 不变。
- [x] 真实 MCP 单任务成功：prompt 包含 `[经验]`/`[注意]`，真实 function call
  写入 episode，最终 evaluator score `1.0` 更新 tree，`DONE` 不入树。
- [x] `write_enabled=False` 真实运行仍能检索经验，前后 SHA-256 manifest 经
  `cmp` 确认完全一致。
- [x] 同任务导入真实 GUI 与 MCP 成功轨迹，验证 MCP 两步路径的
  `val=0.81000` 高于 GUI 四步路径的 `val=0.65610`。
- [x] 共享服务器 provider 删除全局 QEMU kill，改为 container/PID/create-time
  精确归属与 conmon full-ID fallback。
- [x] 固定并验证 VM 依赖 `fastmcp==2.13.3`，消除 `fastmcp-slim` 冲突。
- [x] 清理本轮精确命名资源；未触碰其他用户的容器、QEMU 或 GPU 任务。
- [x] 确定 runner/provider 官方基线，将外部改动拆成三段可审查 patch；在隔离
  目录应用后与现场文件 SHA-256 完全一致，并通过 `py_compile`。
- [x] 新增 M2 baseline/frozen/online 单 arm 启动器，三组结果与 memory 隔离，
  拒绝覆盖已有实验，API key 只从环境读取。
- [x] 固定 1 任务 smoke 和 5 任务清单，排除已进入 seed 的目录重命名任务。
- [x] 新增 M2 汇总器，统一统计 score、有效动作、GUI/MCP 比例与 memory hash。
- [x] 修复真实轨迹中 structured executed action 被陈旧 prediction 覆盖的问题。
- [x] M2 工具本地与 A800 全量测试：均为 `136 passed`。
- [x] 在 A800 串行完成单任务 baseline/frozen/online smoke：三组 score 均为
  `1.0`，frozen memory hash 不变，online 写入成功，精确资源已回收。
- [x] M2 汇总器纳入 `run.json.returncode`，区分 evaluator 结果与完整基础设施成功。
- [x] 为 OSWorld 任务文件和浏览历史下载增加 `HF_ENDPOINT` 镜像优先、官方回退，
  并将补丁保存为 `0004-fix-HuggingFace-task-download-mirror.patch`。
- [x] 为 MCP 依赖安装增加失败标记与 `/tmp/mcpdeps.log` 尾部回传，保存为
  `0005-diagnose-MCP-dependency-install.patch`。

## 后续

- [ ] smoke 通过后扩到固定 5 任务，汇总三组成功率、动作数与 GUI/MCP 比例。
- [ ] 完成固定 5 任务的 frozen/online 两组评测并生成三组 summary。
- [ ] 根据 MCP 依赖日志修复 Playwright 安装链路后，重新完成 online 评测。
- [ ] 评估是否将完整 `/data/songweishuai/projs/rh/OSWorld-mcp-run` fork 纳入
  独立 Git 仓库；当前任务相关差异已由 EvoTree patch series 覆盖。
- [ ] 用真实 a11y tree 样本继续细化 dialog/facts 提取，避免把无关窗口文本
  带入状态签名。
- [ ] 在固定任务集上重复 GUI/MCP 配对实验，报告 TIR 的均值、方差和失败路径
  对排序的影响；不在共享服务器上启动全量评测。

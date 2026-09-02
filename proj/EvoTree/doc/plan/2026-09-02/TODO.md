# TODO 2026-09-02

## 已完成

- [x] 固定五个 OSWorld 任务，串行完成 `baseline`、`frozen`、`online` 三组评测。
- [x] 汇总每个任务和 arm 的 evaluator score、成功率、有效动作数与 GUI/MCP 比例。
- [x] 验证每组 `run.json.returncode=0`，区分 evaluator 结果与基础设施完整性。
- [x] 验证 `frozen` memory 前后 SHA-256 manifest 不变，`online` 产生写入并生成 6 个
  `tree.json`。
- [x] 完成 M2 五任务最终结果归档：`doc/spec/2026-09-02/06_M2五任务三组最终验收.md`。
- [x] 本地与 A800 canonical repo 全量 pytest 均为 `139 passed`。
- [x] 固定 20 个 OSWorld 任务，启动 baseline 难度校准。
- [x] 配置串行接力：online 从空 memory 开始，frozen 复制 online 最终 memory
  并禁用写入，不读取旧 M1/M2 seed。
- [x] 报告器增加真实 prompt 的 `[经验]`/`[注意]` 命中数和 memory 文件变更统计。
- [x] 新增统计后本地与 A800 canonical repo 全量 pytest 均为 `140 passed`。
- [x] prompt 命中统计收紧为每步最后一条 user 消息，并验证 20 个任务 ID 唯一；
  本地与 A800 canonical repo 全量 pytest 均为 `141 passed`。

## 后续

- [ ] 完成 20 任务 baseline，按逐任务得分划分简单、中等和困难子集。
- [ ] 完成 20 任务 online warm-up，核对经验写入和跨任务 prompt 命中。
- [ ] 完成 20 任务 frozen 只读复测，核对 memory 前后散列完全一致。
- [ ] 汇总 baseline/online/frozen 的总体和难度子集指标，判断 TIR 是否提升。
- [ ] 在固定任务集上重复 GUI/MCP 配对实验，报告 TIR 均值、方差和失败路径对排序的影响。
- [ ] 分析 online 未超过 baseline 的原因，区分模型随机性、任务难度与经验注入质量。
- [ ] 用真实 a11y tree 样本继续细化 dialog/facts 提取，避免无关窗口文本进入状态签名。
- [ ] 评估是否将完整 `/data/songweishuai/projs/rh/OSWorld-mcp-run` fork 纳入独立 Git 仓库。

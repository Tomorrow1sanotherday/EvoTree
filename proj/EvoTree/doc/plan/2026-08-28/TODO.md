# TODO 2026-08-28

## 已完成

- [x] 修复服务器 `/data/songweishuai/envs/evotree`：使用 Miniconda
      Python 3.13 重建，阿里镜像安装 pytest。
- [x] 服务器 EvoTree 全量测试：`126 passed`。
- [x] 新增 `osworld_integration.py`：OSWorld observation、MCP tool schema、
      model prediction、env action、evaluator score 的独立 wrapper。
- [x] `use_evotree=False` 返回原始 agent/env，保持裸 baseline。
- [x] smoke 测试验证 prompt 出现 `[经验]` 和 `[注意]`、真实动作进入 episode、
      `finish(score)` 写出 `memory/*/tree.json`，终止动作跳过，冻结模式不写入。
- [x] M1 离线多轨迹验证：同一任务 GUI 三步成功轨迹 + MCP 一步成功轨迹，
      检索按 `val = p * gamma**d` 将 MCP 工具路径排在 GUI 之前。
- [x] 服务器单任务 smoke：`/data/songweishuai/evotree_smoke_20260828` 生成
      `traj.jsonl`、`result.txt` 和 tree 统计；prompt 出现 `[经验]`/`[注意]`，
      `write_enabled=False` 的 frozen 目录无文件。
- [x] `[注意]` 渲染标题统一为带方括号的 `[注意]：`。

## 后续真实数据

- [ ] 在 OSWorld-mcp-run 中用真实模型/VM 跑一个单任务，保存
      `traj.jsonl`、`result.txt` 与 EvoTree tree.json（当前服务器无可用 API key，
      已完成无 VM 的协议级 smoke）。
- [ ] 在真实 VM/API 上对同一任务各跑 GUI/MCP 成功轨迹，记录真实 TIR 对比。
- [ ] 依真实 a11y tree 细化 `build_observation` 的 dialog/facts 提取。

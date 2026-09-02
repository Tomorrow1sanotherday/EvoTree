# TODO 2026-08-26

> 本次新增：真实 OSWorld 轨迹适配层与在线 agent 接入 hook。
> 代码流程：本地先写 → 测试通过 → 补 spec / TODO → 再同步服务器。

## 本次完成 ✅

- [x] `evotree/episode.py`：统一 `Episode` / `EpisodeStep` / `RuntimeStep`
- [x] `evotree/osworld_adapter.py`：OSWorld `traj.jsonl` + `result.txt` 适配
- [x] 明确区分 episode-level success 与中间 reward=0
- [x] 跳过 reset、DONE/terminate、client error
- [x] `evotree/integration.py`：`EvoTreeRuntime` 的 begin/read/after_action/finish hook
- [x] prompt advisory 注入：`inject_cheatsheet()`
- [x] Episode 级成功/失败路径写入 `EvoTreeMemory.write_episode()`
- [x] `retrieval.py` 通过公开 `node_for_signature()` 访问森林，不再依赖私有 `_nodes`
- [x] parser 使用 `JSONDecoder.raw_decode`，支持嵌套工具参数
- [x] README 更新
- [x] 本地全量测试：118 个通过
- [x] spec：`doc/spec/2026-08-26/05_真实轨迹适配与Agent接入.md`

## 服务器同步状态 ⚠️

- [ ] 本轮代码和文档尚未同步：SSH 公钥认证当前返回 `Permission denied (publickey,password)`。
- [ ] 认证恢复后：同步代码、tests、README、spec、TODO。
- [ ] 服务器 `evotree` uv 环境跑全量测试。
- [ ] 测试通过后再在服务器 git commit。

## 下一步：真实 OSWorld 接入

### 1. 读取并确认底座接口

- [ ] 阅读 OSWorld-MCP `run_multienv_e2e.py` 的 agent 决策循环。
- [ ] 阅读具体 agent（`mm_agents/`）的模型调用方法。
- [ ] 找到真实 current screenshot / accessibility tree / window title 的来源。
- [ ] 找到当前候选 MCP tool list 生成位置。
- [ ] 找到 episode 最终 evaluator score 回调位置。

### 2. 建立独立 wrapper（不要直接侵入底座）

- [ ] 将 EvoTree 作为独立模块接入 agent。
- [ ] 每次模型调用前调用 `runtime.before_model()`。
- [ ] 将返回的 `text` 追加到 prompt。
- [ ] 使用 `reranked_tools` 生成工具展示顺序。
- [ ] 每次动作执行后调用 `runtime.after_action()`。
- [ ] 任务结束后调用 `runtime.finish(score=...)`。
- [ ] 增加 `use_evotree=False` 开关，保证裸 baseline 不变。

### 3. 单任务 smoke（接入后的第一验收）

- [ ] 只跑一个任务，不跑全量。
- [ ] 容器使用独有的 `evo-` 前缀名称。
- [ ] 不使用批量容器/qemu 清理命令。
- [ ] 确认 prompt 中真实出现 `[经验]` / `[注意]`。
- [ ] 确认 `traj.jsonl` 能产生真实 GUI/MCP 动作。
- [ ] 确认 `finish(score=...)` 后 tree.json 有统计。
- [ ] 确认 `write_enabled=False` 时不产生写入。

### 4. M1 多轨迹验证

- [ ] 同一任务产出 GUI 成功轨迹。
- [ ] 同一任务产出 MCP 成功轨迹。
- [ ] 验证工具路径与 GUI 路径同时进入同一任务树。
- [ ] 验证成功率、剩余步数、前瞻推荐的统计正确。
- [ ] 验证工具重排是否将高价值工具置前。
- [ ] 验证失败轨迹和 global notes 的写入。

## 代码质量门槛

每次提交前必须：

- [ ] 本地 `python -m pytest tests/ -q` 全绿。
- [ ] 不让公共 API 直接访问其他模块的私有字段。
- [ ] 新增特性写入当天 `doc/spec/<日期>/`。
- [ ] 更新当天 `doc/plan/<日期>/TODO.md`。
- [ ] 明确写出哪些是真实实现、哪些仍是 placeholder。
- [ ] 同步服务器后，在服务器 uv 环境再跑一次完整测试。
- [ ] 服务器代码测试通过后再 git commit。

## 永久安全规则

- [ ] 只删除自己的精确容器名/进程 PID。
- [ ] 绝不使用 `podman rm -f $(podman ps -aq)`。
- [ ] 绝不使用 `pkill -x qemu` / `pkill -f qemu`。
- [ ] 绝不使用"快照差集删除"。
- [ ] 任何查询返回空时，默认视为查询失败，不视为"没有别人"。
- [ ] 共享 root 服务器上宁可留下自己的残留，也绝不冒险碰别人的任务。

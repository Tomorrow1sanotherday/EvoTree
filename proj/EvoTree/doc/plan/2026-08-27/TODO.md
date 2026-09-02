# TODO 2026-08-27

> EvoTree 核心记忆系统的实现进度与待做。对应 idea 文档 §0–§4 与 plan 的 M1/M2。
> 原则：纯 Python、可本地测（不依赖服务器）、每块配单元测试、每块写完补 spec。

## 已完成 ✅

- [x] `evotree/types.py` —— 核心数据类型（Template/StateSig/ActionTmpl/StateNode/ActionEdge）
- [x] `evotree/scoring.py` —— 边价值打分（纯函数，val = p·γ^d + UCB）
- [x] `tests/test_scoring.py` —— 15 个测试全过
- [x] `pyproject.toml` / `__init__.py` —— 项目骨架
- [x] spec: `doc/spec/2026-08-27/01_核心类型与打分.md`

## 已完成（续）✅ —— 完整闭环打通，54 测试全过

- [x] `evotree/forest.py` —— 经验森林（13 测试）
- [x] `evotree/distill.py` —— 三道清洗占位版 + 可替换接口（11 测试）
- [x] `evotree/notes.py` —— 注意栏（8 测试）
- [x] `evotree/retrieval.py` —— 五拍检索 + 前瞻 k=1 + 小抄渲染
- [x] `evotree/memory.py` —— 顶层封装 read/write（10 测试）
- [x] 闭环 Demo 跑通（小抄含推荐/前瞻/注意/重排四机制）
- [x] spec: `doc/spec/2026-08-27/02_完整闭环.md`

## 已完成（续）✅ —— B 阶段起步：真实数据解析（67 测试全过）

- [x] `evotree/distill_osworld.py` —— 解析 OSWorld-MCP 真实 prediction 格式（13 测试）
- [x] `tests/fixtures/rename_dir_gui_6steps.jsonl` —— 真实轨迹 fixture
- [x] 真实数据闭环验收：真实 rename 轨迹 → 森林 → 小抄（跑通）
- [x] spec: `doc/spec/2026-08-27/03_真实数据解析B阶段.md`

## 已完成（续）✅ —— φ 的 window 归一化（76 测试全过）

- [x] `evotree/normalize.py` —— window 归一化：抹文件名保状态标志（9 测试）
- [x] `distill_state` 接入归一化，解决"同类任务换文件状态对不上"（假冷启动）
- [x] 真实 demo 验收：data.xlsx 学、data2.xlsx 读 → cold_start=False（对上了）
- [x] spec: `doc/spec/2026-08-27/04_状态归一化.md`

## ⚠️ B 阶段剩余（依赖真实评测数据 / 非当前瓶颈）

- [ ] **状态签名粒度**（命中 idea 地基①）：连续两步主键相同时前瞻断裂。
      需真实多步轨迹验证。
- [ ] **φ 从 a11y tree 提取**：当前 app/dialog 从 dict 直读。真实需从
      accessibility tree 提取 —— **依赖真实评测观测数据，离线做不了**。
- [ ] **τ 的 embedding 匹配**：当前规则占位。需外部依赖（sentence-transformers），
      **非当前瓶颈**（规则版已能让同类任务归堆），暂缓。
- [ ] **多轨迹验证**（关键，下一个重点）：需同一任务的 GUI+MCP 多条轨迹，验证
      val 把工具路径推到 GUI 之上（TIR 提升机制）。依赖服务器批量产轨迹。

## 历史待做（已全部完成，保留供追溯）

### 1. `evotree/forest.py` —— 经验森林 ✅（下一个写）
- [ ] `ExperienceTree`：单个任务模板 τ 的一棵树，管理状态节点 + 动作边
    - [ ] 状态按 `StateSig.key()` 索引（不同来路自动汇合 → 这就是"森林是图"）
    - [ ] 边按 `edge_key()` 索引（同一动作自动归堆）
    - [ ] `upsert_step()`：写入一步（找/建状态、找/建边、更新 n/n_success）
    - [ ] `record_steps_after_success()`：本轮成功后从终点倒数，填 steps_after_success（cost-to-go 的 MC 计算，对应 D3）
    - [ ] JSON 持久化：`save()`/`load()`，落到 `memory/<template_id>/tree.json`
- [ ] `ExperienceForest`：管理所有树（按 template_id 分册）
- [ ] 维护：低访问低成功边定期删除（**绝不"只留最短路径"**）
- [ ] 配 `tests/test_forest.py`

### 2. `evotree/distill.py` —— 三道清洗
- [ ] `distill_task(instruction) -> Template`（τ）：指令 → 模板 + 槽位
    - pilot 阶段可先用规则/占位；embedding 匹配是后续升级
- [ ] `distill_state(observation) -> StateSig`（φ）：截图/a11y tree → 签名
    - 主键 (app, window, dialog) 确定性提取；facts 先留空/占位
- [ ] `distill_action(raw_action) -> ActionTmpl`（α）：原始动作 → 模板
    - TOOL：工具名 + 参数槽位化（精确）
    - GUI：语义描述，丢坐标
- [ ] **依赖真实数据格式**：参照 `plan/first_success_result/traj.jsonl` 的字段
    （instruction / action / messages 里的 function calling 格式）
- [ ] 配 `tests/test_distill.py`（用构造的假轨迹）

### 3. `evotree/notes.py` —— global_notes 注意栏
- [ ] 任务无关的坑位手册：`(工具名) -> [坑/报错规律]`
- [ ] `add_note()` / `query(action)`：写入 + 查询
- [ ] 硬规则：**只警告不推荐**（不知道任务目标，只配说"小心"）
- [ ] JSON 持久化：`global_notes.json`

### 4. `evotree/retrieval.py` —— 五拍检索
- [ ] 固定五拍（对应 idea §0）：
    1. 定位：指令→哪棵树；观测→φ→当前状态 s
    2. [推荐]：s 的出边打分取前 2–3，各前瞻 1 层（k=1，D4）
    3. [注意]：查 global_notes
    4. 拼小抄（哪栏空整段不出现）
    5. 工具重排
- [ ] 冷启动 = 推荐栏空返回（不是特殊分支）
- [ ] 配 `tests/test_retrieval.py`

### 5. `evotree/memory.py` —— 顶层封装
- [ ] `EvoTreeMemory`：对外只暴露两个方法
    - [ ] `read(observation) -> 小抄`（每步动手前调）
    - [ ] `write(trajectory)`（每轮跑完调，成败都写）
- [ ] `write_enabled` 开关（为 Setting II 冻结评测准备）
- [ ] config：相似度阈值、γ、UCB系数、报警阈值
- [ ] 配 `tests/test_memory.py`（端到端：写入几条假轨迹→读出小抄）

## 之后（M1 地基探针，需要真实数据/服务器）
- [ ] 用真实轨迹跑三道清洗，测**签名归并纯度**（地基①）
- [ ] 测**每条边样本量 n 分布**（地基②）
- [ ] failure-mode 分解，界定**增益天花板**（地基③）

## 注意事项
- 每块写完：跑测试 + 补 `doc/spec/<日期>/` 说明 + 更新本 TODO
- 全部核心写完、测试通过、用户审过 → 再一次性同步到服务器 `proj/EvoTree` 并 git commit
- 服务器同步是"最后"做（用户明确要求），本地先成型

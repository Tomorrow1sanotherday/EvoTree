# 实验三报告

**日期**：2026-09-16 / 09-17
**产物**：`runs/rh_exp3_20260916`（baseline 保留首轮，frozen/online 为 09-17 重跑）
**种子**：`temp/rh_seed_v3`（v2 树 + global_notes 新增执行层知识）
**代码**：commit `636382c`（薄硬币边不进推荐栏 + 冻结臂关探索项）
**任务集**：`temp/rh_tasks_exp2.json` 6 个纯终端任务，max_steps 12

---

## 一、成绩

| 任务 | 内容 | baseline | frozen | online |
|---|---|---|---|---|
| 6f56bf42 | cp 到三个目录 | **1.0** | **1.0** | 0.0 |
| 23393935 | 递归复制 jpg | 0 | 0 | 0 |
| 5c1075ca | 按层级复制 ipynb | 0.0 | 0.0 | 0.0 |
| 4127319a | 统计 php 行数 | 0.0 | 0 | 0 |
| 5ced85fc | 每行追加 br | **1.0** | **1.0** | **1.0** |
| 4d117223 | 改权限 644 | **1.0** | **1.0** | **1.0** |
| **均分** | | **0.50** | **0.50** | **0.33** |

三臂全部 `rc=0` 正常跑完，无超时。

**frozen 与 baseline 逐任务完全一致（6/6 相同）**，不是均分巧合。
online 唯一差异是 6f56bf42，轨迹显示是模型随机性（见第四节）。

---

## 二、好消息：代码修复生效了

exp2 的教训是 `CTRL+ALT+T` 这条**薄硬币边**（`computer_use.key · 2次·0.50`）挤进了推荐栏。

对比同一任务（23393935）的第一步注入：

**exp2（修复前）**
```
经验] 推荐：
  computer_use.type —— 4次·成功率0.50·平均还差2步
  computer_use.key  —— 2次·成功率0.50·平均还差5步（低置信）
      曾用：CTRL+ALT+T（1/2 成）        ← 就是这条
  os.open_shell     —— 6次·成功率0.33·平均还差4步
```

**exp3（修复后）**
```
[经验] 推荐：
  computer_use.type —— 4次·成功率0.50·平均还差2步
      曾用：find ~/photos -type f -name '*.jpg' -exec cp -t ~/cpjpg -- {} +（1/1 成）
      曾用：find ~ -type d -name photos 2>/dev/null（1/1 成）
  os.open_shell —— 6次·成功率0.33·平均还差4步
```

`computer_use.key` 那条**已从推荐栏消失**——`is_too_shaky`（n<3 且 p<0.67）判据按设计拦住了它。

---

## 三、坏消息：三条执行层经验一次都没送达

### 3.1 事故

frozen 臂里 agent 的行为与 exp2 一模一样：

- 23393935：第 1 步就按 `CTRL+ALT+T`，随后连按 **11 次**，最后 `FAIL`
- 4127319a：连按 **11 次** `CTRL+ALT+T`，最后 `FAIL`

而种子里明明写着：

```json
"computer_use.key": ["此环境 Ctrl+Alt+T 反复按也常打不开终端（四轮实验实测，一次连按 10 次全败）。要开终端优先用 os.open_shell 工具。"],
"computer_use.type": ["敲命令前必须先确认终端已打开并持有焦点……", "文件路径常在 ~/Desktop 之下……"]
```

### 3.2 根因：[注意] 栏的匹配是精确字符串比对，而 key 写错了

`retrieval.py:_build_cautions` 的实现：

```python
def _build_cautions(notes, tools):
    for tool_name in sorted(tools):          # tools = candidate_tools ∪ 推荐里的工具名
        for note in notes.query(tool_name):  # 精确 get()
            cautions.append(f"{tool_name}: {note}")
```

而运行时 `candidate_tools` 的真实内容是：

```
真实 tool_list 工具名: ['computer_use']
```

**整个环境只有一个 MCP 工具叫 `computer_use`**（子动作 key/type/click/scroll 是它的 `action` 参数取值）。我们笔记的 key 写成了 `computer_use.key` / `computer_use.type` —— 子动作名，**不是工具名**。

`computer_use` ≠ `computer_use.key`，精确匹配必然失败。

### 3.3 证据

从事件日志（`debug-20260917@*.log`）统计：

| 臂 | `[经验]` 注入次数 | `[注意]` 注入次数 |
|---|---|---|
| frozen | 3 | **0** |
| online | 10 | **0** |

三条以中文写就的警告（"此环境…"、"敲命令前…"、"文件路径常在…"）在整个实验中出现 **0 次**。

---

## 四、注入覆盖面：只有 1 个任务真正拿到了经验

| 任务 | τ 命中种子树 | 运行时实际注入 |
|---|---|---|
| 23393935（照片） | ✅ | ✅ 3 次 |
| 6f56bf42（cp 三目录） | ✅ | ❌ **0 次** |
| 5c1075ca / 4127319a / 5ced85fc / 4d117223 | ❌ | ❌ 0 次 |

6 个任务里**只有 2 个的 τ 能命中种子树**（种子共 4 棵树，另 2 棵是 settings/battery 类，不在这批任务里）；而这 2 个里只有 1 个真的注入了。

**为什么 6f56bf42 也是 0 次**：种子树的节点签名是 `('unknown','','')`（建树时窗口信息不可用），运行时必须**恰好也解析成 unknown** 才能命中该节点。VM 窗口注册慢时退化成 unknown（命中，如 23393935）；正常解析出 `terminal` 时就对不上（不命中，如 6f56bf42）。

也就是说，**注入能否发生取决于 VM 窗口注册时机的运气**，不是可控的实验条件。

---

## 五、结论

### 5.1 这轮实验能说什么

**不能**说"经验注入无效"。三条执行层经验从未送达 agent 面前，frozen 臂实质上是一次**没有经验的对照**——所以 frozen ≡ baseline 是**预期结果**，不构成对方法否证。

**能**说：

1. **薄硬币边判据有效**：exp2 的噪声源（`CTRL+ALT+T · 2次·0.50`）已被拦截。这是本轮唯一干净的正向结论。
2. **执行层瓶颈被再次确认**：agent 的失败模式（连按 11 次 `CTRL+ALT+T`、命令打进虚空）与 exp2 完全一致，两轮复现。
3. **稳定性是好消息**：同配置重跑，frozen 6 个任务成绩与上轮逐任务一致（上轮 3 个 time_limit 这轮都跑出真分数，且与 baseline 相同）。

### 5.2 三个必须修的缺陷

| # | 缺陷 | 修法 |
|---|---|---|
| **P0** | 笔记 key 用子动作名（`computer_use.key`），与真实工具名（`computer_use`）不匹配 → [注意] 栏永远为空 | 笔记改挂 `computer_use`（单键），或在 `_build_cautions` 里做前缀匹配 `tool_name.split(".")[0]` |
| **P1** | 种子树节点签名是 `unknown`，注入与否取决于 VM 时序运气 | 建树时用稳定签名（如固定 `('unknown','','')` 归一化），或检索时对 unknown 节点做兜底匹配 |
| **P2** | 种子覆盖太窄：6 个任务只有 2 个能命中 τ | 让种子树覆盖评测任务集，或改用有信号空间的任务集 |

### 5.3 对 online 臂的观察

online（0.33）比 baseline 低一个任务，差异点是 6f56bf42：

```
baseline/frozen：step1 敲 cp 命令 → step2 再敲一次确认 → DONE   → 1.0
online：        step1 敲 cp file1 dir{1,2,3}/          → DONE   → 0.0
```

online 臂 `explore=True`（UCB 探索项开着），但**这一轮它在该任务上根本没拿到注入**（与 frozen 同样条件），所以这个 0.0 是 temperature=1.0 下的采样差异，**不是探索项有害的证据**。

online 写入正常：运行后记忆从 4 棵树增长到 8 棵（新增 `Append_br_*`、`Change_the_permission_*`、`Copy_all_files_matching_*`、`Use_terminal_command_to_count_*`），读写链路工作正常。

---

## 六、下一步建议

**先修 P0，这是最小改动、最大杠杆**：

1. 把 `rh_seed_v3/global_notes.json` 的 key 从 `computer_use.key` / `computer_use.type` 改成 `computer_use`
2. 本地验证 `render_cheatsheet` 用真实 `candidate_tools=["computer_use"]` 能渲染出三条警告
3. 重跑 frozen 臂，看 agent 是否还会连按 `CTRL+ALT+T`

**P0 已本地验证通过**（2026-09-17，两个 key 合并成 `computer_use` 后）：

```
notes keys: ['computer_use']

=== 修复后小抄（真 candidate_tools=[computer_use]）===
[注意]：
  computer_use: 此环境 Ctrl+Alt+T 反复按也常打不开终端（四轮实验实测，一次连按 10 次全败）。要开终端优先用 os.open_shell 工具。
  computer_use: 敲命令前必须先确认终端已打开并持有焦点（用 os.open_shell 打开后立即敲）——否则命令文本会打进桌面/文件管理器搜索框而不执行，屏幕几乎无变化，极易误判为已完成（实测命令完全正确仍因此 0 分）。
  computer_use: 文件路径常在 ~/Desktop 之下（如此环境照片目录是 ~/Desktop/photos 而非 ~/photos）——先 ls 或 find 确认再执行，别猜。
```

（验证用临时目录在 `temp/` 下建、验完即删，未留残留。）

P0 修好之前，重跑多少次都是在测"没有经验的 agent"。

修好之后再决定 P1/P2：
- **P1**（签名稳定性）决定注入能否可靠发生
- **P2**（覆盖广度）决定实验有没有信号空间——当前 6 个任务里 4 个压根没经验可用，样本太小

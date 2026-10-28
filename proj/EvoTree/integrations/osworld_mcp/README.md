# OSWorld-MCP 外部补丁系列

本目录把 A800 上未受 Git 管理的 `OSWorld-mcp-run` 关键改动保存为可审查的
patch series。补丁目标是 OSWorld 源码树根目录，包含：

1. `0001-feat-add-EvoTree-runner-hooks.patch`
   - 增加默认关闭的 `--use_evotree` 与冻结写入开关。
   - 将 observation、候选工具、模型动作和 evaluator score 接入 EvoTree wrapper。
   - 确保 worker 使用 wrapper 后的 agent/env 配对。
   - 从 `OSWORLD_API_KEY` 读取凭据并在日志中脱敏。
2. `0002-chore-add-TCG-Podman-provider-prerequisite.patch`
   - 保存真实 smoke 所依赖的既有 TCG、Podman、softpipe 和 MCP 注入适配。
   - 这是外部目录在本轮安全修复前的前置状态，不代表 EvoTree 新增逻辑。
   - 该中间版本仍有旧的全局 QEMU 回收，不能单独运行。
3. `0003-fix-scope-QEMU-cleanup-and-pin-FastMCP.patch`
   - 用 container descendant、PID、create time 与 command 精确追踪 QEMU。
   - cgroup inspect 失败时使用包含完整 container ID 的 conmon 后代作为 fallback。
   - 删除全局 `pkill qemu`，固定并验证 `fastmcp==2.13.3`。
4. `0004-fix-HuggingFace-task-download-mirror.patch`
   - 读取 `HF_ENDPOINT`，让任务文件和浏览历史下载优先使用国内镜像。
   - 镜像失败后回退到原始 `huggingface.co` URL。
   - 缓存键仍使用原始 URL，切换下载端点不会重复缓存文件。
5. `0005-diagnose-MCP-dependency-install.patch`
   - 为 VM 内 MCP 依赖安装增加成功/失败标记。
   - 安装失败或校验失败时回传 `/tmp/mcpdeps.log` 尾部，避免只看到笼统的
     `ModuleNotFoundError`。
6. `0006-fix-window-title-column-and-app-list-race.patch`
   - `get_current_apps()` 把 `wmctrl -xl` 的**客户机名**列当成了窗口标题
     （`parts[3]` → 应为 `parts[4]`）。后果：所有窗口的 `title` 都等于主机名
     `user-virtual-machine`，状态签名 `(app, window, dialog)` 里的 window 变成
     常量，完全失去区分度。
   - `_get_obs()` 的取窗口重试循环只在抛异常时重试，窗口列表尚未就绪
     （`cur_id not in app_list`）时不重试，直接得到 `cur_app=None`。TCG 软件
     模拟下开窗后 `wmctrl -xl` 要 >25s 才列得出，这个窗口期很宽。顺带补上
     原循环缺失的 `break`（原实现无论成败都恒定调用三次）。

   2026-09-08 真机实测证据（`temp/rh_probe_wmctrl2.py`）：

   ```text
   cur_app       = 'Org.gnome.Nautilus'
   cur_window_id = '0x04400008'
   apps = {'0x0480000a': {'app_name': 'Gnome-terminal',
                          'title': 'user-virtual-machine'},
           '0x04400008': {'app_name': 'Org.gnome.Nautilus',
                          'title': 'user-virtual-machine'}}
   ```

   两个不同应用共享同一个假 title；窗口 ID 两侧格式一致（都是补零到 8 位），
   **不存在 ID 格式不匹配**——早前的补零猜测已被这次实测否定。

   受影响的下游：`evotree/osworld_integration.py::_clean_window_title` 是本
   bug 的下游兜底（把主机名识别出来丢弃）。上游修好后它退化成保险丝，正则
   `^[\w-]+-virtual-machine$|^localhost$` 足够窄，不会误伤真实标题。

## 基线与目标

### 2026-10-12 并行运行补丁

### 2026-10-13 动作执行修复

`0012-preserve-operations-before-termination.patch` 以r5实际运行副本为基线。
多工具回复遇terminate时保留前面操作，追加DONE/FAIL并停止；移除agent在
最后一个模型turn覆盖整个操作列表的FAIL分支，由runner控制步数。动作
解析异常不再合成DONE。记忆算法不变。

回归脚本 `scripts/verify_osworld_dispatch.py /path/to/mm_agents/owl_agent.py`
用AST提取实际动作组装代码，在不启动VM/模型的情况下检查11种执行语义。
`parallel_pilot` 现在也会检查每次回复提出的调用数/工具顺序与轨迹一致，
避免静默丢操作的运行被计为正常结果。

真实smoke使用完整predict和任务runner，在一次回复中完成重命名或Calc
复制后terminate，要求轨迹动作齐全且evaluator=1，再检验真实模型调用链。
该smoke通过后才重启正式实验，smoke记忆不进入实验种子。

`0007-isolate-parallel-worker-ports-and-assets.patch` 的基线是 A800 当前实际
runner/provider，包含此前保存到 `temp/osworld_port_fix/provider.py` 的动态端口
及 owned-QEMU 等修复；它不能直接接在早期 `0003` 的旧 provider 后面。
必须先 `git apply --check`，不匹配时检查差异，不强行模糊应用。

该补丁为独立 worker 提供有界端口范围、独立锁文件（网络盘使用 SoftFileLock）、
可配置 MCP 资产目录，并让并行 runner 的日志文件带 PID。容器 API 超时
默认 600 秒，避免并行复制容器层超过旧的 60 秒；隔离的 host-network 模式
查宿主监听端口，避免运行中容器枚举的 cgroup 错误。端口范围由
`run_arm --worker-slot` 分配，每个子进程仍是 `--num_envs 1`。
容器名由不同 label 分配；不要仅把旧 runner 的 `num_envs` 改成 4。

新增两个运行变量：`OSWORLD_PORT_LOCK_DIR`（启动器派生每个 slot 的锁文件）
和 `OSWORLD_MCP_ASSET_DIR`（含 server tgz、slim client）。没有隔离变量时
保留旧串行 provider 的默认锁与 `/tmp` 资产路径。

补丁由以下只读基线生成：

```text
runner base:
  /data/songweishuai/projs/rh/OSWorld-MCP/osworld/run_multienv_e2e.py
  sha256 37916ab3a57c8aeb72df6573dfa053d04b2428a71dfd58b123876d808d36488a

provider base:
  /data/songweishuai/projs/rh/OSWorld/desktop_env/providers/docker/provider.py
  sha256 fe4056ebef683bd341d65f17833c191736740761000775c05f4c9af1285f6879
```

完整应用三段 patch 后应得到：

```text
run_multienv_e2e.py
  sha256 480f3045d6131a6bde2b00fd51cc4741f0151b25d22b2835506780378397e057

desktop_env/providers/docker/provider.py
  sha256 690321a7b25ed612deb783372633ac5df7ff710336ff133ffc8bb6ab80d5c3e3
```

## 应用

在包含上述两份基线文件的干净 OSWorld 源码树根目录执行。必须完整应用三段，
不要在 `0002` 和 `0003` 之间启动环境：

```bash
git apply --check /path/to/patches/0001-feat-add-EvoTree-runner-hooks.patch
git apply /path/to/patches/0001-feat-add-EvoTree-runner-hooks.patch
git apply --check /path/to/patches/0002-chore-add-TCG-Podman-provider-prerequisite.patch
git apply /path/to/patches/0002-chore-add-TCG-Podman-provider-prerequisite.patch
git apply --check /path/to/patches/0003-fix-scope-QEMU-cleanup-and-pin-FastMCP.patch
git apply /path/to/patches/0003-fix-scope-QEMU-cleanup-and-pin-FastMCP.patch
git apply --check /path/to/patches/0004-fix-HuggingFace-task-download-mirror.patch
git apply /path/to/patches/0004-fix-HuggingFace-task-download-mirror.patch
git apply --check /path/to/patches/0005-diagnose-MCP-dependency-install.patch
git apply /path/to/patches/0005-diagnose-MCP-dependency-install.patch
git apply --check /path/to/patches/0006-fix-window-title-column-and-app-list-race.patch
git apply /path/to/patches/0006-fix-window-title-column-and-app-list-race.patch
```

也可以在干净 Git 工作树中一次提交整个系列：

```bash
git am /path/to/patches/*.patch
```

若任一步失败，停止并检查基线哈希；不要使用 fuzzy patch 或手工跳过失败 hunk。

## 验证

```bash
python -m py_compile \
  run_multienv_e2e.py desktop_env/providers/docker/provider.py
sha256sum \
  run_multienv_e2e.py desktop_env/providers/docker/provider.py
grep -n 'fastmcp=={expected_fastmcp}' \
  desktop_env/providers/docker/provider.py
HF_ENDPOINT=https://hf-mirror.com python -m py_compile \
  desktop_env/controllers/setup.py
```

还应检查 provider 不含任何全局 QEMU kill。共享服务器上不得用
`pkill -x qemu`、`pkill -f qemu` 或批量 Podman 清理来代替精确回收。

## r6 memory regressions (2026-10-13)

Apply `0013-record-corresponding-tool-call-and-result.patch` after the existing
runner patches. It records the wrapper's corresponding `tool_call` and each
action's `exe_result`, so later audits can check memory attribution and business
failure independently of execution completeness.

The EvoTree wrapper aligns parsed calls with returned actions before execution.
Ambiguous multi-call GUI records without an aligned call raise an error instead
of silently storing the first call. The offline adapter aligns complete legacy
turns before dropping termination records and rejects incomplete multi-call
turns. MCP parameter values remain examples outside the action aggregation key;
executable command text is retained intact.

For real observations containing `app_info`, state identity also includes the
preceding actual action and a digest of available structured sheet information.
The initial frame uses a stable task root, so asynchronous window registration
cannot turn the same reset into incompatible unknown/app nodes. Identical JSON
objects printed repeatedly by Calc's environment getter are decoded as one.
This distinguishes initial, source/destination, modified-content and later
execution contexts without extra model or VM calls. It is an implementation
change to state representation, not a claim that GUI target semantics or transfer
to new task parameters have been solved. Existing snapshots remain readable,
but old snapshots without context do not match contextual nodes; start a fresh
learning chain rather than inheriting r6's incorrectly attributed memory.

Known Calc mutators returning business `false` (or `None` for `copy_sheet`) are
recorded as cautions even when the MCP invocation reports `is_error=False`.
Queries returning false are not classified as failures. Self-loops no longer
produce lookahead advice, and successor hints describe candidates without
claiming one-step completion. Injection reports count unique model turns.

Run `python -m pytest -q` in the EvoTree project for regression checks. Real VM
memory smoke must additionally check executed calls against stored actions,
preserved parameters, fresh-reset seed retrieval and objective evaluator scores.
Fixed model responses in that smoke are harness diagnostics; their scores do not
measure model ability or memory benefit.

`0014-support-model-request-lock-on-shared-filesystems.patch` adds an explicit
`OSWORLD_MODEL_REQUEST_LOCK_MODE=soft` option for locks in project directories
on filesystems where `flock(LOCK_UN)` fails. It selects `SoftFileLock`; the
default keeps the existing `FileLock`. The smoke uses a uniquely scoped lock
inside its own runtime directory and releases it on normal completion.

`0015-normalize-gui-hotkey-names.patch` converts GUI hotkey names to lowercase
before passing them to PyAutoGUI. Uppercase model outputs such as CTRL/ALT/T,
ESC and F2 otherwise produce no intended key event. Verify the real installed
assembler with `python scripts/verify_osworld_hotkeys.py /path/to/mm_agents/owl_agent.py`.

`0016-validate-gui-actions-and-support-middle-click.patch` completes the advertised
middle-click action in both parsing and PyAutoGUI dispatch. It matches the parsed
tool name exactly and rejects unsupported GUI action names with `ValueError`,
allowing the existing format-correction request to run. This fixes r8's real
`action="TAB"` response (the correct form is `action="key", keys=["TAB"]`), which
previously raised an uninitialized `action_type` error outside format correction.
Verify the installed parser with `scripts/verify_osworld_action_schema.py`;
passing the saved failure JSON as a second argument also checks that regression.

`0017-record-real-runner-exceptions.patch` preserves exception type/message and
traceback instead of labeling every exception as a timeout. The failed attempt
is still invalid for evaluation and memory inheritance.

The pilot now supports `--resume` with an identical plan. Completed arms are
validated and reused; incomplete baseline episodes are archived before retry,
and scored baseline episodes are kept byte-for-byte. Partial online arms are
rejected because their memory may already include incomplete attempts. This is
explicit recovery from an infrastructure failure, not automatic score retries.

`0019-bind-current-state-to-capture-and-agent-prompt.patch` applies after the
read-only observation retry patch (`0018`) and existing OwlAgent fixes. It brackets
each screenshot with fresh application `env_info()` reads, records capture IDs,
action sequence numbers, acquisition timestamps and hashes, and retries only
observations when the two reads differ. Persistently unstable state is unknown;
the intended action is never treated as proof. Rendering freshness remains
explicitly unverified even when structured reads agree.

OwlAgent previously used `exe_result` for previous-tool context but omitted the
fresh `app_info` field from the actual model prompt. Patch 0019 includes that
current observation separately for every arm, including baseline. The retrieval
hook and agent now consume the same captured state. These are harness changes;
they do not make the Calc role guard application-independent or establish any
memory success-rate benefit. Original runs and bank snapshots are not rewritten.

Generate files for a new isolated runner using
`scripts/patch_osworld_observations.py --runner OLD --output NEW --patch-file FILE`.
The generator refuses unrecognized source and existing output files.

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

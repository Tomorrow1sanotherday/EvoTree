# Spec 2026-09-01 - Hugging Face 下载镜像回退

## 背景

共享 A800 访问 `huggingface.co` 不稳定，OSWorld 的 `_download_setup` 和浏览历史
setup 会在任务初始化阶段直接重试官方站点，导致任务结果缺失。已验证
`https://hf-mirror.com` 可以下载同一任务文件。

## 行为

外部补丁 `0004-fix-HuggingFace-task-download-mirror.patch` 为 OSWorld 的 setup
下载器增加 `_download_candidates()`：当 `HF_ENDPOINT` 非空且 URL 来自
`huggingface.co` 时，按 `[HF_ENDPOINT URL, 原始 URL]` 顺序下载；未设置变量时
保持原始单 URL 行为。每个候选端点沿用现有三次重试。

缓存文件名仍以原始 URL 的 UUID5 计算，因此切换镜像不会产生第二份缓存。任务
下载和 `_update_browse_history_setup` 共用该候选逻辑。

M2 启动器在未显式设置时设置 `HF_ENDPOINT=https://hf-mirror.com`，因此 baseline、
frozen、online 三组使用相同的下载策略；该变量不包含凭据。

同一启动器将 `OSWORLD_MCP_DEP_TIMEOUT` 默认设为 `1800` 秒。TCG/Slirp 下每次
新 VM 的 `fastmcp==2.13.3` 与 Playwright 安装可能超过原来的 900 秒上限；延长
等待只改变基础设施超时，不改变任务步数或 evaluator 语义。

## 验证

- `hf-mirror.com` 任务文件下载成功，字节数 `181662`，SHA-256 为
  `36ad3069b90c7e3968b355aa655368e0a4078bffb6459621bc6ad9f15662a607`。
- 对外部 setup 文件执行 `python -m py_compile`。
- EvoTree 本地与 A800 全量 pytest 均需保持全绿。

## MCP 依赖诊断

v2 online 已确认任务/VM 镜像下载使用 `hf-mirror.com`，但新 VM 的
`playwright` 安装仍未通过校验。补丁 `0005-diagnose-MCP-dependency-install.patch`
让 provider 在 pip 子进程失败时写入 `/tmp/mcpdeps.failed`，并把安装日志尾部写入
host launcher log；后续重跑可据此区分 PyPI 下载失败、安装冲突和 VM 环境问题。

# Spec 2026-09-01 · OSWorld 外部补丁系列

## 背景

真实评测目录 `/data/songweishuai/projs/rh/OSWorld-mcp-run` 没有 `.git`，此前
EvoTree runner hook、VM 依赖修复和 QEMU 安全回收只能依赖服务器现场文件。
本轮没有直接把该目录初始化成新仓库，避免把 results、VM 定制和未知历史混成
一个错误基线；改为在 EvoTree 仓库保存三段分层 patch。

产物位于 `integrations/osworld_mcp/patches/`，应用方式和完整 SHA-256 清单见
同目录 `README.md`。

## 分层原则

- `0001` 从 OSWorld-MCP 的 runner 上游文件出发，只形成当前真实 runner 所需的
  EvoTree hook、agent/env 配对和 API key 脱敏结果。
- `0002` 从 OSWorld provider 上游文件出发，记录真实 smoke 已经依赖的 TCG、
  Podman、softpipe 与 MCP 注入前置定制。这部分不是本轮新增，单独拆出是为了
  让 `0003` 保持最小且可审查。
- `0003` 只表示本轮 provider 安全差异：精确 QEMU ownership、conmon full-ID
  fallback、`fastmcp==2.13.3` 安装与导入验证。

`0002` 中保留了历史全局 QEMU 清理，不能作为可运行终点。部署必须连续完成
`0003`；最终文件不存在 `pkill qemu`。

## 可复现验证

在隔离目录中复制官方 runner/provider 基线，然后对每段先执行
`git apply --check` 再执行 `git apply`。验证结果：

```text
runner_exact_match
provider_exact_match
480f3045d6131a6bde2b00fd51cc4741f0151b25d22b2835506780378397e057  run_multienv_e2e.py
690321a7b25ed612deb783372633ac5df7ff710336ff133ffc8bb6ab80d5c3e3  desktop_env/providers/docker/provider.py
py_compile_ok
no_global_qemu_kill
```

两个目标哈希与 `/data/songweishuai/projs/rh/OSWorld-mcp-run` 现场文件一致，
因此该系列不是近似说明，而是可以重建当前已通过真实 smoke 的两个文件。

## 安全边界

- patch 与文档不包含 `/data/songweishuai/projs/rh/cofig/model.md` 中的凭据。
- 验证只在用户目录下的精确命名临时目录中进行，不启动容器或 QEMU。
- patch 应用失败时必须回到正确基线，不能用模糊匹配强行应用。
- 外部目录仍不由 EvoTree Git 接管；此系列只版本化任务相关的两个文件差异。

# Spec 2026-09-01 · 真实 OSWorld 验收与 M1 多轨迹

## 验收范围

本轮在 A800 上补齐 `EvoTreeRuntime` 进入 OSWorld-MCP 真实 agent 循环后的
端到端证据。EvoTree 正式仓库基线为 `4357691`；真实 runner 位于
`/data/songweishuai/projs/rh/OSWorld-mcp-run`，该目录没有 Git 元数据，因此
runner/provider 的接线与共享服务器安全修复不属于本仓库提交，具体改动在本文
单独列明。

环境：

- EvoTree venv：`/data/songweishuai/envs/evotree/bin/python`
- OSWorld runtime：`/data/songweishuai/envs/evo-osworld-20260829/bin/python`
- Python package overlay：`/data/songweishuai/envs/osworld/lib/python3.12/site-packages`
- Podman socket：`unix:///data/songweishuai/evo-podman.sock`
- VM 只使用 TCG/CPU，不使用 KVM/GPU

## 真实 MCP 单任务

任务 ID 为 `e0df059f-28a6-4169-924f-b9623e7184cc`，目标是把桌面目录
`todo_list_Jan_1` 重命名为 `todo_list_Jan_2`。结果与记忆目录：

```text
/data/songweishuai/projs/rh/OSWorld-mcp-run/results_evotree_real_20260901_mcp2
/data/songweishuai/evotree_memory_real_20260901_mcp2
```

真实调用 prompt 中同时注入 `[经验]` 和 `[注意]`。轨迹包含四条 runner 记录：

1. `reset`，不进入经验。
2. 调用 `os.rename_directory`，错误使用 `/home/oai/Desktop/...`。
3. 再次调用 `os.rename_directory`，改用 `~/Desktop/...` 后成功。
4. `DONE`，不进入经验。

`result.txt` 为 `1.0`。`finish(score=1.0)` 后，tree 中
`os.rename_directory` 边已有成功统计；错误调用的中间 reward 不被解释为整轮
失败，整轮结果经 `memory.write_episode()` 传播到路径边。tree 中没有 `DONE`
边，符合终止信号不作为动作经验的约束。

## 冻结读取

冻结运行使用 argparse 的实际开关 `--no-evotree_write_enabled`。结果与冻结记忆：

```text
/data/songweishuai/projs/rh/OSWorld-mcp-run/results_evotree_real_20260901_frozen1
/data/songweishuai/evotree_memory_real_20260901_frozen1
```

该轮 prompt 仍包含 `[经验]`、`[注意]`，说明只读检索有效。运行前后分别对
`global_notes.json` 和 `tree.json` 生成 manifest：

```text
/data/songweishuai/evotree_frozen1_before.sha256
/data/songweishuai/evotree_frozen1_after.sha256
```

`cmp` 返回成功并输出 `frozen_memory_unchanged`，两个 manifest 均为两行，哈希
完全相同：

```text
3269ecaa74083d1daf7f192b60ba5c519276d8a6c7df4421a8e87eda38edaa07  env-0/global_notes.json
aa1e04e94e386d455c19441cac78d00810084e39d0f329ef7156e565a9b6e52c  env-0/memory/I_have_a_directory_named__arg0__Can_you_/tree.json
```

该冻结 episode 的 evaluator score 为 `0`，但冻结验收只检查“可读、不可写”，
不把这次任务结果当作写入依据。

## M1 真实多轨迹

同一任务使用两条已有真实成功轨迹：

```text
GUI: /data/songweishuai/projs/rh/OSWorld-mcp-run/results_a/pyautogui/screenshot/sub2api-gpt-5.5/0/os/e0df059f-28a6-4169-924f-b9623e7184cc
MCP: /data/songweishuai/projs/rh/OSWorld-mcp-run/results_evotree_real_20260901_mcp2/mcp/screenshot/gpt-5.6-terra/0/os/e0df059f-28a6-4169-924f-b9623e7184cc
```

两边 `result.txt` 均为 `1.0`。经 `OSWorldEpisodeAdapter` 导入
`/data/songweishuai/evotree_m1_real_20260901` 后，GUI 有 4 个有效步骤，MCP 有
2 个有效步骤。在 `ucb_c=0`、`gamma=0.9` 时：

```text
rank 1 tool os.rename_directory  p=1.000 d=2.0 val=0.81000
rank 2 gui  computer_use.click   p=1.000 d=4.0 val=0.65610
```

候选输入顺序是 `['os.other', 'os.rename_directory']`，检索后变为
`['os.rename_directory', 'os.other']`。两条路径成功率相同，工具路径因成功所需
步骤更少而获得更高 `val = p * gamma**d`，验证了 TIR 的效率提升机制。

## 外部 runner 与 provider

`run_multienv_e2e.py` 保持 `--use_evotree` 默认关闭；开启时为每个 worker 创建
独立的 `env-N` memory，并把 wrapper 后的 agent 与 env 成对交给 worker。
`--api_key` 默认从 `OSWORLD_API_KEY` 读取，日志中的 key 被替换为
`<redacted>`，避免凭据进入命令行或日志。

`desktop_env/providers/docker/provider.py` 的本轮安全修复包括：

- 删除全局 `pkill qemu` 行为，只记录当前精确容器的 QEMU PID、create time 和
  command；回收前再次校验身份。
- Podman cgroup inspect 失败时，用包含完整 container ID 的 conmon 进程寻找
  后代；伪造进程树测试输出 `conmon-fallback-ok`。
- VM 内固定安装并校验 `fastmcp==2.13.3`，安装前显式卸载 `fastmcp` 与
  `fastmcp-slim`，防止命名空间冲突导致 `fastmcp.tools.tool.Tool` 缺失。

`OSWorld-mcp-run` 仍不是 Git 仓库。其任务相关差异现已按上游基线拆成三段
patch，保存在 `integrations/osworld_mcp/patches/`；既有 TCG/Podman 前置适配
和本轮安全修复分开，应用全系列可字节级重建当前两个外部文件。

## 最终验证

- Windows 同步副本：`python -m pytest tests/ -q` -> `126 passed`
- A800 正式仓库：`/data/songweishuai/envs/evotree/bin/python -m pytest tests/ -q`
  -> `126 passed`
- 精确资源查询正常返回，匹配 `evo-*` 容器数为 0，匹配本轮固定端口特征的
  QEMU 数为 0；未使用批量容器删除或全局进程 kill。

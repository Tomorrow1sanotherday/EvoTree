# A800 运行脚本

这些脚本在 A800 上跑，不在本地跑。之前它们以 `.evo-*.sh` 的隐藏文件形式散在仓库根目录、
不受版本控制、靠手工 `scp` 上服务器且改名——`v3` 那次就是这么漏传的。现在纳入 git，
和代码一起同步。

## 上传

服务器上的落点是 `/data/songweishuai/`，**文件名去掉 `-a800` 后缀**（历史约定，
远端叫 `evo-suite20-v2-20260902.sh` 这种带日期的名字）。上传后必须 `chmod +x`，
否则脚本躺在那儿跑不了。

```bash
scp proj/EvoTree/scripts/evo-suite20-v3-a800.sh A800:/data/songweishuai/evo-suite20-v3-20260903.sh
ssh A800 'chmod +x /data/songweishuai/evo-suite20-v3-20260903.sh'
```

## 当前有效的一套（v3）

v3 相对 v2 的唯一实质改动：**MCP 依赖改走宿主机 wheelhouse 离线装**。v2 就是死在
VM 内联网装 fastmcp+playwright 上——`waiting for deps install...` 刷了 2 小时 43 分，
20 个任务只跑完 5 个。

按顺序：

| 脚本 | 作用 | 前置 |
|---|---|---|
| `evo-wheel-download-a800.sh` | 从阿里云镜像下 59 个 wheel 到 wheelhouse | 只需跑一次，**已完成**（rc=0） |
| `evo-wheel-verify-a800.sh` | 建临时 venv 离线装一遍，验证 wheelhouse 完整 | 打印 `OFFLINE_VERIFY_OK` 即通过 |
| `evo-wheel-server-a800.sh` | 在 18765 起 http.server 供 VM 拉 wheel | **前台常驻**，要 tmux 里跑；suite 全程不能停 |
| `evo-smoke-v3-a800.sh` | 单任务 1 步冒烟，验证离线装依赖真的生效 | 需 wheel server 已起 |
| `evo-suite20-v3-a800.sh` | 20 任务 baseline→online→frozen 三阶段串行 | 需 wheel server 已起 + 冒烟通过 |

wheelhouse 固定在 `/data/songweishuai/evotree_mcp_wheelhouse_py310_20260902`，
VM 通过 `OSWORLD_MCP_WHEEL_URL=http://20.20.20.1:18765/` 访问宿主机
（`20.20.20.1` 是 QEMU slirp 网关，从 VM 里看就是宿主机）。

## 历史版本（保留供追溯，不要再跑）

- `evo-baseline20-a800.sh` / `evo-suite20-a800.sh` — v1，端口冲突作废，
  产物已标记为 `INVALID-*-port-collision`
- `evo-suite20-v2-a800.sh` — v2，卡死在 VM 内联网装依赖，20 任务只完成 5 个
- `evo-ownership-smoke-a800.sh` — 容器归属回收验证，v2 时期已通过

## 安全约定

所有脚本遵守 `doc/服务器使用必读手册.md` 的第一铁律：容器名固定 `evo-` 前缀，
清理只用精确名字 `podman rm -f <确切名>`，绝不用 `podman ps -aq` 批量或差集删除。
`evo-suite20-v3-a800.sh` 的 `require_exact_container_absent` / `check_exact_container_gone`
在 inspect 失败时**中止而非继续**，避免把"查不到"误当成"不存在"。

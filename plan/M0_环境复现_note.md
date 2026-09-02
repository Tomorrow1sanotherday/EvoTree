# M0 环境复现 — 侦察 Note

> 状态：**侦察完成（纯只读，未启动任何评测）**。日期 2026-08-27。
> 结论先行：**M0 几乎不用从头做**——服务器上已有前人配好并跑通的完整环境（`result.txt=1.0` 实锤），还留了一份 16 个坑全记录的部署圣经 `DEPLOY_NOKVM.md`。我们的 M0 = "验证这套现成环境现在还能不能一键出分" + "改造清理脚本避免误伤别人"，而非重搭。

---

## 1. 一张图看懂 OSWorld-MCP 怎么跑

```
宿主机(A800 pod, 无KVM)
  └─ run_multienv_e2e.py  (跑在 osworld venv, 纯CPU)
       └─ DesktopEnv → podman 起容器 → 容器内 QEMU(TCG软件模拟) 跑一台 Ubuntu22.04 桌面VM
            ├─ VM内控制服务器(:5000)  负责截图/执行命令
            ├─ pyautogui模式: 动作=鼠标键盘代码, VM内执行
            └─ mcp模式:      动作=MCP工具调用, VM内MCP server(:9292)执行158/131个工具
       └─ agent 调 LLM(走API: llmapi.evolventapi.com, 不占本地卡) 做决策
```

**三个关键事实（直接消解之前的资源顾虑）：**
1. **跑评测不占 GPU** — 部署文档 §7 原文"纯 CPU/TCG，不占 GPU"。别人占 8 卡在跑训练，与我们跑评测（吃 CPU/内存）互不相干。
2. **LLM 走远端 API** — agent 大脑在 `llmapi.evolventapi.com`，本地一张卡都不用。
3. **慢** — TCG 软件模拟，一个任务含 VM 启动约 5–25 分钟；MCP 模式每次还要在 VM 内装 6 分钟依赖（COW 磁盘不持久）。定位是**冒烟/验证**，真刷榜要用带 KVM 的机器。

---

## 2. 现成资产清单（前人留下的，直接复用）

| 资产 | 路径 | 说明 |
|---|---|---|
| 官方 repo | `/data/songweishuai/projs/rh/OSWorld-MCP` | X-PLUG 正版，含部署圣经 + deploy 脚本 |
| OSWorld 主仓库 | `/data/songweishuai/projs/rh/OSWorld` | MCP 是它的扩展，需覆盖集成 |
| **跑通的工作副本** | `/data/songweishuai/projs/rh/OSWorld-mcp-run` | 已集成+打补丁，`results_mcp` 里有 score=1.0 实证 |
| 部署圣经 | `OSWorld-MCP/DEPLOY_NOKVM.md` | **16个坑全记录**，最宝贵 |
| run 脚本 | `OSWorld-MCP/deploy/run_gui.sh` / `run_mcp.sh` | 一键跑 GUI / MCP 模式 |
| osworld venv | `/data/songweishuai/envs/osworld` | uv管理, py3.12, 依赖齐 |
| VM 磁盘镜像 | `/data/songweishuai/data/osworld/Ubuntu.qcow2` | 24.5G, 已就位 |
| 容器镜像 | `localhost/osworld-patched:latest` | 已在 podman storage |
| podman 环境 | `source /data/songweishuai/container/env.sh` | rootless podman(docker替身) |
| LLM API | `http://llmapi.evolventapi.com/v1`, model `sub2api-gpt-5.5` | 直连,无需代理 |

---

## 3. ⚠️ 危险点：默认清理脚本会误伤别人（必须改造）

部署圣经的 TL;DR 和 `run_mcp.sh` 里都有这两行"清理"：

```bash
for c in $(podman ps -aq); do podman rm -f "$c"; done   # 删掉所有容器
for p in $(pgrep -x qemu); do kill -9 "$p"; done          # 杀掉所有 qemu
```

**问题：这会杀掉机器上别人的东西**——当前有 3 个 qemu 跑了 35 天、一堆 `osworld-pool`/`cua-ppt-libreoffice` 容器起了 3–5 周。前人写脚本时机器上大概只有自己的任务，所以"全清"没事；**现在机器上有别人，全清 = 误伤**。

**改造原则（我们跑时必须遵守）：只清自己起的，用名字/label 隔离，绝不 `podman ps -aq` 全清、绝不 `pkill -x qemu` 全杀。** 具体做法在实际跑之前设计（见第 5 节 TODO）。

---

## 4. 怎么跑一次（改造清理后的标准流程，暂未执行）

```bash
# 0) 起 podman API 守护进程(若没起)
source /data/songweishuai/container/env.sh
setsid podman system service --time=0 unix:///data/songweishuai/container/podman.sock \
    >/data/songweishuai/container/podman_service.log 2>&1 </dev/null & disown

# 1) tmux 里跑(和终端解耦,断连不掉)
tmux new-session -d -s evo-smoke 'bash /data/songweishuai/projs/rh/OSWorld-MCP/deploy/run_mcp.sh 2>&1 | tee /tmp/evo_smoke.log'

# 2) 看进度
tail -f /tmp/evo_smoke.log

# 3) 跑完只清自己的(★ 改造版,不用原脚本的全清)
```

**成功标志**：日志出现 `MCP server is up on :9292`，`traj.jsonl` 里动作是真工具名（如 `os.rename_directory`），`result.txt`=1.0。

**结果位置**：`OSWorld-mcp-run/results_mcp/mcp/screenshot/<model>/0/<domain>/<task_id>/`（含 result.txt 得分、traj.jsonl 轨迹、step_*.png 截图）。

---

## 5. M0 待办（还没做的）

- [ ] **设计"只清自己容器/qemu"的隔离清理**（替换原脚本的全清）——跑之前必做，防误伤别人 35 天的任务。
- [ ] **确认那 3 个 qemu / 占卡任务是谁的**（找 songweishuai/团队），或确认我们跑评测能否共存。
- [ ] **实际跑一次 smoke**（改造清理后 + 确认安全后）：验证现成环境现在还能出 score=1.0。
- [ ] 确认 LLM API key 是否仍有效（脚本里写死了一个 key，可能过期）。
- [ ] 记录一次完整跑的耗时（为后续批量评测估算时间成本）。

---

## 5.5 ⚠️ 环境已漂移（2026-08-27 实测，部署圣经写于 06-19 已对不上）

**教训：这套环境不是"开箱即用"，别人这 2 个月改动过，复现要边修边对齐当前状态。** 实测发现三处漂移：

| 部署圣经说的 | 08-27 实际 | 解法 |
|---|---|---|
| podman service 在 `/data/.../container/podman.sock` | 活着的 service 换到 `/run/podman/podman.sock` | 脚本 DOCKER_HOST 改指向（已改） |
| 那 service 用自定义 storage（含全部镜像） | **活着的 service 用默认空 storage（0 镜像）** | 需起"专属 service"绑自定义 storage（见下） |
| `osworld-patched` 镜像已在 storage | **已被 07-23 清理删除** | 按 `osworld_patch/Dockerfile` 重建（进行中） |

**两套 storage 是核心陷阱：**
- **自定义 storage**（`/data/songweishuai/container/config/storage.conf`，vfs 驱动，graphroot=`/data/songweishuai/container/storage`）：含 11 个受保护容器 + `osworld-pool`/`cua-ppt-libreoffice`/`happysixd`/`ubuntu` 镜像。**我们的一切都应在这套。**
- **默认 storage**：空。当前活着的 service（别人起的）恰恰连的是这套 → 我们连它什么都找不到。
- vfs 驱动无分层缓存 → build 镜像慢（每层全量复制），别指望秒建。

**正确姿势（对齐自定义 storage）：**
```bash
source /data/songweishuai/container/env.sh   # 这步设置 CONTAINERS_STORAGE_CONF 指向自定义storage
# build / 起专属service / 跑评测 全都要在 source 之后,才落在有镜像的那套storage
```

**待确认**：`osworld-patched` 是别人故意删的还是顺手清的？osworld-pool（同源、同目的）能否直接替代？—— 找 songweishuai 对齐可省未来的重复踩坑。

## 5.6 ⚠️⚠️ 今天(08-27)踩的最大坑:两套 storage 分裂(耗一整天的总根源)

**一句话:镜像都在"自定义 storage",但评测脚本默认连的 service 用的是"默认空 storage",所以永远 ImageNotFound。今天绝大部分弯路都源于没一开始看清这个。**

- **自定义 storage** = `/data/songweishuai/container/storage`(env.sh 的 `CONTAINERS_STORAGE_CONF` 指定)。**所有镜像在这**:osworld-patched / osworld-pool / cua / ubuntu。
- **默认 storage** = `/root/.local/share/containers`。**空的**。别人跑着的 `/run/podman` service 恰恰连这套。
- **陷阱**:`source env.sh` 后用 podman CLI `images` 看得到镜像(自定义)，让人误以为"镜像在、service 也连对了"——但评测用 docker-py 走 service，连的是默认空 storage，看不到。**我因此误判"镜像被删"、去重建、又怀疑 cgroup、绕了一整天。**

**正解(已验证跑通到起VM):**
1. 起一个**绑定自定义 storage 的专属 podman service**:先 `source env.sh`(继承 `CONTAINERS_STORAGE_CONF`)，再 `setsid podman system service --time=0 unix://<自己的sock>`。脚本见 `.smoke_guard/start_evo_svc.sh`。
2. 验证连对:用 **docker-py**(不是CLI)走这个 sock 列镜像,能看到 osworld-patched = 连对了。
3. 评测脚本 `DOCKER_HOST` 指向这个 sock。

**另一个假象:cgroup 500 "could not find any cgroup in /proc/<pid>/cgroup"**——不是 cgroup 系统坏了(pod 的 cgroup 完全正常)。真相是:**被 inspect 的容器进程已死**(PID 不存在了),podman 读死 PID 的 cgroup 就报这个错。裸 `podman run osworld-patched` 会**秒退**(exit 64: "No boot disk specified")因为没挂 qcow2/没传 BOOT——评测脚本会正确挂载,不会秒退,所以评测流程里不触发此错。**别再用裸 podman run 测这个镜像。**

## 5.7 ⚠️ 严重事故记录(08-27):误删别人容器

**经过**:早期用的 smoke 脚本(`run_smoke_safe.sh`)靠"拍快照→删掉不在快照里的容器"做清理。某次重拍快照时,用某个 service(空 storage)列容器返回 0 → 保护名单为空 → 清理逻辑把它能看到的容器**全当成"我起的"删掉** → 误删了若干他人容器,其中含运行中的任务。

**根因**:(1) 差集清除依赖了不可靠的数据源(会返回空的service);(2) 没有"快照为空则拒绝删除"的保险。

**教训(硬规则,写死)**:
- **清理绝不用"删掉不在名单里的"这种差集逻辑**——数据源一旦返回空就是灾难。
- **只用"精确名字删除"**:给自己的容器固定前缀 `evo-`(已改 provider.py 的 `containers.run` 加 `name=os.getenv("EVO_CONTAINER_NAME","evo-osworld")`),清理只 `podman rm -f evo-osworld`,物理上不可能碰别人的。
- 新脚本 `run_smoke_v2.sh` 已用此法。旧的 `run_smoke_safe.sh` **已废弃,勿用**。

## 5.8 坑:VNC端口5900被别人的x11vnc占用(端口漂移,已解)

**现象**:QEMU启动报 `-display vnc=:0: Failed to find an available port: Address already in use` → 容器秒退。
**根因**:provider用 `DISPLAY=vnc` + `network_mode=host`,qemu-docker框架的 `display.sh` 硬编码 `-vnc :0`(=端口5900);而宿主机5900被别人的 `x11vnc`(PID随时变)占着。部署圣经6-18跑通时5900是空的,现在漂移被占。**x11vnc是别人的,不能杀。**
**解法(只加环境变量,不碰别人/不改镜像)**:qemu-docker的 `config.sh` 把环境变量 `ARGUMENTS` 透传到QEMU参数串**末尾**,后出现的 `-vnc` 覆盖前面的。所以在provider的 `environment.update` 里加:
```python
"ARGUMENTS": "-vnc :20",   # =5920,避开被占的5900;末尾覆盖display.sh的 -vnc :0
```
(已加在provider.py podman分支,DISPLAY后一行。5920先确认空闲。)
**验证**:重跑后日志不再有"Address already in use",容器不再秒退,进入稳定的"Checking if virtual machine is ready"(VM引导中)。

---

## 5.9 ✅ 08-27 最终:打通到 VM 完全就绪+MCP注入(离出分只差agent交互)

**今天从"完全跑不起来"一路打通,过关记录(按顺序):**
1. ✅ socket路径漂移 → DOCKER_HOST 指向对的 service
2. ✅ 两套storage分裂 → 起绑自定义storage的专属service(evo-podman.sock, docker-py验证能看到osworld-patched)
3. ✅ 镜像被删 → 按Dockerfile重build osworld-patched
4. ✅ cgroup 500假象 → 认清是"死进程PID"非cgroup坏,别裸run(缺BOOT秒退)
5. ✅ 容器误删事故 → 改用evo-前缀精确命名+精确删除(见5.7)
6. ✅ VNC :0(5900)被别人x11vnc占 → 改镜像display.sh的vnc=:0→:20(见5.8)
7. ✅ VM引导成功 → QEMU稳定存活>400s, 5000端口HTTP 200
8. ✅ 黑屏坑④ → provider的_apply_softpipe_fix()自动检测黑屏(mean=0.007)并切softpipe,桌面渲染成功(mean=34→62,非黑)
9. ⏳ MCP依赖注入 → 正在VM内装fastmcp+playwright(slirp慢~6min),装完起MCP server(:9292)→agent用工具→出分

**当前状态**:VM完全就绪、桌面可截图、MCP依赖安装中。**这是今天最大成果——环境实质打通了**,剩下就是等agent交互出分。

**复跑要点(下次直接用)**:
- 起专属service: `bash .smoke_guard/start_evo_svc.sh`(绑自定义storage)
- 跑评测: `bash .smoke_guard/run_smoke_v2.sh`(DOCKER_HOST=evo-podman.sock, 容器名evo-osworld, 清理只删这一个名)
- 诊断版(容器退出不删,保留现场): `run_smoke_diag.sh`
- **关键前提**:osworld-patched镜像已含VNC :20补丁(413ca9c0c5d);provider.py已加name=evo-osworld
- **慢**:VM引导~5min + 黑屏softpipe修复~2min + MCP依赖~6min = 单任务启动约13min+,批量评测这台机器不现实(见§6)

---

## 6. 对 EvoTree 的启示（M0 之后的路怎么走）

1. **环境这关基本免费过了** — 省下 idea §15 预估的 1–2 周环境复现。这是意外之喜。
2. **但"慢"是硬约束** — 一个任务 5–25 分钟 + MCP 每次装 6 分钟。EvoTree 要跑 Setting I 三轮 × 多任务，**这台无 KVM 机器上批量评测会非常慢**。M4 全量前需要么优化（把依赖烤进镜像免每次重装，见部署圣经 §6）、么找带 KVM 的机器。
3. **EvoTree 接入点清晰** — 我们的记忆系统是 agent 侧的 wrapper，接在 `run_multienv_e2e.py` 的 agent 决策环节（`mm_agents/`）。不用碰 VM/容器/评测底座那套，只在 agent"每步决策前查记忆、每轮结束后写记忆"处切入。这对我们是好消息：**底座是别人的，我们只加 agent 的大脑**。

---
name: remote-cann-development
description: >
  Unified remote NPU development — sync, exec, and test across multiple NPU backends
  (docker containers, hdspace cloud) via a single Python CLI. Use when:
  (1) syncing code to any remote NPU server
  (2) executing build/test commands on remote NPU
  (3) probing NPU platform info
  (4) managing multiple remote targets (A2/910B + A3/910C)
  (5) setting up hdspace tunnel or Docker SSH connections
  (6) troubleshooting remote compile/run issues (507033, ctypes, rsync)
---

# Remote NPU Development

Unified Python CLI (`npu.py`) for remote NPU development. Supports docker containers (SSH + docker exec) and hdspace cloud containers (tunnel + SSH) through a single `.npus.yaml` config.

## Prerequisites

**本地**：
- [uv](https://docs.astral.sh/uv/) — 脚本使用 PEP 723 inline metadata，`uv run` 自动安装依赖
- SSH config 已配置远程 host（docker 后端）或 hdspace Include（hdspace 后端）

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # 安装 uv
```

**远程**：需要 rsync（sync 命令依赖）

```bash
sudo apt-get install -y rsync   # hdspace（developer 有 sudo）
apt-get install -y rsync        # Docker（root 用户）
```

## Setup

### 1. 创建 `.npus.yaml`

```bash
cp ${CLAUDE_SKILL_DIR}/scripts/npus.example.yaml .npus.yaml
```

Add `.npus.yaml` to `.git/info/exclude`.

### 2. Docker 后端配置

需要 SSH config 中有对应 host alias。详细的 SSH config 配置（直连/跳板机/Docker）见 [references/docker-setup.md](references/docker-setup.md)。

```yaml
# .npus.yaml
remotes:
  a2:
    backend: docker
    host: <ssh-alias>          # ~/.ssh/config 中的 Host
    container: <container>     # docker container name
    workdir: /workspace        # 容器内父目录；实际仓库路径自动追加为 /workspace/<repo>
    soc: ascend910b
    device_isolate: 7          # ASCEND_RT_VISIBLE_DEVICES（共享服务器）
```

### 3. hdspace 后端配置

华为开发者空间容器，通过 hdspace CLI 隧道连接。

**首次配置**：见 [references/hdspace-setup.md](references/hdspace-setup.md) 完整步骤（CLI 安装、Access Key ID / Secret Access Key、SSH Include、容器环境、login shell 陷阱）。

摘要：
```bash
hdspace config                  # 输入 Access Key ID / Secret Access Key
# ~/.ssh/config 顶部加：Include ~/.devenv/.ssh/config
```

```yaml
# .npus.yaml
remotes:
  a3:
    backend: hdspace
    name: <container-name>     # hdspace devenv list → NAME 列
    workdir: /mnt/workspace    # 持久化路径
    soc: ascend910_93
    login_shell: true          # 必须！否则 asc_opc 编译失败
```

使用前启动隧道（保持终端运行）：

```bash
hdspace devenv start-tunnel --name=<NAME> --ports=10022:22
```

### 4. 验证

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py list
```

## Commands

```bash
NPU="${CLAUDE_SKILL_DIR}/scripts/npu.py"

# List remotes
uv run $NPU list

# Sync
uv run $NPU sync push                     # 全仓推到 default remote
uv run $NPU sync push a3                  # 推到指定 remote
uv run $NPU sync push a2 cmake/           # 推特定路径（绕过 gitignore）
uv run $NPU sync pull a2 build_out/       # 从远程拉取
uv run $NPU sync diff a3                  # 双向比较（+新增 ~修改 -远程多余）
uv run $NPU sync clean a2 old/ --force    # 删除远程路径

# Execute
uv run $NPU exec a2 info                  # 探测 NPU
uv run $NPU exec a2 "bash build.sh --pkg --ops=erf_inv --experimental --soc=ascend910b -j16"
uv run $NPU exec a3 "bash build.sh --pkg --ops=erf_inv --experimental --soc=ascend910_93 -j16"
```

## Typical Workflow

| Step | Location | Command |
|------|----------|---------|
| 1. Edit code | Local | Claude Read/Write/Edit tools |
| 2. Push to remote | Local | `uv run $NPU sync push <target>` |
| 3. Build | Remote | `uv run $NPU exec <target> "bash build.sh ..."` |
| 4. Test / Run | Remote | `uv run $NPU exec <target> "./test_binary"` |
| 5. Pull results | Local | `uv run $NPU sync pull <target> build_out/` |
| 6. Analyze | Local | Read output files locally |

## Remote File Exploration

Claude 的 Read/Glob/Grep 只能操作本地文件。探索远程文件（CANN SDK headers、profiling 数据、日志）通过 exec：

```bash
uv run $NPU exec a2 "ls -la \$ASCEND_HOME_PATH/include/"
uv run $NPU exec a2 "cat /path/to/remote/file.h"
uv run $NPU exec a3 "grep -r 'pattern' /path/to/search/"
```

## Config reference

```yaml
remotes:
  <name>:
    backend: docker | hdspace
    # docker 后端
    host: <ssh-alias>            # SSH config 中的 Host
    container: <name>            # Docker 容器名
    # hdspace 后端
    name: <container-name>       # hdspace devenv list → NAME
    # 通用
    workdir: /path               # 远程工作目录
    soc: ascend910b              # build.sh --soc
    cann: /path/to/cann          # CANN 路径（可选，自动检测）
    device_isolate: 7            # ASCEND_RT_VISIBLE_DEVICES（可选）
    login_shell: true            # 用 bash -l（hdspace 必须，docker 可选）
    port: 10022                  # SSH 端口（可选，覆盖 ~/.devenv/.ssh/config）

default: <name>                  # 不指定 target 时使用
```

## Backend differences

| | docker | hdspace |
|---|---|---|
| 连接 | SSH → docker exec | hdspace 隧道 → SSH |
| Shell | bash | bash -l（login shell 必须，加载 /etc/profile） |
| 用户 | 通常 root | developer（有 sudo） |
| 持久化 | Docker volume | /mnt/workspace（196GB） |
| rsync | 两步（host staging → docker cp） | 直连 |
| CANN 路径 | /usr/local/Ascend/cann-*/ | /home/developer/Ascend/cann-*/ |
| NPU 设备隔离 | ASCEND_RT_VISIBLE_DEVICES | 通常独占，不需要 |
| 非交互 shell 陷阱 | 无 | 不用 login shell → buildtools Python 不在 PATH → asc_opc 失败 |

## Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| hdspace `kex_exchange_identification` | 隧道未启动或 SSH config 未 Include | `hdspace devenv start-tunnel` + `Include ~/.devenv/.ssh/config` |
| hdspace `Connection refused` | 隧道端口与 config 不匹配 | `lsof -i \| grep hdspace` 查实际端口，手动改 `~/.devenv/.ssh/config` |
| ascend950 `Invalid socVersion` | CANN 8.5 不认识 ascend950 | 在 CANN 9.0+ 环境编译（A3/hdspace） |
| hdspace SSH `Permission denied` | 用了 root 或错误 key | 用户是 developer，key 在 `~/.devenv/.ssh/IdentityFile/` |
| `asc_opc` Python ctypes `_PyErr_SetLocaleString` | 非 login shell，buildtools Python 不在 PATH | 设 `login_shell: true`（已自动处理） |
| 507033 `ACL_ERROR_RT_DEV_PERMISSION_DENIED` | NPU 设备被占用 | 设 `device_isolate` 到空闲 device |
| `cmake/third_party/` 未同步 | `.gitignore` 中 `third_party/` 匹配到 | `sync push <target> cmake/third_party/` 手动推 |
| rsync `command not found` | 远程未装 rsync | `sudo apt-get install -y rsync` |
| 编译产物在 NPU 上输出垃圾值 | soc 版本不匹配（如 910B 硬件用了 ascend910_93） | 确认 `.npus.yaml` 中 soc 与硬件一致 |
| hdspace 容器重启后 CANN 丢失 | /home/developer 不持久化 | 代码和产物放 /mnt/workspace/ |

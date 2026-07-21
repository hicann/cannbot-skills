# 华为开发者空间（hdspace）完整配置指南

## 概述

华为开发者空间提供免费的 NPU 云容器（910C 等），通过 `hdspace` CLI 建立 SSH 隧道连接。

- 容器管理：https://developer.huaweicloud.com/space/devportal/platform/devEnvironment?tab=container
- CLI 下载：同上页面（点击"远程连接"按钮 → 客户端远程连接）

## 1. 安装 hdspace CLI

从开发者空间页面下载对应平台的 CLI，放入 PATH：

```bash
# 验证
hdspace version
# hdspace CLI Version 2.3.0
```

## 2. 配置 Access Key ID / Secret Access Key

```bash
hdspace config
# 输入 Access Key ID 和 Secret Access Key
# 获取地址：https://console.huaweicloud.com/iam/?#/mine/accessKey
```

## 3. SSH Include 配置

hdspace 的 SSH 配置自动维护在 `~/.devenv/.ssh/config`，包含：
- Host 名（格式：`{name}.{instance_id}.0`）
- 端口（每次 start-tunnel 可能变化）
- IdentityFile（自动生成的密钥）
- 用户（`developer`，非 root）

需要 include 到主 SSH config 让 ssh/rsync 能找到：

```bash
# 在 ~/.ssh/config 顶部添加（放在 Host * 之前）
Include ~/.devenv/.ssh/config
```

## 4. 启动隧道

```bash
# 查看可用容器
hdspace devenv list
# +----+----------------------------------+-------+-----+
# | NO | ID                               | NAME  | ... |
# +----+----------------------------------+-------+-----+
# | 1  | c7310c9af9aa...                  | DevHW | ... |
# +----+----------------------------------+-------+-----+

# 启动隧道（保持终端运行）
hdspace devenv start-tunnel --name=DevHW --ports=10022:22
```

隧道启动后 `~/.devenv/.ssh/config` **应该**自动更新端口。之后可直接 SSH：

```bash
ssh devhw.c7310c9af9aa444c8d8a5b3bcd727277.0
# developer@...：~$
```

### 隧道管理

```bash
# 关闭隧道
pkill -f "hdspace devenv start-tunnel"

# 检查隧道是否运行
pgrep -f "hdspace devenv start-tunnel"

# 检查实际监听端口
lsof -i :10022   # 或你指定的本地端口
```

### 端口不匹配排查

如果 `Connection refused`：
1. 检查实际监听端口：`lsof -i | grep hdspace`
2. 对比 `~/.devenv/.ssh/config` 中的 Port
3. 如果不一致，手动修改 config 中的 Port 或用不同 `--ports` 重启隧道
4. ssh-key-reset 后可能需要重建隧道：`echo "yes" | hdspace devenv ssh-key-reset --name=<NAME>`

## 5. 容器环境信息

### 典型配置

| 项 | 值 |
|---|---|
| NPU | Ascend910C (IT22HMDA_4_S)，1-2 芯片 |
| CANN | 9.0.0，`/home/developer/Ascend/cann-9.0.0/` |
| 编译目标 | `--soc=ascend910_93` |
| 用户 | `developer`（非 root，有 sudo） |
| 持久化路径 | `/mnt/workspace/`（196GB，重启后保留） |
| 非持久化 | `/home/developer/`（overlay，重启后丢失） |
| Python | 3.12，buildtools 在 `/opt/buildtools/python-3.12.9/` |

### 关键路径

```
/mnt/workspace/              # 持久化：代码、编译产物放这里
/home/developer/             # 非持久化：CANN 在这，重启后恢复
/home/developer/Ascend/cann-9.0.0/  # CANN SDK
/usr/local/Ascend/driver/    # NPU driver（系统级）
/opt/buildtools/python-3.12.9/  # CANN 编译工具链 Python
```

## 6. Login Shell 陷阱

容器的 `~/.bashrc` 有非交互式 guard：

```bash
case $- in
    *i*) ;;
      *) return;;
esac
```

非交互式 `ssh remote 'cmd'` 跳过 `.bashrc`，导致 `/etc/profile` 不加载。而 `/etc/profile` 包含 buildtools Python 的 PATH 设置：

```bash
# /etc/profile
export PATH=/opt/buildtools/python-3.12.9/bin:$PATH
```

没有这个 PATH，`asc_opc`（kernel binary 编译器）调用 Python 时找到的是系统 Python，它的 `_ctypes.so` 和 buildtools 的 Python 二进制 ABI 不匹配，报 `_PyErr_SetLocaleString undefined`。

**解决**：在 `.npus.yaml` 中设 `login_shell: true`，脚本自动用 `bash -l` 执行。

## 7. 容器开关机

免费容器有机时限制，用完后应关机节约机时避免被封。

```bash
hdspace devenv stop --name=DevHW      # 关机
hdspace devenv start --name=DevHW     # 开机
hdspace devenv list                   # 查看状态（Running/Ready/Stopping）
```

> 关机后 `/mnt/workspace/` 数据保留，`/home/developer/` 重置。

## 8. 申请 NPU 容器

在 [GitCode CANN 组织](https://gitcode.com/cann) 下任意仓库（如 [ops-math](https://gitcode.com/cann/ops-math)）点击"云开发"按钮，即可免费获得 NPU 容器（910C）。

## 9. 远程安装 rsync

容器默认不带 rsync，首次使用 sync 命令前需安装：

```bash
ssh <host> "sudo apt-get install -y rsync"
```

## 10. 代码同步方式

### 方式 1：rsync（推荐，通过 npu.py）

```bash
uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py sync push a3       # 全仓推
uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py sync pull a3 build_out/  # 拉取产物
```

### 方式 2：bare git repo（无需 rsync）

容器没有私仓 SSH key，通过本地 bare repo 中转：

```bash
# 一次性
ssh <host> "git init --bare /mnt/workspace/repo.git"
git remote add devhw <host>:/mnt/workspace/repo.git

# 推送
git push devhw <branch>

# 远程 clone
ssh <host> "cd /mnt/workspace && git clone /mnt/workspace/repo.git"
# 后续：git push devhw + ssh <host> "cd /mnt/workspace/repo && git pull"
```

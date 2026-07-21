# Docker 后端配置指南

## SSH Config 配置

在 `~/.ssh/config` 中为远程服务器创建 alias，`.npus.yaml` 的 `host` 字段引用此 alias。

### 直连

```
Host npu
    HostName <server-ip>
    User root
    RequestTTY no
```

### 通过跳板机

```
Host npu
    HostName <server-ip>
    User root
    ProxyJump <jump-user>@<jump-host>
    RequestTTY no
```

**关键点**：
- `RequestTTY no` — 非交互式 SSH 命令必须
- `ProxyJump` — 多跳 SSH 透明处理
- **不要** 用 `RemoteCommand` 处理 Docker — 在 `.npus.yaml` 中设 `container` 字段

### 验证连接

```bash
# 1. SSH 连通性
ssh npu "echo hello"

# 2. Docker 容器存在
ssh npu "docker ps --filter name=<container>"

# 3. NPU 可用
uv run ${CLAUDE_SKILL_DIR}/scripts/npu.py exec <target> info
```

## .npus.yaml Docker 配置

```yaml
remotes:
  a2:
    backend: docker
    host: npu                    # SSH config 中的 Host alias
    container: my-cann-container # Docker 容器名
    workdir: /workspace          # 容器内父目录，实际仓库路径为 /workspace/<repo>
    soc: ascend910b
    device_isolate: 7            # 共享服务器设备隔离
    cann: /usr/local/Ascend/cann-8.5.0  # 可选，自动检测
```

## sync 工作原理（docker 后端）

Docker 容器不能直接 rsync，采用两步方式：

1. **push**：rsync 到 host 的 staging 目录 → `docker cp` 到容器
2. **pull**：`docker cp` 从容器到 host staging → rsync 到本地

npu.py 内部自动处理，用户无感。

## Troubleshooting

| 问题 | 原因 | 解决 |
|------|------|------|
| `Permission denied` | SSH key 未配置 | `ssh-copy-id npu` |
| `Connection timed out` | 跳板机不通 | 检查 VPN、`ProxyJump` 配置 |
| `command not found: npu-smi` | 容器内无 CANN | 检查容器 CANN 安装 |
| `rsync: command not found` | 远程未装 rsync | `apt-get install -y rsync` |
| 507033 device permission | NPU 被占用 | `device_isolate` 换空闲 device |

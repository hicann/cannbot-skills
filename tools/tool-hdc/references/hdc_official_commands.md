# hdc 命令速查

本文件按使用场景整理 hdc 常用命令。命令语法和参数以实际环境 `hdc -h` 及目标设备系统版本为准；`tool-hdc/SKILL.md` 中的使用约束同样适用。

## 全局参数

全局参数放在 `hdc` 与子命令之间。

| 参数 | 作用 | 备注 |
| --- | --- | --- |
| `-t <connect-key>` | 选择目标设备 | 多设备场景必须指定；USB 用设备 SN，TCP 用 `IP:port` |
| `-l <0-6>` | 设置当前运行日志级别 | 默认 3 |
| `-s [IP:]port` | 指定 server 监听地址和端口 | 远程连接场景使用 |
| `-p` | 跳过 server 查询直接执行 | server 未启动时不会自动拉起 |
| `-m` | 前台运行 server | 便于观察服务日志 |
| `-e <ip>` | 指定端口转发时本地监听 IP | 需与 `-m` 配合（API 20+） |

```bash
hdc -t <connect-key> shell echo hello
```

## 设备发现与连接

| 命令 | 用途 |
| --- | --- |
| `hdc list targets` | 列出已连接设备；无设备时返回 `[Empty]` |
| `hdc list targets -v` | 显示连接方式、状态和设备名 |
| `hdc wait [-t key]` | 阻塞等待设备连接就绪 |
| `hdc tconn IP:port` | 通过 TCP 连接设备 |
| `hdc tconn IP:port -remove` | 断开 TCP 连接 |
| `hdc tmode port [port]` | 打开设备网络调试通道；执行后守护进程会重启，需要重连 |
| `hdc tmode port close` | 关闭设备网络调试通道 |
| `hdc reconnect <connect-key>` | 重置 USB 会话并重新枚举设备（API 26+） |

设备状态说明：

- `Ready` / `Connected`：设备可用。
- `Unauthorized`：设备侧尚未确认调试授权，需要在设备上点击信任。
- `Offline`：设备已断开或服务异常，先检查连接再执行 `hdc kill -r`。

## shell 与设备端工具

```bash
hdc shell [command]
hdc shell -b <bundle> [command]
```

- 不传 `command` 时进入交互式 shell。
- `-b <bundle>` 用于在可调试应用的数据目录内执行命令（API 15+）；应用需使用调试证书签名并已启动。
- 设备端命令由 toybox 提供，可用 `hdc shell help -a` 查看完整列表。

常用设备端工具：

| 工具 | 用途 | 示例 |
| --- | --- | --- |
| `aa` | 启动/停止 Ability | `hdc shell aa start -a <ability> -b <bundle>` |
| `bm` | 查询和管理应用包 | `hdc shell bm dump -a` |
| `hilog` | 查看设备日志 | `hdc shell hilog` |
| `param` | 读取/设置系统参数 | `hdc shell param get <name>` |
| `hidumper` | 导出系统信息 | `hdc shell hidumper -c base` |
| `hiperf` / `hitrace` | 性能采样与打点 | `hdc shell hiperf -h` |

> `aa force-stop`、`rm`、`uninstall`、`target boot` 等属于破坏性操作，执行前必须让用户确认。

## 应用安装与卸载

```bash
hdc install [-r] <path.hap>
hdc install -s <path.hsp>
hdc uninstall [-k] <bundle>
```

| 参数 | 说明 |
| --- | --- |
| `-r` | 覆盖安装 |
| `-s` | 安装 HSP 时指定文件 |
| `-p` | 指定多 HAP/HSP 所在目录（API 22+ 也支持 APP） |
| `-w <seconds>` | 设置 bm 等待时长，常用 180~600 |
| `-g` | 调试包自动授予 user_grant 权限（API 24+） |
| `-k` | 卸载时保留应用数据 |

安装成功后可看到类似 `install bundle successfully` 的提示；卸载成功后可看到 `uninstall bundle successfully`。自动化脚本仍应以返回码和实际状态为准。

## 文件传输

```bash
hdc file send <local> <remote>
hdc file recv <remote> <local>
```

| 参数 | 说明 |
| --- | --- |
| `-a` | 保留文件修改时间 |
| `-sync` | 仅传输 mtime 更新的文件 |
| `-m` | 同步权限和属主信息 |
| `-cwd <path>` | 指定源路径工作目录 |
| `-b <bundle>` | 直接传输到可调试应用沙箱（API 15+） |

注意：

- `file recv` 的参数顺序是“远端在前、本地在后”，与 `file send` 相反。
- `-z` 压缩选项未开放，不要使用。
- 目标目录不存在或无写权限时可能报 `E005005`，先用 `hdc shell mkdir -p` 建目录。
- 传输完成后建议校验两端字节数，不要只看返回码。

```bash
hdc file send x1.bin /data/local/tmp/board_test/
hdc file recv /data/local/tmp/board_test/output0.bin ./
```

## 端口转发

```bash
hdc fport ls
hdc fport tcp:1234 tcp:1080
hdc rport tcp:1234 tcp:1080
hdc fport rm tcp:1234 tcp:1080
```

- `fport`：主机端口转发到设备端口。
- `rport`：设备端口反向转发到主机。
- 主机侧目前只支持 `tcp`；设备侧还支持 `dev`、`localabstract` 等类型。
- 本地端口被占用时会报 `TCP Port listen failed`，换端口后重试。

## 服务进程与版本

```bash
hdc start [-r]       # 启动 server；-r 表示重启
hdc kill [-r]        # 停止 server；-r 表示停止后重启
hdc checkserver      # 查看 client/server 版本是否一致
hdc version          # 查看 hdc 版本
hdc jpid             # 查看设备端已打开应用的 pid
```

遇到命令卡住、设备失联或 server 状态异常时，先执行 `hdc kill -r`，再重新连接设备。

## 日志与调试

```bash
hdc hilog
hdc shell hilog > local.log
hdc -l 5 shell echo hello
hdc bugreport
```

- `hdc hilog` 打印设备日志；保存到本地时用重定向。
- 临时提高日志级别可用 `-l`。
- `bugreport` 用于导出系统信息，文件可能较大，按需使用。

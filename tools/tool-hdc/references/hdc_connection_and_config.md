# hdc 连接方式与环境配置

本文件说明 hdc 的常见连接方式、环境变量和版本特性。具体命令语法以实际环境 `hdc -h` 为准。

## 连接方式选择

| 场景 | 适用条件 | 推荐做法 |
| --- | --- | --- |
| USB | 设备与开发机直接连线 | 最稳定的方式，插线后用 `hdc list targets` 确认 |
| TCP | 设备和开发机在同一网络 | 在设备上打开无线调试，记录 `IP:port`，执行 `hdc tconn` |
| 远程 | 设备插在另一台电脑上 | 在设备所在电脑启动 server，开发机通过 `-s` 连接 |

## USB 连接检查

按下面四项逐一确认：

| 检查项 | 正常表现 | 异常处理 |
| --- | --- | --- |
| USB 调试开关 | 开发者选项中已打开 | 打开开关或重启设备 |
| 数据线 | 支持数据传输 | 更换原装或带数据功能的数据线 |
| USB 接口 | 主板直出接口稳定识别 | 换接口，避免使用扩展坞或前置面板 |
| 驱动 | 设备管理器或 `lsusb` 中出现 HDC 设备 | 重新安装驱动或换电脑验证 |

```bash
hdc list targets
```

返回设备 SN 表示连接成功；返回 `[Empty]` 表示系统未识别到设备。

## TCP 连接

> TCP 调试适合测试环境，不建议作为生产环境的唯一连接方式。连接数量有限，超限会阻塞。

1. 在设备上打开“无线调试”，记录显示的 `IP:port`。
2. 确认开发机可以访问该地址。
3. 连接并验证：

```bash
<device-ip>:<port>
hdc list targets
<device-ip>:<port> -remove   # 断开
```

如果需要打开设备网络调试通道：

```bash
hdc tmode port 65512
hdc tmode port close
```

执行 `tmode` 后设备端守护进程会重启，需要重新连接。

## 远程连接

远程场景下，设备插在服务器上，开发机通过网络访问服务器的 hdc server。

```bash
# 服务器侧（设备所在机器）
hdc kill
hdc -s <server-ip>:8710 -m

# 开发机侧
hdc -s <server-ip>:8710 list targets
hdc -s <server-ip>:8710 shell echo hello
```

注意事项：

- `-s` 只对当前命令生效。
- 不写 IP 时默认监听回环地址。
- 显式指定 `-s` 后，`OHOS_HDC_SERVER_PORT` 不再生效。
- 监听在非回环地址时要评估网络安全。
- 连接失败常见提示：`Connect server failed.`（server 不可达）、`-s content port incorrect.`（端口非法）。

## 环境变量

以下变量用于调整 server 行为，修改后需要重开命令行窗口：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OHOS_HDC_SERVER_PORT` | `8710` | server 监听端口，范围 1~65535 |
| `OHOS_HDC_LOG_LEVEL` | `3` | server 日志级别；与 `-l` 同时设置时该变量优先 |
| `OHOS_HDC_HEARTBEAT` | 开启 | 设为 1 可关闭心跳 |
| `OHOS_HDC_CMD_RECORD` | 关闭 | 设为 1 记录命令（API 20+），日志在系统临时目录下 |
| `OHOS_HDC_ENCRYPT_CHANNEL` | 关闭 | 设为 1 加密 server 与 daemon 的 TCP 通道（API 20+） |
| `OHOS_HDC_SUBSERVER_LOG_FILE` | 关闭 | 设置子服务日志保留数量（API 26+） |

## 版本与特性

不同 hdc 版本支持的能力不同：

| hdc 版本 | API 版本 | 主要特性 |
| --- | --- | --- |
| 3.1.0a | 12 | `wait -t`、中文文件名参数 |
| 3.1.0e | 15+ | `file send/recv -b`、`shell -b` |
| 3.2.0b | 20 | 端口转发 `-e`、命令录制、通道加密 |
| 3.2.0e | 26+ | `reconnect`、`spawn-sub/killall-sub`、`shell -b` 交互模式 |

低版本使用高版本特性时可能报 `Unknown file option` 或 `E005004`，确认版本后再决定是否升级 SDK。

```bash
hdc version
hdc shell hdcd -v
hdc checkserver
```

## 工具获取与 PATH

hdc 随 HarmonyOS SDK 提供，常见位置：

1. DevEco Studio 内置 SDK：`{DevEco Studio}/sdk/default/openharmony/toolchains/`
2. Command Line Tools：`{CLT}/sdk/default/openharmony/toolchains/`

将 toolchains 目录加入 PATH：

```bash
export PATH="{DevEco Studio}/sdk/default/openharmony/toolchains:$PATH"
```

如果要把 hdc 单独拷出使用，需要同时拷贝同目录下的 `libusb_shared` 依赖。

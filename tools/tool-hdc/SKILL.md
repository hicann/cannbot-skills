---
name: tool-hdc
description: "hdc（HarmonyOS Device Connector）命令行：设备连接、HAP 安装/卸载、文件传输、shell、日志与错误码排查。触发场景：用户提到 hdc、设备连接/调试、向麒麟设备推拉文件、安装 HAP、抓取 hilog、执行设备端 shell 时使用。"
---

# hdc 设备调试工具

hdc（HarmonyOS Device Connector）是 HarmonyOS 官方命令行调试工具，用于与设备交互调试、数据传输、日志查看与应用安装，支持 Windows/Linux/MacOS。典型形态：开发机（服务器/PC）通过 USB 或 TCP 直接连接麒麟设备。

> **目录定位**：hdc 属 HarmonyOS SDK 工具（非 CANN 工具），直接服务算子开发工作流（离线模型设备部署验证的设备侧操作），位于 `tools/` 下（与 asys 等 CANN 维测工具并列，定位不同）。

## 架构

```
电脑端                          设备端
┌─────────────┐   ┌─────────────┐      ┌──────────────┐
│ hdc client  │──▶│ hdc server  │─USB/TCP─▶│ hdcd daemon │
│ (命令即启退) │   │ (后台常驻,   │      │ (响应请求)    │
└─────────────┘   │  默认监听8710)│      └──────────────┘
                  └─────────────┘
```

## 工作流程

1. **解析场景**：判断用户意图属于哪类（连接管理 / 文件传输 / 应用管理 / 日志抓取 / shell 执行 / 报错排查）
2. **查文档**：从 [references/](references/) 取对应命令的官方用法与参数
3. **生成命令**：输出一条或多条 hdc 命令，每条附用途说明与关键参数解释
4. **真实环境验证**：
   - 先执行 `hdc list targets` 确认有可用设备（[脚本](scripts/find_hdc.sh)可定位 hdc 并自检）
   - 有设备时实际执行命令，以实际回显作为验证结果
   - **破坏性命令（删除、卸载、重启、强停类）只生成、不执行**，并提示用户确认

## 输出要求

每条命令按以下格式给出：

```bash
hdc <命令>          # 用途说明；关键参数解释
```

- 验证通过的命令附真实环境执行回显
- 未经真实环境验证的命令明确标注"未验证，来源：官方文档"

## 命令速查（场景 → 命令）

| 场景 | 命令 | 详见 |
|------|------|------|
| 查询已连接设备 | `hdc list targets [-v]` | [hdc_official_commands.md](references/hdc_official_commands.md) |
| 多设备指定某台执行 | `hdc -t <connect-key> <command>` | 同上 |
| 无线连接/断开设备 | `hdc tconn IP:port [-remove]` | [hdc_connection_and_config.md](references/hdc_connection_and_config.md) |
| 安装/覆盖安装 HAP | `hdc install [-r] <path.hap>` | [hdc_official_commands.md](references/hdc_official_commands.md) |
| 卸载应用（破坏性） | `hdc uninstall <bundle>` | 同上 |
| 本地 → 设备推文件 | `hdc file send <local> <remote>` | 同上 |
| 设备 → 本地拉文件 | `hdc file recv <remote> <local>` | 同上 |
| 抓取设备日志到本地 | `hdc shell hilog > local.log` 或 `hdc file recv` | 同上 |
| 设备端执行命令 | `hdc shell [-b <bundle>] [command]` | 同上 |
| 查询应用 bundle 名 | `hdc shell bm dump -a` | 同上 |
| 端口转发 | `hdc fport / rport / fport ls / fport rm` | 同上 |
| 服务异常恢复 | `hdc kill -r` | [hdc_error_codes_and_faq.md](references/hdc_error_codes_and_faq.md) |
| 命令报错误码 | 按码查表 | [hdc_error_codes_and_faq.md](references/hdc_error_codes_and_faq.md) |
| 文件传输与路径处理避坑（引号静默失败/路径斜杠/双侧校验/异步轮询） | 按专题文档操作 | [hdc_transfer_and_path_pitfalls.md](references/hdc_transfer_and_path_pitfalls.md) |

## 使用前必读的六条约束

1. **返回信息不可用于脚本逻辑判断**：错误文本仅供人读，自动化脚本应使用标准错误码判断（见 error-codes 文档）
2. **多设备必须带 `-t`**：两台及以上设备连接时，所有命令必须 `hdc -t <connect-key>` 指定目标
3. **`file recv` 参数顺序是远端在前、本地在后**（与 `file send` 相反），写反会报路径非法
4. **TCP 调试仅建议测试环境使用**，最多 16 个连接，超限阻塞
5. **脚本化调用时传输可能静默失败**：命令经脚本/多层 shell 拼接时，路径引号与转义可能被中间层吞掉 → rc=0 但文件未传输；路径不加引号（无空格时），Windows 路径用全反斜杠绝对路径（见 [hdc_transfer_and_path_pitfalls.md](references/hdc_transfer_and_path_pitfalls.md)）
6. **传输必双侧字节校验**：send 后查设备侧字节数、recv 后查本地侧字节数（=元素数×dtype 宽），仅信 rc=0 不可靠；`aa start` 返回成功 ≠ 推理已执行，以输出文件生成与字节数为准

## 参考文档索引

| 文档 | 内容 | 来源 |
|------|------|------|
| [hdc_official_commands.md](references/hdc_official_commands.md) | 常用命令速查：全局参数、连接管理、shell 与配套工具、install/uninstall、file send/recv、fport/rport、服务进程管理 | hdc 命令说明整理 |
| [hdc_connection_and_config.md](references/hdc_connection_and_config.md) | USB/TCP/远程连接场景、环境变量配置、版本特性 | hdc 命令说明整理 |
| [hdc_error_codes_and_faq.md](references/hdc_error_codes_and_faq.md) | 常见错误码速查与故障排查 | hdc 命令说明整理 |
| [hdc_examples.md](references/hdc_examples.md) | 场景示例与端侧部署命令序列（源自历史开发记录，已脱敏） | 工程实践整理 |
| [hdc_transfer_and_path_pitfalls.md](references/hdc_transfer_and_path_pitfalls.md) | 文件传输与路径处理实战避坑：引号静默失败、路径斜杠规范、双侧字节校验、异步执行轮询、cmd 语法差异 | 工程实战沉淀 |

## 脚本工具

| 脚本 | 用途 |
|------|------|
| `scripts/find_hdc.sh` | 定位 hdc 可执行文件（PATH/常见 SDK 路径），自检版本并执行 `hdc list targets` 确认设备 |

## 与其他 Skill 的关系

- `tool-atc`：模型/单算子转换为离线模型（om）后，推送到设备执行、结果拉回等环节使用本 skill
- 端侧部署类 skill（如 model/ 领域的端侧推理技能）：设备侧操作（安装、传输、日志）引用本 skill，不重复维护命令细节
- 回传输出与 golden 的精度比对由测试/精度调试类 skill（如 `ascendc-precision-debug`）承接，本 skill 只负责到"输出回传 + 字节数校验"
- 设备运行时拒载定性（如无符号 dtype 被 NNRt/HIAI 栈拒绝）、HAP 推理参数协议的源码依据、端侧验证工作流编排归 `ops-kirin-device-verify`；本 skill 只收 hdc 命令层规律

## 信息来源

- 命令语法、参数和错误码参考华为开发者官方文档《hdc》的命令说明，按本项目端侧验证场景重新整理：
  https://developer.huawei.com/consumer/cn/doc/doccenter-capabilities/hdc
- `hdc_examples.md` 中的部署序列来自历史开发记录（已去除设备标识、路径等个人信息并抽象化）
- 文中部分命令未在真实设备验证，**以官方文档为准**

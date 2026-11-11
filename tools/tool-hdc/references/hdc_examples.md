# hdc 场景示例与端侧部署序列

本文件给出端侧算子验证中常用的 hdc 操作序列。命令语法以实际环境 `hdc -h` 和目标设备系统版本为准；未在真实设备上复验的步骤会明确标注。

## 基础示例

### 抓取设备日志

```bash
# 持续打印，Ctrl+C 结束
hdc shell hilog > local.log

# 使用 hilog 落盘能力，再拉回落盘目录（路径以设备实际为准）
hdc shell "hilog -w start"
hdc file recv /data/log/hilog ./hilog_log/
```

### 安装 HAP 并确认 bundle 名

```bash
hdc install -r app.hap
hdc shell bm dump -a
hdc shell bm dump -n <bundle>
```

`bm dump -a` 用于列出设备上的 bundle 名；不要凭猜测填写 bundle。

### 推送输入并拉回输出

```bash
hdc shell mkdir -p /data/local/tmp/board_test/
hdc file send input1.bin /data/local/tmp/board_test/
hdc file send model.om /data/local/tmp/board_test/
hdc file recv /data/local/tmp/board_test/output0.bin ./output0.bin
```

### 多设备环境

```bash
hdc list targets
hdc -t <connect-key> shell echo hello
```

连接多台设备时，每条命令都要带 `-t`。

## 端侧算子部署验证序列

场景：把自定义算子的离线模型部署到 HarmonyOS 麒麟设备，执行推理并回传结果。

占位符说明：

- `<connect-key>`：设备标识。
- `<bundle>`：推理宿主应用包名。
- `<ability>`：宿主应用的 Ability 名。
- `<device_dir>`：设备端工作目录。
- `<model>.omc` / `<model>.om`：离线模型。
- `<input>.bin`：输入数据。

> 推理宿主 HAP（负责加载模型和执行推理）由应用工程构建，需使用调试证书签名。本 skill 不负责生成 HAP；没有现成 HAP 时，流程无法启动。

```bash
# 1. 确认设备在线
hdc list targets
hdc -t <connect-key> shell param get <product-param>

# 2. 安装宿主 HAP，确认 bundle 名
hdc -t <connect-key> install <host_app>.hap
hdc -t <connect-key> shell bm dump -a

# 3. 准备设备目录并推送输入/模型
hdc -t <connect-key> shell mkdir -p <device_dir>/
hdc -t <connect-key> file send <input1>.bin <device_dir>/
hdc -t <connect-key> file send <input2>.bin <device_dir>/
hdc -t <connect-key> file send <model>.omc <device_dir>/

# 4. 清理旧输出，强停应用后启动推理
hdc -t <connect-key> shell "rm -f <device_dir>/output0.bin"
hdc -t <connect-key> shell aa force-stop <bundle>
hdc -t <connect-key> shell aa start -a <ability> -b <bundle> \
    --ps path <input1>.bin,<input2>.bin \
    --ps omPath <model>.omc

# 5. 查看推理日志
hdc -t <connect-key> shell hilog | grep <app-log-tag>

# 6. 拉回输出并校验
hdc -t <connect-key> file recv <device_dir>/output0.bin ./output0.bin
wc -c ./output0.bin
```

> 第 4 步中的 `rm -f`、`aa force-stop` 属于破坏性操作，执行前必须让用户确认。

## 经验与判定

1. **应用未退出时重复 `aa start` 可能不会重新推理**：Ability 复用会让新参数失效，输出文件也不更新。两次推理之间先执行 `aa force-stop`。
2. **先校验字节数，再比精度**：输出字节数应等于元素总数乘以 dtype 宽度。字节数不对时，先确认推理是否真正执行。
3. **`--ps` 的 key 由宿主应用定义**：`path`、`omPath` 只是示例，实际名称需要向应用方确认。
4. **`file recv` 的参数顺序是远端在前、本地在后**，与 `file send` 相反。
5. **`aa start` 返回成功不代表推理完成**：常见原因有两类——Ability 复用未重新执行，或设备运行时拒载模型。统一以输出文件生成和字节数为准，用 `hdc shell hilog` 取证。
6. **批量用例要防覆盖**：`file send` 一次只传一个文件；设备侧文件名加用例前缀，固定名输出每次拉回后立即改名。推理是异步的，可用 `stat -c %s` 轮询输出文件直到字节数达标。更多路径与传输问题见 [hdc_transfer_and_path_pitfalls.md](hdc_transfer_and_path_pitfalls.md)。

## 验证状态

- 文中部分命令未在真实设备验证，使用前按官方文档核对。
- 命令语义以 hdc 帮助和官方说明为准；部署序列来自历史开发记录的脱敏整理。
- 在真实设备上验证后，可以把新的回显补充为示例。

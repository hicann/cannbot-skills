# hdc 文件传输与路径处理实战避坑

> hdc 命令语义本身简单，实战问题集中在**路径解析**与**结果可信度**两处。本文件汇总
> 脚本化/自动化使用 hdc 时的传输与路径规律；命令语法以官方文档为准。

## 1. 路径引号与转义：rc=0 静默失败

**现象**：命令经脚本或多层 shell 拼接下发给 hdc 时，若路径带双引号或使用 `\\` 转义，
hdc 可能返回 rc=0 且无报错，但文件实际未传输（目标位置文件不存在）。

**根因**：中间 shell 层吞掉引号/转义后，参数被错误解析；hdc 对不可解析路径不报错、
静默忽略。**rc 与返回信息均不可信，必须以文件存在性为准**。

**规则**：
- 路径不含空格时**不加引号**（引号仅用于确实含空格且直接交互执行的场景）
- 脚本调用时命令以**单字符串整体下发**，避免多层拼接转义
- 注意区分：官方文档中 `install -w/-u` 等「需加引号」的约定与路径转义问题无关，勿混用

## 2. Windows 路径斜杠规范

**现象**：hdc 运行在 Windows 系统时，本地路径使用正斜杠 `/` 或混合斜杠 → 报
`no such file`，路径被当作相对路径拼到 `C:\Users\<user>\` 前缀下。`file send`
（本地→设备）与 `file recv`（设备→本地）**两个方向均会命中**。

**规则**：Windows 侧路径统一使用**全反斜杠的绝对路径**（如 `C:\data\output0.bin`）。

## 3. 传输必双侧字节校验

仅信 rc=0 不够，推送/回传后立即校验：

```bash
# file send 后：设备侧字节数 == 本地字节数
hdc -t <key> shell stat -c %s <device_dir>/<file>    # 与本地 stat -c %s / os.path.getsize 比对

# file recv 后：本地文件存在 + 字节数 == 元素数 × dtype 宽度
hdc -t <key> file recv <device_dir>/output0.bin <local_path>
```

字节数不匹配 = 传输失败或对端未真正产出数据，先查链路再进行后续处理。

## 4. 一次一个文件与防覆盖

- `file send` **一次只传一个文件**（多个文件参数会被静默忽略，只落第一个），逐个传
- 多用例场景设备侧文件名加用例前缀（如 `case2_input1.bin`）防同名覆盖
- 固定名输出文件（如 `output0.bin`）每次 recv 后立即改名保存

## 5. 异步执行：返回成功 ≠ 执行完成

设备端应用异步执行推理类任务时，`aa start` 等启动命令返回成功不代表已产出结果。
以**输出文件生成 + 字节数达标**为准，轮询检测：

```bash
# 轮询直至字节数 == 元素数 × dtype 宽度（设 deadline + 轮询间隔）
hdc -t <key> shell stat -c %s <device_dir>/output0.bin
```

返回成功但始终无输出时，用设备日志定位：`hdc shell hilog -x | grep <tag/bundle>`。

## 6. Windows cmd 语法差异

在 Windows 上执行 hdc.exe（cmd 批处理环境，非 POSIX shell）时，本地侧语法：

| 需求 | cmd 写法（勿用 bash 语法） |
|------|--------------------------|
| 目录不存在才创建 | `if not exist C:\data\ mkdir C:\data\` |
| 文件存在判断 | `if exist C:\data\output0.bin (echo ok) else (echo missing)` |
| 文本过滤 | `findstr <pattern>`（替代 grep） |
| 文件大小 | 设备侧是 Linux，可用 `hdc shell stat -c %s`（Windows 侧 `stat` 不可用时） |

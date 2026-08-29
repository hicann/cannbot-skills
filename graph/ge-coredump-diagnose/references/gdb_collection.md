# 崩溃现场采集手册

诊断流程向用户要材料时使用。每节按"什么时候要 → 怎么说 → 用户怎么做 → 采完提交什么"组织，命令直接复制给用户在崩溃机执行，按章节顺序从前往后要。

---

## 0. 确认 core 文件

**什么时候要**：最前面。先问用户是否有崩溃产生的 core 文件，有则记录路径直接进入第 1 章；没有且程序还能运行时，告诉用户怎么生成并重新触发一次崩溃。

**话术**：

```text
请问崩溃后是否留下了 core 文件？如果有，我接下来会给你一条 gdb 命令，用 core 和崩溃进程的可执行文件一起导出完整堆栈，你只需要填这两个文件的路径。如果没有且程序还能再次运行，请按下面的方法打开 core 生成后重新触发一次崩溃；如果当前无法复现，直接说明现有 backtrace 或日志情况即可。
```

**没有 core 时，让用户这样生成**：

1. 检查 core 生成开关：

   ```bash
   ulimit -c
   ```

   输出 `0` 表示 core 生成被关闭，打开它（只在当前会话生效）：

   ```bash
   ulimit -c unlimited
   ```

2. 确认 core 落盘位置和命名规则：

   ```bash
   cat /proc/sys/kernel/core_pattern
   ```

   - 输出是 `core` 这类文件名：core 生成在进程崩溃时的工作目录；
   - 输出以 `|` 开头（如 `|/usr/lib/systemd/systemd-coredump`）：core 由 systemd-coredump 接管，不会落成普通文件，用下面的命令查看和导出：

     ```bash
     coredumpctl list | grep <进程名>
     coredumpctl info <PID>
     coredumpctl dump <PID> -o core.<PID>
     ```

     导出的 `core.<PID>` 就是 core 文件，后续 gdb 命令直接用它；
   - 输出是其他路径模式：core 生成到该模式指定的目录，去那里找。

3. 用原启动命令重新运行，复现崩溃，然后确认 core 已生成：

   ```bash
   ls -lh <core_pattern 指示的目录>/core*
   ```

4. 顺带记录环境信息（容器/裸机、OS 版本），core 相关行为有时和内核配置有关。

如果当前无法重启进程或无法再次复现，不强求补生成 core：已有完整 backtrace 时先按 SKILL 第 1 节判断是否纳入，再按现有材料继续做降级分析；只有日志且没有完整 backtrace 时，按 P0 停在材料盘点，并在报告中写明无法取得崩溃现场。

---

## 1. 导出崩溃现场

**什么时候要**：core 确认有了之后。本节产出的是完整 backtrace（P0 材料，问题描述 + 完整 backtrace 缺了真没法开始定位）。用户已有完整 backtrace（日志带出、或已导出过）时跳过本节；没有 backtrace 但有 core 时，用本节命令生成。

**话术**：

```text
当前只能确认进程发生崩溃，无法定位崩溃函数。请在崩溃机执行以下命令，导出同一次崩溃的所有线程堆栈并保存为文件；不能只截取栈顶几行。
```

**命令**（把可执行文件和 core 路径替换为实际绝对路径）：

```bash
gdb -q -batch \
  -ex 'set pagination off' \
  -ex 'set confirm off' \
  -ex 'set print thread-events off' \
  -ex 'info threads' \
  -ex 'thread apply all bt full' \
  -ex 'info registers' \
  -ex 'p $_siginfo' \
  -ex 'info proc mappings' \
  -ex 'info sharedlibrary' \
  /path/to/executable /path/to/core \
  > gdb_core.txt 2>&1
```

命令含义：`info threads` 列出所有线程；`thread apply all bt full` 导出所有线程完整调用栈和局部变量；`info registers` 导出崩溃时寄存器；`p $_siginfo` 导出信号详细信息（si_addr 崩溃访问地址、si_code 访问性质，判断空指针/越界的关键）；`info proc mappings` 导出进程地址映射；`info sharedlibrary` 列出动态库及符号加载状态。这些合起来就是完整现场，后续分析不再反复要用户重跑。个别 gdb 版本不支持 `p $_siginfo` 时报错不影响其余输出，si_addr/si_code 视为采不到，按 unverified 处理。部分旧版或非 Linux gdb 可能不支持 `info proc mappings`，此时改用 `info files`（必要时再用 `maintenance info sections`）获取映射信息。

**命令文件方式**（命令长，或需要反复执行时用）。创建 `gdb_commands.txt`，内容为：

```text
set pagination off
set confirm off
set print thread-events off
info threads
thread apply all bt full
info registers
p $_siginfo
info proc mappings
info sharedlibrary
quit
```

执行（同样替换两个路径）：

```bash
gdb -q -batch -x gdb_commands.txt \
  /绝对路径/到/可执行文件 \
  /绝对路径/到/core文件 \
  > gdb_core.txt 2>&1
```

`-x` 让 gdb 逐行读取命令文件执行。两种方式产出相同的 `gdb_core.txt`，选一种即可。

**core 的去向**：core 不需要提交给诊断方。按本章在崩溃机执行命令时，提交符号化后的 `gdb_core.txt` 即可；如果用户已有完整 backtrace，则提交其原始日志或文本来源即可。但 core 和可执行文件要**保留在崩溃机上别删**：涉及内存类根因（UAF/越界/踩内存）确认时，还要回到崩溃机用第 3 章的方式继续挖。

**提交什么**：

```text
gdb_core.txt       # 通过本章采集时提交；已有完整 backtrace 时可用其原始日志/文本来源替代
gdb 版本           # 通过本章采集时顺带告知，文本即可
```

core 文件、可执行文件、`.so` **不需要提交**：二进制太大且跨环境无法解析，所有 gdb 操作都在崩溃机上完成，诊断只需要文本产物。用户已有完整 backtrace 时，保留并提交其原始日志或文本来源。

要求：不手工删除中间线程或栈帧；`gdb_core.txt` 过大可压缩，保留原始压缩文件。

---

## 2. 堆栈只有地址没有行号：装 debug 包重新复现一遍

**什么时候要**：backtrace 只有 `??`、十六进制地址或函数名不完整，没有 `file:line`。

**话术**：

```text
当前 backtrace 只有地址，是崩溃环境里的包不带符号。请安装与目标版本一致的 CANN debug 包（就是对应子包的 debug 版本，正常安装替换即可），然后用原启动命令重新复现一次，生成新的 core；再对这次新 core 和对应的可执行文件/.so 执行第 1 章命令，提交新的 gdb_core.txt。不要拿新安装的 debug 包去解析之前那次崩溃的旧 core。
```

**判断成败**：新输出里有 `file:line` 就是成功；仍然只有 `??` 或 `No symbol table info available` 就是符号还没加载上，优先检查 debug 包版本是否和崩溃二进制完全一致（版本号要一模一样）。

**给不了的后果**：只能做地址/函数级初步分析，不能可靠定位源码行和根因，报告降置信度并写明限制。

---

## 3. 前两章之后还分析不出来

到这里就没有流程化动作了。前两章的标准材料（完整 backtrace，通常来自 `gdb_core.txt`，含 file:line）已经够支撑源码分析；如果仍然定不了根因，说明缺的不是标准采集，而是针对当前假设的专项证据。此时由诊断方根据具体假设设计手段，指导用户在崩溃机上交互执行，一轮一收：

- **专项 gdb 操作**：用保留的 core 查具体值——`p <变量>` 看参数实际值、`x/i $pc` 看崩溃指令、`info locals` 看局部变量、`thread <N>` + `bt full` 看特定线程；诊断方给命令，用户贴回输出。
- **加诊断手段**：临时日志、Debug 构建复现、ASAN/TSAN、单线程/多线程对照等，按验证哪个假设来选，由诊断方说明预期结果并解读输出。

原则：每轮只问当前假设需要的那一个证据，不撒网式让用户跑命令。

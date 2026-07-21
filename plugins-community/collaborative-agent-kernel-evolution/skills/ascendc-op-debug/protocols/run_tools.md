# Protocol: Layer 3 — 工具链调用

> 仅在 Layer 1（代码审查）和 Layer 2（日志分析）均未定位时使用。
> 本文件自包含，无外部依赖。脚本路径均相对于 `skills/ascendc-op-debug/`。

## 工具选择决策树

```
代码审查未定位
      │
      ├─ 偶发不一致 / 同步疑问    → msSanitizer synccheck / racecheck
      ├─ 内存越界 / 对齐错误      → msSanitizer memcheck
      ├─ 需要逐行断点 / 看中间值  → msDebug（交互调试，替代 DumpTensor）
      ├─ 有 AI Core dump 文件     → msaicerr
      └─ 挂起 / 不知道卡在哪      → msDebug Ctrl+C 中断 + 查 PC
```

---

---

## msDebug（交互式断点调试）

msSanitizer 是"跑完看报告"，msDebug 是"断点进去看变量"——两者互补。
**适用场景**：代码看不出来 + msSanitizer 不报错 + 需要在运行时观察 UB/GM 中间值。

### 编译要求

```cmake
add_ops_compile_options(ALL OPTIONS -g -O0)
# -g：生成调试符号；-O0：禁用优化（避免代码乱序）
```

### 环境准备

```bash
# 检查 debug switch（需要为 1）
cat /proc/debug_switch
# 若不为 1，需要 root 开启：
sudo bash -c "echo 1 > /proc/debug_switch"

# 找到算子的 .o 文件并设置环境变量
find /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize \
    -name "*.o" | grep {op_name}
export LAUNCH_KERNEL_PATH=/path/to/kernel.o
```

### 快速启动

```bash
# ① 自动化调试脚本（推荐）：自动打断点 + 采集变量 + 生成 HTML 报告
python3 ${CLAUDE_SKILL_DIR}/scripts/msdebug_auto_debug.py \
    --source-file op_kernel/{op_name}.cpp \
    --executable ./test_op \
    --breakpoints "45,67,89" \
    --output debug_report.html \
    --format html

# ② 交互式调试
msdebug ./your_executable

# ③ coredump 分析
msdebug --core corefile /path/to/kernel.o
```

### 常用命令速查

| 需求 | 命令 |
|---|---|
| 打断点 | `b my_op.cpp:45` |
| 运行 | `run` |
| 下一步（不进函数）| `next` |
| 步入函数 | `step` |
| 打印变量 | `print xLocal` |
| 打印所有局部变量 | `var` |
| 读 UB 内存（float）| `memory read -m UB -f float[] 0x0 -s 256 -c 1` |
| 读 GM 内存（half）| `memory read -m GM -f float16[] 0x1240c0015000 -s 128 -c 1` |
| 跳过前 N 个元素 | `memory read -m UB -f float[] 0x0 -s 10 -c 1 -E 100` |
| 查所有核状态 | `ascend info cores` |
| 切换到第 N 核 | `ascend aiv N` |
| 调用栈 | `bt` |
| 中断挂起的程序 | `Ctrl+C` |
| 退出 | `quit` |

### 典型场景

**场景1：定位精度 bug（替代 DumpTensor，无需改代码重编译）**
```bash
(msdebug) b my_op.cpp:67     # 计算后打断点
(msdebug) run
(msdebug) print xLocal       # 查看输入
(msdebug) print outputLocal  # 查看输出，与 golden 对比
```

**场景2：定位挂起（kernel 卡死在哪）**
```bash
(msdebug) run
# 等待几秒后 Ctrl+C 中断
(msdebug) ascend info cores  # 看哪个核卡住
(msdebug) ascend aiv 3       # 切到卡住的核
(msdebug) bt                 # 看调用栈
(msdebug) register read $PC  # 看当前执行位置
```

**场景3：coredump 分析**
```bash
msdebug --core corefile /path/to/kernel.o
(msdebug) ascend info summary   # 查异常类型
(msdebug) bt                    # 定位崩溃代码行
(msdebug) register read -a      # 查寄存器状态
```

---

## msSanitizer（同步 / 竞争 / 内存检测）

对应 evidence=`tool_sanitizer` 的 hypothesis：H11（sync_missing）、H12（event_id_reuse）、H13（alignment_violation）

### 编译要求（必须，否则 sanitizer 无效）

```cmake
# kernel 侧 CMakeLists.txt
add_ops_compile_options(ALL OPTIONS -sanitizer -g)
```

### 环境变量

```bash
export PYTORCH_NO_NPU_MEMORY_CACHING=1   # 禁用内存池，避免干扰检测
export TRITON_ALWAYS_COMPILE=1            # Triton 场景额外设置
```

### msSanitizer 路径

```bash
# CANN toolkit 内置，无需单独安装
/usr/local/Ascend/ascend-toolkit/latest/tools/mssanitizer/bin/mssanitizer
# 或
which mssanitizer
```

---

### 检测优先级（顺序执行）

```
1. memcheck   → 排除内存越界 / 对齐问题（最基础）
      ↓
2. synccheck  → 检测 SetFlag/WaitFlag 配对问题（最常见）
      ↓
3. racecheck  → 检测 RAW/WAW/WAR 数据竞争
      ↓
4. initcheck  → 检测未初始化内存读取（较罕见）
```

---

### 快速启动命令

```bash
# ① 使用自动化诊断脚本（推荐，自动运行所有检测 + 生成 HTML 报告）
python3 ${CLAUDE_SKILL_DIR}/scripts/mssanitizer_diagnose.py \
    ./your_test_executable --check all --output report.html

# ② 单项检测
python3 ${CLAUDE_SKILL_DIR}/scripts/mssanitizer_diagnose.py \
    ./your_test_executable --check synccheck
python3 ${CLAUDE_SKILL_DIR}/scripts/mssanitizer_diagnose.py \
    ./your_test_executable --check memcheck

# ③ 直接调用 msSanitizer（调试特定 kernel）
mssanitizer --tool=synccheck bash run.sh
mssanitizer --tool=racecheck bash run.sh
mssanitizer --tool=memcheck --leak-check=yes bash run.sh
mssanitizer --tool=initcheck bash run.sh

# ④ 只检测特定 kernel 或特定 block（缩小范围）
mssanitizer --tool=synccheck --kernel-name={op_kernel_name} bash run.sh
mssanitizer --tool=synccheck --block-id=0 bash run.sh
```

---

### 报告速查表

| 输出关键词 | 含义 | 严重度 | 对应 hypothesis |
|---|---|---|---|
| `Unpaired set_flag` | SetFlag 无对应 WaitFlag | ERROR | H11 |
| `Potential RAW hazard` | Read-After-Write 竞争 | ERROR | H11 |
| `Potential WAW hazard` | Write-After-Write 竞争 | ERROR | H11 |
| `illegal read/write of size X` | 数组越界 | ERROR | H13 |
| `out of bounds of size X` | 多核内存重叠 | WARNING | H01 |
| `misaligned access` | 未对齐访问 | ERROR | H13 |
| `LeakCheck: detected memory leaks` | 内存泄漏 | ERROR | — |
| `uninitialized read of size X` | 未初始化读 | ERROR | — |

---

### mhc_post_fusion 超时专项（error 507034）

**症状**：Vector Core 超时，synccheck 可能不报错

**专项诊断脚本**（包含完整 6 步诊断）：

```bash
cd /path/to/mhc_post_fusion/output
bash ${CLAUDE_SKILL_DIR}/scripts/mhc_post_fusion_diagnose.sh
```

**根因**：高频 SetFlag/WaitFlag（~200次/ComputeXHat）+ 仅 2 个 event ID 复用 → 硬件状态机溢出

**修复优先级**：

```cpp
// ✅ 方案1（最优）：改用 TQue，框架自动管理同步
LocalTensor<float> buf = queue.AllocTensor<float>();
queue.EnQue(buf);
buf = queue.DeQue<float>();  // 自动 WaitFlag
queue.FreeTensor(buf);

// ✅ 方案2：批量 DMA，减少同步频率
for (int c = 0; c < 24; c++) DataCopy(batchBuf[c], gm[offset[c]], size);
SetFlag<HardEvent::MTE2_V>(eId);   // 一次同步代替 24 次
WaitFlag<HardEvent::MTE2_V>(eId);

// ❌ 方案3（治标）：增加独立 event ID，可能超出硬件上限
```

---

### 常见 Pattern 速查

**Pattern 1：Event ID 复用（H12 症状：hang/timeout）**

```cpp
// ❌ 循环内高频复用同一 event_id
for (int h = 0; h < 4; h++)
    for (int c = 0; c < 24; c++) {
        SetFlag<HardEvent::MTE2_V>(eIdMte2V);  // 每次都是 eIdMte2V
        WaitFlag<HardEvent::MTE2_V>(eIdMte2V);
    }
```

**Pattern 2：跨流水线同步缺失（H11 症状：racecheck hazard）**

```cpp
// ❌ MTE3 写完，MTE2 立刻读，无 barrier
DataCopy(gm[0], ubuf, size);      // MTE3 写 GM
DataCopy(ubuf2, gm[0], size);     // MTE2 读 GM（可能读到旧数据）

// ✅ 加 MTE3→MTE2 barrier
DataCopy(gm[0], ubuf, size);
pipe_barrier(PIPE_MTE3);
DataCopy(ubuf2, gm[0], size);
```

**Pattern 3：DataCopy 长度越界（H13 症状：memcheck illegal write）**

```cpp
// ❌ 长度超出 buffer
DataCopy(buf, gm, 2 * TILE_LENGTH);  // buf 只有 TILE_LENGTH 大小

// ✅ 边界检查
uint32_t copySize = std::min(remaining, TILE_LENGTH);
DataCopy(buf, gm[offset], copySize);
```

---

## msaicerr（AI Core Dump 分析）

对应 evidence=`tool_msaicerr` 的 hypothesis：H14

**触发条件**：运行时有 AI Core error dump 文件生成，或 plog 中出现 AICore exception

```bash
# 检查 dump 文件是否存在
ls ~/ascend/log/dump/

# 使用 msaicerr-helper skill（CAKE2 项目独立 skill）
# skills/msaicerr-helper/SKILL.md

# 步骤1：解析 AI Core 错误报告
python3 ${CLAUDE_PLUGIN_ROOT}/skills/msaicerr-helper/scripts/msaicerr.py \
    --input ~/ascend/log/dump/ --output error_report/

# 步骤2：解析 tiling 数据（如有）
python3 ${CLAUDE_PLUGIN_ROOT}/skills/msaicerr-helper/scripts/parse_tiling.py \
    --input tiling.bin --output tiling_parse/

# 步骤3：单算子测试
python3 ${CLAUDE_PLUGIN_ROOT}/skills/msaicerr-helper/scripts/test_single_op.py \
    --op_name {op_name} --shape {shape}
```

**错误码对照（plog 速查）**：

| 错误码 | 含义 |
|---|---|
| `161xxx` | Runtime 通用错误 / 参数错误 |
| `507034` | Vector Core 超时 → 见 mhc_post_fusion 专项 |
| `561002` | Executor 内部错误 |
| `561003` | 属性配置错误 |
| `561107` | JSON op 描述错误 |
| `561112` | 二进制包错误，需重新编译 |
| DDR out-of-range | 内存越界 → 先排查 H02（workspace）|

# clock 和 asc_time_stamp

资料基线及版本适用性见 [平台与编程模式支持矩阵](platform_matrix.md#来源基线)。示例为调用片段，使用前需核对目标版本和编程模式。

## `clock`

### 接口功能

- **输入**：无参数。
- **行为**：读取调用时刻的周期计数。
- **输出**：`uint64_t` 数值；区间周期差由调用者计算 `end - start`，需要展示时另行打印或写回。
- **不保证**：不自动生成日志、不自动换算为时间，也不保证异步流水操作在读数前完成。不能把局部周期差直接解释为整算子耗时。
- **最小示例**：下例 `RunTheCode()` 是待测代码占位符，应替换为真实代码并明确测量所需的同步边界。

```cpp
#include "utils/debug/asc_time.h"

uint64_t start = clock();
RunTheCode();
uint64_t end = clock();
uint64_t cycles = end - start;
```

`clock()` 返回从程序开始到调用时刻经历的时钟周期数，返回类型为 `uint64_t`。它适合测量同一线程中一段局部代码的相对周期差；不要把不同线程或不同核的值直接当作一个全局同步时间线。

测量时：

- 只让一个代表线程打印结果，或将结果写回 GM 后由 Host 汇总。
- 在测量区域外放置 `printf`/`asc_dump`，否则调试输出本身会污染结果。
- 说明冷启动、编译优化、同步和线程调度对结果的影响。
- 需要跨阶段结构化记录时，使用 `asc_time_stamp`，不要自行把 `clock` 输出伪装成 timestamp dump。

## `asc_time_stamp`

### 接口功能

- **输入**：`uint32_t desc_id`，用于区分打点位置。
- **行为**：开启编译开关后，在支持的 SIMD/NPU 场景记录调用位置的时间点。
- **输出**：含 `descId`、`rsv`、`timeStamp`、`pcPtr` 和 `entry` 的结构化记录，无返回值。其中 `entry` 是算子开始执行的 cycle 数，不是入口地址。
- **不保证**：不自动配对开始/结束标记、不自动计算阶段耗时或推导性能瓶颈；不能替代完整 profiling。
- **最小示例**：下例使用用户 ID `0x10000`，编译时需开启后述宏。

```cpp
#include "utils/debug/asc_time.h"

asc_time_stamp(0x10000);
```

`asc_time_stamp` 在 SIMD Kernel 中按用户描述符输出结构化时间戳信息，包括当前 cycle、PC 指针数值和算子开始执行的 cycle 等字段。默认关闭，需要增加：

```text
-DASCENDC_TIME_STAMP_ON
```

用户自定义 ID 建议大于 `0xffff`，因为 `[0, 0xffff]` 预留给 AscendC 内部模块。该能力仅用于 NPU 实机调试，本参考基线不支持算子集成到计算图执行（入图）的场景。

按本参考基线，单次输出不可超过 1 MB；自定义算子工程中，每核所有 Dump 接口的累计输出也不可超过 1 MB，需计入框架头尾信息。

## “时间戳没有输出”排查

1. 检查调用是否位于 SIMD 支持的 Kernel 路径。
2. 确认重新编译时确实传入 `-DASCENDC_TIME_STAMP_ON`。
3. 确认不是 CPU 调试或算子入图场景。
4. 检查每核累计 dump 数据量和 `ASCENDC_DUMP` 设置。
5. 检查 Kernel 是否在执行到打点位置前就失败或提前返回。

## 与 profiling 的边界

本技能只覆盖 `clock` 和 `asc_time_stamp`。`asc_prof_start`、`asc_prof_stop`、`asc_mark_stamp`、`TRACE_START`、`TRACE_STOP` 不在本技能范围，不要用本参考资料推导它们的签名或行为。

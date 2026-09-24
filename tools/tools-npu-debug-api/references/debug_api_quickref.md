# 调测接口快速参考

资料基线及版本适用性见 [平台与编程模式支持矩阵](platform_matrix.md#来源基线)。使用前需核对目标版本和编程模式。

本表用于第一轮选型。具体产品支持和 API 变体以当前 `asc-devkit/docs/zh/api` 和公开头文件为准。

| 接口 | 头文件 | 典型模式 | 成功行为 | 失败/缺失输出重点 |
| --- | --- | --- | --- | --- |
| `printf` / `AscendC::printf` / `PRINTF` | 全局调用使用 `utils/debug/asc_printf.h`；`AscendC::printf` 使用 `kernel_operator.h` | SIMD；Ascend 950PR/950DT 上支持 SIMT/VF 变体 | 输出格式化文本 | 格式符、参数类型、FIFO、打印线程和 `ASCENDC_DUMP` |
| `assert` / `ascendc_assert` | `utils/debug/asc_assert.h` | SIMD；Ascend 950PR/950DT 上支持 VF/SIMT 相关变体 | 条件真则继续 | `NDEBUG`、`ASCENDC_DUMP`、模式限制、自定义消息容量 |
| `__trap` | `utils/debug/asc_assert.h` | SIMT/SIMT VF，平台需核对 | 触发条件时中断 Kernel | 仅支持公开声明的场景，不作为 SIMD API |
| `asc_dump_*` / `asc_dump` | `utils/debug/asc_dump.h` | SIMD；Ascend 950PR/950DT 上有额外 VF/Buffer 变体 | 输出指定地址空间数据 | 重载、`T`、`dump_size`、32 字节补齐、每核上限 |
| `clock` | `utils/debug/asc_time.h` | SIMD；Ascend 950PR/950DT 上支持 SIMT | 返回 `uint64_t` cycle | 线程局部性、同步、冷启动和调试扰动 |
| `asc_time_stamp` | `utils/debug/asc_time.h` | SIMD、NPU 实机运行 | 写出带 `desc_id` 的结构化时间戳 | `-DASCENDC_TIME_STAMP_ON`、入图限制、1 MB/核上限 |

## 详细说明

主文件提供接口选择依据；按接口读取以下文档中的输入、行为、输出、非保证项和最小示例：

- [printf](printf.md)：格式化文本观测。
- [assert、ascendc_assert 和 __trap](assert_and_trap.md)：条件检查与显式中断的区别。
- [asc_dump](asc_dump.md)：按地址空间观测数据。
- [clock 和 asc_time_stamp](clock_and_timestamp.md)：返回周期数值与输出结构化记录的区别。

## 最小选型规则

- 确认某个分支是否执行：优先少量 `printf`。
- 想让错误状态立即可见并中止算子：用 `assert`/`ascendc_assert`；不要把 `printf` 后继续执行当作错误处理。
- 在 SIMT 线程检测到不可恢复状态时中断核函数：用条件判断 + `__trap`。
- 需要查看张量或片上缓存：用地址空间对应的 `asc_dump_*`，先确认内存范围。
- 需要返回值参与计算或写回 GM：用 `clock`。
- 需要统一的、可解析的 SIMD 时间点记录：用 `asc_time_stamp`。

## 资料来源

- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/printf.md`
- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/assert.md`
- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/__trap.md`
- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/asc_dump.md`
- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/clock.md`
- `asc-devkit/docs/zh/api/Utils-API/tuning_interface/asc_time_stamp.md`

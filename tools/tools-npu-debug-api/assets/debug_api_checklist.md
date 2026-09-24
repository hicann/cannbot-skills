# 核函数调试接口检查清单

仅检查本次使用的接口；不适用项标记为“不适用”并说明原因。

## 环境与接口

- [ ] 已确认 SoC、`__NPU_ARCH__` 和 CANN 版本。
- [ ] 已确认 SIMD、SIMT、SIMT VF 或 SIMD VF 编程模式。
- [ ] 已选择与目标一致的头文件和地址空间重载。
- [ ] `printf` 的格式符、参数数量和类型已匹配。
- [ ] 断言测试没有被 `NDEBUG` 或 `ASCENDC_DUMP=0` 静默关闭。
- [ ] `__trap` 只出现在文档支持的 SIMT 路径。
- [ ] `asc_dump` 的 `dump_size` 不超过实际元素数，并考虑 32 字节补齐。
- [ ] 已检查 FIFO、每核容量、单条记录和线程过滤。
- [ ] `clock` 的测量区不包含调试输出，并控制线程/同步影响。
- [ ] `asc_time_stamp` 已使用 `-DASCENDC_TIME_STAMP_ON` 重新编译，并确认不是入图场景。
- [ ] 未把 `asc_prof_start`、`asc_prof_stop`、`asc_mark_stamp`、`TRACE_START` 或 `TRACE_STOP` 当作本技能接口。

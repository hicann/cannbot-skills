---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: ".cpp Dispatcher 入口文件模板与规则"
description: "每算子在顶层生成 <kernel>.cpp 内核入口：extern \"C\" __global__ __aicore__ void {kernel_name}(...)，用 GetBlockIdx()/GetBlockNum() 组线程参数，经 DISPATCH_VALUE_SIZE + Simt::VF_CALL 调 _vf。"
confidence: single_run
original_id: SIMT_PATTERNS.md#8-cpp-dispatcher-文件模板
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, dispatcher, kernel-entry]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
每个算子目录除 `v35/<kernel>.h` 外，还需在顶层生成 `<kernel>.cpp` 作为 AscendC 内核入口（对应 source 的 `__global__` 启动点）。固定规则：

- 入口函数名 = `{kernel_name}`（不带 `_vf` 后缀）。
- 修饰符固定写法：`extern "C" __global__ __aicore__ void {kernel_name}(...)`。
- 首行：`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);`。
- `block_index`：在 `.cpp` 内用 `GetBlockIdx()` 获取后传入 `_vf`。
- `total_thread_num`：`THREAD_NUM * GetBlockNum()`，作为 `_vf` 的 `thread_all` 参数传入。
- `system_cycle`：仅当 `_vf` 需要时 `static_cast<uint64_t>(AscendC::GetSystemCycle())`。
- `DISPATCH_VALUE_SIZE`：始终需要，把运行时 `value_size` 映射为 `DTYPE` 模板参数。
- `DISPATCH_EVICT_STRATEGY`：仅当 kernel 使用 ScoreFunctor / 淘汰策略时包含。
- `Simt::VF_CALL<...>`：固定包装 kernel 调用，第一参数为 `Simt::Dim3{THREAD_NUM}`。
- include：`"./v35/{kernel_name}_kernel.h"`、`"../../include/simt_vf_dispatcher.h"`。

标准骨架（含淘汰策略）：
```cpp
extern "C" __global__ __aicore__ void {kernel_name}(/* GM_ADDR 与标量参数，同 _vf；含 value_size、evict_strategy 等 */) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  uint64_t system_cycle = static_cast<uint64_t>(AscendC::GetSystemCycle());
  const uint64_t total_thread_num = THREAD_NUM * GetBlockNum();
  DISPATCH_VALUE_SIZE(value_size,
    DISPATCH_EVICT_STRATEGY(evict_strategy,
      (Simt::VF_CALL<{kernel_name}_vf<uint64_t, DTYPE, uint64_t, STRATEGY>>(
          Simt::Dim3{static_cast<uint32_t>(THREAD_NUM)}, /* GM 参数... */,
          n, /* ... */, global_epoch, total_thread_num, system_cycle,
          GetBlockIdx(), max_bucket_shift,
          capacity_divisor_magic, capacity_divisor_shift))));
}
```
无淘汰策略的简单算子（如 clear）去掉 `DISPATCH_EVICT_STRATEGY` 层，仅保留 `DISPATCH_VALUE_SIZE` + `Simt::VF_CALL`。

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT kernel 头文件骨架：include guard / 头文件 / 命名空间"
description: "kernel .h 用 ASCENDC_{NAME_UPPER}_KERNEL_H_ 守卫，include kernel_operator.h + types.h + utils.h（ScoreFunctor 时才加 score_functor.h），置于 npu::hkv + using AscendC。"
confidence: single_run
original_id: SIMT_PATTERNS.md#6-完整的函数头文件模板
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, header-template, scaffold]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
每个 kernel 的 `v35/<kernel>.h` 头文件按固定骨架生成：

- include guard 命名：`ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_`。
- 头文件包含：`<kernel_operator.h>`、`<cstdint>`、`"../../../include/types.h"`、`"../../../include/utils.h"`；`score_functor.h` 仅在使用 ScoreFunctor（淘汰策略）时才包含。
- 命名空间：置于 `namespace npu { namespace hkv { ... } }` 内，并 `using namespace AscendC;`。
- 在命名空间内定义 `constexpr uint32_t THREAD_NUM = ...;`（值按算子类型选定）。
- `_vf` 函数体的固定结构顺序：空指针检查 → `GM_ADDR → __gm__` 指针转换 → 步进式（grid-stride）主循环 `for (kv_idx = block_index*blockDim.x+threadIdx.x; kv_idx < n; kv_idx += total_thread_num)`。

```cpp
#ifndef ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
#define ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
#include <kernel_operator.h>
#include <cstdint>
#include "../../../include/types.h"
#include "../../../include/utils.h"
// #include "../../../include/score_functor.h"  // 仅 ScoreFunctor 时
namespace npu { namespace hkv {
using namespace AscendC;
constexpr uint32_t THREAD_NUM = 512;
// template<...> __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void {kernel_name}_vf(...) { ... }
}}  // namespace hkv / npu
#endif  // ASCENDC_{KERNEL_NAME_UPPER}_KERNEL_H_
```

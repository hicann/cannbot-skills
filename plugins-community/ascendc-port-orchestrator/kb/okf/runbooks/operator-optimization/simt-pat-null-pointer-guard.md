---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "kernel 入口关键 GM 指针空值保护"
description: "在 SIMT kernel 函数开头对关键 GM_ADDR 指针参数逐一做 nullptr 检查并提前 return，再做任何指针转换或计算。"
confidence: single_run
original_id: SIMT_PATTERNS.md#5-空指针保护
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, null-check, defensive]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
在 kernel 函数开头、进行 `GM_ADDR → __gm__` 指针转换和主循环之前，必须对关键指针参数做空值保护，任一为空则直接 return：

```cpp
// 示例：find_and_update_kernel_vf
if (buckets_gm == nullptr) return;
if (keys_gm == nullptr) return;
if (value_ptrs_gm == nullptr) return;
```

检查对象是关键的输入/输出 GM 指针（如 buckets、keys、value_ptrs）。这是 kernel 体结构的第一步（空指针检查 → 指针转换 → 主循环），避免对空地址做 `reinterpret_cast` 后解引用。

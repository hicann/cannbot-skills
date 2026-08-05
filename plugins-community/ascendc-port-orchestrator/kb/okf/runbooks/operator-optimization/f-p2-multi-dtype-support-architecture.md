---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-dtype support architecture"
description: "Templated kernel + per-type dispatcher: cpp template <typename T> __simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... } extern \"C\" __global__ __aicore__ void kernel_fp32(...) { Sim"
severity: low
confidence: single_run
original_id: F-P2
timestamp_inferred: true
tags: [precision, optimization, f-p2, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Templated kernel + per-type dispatcher:
```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... }

extern "C" __global__ __aicore__ void kernel_fp32(...) {
    Simt::VF_CALL<kernel_vf<float>>(Simt::Dim3{threads}, ...);
}
// Same for fp16, bf16
```

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P2，convert_patterns_to_okf.py）。confidence 未升格。 -->

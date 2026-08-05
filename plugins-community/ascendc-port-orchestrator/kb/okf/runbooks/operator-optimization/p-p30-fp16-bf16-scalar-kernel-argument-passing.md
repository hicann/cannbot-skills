---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "fp16/bf16 Scalar Kernel Argument Passing"
description: "Problem: extern \"C\" __global__ __aicore__ kernel entry cannot directly take half/bfloat16_t scalar parameters. The ABI does not support this and it causes value corruption or undefined behaviour. Anti"
severity: high
confidence: single_run
original_id: P-P30
timestamp_inferred: true
tags: [platform_compat, optimization, half, bfloat16_t, p-p30, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: `extern "C" __global__ __aicore__` kernel entry cannot directly take `half`/`bfloat16_t` scalar parameters. The ABI does not support this and it causes value corruption or undefined behaviour.

**Anti-pattern**:
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, half num, int64_t size) {  // ❌ half cannot cross the extern "C" boundary
```

**Correct pattern** (uint16_t bit-pattern):
```cpp
extern "C" __global__ __aicore__ void init_kernel_fp16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  half num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;  // rebuild from bit pattern
  // ... use num ...
}

// Same idea for bf16:
extern "C" __global__ __aicore__ void init_kernel_bf16(
    GM_ADDR data, uint16_t num_bits, int64_t size) {
  bfloat16_t num;
  *reinterpret_cast<uint16_t*>(&num) = num_bits;
  // ...
}
```

**Host-side call**:
```cpp
half h_val = ...;
uint16_t bits = *reinterpret_cast<uint16_t*>(&h_val);
aclrtlaunch_init_kernel_fp16(..., bits, size);
```

**Trigger condition**: Any kernel with a scalar parameter that is not float/int/int64_t (especially initial-value parameters of init/fill kernels).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P30，convert_patterns_to_okf.py）。confidence 未升格。 -->

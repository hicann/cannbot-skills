---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Radix Sort Twiddle Transform for Float/Signed Types"
description: "Problem: Radix sort sorts by bit pattern. IEEE 754 float bit patterns do not directly reflect numeric magnitude (negative bit patterns are \"larger\" than positive ones). Twiddle transformation rules: -"
confidence: single_run
original_id: P-P44
timestamp_inferred: true
tags: [sort, optimization, p-p44, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: Radix sort sorts by bit pattern. IEEE 754 float bit patterns do not directly reflect numeric magnitude (negative bit patterns are "larger" than positive ones).

**Twiddle transformation rules**:
- **Unsigned int**: no transform needed
- **Signed int**: XOR sign bit mask (`0x80000000` for int32)
- **Float**: positive XOR `0x80000000` (flip sign bit); negative XOR `0xFFFFFFFF` (flip all bits)
- **Descending**: additional NOT of all bits after the transform

```cpp
// Vectorized implementation:
And(signBits, data, 0x80000000);         // extract sign bit
CompareNE(mask, signBits, 0);             // mask for negatives
Select(xorMask, mask, 0xFFFFFFFF, 0x80000000);  // pick XOR mask
Xor(twiddled, data, xorMask);            // apply transform
```

**Evidence**: CANN sort_radix_sort_more_core.h:314-562. E1 level.

**Stop condition**: Only needed for radix sort. The hardware Sort API (Concat+Sort+Extract) handles this internally.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P44，convert_patterns_to_okf.py）。confidence 未升格。 -->

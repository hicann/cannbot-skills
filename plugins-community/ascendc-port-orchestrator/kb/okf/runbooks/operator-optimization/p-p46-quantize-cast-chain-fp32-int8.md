---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Quantize Cast Chain (fp32 → int8)"
description: "Problem: AscendC has no direct fp32->int8 Cast instruction. A multi-step conversion is required. Cast chain: cpp // 1. fp32 -> int32 (round to nearest) Cast(int32Local, fp32Local, CAST_RINT, count); /"
confidence: single_run
original_id: P-P46
timestamp_inferred: true
tags: [reduction_quant, optimization, p-p46, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: AscendC has no direct fp32->int8 Cast instruction. A multi-step conversion is required.

**Cast chain**:
```cpp
// 1. fp32 -> int32 (round to nearest)
Cast(int32Local, fp32Local, CAST_RINT, count);

// 2. Set dequant scale (hardware requirement)
SetDeqScale(half(1.0f));  // scale=1.0 means pure type conversion

// 3. int32 -> fp16 (use hardware deq path)
Cast(fp16Local, int32Local, CAST_NONE, count);

// 4. fp16 -> int8 (truncate, auto-clip to [-128, 127])
Cast(int8Local, fp16Local, CAST_TRUNC, count);
```

**arch35 VF register variant** (more efficient):
```cpp
Truncate<float, CAST_RINT>(fp32Tmp, fp32Src);  // in-register round
Cast<half, float>(fp16, fp32Tmp);
Cast<int8, half>(int8, fp16);
```

**Precondition**: Input must already be scaled into the [-127, 127] range. Values outside the range are truncated by TRUNC.

**Evidence**: CANN add_rms_norm_dynamic_quant_helper.h:179-190, regbase_common.h:229-232. E1 level.

**Stop condition**: Cast chain differs when outputting int4 or fp8. fp8 (E4M3) uses Cast(RINT) directly from fp16.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/reduction_quant.md（P-P46，convert_patterns_to_okf.py）。confidence 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMD bf16 mixed precision (MicroAPI)"
description: "High-level Cast() does not support bf16↔float. Use register-level MicroAPI Cast: cpp __VEC_SCOPE__ { RegTensor<bfloat16_t> vreg_bf16; RegTensor<float> vreg_f32; MaskReg preg; AscendC::MicroAPI::DataCo"
severity: medium
confidence: single_run
original_id: F-P3
timestamp_inferred: true
tags: [precision, optimization, muls, add, f-p3, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

High-level `Cast()` does not support bf16↔float. Use register-level MicroAPI Cast:
```cpp
__VEC_SCOPE__ {
    RegTensor<bfloat16_t> vreg_bf16;
    RegTensor<float> vreg_f32;
    MaskReg preg;
    AscendC::MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vreg_bf16, ub_addr);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTrait>(vreg_f32, vreg_bf16, preg);
    // ... float computation ...
}
```

**Simplified alternative**: when precision permits, accumulate directly in bf16 (`Muls` + `Add` natively support bf16).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P3，convert_patterns_to_okf.py）。confidence 未升格。 -->

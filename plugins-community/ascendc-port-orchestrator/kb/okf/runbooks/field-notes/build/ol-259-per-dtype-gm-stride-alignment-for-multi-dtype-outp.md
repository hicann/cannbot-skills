---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Per-dtype GM stride alignment for multi-dtype output tensors — each output uses its own dtype's alignment unit, not a uniform stride"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any kernel emitting multiple output tensors with different dtypes)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any kernel emitting multiple output tensors with different dtypes)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-259
timestamp_inferred: true
tags: [x_out, ascendc, ol-259]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (any kernel emitting multiple output tensors with different dtypes)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`

**Principle**: when a single kernel writes multiple output tensors whose dtypes differ (e.g., fp32 `x_out` and int8 `y1`), each output's GM row stride must use its OWN dtype's alignment unit. fp32: `AlignFp32(N)` (8-element blocks). int8: `AlignInt8(N)` (32-element blocks). Using uniform alignment causes cross-row data corruption in multi-row kernels.

**Concrete anchor**:
```cpp
int32_t stride_x_out = AlignFp32(N);   // fp32: 8-element blocks
int32_t stride_y1    = AlignInt8(N);   // int8: 32-element blocks
DataCopy(gm_x_out[row * stride_x_out], x_out_local, N);
DataCopy(gm_y1[row * stride_y1], y1_local, n_al_quant);
```

**When this fires**: multi-row + multi-dtype + at least one output dtype differs from the alignment unit used in the kernel's row-stride calculation.

**Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): int8 row stride computed with AlignFp32 instead of AlignInt8 caused multi-row failures. Fix → 196/196 PASS_T1.

**Cross-ref**: PB-50 (int8 Cast tail-drop — same quantization surface, different bug), OL-167 (DataCopy count truncation).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-259（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

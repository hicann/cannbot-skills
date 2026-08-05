---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bf16 Scalar Cast Failure"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "static_cast<float>(bf16_var) produces wrong values in scalar context"
confidence: single_run
original_id: PB-4
timestamp_inferred: true
tags: [ascendc, pb-4]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Symptom**: `static_cast<float>(bf16_var)` produces wrong values in scalar context
- **Affected**: bisheng 2026-03-21 (CANN 9.0.0)
- **Workaround**: Use bit-manipulation helpers (`bf16_scalar_to_float`, `simt_to_float`, `simt_from_float`) OR SIMD `Cast()` intrinsic
- **Status**: OPEN
- **Evidence**: `tests/repro/bf16_cast_repro.cpp` (7 test cases), P-P27 pattern
  - 7_MoeGatingTopKSoftmax Phase C iter 1 (2026-04-17): `static_cast<bfloat16_t>(float)` caused compile error; fixed via SIMD Cast through fp32 scratch buffer (same workaround)
  - 14_AdaptiveInstanceNormalization2DBackward kw-1 iter 1 (2026-05-03): `(bfloat16_t)gw_partial` C-style cast in scalar emit context — bisheng "not support bf16 type cast". Fixed via `EmitScalarFromFloat` helper using SIMD Cast tensor-based path (1-element local tensor → Cast → GetValue). Confirms scalar bf16 cast remains broken on bisheng 2026-03-21 / CANN 9.0.0.
  - op#28 MultimodalRopePositionComputationWithGridBasedIndexing (2026-04-22): bit-manip helpers `simt_to_float<bfloat16_t>` (shift-by-16) and `simt_from_float<bfloat16_t>` (explicit IEEE RNE) compiled and ran cleanly inside `__simt_vf__` functions; PB-4 workaround is the durable path for bf16 scalar conversion in pure-SIMT contexts.
- **Detail**: Scalar bf16→float cast emits wrong instruction sequence. SIMD Cast() with `RoundMode::CAST_NONE` works fine.

<!-- 迁移自 porter kb/target/ascendc/（PB-4，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

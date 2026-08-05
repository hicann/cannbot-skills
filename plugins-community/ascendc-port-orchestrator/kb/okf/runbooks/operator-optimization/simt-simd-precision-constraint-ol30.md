---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "OL-30: SIMD optimization must not degrade group/block precision"
description: "A SIMD speedup that changes an op's group/block precision semantics (per-32 shared exponent gone tile-wide) is a precision bug, not production-usable; keep per-group semantics or label approximate."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#precision-constraint-ol-30
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, ol-30, mxfp4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Hard constraint on any SIMT→SIMD (or SIMD-favoring) optimization: **SIMD performance must never be
bought with a precision downgrade.** The common trap is converting a per-group operation into a
tile-wide operation to make it vectorizable — this silently changes the algorithm's precision
semantics.

Confirmed failures of this trap:
- **MXFP4**: letting 1024 elements share one exponent when the spec requires one exponent per 32
  elements → precision bug.
- **A3 hand-written SIMD**: `BATCH=512` sharing an exponent → precision bug (confirmed).
- **Counter-example (safe)**: SG forward is per-token — each token is independent, so vectorizing it
  introduces no precision issue.

**Rule**: if an optimization changes the op's group/block precision semantics, it cannot ship as a
production kernel. Either preserve the per-group semantics, or explicitly label the fast path as
"approximate" and also provide an exact version. A faster-but-wrong kernel is not an acceptable
outcome of a SIMD conversion.

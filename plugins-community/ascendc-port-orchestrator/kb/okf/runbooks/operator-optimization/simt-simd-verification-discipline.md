---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "P-P9 verification discipline: A/B benchmark the SIMT/SIMD choice"
description: "After choosing SIMT vs SIMD, A/B benchmark on the same NPU and session (OL-27) to confirm; SIMD must not lower precision (OL-30); if SIMD is slower than SIMT, just use SIMT and stop optimizing."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#pattern-p-p9-summary
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, benchmark, p-p9, verification]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

The P-P9 defaults and the non-negotiable verification steps after you pick SIMT vs SIMD.

**Defaults**:
- atomicAdd / indirect addressing → **SIMT**.
- group-local dependency with `group < 256` → **SIMT**.
- contiguous read + pure vector compute + no group dependency → **SIMD**.

**Verification is mandatory** — the pick is a hypothesis, not a conclusion:
- **A/B benchmark to confirm** the choice, measured on the **same NPU and same session** (OL-27) so
  the two variants are comparable.
- **SIMD must not reduce precision** (OL-30) — a faster-but-wrong SIMD path is disqualified.
- **If SIMD is slower than SIMT, just use SIMT** — do not keep sinking effort into optimizing the
  SIMD variant.

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMD performance optimization must not come at the cost of precision"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "SIMD kernel optimization attempts, especially eliminating per-group/per-block loops"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-30
timestamp_inferred: true
tags: [ascendc, ol-30]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process, trust_calibration
- **Loaded by**: Builder, Optimizer, QA
- **Trigger**: SIMD kernel optimization attempts, especially eliminating per-group/per-block loops
- **Lesson**: MXFP4 SIMD V4 "fast" replaced per-32-group exponent with a tile-wide shared exponent (1024 elements shared).
  Performance did improve (small tensors 1.08x faster than SIMT), but **precision no longer matches MXFP4 spec**.
  A3's hand-written SIMD has the same issue — BATCH=512 sharing one exponent is exactly the root cause of its precision bug.
  **Never publish a "faster but numerically wrong" kernel as production**.
  **Hard rules**:
  1. After any performance optimization, must compare against PyTorch spec precision (not only against the previous AscendC version)
  2. If an optimization requires changing algorithmic precision semantics (e.g. enlarging group_size), must be **explicitly labeled "approximate"** in docs
  3. "0 mismatch precision" is only meaningful when compared against the PyTorch spec. Comparing against our own CPU ref does not count (our ref may have the same bug)
  4. The A3 implementation's precision bug is precisely because per-group precision was abandoned for SIMD performance — this is not an acceptable trade-off
- **Evidence**: MXFP4 SIMD V4 (2026-04-07), A3 hand-written SIMD precision bug analysis

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-30（category=process, trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

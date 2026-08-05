---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`Select(dst, mask, src0, src1)` mask polarity — mask bit=1 → src0 (NOT src1)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel precision all-fail or systemic mismatch. Worker wrote a \"drop-mask\" (e.g. Compares(mask, v, threshold, LT) → mask=1 where value < threshold, i.e. where p"
confidence: single_run
original_id: EC-31
timestamp_inferred: true
tags: [ascendc, ec-31]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel precision all-fail or systemic mismatch. Worker wrote a "drop-mask" (e.g. `Compares(mask, v, threshold, LT)` → mask=1 where value < threshold, i.e. where position should be DROPPED), then `Select(dst, mask, workBuf, NEG_INF)`, assuming mask=1 means "use workBuf (keep)" and mask=0 means "use NEG_INF (drop)". Actual behavior inverts this: mask=1 picks src0 (workBuf) for the drop positions → keeps what should be dropped and vice versa.
- **Root cause**: `Select` semantics are documented as "mask bit=1 → pick src0, mask bit=0 → pick src1". Worker wrote mask as "drop-bit is 1" but treated Select as "keep-bit is 1". Semantic mismatch.
- **Fix**: ALWAYS build a positive **keep-mask**: the mask bit is 1 where the element should be KEPT (i.e. written to dst as src0), and 0 where it should be filled with sentinel (src1). For "keep v >= threshold", use `Compares(..., GE)` not `LT`. For complex conditions with ties, construct: `keep = (v >= threshold) AND (v > cutoff OR (v == cutoff AND idx > cutoff_idx))` using `Compares` + `And` + `Or`. Then `Select(dst, keep_mask, v, NEG_INF)`.
- **Alternative (safer)**: Use scalar `SetValue` per kept position in an iteration — no mask polarity ambiguity. Slower but bug-resistant. Prefer this for complex per-column emit logic until the mask-construction approach is well-tested.
- **Detection**: If a kernel's "Phase 4 emit" or similar produces 0/N PASS with all-positions-flipped signature (kept and dropped positions swapped), check mask polarity first. This is distinguishable from EC-28 (-inf sentinel value) because the count of mismatched positions is typically 50% of kept or 50% of all — not the "few boundary positions" signature of tie-break bugs.
- **Evidence**: 9_TopKTopP cold-run Phase D iter 1 (2026-04-18). Worker had `mask = (v < threshold)` + `Select(dst, mask, workBuf, -inf)` intending mask=1 = drop → selected workBuf for drops, -inf for keeps. Fix: rebuilt as keep-mask. iter went 0/50 → closed multi-bug chain.

<!-- 迁移自 porter kb/target/ascendc/（EC-31，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Edge-case discovery methodology (quantization ops)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "any op involving bit manipulation, quantization, or fixed-point arithmetic"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-29
timestamp_inferred: true
tags: [ascendc, ol-29]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process, algorithm_selection
- **Loaded by**: Analyzer, QA
- **Trigger**: any op involving bit manipulation, quantization, or fixed-point arithmetic
- **Lesson**: During MXFP4 migration, several classes of edge cases were found:
  1. **Shared exponent extreme-value divergence**: having 1e+38 and 1.0 in the same group causes 1.0 to underflow to 0 (correct behavior but needs verification)
  2. **Integer overflow**: `1 << expdiff` when expdiff > 30 is C++ UB (source produces 0, CPU produces inf)
  3. **Rounding boundary**: behavior of round-to-nearest-even at 0.5 differs between implementations (source bit ops vs PyTorch floor+0.5)

  **Edge-case discovery method**:
  - Extreme-value combinations: mix extreme and normal values in the same group/block
  - Integer overflow: find all `<<` and `>>` ops, check the upper bound on shift amount
  - Rounding boundary: construct values exactly on the quantization boundary (e.g. MXFP4's 1.5 × 2^exp boundary)
  - Zero and subnormals: 0.0, -0.0, smallest positive subnormal
  - Saturation: max representable value + 1 ULP
  - Three-way comparison: PyTorch (spec) vs source vs AscendC; any pair disagreeing is a bug
- **Evidence**: MXFP4 session (2026-04-07)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-29（category=process, algorithm_selection，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

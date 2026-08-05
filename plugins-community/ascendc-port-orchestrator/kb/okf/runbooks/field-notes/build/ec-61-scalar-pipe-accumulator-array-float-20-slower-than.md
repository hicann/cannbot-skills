---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Scalar-pipe accumulator array (float[]) ~20× slower than VEC-pipe accumulator (TBuf<VECCALC>) for per-element reduction ops on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=reduction (per-element accumulator)"
phenomenon: build_failure
signal:
  - "Kernel produces correct output but perf ratio < 0.1×. aiv_vec_ratio near 0, high scalar pipe utilization."
confidence: single_run
original_id: EC-61
timestamp_inferred: true
tags: [aiv_vec_ratio, ascendc, ec-61]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5; op_class=reduction (per-element accumulator)`

- **Severity**: PERFORMANCE (kernel works correctly but 20-50× slower than achievable)
- **Status**: CONFIRMED 2026-05-22 27_MaxPool3d a3-ds (0.047× perf ratio, scalar pipe bottleneck)
- **Symptom**: Kernel produces correct output but perf ratio < 0.1×. `aiv_vec_ratio` near 0, high scalar pipe utilization.
- **Root cause**: `float acc[N]` (S-pipe scalar array). Every `acc[i] = val` is a scalar store (1 element/cycle) vs VEC pipe (8+ elements/cycle for fp32).
- **Fix**: Replace `float acc_[TILE_W]` with `TBuf<TPosition::VECCALC> accBuf_`. Use VEC `Duplicate` for init, `Max` for accumulation, `Cast` + direct `DataCopy` for output.
- **Evidence**: 27_MaxPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: P-P47 (VEC halving for reductions); OL-161 (V220 SIMD UB element duplication).

<!-- 迁移自 porter kb/target/ascendc/（EC-61，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

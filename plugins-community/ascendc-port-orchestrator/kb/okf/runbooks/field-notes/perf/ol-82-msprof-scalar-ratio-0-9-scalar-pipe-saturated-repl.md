---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof scalar_ratio > 0.9 = scalar pipe saturated → replace with VEC instructions"
description: "the scalar pipe (S pipe) processes 1 element per issue; the VEC pipe processes 64-256 elements per cycle. When scalar_ratio > 0.9, the kernel is almost entirely stuck on the S pipe. Usually the bottle"
phenomenon: perf_regression
signal:
  - "msprof shows `aiv_scalar_ratio > 0.9` on the worst/dominant case"
confidence: single_run
original_id: OL-82
timestamp_inferred: true
tags: [compare, comparescalar, select, datacopy, ascendc, performance, ol-82]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
msprof shows `aiv_scalar_ratio > 0.9` on the worst/dominant case

## 教训 / 根因
the scalar pipe (S pipe) processes 1 element per issue; the VEC pipe processes 64-256 elements per cycle. When scalar_ratio > 0.9, the kernel is almost entirely stuck on the S pipe. Usually the bottleneck is a per-element loop (`for i in N: buf.GetValue(i)` / `buf.SetValue(i, v)`). **Replace with VEC instructions first**:
  - scalar insertion sort / scalar selection topk → SIMD `ReduceMax/ReduceMin(calcIndex=true)` × k iterations + mask-out
  - scalar accumulate / product → `ReduceSum/BlockReduceSum`
  - scalar clamp/activation → SIMD `Muls/Adds/Max/Min`
  - scalar compare → `Compare` / `CompareScalar` + `Select`
  - GM scalar read (`gm.GetValue`) → SIMD `DataCopy` bulk load

## 证据
- 7_MoeGatingTopKSoftmax (2026-04-17): scalar insertion topk over N=2048..7168 → `scalar_ratio=0.975` on worst cases 17/47. Switched to SIMD `ReduceMax<float>(calcIndex=true)` × k + `SetValue(idx, -inf)` mask-out → `scalar_ratio 0.975 → 0.271`, `vec_ratio 0.017 → 0.679`, sum-ratio `0.142x → 1.097x` (+673%)
  - 1_RotaryMul (2026-04-23): `rotarymul_kernel_bf16_interleave` avg_dur=956us (173x slower than CANN ref `RotaryPositionEmbedding` avg_dur=5.5us), `aiv_scalar_ratio=0.971`, `aiv_vec_ratio=0.012`. Pattern: scalar even/odd permute loop `for j<DHALF: GetValue(2j); GetValue(2j+1); SetValue(-oddVal); SetValue(evenVal)`. **Kind-2 fix template**: strided CopyIn to SoA layout (even-array, odd-array separately), then fully vectorized Muls/Mul/Add on SoA halves, re-interleave on CopyOut. Directive archived at `output/npukernelbench/src/kernels/1_RotaryMul/optimization_directive.md`. Worker respawn for Kind-2 blocked by firewall this session → DEBT-037.
  - Cross-ref: MSPROF_AGENT_GUIDE.md line 243 (scalar_ratio > 0.2 warning) — OL-82 is a strengthened criterion (>0.9 = mandatory vectorization)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-82（category=performance，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->

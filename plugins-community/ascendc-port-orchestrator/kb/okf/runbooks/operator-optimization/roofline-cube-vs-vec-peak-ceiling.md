---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Pick CUBE peak for matmul/attention, VEC peak for elementwise/reduction"
description: "Use the CUBE ceiling (~373 TFLOPS fp16) for Matmul/Mmad-dominated ops and the VEC ceiling (~56 TFLOPS fp16) for elementwise/reduction; conflating them understates ~6.7x."
confidence: single_run
original_id: ROOFLINE_MODEL.md#cube-vs-vec-peak-do-not-conflate
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, cube, vec, peak-tflops]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Ascend950PR (A5) has two compute units with very different peaks. Pick the ceiling that matches the op's dominant instruction:

- **CUBE unit** (Matmul / Mmad) — matmul, FlashAttention, any op dominated by `Matmul`/`Mmad`.
  - FP16 ~373 TFLOPS, BF16 ~368 TFLOPS (measured 2026-06-12, .171 NPU1 957b, torch.matmul 8192³).
  - FP32 ~24 TFLOPS — cube is fp16/bf16-optimized, fp32 not favored.
- **VEC unit** (AI Vector Cores) — elementwise, reduction, softmax.
  - FP32 ~28 TFLOPS (56 cores × 512 FLOPS/cycle × 1 GHz), FP16 ~56 TFLOPS (2x via half-precision packing).

**Why it matters:** using the VEC peak (~56 TFLOPS fp16) as the ceiling for a cube-bound op understates it ~6.7x. An op running at 24% of its real cube ceiling then looks like 80%+ "efficient", and the perf gate wrongly stops optimizing.

**API:** `roofline_eval.py` encodes this split via `peak_cube_*` fields + `_peak_tflops(op_type)`; the ridge point is computed per-unit.

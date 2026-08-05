---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "port_a3 complexity routing — algorithmic class (not surface LOC/matmul/CrossCore count) decides L4; flat_quant anchors the L2/L3-shippable band"
description: "Surface metrics (2000+ LOC, cube+vec, CrossCoreSetFlag, matmul stages) are necessary-not-sufficient for L4; only forward softmax/attention is L4. flat_quant is the L2/L3 calibration anchor."
confidence: single_run
original_id: OL-185
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-185, port-a3, l4-routing]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
When a port_a3 op superficially matches FA-class L4 (high LOC, cube+vec pipeline, many CrossCoreSetFlag sync sites, matmul stages) but its **algorithmic semantic is NOT softmax/attention** (quantization, GEMM-reduce, fused-elementwise, or the backward/gradient of a non-FA op), the correct calibration anchor is `flat_quant`, **not** OL-159's FA class.

**Calibration data** — `flat_quant` shipped in a single cold-start (9 kw spawns + 1 researcher + 1 probe) with these characteristics:
- 2014 LOC across 8 V220 source files (`op_kernel/*.h` + `apt.cpp`)
- 8+ `CrossCoreSetFlag` / `matmul::Matmul` / `MatmulImpl` call-sites across multiple files
- TILING_KEY=1 path: hand-rolled `FlatQuantVec` (AIV) + `FlatQuantCube` (AIC) cube primitives, `MM_BASE_MODE`, NO `REGIST_MATMUL_OBJ`
- ~150-line `kernels.cpp` dispatcher hand-authored by the worker

Verified on soc=Ascend950PR_9579 (V351), cann=9.0.0, bisheng=15.0.5; terminal=done, precision=PASS_WITHIN_TOLERANCE (tier1 8/8 inclusive), perf=1.63× A3-CANN (2026-05-22 kw-9).

**Anti-pattern (caught on lightning_indexer_grad, 2026-05-23)**: a worker reads OL-159 (FA-class template-assembly requirement) and reflexively classifies ANY op with similar surface-LOC + CrossCoreSetFlag count as FA-class L4. **Surface metrics (LOC, matmul stages, CrossCore count) are necessary-not-sufficient for L4 routing.** The load-bearing axis is **algorithmic class**: LIG_grad (2549 LOC, 9 files, 8 CrossCoreSetFlag, `service_cube.h` matmul, sub-ops gemm+gather+scatter+reduce+relu_grad) contains **NO softmax/attention** — it is a backward op → L2/L3, not L4.

**Decision rule** (replaces OL-159's broad surface trigger):
1. **Classify as L4 IFF**: forward softmax/attention with Q×K@V tile-scheduling, online softmax (running max/sum), causal/sliding-window mask handling. Op-name pattern `flash_attention*`, `incre_flash_attention`, `mha_*` forward, `attention_score_*` (NOT `*_grad`).
2. **Treat as L2/L3** (single kw spawn, may need bump-cap) when the op is ≥2000 LOC + cube+vec + cross-core BUT is a `*_grad` / `*_backward`, or otherwise lacks softmax/attention semantics (quant, GEMM-reduce, fused-elementwise).

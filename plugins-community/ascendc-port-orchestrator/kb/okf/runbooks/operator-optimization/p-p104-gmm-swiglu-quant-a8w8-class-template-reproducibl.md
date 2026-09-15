---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "GMM SwiGLU Quant A8W8-class template — reproducible A5 cube-MIX grouped-matmul-swiglu-quant framework (recipe phase-order + host/kernel tiling + stitching spec-map + 12-variant X-macro)"
description: "For any GMM SwiGLU Quant A8W8 op on A5 (grouped-matmul-swiglu-quant family). The worked template: MIX_AIC_1_2 Cube↔Vector pipeline with CrossCoreSetFlag/WaitFlag(0x8/0x9) handshake + Backpressure dept"
severity: high
confidence: single_run
original_id: P-P104
timestamp_inferred: true
tags: [grouped_matmul, optimization, p-p104, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For any GMM SwiGLU Quant A8W8 op on A5 (grouped-matmul-swiglu-quant family). The worked template: MIX_AIC_1_2 Cube↔Vector pipeline with CrossCoreSetFlag/WaitFlag(0x8/0x9) handshake + Backpressure depth=14 + BasicBlock global indexing + ProcessDSQ fused dequant-swiglu-quant body (manual SiLU⊙Gate 5-op + per-token WholeReduceMax quant). Spec-map: 3 dequantDtype × 2 wFormat × 2 transB = 12 kernel variants via X-macro, 3-in-lockstep. Host tiling: CalcBasicBlock(baseM=128/N=256/K=128) + CalcUBFactorDimX(N→{1,2,4}) + CalcWorkspaceSize(M*N*4+20MB) + SelectA8W8Launcher dispatch. K2 invariant: host CalcBasicBlock ↔ kernel template params 1:1. Meta-lessons: V1 split/unsplit dead weight, cross-core sync arch35-specific, SwiGLU formula canonical SiLU⊙Gate, quant uses 1/127 multiply not divide (CAND-PP103). Cross-ref: **P-P102** (cube-MIX scaffold), **P-P70** (fused dequant→activation→quant pipeline), **CAND-V351-AIV-WholeReduceMax-fp32-mask-cap** (fp32 mask=64 cap), **CAND-PP102** (two-kernel split broken on V351). Full body: `patterns/domains/gmm_swiglu_quant_a8w8_class_template.md`. `applies_to: soc=Ascend950PR/V351; cann=9.0.0; op_class=grouped-matmul-swiglu-quant/CUBE_MIX (A8W8 path)`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P104，convert_patterns_to_okf.py）。confidence 未升格。 -->

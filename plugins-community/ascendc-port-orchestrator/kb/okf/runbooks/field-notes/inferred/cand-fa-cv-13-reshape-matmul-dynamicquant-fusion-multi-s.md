---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Reshape + Matmul + DynamicQuant fusion — multi-stage pipeline with L1-resident intermediates"
description: "verified_on: cv-agent reshape_matmul_rowwise_quant_int8 kernel (reshape view → mm → dynamic_quant, all in one kernel) Pattern: Fuse reshape (zero-copy view), matmul, and dynamic quantization into a si"
phenomenon: build_failure
signal:
  - "verified_on: cv-agent reshape_matmul_rowwise_quant_int8 kernel (reshape view → mm → dynamic_quant, all in one kernel)"
confidence: inferred
status: stub
original_id: CAND-FA-CV-13
timestamp_inferred: true
tags: [candidate, inferred, reshape, reshapematmulquantkernel, cand-fa-cv-13]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent reshape_matmul_rowwise_quant_int8 kernel (reshape view → mm → dynamic_quant, all in one kernel)`

**Pattern**: Fuse reshape (zero-copy view), matmul, and dynamic quantization into a single AscendC kernel. The reshape is free (reinterpret strides), the matmul output stays in L1, and the VEC quant stage reads directly from L1. Two intermediates (reshaped view, matmul result) never touch GM.

**Pipeline**:
1. Reshape: x(m,n) → x_view(m*n/k, k) — zero-copy, just reinterpret strides
2. Matmul: x_view @ h(k,k) → result(m*n/k, k) in L1
3. DynamicQuant: row-wise max_abs → scale → round + clip → int8 output
4. Reshape back: int8 result → (m, n) output

**Why fusion matters**: Without fusion, the matmul output (fp16/bf16, size m*n) must be written to GM, then read back by a separate quant kernel. For m=4096, n=2048: 16MB write + 16MB read avoided.

**L1 budget constraint**: The matmul output buffer + quant workspace must fit in L1 simultaneously. This constrains the inner dimension K to K_max = L1_size / (element_size * 2). If K exceeds this, the pipeline must split into tiles.

**Detection**: grep for `reshape` AND `Matmul\|GEMM` in the same kernel .cpp. If reshape+matmul are separate ops → gap. If in same kernel → fusion applied.

**Evidence**: cv-agent `reshape_matmul_quant/` kernel with `ReshapeMatmulQuantKernel` class — single Process() body with reshape→matmul→quant stages.

**Cross-ref**: CAND-FA-CV-8 (fused matmul+elementwise — same principle, different tail stage: quant vs activation), CAND-FA-CV-11 (INT8 quant matmul — this pattern adds reshape fusion before matmul), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — L1 workspace management generalizes to reshape+matmul+quant pipeline)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-13，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

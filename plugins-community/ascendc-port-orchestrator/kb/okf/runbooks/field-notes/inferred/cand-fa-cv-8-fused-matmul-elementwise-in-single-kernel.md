---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Fused matmul + elementwise in single kernel — L1-resident intermediate, no GM round-trip"
description: "verified_on: cv-agent matmul_leakyrelu kernel (Cube matmul → L1 resident → VEC leaky_relu → GM output) Pattern: When an op is matmul followed by elementwise activation (GELU, ReLU, LeakyReLU, SiLU, ad"
phenomenon: build_failure
signal:
  - "verified_on: cv-agent matmul_leakyrelu kernel (Cube matmul → L1 resident → VEC leaky_relu → GM output)"
confidence: inferred
status: stub
original_id: CAND-FA-CV-8
timestamp_inferred: true
tags: [candidate, inferred, mmoutbuf_, cand-fa-cv-8]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent matmul_leakyrelu kernel (Cube matmul → L1 resident → VEC leaky_relu → GM output)`

**Pattern**: When an op is matmul followed by elementwise activation (GELU, ReLU, LeakyReLU, SiLU, add, mul), fuse both into a single AscendC kernel. The matmul output stays in L1 buffer and is immediately consumed by the VEC elementwise stage — zero GM round-trip for the intermediate tensor.

**Mechanism**:
1. Cube unit computes matmul output into L1 workspace tensor (not GM)
2. VEC unit reads from the SAME L1 buffer (no DataCopy to GM, no DataCopy back)
3. VEC applies elementwise activation (Mul+Max+Mul for LeakyReLU, etc.)
4. VEC writes final result to GM output tensor

**Buffer lifecycle**:
```
[GM A] → L1 buf_a → Cube → L1 buf_c (matmul output, resident)
                                   ↓
                              VEC reads buf_c
                                   ↓
                              VEC applies activation
                                   ↓
                              [GM Y] ← VEC writes buf_c (now = activated output)
```

**Savings**: Eliminates one GM write (matmul output) + one GM read (activation input). For matmul with M=N=K=4096 fp16: saves 64MB GM write + 64MB GM read per invocation.

**Detection**: grep for `Matmul.*IterateAll` and `Activation|Relu|Gelu|Silu|LeakyRelu` in the same kernel .cpp. If matmul output goes to GM (SetGlobalBuffer before activation) → gap. If matmul output stays in L1/TBuf → pattern applied.

**Evidence**: cv-agent `matmul_leakyrelu/kernel/matmul_leakyrelu.h` — single kernel class with `MatmulObj.IterateAll()` writing to L1-local `mmOutBuf_`, immediately followed by `LeakyRelu(mmOutBuf_, yOutQueue_)` in the same Process() body.

**Cross-ref**: CAND-NSA-1 (Matmul<>::IterateAll + local SetFlag for AIV chaining — same L1-resident principle for FA), CAND-FA-CV-1 (ring buffer with WorkspaceQueue — L1 workspace management generalizes to fused matmul)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-8，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

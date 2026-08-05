---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "cross-attention (Sq≠Skv rectangular) backward — ZERO kernel change; dim-generic dense FA-bwd kernels + pure host rectangular remap; no template needed"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382) unverified_on: soc=Ascend950PR (A5/V300) Principle:"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward"
confidence: inferred
status: stub
original_id: CAND-FA-CROSS-BWD-1
timestamp_inferred: true
tags: [candidate, inferred, npu_fusion_attention_grad, cand-fa-cross-bwd-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=attention-backward`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (A3 Ascend910_9382)`
`unverified_on: soc=Ascend950PR (A5/V300)`

**Principle**: Cross-attention backward (Q from sequence length Sq, K/V from Skv, **Sq ≠ Skv**) is the CHEAPEST FA-class-backward variant to generate from a dense FA-bwd kernel — it needs **ZERO kernel change**, only a pure HOST rectangular remap. If the dense FA-bwd kernels are already **dim-generic** (the 5 GEMMs + the softmax/dS VEC stages take m/n/k + strides as per-launch parameters, not hard-coded square S), cross-attention is just: host passes rectangular shapes — q/dO=[G,Sq,D], k/v=[G,Skv,D], S/P/dP/dS=[G,Sq,Skv]; the 5 GEMMs get rectangular m/n/k+strides; softmax/dS reduce over Skv. Same per-pair FA-2 math. Load-bearing lesson: **author the derive-from-forward kernels dim-generic from the start, and rectangular/cross variants come for free.**

**Concrete anchor**: dense FAG GEMM entry `gemm<CT>(..., m, n, k, transA, transB, strideA, strideB, strideC)` already parameterized → cross passes m=Sq, n=Skv (per output) with no kernel edit; softmax row-reduce dim = Skv.

**Evidence**: flash_attention_grad_cross (white-box gqa-bwd-wp1, 2026-06-19, A3/V220 Ascend910_9382), Sq≠Skv cases. Precision vs fp64 cross autograd (cannbot single-judge, 900 records): fp16 PASS 300/300, bf16 PASS 300/300, fp32 T2 dtype-floor (= OL-109). Deterministic 6/6. Perf (P97, vs RAW vendor `npu_fusion_attention_grad` rectangular Sq≠Skv no-mask, lead independently re-ran in-container): median 0.960× (proxy 0.889×; ~8% run-variance, both > 0.6× floor; small cases beat vendor 1.4–1.5×). Out of box, no ko.

**Other instances (predicted)**: any rectangular-S attention backward (encoder-decoder cross-attn, retrieval/memory attention, prefix attention). The dim-generic-kernel lesson transfers to ANY variant whose only change is shape/stride.

**3-variant generalization (with CAND-FA-GQA-BWD-1, CAND-FA-SWA-BWD-2)**: GQA (1.391×), SWA (0.354→0.852× via cheap mask-cache), Cross (0.960×) — NONE of the three FA-class-backward variants needs the forward fused single-launch template to clear precision AND the perf floor. The simple multi-launch cube+vec skeleton (PB-34/35-sidestepping) + cheap non-template host levers generalize across the variants; the heavy fused template is not what generalizes.

**Promote when**: a second rectangular/cross attention backward reproduces "dim-generic dense kernels + host remap, zero kernel change, clears precision + perf-gate". Cross-ref: CAND-FA-GQA-BWD-1, CAND-FA-SWA-BWD-1/2, CAND-FA-MULTI-LAUNCH-PERF-GAP.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CROSS-BWD-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

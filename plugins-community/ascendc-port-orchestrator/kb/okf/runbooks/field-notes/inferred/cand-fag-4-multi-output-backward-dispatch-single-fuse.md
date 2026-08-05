---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Multi-output backward dispatch — single fused bwd kernel produces all primary input gradients (dq, dk, dv) in one pass via shared recompute + per-output cube ladder, NOT three independent backward launches"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=multi_output_backward / fused_attention_backward / any_bwd_op_whose_forward_was_one_fused_kernel_producing_N_outputs derived-from: c"
phenomenon: build_failure
signal:
  - "A fused forward op has N>1 primary inputs (e.g. attention's Q, K, V) and the user-facing bwd interface produces N gradient tensors of the same input shapes. A n"
confidence: inferred
status: stub
original_id: CAND-FAG-4
timestamp_inferred: true
tags: [candidate, inferred, fagtilingtype, processdqkv, flashsoftmaxgrad, cand-fag-4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=multi_output_backward / fused_attention_backward / any_bwd_op_whose_forward_was_one_fused_kernel_producing_N_outputs`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/ — three peer outputs (dq, dk, dv) share one Init signature (`Init(..., dq, dk, dv, dpse, dqRope, dkRope, dsink, workspace, ...)`), one tiling-data struct (`FagTilingType`), and one fp32-workspace plan (`postTilingData.{dq,dk,dv}WorkSpaceOffset`); the main vec-block headers issue a three-stage vec pipeline `ProcessVec2 → ProcessVec3 → ProcessVec4` where the intermediate ds is reused for both dq and dk/dv cube ladders; the post-kernel `ProcessDqkv` iterates `for qkvIdx in {0,1,2}` over the three outputs with one ping-pong loop body`
`unverified_on: a5_ops`

**Trigger**: A fused forward op has N>1 primary inputs (e.g. attention's Q, K, V) and the user-facing bwd interface produces N gradient tensors of the same input shapes. A naive design would launch N separate backward kernels — but the bwd's most expensive shared subwork is (a) recomputing P = softmax(scale·Q·Kᵀ + bias) and (b) computing dP = dY @ Vᵀ then dS = P · (dP - rowsum(P·dP)). dS is the common ancestor of dQ, dK, and dV; computing it once and consuming it three times is the structural win.

**Recommendation**: Build the bwd as **one kernel** that:

1. Takes ALL primary-input GM pointers (q, k, v, dy, plus saved tensors per CAND-FAG-3) AND all gradient-output pointers (dq, dk, dv) in a single Init signature.
2. Internally chains **shared recompute → shared ds → per-output cube ladder**:
   - Stage A (recompute, vec): from saved (max, sum) and Q/K, reconstruct P. CAND-FAG-3 anchor.
   - Stage B (dY @ Vᵀ → dP, cube): one matmul.
   - Stage C (ds = P · (dP - softmaxgrad-correction), vec): the `FlashSoftmaxGrad` step from FAG design doc §1.3 — emits ds shared by Stages D, E, F.
   - Stage D (dQ += ds @ K, cube): writes to dq's fp32 workspace via SetAtomicAdd.
   - Stage E (dK += dsᵀ @ Q, cube): writes to dk's fp32 workspace.
   - Stage F (dV += Pᵀ @ dY, cube): writes to dv's fp32 workspace. Uses P (not ds).
3. Treat the three outputs as **a uniform array indexed by qkvIdx ∈ {0,1,2}** in the POST kernel (CAND-FAG-1) — same ping-pong, same Cast loop, only the per-output scale and write-offset differ. Skip the Muls(scale) for the qkvIdx==2 (dv) branch because dv does NOT inherit the attention scale.

**Concrete anchor** (public AscendC):
```cpp
// Single Init takes all gradient outputs together (NOT three separate kernel signatures)
__aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR dy,
                            GM_ADDR softmaxMax, GM_ADDR softmaxSum,
                            GM_ADDR dq, GM_ADDR dk, GM_ADDR dv,
                            GM_ADDR workspace, FagTilingDataLike *tilingData, TPipe *pipeIn);

// In the per-tile hot loop:
// Stage A — recompute P from saved (max, sum)
ReconstructP_FromSavedStats(pUb, scoresUb, savedMaxRow, savedSumRow, n);
// Stage B — dY @ Vᵀ on AIC, hands ds source to AIV via cube ladder (anchor in CAND-FA1)
IterateMmDpFromDyVt(dpL0c, dyL1, vL1Trans);
// Stage C — ds = P · (dP - rowsum_dot_correction); shared output
ComputeDs(dsUb, pUb, dpUb, n);    // ds is now in UB, ready for three consumers
// Stages D/E/F — three cube matmuls, all using ds (or P), all writing fp32 workspace atomically
SetAtomicAdd<float>();
IterateMmDqFromDsK (dqWorkSpaceGm, dsL1, kL1);       // ds @ K → dq
IterateMmDkFromDsTQ(dkWorkSpaceGm, dsL1Trans, qL1);  // dsᵀ @ Q → dk
IterateMmDvFromPTDy(dvWorkSpaceGm, pL1Trans, dyL1);  // Pᵀ @ dY → dv
SetAtomicNone();

// POST kernel — uniform 3-output cast/scale loop
for (int qkvIdx = 0; qkvIdx < 3; ++qkvIdx) {
    // ... ping-pong DataCopy → (Muls for qkvIdx<2 only) → Cast → DataCopy to dqkv[qkvIdx] ...
}
```

**Why it works**:
- Three separate bwd kernels would each pay (a) saved-tensor copy-in, (b) Q/K/V copy-in, (c) recompute P, (d) compute dP, (e) compute ds — 5 redundancies × 2 = 10x duplicated work. One fused bwd pays each once
- ds is the common gradient ancestor; the algorithm's correctness proof (FlashAttention paper §4.3) is what makes this fusion safe — there is no other intermediate that achieves the same sharing
- Per-output fp32 workspaces decouple the three cube ladders' atomic-add domains — they never contend on the same address, so SetAtomicAdd safety is per-output
- The 3-output POST-kernel uniformity is a code-size and instruction-cache win: one loop body with a per-iter qkvIdx-conditional Muls skip beats three duplicated POST blocks

**Determinism**: NOT deterministic by default — Stages D/E/F use atomic-add across cores. To get a deterministic multi-output bwd, layer CAND-FAG-2 (coordinate-partitioned dispatch) on top — its assignment formula must produce a bijection over (b, n2, g, s1Outer, s2Outer) tiles regardless of which output is being written. The FA-grad reference's deter mode does exactly this: the same coordinate dispatcher serves all three of dq, dk, dv writebacks.

**Other instances predicted**:
- MoE backward: dExpert-weight, dGate-weight, dInput all share an intermediate "routed-input × routed-output-grad" tensor
- Cross-attention backward: dQ_decoder, dK_encoder, dV_encoder share the same ds
- Fused gated-linear-unit (GLU) backward: dGate, dUp, dDown share a ds-equivalent intermediate
- Convolution backward: dW and dInput share the unrolled-input × dY product if expressed as gemm
- LayerNorm + Linear fused backward: dLN-input, dLN-gamma, dLN-beta, dLinear-W share the dLN-output intermediate

**Risks before promotion**:
- The fused-bwd kernel is one of the largest single-source kernels in the FA-grad reference (multi-thousand-line single header); UB / L1 budget pressure scales nonlinearly with output count. For >3 outputs (e.g. attention with bias gradient), the fusion may overflow UB and force per-output spill
- Cube-ladder ordering matters: dQ depends on K (not its grad), dK depends on Q (not its grad), dV depends on P (not its grad) — independent in graph terms, so the L1 reuse policy must keep K, Q, V, dY co-resident across the three ladders. If L1 is too small for all four, the fused kernel degrades to per-output reload — worse than three separate kernels
- The single Init signature accumulates many arguments (12+) — exceeds the "Init args" budget of some host frameworks; may need a packed-args struct on the host side
- The qkvIdx==2 skip-Muls branch is a hardcoded shape assumption (dv doesn't carry attention scale) — if a future variant adds a scale-like factor to dv (e.g. quantization dequant scale), the POST kernel's uniform loop breaks silently

**Cross-reference**:
- CAND-FAG-1 (three-kernel pre/main/post split): COMPOSED with this candidate — the "M" in pre/M/post is exactly the multi-output fused kernel described here. CAND-FAG-1 = "how to write the three outputs"; this candidate = "why one MAIN kernel produces all three"
- CAND-FAG-2 (deterministic coordinate dispatch): COMPOSED — provides the optional determinism layer for the multi-output bwd
- CAND-FAG-3 (saved-tensor restore): COMPOSED — provides Stage A input
- P-P89 (workspace contract for fused ops): same multi-output workspace shape; promote-merge if multi-op evidence accumulates (attention bwd is one op, MoE bwd would be another)
- CAND-FA1 / CAND-FA3 (cross-core sync, slot rotation): orthogonal — both apply within Stage B and Stage D/E/F's cube ladders unchanged

**Promote when**: a5_ops ships a multi-output fused-bwd op (attention bwd, MoE bwd, fused-norm-Linear bwd) with measured perf ≥1.0× vs the "three independent bwd kernels" baseline AND L1/UB budget verified ≥80% utilization (proves the fusion is paying for its complexity).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FAG-4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

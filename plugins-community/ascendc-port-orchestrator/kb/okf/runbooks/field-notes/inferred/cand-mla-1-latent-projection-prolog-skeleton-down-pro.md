---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Latent-projection prolog skeleton — down-project × pre-norm × up-project producing two heads from one normalized intermediate (cube-vec-cube ladder)"
description: "applies_to: any soc with public AscendC Matmul-tile primitives + RmsNorm + cross-core sync; cann=9.0.0+; op_class=latent_attention_prolog / low_rank_dual_projection_prefix / mla_prefix derived-from: c"
phenomenon: build_failure
signal:
  - "Op shape is \"compress hidden dim by a low-rank projection, normalize the latent, then expand into TWO downstream heads (e.g. content and rotary, or up-projectio"
confidence: inferred
status: stub
original_id: CAND-MLA-1
timestamp_inferred: true
tags: [candidate, inferred, gamma, c_norm, crosscoresetflag, crosscorewaitflag, cand-mla-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public AscendC Matmul-tile primitives + RmsNorm + cross-core sync; cann=9.0.0+; op_class=latent_attention_prolog / low_rank_dual_projection_prefix / mla_prefix`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h (top-level Process/AicProcess at ~L755-L880 — two-stage MM_CQ → RMSNORM_CQ → MM_QCQR ladder), attention/mla_prolog/docs/aclnnMlaProlog.md (formula block: c^Q = RmsNorm(x·W_DQ); q^C = c^Q·W_UQ; q^N = q^C·W_UK)`
`unverified_on: a5_ops (no MLA-prolog-class op currently shipped; closest analog is fused norm-then-matmul chains in workspace/3_FusionAttention)`

**Trigger**: Op shape is "compress hidden dim by a low-rank projection, normalize the latent, then expand into TWO downstream heads (e.g. content and rotary, or up-projection plus a second branch)" — characteristic of MLA prolog, low-rank-adapter-style prefixes, factorized-attention prefixes, and certain MoE expert prefixes where one small intermediate feeds multiple downstream matmuls. Hidden dim `He` is large (>=4K), the latent dim `Hc` is small (~1K-2K), and the two downstream up-projections share the latent — meaning the latent must be (a) normalized once, (b) cached in GM or wide-UB, (c) read twice by the up-matmuls without recomputation.

**Recommendation**: Structure as a three-stage cube-vec-cube ladder with the normalized latent as the cross-stage carrier in GM workspace. Stages:
  1. **Down-project (cube)**: `c_pre = x · W_D` where `(B*S, He) @ (He, Hc) -> (B*S, Hc)`. Emit to a GM workspace slot (pre-norm latent buffer). Signal vec.
  2. **Normalize (vec)**: read the pre-norm latent from GM, apply RmsNorm with `gamma` of length `Hc`, write the normalized latent to a second GM workspace slot. Signal cube. (This is the only stage that touches the latent's row state — see CAND-MLA-3 for the per-row rmsnorm shape.)
  3. **Up-project (cube)**: cube reads `c_norm` and performs the dual up-projection. The fused up-matmul has output dim `N*(D+Dr)` where the `N*D` slice is the content head and the `N*Dr` slice is the rope head — a single matmul over a concatenated weight `W_U = [W_UQ | W_QR]` shape `(Hc, N*(D+Dr))`, with the two heads separated post-matmul by offset slicing in the consumer.

The dual-output fusion in step 3 is load-bearing: it amortizes the GM read of `c_norm` (1 read instead of 2) and shares the L1 A-tile across both N partitions. The post-matmul slice into content/rope heads is a cheap address-only operation (no data movement) because the slice is along the N dimension which is the cube's output dim and is already laid out contiguously per head.

The cross-stage handoff between stages 1->2 and 2->3 uses the user-owned cross-core flag protocol from CAND-FA1 (paired AIC/AIV via `CrossCoreSetFlag` / `CrossCoreWaitFlag`); the latent GM slot follows the workspace-slot-rotation discipline of CAND-FA3 when the step is iterated over a batch dimension.

**Concrete anchor** (public-API-only skeleton; orchestrator level — the per-stage matmul body uses the cube-tile-mmad primitives or `Matmul<>` per CAND-FA1 hard-exclusion clause):
```cpp
constexpr uint32_t downDoneId = 1;   // user-owned flag IDs, all <= FFTS_MAX_FLAG (7)
constexpr uint32_t normDoneId = 2;

// Stage 1 (cube) — down-project x by W_D, emit pre-norm latent to GM
// (B*S, He) @ (He, Hc) -> preNormLatentGm[B*S, Hc]
AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(downDoneId);

// Stage 2 (vec) — gate on cube, normalize, emit normalized latent to GM
AscendC::CrossCoreWaitFlag(downDoneId);
RmsNormRow(cNormUb, preNormLatentGm, gammaUb, /*col=*/Hc, epsilon);
DataCopy(normLatentGm[rowOffset], cNormUb, Hc);
AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(normDoneId);

// Stage 3 (cube) — gate on vec, up-project into fused (content | rope) heads
// (B*S, Hc) @ (Hc, N*(D+Dr)) -> upResGm[B*S, N*(D+Dr)]
AscendC::CrossCoreWaitFlag(normDoneId);
// Downstream consumers slice: content = mmUpRes[:, :N*D]; rope = mmUpRes[:, N*D:]
```

**Why it works**:
- The latent dim `Hc` is the smallest tensor on the cross-stage path. Putting the cross-stage hand-off AT the latent (rather than upstream of the down-project or downstream of the up-project) minimizes GM round-trip volume and the working set the vec stage must keep resident.
- Fusing the two up-projections into one matmul along N halves the down-projection's L1 A-tile reuse pressure: the cube reads `c_norm` once, computes the concatenated output tile, and the downstream slice is purely an offset alias.
- The cube-vec-cube ladder gives the cube engine two compute windows per token-group (down + up), interleaved with one vec window (norm), allowing partial overlap when the producer-consumer flags are pipe-tight (PIPE_FIX for cube emit, PIPE_MTE3 for vec emit) and when the workspace slot rotation (CAND-FA3) admits one in-flight generation of overlap.

**Determinism**: The skeleton is deterministic when (a) each row's down-project / norm / up-project is owned by a single AIC/AIV pair (no cross-core writes participate in the latent), (b) RmsNorm uses the row-reduction shape of CAND-MLA-3 with a fixed Cast+Square+Sum order, and (c) the up-projection's K-dim reduction order is fixed by the tiling (same as any deterministic matmul). The fusion of two up-projections into a single matmul does not change any reduction order per output element — the only change is that two output regions are produced in one pass.

**Other instances predicted**:
- MLA-style prolog operators (this verified instance) for inference and training prefixes
- LoRA-style low-rank prefixes (`x · A · B` factorization) where `A` projects to a low rank and `B` projects back — the dual-output extension is when `B` is itself a concatenation
- DeepSeek-V2 / V3 latent attention prefixes that share an MQA-style compressed KV
- MoE prefixes that share a routing pre-projection feeding both gate scores and load-balancing statistics from one normalized latent
- Any "compute embed once, use twice" pattern where the second use is along a different head axis (the up-fusion variant amortizes the embed read)

**Risks before promotion**:
- The dual-output up-projection requires the two consumers' inner-dim layouts to be compatible — if the rope head needs an interleaved-pair layout (CAND-MLA-4) along the same axis as the content head's flat layout, the fusion breaks the rope head's locality. Verify by matching the consumer's GatherMask stride against the fused matmul's output stride before promoting per shape.
- Three-stage cross-core flag chains consume 2 of the `FFTS_MAX_FLAG = 7` user-owned IDs; layered on top of CAND-FA1's existing chain or co-existing pipelines, exhaustion is plausible — track flag-ID accounting at the kernel level.
- Latent GM slot rotation (CAND-FA3) is REQUIRED if the prolog is iterated across a batch dimension with an inflight-depth > 1; without rotation the second down-project would overwrite the latent the up-project of generation N-1 is still reading. The MLA reference uses `stepBatchSize` chunking + a `curBlockTokenOffset` rotation discipline that combines with FA3's modulo slot indexing.
- a5_ops has no MLA-class op shipping yet — pattern is structurally derived but not measured.

**Cross-reference**:
- CAND-FA1 (cross-core flag handoff) — supplies the stage handoff primitive used at each of the two seams in this ladder; the hard-do-not-apply clause about `Matmul<>` carries through
- CAND-FA3 (GM workspace slot rotation) — supplies the multi-generation slot discipline when the prolog is iterated per batch chunk
- CAND-MLA-3 (per-row RmsNorm with shared-tmp UB) — supplies the vec stage's row-normalize implementation
- CAND-MLA-4 (interleaved-pair RoPE via GatherMask) — supplies the downstream rope-head consumer of the fused up-projection's rope slice

**Promote when**: an a5_ops op with the latent-projection shape ships (e.g. a future MLA prolog port, a LoRA-bias-fused matmul prefix, or a fused down-norm-up triplet that is currently three separate ops), AND the shipped kernel demonstrates measurable improvement from the dual-output up-projection fusion vs two sequential up-matmuls, AND the cross-stage flag chain is verified disjoint from any high-level `Matmul<>` library use per CAND-FA1's exclusion clause.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-MLA-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

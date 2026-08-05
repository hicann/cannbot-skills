---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Whole-VEC barrier for cross-partition stage transitions when consecutive vec stages have different core-partitionings"
description: "applies_to: any soc with WaitAllCore / SyncAll on vec engine; cann=9.0.0+; op_class=multi_stage_vec_pipeline_with_repartition derived-from: cann-source (mla-class prolog, 2026-05-10 multicann) verifie"
phenomenon: build_failure
signal:
  - "A vec-side pipeline has consecutive stages where the across-core partitioning differs — stage N partitions the workload along axis X (e.g. batch / token rows),"
confidence: inferred
status: stub
original_id: CAND-MLA-3
timestamp_inferred: true
tags: [candidate, inferred, pipe_mte3, pipe_v, syncall, cand-mla-3]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with WaitAllCore / SyncAll on vec engine; cann=9.0.0+; op_class=multi_stage_vec_pipeline_with_repartition`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h (vec-side process body at ~L880-L955 — engine-scoped WaitAllCore on PIPE_MTE3 appears at three distinct seams: between the down-projection rmsnorm and the subsequent matmul gate; between rmsnorm finish and the dual-output up-projection; between the up-projection and the dequant stage), inline source comment explicitly states the seam exists because two consecutive vec stages have different core-partitioning strategies and require an all-vec barrier between them`
`unverified_on: a5_ops`

**Trigger**: A vec-side pipeline has consecutive stages where the across-core partitioning differs — stage N partitions the workload along axis X (e.g. batch / token rows), stage N+1 partitions along axis Y (e.g. head dim / N axis). At the seam, each vec core has produced an output region that core N+1 (of the next stage) needs to read but did NOT itself produce. A paired AIC/AIV CrossCore flag (CAND-FA1) is insufficient — it synchronizes one AIV with one paired AIC, NOT one AIV with all peer AIVs. Without a whole-vec barrier, stage N+1 starts reading regions that some peer AIV is still writing.

**Recommendation**: Insert `AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(flagId)` (or the equivalent whole-vec barrier helper on the target SoC) at the seam between two vec stages that have different partitioning. The barrier semantics:

  - All AIV sub-blocks wait until every AIV sub-block has reached this point
  - The barrier flag is scoped to the vec engine (NOT cube)
  - The pipe argument (`PIPE_MTE3`) ensures the GM write that the AIV is about to depend on has retired before the barrier completes

The barrier is REQUIRED whenever a re-partitioning seam exists; it is NOT required between consecutive vec stages that share the same core partitioning along the same axis (those need only an intra-core `SetFlag/WaitFlag<HardEvent>` pair).

A typical layered structure mixing all three sync layers:
  - intra-core: `SetFlag<HardEvent::MTE3_V>` / `WaitFlag<HardEvent::MTE3_V>` (within one AIV's pipe chain)
  - whole-vec: `WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId)` (at a re-partitioning seam within vec)
  - cross-engine: `CrossCoreSetFlag<0x2, PIPE_MTE3>(stageDoneId)` / `CrossCoreWaitFlag(stageDoneId)` (between vec and cube, per CAND-FA1)

These three layers compose without conflict — they target disjoint scopes. Using the wrong layer is the failure mode this candidate flags.

**Concrete anchor** (verified pattern from the MLA prolog vec process; two re-partitioning seams shown):
```cpp
// Stage A (vec): row-partitioned RmsNorm over the latent
RmsNormRow(cNormUb, cPreGm, gammaUb, /*col=*/Hc, eps);  // each AIV owns a row subset
DataCopy(cNormGm[rowOff], cNormUb, Hc);                 // emit via MTE3

// Re-partitioning seam: Stage B partitions by head-index, not by row.
// A whole-vec barrier is required because Stage B core will read rows that
// peer AIV cores wrote in Stage A.
constexpr uint32_t allVecBarrierId = 3;  // user-owned flag ID, in 0..7 range
AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId);

// Stage B (vec): head-partitioned RoPE on cross-batch rows
RotaryPosEmb(qRopeUb, qRotUb, cosUb, sinUb, sharedTmp,
             /*rows=*/rowsForThisHead, /*cols=*/Dr, sinCosStride);
DataCopy(qRopeOutGm[headOff], qRopeUb, rowsForThisHead * Dr);

// Another seam if Stage C re-partitions again (e.g. dequant by token)
AscendC::WaitAllCore<SYNC_MODE_ALL_VEC, PIPE_MTE3>(allVecBarrierId);
```

**Why it works**:
- Re-partitioning is a SoC-level data-flow phenomenon: stage N's output region is the union over all AIV cores of per-core sub-regions, and stage N+1's input region is also the union over all AIV cores of per-core sub-regions, but the two unions are partitioned differently. The barrier converts the cross-region dependency into a single "all of stage N has retired before any of stage N+1 starts" guarantee, which is sufficient for any read pattern in stage N+1.
- The pipe argument (`PIPE_MTE3`) ensures the GM write that retires the barrier is the actual data write, not a register write — so consumers in the next stage are guaranteed to see the data, not just the control signal.
- Whole-vec barriers are cheaper than `SyncAll<true>()` because they exclude the cube engine; if no AIC is participating in this seam, broadcasting the barrier to AIC adds unnecessary cube-side stall (cube can keep cube-internal work going). Use the engine-scoped form whenever the seam is engine-internal.

**Determinism**: Adding a barrier never breaks determinism — it only narrows the in-flight schedule. Determinism of the pipeline is governed by the per-element reductions and writes inside each stage; the barrier ensures stage N's writes are visible to stage N+1's reads, which is a precondition for the per-element compute to even be well-defined.

**Other instances predicted**:
- MLA-class prolog (this verified instance) where rmsnorm-by-row feeds qcqr-by-head
- FlashAttention-style vec pipelines where softmax-by-row feeds output-rescale-by-embed
- Fused norm + scatter where norm-by-row feeds scatter-by-target-slot
- MoE expert dispatch where routing-by-token feeds per-expert-shuffle-by-expert
- Any two-stage vec pipeline where stage 1 is "all cores process disjoint row subsets" and stage 2 is "all cores process disjoint column subsets" of the same intermediate tensor
- Permute / transpose epilogues where the producer and consumer partition the tensor along different axes

**Risks before promotion**:
- Using `SyncAll<true>()` instead of `WaitAllCore<SYNC_MODE_ALL_VEC>` is a correctness-equivalent but wasteful alternative — it stalls cube needlessly. Verify the seam is engine-internal before choosing the cheaper form.
- Using an intra-core `WaitFlag<HardEvent::MTE3_V>` instead of the whole-vec barrier is a CORRECTNESS BUG — it only orders the producer's pipe within one AIV, not across peer AIVs. Audit code that has a stage transition + intra-core pipe wait but no whole-vec barrier; this is the most plausible silent-corruption mode for re-partitioning seams.
- Excessive whole-vec barriers serialize the vec engine; ideally place one per re-partitioning seam, not per stage. The MLA reference inserts three barriers across the ~75-line vec process — verify counter-cases that introduce barriers without a re-partition.
- The `PIPE_MTE3` choice assumes the producer's last operation is a GM write; if the producer's last op is a vec compute (PIPE_V) with no GM write, use `PIPE_V` instead — wrong pipe = released before data retires.
- a5_ops has no shipping kernel with vec-engine re-partitioning seams currently; promoting requires a port (e.g. fused norm-then-scatter, fused multi-head rope) that exercises this shape.

**Cross-reference**:
- CAND-FA1 (cross-core flag handoff) — orthogonal layer (this is intra-vec, CAND-FA1 is cross-engine); both compose in one kernel
- P-P75 (intra-core SetFlag/WaitFlag<HardEvent> for SIMD pipe sync) — the intra-core layer; this candidate is the engine-scoped layer that sits between intra-core and SyncAll
- CAND-MLA-1 (latent-prolog skeleton) — the multi-stage ladder where this barrier appears at the vec-side re-partitioning seams
- OL- entries on `SyncAll` overuse — this is the "more-targeted-than-SyncAll" replacement

**Promote when**: an a5_ops vec-pipeline op ships with a documented re-partitioning seam (e.g. fused-norm-then-scatter, fused multi-head rope, fused softmax + output rescale) AND the seam's whole-vec barrier is measured cheaper than the `SyncAll<true>()` alternative AND the engine-scoped barrier is verified necessary (i.e. a peer-AIV-read dependency demonstrably exists at the seam).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-MLA-3，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

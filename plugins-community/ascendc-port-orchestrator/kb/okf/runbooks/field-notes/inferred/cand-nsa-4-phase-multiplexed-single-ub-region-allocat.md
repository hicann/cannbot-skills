---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Phase-multiplexed single UB region — allocate one `TBuf<>` covering all of UB and re-slice it per phase via typed `GetWithOffset<T>` views (UB-side counterpart to P-P89's GM-side workspace contract)"
description: "applies_to: any soc with public TBuf<TPosition::VECCALC + GetWithOffset<T; cann=9.0.0+; op_class=multi_phase_fused_op_with_disjoint_per_phase_ub_needs derived-from: cann-source (nsa-class compressed a"
phenomenon: build_failure
signal:
  - "A fused op runs ≥3 sequential phases per outer iter (e.g. softmax → aux-score → TopK) whose per-phase peak UB tensor sets are mostly disjoint — phase A needs {t"
confidence: inferred
status: stub
original_id: CAND-NSA-4
timestamp_inferred: true
tags: [candidate, inferred, tque, pipebarrier, tbuf, cand-nsa-4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public TBuf<TPosition::VECCALC> + GetWithOffset<T>; cann=9.0.0+; op_class=multi_phase_fused_op_with_disjoint_per_phase_ub_needs`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — softmax phase + scoring phase + topK phase share the same UB region`
`unverified_on: a5_ops`

**Trigger**: A fused op runs ≥3 sequential phases per outer iter (e.g. softmax → aux-score → TopK) whose per-phase peak UB tensor sets are mostly disjoint — phase A needs `{tensorsA[]}`, phase B needs `{tensorsB[]}`, phase C needs `{tensorsC[]}`, and `max(sumA, sumB, sumC) << sumA + sumB + sumC`. Naively giving each phase its own `TBuf<>` overflows UB; per-phase `TQue` rotation does not solve the problem because the buffers are scratch, not pipelined input/output. The phases run serially (a `PipeBarrier<PIPE_V>` or `SetFlag/WaitFlag` boundary separates them), so the UB region's "owner" cleanly changes at each phase boundary.

**Recommendation**: Allocate a single `TBuf<>` covering the kernel's full UB budget (or its largest single-phase peak). At each phase entry, compute byte offsets for that phase's tensors and obtain typed views via `GetWithOffset<T>(elem_count, byte_offset)`. The same byte region under-pins different typed views in different phases — phase A may see it as `LocalTensor<float>`, phase B as `LocalTensor<half>`, phase C as `LocalTensor<int32_t>` — and the phase barrier (`PipeBarrier<PIPE_V>` or `SetFlag<HardEvent::V_MTE3> + WaitFlag<HardEvent::MTE3_V>` when MTE writes intervene) guarantees the previous phase's writes have retired before the next phase's reads begin.

This is the UB-side analog of P-P89's GM workspace contract: ONE byte buffer, host- or kernel-computed offsets, typed re-slicing. Difference: P-P89 covers GM scratch with host-emitted offsets in tilingdata; CAND-NSA-4 covers UB scratch with kernel-computed offsets at phase entry (since UB layout depends on the phase's runtime per-phase row counts and aligned col widths, which are tiling-derived but not flat host constants).

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// Init: one TBuf for all of UB
TBuf<TPosition::VECCALC> allUb;
pipe.InitBuffer(allUb, /*bytes=*/192 * 1024);
LocalTensor<uint8_t> base = allUb.Get<uint8_t>();

// Phase A entry: softmax tensors
int64_t off = 0;
LocalTensor<float> qkScores = allUb.GetWithOffset<float>(rows * cols, off);
off += rows * cols * sizeof(float);
LocalTensor<float> softmaxOut = allUb.GetWithOffset<float>(rows * cols, off);
// ... phase A compute ...
PipeBarrier<PIPE_V>();   // boundary — phase A's writes retire before phase B's reads

// Phase B entry: aux-score tensors REUSE the same byte region with new typed views
off = 0;
LocalTensor<float> scoreScratch = allUb.GetWithOffset<float>(rows2 * scoreLen, off);
off += rows2 * scoreLen * sizeof(float);
LocalTensor<half> packedScores = allUb.GetWithOffset<half>(rows2 * scoreLen, off);
```

For phases separated by an MTE3 emission (e.g. one phase writes intermediate results to GM and a later phase reads them back), use `SetFlag<HardEvent::V_MTE3>` + `WaitFlag<HardEvent::V_MTE3>` at the boundary instead of `PipeBarrier<PIPE_V>` — `PipeBarrier` only orders within the vector pipe and does NOT order against the MTE pipe.

**Why it works**: A `TBuf<TPosition::VECCALC>` of size `B` byte-allocates UB once; subsequent `GetWithOffset<T>(count, off)` views are address arithmetic only and do not allocate. Per-phase reuse is safe ONLY because (a) the phase boundary (`PipeBarrier` / hard-event flag) hard-orders the prior phase's writes before the next phase's reads, (b) phases are mutually exclusive in time (no overlap), and (c) the typed view's lifetime is bounded by the phase scope — using a stale view from phase A inside phase B is a programmer error. The single-`TBuf<>` form avoids the "OL-94 TQue vs TBuf sync decision table" complexity of per-tensor TQue rotation when the tensors are scratch (no pipelined producer/consumer pattern across iterations on the same buffer).

**Determinism**: The phase boundary primitive (`PipeBarrier<PIPE_V>` or `SetFlag/WaitFlag<HardEvent::V_*>`) is deterministic — it stalls the consumer until the producer pipe drains. Per-phase compute uses only public vec/MTE primitives. As long as each phase's compute itself is deterministic (no atomic, no cross-core mid-phase), the multi-phase chain is deterministic by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when phases overlap in time (e.g. phase A's tail runs on MTE while phase B starts on VEC) — there is no UB-byte coherence between concurrent phases. Use separate `TQue`s or separate `TBuf`s in that case.
- Do NOT omit the phase boundary primitive (`PipeBarrier` / `SetFlag` / `WaitFlag`) — the second phase MAY race read-after-write against the first phase's tail and silently see uninitialized bytes (V220-class AIV does not auto-serialize across logical phases sharing UB).
- Do NOT use this pattern when one of the phases needs `TQue`-style double-buffer pipelining ON THE SAME tensor (e.g. streaming input load overlapped with compute on the prior tile) — that is exactly what `TQue<DEPTH=4>` (OL-63) was designed for; do not replace TQue with TBuf re-slicing in pipelined contexts.
- Do NOT use `LocalTensor<T2>` typed views from phase A inside phase B — the view object holds a base + offset; using it after a phase boundary that re-purposed the region is a use-after-free-class hazard at the language level (no compile error, silent wrong-bytes read).
- Do NOT use the reinterpret-cast form (`localTensor.template ReinterpretCast<T>()`) across phase boundaries to "convert" a phase-A typed view to a phase-B typed view — re-obtain via `GetWithOffset<NewT>(count, off)` at phase entry to make the lifetime explicit.

**Other instances predicted**:
- Any fused-attention forward that does softmax → aux-score → TopK or softmax → mask-select → emit (3+ phases per row tile).
- Fused norm + scatter + gather where each phase's working tensors are mutually disjoint.
- Fused dequant → matmul-prep → quant pipelines where dequant scratch, matmul A/B prep buffers, and quant scratch are large and disjoint.
- Multi-stage MoE per-expert dispatch where routing-mask, gather-buffer, and per-expert-output stages each peak in different UB regions.
- Fused LayerNorm + Linear where the LayerNorm's stats buffers and the Linear's matmul-prep buffers do not coexist.

**Risks before promotion**:
- a5_ops has not yet shipped a fused op with 3+ disjoint-UB-need phases sharing one `TBuf<>`; the pattern is unverified on a5_ops perf and precision.
- Phase-boundary primitive choice (`PipeBarrier<PIPE_V>` vs `SetFlag/WaitFlag<HardEvent>`) is failure-mode-different — choosing `PipeBarrier<PIPE_V>` when an MTE write actually crosses the boundary is the OL-94 mis-application class (silent stale data on the second phase).
- Per-phase `GetWithOffset<T>(count, off)` must respect the architecture's UB alignment (32B vector block on V220 / V351); call sites that compute `off` from runtime tiling must `AlignUp(off, 32)` before the next view, otherwise the next typed view's MTE2/MTE3 issues mis-align and either fault or silently corrupt.
- Debugging a wrong-bytes read across a phase boundary is hard — there is no compile-time check that "phase A's `qkScores` view is unused after the boundary". Code-review discipline is required (or static-analysis pass over `GetWithOffset` call sites).

**Cross-reference**:
- P-P89 (GM workspace contract for fused ops — public outputs separate; opaque scratch sliced by host offsets) — this candidate is the UB-side analog. Cross-reference both when shipping a multi-phase fused op: GM scratch follows P-P89, UB scratch follows CAND-NSA-4.
- OL-94 (TQue vs TBuf sync decision table) — directly relevant: the phase-boundary sync primitive choice MUST consult OL-94. `PipeBarrier<PIPE_V>` is correct only when no MTE write crosses the boundary.
- P-P66 (`TQueBind<VECIN, VECOUT, 1>` in-place buffer reuse) — related "share one buffer across roles" pattern, but P-P66 is within ONE pipelined compute, this candidate is across SERIAL phases.
- OL-63 (TQue depth-4 for elementwise) — orthogonal; OL-63 governs streaming pipelined tensors, this candidate governs phase-scratch reuse.

**Promote when**: an a5_ops fused op (e.g. a future fused attention + scoring or fused norm + scatter + gather) ships with ≥3 phases sharing one `TBuf<>` AND total kernel UB peak measured below the alternative of per-phase-`TBuf` allocation (proving the reuse saved UB) AND precision PASS shows no cross-phase wrong-read regressions vs a per-phase-`TBuf` baseline.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-NSA-4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

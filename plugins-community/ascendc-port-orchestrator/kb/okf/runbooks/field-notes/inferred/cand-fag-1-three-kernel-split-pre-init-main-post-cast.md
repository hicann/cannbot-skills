---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Three-kernel split (pre-init → main → post-cast) for fp32-workspace gradient accumulation in low-precision backward ops"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=low_precision_backward_with_multi_source_gradient_accumulation / flash_attention_backward / fused_norm_backward / scatter_grad / any"
phenomenon: build_failure
signal:
  - "A backward op produces gradients (e.g. dq/dk/dv) in a low-precision output dtype (fp16/bf16/fp8) BUT the accumulation across multiple producing blocks must happ"
confidence: inferred
status: stub
original_id: CAND-FAG-1
timestamp_inferred: true
tags: [candidate, inferred, initoutput, cast, muls, datacopy, setatomicnone, cand-fag-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=low_precision_backward_with_multi_source_gradient_accumulation / flash_attention_backward / fused_norm_backward / scatter_grad / any_bwd_op_with_atomic_add_across_cores_on_low_precision_output`
`derived-from: cann-source (fa-grad-class backward, 2026-05-10 multicann)`
`verified_on: cann ops-transformer flash_attention_score_grad/op_kernel/arch35/ — three peer kernel headers covering the pre/main/post split (pre header initializes the fp32 workspace, post header rescales fp32→output-dtype, main header does the actual fwd-recompute + bwd matmul chain); the post header's accumulation-then-cast write-back pattern is the load-bearing structural piece`
`unverified_on: a5_ops (no backward op currently shipped)`

**Trigger**: A backward op produces gradients (e.g. dq/dk/dv) in a low-precision output dtype (fp16/bf16/fp8) BUT the accumulation across multiple producing blocks must happen in fp32 to avoid catastrophic precision loss from atomic-add in low precision. The op cannot simply atomic-add into the user-facing fp16 dq buffer because (a) `SetAtomicAdd<fp16>()` either does not exist, has worse cumulative error, or has slower hardware path, and (b) cross-block partial sums for a single output element may number in the hundreds (one per S2 tile crossing the row).

**Recommendation**: Split the launch into three peer kernels chained on the same workspace via host-side enqueue order. Public-API surface is `InitOutput`, `Cast`, `Muls`, `DataCopy`, `SetAtomicAdd<float>`/`SetAtomicNone`, plus a host-side workspace plan exposed via `tilingData->postTilingData.{dq,dk,dv}WorkSpaceOffset`.

1. **PRE kernel** (AIV-only): on each used core, `InitOutput<float>(dqWorkSpaceGm[off], len, 0)` for the per-core slice of each output's fp32 workspace. When the output dtype IS fp32, skip the fp32 workspace entirely and `InitOutput` the user-facing GM directly. Optional ancillary clears (e.g. dropout-mask working buffer, ds-sink workspace) belong here.
2. **MAIN kernel**: do the recompute + matmul chain; all writes to dq/dk/dv go through `SetAtomicAdd<float>()` into the fp32 workspace (NOT the fp16/bf16 user output). `SetAtomicNone()` is called at the end of each atomic-region to keep subsequent writes ordered.
3. **POST kernel** (AIV-only, ping-pong): tile across each of the three fp32 workspaces, `DataCopy` a tile into UB → `Muls(tile, tile, scale, n)` (e.g. dq/dk inherit the attention scale) → `Cast(outTile, tile, RoundMode::CAST_ROUND, n)` → `DataCopy` final low-precision tile to dq/dk/dv GM. Ping/pong queue pair (`inQuePing`, `inQuePong`, `outQuePing`, `outQuePong`) overlaps the cast/scale of one tile with the GM load of the next. Skip the scale-Muls for dv (it does not carry the attention scale).

**Concrete anchor** (public AscendC):
```cpp
// PRE kernel — clear fp32 workspace before MAIN runs
if constexpr (IsSameType<OutT, float>::value) {
    InitOutput<OutT>(dqGm[dqOffset], initDqSize, 0);  // fp32: write user output directly
} else {
    InitOutput<float>(dqWorkSpaceGm[dqOffset], initDqSize, 0);  // low-prec: clear fp32 scratch
}

// MAIN kernel — all atomic accumulation into fp32 workspace
SetAtomicAdd<float>();
DataCopy(dqWorkSpaceGm[off], dqUbFp32, n);  // partial sum from one block, accumulated atomically
SetAtomicNone();

// POST kernel — ping-pong scale + cast + write final output
LocalTensor<float> inPing  = inQuePing.AllocTensor<float>();
DataCopy(inPing, dqWorkSpaceGm[pingIdx], pingSize);
inQuePing.EnQue(inPing); inQuePing.DeQue<float>();
Muls(inPing, inPing, scale, pingSize);
LocalTensor<OutT> outPing = outQuePing.AllocTensor<OutT>();
Cast(outPing, inPing, RoundMode::CAST_ROUND, pingSize);
outQuePing.EnQue(outPing); outQuePing.DeQue<OutT>();
DataCopy(dqGm[pingIdx], outPing, alignUp16(pingSize));
```

**Why it works**:
- fp32 atomic-add is hardware-supported with deterministic semantics on V220/950PR; fp16 atomic-add either is unsupported or has cumulative error proportional to producer count
- Splitting PRE/MAIN/POST also keeps the MAIN kernel's UB budget free of cast scratch — a fp16 output of size B·N·S·D would otherwise need an extra cast buffer in the hot path
- POST's ping-pong overlap hides Cast latency behind DataCopy, achieving near-MTE2-bound throughput on the rescale
- `InitOutput` is the only public API that emits a fixed-value write through MTE3 without going through UB allocation, making it the right primitive for workspace zeroing

**Determinism**: PRE/POST are deterministic by construction (each output element is written by exactly one core in POST). The MAIN kernel's atomic-add is the determinism risk — see CAND-FAG-2 for the deterministic-mode alternative that replaces atomic-add with a partition-by-coordinate dispatch scheme.

**Other instances predicted**:
- Any backward op of a fused reduction (LayerNorm backward, RMSNorm backward, Softmax backward) when accumulating dWeight / dBias across batch
- Scatter-add gradients when the scatter index set spans more cores than the index cardinality
- MoE expert gradient accumulation back into the shared input embedding
- Cross-entropy + softmax fused backward where dlogits accumulates per-class across batches
- Beam-search / sequence-parallel backward where gradient at a position is summed from multiple ranks/cores

**Risks before promotion**:
- a5_ops has no shipped backward op exercising this three-kernel split — the pattern's launch-overhead-vs-precision tradeoff (3× kernel launches vs 1×) is unmeasured on this codebase
- The fp32 workspace size is `MAX_CUBE_CORE_NUM × CUBE_BASEM × HEAD_DIM_ALIGN` per output for the BN2 path — for tall S/D this can exceed reserved workspace; check `RESERVED_WORKSPACE_SIZE` budget before adopting
- If the output dtype is already fp32, the three-kernel split is wasteful — the FA-grad reference explicitly bypasses workspaces and writes directly to user GM in that case (see anchor `if constexpr (IsSameType<OutT,float>::value)`)
- For very small problems (single-core-sufficient), MAIN-only without atomic-add is faster — gate on producer count

**Cross-reference**:
- P-P89 (GM workspace contract for fused ops): this candidate is the BACKWARD specialization — outputs are public, fp32 scratch is the opaque workspace sliced via `dqWorkSpaceOffset` / `dkWorkSpaceOffset` / `dvWorkSpaceOffset`. Same workspace-contract shape as P-P89; promote-merge if multi-op evidence accumulates
- CAND-FA1 (manual cross-core flag handoff): orthogonal — this candidate is about WHAT goes through the workspace, CAND-FA1 is about HOW writes are ordered
- CAND-FA3 (GM workspace slot rotation): orthogonal — this candidate uses a flat per-output workspace, not a rotating ring

**Promote when**: a5_ops ships a backward op with measured precision improvement vs single-kernel fp16-atomic-add baseline AND measured launch-overhead acceptable vs the precision win.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FAG-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

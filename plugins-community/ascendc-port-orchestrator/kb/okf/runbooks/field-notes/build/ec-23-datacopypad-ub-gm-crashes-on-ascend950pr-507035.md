---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "DataCopyPad UB→GM crashes on Ascend950PR (507035)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "DataCopyPad(gmTensor, localTensor, extParams) for UB→GM direction causes 507035 vector core exception, even with properly aligned blockLen."
confidence: single_run
original_id: EC-23
timestamp_inferred: true
tags: [507035, ascendc, ec-23]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: `DataCopyPad(gmTensor, localTensor, extParams)` for UB→GM direction causes 507035 vector core exception, even with properly aligned blockLen.
- **Root cause**: DataCopyPad in UB→GM direction is not supported (or buggy) on Ascend950PR. GM→UB direction works fine.
- **Fix**: Use `DataCopy` with aligned count instead. Handle non-aligned tails by pre-padding input to aligned size in the host (pybind) layer.
- **Evidence**: Sort kernel development (2026-04-14), confirmed on Ascend950PR_9589 CANN 9.0.0.
- **Cleaner mitigation (29_DynamicQuant kw-2, 2026-05-01)**: when using a tile-loop kernel with `TILE=N`, **pre-pad the input row stride to `align_up(D, N)` in the pybind layer** (pad output stride symmetrically). Every tile (including last) then uses plain `DataCopy(local, gm[r*in_stride+off], TILE)` — no DataCopyPad, no partial-count handling, no risk of EC-23. Pybind narrows back to original D after kernel via `.narrow(-1, 0, D_orig)`. Cost: a few KB of zero-padded GM per row. Benefit: kernel code becomes branchless across tile types and EC-23-immune by construction. Recommended for all tile-loop kernels touching variable D.
- **Non-crash precision-corruption variant (8_QuantScatter kw-1, 2026-05-03)**: when the writeback alignment exceeds the per-slot stride (e.g. `DataCopy<int8>(gm[i*D], ub, AlignUp(D, 32))` with `D < 32`), the **extra bytes silently corrupt the next row of the same logical tensor** — NO crash, NO 507035. Symptom signature: `mismatch_count = batches × overshoot_bytes` exact (e.g. 8 batches × 16-byte overshoot = 128/1024 mismatch on Pass B). When you see a precision Pass B failure with this exact factorization, check the writeback path's count argument vs the per-slot stride before suspecting algorithm bugs. **Same mitigation applies**: pad output GM stride to `align_up(D, 32)` in pybind, narrow back via `.narrow(-1, 0, D_orig)` before return.
- **Portability shim from CANN ops-nn (2026-05-12, CAND-A3A5-3 auto-merged via Mode 5 C37)**: cross-op evidence from `group_norm_silu_quant_base.h` lines 17-25 + `rms_norm_quant.cpp` line 245 (ReduceSum fork) shows CANN's own norm/quant kernels use a portability pattern: `#if defined(__CCE_AICORE__) && __CCE_AICORE__ == 220` branches use `DataCopyPad` for UB→GM (allowed on V220); the `#else` branch (V351 / arch35) falls back to `DataCopy` with caller-supplied `align_up(count, 8)` sizing. When porting any norm/quant op kernel with partial-block UB→GM writes, **reuse this shim instead of authoring a new one**. The shim is typically a small inline helper at the top of `<op>_base.h`; copy the pattern verbatim and parameterize over the dtype.
- **Cross-op confirmation that V351 DataCopyPad both directions work (group_norm_silu kw-1, 2026-05-24)**: port_a3_to_a5 V220→V351 AUTHORED-from-source cold start. Upstream V220 `group_norm_silu_base.h::IsDataCopyPadSupport()` returns `false` for `__NPU_ARCH__ == 3510` (V351/arch35) — only `220` / `3003` / `3113` are listed as supported. Empirically, DataCopyPad worked cleanly in BOTH directions on V351: GM→UB load of a 4-element fp32 (16 bytes, non-32B-aligned) AND UB→GM store of the same. Case 7 bit-exact PASS. Worker did NOT fall back to the portability shim; used DataCopyPad unconditionally. Combined with the 2026-05-18 task #22 aog-hardware-probe (UB→GM blockLen ∈ {31,33,47,63}) + OL-167 scope clarification, the upstream V220 `IsDataCopyPadSupport` guard is **stale for V351** — V351 workers SHOULD use DataCopyPad freely (per OL-167 / P-P98) and need not author / inherit the V220-style `#if __CCE_AICORE__ == 220` portability shim unless they explicitly target V220 as a co-build.
- **2nd cross-op confirmation (modulate kw-1, 2026-06-21, port_a3_to_a5 V220→arch35, A5 Ascend950PR_9579)**: a self-contained VEC affine kernel used DataCopyPad freely in BOTH directions (GM→UB load + UB→GM store) on V351 — no 507035, no portability shim — across 225/225 PASS (fp16/fp32/bf16). Adds a second port_a3 cross-op data point (after group_norm_silu) that the upstream V220 `IsDataCopyPadSupport`==false-for-3510 guard is stale for V351.
- **3rd cross-op confirmation, int64 output dtype (top_k_top_p_sample kw-4, 2026-06-24, port_a3_to_a5 V220→V351, Ascend950PR_9579)**: a sampling kernel used `DataCopyPad(outIdxGm[rowId], out, DataCopyExtParams{1, sizeof(int64_t), ...})` for a single-element **int64** (8-byte) UB→GM index write (`top_k_top_p_sample_kernel.h:324-325`) — no 507035, no portability shim — across a 16-case bit-exact PASS suite. Adds the **int64** dtype to the V351 both-directions-OK evidence base (prior confirmations: fp16/fp32/bf16 via group_norm_silu + modulate), reinforcing that the upstream V220 `IsDataCopyPadSupport`==false-for-3510 guard is stale for V351 across all common index/value dtypes.

<!-- 迁移自 porter kb/target/ascendc/（EC-23，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

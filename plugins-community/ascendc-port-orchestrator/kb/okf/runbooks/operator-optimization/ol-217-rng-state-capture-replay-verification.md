---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A non-deterministic reference that RETURNS its RNG state is bit-exact verifiable by capture-and-replay — \"non-deterministic\" ≠ \"unverifiable\""
description: "A non-deterministic reference that RETURNS its RNG state (seed/offset) is bit-exact verifiable by capture-and-replay; only truly-internal RNG state is unverifiable-by-construction."
confidence: single_run
original_id: OL-217
classified_by: llm-assisted
timestamp_inferred: true
tags: [verification, optimization, ol-217, rng, dropout, capture-replay]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: verifier-side methodology, chip-independent by construction — rng-driven non-deterministic references (dropout / stochastic-rounding / sampling / noise-injection). All SoC / all CANN.

**Principle**: before labeling a non-deterministic reference "unverifiable-by-construction" and scoping its cases out, ask the discriminator question — **does the op expose / return the random state that drives the non-determinism (seed / offset / RNG handle)?**
- **YES** → it is verifiable. Capture the returned state from the reference call, feed that exact state to the candidate, and compare the candidate against the SAME reference call whose state you hold → bit-exact. Run-to-run variation is irrelevant because you are not comparing call-N vs call-M.
- **NO** (state internal, never surfaced) → genuinely unverifiable-by-construction.

The trap (an ANTI_PRESSURE P5 failure-wrapping) is to prove the reference is non-deterministic (its RNG counter advances per call) and then conclude the cases are unverifiable. A reproduced root-cause for the *non-determinism* is NOT a root-cause for *unverifiability* — different claims. Comparing candidate-call-N against reference-call-M (M≠N) is the wrong comparison; hold the reference call's returned state and replay it.

**Tiling-invariance corollary**: if the reference's RNG counter is keyed to **absolute output coordinates** (e.g. `(batch, head, group, s1·base+offset, s2)` + the returned base counter) rather than to loop/core indices, then a candidate with a DIFFERENT tile/core layout still reproduces the identical random stream — the mask depends only on `(seed, offset, absolute-position, aligned-stride)`, not on the candidate's core-split. Use this to justify free tiling decisions when porting RNG-driven ops: replay the base counter, tile however you want.

**Concrete anchor**: benchmark `npu_fusion_attention` returns a 7-tuple `(attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length)`; `seed=ref[4]`, `offset=ref[5]` are the exact Philox state of that call. Feed them to the `hasDrop=true` candidate instance → it replays the identical dropmask → `max_abs_diff=0.0` vs the same ref call. The Philox counter is derived from absolute element positions + the returned base offset (per-row stride `s2SizeAligned = CeilDiv(S2,16)*16`), so the bit-match is independent of the candidate's tiling.

**Evidence**: FA-A5 `3_FusionAttention`, 2026-06-11 (kw-1→kw-2): kw-1 measured the dropout reference as non-deterministic (offset advances 0→12→…→384 per call, seed constant) and wrongly scoped the 33 `keep_prob<1.0` cases out as "unverifiable-by-construction"; kw-2 applied capture-and-feed (ref[4]/ref[5] → candidate) and recovered all of them bit-exact.

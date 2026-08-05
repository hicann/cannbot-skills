---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "arch35 WelfordUpdate / WelfordFinalize replace hand-rolled mean/variance loops"
description: "arch35 exposes WelfordUpdate / WelfordFinalize as public primitives for numerically stable running mean+variance; prefer them over V220-style separate sum_x / sum_x² accumulation in norm-forward kernels."
original_id: OL-135
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-135, normalization-fwd, welford, numerical-stability]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

arch35 exposes `WelfordUpdate<T, AccT, Initialize>(...)` and `WelfordFinalize<UseUnbiased>(...)`
as public AscendC primitives that compute running mean + running variance with the
numerically stable Welford algorithm. V220 has neither; the V220 equivalent norm kernel
hand-rolls the loop, accumulating `sum_x` + `sum_x²` separately and computing
`var = E[x²] - E[x]²` at the end (less numerically stable).

**Applies to** `soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=normalization-fwd`
(covers LayerNorm / GroupNorm / RmsNorm / InstanceNorm / BatchNorm variants). Verified on
Ascend950PR (`ada_layer_norm/op_kernel/arch35/ada_layer_norm_impl.h`). Unverified on
Ascend910_V220 — V220 has no Welford primitive; V351-only by API availability.

When porting a V220 norm kernel to A5 arch35 OR writing a new A5 norm kernel from scratch,
prefer `WelfordUpdate` / `WelfordFinalize` over the hand-rolled accumulation.

### Concrete anchor (from `ada_layer_norm_impl.h`)

```cpp
// Per-tile update — runs once per row segment
WelfordUpdate<T, float, /*Initialize=*/false>(
    meanLocal, varLocal,        // running mean / variance accumulators
    meanLocal, varLocal,        // in-place update (overwrites the same)
    xTileLocal,                 // input slice [tile_size]
    scratchLocal,               // temporary buffer
    param);

// After all tiles consumed — turn running counters into final mean/var
WelfordFinalize</*UseUnbiased=*/true>(
    meanLocal, varLocal,
    meanLocal, varLocal,
    nElements,                  // total count (host-known)
    param);
```

### Why use this

1. **Numerical stability** — Welford avoids cancellation in the `E[x²] - E[x]²` form.
2. Compile-time `Initialize` flag avoids a zero-init branch.
3. The primitive is **arch-tuned** — uses reg-based MicroAPI internally on V351, so no
   `Reg::LoadAlign` boilerplate is needed in the caller.
4. The caller doesn't have to manage the running-count state machine.

### Anti-pattern (DO NOT)

- Hand-roll `sum_x += x; sum_xx += x*x` then `var = sum_xx / N - (sum_x / N)²` on A5.
- Mix `WelfordUpdate` with V220-style accumulation in the same kernel — use only one strategy
  per row.

**Predicted other instances**: `rms_norm_quant`, `group_norm_silu_quant` could use
`WelfordUpdate` if their algorithm computes mean/var (RMSNorm computes RMS only, so
`WelfordFinalize` may not apply directly; GroupNorm does both). LayerNorm + InstanceNorm +
BatchNorm forward are direct fits.

**Cross-ref**: P-P91 variant-split (fp16/bf16/fp32 paths each use Welford);
ASCENDC_API_CATALOG.md. (Source text truncated at cross-ref.)

Source: cann-learner CAND-A3A5-10, promoted 2026-05-12 (Mode 5 batch 2); C36 lift generalized
op-class from "ada_layer_norm" → "normalization-fwd".

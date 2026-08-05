---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "V220 PipeBarrier<PIPE_V> coalescing — remove redundant intra-pipe barriers"
description: "Domain: memory_access / sync Arch: V220 / arch22 (Ascend910B-series) only Companion: P-P77 (PipeBarrier precision regression — 6/10 wrong outputs from extra barriers on unrelated V ops) ### When to ap"
severity: high
confidence: single_run
original_id: P-P92
timestamp_inferred: true
tags: [memory_access, optimization, setflag, waitflag, pipe_v, p-p92, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Domain**: memory_access / sync

**Arch**: V220 / arch22 (Ascend910B-series) only
**Companion**: P-P77 (PipeBarrier precision regression — 6/10 wrong outputs from extra barriers on unrelated V ops)

### When to apply

A SIMD elementwise/fused kernel that:
1. Uses TBuf-based pipeline with explicit `SetFlag`/`WaitFlag` at PIPE crossings
2. Emits > 5 `PipeBarrier<PIPE_V>()` calls per row-loop iteration
3. Profiler shows `aiv_vec_ratio < 60%` on what should be VEC-bound
4. Runs on V220 / arch22 (Ascend910B2C, Ascend910B4, etc.)

### Principle

On V220 arch22, the **VEC pipe is in-order within its own pipe**. Consecutive `PIPE_V` operations auto-serialize via data dependencies — the hardware tracks which VEC op reads which UB region and which VEC op wrote it last. An explicit `PipeBarrier<PIPE_V>()` between two back-to-back VEC ops performs ZERO useful synchronization: the data dependency already serializes them. Worse, each `PipeBarrier<PIPE_V>()` forces a **pipe drain** (~tens of cycles latency), flushing the VEC pipeline and waiting for all prior VEC ops to retire before proceeding.

In a row-loop kernel with N such barriers per row × M rows, the cumulative drain overhead **dominates total runtime**, turning a compute-bound kernel into a sync-overhead-bound one.

**What PipeBarrier<PIPE_V> does NOT protect against**: data races between PIPE_V and PIPE_MTE2 / PIPE_MTE3 / PIPE_S. Those are cross-pipe operations and the in-order guarantee does NOT cross pipe boundaries. `SetFlag`/`WaitFlag` at phase boundaries is still REQUIRED for those.

### Fix

```cpp
// BEFORE (anti-pattern): 24 PipeBarrier<PIPE_V> calls
Mul(buf, buf, scale, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT — Mul writes buf, Cast reads buf, data dep serializes
Cast(tmp, buf, CAST_NONE, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT
Sigmoid(buf, tmp, count);
PipeBarrier<PIPE_V>();       // ← REDUNDANT
// ... etc — 24 total per row

// AFTER (P-P92 fix): 0 PipeBarrier<PIPE_V> calls
Mul(buf, buf, scale, count);
// No barrier — VEC pipe auto-serializes Mul→Cast via buf data dependency
Cast(tmp, buf, CAST_NONE, count);
// No barrier — VEC pipe auto-serializes Cast→Sigmoid via tmp data dependency
Sigmoid(buf, tmp, count);
// ... cross-pipe sync (V→MTE3, V→S) still uses SetFlag/WaitFlag at phase boundaries
```

**RETAIN** these cross-pipe sync points:
- `SetFlag<HardEvent::V_MTE3>(ev)` / `WaitFlag<HardEvent::V_MTE3>(ev)` — ensure MTE3 sees VEC writes before DMA-out
- `SetFlag<HardEvent::MTE2_V>(ev)` / `WaitFlag<HardEvent::MTE2_V>(ev)` — ensure DMA-in completes before VEC reads
- `SetFlag<HardEvent::V_S>(ev)` / `WaitFlag<HardEvent::V_S>(ev)` — scalar pipe must see VEC broadcasts
- `SetFlag<HardEvent::S_V>(ev)` / `WaitFlag<HardEvent::S_V>(ev)` — VEC must see scalar writes

### Evidence

op#11 DequantSwigluQuant kw-3 H1 (2026-05-13):
- **Pre-H1**: 24 `PipeBarrier<PIPE_V>()` calls → 354 µs wall-clock on [128,4096] → 0.19× CANN (5.4× slower)
- **Post-H1**: 0 `PipeBarrier<PIPE_V>()` calls → 52 µs wall-clock → **6.8× speedup**, 1.27× CANN (faster)
- **Precision**: 49/50 PASS_T1 preserved bit-exact (H1 is sync-only, zero arithmetic change)
- **Determinism**: 50/50 identical preserved (VEC in-order semantics unchanged)
- **Researcher prediction was 1.5–2.0×; actual was 6.8×** — underestimated because the cumulative drain cost of 24 barriers per row × 456 rows wasn't modeled

### Limits

- **V220 only**: confirmed on Ascend910B2C (arch22). A5/V351 reg-based SIMD may have different pipe semantics — verify before applying.
- **Intra-VEC only**: only `PipeBarrier<PIPE_V>()` is safe to remove. Do NOT remove `PipeBarrier<PIPE_MTE2>()`, `PipeBarrier<PIPE_MTE3>()`, `PipeBarrier<PIPE_S>()`, or `PipeBarrier<PIPE_ALL>()`.
- **Phase-boundary flags stay**: SetFlag/WaitFlag at PIPE crossings remain REQUIRED. This pattern only removes intra-VEC-pipe barriers between consecutive VEC ops.
- **Detection pre-condition**: kernel must already have correct `SetFlag`/`WaitFlag` at phase boundaries. If phase-boundary sync is wrong, removing intra-VEC barriers won't fix it.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P92，convert_patterns_to_okf.py）。confidence 未升格。 -->

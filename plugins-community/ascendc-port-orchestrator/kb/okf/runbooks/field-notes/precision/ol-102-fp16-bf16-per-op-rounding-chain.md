---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "fp16/bf16 composite ops must replicate CPU's per-op rounding chain (cast back to T between each mul and each add)"
description: "Under CPU-truth, an fp16/bf16 multiply-accumulate kernel must round back to native dtype after each mul and each add (matching CPU); summing in fp32 throughout is MORE precise but bit-DIFFERENT and fails the bit-match — more precise != correct here."
phenomenon: precision_issue
signal:
  - "fp16/bf16 kernel does multi-step arithmetic (Σ v_c*w_c or similar) and the reference does the same op in native dtype with no fp32 promotion"
  - "an 'all fp32 sum, cast at end' kernel shows ~1 ULP max-abs-diff and a MARE catastrophe at value boundaries"
confidence: single_run
original_id: OL-102
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-102, fp16, bf16, rounding-chain]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Kernel performs `Σ_c v_c * w_c` (or similar reduction) where `v_c, w_c` are fp16/bf16 AND the reference does the same operation in native dtype (no fp32 promotion). A kernel that sums in fp32 and casts once at the end diverges from the CPU reference bit pattern even though it is numerically more precise.

## 根因 / 教训

PyTorch on CPU for an fp16 multiply-accumulate does:
1. fp16 a, b → cast to fp32 → multiply in fp32 → cast result BACK to fp16 (1 round).
2. fp16 sum + fp16 next → cast both to fp32 → add → cast BACK to fp16 (1 round per add).

For a 4-corner bilinear reduce: 4 mults × 1 round + 3 adds × 1 round = **7 rounding points**.

A kernel that does "sum in fp32 throughout, cast to T at end" has 1 rounding point — MORE precise but a DIFFERENT bit pattern from CPU. Under the CPU-truth standard, more precise ≠ correct.

### Correct kernel pattern — replicate the per-op round chain

```cpp
T p0 = simt_from_float<T>( simt_to_float<T>(v0) * simt_to_float<T>(w0) );  // round per mul
T p1 = simt_from_float<T>( simt_to_float<T>(v1) * simt_to_float<T>(w1) );
T p2 = simt_from_float<T>( simt_to_float<T>(v2) * simt_to_float<T>(w2) );
T p3 = simt_from_float<T>( simt_to_float<T>(v3) * simt_to_float<T>(w3) );

T s01 = simt_from_float<T>( simt_to_float<T>(p0)  + simt_to_float<T>(p1) );  // round per add
T s02 = simt_from_float<T>( simt_to_float<T>(s01) + simt_to_float<T>(p2) );
T s   = simt_from_float<T>( simt_to_float<T>(s02) + simt_to_float<T>(p3) );

out[i] = s;
```

For `T=fp32`, `simt_to_float` / `simt_from_float` are identity, so this collapses to plain fp32 — no perf penalty.

### Anti-pattern (more precise but wrong)

```cpp
// WRONG: "all fp32 sum at end"
float s = simt_to_float<T>(v0) * simt_to_float<T>(w0)
        + simt_to_float<T>(v1) * simt_to_float<T>(w1)
        + ...;  // 1 final cast — diverges from CPU per-op chain
out[i] = simt_from_float<T>(s);
```

### Empirical impact

- op#28 bf16 case 11: all-fp32-sum-at-end → max abs diff `1.5625e-2` (1 bf16 ULP at value 2.0) → MARE catastrophe; per-op rounding chain → **0.0 (bit-exact)**.
- op#1 RotaryMul stage-pilot (pure Mul/Add chain, no transcendentals), CPU-truth over 50 cases: V0 (fp32-promote + 1 final RINT cast) → 4/50; V_CDE half-mode (direct-T compute, no fp32 promote) → 41/50; + V_CD interleave (per-Mul round)… (source text truncated in the batch excerpt here).

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — op#28 MultimodalRopePos kw-fix iter 2 (V2 per-op rounding fix bit-matched CPU on bf16 case 11).

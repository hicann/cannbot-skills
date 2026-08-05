---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT→SIMD conversion (memory-bound elementwise kernel)"
description: "Trigger condition: msprof shows MTE2=0% AND throughput < 50% of theoretical bandwidth Scenario: a SIMT kernel performs elementwise/per-group ops; all GM accesses go through dcache (VEC pipe); the MTE2"
confidence: single_run
original_id: P-P33
timestamp_inferred: true
tags: [memory_access, optimization, p-p33, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger condition**: msprof shows MTE2=0% AND throughput < 50% of theoretical bandwidth

**Scenario**: a SIMT kernel performs elementwise/per-group ops; all GM accesses go through dcache (VEC pipe); the MTE2 DMA engine is completely idle.

**Diagnosis**:
```
msprof data:
  aiv_vec_ratio: high (>70% for large tensors)
  aiv_mte2_ratio: ~0%
  aiv_mte3_ratio: ~0%
  Throughput: actual 125 GB/s vs theoretical 400 GB/s (31%)
```

**Reason**: in SIMT mode, GM reads/writes go through dcache (128B cacheline), not through MTE2 DMA. The VEC pipe carries both compute and memory access, and the two cannot parallelize.

**Optimization**: switch to SIMD or hybrid mode:
```
Option A (pure SIMD): DataCopy(MTE2) → VEC compute → DataCopy(MTE3)
  - Applicable when the computation can be expressed with SIMD vector instructions
  - TQue<VECIN,4> + TQue<VECOUT,2> for automatic pipeline overlap

Option B (hybrid SIMT+SIMD):
  - SIMD DataCopy bulk-loads into UB
  - GetPhyAddr() obtains the UB physical address
  - SIMT VF_CALL performs irregular computation (e.g. bit ops)
  - SIMD DataCopy writes back to GM
  - Applicable when the computation contains operations SIMD does not support (e.g. reinterpret float↔int)
```

**Expected effect**: 2-3x throughput gain on large tensors (MTE2+VEC dual-pipe parallelism)

**Constraints**:
- MXFP4-specific analysis required: the PyTorch version's algorithm uses float math (log2, floor, pow2) and may be fully expressible in SIMD
- The source version uses bit ops (reinterpret cast, bit shift) and must use hybrid mode

**Important limitation (verified 2026-04-07)**:
the SIMD version of MXFP4 is **4-20x slower** than SIMT. Reason: MXFP4 quantization needs per-element x_exp and shift amount, which cannot be expressed with SIMD vector instructions (different shift per element). SIMD degenerates to per-element GetValue/SetValue scalar operations, far slower than SIMT's 128-thread parallelism.

**P-P33 applicability update**:
- Applies: computation fully expressible with SIMD vector instructions (Add, Muls, Cast — same op for every element)
- Applies: SG forward/backward (contiguous DataCopy + Muls + Add, all vector ops)
- Not applicable: per-element heterogeneous computation (e.g. MXFP4's per-element log2/pow2/shift)
- Not applicable: quantization / bit-ops requiring per-element conditional branching

**Decision criterion**: check whether the inner loop executes the **exact same instruction sequence** for every element (same Muls/Add/Cast). If each element needs different ops (different shift amount, different branch), SIMT is better.

**SIMD V4 "fast" experiment (2026-04-07)**:
tile-wide shared exponent (no per-group loop) is **1.08x faster than SIMT** on small tensors, proving SIMD itself is not slow.
But **precision is broken**: using a tile-level exponent instead of a per-32-group exponent lowers quantization precision (does not meet MXFP4 spec, cannot be used as production).

**Full comparison (same-NPU A/B)**:
| Version | 4K(ms) | 4M(ms) | Precision | Production-capable |
|:---:|:---:|:---:|:---:|:---:|
| SIMT (128 threads) | 0.018 | **0.253** | OK PyTorch exact | OK **production** |
| SIMD V3 (per-group vectorized) | 0.029 | 1.724 | OK PyTorch exact | NO, slower than SIMT |
| SIMD V4 fast (tile-wide) | **0.017** | 0.813 | WARN **precision degraded** | NO, violates spec |

**WARNING — precision**: SIMD V4 fast replaces the per-32-group exponent with a tile-wide shared exponent.
This means 1024 elements share one exponent, while the MXFP4 spec requires one per 32 elements.
When intra-tile values vary widely (some near 0, some large), small values underflow to 0.
**The A3 hand-written SIMD implementation has the same issue — that is the root cause of its precision bug.**

**P-P33 final conclusion**:
1. The SIMD perf bottleneck is not SIMD mode itself, but the **per-group serial loop**
2. Eliminating the per-group loop (tile-wide processing) lets SIMD be faster than SIMT
3. But eliminating the per-group loop = abandoning per-group precision = **violates spec**
4. For group-local quantization operators, **SIMT is the only approach that meets both precision and performance**
5. SIMD applies when group_size >= tile_size, or when per-group precision is not required

**Evidence**: MXFP4 full-chain verification (2026-04-07): msprof + SIMD V1/V2/V3/V4 A/B + PyTorch spec comparison

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P33，convert_patterns_to_okf.py）。confidence 未升格。 -->

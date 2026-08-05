---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Row-Scalar VEC Multiply — avoiding per-row scalar multiply on the Scalar pipe"
description: "Scenario: in a SIMD kernel, each row has a scalar coefficient; the canonical form is Muls(work_row, work_row, scale_k, H), with a different scale_k per row. A direct GetValue(scale[i]) → Muls(...) tri"
severity: high
confidence: single_run
original_id: P-P62
timestamp_inferred: true
tags: [memory_access, optimization, scale_k, mul, rowmuls, k_base, k_muls_flex, p-p62, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Scenario**: in a SIMD kernel, each row has a scalar coefficient; the canonical form is `Muls(work_row, work_row, scale_k, H)`, with a different `scale_k` per row. A direct `GetValue(scale[i]) → Muls(...)` triggers the whole MTE2_S → S_V → V_S Scalar-VEC sync chain. Empirically on op#11, `scalar_ratio ~= 0.44` (close to Scalar-VEC serialization).

**Correct pattern (RowMuls pattern, from `ops-transformer/attention/incre_flash_attention/op_kernel/ifa_public_define.h`)**:

```cpp
// src1Ub shape: [dealRowCount, 8_fp32_or_16_fp16]
// Each row's scalar has been pre-filled into a 32B block (via Brcb or Duplicate)
BinaryRepeatParams params;
params.src0BlkStride = 1;
params.src1BlkStride = 0;       // KEY: src1 does not step across blocks (the whole block is a single scalar broadcast)
params.dstBlkStride  = 1;
params.src0RepStride = columnCount / blockElementNum;  // row_stride in blocks
params.src1RepStride = 1;       // each repeat advances by 1 block = the next row's scalar
params.dstRepStride  = columnCount / blockElementNum;

AscendC::Mul(dst, src0, src1Ub, /*elements_per_repeat*/ REPEAT_ELEMENT_NUM,
             /*repeatTimes*/ dealRowCount, params);
```

**How to fill src1Ub** (two choices):
- `Brcb(src1Ub, scalars_ub, dealRowCount, brcbParams)` — AscendC API ref 07_0089, a dedicated hardware "8 scalars → 8 × 32B blocks" instruction
- `Duplicate<T>(src1Ub_block_i, scalar_i, blockElementNum)` in a single loop (low overhead since it is only init-time)

**Principle**: `src1BlkStride=0` + `src1RepStride=1` make VEC `Mul` treat each row's 32B block as "one scalar broadcast across the row", completely on the VEC pipe without touching the Scalar pipe. Equivalent to "per-row scalar × whole-row vector" but latency ~= 1 VEC Mul instead of N Scalar-syncs.

**Anti-pattern**:
```cpp
for (int i = 0; i < N; i++) {
  T scale = scaleUb.GetValue(i);   // Scalar pipe, ~100 cycle
  Muls(work[i*H], work[i*H], scale, H);  // triggers S_V sync
}
```

**Evidence of CANN internal usage**:
- `ops-transformer/attention/incre_flash_attention/op_kernel/ifa_public_define.h` `RowMuls`
- `ops-transformer/attention/sparse_flash_attention/op_kernel/arch32/sparse_flash_attention_service_vector_mla.h`
- `ops-transformer/attention/nsa_selected_attention_infer/op_kernel/nsa_public_define.h`

Standard per-row scale pattern for flash-attention-class operators.

**op#11 opportunity**: DequantSwigluQuant has per-row dequant_scale_k; currently uses `GetValue→Muls` giving `scalar_ratio 0.44`. Switching to the RowMuls pattern should be significantly faster.

**Measured data** (`probe_findings/2026-04-21_Q_scalar_broadcast.md`, N=64 rows × H=8 fp32 in one block, warmup 10 + 20 measured, median):

| Variant | median time (us) | ratio vs K_base |
|---------|------------------|-----------------|
| `K_base` (per-row GetValue + Muls) | 758.53 | 1.00x |
| `K_muls_flex` ("Muls flexible scalar position" arg-order variant) | 778.52 | 0.97x — **useless**; argument order does not change the pipe path |
| `K_brcb` (single Brcb + single wide Mul with src1BlkStride=0/src1RepStride=1) | **29.95** | **25.3x** WINNER |

Brcb path wins decisively; this pattern is exactly that path. Note: the probe uses H=8 (block-width matched) which amplifies the ratio; at the realistic H=2048 the ratio will converge but is still expected to be significant (follow-up probe with shape-matched H TBD).

**Counter-lesson** (from this probe): the "flexible" in `Muls (flexible scalar position)` refers to **argument position** (dst, src, scalar vs dst, scalar, src), **not** scalar source (UB vs register). Both overloads go through the Scalar pipe; neither bypasses the sync chain. Names in the hiascend.com API list can mislead; you must read the signature.

**Applicability (critical — lesson from op#11 Kind-1 respawn 2026-04-21)**:
- Brcb's 25.3x comes from **one Brcb + 1 wide Mul covering N_ROWS rows**. If the kernel is currently `for r in N: process_one_row(r)` (1 row per iter), each iter has only 1 scalar; Brcb degenerates to "put a single scalar in 1 block then wide Mul" — equivalent to (actually **worse than**) the original `Muls(dst, src, scalar, H)`.
- The precondition of P-P62 is **the kernel has a multi-row parallel axis to amortize over**. A single-row loop does not satisfy it.
- If the kernel is single-row-per-iter, applying P-P62 requires a **Kind-2 architecture rewrite**: batch R ≥ 8 rows per iter. This means UB must hold R row buffers simultaneously (H=4992 fp32 × R=8 = 160 KB for work alone, plus other buffers can exceed 192 KB); usually not cost-effective.
- Actual op#11 result: static audit found only 2 scalar Muls in the kernel (`as_scalar` + `dyn_scale`); the other per-row operations are already H-wide vector ops. Brcb has no room to fold; Kind-1 retrofit failed.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P62，convert_patterns_to_okf.py）。confidence 未升格。 -->

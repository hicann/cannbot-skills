---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Software MXFP8 dequant on the VEC pipe for reduction-class ops"
description: "LoadDataWithMxScaling (hardware MX) is CUBE-only; VEC reduction ops reading MXFP8 must software-dequant: Cast<float,fp8_e4m3fn_t> then per-32-block E8M0 scalar Muls before the reduction."
confidence: single_run
original_id: OL-170
classified_by: llm-assisted
timestamp_inferred: true
tags: [mxfp8, optimization, ol-170, vec-pipe, dequant, layernorm]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Applies to `soc=Ascend950PR, cann=9.1.0, bisheng=15.0.5+2026-04-13,
op_class=mx-format-vec-only` (LayerNorm, RMSNorm, Softmax, any reduction-class op reading MXFP8).
Verified on `soc=Ascend950PR_957c, cann=9.1.0.B010, benchmark=MxFp8LayerNorm`. Unverified on
Ascend910_V220 (A3 has `fp8_e4m3fn_t` but ToFloat constraints differ).

**Principle**: The hardware MX path (OL-145, `LoadDataWithMxScaling`) is **CUBE-only** — it ships
`(data, e8m0_scale)` from L1 → L0A/L0B → Mmad for matmul-class ops. For pure-VEC
reduction/normalization ops (LayerNorm, RMSNorm, Softmax) that consume MXFP8 inputs, the kernel must
do **software dequant in the vector pipe** before the reduction:

1. Load `D` uint8 bytes of E4M3 data + `D/32` uint8 bytes of E8M0 scale per row into UB.
2. Reinterpret the data uint8 buffer as `fp8_e4m3fn_t` (zero-cost view cast) and call
   `Cast<float, fp8_e4m3fn_t>(dst_fp32, src_fp8, RoundMode::CAST_NONE, D)` — the native AscendC
   dequant of the E4M3 mantissa to fp32 (V351-supported fp8 types per OL-145).
3. Per 32-element block, decode the E8M0 byte to an fp32 scale (`scale = 2^(byte - 127)`) and
   broadcast-multiply the block via `Muls(xF32[b*32], xF32[b*32], scale, BLOCK_SIZE)`. The E8M0 →
   fp32 conversion is a single bit-shift:
   `uint32_t bits = uint32_t(byte) << 23; float scale = *reinterpret_cast<float*>(&bits);`.
4. From here the kernel sees standard fp32 — apply the normal reduction tree (OL-115 / P-P45) and
   affine math.

**Why software, not hardware MX**: `LoadDataWithMxScaling` is wired to the Cube fixed-point pipeline
(L0A/L0B → Mmad), not to VEC. The VEC pipe has no native "load fp8 with per-block scale and dequant
on read" intrinsic on Ascend950PR. The two-step software dequant (E4M3 native cast + E8M0
scalar-broadcast Muls) costs about 1 VEC instr per 32-element block on top of the standard fp32
reduction — negligible vs the MTE2/HBM cost in the bandwidth-bound regime.

**Concrete anchor** (op#MxFp8LayerNorm kw-1, 2026-05-20):
```cpp
auto xDataFp8 = xData.ReinterpretCast<fp8_e4m3fn_t>();
Cast(xF32, xDataFp8, RoundMode::CAST_NONE, norm_size_);
PipeBarrier<PIPE_V>();
for (int32_t b = 0; b < n_blocks_; b++) {
    uint8_t sb = scalesLocal.GetValue(b);
    float scale = E8m0_byte_to_scale(sb);   // = 2^(sb - 127) via uint32<<23 reinterpret
    Muls(xF32[b * 32], xF32[b * 32], scale, 32);
}
PipeBarrier<PIPE_V>();
```

**Constraints**: `D` (last/normalized dim) MUST be a multiple of 32. The block axis is the inner axis
and never crosses row boundaries — this gives the op batch-invariance by construction. The scales
tile must be padded to a 32-byte multiple for `DataCopy` alignment (allocate
`n_blocks_padded = align_up(n_blocks, ...)`).

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A5 narrow-float datatype family — FP8 / HiFloat8 / FP4x2 / fp8_e8m0_t scale [V351, narrow-float]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,matmul-fused,norm-quant"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,matmul-fused,norm-quant"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-144
timestamp_inferred: true
tags: [ascendc, ol-144]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,matmul-fused,norm-quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 (Ascend/agent-skills) ascendc-operator-A5-migration SKILL.md §229-235; l2-guide §38-98, 354-410; api docs c_api/reg/reg_vector/asc_{float2e4m3,float2e5m2,e4m32float,e5m22float,bfloat162e1m2x2,bfloat162e2m1x2,e1m2x22bfloat16,e2m1x22bfloat16}`

**Rule**: When authoring fp8 / mxfp8 / mxfp4 kernels on A5, the **canonical type names** are:

| Type | C++ name | Bits | Use |
|---|---|---:|---|
| FP8 E4M3FN (Finite, No-NaN representation per OFP8) | `fp8_e4m3fn_t` | 8 | Standard FP8 path (training-grade) |
| FP8 E5M2 | `fp8_e5m2_t` | 8 | Standard FP8 path (wider range, less precision) |
| HiFloat8 | `hifloat8_t` | 8 | Ascend-specific FP8 variant (e8m0-style range) |
| FP4x2 E1M2 | `fp4x2_e1m2_t` | 4+4 packed | mxfp4 path (pair of fp4 in one byte) |
| FP4x2 E2M1 | `fp4x2_e2m1_t` | 4+4 packed | mxfp4 path (different bit allocation) |
| Scale factor | `fp8_e8m0_t` | 8 (e8m0 exponent-only) | MicroScaling shared-scale matrix |

**Why this matters**:
- **All five are A5-new types** — V220 (A3) has none of them. There is NO direct port path from A3 for kernels using these; they must be authored fresh.
- **FP4 types are PACKED** — two fp4 values in one byte. Reg-based Cast intrinsics take a `_v2/_v3/_v4` index argument selecting which of 4 sub-positions to write within a VL.
- **fp8_e8m0_t is exponent-only** — used as the per-tile scale factor in MicroScaling format (see OL-145). Range 2⁰ to 2²⁵⁴.

**A5 Cast intrinsics** (reg-based vector, all live in `AscendC::MicroAPI`):

| Direction | API root | Notes |
|---|---|---|
| `float` → `fp8_e4m3fn_t` | `asc_float2e4m3` | Single-step, RoundMode variants `_rn / _rna / _rd / _ru / _rz` |
| `float` → `fp8_e5m2_t` | `asc_float2e5m2` | Same |
| `fp8_e4m3fn_t` → `float` | `asc_e4m32float` | Dequant single-step |
| `fp8_e5m2_t` → `float` | `asc_e5m22float` | Dequant single-step |
| `bfloat16_t` → `fp4x2_e1m2_t` | `asc_bfloat162e1m2x2_{rn,rna,rd,ru,rz}[_v2..v4]` | Packed; `_vN` selects 1 of 4 output positions |
| `bfloat16_t` → `fp4x2_e2m1_t` | `asc_bfloat162e2m1x2_*` | Packed; same variant pattern |
| `fp4x2_e1m2_t` → `bfloat16_t` | `asc_e1m2x22bfloat16` | Unpack to bf16 |
| `fp4x2_e2m1_t` → `bfloat16_t` | `asc_e2m1x22bfloat16` | Unpack to bf16 |

**High-level MicroAPI Cast wrapper** (preferred over raw `asc_*` intrinsics for production code):

```cpp
// FP32 → FP8 E5M2 (SAT mode — quantization, prevents overflow)
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_FLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::SAT,         // quant → saturate
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT
};
AscendC::MicroAPI::RegTensor<fp8_e5m2_t> dst;
AscendC::MicroAPI::Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(dst, src, preg);

// FP32 → HiFloat8 (NO_SAT — HiFloat8 has built-in range guard)
constexpr AscendC::MicroAPI::CastTrait CAST_FP32_TO_HIFLOAT8 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,      // hifloat8 = no-sat per CANN team
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_ROUND
};
```

**SatMode rule** (from PR 103 l2-guide §109-112):
- **量化 (high→low precision)**: `SatMode::SAT` — saturate at narrow-type extremes, prevent overflow
- **反量化 (low→high precision)**: `SatMode::NO_SAT` — wider type holds any narrow value, no overflow possible
- **Exception**: `hifloat8_t` quant uses `NO_SAT` — its internal representation already handles range, additional saturation would be wrong

**Storage after quant** (see OL-146 for Pack details):
- FP8 / HiFloat8 → 1-step Cast then 2-level `Pack` to pack 4 lanes per register-width, then `DataCopyUnAlign` to GM
- FP4x2 already packed pre-Cast (2 fp4 per byte), still needs `Pack` for register-wide compression

**L1-path standard SIMD Cast is also supported on dav_3510** (not only the L2 MicroAPI form): the standard 4-arg `Cast(dst_fp32, src_fp8, RoundMode::CAST_NONE, count)` overload is implemented for `fp8_e4m3fn_t ↔ float` and `hifloat8_t ↔ float` dtype-pairs in the dav_3510 vconv codegen. This means L1 Memory-based first-pass kernels can dequant/quant fp8 WITHOUT entering `__VEC_SCOPE__` + `RegTensor` + `CastTrait` discipline. The choice between L1 standard SIMD Cast and the L2 MicroAPI `asc_e4m32float` / `asc_float2e4m3` form is an L1/L2 perf decision (per OL-152), NOT a correctness one. Reach for L2 MicroAPI once perf profiling identifies the dequant/quant as the hot path.

Supported standard-SIMD Cast dtype-pairs in dav_3510 vconv codegen (verified via SDK header read at `<CANN>/x86_64-linux/asc/impl/basic_api/dav_3510/kernel_operator_vec_vconv_impl.h` lines 1361, 1369, 1826, 1834): `Tuple<float, fp8_e4m3fn_t>`, `Tuple<fp8_e4m3fn_t, float>`, plus hifloat8 variants. `fp8_e5m2_t` and `fp4x2_*` are **NOT** in the standard-SIMD Cast support list — those still require the L2 MicroAPI path.

**Evidence**:
- PR 103 l2-guide §354-410 codifies single-step FP32→FP8/HiFloat8 paths (with concrete 4-arm `if constexpr` template specialization)
- PR 103 imports `bfloat162e1m2x2` (mxfp4 cast) intrinsic doc — confirms public availability
- All 8 reg-vector Cast intrinsics ship in `asc-devkit/docs/api/context/c_api/reg/reg_vector/` (public, doc-grade)
- MxFp8LayerNorm kw-1 (2026-05-21, Ascend950PR_957c, CANN 9.1.0.B010): standard SIMD `Cast(xF32, xDataLocal.ReinterpretCast<fp8_e4m3fn_t>(), RoundMode::CAST_NONE, D)` compiled cleanly first try and Pass-A 8/8 + Pass-B 11/11 PASS — confirms L1 path viable for fp8_e4m3fn ↔ float dequant without forcing MicroAPI scoping at first authoring. SDK-header probe before coding (OL-130 lookup chain step 2) saved an estimated 1+ compile-fix iters that would otherwise have gone into MicroAPI scoping setup.

**Other instances (predicted)**: applies to every fp8 / mxfp8 / mxfp4 kernel we ship — `flash_attention_score`, `flat_quant`, `grouped_matmul_swiglu_quant`, `fused_quant_mat_mul`, plus future kernels. Also applies to standalone Cast / `Pack` test fixtures.

**Cross-reference**:
- OL-145 (MicroScaling format — uses `fp8_e8m0_t` scale)
- OL-146 (CastTrait constants table)
- P-P93 (CPU reference clamp rule — narrow-float ranges)
- P-P94 (MERE/MARE precision thresholds: FP8 E4M3 = 2⁻³, FP8 E5M2 = 2⁻²)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-144（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

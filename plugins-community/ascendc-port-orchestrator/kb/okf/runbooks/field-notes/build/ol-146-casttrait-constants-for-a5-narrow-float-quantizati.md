---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CastTrait constants for A5 narrow-float quantization — full reference table [V351, microapi-cast]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,fused-quant"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,fused-quant"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-146
timestamp_inferred: true
tags: [ascendc, ol-146]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=quant,fused-quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §38-112, §191-228, §354-410`

**Rule**: On A5, type-conversion via MicroAPI `Cast<DST, SRC, TRAIT>(dst, src, maskReg)` requires a `constexpr CastTrait` template-argument with four fields. The trait constants are **direction-specific** and must follow the SatMode rule (quant=SAT, dequant=NO_SAT, except hifloat8 which is always NO_SAT due to its internal range encoding).

**Why this matters**:
- A3's `Cast(dst, src, RoundMode::CAST_RINT, count)` style is **deprecated on A5**. Calls compile (vendor primitives still expose the 4-arg form) but use suboptimal default SatMode and lose explicit control.
- Each direction has a canonical trait; mixing them silently picks wrong rounding / saturation and produces precision drift undetectable by build-time checks.

**Reference table — canonical CastTrait constants** (copy-paste these into kernel header):

```cpp
namespace AscendC::MicroAPI {
namespace { // file-scope

// ============= Common widening (dequant-direction) =============
constexpr CastTrait CAST_B16_TO_B32 = {           // half/bf16 → fp32
    RegLayout::ZERO, SatMode::UNKNOWN,
    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// ============= Common narrowing (quant-direction) =============
constexpr CastTrait CAST_FP32_TO_FP16 = {
    RegLayout::ZERO, SatMode::NO_SAT,             // fp32→fp16 has no real overflow risk at our usual ranges
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr CastTrait CAST_FP32_TO_BF16 = {
    RegLayout::ZERO, SatMode::NO_SAT,
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

// ============= A5 new narrow floats (quant direction) =============
constexpr CastTrait CAST_FP32_TO_FLOAT8 = {       // fp32 → fp8_e4m3fn OR fp8_e5m2
    RegLayout::ZERO, SatMode::SAT,                // SAT — clamp at fp8 range
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr CastTrait CAST_FP32_TO_HIFLOAT8 = {     // fp32 → hifloat8
    RegLayout::ZERO, SatMode::NO_SAT,             // HiFloat8 internal range guard — DO NOT additionally saturate
    MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};

// ============= A5 new narrow floats (dequant direction) =============
constexpr CastTrait CAST_FLOAT8_TO_FP32 = {       // fp8_e4m3fn OR fp8_e5m2 → fp32
    RegLayout::ZERO, SatMode::NO_SAT,             // dequant always NO_SAT
    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

constexpr CastTrait CAST_HIFLOAT8_TO_FP32 = {     // hifloat8 → fp32
    RegLayout::ZERO, SatMode::NO_SAT,
    MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

// ============= Integer-int8 quant chain (three-step) =============
constexpr CastTrait CAST_FP32_TO_INT16 = {        // step 1 of FP32→INT8
    RegLayout::ZERO, SatMode::SAT,                // SAT
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr CastTrait CAST_INT16_TO_FP16 = {        // step 2
    RegLayout::ZERO, SatMode::NO_SAT,             // int16 fits in fp16 exponent range — no saturation
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr CastTrait CAST_FP16_TO_INT8 = {         // step 3
    RegLayout::ZERO, SatMode::SAT,                // SAT — clamp at int8 ±127
    MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

} // namespace
} // namespace AscendC::MicroAPI
```

**The 4 fields explained**:

| Field | Allowed values | Default | Meaning |
|---|---|---|---|
| `RegLayout` | `ZERO` | `ZERO` | Register-bit layout convention — always `ZERO` in current arch35 codegen |
| `SatMode` | `SAT`, `NO_SAT`, `UNKNOWN` | `UNKNOWN` | Saturation on overflow. **SAT for quant, NO_SAT for dequant** (hifloat8 quant is the exception) |
| `MaskMergeMode` | `ZEROING` | `ZEROING` | Predicate-masked lanes — zero them. Always `ZEROING` in arch35 codegen |
| `RoundMode` | `CAST_RINT` (round-to-nearest-even), `CAST_ROUND` (round-half-away-from-zero), `CAST_FLOOR`, `CAST_CEIL`, `CAST_TRUNC`, `UNKNOWN` | `UNKNOWN` | IEEE rounding. `CAST_RINT` is default for fp32→narrow; `CAST_ROUND` for hifloat8; `CAST_TRUNC` for explicit truncation |

**FP32 → INT8 three-step quantization** (A5 path — uses INT16 intermediate):

```cpp
// On A5 — NO PipeBarrier between steps, all in __VEC_SCOPE__
AscendC::MicroAPI::RegTensor<int16_t> tmpInt16;
AscendC::MicroAPI::RegTensor<half> tmpHalf;
AscendC::MicroAPI::RegTensor<int8_t> quantInt8;

AscendC::MicroAPI::Cast<int16_t, float, CAST_FP32_TO_INT16>(tmpInt16, src, preg);
AscendC::MicroAPI::Cast<half, int16_t, CAST_INT16_TO_FP16>(tmpHalf, tmpInt16, preg);
AscendC::MicroAPI::Cast<int8_t, half, CAST_FP16_TO_INT8>(quantInt8, tmpHalf, preg);

// Two-level Pack to compress
Pack((RegTensor<uint16_t>&)tmpInt16, (RegTensor<uint32_t>&)quantInt8);
Pack((RegTensor<uint8_t>&)quantInt8, (RegTensor<uint16_t>&)tmpInt16);

// Non-aligned store
DataCopyUnAlign(output, quantInt8, uValue, postUpdateStride);
```

**vs A3 path** (uses INT32 intermediate + PipeBarrier + SetDeqScale):

```cpp
// On A3 — explicit sync, SetDeqScale, no Pack
Cast(src.ReinterpretCast<int32_t>(), src, RoundMode::CAST_RINT, size);
PipeBarrier<PIPE_V>();
SetDeqScale((half)1.0f);
PipeBarrier<PIPE_V>();
Cast(src.ReinterpretCast<half>(), src.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
PipeBarrier<PIPE_V>();
Cast(dst, src.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
// DataCopyPad (A3) — not DataCopyUnAlign
```

**Key A3/A5 differences for INT8 quant**:
1. **Intermediate type**: A3 uses INT32 (4-byte alignment requirement); A5 uses INT16 (saves register pressure)
2. **Sync**: A3 needs `PipeBarrier<PIPE_V>` between each step; A5 in `__VEC_SCOPE__` does not
3. **Helper APIs**: A3 needs `SetDeqScale((half)1.0f)`; A5 doesn't
4. **Pack**: A3 doesn't need Pack; A5 mandates two-level Pack before store
5. **Store**: A3 uses `DataCopyPad`; A5 uses `DataCopyUnAlign`

**FP32 → FP8 / HiFloat8 paths** (A5 single-step):

```cpp
// FP32 → FP8 E5M2 (or E4M3FN — same CAST_FP32_TO_FLOAT8 trait)
RegTensor<fp8_e5m2_t> quantFp8;
Cast<fp8_e5m2_t, float, CAST_FP32_TO_FLOAT8>(quantFp8, src, preg);
Pack((RegTensor<uint16_t>&)tmpHalf, (RegTensor<uint32_t>&)quantFp8);
Pack((RegTensor<uint8_t>&)quantFp8, (RegTensor<uint16_t>&)tmpHalf);
DataCopyUnAlign(output, quantFp8, uValue, postUpdateStride);

// FP32 → HiFloat8 (note NO_SAT + CAST_ROUND)
RegTensor<hifloat8_t> quantHi8;
Cast<hifloat8_t, float, CAST_FP32_TO_HIFLOAT8>(quantHi8, src, preg);
// ... same Pack + DataCopyUnAlign
```

**Detection signature** (anti-pattern catches):
- Search for `Cast<.*, .*>(.*, .*, RoundMode::.*, .*size)` in arch35/ — old A3-style 4-arg form, should be 3-arg MicroAPI form
- Search for `SetDeqScale` in arch35/ — A3 helper, no role on A5
- Search for `PipeBarrier<PIPE_V>` inside `__VEC_SCOPE__` blocks — sync inside vec-scope is wrong; remove

**Evidence**:
- PR 103 l2-guide §38-112 codifies the 6 most-common traits
- PR 103 l2-guide §400-410 documents the complete quant-path-vs-Pack table including all narrow-float destinations

**Other instances (predicted)**: applies to every fp8/mxfp8/mxfp4 kernel + every quant op (rms_norm_quant, group_norm_silu_quant, add_rms_norm_quant, flat_quant, grouped_matmul_swiglu_quant, fused_quant_mat_mul).

**Cross-reference**:
- OL-144 (narrow-float type family)
- OL-145 (MicroScaling format — uses these casts pre-store)
- OL-152 (Memory↔Register API mapping — Batch 3) — full A3→A5 API substitution table
- P-P93 (CPU clamp rule — narrow-float quant test reference)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-146（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

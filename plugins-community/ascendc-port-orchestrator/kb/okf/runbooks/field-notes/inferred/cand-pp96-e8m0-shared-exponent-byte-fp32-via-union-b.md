---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "E8M0 shared-exponent byte → fp32 via union bit-reinterpret in AscendC scalar context"
description: "applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=15.x; op_class=mx-quant,mxfp8,mxfp4,microscaling-decode; kernel_type=AIV scalar pipe verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=15.x; op_class=mx-quant,mxfp8,mxfp4,microscaling-decode; kernel_type=AIV scalar pipe"
confidence: inferred
status: stub
original_id: CAND-PP96
timestamp_inferred: true
tags: [candidate, inferred, fp8_e8m0_t, union, memcpy, cast, cand-pp96]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=15.x; op_class=mx-quant,mxfp8,mxfp4,microscaling-decode; kernel_type=AIV scalar pipe`
`verified_on: soc=Ascend950PR_957c; cann=9.1.0.B010 (MxFp8LayerNorm — single op evidence)`
`unverified_on: soc=Ascend910_V220 (no E8M0 type support per OL-144); op_class=mxfp4-matmul, mxfp8-attention, future MX-format consumers`

**Principle**: when decoding MicroScaling E8M0 shared-exponent scale bytes (`fp8_e8m0_t`, see OL-144) in scalar context on the AIV pipe, libm is unavailable so `ldexpf(1.0f, byte - 127)` cannot be used. The canonical primitive is a union-based bit reinterpret that constructs the IEEE-754 fp32 with the byte placed directly into the biased-exponent field. For each byte `b ∈ [0, 254]`, the corresponding scale `2^(b - 127)` is exactly representable as a normal fp32 (sign=0, biased_exp=b, mantissa=0).

**Concrete anchor** (3-line primitive — drop-in for any kernel that consumes per-block E8M0 scales):

```cpp
__aicore__ inline float E8m0_byte_to_scale(uint8_t b) {
    union { uint32_t u; float f; } cvt;
    cvt.u = static_cast<uint32_t>(b) << 23;  // IEEE 754: sign=0, biased_exp=b, mantissa=0 → 2^(b-127)
    return cvt.f;
}
```

The `union` bit-cast is well-defined in AscendC scalar code under bisheng (CANN 9.1.0.B010 confirmed); we don't need to fall back to `memcpy`-style type punning. Compiles cleanly on the AIV scalar pipe.

**Where this bites if missing**: workers reaching for `ldexpf(1.0f, b - 127)` get an "undefined symbol" link error (libm not available on AIV); workers reaching for `Cast` from uint8 to float get **PB-26 territory** (`Cast<float, uint8_t>` is unsupported, silent garbage). The union bit-reinterpret is the only correct primitive for E8M0 → fp32 in kernel-side scalar code.

**Why this is reusable beyond LayerNorm**: any MX-format consumer needs per-block scale broadcast. Examples:
- mxfp8 attention (Q/K/V quantized with E4M3 mantissa + E8M0 shared scale per 32-element block).
- mxfp4 matmul-class kernels (FP4x2 packed weights + E8M0 per-block scale per OL-144 / OL-145).
- Any future MX-format dequant where the per-block scale is consumed scalar-wise (not via the `fp8_e8m0_t` → bf16 reg-vector Cast in `__VEC_SCOPE__`).

**Anti-patterns**:
- ❌ `Cast<float, uint8_t>(scale_fp32, scale_u8, RoundMode::CAST_NONE, 1)` — uint8 → float Cast is unsupported on A5 (PB-26 family), silently produces garbage.
- ❌ `ldexpf(1.0f, byte - 127)` — libm unavailable on AIV.
- ❌ `pow(2.0f, byte - 127)` — same libm issue, plus floating-point pow is slow and loses bit-exactness vs the exact integer-bit construction.
- ❌ Computing the scale on host and broadcasting an already-fp32 scale tensor — works but doubles input bandwidth on every quant kernel; the kernel-side bit reinterpret is strictly cheaper.

**Evidence**:
- MxFp8LayerNorm kw-1 (2026-05-21 Ascend950PR_957c, CANN 9.1.0.B010): primitive used in step 3 (per-block scale application) — 24 to 128 scalar calls per row depending on D, each computing one `2^(b - 127)` scale exactly. Pass-A 8/8 + Pass-B 11/11 PASS_WITHIN_TOLERANCE confirms numeric correctness (would fail dequant precision immediately if the bit-reinterpret produced wrong scales).

**Promotion gate**: 2+ op evidence required. Next candidate: any future mxfp8 / mxfp4 op landing with the same scalar-pipe E8M0 decode pattern (e.g. an independent fused mxfp8-attention prototype, OR an mxfp4-matmul kernel decoding scales scalar-wise). Then promote to `patterns/domains/quant.md` as a P-Pxx.

**Cross-ref**:
- OL-144 (A5 narrow-float datatype family — `fp8_e8m0_t` exponent-only scale type definition)
- OL-145 (MicroScaling format — per-block 32-element scale convention)
- PB-26 (`Cast<float, uint8_t>` unsupported — the trap this primitive avoids)
- OL-152 (L2 register-based path uses reg-vector Cast for fp8_e8m0_t → bf16; this candidate is the L1-scalar-pipe counterpart for the same decode)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP96，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

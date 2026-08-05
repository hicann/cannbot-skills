---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`static_cast<float>(bfloat16_t)` fails in bisheng"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-5
timestamp_inferred: true
tags: [static_cast, bfloat16_t, float, half, ascendc, ec-5]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: not support bf16 type cast
  ```
  or:
  ```
  error: static_cast from 'bfloat16_t' to 'float' is not allowed
  ```
- **Root cause**: Bisheng compiler (CANN 9.0.0 and 9.0.T501) does not support scalar `static_cast` between `bfloat16_t` and `float` in either direction. The `half` (fp16) type works fine with `static_cast`. This is a known bisheng limitation (PB-4 in PLATFORM_BUGS.md).
- **Fix (SIMT kernel — use bit-manipulation)**:
  ```cpp
  // BEFORE (fails):
  bfloat16_t val = input[i];
  float fval = static_cast<float>(val);    // ❌ bisheng rejects this

  // AFTER (bit-manipulation workaround):
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }

  template <>
  __aicore__ inline float simt_to_float<bfloat16_t>(bfloat16_t v) {
    uint16_t bits;
    __builtin_memcpy(&bits, &v, sizeof(bits));
    uint32_t f32bits = static_cast<uint32_t>(bits) << 16;
    float result;
    __builtin_memcpy(&result, &f32bits, sizeof(result));
    return result;
  }

  // Reverse: float → bfloat16_t
  template <typename T>
  __aicore__ inline T simt_from_float(float v) { return static_cast<T>(v); }

  template <>
  __aicore__ inline bfloat16_t simt_from_float<bfloat16_t>(float v) {
    uint32_t f32bits;
    __builtin_memcpy(&f32bits, &v, sizeof(f32bits));
    uint16_t bits = static_cast<uint16_t>(f32bits >> 16);  // truncate
    bfloat16_t result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
  }
  ```
- **Fix (SIMD kernel — use Cast intrinsic)**:
  ```cpp
  // Cast(bf16→float) is lossless and works:
  Cast(floatBuf, bf16Buf, RoundMode::CAST_NONE, count);
  float w = floatBuf.GetValue(i);
  ```
- **WARNING**: `Cast(bf16→half)` is LOSSY — bf16 exponent=8bit overflows half exponent=5bit, producing `inf` for large values. Always cast bf16→float (lossless).
- **Related**: P-P27 (bf16 scalar via Cast + GetValue)

<!-- 迁移自 porter kb/target/ascendc/（EC-5，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

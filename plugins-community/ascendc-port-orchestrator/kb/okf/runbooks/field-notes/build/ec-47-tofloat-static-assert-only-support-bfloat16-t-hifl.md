---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`ToFloat<>` static_assert \"only support bfloat16_t/hifloat8_t/fp8_*/fp4_*\" after BF16 guard removal [V351, port_a3_to_a5]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-47
timestamp_inferred: true
tags: [ascendc, ec-47]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0; op=ctc_loss_v3`
`source: PR 103 SKILL.md §455-471 (cites our ctc_loss_v3 as the canonical example)`

**Symptom**:
```
kernel_scalar_convert.h: error: static assertion failed: ToFloat only support
bfloat16_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/fp4x2_e1m2_t/fp4x2_e2m1_t data type on current device!
```
Call chain: `ToFloat<>(val)` → `Cast<>(bVal)` → `static_assert` fail.

**Root cause**: On A5, the templated `ToFloat<>` helper only specializes for the low-precision narrow-floats. When A3-source code removed its BF16 conditional-compile guards (per OL-142 / EC-49) but the underlying tensor type changed in the process, the call to `ToFloat<>` may now receive a wider type (`half` / `float`) that `ToFloat<>` refuses by design.

**Concrete example (ctc_loss_v3, cited in PR 103)**:
```cpp
// BEFORE — under #if guarded BF16 path, ToFloat sees bfloat16_t (OK)
logProbBlank = ToFloat(logProbBlankTensor.GetValue(0));

// AFTER guard removal — type chain shifts; ToFloat now sees half (REJECTED)
// Fix: explicit ReinterpretCast<bfloat16_t>() before ToFloat
logProbFirstChar = ToFloat(logProbFirstTensor
    .template ReinterpretCast<bfloat16_t>()
    .GetValue(0));
```

**Fix pattern** (universal):
- Before calling `ToFloat<>(x)`, insert `.template ReinterpretCast<bfloat16_t>()` if the underlying memory holds BF16-encoded bits but the static type is `half`/`float`/etc.
- For genuine `half` / `float` values, use plain `static_cast<float>(x)` instead of `ToFloat<>`.

**Detection signature**: search for `ToFloat<` calls in newly-ported arch35/ kernels; cross-check against `LocalTensor<T>` declarations to confirm T is in the allowed set OR a `ReinterpretCast` is present.

**Evidence**:
- ctc_loss_v3 (2026-05-13): we hit this during the L1 port; the PR 103 authors saw our archive and cite it explicitly in their fast-track table
- PR 103 EC table line 452: "ToFloat<> static_assert 失败 | A5 上 ToFloat 仅支持 BF16/FP8/HiFloat8 等新类型"

**Mitigation gate**: post-worker `aog-self-critic` should grep arch35/ kernel headers for `ToFloat<` and emit a soft-warning if the surrounding `LocalTensor` type is `half` / `float` without `ReinterpretCast`.

**Cross-reference**: EC-49 (BF16 guard removal) often causes this; the fix sequence is "remove guard → recompile → if static_assert fires, apply this fix".

<!-- 迁移自 porter kb/target/ascendc/（EC-47，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

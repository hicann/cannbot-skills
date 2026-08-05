---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A5 overflow-mode SPR toggle for RMSNorm/Softmax perf (save/restore, arch-guarded, bounded-output only)"
description: "A5 exposes a global overflow-detection SPR (ID 60) that can be temporarily disabled to skip per-instruction overflow checks on bounded-output algorithms (RMSNorm, Softmax); must be saved before disable and restored after, guarded with __NPU_ARCH__ == 3510."
original_id: OL-148
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-148, microapi-spr, overflow-mode, norm-softmax]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

A5 exposes a global overflow-detection SPR (Special Purpose Register, ID 60) that can be
temporarily disabled to skip per-instruction overflow checks. For ops whose **output range is
mathematically bounded** (RMSNorm → bounded by ratio, Softmax → in [0,1]), disabling the check
provides perf without correctness risk. The SPR is global state — it MUST be saved before
disable and restored after.

**Applies to** `soc=Ascend950PR; cann=9.0.0; bisheng=all;
op_class=norm,softmax,attention-with-softmax`. Verified on Ascend950PR / cann 9.0.0. Source:
PR 103 `l2-register-based-guide.md` §156-185.

### Why this matters

- Default overflow detection adds per-instruction cycles on the VEC pipeline — a measurable
  bottleneck on long norm/softmax inner loops.
- Disabling globally is unsafe (other concurrent compute paths in the same kernel may need
  it). Localized save/restore around the bounded section is the canonical pattern.
- The wrapper MUST be `__NPU_ARCH__ == 3510` guarded — A3 SPR semantics differ; the same code
  on V220 silicon may crash or silently misbehave.

### Canonical pattern (RMSNorm fast-path)

```cpp
#define GLOBAL_OVERFLOW_MODE_CTRL 60

template <typename T>
__aicore__ inline void RMSNormProcess(...) {
    // Save current SPR value (only on A5)
#if (__NPU_ARCH__ == 3510)
    int64_t globalOriOverflowMode =
        AscendC::GetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>();
#endif

    // Disable overflow detection — safe because RMSNorm outputs are bounded
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(0);
#endif

    // ... bounded-range computation (RMSNorm body) ...

    // RESTORE — mandatory, never elide
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<GLOBAL_OVERFLOW_MODE_CTRL, GLOBAL_OVERFLOW_MODE_CTRL>(globalOriOverflowMode);
#endif
}
```

### Three discipline rules

1. **Must save/restore** — the SPR is global state; an unbalanced disable leaks into subsequent
   unrelated compute and silently changes its correctness.
2. **Must guard with `__NPU_ARCH__ == 3510`** — A3 has different SPR semantics; same code on
   V220 may crash. (Note: V220 BF16 guards 3003/3113 are NEGATIVE; this guard is POSITIVE,
   emitted only on A5.)
3. **Bounded-output justification required** — disabling is safe ONLY if the entire bracketed
   section produces values within float-representable range. Apply ONLY to RMSNorm / Softmax /
   sigmoid / similar bounded-output algorithms. Do NOT apply to arbitrary vector compute.

### Anti-patterns

```cpp
// WRONG — no restore (leak)
AscendC::SetCtrlSpr<60, 60>(0);
// ... compute ...
// (forgot to restore)

// WRONG — no arch guard (crashes V220 if same code compiled for both)
AscendC::SetCtrlSpr<60, 60>(0);  // A3 SPR 60 has different meaning

// WRONG — applied to unbounded compute (silently produces inf/NaN)
AscendC::SetCtrlSpr<60, 60>(0);
Mul(big_a, big_b, count);        // (source text truncated here)
```

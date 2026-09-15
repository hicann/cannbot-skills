---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Member variable shadowing in AscendC `Init()` — `type name_ = expr` creates a LOCAL that shadows the class member, leaving the member at its initializer value (typically 0)"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-264
timestamp_inferred: true
tags: [hwnumaligned_, ascendc, ol-264]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (group_norm_silu D>1 fix, 2026-06-26)`

### Principle

In C++, writing `uint32_t hwNumAligned_ = CeilDivT<uint32_t>(hwNum_, 8) * 8;` inside `Init()` is a **declaration + initialization** — it creates a NEW local variable `hwNumAligned_` that shadows the class member `hwNumAligned_` (initialized to 0 via `uint32_t hwNumAligned_{0}` in the class body). The class member remains 0 for all downstream use. The local variable goes out of scope when `Init()` returns.

This is especially dangerous in AscendC because:
1. `Init()` is called once before `Process()` — the member appears to be "initialized" but actually stays at 0
2. If the member controls UB offset computation (e.g., `ubOff = d * hwNumAligned_`), the result is always `d * 0 = 0` — all slices write to the same UB address
3. The kernel compiles and runs clean — no compiler warning, no runtime error, just silently wrong output

### Fix

Remove the type prefix:

```cpp
// BEFORE (WRONG — local variable shadows class member):
void Init(...) {
    uint32_t hwNumAligned_ = CeilDivT<uint32_t>(hwNum_, 8) * 8;  // LOCAL, shadows member
    // ... member hwNumAligned_ is STILL 0
}

// AFTER (correct — assignment to class member):
void Init(...) {
    hwNumAligned_ = CeilDivT<uint32_t>(hwNum_, 8) * 8;  // member assignment
    // assertion: hwNumAligned_ > 0 for any hwNum_ > 0
}
```

### Prevention (defensive coding)

Add a post-Init assertion in Process() for any critical member used in offset math:

```cpp
__aicore__ void Process(...) {
    if (hwNumAligned_ == 0) {
        // UB: Init() bug — member not assigned
        // Fallback: compute inline as safety net
        hwNumAligned_ = CeilDivT<uint32_t>(hwNum_, 32/sizeof(T)) * (32/sizeof(T));
        // Or: pipe a FATAL diagnostic then return
    }
    // ...
}
```

### Evidence

group_norm_silu precision fix (2026-06-26, A5/Ascend950PR, CANN 9.0.0):
- The kernel had `uint32_t hwNumAligned_ = CeilDivT<uint32_t>(hwNum_, 8) * 8;` in `Init()`
- The class had `uint32_t hwNumAligned_{0};` as a member declaration
- For D>1 cases, `ubOff = d * hwNumAligned_` always computed `d * 0 = 0` — all D output slices wrote to UB offset 0
- Result: all D output slices were identical (all matched the last d's reference because the last slice's compute overwrote the buffer and all slices read from offset 0)
- SAME session also had a Duplicate red herring (EC-80) — Duplicate semantics were wrong too, but fixing Duplicate alone still produced garbage because hwNumAligned_=0 meant ubOff=0 regardless
- The ONE-CHARACTER fix: remove `uint32_t` → `hwNumAligned_ = CeilDivT(...);`
- After fix: all 21 precision tests PASS (fp16/bf16 0 diff, fp32 <2e-6)

### Other instances (predicted)

Any AscendC kernel where a member variable is "initialized" in Init() with a type prefix — this is a general C++ footgun, not AscendC-specific. Common victims:
- `hwNumAligned_` / `ubStride_` / `tileSize_` (UB layout members)
- `totalBlocks_` / `blocksPerRow_` (launch-dimension members)
- `gammaSize_` / `betaSize_` (affine-parameter dimension members)
- Any member used in offset or size computation in Process()

### Cross-references

- EC-80 (Duplicate semantics — the OTHER bug in the same group_norm_silu session; session had BOTH Duplicate AND shadowing, and both needed fixing)
- OL-215 (UB buffer layout validator — pre_build_check.py would have caught hwNumAligned_==0? No: it's a runtime-init value, not statically analyzable from the header alone)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-264（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

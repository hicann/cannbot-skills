---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Use bit-pattern -inf (0xFF800000), not -FLT_MAX literal, for fp32 reduction-mask sentinels"
description: "When the reference masks with float('-inf'), the kernel sentinel must be IEEE-754 -inf (0xFF800000), not -3.402823e+38f (=-FLT_MAX, 0xFF7FFFFF); the two differ only on fp32, giving fp32-only failures with max_diff ≈ ±FLT_MAX/2."
phenomenon: precision_issue
signal:
  - "fp32-only failures with max_diff magnitude ≈ ±FLT_MAX/2 (≈ ±1.7e+38, often surfacing as ~±4e+31 on mean-of-N reductions); torch.unique(ref) includes -inf while torch.unique(cand) includes -3.4028230607370965e+38"
confidence: single_run
original_id: OL-119
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, kernel-design, ol-119, sentinel, neg-inf]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
fp32-only failures (fp16/bf16 bands PASS) with one of these tells:
- `max_diff` magnitude ≈ ±FLT_MAX/2 (typically ±3-5e+31 on aggregate reductions, larger on raw values)
- `torch.unique(ref)` includes `-inf`; `torch.unique(cand)` includes `-3.4028230607370965e+38`
- an all-same-value mean = max/2 pattern in cand on rows where the reference produces `-inf`

Loaded by aog-kernel-worker (Phase B: any masked reduction / select-with-sentinel / topk init) and aog-precision-probe.

## 根因 / 教训
When a kernel uses a sentinel to mask positions in a `Select<float>` / `Duplicate<float>` for a reduction (mask-then-Max, mask-then-min, topk init, attention mask), and the reference uses `float('-inf')` (or `torch.finfo(fp32).min`, which IS `-inf` for float types), the sentinel MUST be the IEEE-754 representation of `-inf` (bit pattern `0xFF800000`), NOT the literal `-3.402823e+38f` (which equals `-FLT_MAX`, bit pattern `0xFF7FFFFF`).

These are TWO DIFFERENT VALUES. They saturate to the same down-cast on fp16/bf16 (both narrower than fp32 max), so fp16/bf16 PASS while fp32 produces `max_diff` around ±FLT_MAX/2 (≈ ±1.7e+38, often ≈ ±4.05e+31 on mean-of-N reductions or larger).

### Concrete anchor
```cpp
// BAD: -3.402823e+38f is -FLT_MAX (0xFF7FFFFF), NOT -inf
constexpr float NEG_INF = -3.402823e+38f;
Duplicate<float>(buf, NEG_INF, count);
Select<float>(dst, mask, valid_path, NEG_INF, count);

// GOOD: explicit bit-pattern union for true -inf (0xFF800000)
inline __aicore__ float NegInfF32() {
    union { uint32_t u; float f; } v;
    v.u = 0xFF800000U;
    return v.f;
}
Duplicate<float>(buf, NegInfF32(), count);
Select<float>(dst, mask, valid_path, NegInfF32(), count);
```
(Alternative: `-__builtin_inff()` if bisheng accepts it — verify before relying on it.)

### Detection workflow
1. Symptom triage: failure band is fp32-only; max_diff looks like ±FLT_MAX/2 or a related power of 2 above ±1e+30.
2. Probe a single failing case + dump `torch.unique(ref).cpu()[:5]` and `torch.unique(cand).cpu()[:5]`. Look for `-inf` on ref and `-3.4028230607370965e+38` on cand.
3. Grep kernel source: `grep -E '\-3\.4028|FLT_MIN|FLT_MAX|0xFF7FFFFF' kernel/*.h *.cpp` — every match is a candidate fix site.
4. Replace literals with the `NegInfF32()` helper. Rebuild + reverify.

### Evidence
- 26_MoeGroupScoreAggregationAndMasking kw-1 iter 3 (2026-05-02): iter 1 had 22 fp32 cases FAIL with `max_diff = 4.05e+31`, mis-diagnosed as uninit memory (added a one-time `Duplicate(NEG_INF)` before `Select`) → iter 2 still 4.05e+31. iter-2 probe: ref unique `[-inf, ...]`, cand unique `[-3.4028230607370965e+38, ...]` → root-caused to the -FLT_MAX vs -inf sentinel.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Tiling-tensor cache + C++ convenience wrapper — cut host-side overhead in pybind11 kernels"
description: "Cache the tiling NPU tensor keyed on shape/dtype to skip alloc+memcpy+device-copy on hit, plus a C++ convenience fn for layout conversion; +0.13× overall (0.585→0.715× vs CANN native)."
confidence: single_run
original_id: OL-250
classified_by: llm-assisted
timestamp_inferred: true
tags: [host-overhead, optimization, ol-250, tiling-cache, pybind11]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** all pybind11-based A5 custom kernels invoked repeatedly (benchmark warmup+repeats, inference serving), where hidden per-call host overhead compounds.

Two forms of hidden overhead:

1. **Tiling re-allocation + CPU→NPU copy.** Every call allocates a CPU tensor (`torch::empty`), memcpys the tiling struct into it, then `.to(device)`. That is ~5-10 µs of wasted work when shapes haven't changed — tiling is a pure function of `(N, C, inD, inH, inW, outD, outH, outW, dtype)`.
2. **Python-side dispatch overhead.** When the Python wrapper does `x.permute().contiguous().reshape()` before the raw C++ entry, each op crosses the Python→C++ boundary and dispatches through the PyTorch dispatcher. A C++ convenience function that does the same permute internally removes 2-3 crossings per call.

Both fixes are **zero-cost to the kernel** — it receives identical inputs — and they compound.

**Decision rule:**
1. **Always cache the tiling NPU tensor** in a static variable keyed by all shape/dtype params. On a cache hit, skip alloc + memcpy + device copy. Safe because (a) the tiling struct is read-only by the kernel, and (b) `aclrtSynchronizeStream` after each launch guarantees the previous kernel finished reading before the next launch reuses the buffer.
2. **Expose a C++ convenience function** `op_npu(input, params...)` that handles layout conversion internally, so the Python wrapper makes ONE pybind call instead of permute→raw→unpermute across 3+ calls.
3. **These are NOT alternatives to OL-248's raw API** — the raw API remains the canonical measurement path; the convenience wrapper is the production path balancing usability with performance.

### Concrete anchor
```cpp
struct TilingCacheEntry {
    int64_t cache_N, cache_C, cache_inD, cache_inH, cache_inW;
    int64_t cache_outD, cache_outH, cache_outW;
    torch::Tensor tiling_npu;
    int32_t  usedCoreNum;
};
static TilingCacheEntry g_tilingCache = {0};

// Inside op_npu_raw():
bool cacheHit = (N == g_tilingCache.cache_N && C == g_tilingCache.cache_C &&
                 inD == g_tilingCache.cache_inD && /* ... all shape fields ... */);
if (!cacheHit) {
    auto tiling_cpu = torch::empty({sizeof(TilingData)}, ...);
    std::memcpy(tiling_cpu.data_ptr(), &td, sizeof(TilingData));
    // ... .to(device), refresh g_tilingCache ...
}
```

### Evidence
adaptive_avg_pool3d (Ascend950PR, CANN 9.1.T500, 2026-06-23, 50-case wall-clock A/B): **+0.13× overall (0.585 → 0.715× vs CANN native).**

### Related
- OL-248 (raw API — the canonical measurement path this pairs with).

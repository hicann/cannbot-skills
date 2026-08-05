---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Launch a SIMT VF_CALL from inside a SIMD KERNEL_TYPE_AIV_ONLY entry point via GetPhyAddr pointer bridge"
description: "Simt::VF_CALL runs inside an AIV_ONLY Process() with no separate entry or build change — bridge UB/GM via GetPhyAddr() for a SIMT-count then SIMD-merge two-phase kernel. V351-only, not V220/A3."
confidence: single_run
original_id: OL-251
classified_by: llm-assisted
timestamp_inferred: true
tags: [simt, optimization, ol-251, aiv-only, getphyaddr, histogram]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**Principle.** `Simt::VF_CALL<vf_func>(Simt::Dim3{N}, args...)` can be called directly from within a `KERNEL_TYPE_AIV_ONLY` entry point's `Process()` method — **no separate kernel entry point, no `#if __NPU_ARCH__` guard, and no build-system changes**. The SIMD kernel allocates UB buffers via `TQue`, extracts their raw physical addresses via `GetPhyAddr()`, passes them to the SIMT vector function (VF), and after `VF_CALL` returns uses standard SIMD APIs (`SetAtomicAdd` + `DataCopy`, or further VEC compute) on the same UB buffers.

This enables a **two-phase kernel architecture**:
- **Phase 1 (SIMT):** parallel accumulation / counting / index-compute on raw GM/UB pointers using `__simt_vf__` + `LAUNCH_BOUND(N)` + `atomicAdd` + `Simt::ThreadBarrier()`.
- **Phase 2 (SIMD):** merge / normalize / post-process using SIMD `TQue` / `DataCopy` / `SetAtomicAdd` / VEC ops.

**The bridge — extracting raw pointers from SIMD tensor objects:**
- `GlobalTensor<T>::GetPhyAddr(offset)` returns a non-const `__gm__ T*` — pass to the VF as `reinterpret_cast<GM_ADDR>(ptr)` or use the `__gm__ T*` directly.
- `LocalTensor<T>::GetPhyAddr()` returns `uint64_t` (integer physical address) — cast with `reinterpret_cast<__ubuf__ T*>(uint64_val)` to get the UB pointer.

**Decision rule — use this when:**
1. The MAIN compute phase needs SIMT parallelism (many threads doing independent indexed GM r/w), AND
2. the entry point is already `KERNEL_TYPE_AIV_ONLY` (SIMD) — adding a separate SIMT entry point would be invasive, AND
3. the post-compute merge benefits from SIMD APIs (`SetAtomicAdd<T>()` + `DataCopy` for aligned GM merge, or VEC ops for normalization), AND
4. the SIMT phase's UB working set fits a `TQue`-allocated buffer (`bins × sizeof(scalar) ≤ UB capacity`).

**Do NOT use when:**
- The whole kernel is pure SIMT (no SIMD merge) → use the pure-SIMT entry pattern from OL-150.
- The SIMT working set exceeds UB capacity → tile it or go pure-SIMT.
- Target includes V220/A3 → **SIMT is V351-only**; fall back to pure-SIMD.

### Concrete anchor
```cpp
constexpr uint32_t HIST_THREAD_NUM = 512;

__simt_vf__ __aicore__
LAUNCH_BOUND(HIST_THREAD_NUM) inline void hist_count_simt(
    GM_ADDR x_gm_addr,
    __ubuf__ int32_t* local_hist,   // UB pointer passed FROM the SIMD side
    int64_t my_start, int64_t my_count,
    float min_v, float max_v, float inv_bin_size, /* ... */);
```

### Evidence
histogram_v2 SIMT optimization (Ascend950PR, CANN 9.0.0, 2026-06-22). Unverified on Ascend910_V220 (SIMT unsupported on V220 — pattern is V351-only).

### Related
- OL-150 (pure-SIMT entry pattern — use when no SIMD merge is needed).

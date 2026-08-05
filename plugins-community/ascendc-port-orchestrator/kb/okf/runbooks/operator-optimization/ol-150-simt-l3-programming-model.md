---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT (L3) programming model: __simt_vf__ / LAUNCH_BOUND / Simt::VF_CALL for Scatter/Gather"
description: "A5 Vector Core supports SIMT mode (up to 2048 threads, each addressing GM directly), replacing Memory-based Scatter/Gather when index logic is simple, no UB transit is needed, and data volume ≫ 2048; typical 50-200% gain and much shorter kernels."
original_id: OL-150
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-150, simt-l3, scatter-gather, index-arith]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

A5 Vector Core supports SIMT (Single Instruction Multiple Thread) mode — up to 2048 threads
execute the same kernel function, each independently addressing GM. Use SIMT to replace
Memory-based Scatter/Gather kernels when (a) the op has index-based GM r/w, (b) index logic is
simple, (c) no UB transit is needed, and (d) data volume ≫ 2048. Typical gain: 50-200% vs
Memory-based on Scatter/Gather workloads.

**Applies to** `soc=Ascend950PR; cann=9.0.0; bisheng=all;
op_class=scatter,gather,index-arith`. Verified on Ascend950PR / cann 9.0.0. Source: PR 103
`references/l3-simt-optimization-guide.md` §10-122.

### Why this matters

- Memory-based Scatter is GM→UB→compute→UB→GM (4 steps + UB occupancy + TPipe overhead). SIMT
  goes GM→compute→GM (1 step). For pure index-based access, the UB round-trip is pure overhead.
- A3 (V220) has NO SIMT — this is V351-only. Greenfield A5 kernel work for Scatter/Gather
  should default to SIMT, not port Memory-based code first.
- SIMT kernels are MUCH shorter — a typical Memory-based Scatter is ~160 lines; the equivalent
  SIMT is ~86 lines.

### Three core elements

1. **SIMT kernel function** — declared with `__simt_vf__` + `LAUNCH_BOUND(THREAD_NUM)`:

```cpp
constexpr uint32_t SIMT_THREAD_NUM = 2048;

__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void ComputeSimt(
    __gm__ T* input, __gm__ T* output, int32_t totalElements)
{
    // grid-stride loop — each thread handles a stride
    for (int32_t idx = static_cast<int32_t>(Simt::GetThreadIdx());
         idx < totalElements;
         idx += static_cast<int32_t>(Simt::GetThreadNum())) {
        output[idx] = compute(input[idx]);   // direct GM read+write
    }
}
```

- `__simt_vf__` annotates "this function runs as a SIMT vector function".
- `LAUNCH_BOUND(N)` declares max thread count — the compiler uses it for register allocation.
- The body uses `__gm__` raw pointer access (NOT `LocalTensor`, NOT `RegTensor` — those are
  Memory-based / Register-based concepts).

2. **Thread indexing** — `Simt::GetThreadIdx()` / `Simt::GetThreadNum()`:

```cpp
Simt::GetThreadIdx()   // current thread's index, 0 .. threadNum-1
Simt::GetThreadNum()   // total threads at launch (may be < LAUNCH_BOUND)

// Grid-stride loop pattern (handles any element count):
for (int32_t idx = Simt::GetThreadIdx(); idx < N; idx += Simt::GetThreadNum()) { ... }
```

3. **SIMT launch** — `Simt::VF_CALL`:

```cpp
// In _apt.cpp (host kernel launches):
Simt::VF_CALL<ComputeSimt>(input_gm, output_gm, totalElements);
// Hardware launches up to LAUNCH_BOUND threads; each runs ComputeSimt with same args
```

### Two writing styles

```cpp
// Style A: free function (simplest)
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ComputeSimt(
    __gm__ T* input, __gm__ T* output, int32_t total) { ... }

// Style B: class static member (more flexible — encapsulate state)
template <typename T>
class MoeSrcToDst...   // (source text truncated here)
```

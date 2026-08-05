---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Hardware Sort Pipeline (Concat+Sort+Extract)"
description: "Problem: Software sorting (scalar merge sort, bubble sort, etc.) is extremely slow on NPU. Ascend hardware has a dedicated bitonic sort network that processes 32 elements per cycle. Pattern: Three-ste"
confidence: single_run
original_id: P-P42
timestamp_inferred: true
tags: [sort, optimization, sortedlocal, sorttmplocal, concattmplocal, merge_sort, radix_sort, p-p42, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: Software sorting (scalar merge sort, bubble sort, etc.) is extremely slow on NPU. Ascend hardware has a dedicated bitonic sort network that processes 32 elements per cycle.

**Pattern**: Three-step pipeline using the built-in AscendC Sort API:
```cpp
// Step 1: Pack into sort struct (value, index) pairs
uint32_t concatRep = alignedN / 16;
AscendC::Concat(concatLocal, xLocal, concatTmpLocal, concatRep);

// Step 2: Hardware sort (bitonic network, descending by default)
uint32_t sortRep = alignedN / 32;
AscendC::Sort<float, true>(sortedLocal, concatLocal, indexLocal, sortTmpLocal, sortRep);

// Step 3: Unpack results
AscendC::Extract(sortedValueLocal, sortedIndexLocal, sortedLocal, extractRep);
```

**Ascending-sort trick**: The hardware sorter defaults to descending. For ascending, flip the sign bit before sorting:
```cpp
// Ascending: flip sign bit → descending sort → flip sign bit again
Adds(xLocal_int32, xLocal_int32, (int32_t)0x80000000, N);  // flip sign bit
Sort<float, true>(...);  // hardware sort (descending)
Adds(result_int32, result_int32, (int32_t)0x80000000, N);   // flip back
```

**bf16 handling**: The hardware sorter only supports fp32 and fp16. bf16 must be Cast to fp32 for sorting, then Cast back.

**UB space requirements**:
- `sortedLocal`: 8 × alignedN bytes (sort struct)
- `sortTmpLocal`: 8 × alignedN bytes (temporary buffer)
- `concatTmpLocal`: GetConcatTmpSize() bytes
- **Total**: ~20-24 bytes/element → N=4096 fp32 ≈ 80-96KB

**Evidence**: CANN sort_merge_sort.h:198-243, sort_tiling_arch35.cpp:746-761. E1 level.

**A5 verification update (2026-04-14 / 2026-04-18)**: In practice on Ascend950PR we use the **Advanced Sort API**:
```cpp
// Recommended default: MERGE_SORT (EC-33 addendum — RADIX_SORT has sporadic VMS 343 on A5 CANN 9.0.0)
constexpr SortConfig sortCfg = {SortType::MERGE_SORT, true};  // must be global constexpr (EC-24)
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ > 0)                // must have __NPU_ARCH__ guard (EC-25)
Sort<float, false, sortCfg>(dstLocal, srcLocal, tmpLocal, calCount);
#endif
```
**SortType selection** (2026-04-18 update):
- **`MERGE_SORT` (default on A5)**: stable, no VMS 343 observed. Preferred.
- **`RADIX_SORT`**: sporadic runtime `aicore 343 "Incorrectly sorted data entered by the VMS"` observed on Ascend950PR CANN 9.0.0 (see EC-33 addendum). Use only when perf profiling proves a significant advantage over MERGE_SORT AND representative inputs are confirmed not to trigger VMS 343.

This API uses sort internally, supports arbitrary N, and does not require manual Concat+Sort+Extract.
- Requires `#ifdef __NPU_ARCH__` guard (EC-25)
- SortConfig must be global constexpr (EC-24)
- DataCopyPad UB→GM crashes; use DataCopy + host padding instead (EC-23)
- Sorting on a non-last dimension: pybind-layer permute+contiguous → kernel sorts only the last dim → permute result back

**Stop condition**: When N > 4096 (fp32) or N > 1024 (fp16), UB space is insufficient; switch to radix sort. int types are not supported by this pipeline (use radix sort).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P42，convert_patterns_to_okf.py）。confidence 未升格。 -->

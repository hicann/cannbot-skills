---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "DataCopy alignment overwrite in strided/chunked copies"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions"
confidence: single_run
original_id: EC-16
timestamp_inferred: true
tags: [ascendc, ec-16]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures on specific test cases where non-aligned chunk sizes cause data corruption in adjacent output regions
- **Root cause**: DataCopy requires 32-byte aligned element counts. When `chunk_size % ALIGN != 0`, naively aligning up writes extra elements past the chunk boundary, corrupting adjacent tensor data.
- **Fix (overlapping tail write)**:
  ```
  1. Copy floor_aligned(chunk) elements normally
  2. Copy last ALIGN elements starting at (chunk - ALIGN), overlapping with already-written region
  ```
  The overlap is harmless (same values re-written), and tail elements are placed correctly without overflow.
- **Condition**: Strided/chunked DMA with non-aligned chunk boundaries (e.g., cat along non-last dim)
- **Evidence**: Cat V1 failed 3/51 cases, fixed in V2 (2026-04-09)

## Quick Lookup Table

| EC | Error keyword | One-line fix |
|----|--------------|--------------|
| EC-1 | `calling a __host__ function from __aicore__` | Add `__aicore__ inline` to helper |
| EC-2 | `cannot initialize '__gm__ T*' with 'GM_ADDR'` | `reinterpret_cast<__gm__ T*>(gm_addr)` |
| EC-3 | `LAUNCH_BOUND exceeds maximum` | Reduce to 512 |
| EC-4 | `redefinition of 'blockDim'` | Guard with `#if defined(ASCENDC_CPU_DEBUG)` |
| EC-5 | `not support bf16 type cast` | Use `simt_to_float()` bit-manipulation (P-P27) |
| EC-6 | `call to 'GetBlockIdx' is ambiguous` | Remove `using namespace AscendC::Simt;` (OL-14) |
| EC-7 | `no member 'atomicAdd' in 'Simt'` | Use unqualified `atomicAdd()` (global built-in) |
| EC-8 | `unknown type name 'GM_ADDR'` | Add `#include <kernel_operator.h>` as first include |
| EC-9 | `redefinition of 'ITER'` / missing symbols | Wrap all code in `namespace ascendc_ops {}` |
| EC-10 | `undefined reference to 'aclrtlaunch_'` | Add `extern "C" {}` around declaration |
| EC-11 | `merge_mix_obj.sh Error 1` at 95% | Add `-DCMAKE_BUILD_TYPE=Release` |
| EC-12 | `cannot initialize 'int64_t (*)(void)'` + `expanded from macro 'block_num'` | Rename param: `blk_idx`/`blk_cnt` |
| EC-13 | `no member named 'SyncFunc' in namespace 'AscendC'` | Use `SetFlag`/`WaitFlag` with `FetchEventID` |
| EC-14 | `static assertion failed: must use AllocTensor...depth is zero` | Change TQue depth from 0 to ≥1 |
| EC-15 | `the range of 1st parameter must be [4, 6]` | No `PipeBarrier<PIPE_S>`, use SetFlag/WaitFlag for S pipe |
| EC-16 | Non-aligned chunk DataCopy corrupts adjacent data | Overlapping tail write: copy last ALIGN elems separately |
| EC-17 | Sub-align chunk overwrite in compact output | nblk=1 + padded alloc + narrow view |

<!-- 迁移自 porter kb/target/ascendc/（EC-16，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

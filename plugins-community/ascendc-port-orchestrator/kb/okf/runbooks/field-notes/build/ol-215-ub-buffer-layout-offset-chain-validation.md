---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "UB buffer layout must be validated before first compile — one-line offset bug cascades to all downstream buffers"
description: "In a 10+ UB-buffer offset-chain, one copy-paste offset error shifts all downstream buffers and silently corrupts data — validate the chain statically before the first compile."
phenomenon: build_failure
signal:
  - "A kernel with 10+ UB buffers hangs or produces corrupted output at runtime, with the compiler giving ZERO errors — a single offset expression in the offset-chain was mistyped"
confidence: single_run
original_id: OL-215
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, ol-215, ub-layout, offset-chain, static-check]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Ascend950PR / CANN 9.0.0 (LightningIndexerGrad P132+P134, 2026-06-05; same pattern unverified on Ascend910_V220). AscendC kernels with 10+ UB buffers use an offset-chain pattern where each buffer's offset references the previous buffer's offset + size:
```cpp
constexpr static int64_t buf1Offset = 0;
constexpr static int64_t buf1Size = 4096;
constexpr static int64_t buf2Offset = buf1Offset + buf1Size;  // = 4096
constexpr static int64_t buf2Size = 8192;
constexpr static int64_t buf3Offset = buf2Offset + buf2Size;  // = 12288
```
A **single copy-paste error** in one offset expression propagates to ALL subsequent buffers. **The compiler gives ZERO errors** — UB layout is entirely the kernel author's responsibility. Symptom is a runtime hang or silently corrupted data, not a build/compile diagnostic.

## 根因 / 教训

Two concrete instances in LightningIndexerGrad:
- **P134**: `reluInPingUbOffset = 0 + indicesUbSize` (= 8192) should have been `indicesUbOffset + indicesUbSize` (= 16384). The `0 +` base was a copy-paste from the first buffer's pattern. All 14 subsequent buffers shifted by 8192 bytes; the 8192-byte overlap between `indicesUb` and `reluInPingUb` silently corrupted data.
- **P132**: `MAX_UB_SIZE = TOTAL_SIZE/sizeof(float)/2` (= 24192 floats, 96768 bytes) when only 38912 bytes were actually available after the declared buffers. The 2.5× overflow corrupted NPU pipeline/event state BEYOND UB boundaries.

Both were found only through manual inspection after runtime hangs. **Lesson / fix**: run `src/scripts/orchestrator/pre_build_check.py <kernel.h>` static analysis BEFORE compilation (no NPU/CANN needed, exit 0 = clean). It (1) parses all `constexpr static int64_t <name>Offset/Size`, (2) verifies chain consistency `offset[N] == offset[N-1] + size[N-1]`, (3) verifies no overlaps between any two buffer intervals, (4) verifies total declared size ≤ TOTAL_SIZE, (5) verifies `MAX_UB_SIZE ≤` actual remaining workspace bytes, (6) reports exact line numbers + fix suggestions. It would have caught both bugs in < 1 second before the first compile.

**Cross-references**: OL-213 (SyncAll audit — complementary static check), `src/scripts/orchestrator/pre_build_check.py` (the validator), P-P98 (DataCopyPad — alignment counterpart).

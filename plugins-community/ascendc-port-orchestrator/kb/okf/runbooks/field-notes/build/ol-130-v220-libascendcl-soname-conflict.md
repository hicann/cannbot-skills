---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 A3 container has a libascendcl.so SONAME conflict between CANN 9.0.0 and ascend-toolkit 8.3"
description: "The npu-a3 container ships two CANN installs whose libascendcl.so share a SONAME; the linker may resolve the wrong one, causing silent kernel-launch failures. Build+run against the same CANN torch_npu uses (toolkit 8.3)."
phenomenon: build_failure
signal:
  - "double free or corruption (!prev) on process exit; runtime segfault (exit 139) on kernel launch; TQue<VECOUT> crash with error 507035 — all on the A3 npu-a3 container"
confidence: single_run
original_id: OL-130
classified_by: llm-assisted
timestamp_inferred: true
tags: [build, soname-conflict, ol-130, v220]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
On the A3 `npu-a3` container, kernels fail at launch in ways that look unrelated to the kernel algorithm (observed across the DS batch 2026-05-02):
- `double free or corruption (!prev)` on process exit (31_IOU)
- Runtime segfault (exit 139) on kernel launch (15_Pad)
- `TQue<VECOUT>` crash with error 507035 (25_NLLLoss)

## 根因 / 教训
The A3 container `npu-a3` has TWO CANN installations that both ship `libascendcl.so` with the **same SONAME**. CANN 9.0.0 has `aclrtLaunchKernelWithHostArgs`; toolkit 8.3 does not. When both are loaded, the linker may resolve to the wrong one → kernel launch fails silently (segfault, double-free, garbage output).

## 解决配方
Rebuild the kernel against the SAME CANN installation that torch_npu uses at runtime. For this container, use ascend-toolkit 8.3's headers + libs for BOTH the build AND `LD_LIBRARY_PATH` at runtime.

### Evidence
- op#31 IOU pp-1 (2026-05-02): a 3-iter bisection confirmed the kernel algorithm was correct; a library-isolation probe identified the SONAME conflict. op#31 kw-6: rebuilt against toolkit 8.3 → crash resolved, kernel runs clean.

### Cross-ref
- OL-127 (CANN API surface gaps)
- OL-128 (TBuf→MTE3 coherence)
- PB-24 (TBuf GetValue interleaving)

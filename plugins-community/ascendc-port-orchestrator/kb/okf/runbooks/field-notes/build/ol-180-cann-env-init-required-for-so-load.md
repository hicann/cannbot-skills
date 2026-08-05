---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC kernel .so loading requires CANN environment initialization"
description: "Before torch.ops.load_library(.so), CANN env must be initialized (source set_env.sh or set LD_LIBRARY_PATH/ASCEND_HOME_PATH), else .so load fails with undefined-symbol errors."
phenomenon: build_failure
signal:
  - "torch.ops.load_library(<kernel>.so) fails with undefined symbol errors when the CANN environment was not initialized in the process first"
confidence: single_run
original_id: OL-180
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, build, ol-180, cann-env, so-loading]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
`torch.ops.load_library(<kernel>.so)` fails with undefined-symbol errors because the CANN runtime environment was never initialized in the process before the load.

## 根因 / 教训
Loading an AscendC kernel `.so` requires the CANN environment to be live first: source `set_env.sh`, or configure `LD_LIBRARY_PATH` / `ASCEND_HOME_PATH` so the dynamic loader can resolve the CANN runtime symbols the kernel references. Without this, the `.so` load fails with undefined-symbol errors (the missing symbols are the CANN runtime, not the kernel's own code).

Verified on soc=Ascend910_9382 (V220), cann=8.3.RC1. Evidence: 25_NLLLoss a3-ds kw-5 (2026-05-21) — `.so` load failed until the CANN env was properly initialized.

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`deploy_to_a5.sh` syncs `current_task/` not `workspace/<op>/kernel/`"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Worker edits workspace/<op>/kernel/topktopp_kernel.h, runs bash src/scripts/deploy_to_a5.sh --build, build succeeds, precision result is unchanged from previous"
confidence: single_run
original_id: EC-30
timestamp_inferred: true
tags: [ascendc, ec-30]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Worker edits `workspace/<op>/kernel/topktopp_kernel.h`, runs `bash src/scripts/deploy_to_a5.sh --build`, build succeeds, precision result is **unchanged** from previous iteration despite the edit. Verification reports same FAIL signature as before the change.
- **Root cause**: `deploy_to_a5.sh` syncs `~/workspace/AscendOpGenAgent/current_task/` to A5 container for build. It does **not** read from `workspace/<op>/`. So edits to `workspace/<op>/kernel/*.h` are invisible to the build.
- **Fix**: Worker must manually copy before each build:
  ```bash
  cp $LOCAL_PROJECT/workspace/<op>/kernel/* \
     ${LOCAL_TASK:-$HOME/workspace/AscendOpGenAgent/current_task}/kernel/
  bash src/scripts/deploy_to_npu.sh --build
  ```
  Or fix the deploy script to accept `ASCENDC_WORKSPACE` env var and sync from that. Until fixed, worker/probe must always pre-copy.
- **Detection**: Verification run produces identical results across two consecutive iters despite kernel edits. Confirm by `md5sum workspace/<op>/kernel/topktopp_kernel.h` vs `md5sum <current_task on A5>/kernel/topktopp_kernel.h` — different md5 = phantom build.
- **Evidence**:
  - 9_TopKTopP V2 iter 1 (2026-04-17). Worker rewrote kernel, built, got identical 34/50 FAIL as v1 — phantom 30 minutes debugging before noticing A5 current_task/ had stale v1 kernel.
  - 9_TopKTopP cold-run probe (2026-04-18). Probe explicitly ran `cp workspace/topktopp_v31/kernel/* current_task/kernel/` before every `deploy_to_a5.sh --build` call — confirmed the workaround works and is load-bearing. Spawned inside aog-kernel-worker + aog-precision-probe specs as a standing requirement.

<!-- 迁移自 porter kb/target/ascendc/（EC-30，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

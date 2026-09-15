---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Stale `LOCAL_TASK` staging masquerades as runtime non-determinism — `diff workspace/{op}/kernel LOCAL_TASK/kernel` before declaring a stuck signature"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "Phase D shows intermittent precision regressions on a fixed input (e.g. \"47/50 modal, 46–49 across runs\", \"case 26 differs ~10 % of runs\"). Worker exhausts its"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-96
timestamp_inferred: true
tags: [local_task, ascendc, ol-96]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: workflow / staging-discipline
- **Loaded by**: aog-kernel-worker / aog-precision-probe / orchestrator when an op shows "intermittent" precision regressions on inputs that should be deterministic by-construction
- **Symptom**: Phase D shows intermittent precision regressions on a fixed input (e.g. "47/50 modal, 46–49 across runs", "case 26 differs ~10 % of runs"). Worker exhausts its precision-fix budget chasing a phantom AIV pipe-sync race; aog-precision-probe spawns to investigate.
- **Real cause**: between Phase C (build) and Phase D (verify), worker iter 3..5 Edited fixes into `workspace/{op}/kernel/*.h|.cpp` — but the build that downstream `verification_ascendc.py` runs against was started against a stale `LOCAL_TASK/kernel/` snapshot. The `cp workspace → LOCAL_TASK` step is manual per V3.4.2 deploy contract and easy to skip across multiple Edit/Build/Verify iters. The "non-determinism" is purely the build pipeline reading whichever `LOCAL_TASK` snapshot happened to be live when the .so was compiled — once you `cp + rebuild`, the symptom evaporates and worker's fixes (already correct on disk in `workspace/`) take effect.
- **Fix / mandatory check before invoking aog-precision-probe**:
  ```bash
  diff -r workspace/{op}/kernel/ "$LOCAL_TASK/kernel/"
  # if non-empty: cp -a workspace/{op}/kernel/* "$LOCAL_TASK/kernel/"
  #               + bash src/scripts/deploy_to_npu.sh --build
  #               + re-run Phase D
  ```
  Only spawn `aog-precision-probe` if `diff` is empty AND the symptom still reproduces on a fresh build. Otherwise you are paying probe-cost to debug a staging gap.
- **Detection signature in PROGRESS.md**: worker exit handoffs that include phrases like `intermittent`, `~N% of runs`, `non-deterministic intra-process`, `47/50 modal range 46–49` — and where determinism preflight (Phase O2.5 step 3b, OL-88) had passed for the reference. Pure-functional ops (no atomicAdd, no concurrent writes, single-AIV-per-row) **cannot** legitimately exhibit intra-process non-determinism — if you see it on such an op, suspect staging FIRST, kernel race SECOND.
- **Cross-ref**: V3.4.2 `src/scripts/deploy_to_npu.sh` mandate; PB-15 (single container serial discipline); P-P77 (TQue<VECOUT,2> auto-sync — the practitioner-side fix that the aog-kernel-worker correctly applied; staging-stale prevented its effect from being visible). Probe agents: when classifying a stuck signature, run the diff as the *first* step of investigation, not after exhausting all hypothesis testing.
- **Evidence**: op#27 MultiMaskAttentionAggregation (a3 V220, 2026-04-28) — kw-1 exit signature `non-deterministic_intra-process_aiv_pipe_sync` after 5 fix iters (47/50 modal, range 46–49). Probe pp-1 entered, ran `diff workspace/27_*/kernel/ LOCAL_TASK/kernel/` → 3 hunks of mismatch (gmMaskSum_ size, outQue_ TQue replacing finalTBuf_, V→S sync block). After `cp + deploy_to_npu.sh --build`: 50/50 × 3 sequential, 100/100 bit-exact (5 hot cases × 20 runs). Worker's diagnoses (TBuf→TQue conversion, V→S sync, mask_sum Align8 padding) were process-correct, BUT the build was running on a stale tree, so the symptom appeared as residual non-determinism rather than fix-confirmed PASS.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-96（category=workflow / staging-discipline，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

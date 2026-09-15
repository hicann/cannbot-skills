---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "identify the correct OPTIMIZATION VEHICLE before spending — some ops ship a dedicated custom test rig, NOT the standard `orch --optimize` O2.5 pipeline; running the wrong vehicle burns quota stuck at O2.5"
description: "Provenance: scan (cann_scan), selective_scan_fwd_simd 2026-07-24 — a large amount of token burned running orch --optimize on an op that has its own rig, before the vehicle mismatch was diagnosed. Revi"
phenomenon: precision_issue
signal:
  - "orch --optimize on such an op stalls at O2.5 demanding input_gen.py/manifest.json/ref_runnable.json/edge_dataset.pt — but the op does NOT use the standard O2.5"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-282
timestamp_inferred: true
tags: [selective_scan_fwd_simd, ascendc, ol-282]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

> Provenance: scan (cann_scan), `selective_scan_fwd_simd` 2026-07-24 — a large amount of token burned running `orch --optimize` on an op that has its own rig, before the vehicle mismatch was diagnosed. Reviewed + reframed by main (kb-manager): this is a methodology/docs lesson, NOT an unconditional a5_ops harness-code gap.

`applies_to: any op-perf A/B; especially consumer/custom-project ops (cann_scan etc.) that ship a dedicated rig; backend=ascendc`
`verified_on: selective_scan_fwd_simd — orch --optimize failed at O2.5 (demanded input_gen/manifest/ref_runnable/edge_dataset); the real vehicle = the ssf_ rig (ssf_run_case.py / ssf_sweep.py / precision_common.py + a valid edge_dataset.pt ground truth)`

**Symptom**: `orch --optimize` on such an op stalls at O2.5 demanding `input_gen.py`/`manifest.json`/`ref_runnable.json`/`edge_dataset.pt` — but the op does NOT use the standard O2.5 input-gen pipeline; it has its own rig. The caller can burn a large amount of token before realizing the vehicle is wrong.

**Principle**: `orch --optimize`'s O2.5 assumes the STANDARD input-gen pipeline. An op with a dedicated/custom rig (its own run/sweep/prof scripts + its own ground-truth dataset) needs THAT rig as its optimization vehicle, not orch's O2.5. Before spending on a perf A/B, identify the correct vehicle: standard orch O2.5, or a dedicated rig?

**Recommendation (scope-bounded — anti-over-engineering)**: for a5_ops ops that legitimately need a dedicated rig, record the correct vehicle per op (a `workspace/<op>` vehicle marker or an op→vehicle docs mapping). **The harness cannot auto-know a CONSUMER's custom-project rig** (e.g. cann_scan's `ssf_`), so this is primarily a methodology + docs lesson; a harness vehicle-identification CHECK is warranted ONLY if MULTIPLE a5_ops ops recurring-ly ship dedicated rigs — verify recurrence before building.

**Cross-ref**: OL-283 (rig-kernel-drift — the OTHER rig-A/B trap). backend=ascendc.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-282（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "before ANY rig-based A/B, align the rig's BUILT kernel to the current production kernel (md5-verify) — a stage rig's built kernel can lag production by generations, making the A/B baseline wrong / the optimization inapplicable"
description: "Provenance: scan (cann_scan), selective_scan_fwd_simd R2 2026-07-24 — nearly measured an A/B on a rig base 2 generations behind production; intercepted before measurement."
phenomenon: perf_regression
signal:
  - "you apply an optimization \"modify construct X\" to a stage rig's already-built kernel, but X does not exist in that kernel (it lags production) → nothing to modi"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-283
timestamp_inferred: true
tags: [selective_scan_fwd_simd, ascendc, ol-283]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

> Provenance: scan (cann_scan), `selective_scan_fwd_simd` R2 2026-07-24 — nearly measured an A/B on a rig base 2 generations behind production; intercepted before measurement.

`applies_to: any rig-based perf A/B (a5_ops ko/optimizer runs included); backend=ascendc`
`verified_on: selective_scan_fwd_simd — rig built kernel = pre-KO-2 ko1 (55d50c55, 0 Broadcast); production = 9199311c (KO-2, has the 3 UB Broadcasts). R2 = "replace those 3 Broadcasts with register Gather" — the 3-Broadcast target does not exist in the ko1 rig base → R2 inapplicable / A/B void`

**Symptom**: you apply an optimization "modify construct X" to a stage rig's already-built kernel, but X does not exist in that kernel (it lags production) → nothing to modify → the A/B is invalid / measures the wrong thing.

**Principle**: a stage rig's BUILT kernel is a snapshot that can be generations behind the current production kernel. An A/B against a stale rig base is a WRONG BASELINE — the optimization may target constructs the stale base doesn't have, or the "before" number isn't the real current-production number.

**Recommendation**: before any rig A/B, **align the production kernel into the rig + rebuild**, and gate the A/B on `rig_kernel_md5 == current_production_kernel_md5`. Bind the A/B baseline to the exact current production kernel md5, so a drifted base is caught before measurement rather than after. (Perf-A/B analog of the CLAUDE.md same-condition-A/B rule + OL-281's "provision the real thing, not a stale snapshot".)

**Cross-ref**: OL-282 (vehicle-identification — the OTHER rig-A/B trap); CLAUDE.md "性能对比必须是同条件 A/B". backend=ascendc.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-283（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "MIX_AIC: precision-correct ≠ perf-normal — software-pipeline cube/vec across staggered taskIds; a serialized cube↔vec ping-pong is correct but half-idle"
description: "On a MIX_AIC unit, a serialized cube↔vec schedule is precision-correct but half-idle; the perf-normal form software-pipelines cube/vec across taskId%K-staggered iterations."
confidence: single_run
original_id: OL-200
classified_by: llm-assisted
timestamp_inferred: true
tags: [scheduling, optimization, ol-200, mix-aic, cube-vec-pipeline, whitebox]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
A **MIX_AIC** unit couples one cube core (AIC) with N vector cores (AIV, typically 2). A kernel that runs the cube matmul to completion, THEN the vector stage, THEN the next cube — a **serialized cube↔vec ping-pong** — is numerically **correct** and passes precision verification, but leaves the cube idle during every vector phase and vice-versa, so it is **NOT perf-normal**. The performant reference instead **software-pipelines**: it stages cube and vector work across consecutive taskIds (a `taskId % K`-staggered schedule, e.g. Vec on `i%4` while Cube runs `(i-1)%4`) so both engines stay busy.

**Precision-pass is necessary but not sufficient for a MIX_AIC op** — black-box generation that only optimizes for a correct result naturally converges on the serialized form, which is the dominant perf gap (not a tiling/dtype detail). **Whitebox-direct**: read the perf-normal reference's `Process()` loop, identify the cross-taskId stagger, and emit a kernel that reproduces the overlap — do NOT re-run-and-hope the generator stumbles onto pipelining.

**Concrete anchor**: LIG (`lightning_indexer_grad`, backward) V220 reference `Process()` interleaves stages by taskId — schematically `Vec1(i%4); Cube1((i-1)%4); Vec2((i-2)%4); Cube2((i-3)%4)` so AIC and AIV phases of different iterations overlap. The black-box-generated kernel instead emitted `while (sched.HasNext()) { cube(); vec(); }` cross-core ping-pong: precision-correct (34/34 PASS) but each engine idle ~half the time.

**Whitebox check method**: diff the generated `Process()` schedule against the reference's staggered loop; if the generated loop advances cube and vec in lockstep within one iteration (no `i%K` offset between them), the pipeline is collapsed.

Verified on soc=Ascend910_V220; op=`lightning_indexer_grad` backward (2026-05-29, DS whitebox vs CANN V220 source, a5_ops merge `8d8c5538`); generated kernel `85df8121`: 34/34 precision PASS but serialized. Unverified on Ascend950PR (arch35 MIX dispatch + KFC sync differ — the staggering principle is expected to transfer but the concrete cross-taskId schedule must be re-derived from the A5 reference).

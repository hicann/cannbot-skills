---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Exhaust host-contract reconstruction before concluding a deeper device root"
description: "Derive every host-tiling field from the selected arch22 contract; target tiling is advisory, not copy-ready."
confidence: single_run
original_id: OL-204
classified_by: llm-assisted
timestamp_inferred: true
tags: [debugging-discipline, optimization, ol-204, port-a3, faithful-copy, host-tiling]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
Host tiling is part of the operator contract: workspace formula, block-dim, split params, per-head
distribution indices, and tiling-key must all be derived from the selected arch22 source and declared
semantics. Target tiling may reveal omitted categories and test hypotheses, but its body must not be
copied as the generated implementation. Before escalating to device disassembly, mechanically prove
host-field completeness, internal consistency, selected-source traceability, and source-truth coverage.

**Historical anchor (pre-RFC copy policy)**: FA-A5 omitted
`multiCoreParamsRegbase.bnStartIdx[48]` / `sparseStartIdx[48]`; the device then read uninitialized
work-distribution data and raised L0C-OOR. The durable lesson is host-contract completeness.

**Cross-ref**: OL-187 (port_a3 must emit all upstream-dispatched variants), OL-133 (arch35 `ASCENDC_TPL_ARGS_DECL` compile-time axis enumeration), `feedback_port_a3_must_apply_a5_knowledge_differentially`, aog-self-critic C5 (premature platform-blame — same family: blaming a deep/novel cause before exhausting the cheap source-grounded one), PB-41 (the workspace-formula instance of the same "copy the host contract faithfully" rule).

Verified on soc=Ascend950PR, cann=9.0.0 (FA-A5 `3_FusionAttention` independent prototype whole-port, 2026-06-02): workspace + SetSysWorkspaceForce copied → 31→35; multi-head `bnStartIdx`/`sparseStartIdx` un-copied = the next root (under MEASURED test, not disassembly). Other predicted instances: any whole-port of a tiled vendor op (FlashAttention, GroupedMatmul, fused norm+matmul, MoE) where the host tiling has work-distribution / per-instance offset arrays.

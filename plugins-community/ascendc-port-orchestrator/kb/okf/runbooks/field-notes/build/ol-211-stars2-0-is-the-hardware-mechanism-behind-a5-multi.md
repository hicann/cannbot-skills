---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "STARS2.0 is the hardware mechanism behind A5 multi-core core-split + cross-core sync — Group scheduling (≤8 Groups, by-Die L2-affinity), compute-partition (AIC/AIV/SDMA into ≤16 resource pools), and a 128K/4096 hardware sync-flag pool"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (multi-core / cross-core-sync / MIX cube-vec)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (multi-core / cross-core-sync / MIX cube-vec)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-211
timestamp_inferred: true
tags: [ascendc, ol-211]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (multi-core / cross-core-sync / MIX cube-vec)`
`verified_on: soc=Ascend950PR (Ascend950 NPU whitepaper §3, §4.4, 2026-06-07)`

**Principle**: A5 (Ascend950) ships a hardware task-and-resource scheduler **STARS2.0** (System Task and Resource Scheduler, 2nd-gen) that is the silicon behind the multi-core core-split and cross-core synchronization primitives the op-gen pipeline relies on. It is NOT a software runtime — it is a full-chip dispatch processor that schedules AIC/AIV/CPU/DVPP/SDMA/UB/CCU engines, chains data flow, and synchronizes between task streams. Knowing its capacity envelope tells you what multi-core layouts and sync schemes are physically available before you hand-roll one.

**Concrete capacity anchors (whitepaper §4.4)**:
- **Group scheduling — up to 8 Groups**: AI-Core resources partition into ≤8 Groups; per-Group AI-Core count + security attributes are software-configurable. The intended use is **by-Die grouping for L2-cache locality affinity** (group an op's cores onto one Die so its working set stays L2-local).
- **Compute partition (算力切分)** — AIC/AIV/SDMA split into **up to 16 resource pools**; other accelerators into ≤8 pools; pools bind to VMs for isolation.
- **Hardware sync flags** — **up to 128K single-bit OR up to 4096 32-bit multi-bit flags** for inter-task-stream synchronization. This is the HW pool that the `CrossCoreSetFlag`/`CrossCoreWaitFlag` flagId space (API-exposed 0–10 per Group) draws from — the per-Group API range is an allocation convention, not the silicon ceiling.
- **HSCB (High-Speed Control Bus)** — dedicated bus between STARS and AIC/AIV; reduces dispatch overhead to **ns-level**, supports broadcast scheduling, and is interference-free vs the normal NoC (which carries data traffic). This is why A5 launch overhead is materially lower than A3 (cross-ref the ~10-15 µs A5 vs ~60-100 µs A3 launch observation).
- **2048 task streams** host→device — STARS prefetches/schedules per stream config and reports completion back to host.

**Evidence**: Ascend950 NPU 架构白皮书 (2026-06-07) §3 (architecture overview lists STARS2.0 as a top-level spec) + §4.4 (软硬协同高效调度：STARS2.0 — Group/compute-partition/flag/HSCB/task-stream details). Grounds the operational facts already used in OL-210 (global cross-group flags), P-P102 (WorkspaceQueue cross-core sync), OL-190 (MIX_AIC_1_2 group barrier).

**Other instances (predicted)**: any A5 multi-core perf work (FA-A5 multi-core, MoE expert-parallel, fused-norm+matmul) that needs to reason about core-split granularity, L2-affinity grouping, or whether a cross-core sync scheme fits the hardware flag budget; framework-launch orchestration that wants to choose Group layout for L2 locality.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-211（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

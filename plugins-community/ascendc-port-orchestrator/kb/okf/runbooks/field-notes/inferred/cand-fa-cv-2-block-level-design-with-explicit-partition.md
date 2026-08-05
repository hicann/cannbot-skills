---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Declare partition, ring-buffer, workspace, and sync contracts before AscendC kernel implementation"
description: "For an L4 cube-vector fused op, record block partitioning, tile sizes, ring slots, workspace tensors, ownership, and cross-core sync in a structured design contract before assembling kernel bodies."
phenomenon: build_failure
signal:
  - "A fused AscendC kernel is emitted without a reviewable block/workspace/synchronization contract."
confidence: inferred
status: stub
original_id: CAND-FA-CV-2
timestamp_inferred: true
tags: [candidate, inferred, ascendc, prelaunch, ring_slots, workspace, cand-fa-cv-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 候选（未验证 —— 默认检索不返回，需 --status all 才可见）

`applies_to: workflow=L4_fused_op_design; backend=ascendc`
`derived-from: FA-class template-assembly and WorkspaceQueue evidence`

**Trigger**: a new L4 fused op needs algorithm design before AscendC coding. The worker
must decide block partitioning, tile sizes, ring-buffer depth, workspace tensors, and
cross-core synchronization before it starts emitting kernel bodies.

**Pattern**:

1. Record `block_num`, `block_M/N`, prelaunch/ring slots, workspace tensor
   names/shapes/dtypes, cube/vector ownership, and flag IDs in a structured
   decision manifest or Phase-A artifact.
2. Fill per-tile Load/Gemm/Softmax stages only after that contract is reviewable.
3. Mechanically verify that emitted tiling and workspace layout match the contract.

**Detection**: reject an L4 cube↔vec emission that lacks decisions for
`block_partition`, `cube_vec_split`, `cross_core_sync`, `ub_tiling`, and
`ring_slots`. Do not infer them from a monolithic header after generation.

**Evidence**: the FA reference design used block_M=64, block_N=64, prelaunch=2,
ring_slots=3, four workspace tensors, and explicit C/V ownership. Making those
decisions first exposed drift that a one-step model-to-code path had missed.

**Promote when**: a second L4 fused operator independently confirms the same contract
keys predict a correct multi-core AscendC assembly.

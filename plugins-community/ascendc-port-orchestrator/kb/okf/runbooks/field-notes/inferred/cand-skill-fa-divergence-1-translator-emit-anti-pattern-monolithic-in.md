---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Generator-emission anti-pattern: monolithic inline CrossCore flag spam vs WorkspaceQueue discipline"
description: "Reject a fused-attention AscendC emission when a large monolithic header scatters manual CrossCore flags through tile loops and has no queue ownership abstraction."
phenomenon: runtime_failure
signal:
  - "More than five inline CrossCore Set/Wait calls, no WorkspaceQueue abstraction, and a single generated header of at least 500 lines."
confidence: inferred
status: stub
original_id: CAND-SKILL-FA-DIVERGENCE-1
timestamp_inferred: true
tags: [candidate, inferred, ascendc, fused-attention, workspace_queue, cross-core-sync, cand-skill-fa-divergence-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 候选（未验证 —— 默认检索不返回，需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=fused-attention`
`verified_on: differential run of a monolithic generated kernel against a split-header WorkspaceQueue kernel`

**Anti-pattern**:

- one large header contains cube, vector, scheduler, and synchronization logic;
- inline `CrossCoreSetFlag<0x2, PIPE_FIX/MTE3>(id)` calls appear in tile loops;
- flag IDs are literals and workspace slots have no ownership abstraction.

**Preferred pattern**: split cube, vector, workspace queue, matmul tile, and shared
contracts; wrap CrossCore Set/Wait in `WorkspaceQueue`; instantiate dtype entry
points explicitly.

**Evidence**: the split-header WorkspaceQueue kernel ran at multiple shapes, while a
monolithic 757-line generated kernel failed at stream synchronization with AICore
error 507015. This does not prove WorkspaceQueue alone is the root-cause fix, but it
is sufficient to reject the unsafe emission shape.

**Detection gate**: block generation when all three conditions hold:

1. more than five inline `CrossCoreSetFlag|CrossCoreWaitFlag` calls;
2. no `WorkspaceQueue` or equivalent queue abstraction;
3. a single fused-attention header of at least 500 lines.

**Remediation**: require the standard FA template blocks and validate the same rule
on a second GQA/attention variant before promotion.

**Cross-ref**: CAND-FA-CV-1, CAND-FA-CV-4, PB-34, PB-35.

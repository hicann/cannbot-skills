---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Ring buffer workspace with WorkspaceQueue for multi-stage AIC↔AIV pipeline overlap"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=mixed_aic_aiv_fused_kernel_with_kv_iteration_and_prelaunch_overlap derived-from: cv-agent tile2asc flash_attention design (block_level + cub"
phenomenon: build_failure
signal:
  - "A mixed AIC/AIV fused kernel (e.g., FA, MoE gating, grouped matmul) needs ≥2 stages where cube and vector exchange GM-resident intermediates, AND stages should"
confidence: inferred
status: stub
original_id: CAND-FA-CV-1
timestamp_inferred: true
tags: [candidate, inferred, workspacequeue, crosscoresetflag, workspace_queue.h, cand-fa-cv-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=mixed_aic_aiv_fused_kernel_with_kv_iteration_and_prelaunch_overlap`
`derived-from: cv-agent tile2asc flash_attention design (block_level + cube.h + workspace_queue.h)`
`verified_on: cv-agent stock FA 16/16 PASS on A3/V220 (independent prototype F10.A.1 2026-05-25); DS env build+load+execute confirmed`
`unverified_on: V351/A5; a5_ops 61-case fixture; perf vs CANN baseline`

**Trigger**: A mixed AIC/AIV fused kernel (e.g., FA, MoE gating, grouped matmul) needs ≥2 stages where cube and vector exchange GM-resident intermediates, AND stages should overlap (prelaunch next stage before current stage fully completes).

**Pattern**: cv-agent FA uses `WorkspaceQueue<T, DEPTH>(gm_tensor, elem_size, sig_ready, sig_free)` for ring-buffer GM scratchpad management. Producer (cube/vec) writes into slot via `queue.AllocSlot(pipe)` → DataCopy(slot, src) → `queue.ReleaseSlot()`. Consumer acquires via `queue.WaitSlot()` → DataCopy(dst, slot) → `queue.FreeSlot()`. Ring buffer depth = prelaunch + 1 (e.g., prelaunch=2 → 3 slots). Cross-core flag IDs use vendor recipe: slot-indexed flags (0x8+slot for C→V, 0x10+slot for V→C).

**Our gap**: a5_ops FA (PR #146) uses raw `CrossCoreSetFlag<0x2, PIPE_FIX>(0x8+slot)` + manual GM offset arithmetic — no `WorkspaceQueue` abstraction, no ring-buffer lifecycle management, no slot ownership tracking.

**Detection**: grep for `CrossCoreSetFlag` + `DataCopy.*workspace` in kernel .h files. If flag IDs are hand-computed AND workspace_s/p/o tensors are raw-addressable (no queue abstraction) → WorkspaceQueue pattern missing.

**Evidence**: cv-agent `flash_attention_cube.h:32-34` + `workspace_queue.h` full implementation (Init/InitFreeSlotsMte2/AllocSlot/WaitSlot/ReleaseSlot/FreeSlot). 16/16 PASS stock fixture on V220.

**Cross-ref**: CAND-FA1 (manual CrossCoreSetFlag — this is the higher-level abstraction for multi-stage pipelines)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Separate Cube/Vector class architecture with MTE2_MTE1 sync boundary"
description: "applies_to: soc=Ascend910_9382 (V220); op_class=mixed_aic_aiv_fused_kernel derived-from: cv-agent flash_attention_cube.h + flash_attention_vec.h + flash_attention_kernel.h Trigger: Mixed AIC+AIV kerne"
phenomenon: build_failure
signal:
  - "Mixed AIC+AIV kernel where cube and vector stages have independent state (tiling config, GM tensor handles, UB buffers, pipeline queues). Monolithic single-clas"
confidence: inferred
status: stub
original_id: CAND-FA-CV-4
timestamp_inferred: true
tags: [candidate, inferred, flashattentionkernel, fakernel, __aicore__, flash_attention_vec.h, flash_attention_kernel.h, cand-fa-cv-4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220); op_class=mixed_aic_aiv_fused_kernel`
`derived-from: cv-agent flash_attention_cube.h + flash_attention_vec.h + flash_attention_kernel.h`

**Trigger**: Mixed AIC+AIV kernel where cube and vector stages have independent state (tiling config, GM tensor handles, UB buffers, pipeline queues). Monolithic single-class architecture forces shared state that complicates both sides.

**Pattern**: cv-agent FA splits into 3 classes:
1. `FlashAttentionCube<QType>` — cube-only: L1 buffers (qBufL1_, kvBufL1_, pBufL1_), L0A/B/C queues, LoadQ/LoadKV/Mmad1/Mmad2 methods. Init receives tiling struct + all GM tensor handles.
2. `FlashAttentionVec<QType>` — vector-only: UB buffers for softmax state (m_i, sumexp, acc_o), MTE3 output queue.
3. `FlashAttentionKernel` — orchestrator: Init allocates pipe + instantiates both Cube and Vec objects, Process() runs kv_loops with prelaunch pipeline (LoadKV while Cube processes prev iteration).

Sync boundary: `SetFlag<HardEvent::MTE2_MTE1>()` on cube side (cube→vec L1→VEC data ready), `WaitFlag<HardEvent::V_MTE2>()` on vec side (vec→cube UB→L1 data for next iter).

**Our gap**: a5_ops FA (PR #146) uses monolithic single-class with all buffers + methods in one `FaKernel` class. No cube/vec separation → harder to reason about AIC vs AIV lifetimes.

**Detection**: grep for `__aicore__` in kernel .h. If a single class contains BOTH `queL0A_/queL0B_` (cube L0 queues) AND `m_i/sumexp/acc_o` (vec UB state) → monolithic architecture, no cube/vec separation.

**Evidence**: cv-agent files — `flash_attention_cube.h:13` (FlashAttentionCube class), `flash_attention_vec.h` (FlashAttentionVec class), `flash_attention_kernel.h` (orchestrator).

**Cross-ref**: CAND-FA-CV-1 (ring buffer with WorkspaceQueue — the queue abstraction pairs with cube/vec separation), CAND-V351SYNC-1 (V351 mode=4 sync protocol)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

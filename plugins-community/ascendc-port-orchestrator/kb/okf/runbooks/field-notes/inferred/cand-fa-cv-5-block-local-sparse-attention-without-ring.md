---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Block-local sparse attention without ring buffer — composite stages sufficient when no KV iteration"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=block_sparse_attention_without_cross_block_interaction derived-from: cv-agent tile2asc block_sparse_attention design (block_level + kernel)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=block_sparse_attention_without_cross_block_interaction"
confidence: inferred
status: stub
original_id: CAND-FA-CV-5
timestamp_inferred: true
tags: [candidate, inferred, t.mma, cand-fa-cv-5]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; op_class=block_sparse_attention_without_cross_block_interaction`
`derived-from: cv-agent tile2asc block_sparse_attention design (block_level + kernel)`
`verified_on: cv-agent stock block_sparse_attention design documents pattern; not independently runtime-verified by DS`
`unverified_on: V351/A5; DS A3 runtime verification pending`

**HARD GUARD (2026-05-27, per F10 root-cause: designer selected CV-5 for standard SDPA FA → K-loop collapse + handoff broken)**:
- **DO NOT use CAND-FA-CV-5 for standard SDPA / FlashAttention with multi-KV-block (Skv > block_N).**
- CAND-FA-CV-5 applies **ONLY** when `exactly 1 KV block per Q block` (Skv ≤ block_size, block-sparse with no cross-block interaction).
- For standard FA (3_FusionAttention, Skv up to 512 ≫ block_N = 64/128 → multi KV block → KV loop required): use **CAND-FA-CV-1** (ring buffer + WorkspaceQueue + KV loop).
- **Mis-application consequence**: single `T.mma` without K-loop (KV blocks beyond first ignored), no workspace_meta ring (softmax state handoff broken) → A3 zero-output / A5 deadlock.
- **Detection**: grep design `*.py` for `CAND-FA-CV-5` reference AND `Skv > block_N` → BLOCK with `fa_pattern_mismatch: CV-5 requires Skv ≤ block_N, use CV-1`.

**Pattern**: When attention is block-local (each sequence divided into fixed-size blocks, no cross-block interaction), the full prelaunch/ring-buffer/WorkspaceQueue pipeline (CAND-FA-CV-1) is unnecessary. A simpler 4-stage pipeline (C1→V1→C2→V2) with 3 workspace tensors (s/p/o, no meta) and NO KV loop suffices:
- C1: Q_block @ K_block^T → workspace_s (fp32 scores)
- V1: scale + softmax → workspace_p (fp16 weights)
- C2: P @ V_block → workspace_o (fp32 partial output)
- V2: cast + write output

Block dimension: `block_num = batch * n_heads * n_blocks` where `n_blocks = seq_len / block_size`. No prelaunch needed because there's exactly 1 KV block per Q block.

**Our gap**: a5_ops FA (PR #146) assumed ALL attention variants need the full ring-buffer pipeline. For block-sparse or local-attention variants, the simpler composite-stage pattern reduces UB pressure and eliminates ring-buffer sync complexity.

**Detection**: grep for `kv_loops\|prelaunch\|ring_slots` in kernel .h files. If block-sparse op has these (unnecessary) → over-engineered for block-local pattern.

**Evidence**: cv-agent `block_sparse_attention/design/block_level/block_sparse_attention.py` — explicit comment "we don't need the prelaunch / ring-buffer pattern." 3 workspace tensors (vs 4 in base FA), no kv_loops variable.

**Cross-ref**: CAND-FA-CV-1 (ring buffer with WorkspaceQueue — the full pipeline), CAND-FA-CV-2 (block-level design — this is a counter-example showing design adapts to attention variant)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-5，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

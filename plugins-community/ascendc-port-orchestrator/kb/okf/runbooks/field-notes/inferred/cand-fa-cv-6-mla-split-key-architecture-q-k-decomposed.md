---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "MLA split-key architecture — Q/K decomposed into nope (position-independent) + rope (position-dependent) for KV cache compression"
description: "applies_to: soc=all; cann=all; op_class=multi_head_latent_attention_with_compressed_kv_cache derived-from: cv-agent tile2asc flash_attention_mla model.py + kernel verified_on: cv-agent stock MLA desig"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; op_class=multi_head_latent_attention_with_compressed_kv_cache"
confidence: inferred
status: stub
original_id: CAND-FA-CV-6
timestamp_inferred: true
tags: [candidate, inferred, nope, rope, q_nope, q_rope, k_nope, cand-fa-cv-6]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; op_class=multi_head_latent_attention_with_compressed_kv_cache`
`derived-from: cv-agent tile2asc flash_attention_mla model.py + kernel`
`verified_on: cv-agent stock MLA design documents pattern; cv-agent kernel artifacts present (cube.h + vec.h + kernel.h triplet confirms CAND-FA-CV-4 applies)`
`unverified_on: DS A3 runtime verification pending; a5_ops harness integration pending`

**Pattern**: MLA (Multi-head Latent Attention, per DeepSeek-V2/V3) splits Q and K into two components:
- `nope` (no-position): position-independent semantic content — can be absorbed into compressed KV cache
- `rope` (RoPE): position-dependent rotary embedding — computed on-the-fly, NOT cached

Input: 4 separate tensors (`q_nope`, `q_rope`, `k_nope`, `k_rope`) instead of standard single Q/K. GQA repetition via `_repeat_kv` handles kv_heads ≠ n_heads. Kernel retains cube.h/vec.h/kernel.h triplet (CAND-FA-CV-4 confirmed — MLA is same architectural family as standard FA).

**Our gap**: a5_ops has NO MLA implementation. Standard FA (3_FusionAttention) uses single Q/K inputs. CV-agent's MLA kernel exists with full design→translation chain complete — immediate adoption candidate for F10.E.2 (second op after FA).

**Key difference from standard FA**: Not just a different attention variant — it's a **different input contract** (4 tensors vs 2). This affects case_gen schema, model.py interface, and device block-level partition design.

**Detection**: grep for `nope\|_rope\|q_rope\|k_nope` in model.py or design files. Absence = standard FA, not MLA.

**Evidence**: cv-agent `flash_attention_mla/model.py:20-28` (forward signature with 4 inputs + _repeat_kv). Kernel has cube.h/vec.h/kernel.h triplet (same architectural pattern as CAND-FA-CV-4).

**Cross-ref**: CAND-FA-CV-4 (cube/vec class separation — MLA confirms this pattern applies beyond standard FA), CAND-FA-CV-2 (block-level design — MLA has design files in trace.md confirming 6-stage workflow)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-6，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "GQA index-decode (B, N2, G, S1) via device-side mod/div + KV-offset uses n2_idx ONLY — avoid Python `repeat_interleave` materialization"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_GQA_or_MQA / FlashAttention_variant / streaming_attention_with_grouped_kv derived-from: cann-source (FA reference V2"
phenomenon: build_failure
signal:
  - "Multi-head attention with grouped-query heads (Q has N1 heads, K/V has N2 heads, N1 = G N2 with G ≥ 2). Logical iteration space is (B, N2, G, S1_outer). Naive"
confidence: inferred
status: stub
original_id: CAND-FA-VEC-D-TILE-1
timestamp_inferred: true
tags: [candidate, inferred, repeat_interleave, idx, int64_t, cand-fa-vec-d-tile-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_GQA_or_MQA / FlashAttention_variant / streaming_attention_with_grouped_kv`
`derived-from: cann-source (FA reference V220 split-S1 + V351 train top-level Process, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-GQA-DIM`
`verified_on: cann ops-transformer FA reference V220 + V351 (kernel-structural evidence; no a5_ops kernel measurement on this exact pattern yet)`

**Trigger**: Multi-head attention with grouped-query heads (Q has N1 heads, K/V has N2 heads, N1 = G * N2 with G ≥ 2). Logical iteration space is (B, N2, G, S1_outer). Naive port temptation: pre-materialize K/V as (B, N1, S, D) on the Python side via `repeat_interleave(K, G, dim=1)` so the kernel sees a 1:1 N1 mapping. That doubles+ HBM K/V traffic.

**Why "candidate"**: Pattern is structural (algorithm shape, not API surface), derived from CANN FA reference's index-decode. Promotion to canonical requires a 2nd op-class verification — e.g. a separate GQA-style sparse attention kernel showing measurable HBM-traffic reduction vs the `repeat_interleave` port. The 507035 hang seen in independent prototype DEBT-FA-GQA (D=512 row-tiled fp16) is currently UNRESOLVED, so the symptom-anchor below is a hypothesis-link, not validated.

**Recommendation**:
1. Treat the 4-axis (B, N2, G, S1_outer) as a single flat counter `idx` over `[0, B * N2 * G * S1Outer)`.
2. Distribute `idx` across cores via `multi_core_offset = block_idx * splitFactor` style core-split.
3. Inside each core, decode the 4 axis indices via repeated mod/div:
   ```cpp
   int64_t b_idx     = idx / (n2_size * g_size * s1_outer_size);
   int64_t n2_idx    = (idx / (g_size * s1_outer_size)) % n2_size;
   int64_t g_idx     = (idx / s1_outer_size) % g_size;
   int64_t s1o_idx   = idx % s1_outer_size;
   ```
4. **Q offset uses both n2_idx and g_idx** (Q has N1 = N2*G heads): e.g. for BNSD,
   `q_off = b_idx * (N1 * S1 * D) + n2_idx * (g_size * S1 * D) + g_idx * (S1 * D) + s1o_idx * (s1_base * D)`.
5. **K/V offset uses ONLY n2_idx** (KV has N2 heads, shared across G query-groups within same n2): e.g. for BNSD,
   `kv_off = b_idx * (N2 * S2 * D) + n2_idx * (S2 * D) + s2_offset`.

The kernel reads K[n2_idx] once per (B, N2, S1_outer) tuple and uses it for all G query groups in that tuple — no K/V duplication needed.

**Concrete anchor** (public-API only — `GlobalTensor::operator[]` + plain `int64_t` arithmetic):
```cpp
// Inside kernel; tilingData provides b_size, n2_size, g_size, s1_outer_size, s1_base, d_size, layout strides.
int64_t flat_idx = block_idx * split_factor + inner_idx;  // inner_idx loops over the per-core range
int64_t b_idx   = flat_idx / (n2_size * g_size * s1_outer_size);
int64_t n2_idx  = (flat_idx / (g_size * s1_outer_size)) % n2_size;
int64_t g_idx   = (flat_idx / s1_outer_size) % g_size;
int64_t s1o_idx = flat_idx % s1_outer_size;

int64_t q_off  = b_idx * n1_s1_d + n2_idx * g_s1_d + g_idx * s1_d + s1o_idx * s1_base_d;
int64_t kv_off = b_idx * n2_s2_d + n2_idx * s2_d;  // no g_idx — KV shared across G

LocalTensor<half> q_tile = q_que.AllocTensor<half>();
DataCopy(q_tile, q_gm[q_off], s1_base * d_size);
// K, V re-used for all G iterations within this (b_idx, n2_idx, s1o_idx)
```

**Reject_cond**: skip this pattern when **G = 1** (MHA — no grouping benefit). Also skip when the op's GQA shape is **already materialized upstream** (e.g. a sparse-attention variant where K/V is logically shaped (B, N1, S, D) for indexing reasons and de-replication is non-trivial). And skip when **D is so small** (D ≤ 32) that the HBM K/V traffic is already negligible relative to compute.

**Symptom anchor**: independent prototype `fa_v220` row-tiled fp16 D=512 fp16 GQA case hangs at `LaunchAscendKernel 507035` (DEBT-FA-GQA). HYPOTHESIS-LINK: the current row-tiled-VEC-only kernel pre-materializes the G replicas of K/V via Python `repeat_interleave(K, G, dim=1)`, which then makes the kernel's UB budget tight when D=512 and triggers the 507035 silent-launch failure. Validating this candidate's recommendation (device-side n2_idx-only KV offset) on the independent prototype kernel and measuring whether the 507035 hang resolves is one of two next-step actions for DEBT-FA-GQA.

**Other-instances-predicted**: any GQA-aware attention port (FlashAttention-GQA, sparse FlashAttention, multi-query attention variants), incremental KV-cache decoding kernels, and any L4-fused op whose "logical N1" decomposes to N2*G with a shared K/V head per group.

**Promote when**: measured on independent prototype `fa_v220` D=512 GQA AND one additional GQA-shape op (target: a separate sparse-attention or KV-cache decode kernel), with HBM-K/V traffic reduction ≥ G×-1 confirmed via msprof (each n2_idx-only KV offset eliminates G-1 duplicate K/V reads).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-VEC-D-TILE-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

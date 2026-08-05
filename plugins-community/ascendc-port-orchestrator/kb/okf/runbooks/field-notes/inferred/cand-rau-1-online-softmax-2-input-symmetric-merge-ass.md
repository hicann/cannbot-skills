---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Online-softmax 2-input symmetric merge — associative/commutative reducer over (max, sum, accum) triples for ring/tree fold"
description: "applies_to: any SoC with public AscendC VEC Max/Sub/Exp/Mul/Add/Div; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul / ring_attention derived-from: cann-source (r"
phenomenon: build_failure
signal:
  - "An attention-class kernel produces partial (attn_out, softmax_max, softmax_sum) state from N independent stages (ring-attention shards, KV-cache chunks, sliding"
confidence: inferred
status: stub
original_id: CAND-RAU-1
timestamp_inferred: true
tags: [candidate, inferred, exp, out_i, sum_i, cand-rau-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with public AscendC VEC Max/Sub/Exp/Mul/Add/Div; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul / ring_attention`
`derived-from: cann-source (ring-attn-class update, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update SBH + TND variants (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: An attention-class kernel produces partial (attn_out, softmax_max, softmax_sum) state from N independent stages (ring-attention shards, KV-cache chunks, sliding-window blocks) and needs to combine those partials into one final triple. The combination must be order-independent (the ring/tree can fold in any order) and must preserve numerical stability under the max-shift form.

**Relationship to CAND-FA2**: CAND-FA2 expresses the same online-softmax algebra as a *sequential recurrence* (one tile after another, carrying running state forward). This pattern expresses it as a *binary reducer*: given two equal-rank triples `(m_a, s_a, o_a)` and `(m_b, s_b, o_b)`, produce a merged triple `(m, s, o)`. Both forms agree exactly; the merge form is what makes ring-attention (and any tree/ring fold of partial attention outputs) work correctly regardless of communication schedule.

**Identity (the algebra is associative+commutative)**:
```
m       = max(m_a, m_b)
s_a'    = s_a * exp(m_a - m)
s_b'    = s_b * exp(m_b - m)
s       = s_a' + s_b'
scale_a = s_a' / s         # per-row, applies to all head_dim columns of o_a
scale_b = s_b' / s
o       = scale_a * o_a + scale_b * o_b
```
Associativity proof sketch: combining (a,b) then with c yields the same (m,s,o) as combining b,c then with a — because all three exponents re-center on the final max and the sums re-scale accordingly. Commutativity is by inspection (symmetric in a,b).

**Why this matters for ring-attention**: Each ring step ships a (m,s,o) triple between cards/cores. On receipt, the consumer merges with its current accumulator using this kernel. Without the algebraic property, the result would depend on ring traversal order, breaking determinism across ring topologies and across re-runs.

**Shape**: per-row state (`m`, `s`) has shape `[bn, seq, softmax_tail]` where softmax_tail is the per-row reduction width (usually 8 to align to one fp32 block). `o` has shape `[bn, seq, head_dim]`. The merge runs row-by-row (per `(bn, seq)` index), broadcasting the per-row scale across head_dim.

**Concrete anchor** (public-API VEC primitives, worker-local LocalTensor names):
```cpp
// Inputs in UB: maxA, maxB, sumA, sumB (each R*softmaxTail fp32); outA, outB (each R*headDim T).
// Scratch: scaleA, scaleB (R*softmaxTail fp32).
constexpr uint64_t mask[2] = {UINT64_MAX, 0};
AscendC::BinaryRepeatParams rpSoft = {1, 1, 1, 8, 8, 8};
uint8_t rt = (R * softmaxTail + 64 - 1) / 64;

// Step 1: merged max
AscendC::Max(maxOut, maxA, maxB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();

// Step 2: per-input exp(m_i - m)
AscendC::Sub(scaleA, maxA, maxOut, mask, rt, rpSoft);
AscendC::Sub(scaleB, maxB, maxOut, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Exp(scaleA, scaleA, mask, rt, {1, 1, 8, 8});
AscendC::Exp(scaleB, scaleB, mask, rt, {1, 1, 8, 8});
AscendC::PipeBarrier<PIPE_V>();

// Step 3: scaled sums and merged sum
AscendC::Mul(scaleA, sumA, scaleA, mask, rt, rpSoft);
AscendC::Mul(scaleB, sumB, scaleB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Add(sumOut, scaleA, scaleB, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();

// Step 4: per-row out-scale (scaleA, scaleB are reused as scale_a, scale_b)
AscendC::Div(scaleA, scaleA, sumOut, mask, rt, rpSoft);
AscendC::Div(scaleB, scaleB, sumOut, mask, rt, rpSoft);
AscendC::PipeBarrier<PIPE_V>();
// Step 5: broadcast scaleA/scaleB across head_dim and combine — see CAND-RAU-3 for stride shape
```

**Numerics**:
- Stable: every `Exp` argument lies in `(-∞, 0]`.
- An "empty" input (all masked, m_i ≈ -3e38, s_i = 0): contributes `exp(-3e38 - m) ≈ 0` and `0 * 0 = 0`, harmless.
- Both inputs empty: `s = 0`, the Div introduces 0/0 = NaN; worker must short-circuit empty-pair merges per op spec.

**Determinism**: Deterministic when (a) each row is single-AIV-owned, (b) the input GM tensors are themselves deterministic outputs of upstream stages, and (c) operands are read in a fixed (prev,cur) order. The algebra is order-invariant only at the math level (exact in real arithmetic); fp32 rounding errors do depend on order, so ring fold order should be fixed across runs for bit-exact reproducibility. Document this caveat — it is the same caveat as any fp tree-reduction.

**Hard do-not-apply**:
- Do NOT apply when the upstream stages produced their per-stage probabilities (already divided by per-stage sum); this merge expects *unnormalized* per-stage state — i.e. `out_i` must be `sum_j(exp(s_ij - m_i) * V_j)` NOT divided by `sum_i`. Confirm upstream contract before adopting.
- Do NOT collapse the two Div's into one by computing `scale = s_scaled / s` outside the loop and broadcasting; the kernel issues two Div's because both ratios are needed separately for the two outputs.

**Other instances predicted**:
- Ring-attention (the canonical case): each card holds one shard's (m,s,o), ring rotates them, each step merges.
- FlashAttention-v3 cross-block merge when blocks are processed concurrently by different AIVs and merged at the end.
- Distributed softmax/cross-entropy: each shard computes local (m,s) plus logsumexp, merged tree-wise.
- MoE expert output combining when each expert produces softmax-weighted state and the gate-weighted sum is computed via this merge form.
- Any "split-K" reduction over an inner dim where the reduction is `sum of exp(...)`.

**Risks before promotion**:
- Numerical: re-runs must use the same ring order to get bit-exact match. Build harness must record and fix the order; otherwise A/B perf and det-check both drift.
- Boundary: all-empty pair (both `m = -inf`) produces NaN; worker MUST gate.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that ships this merge (ring-attention update, KV-cache merge, or partial-FlashAttention finalize) and passes Pass A + Pass B + det + perf on 3_FusionAttention or a similar op.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-RAU-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

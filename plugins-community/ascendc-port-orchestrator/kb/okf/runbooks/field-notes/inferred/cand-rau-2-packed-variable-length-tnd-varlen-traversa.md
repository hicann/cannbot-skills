---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Packed-variable-length (TND/varlen) traversal via cumulative-offset pointer table with per-AIV batch-boundary advance"
description: "applies_to: any SoC with int64 GM scalar reads via .GetValue(); cann=9.0.0+; op_class=variable_length_sequence / packed_TND / ring_attention_varlen / flash_attn_varlen derived-from: cann-source (ring-"
phenomenon: build_failure
signal:
  - "An attention-class or sequence-class kernel consumes a TND/packed-varlen layout where N variable-length sequences are concatenated end-to-end along the T (token"
confidence: inferred
status: stub
original_id: CAND-RAU-2
timestamp_inferred: true
tags: [candidate, inferred, dimtindexcore, curbatchindex, dimtcore, actualseqqlen, cand-rau-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with int64 GM scalar reads via .GetValue(); cann=9.0.0+; op_class=variable_length_sequence / packed_TND / ring_attention_varlen / flash_attn_varlen`
`derived-from: cann-source (ring-attn-class update TND variant, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update_tnd.h (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: An attention-class or sequence-class kernel consumes a TND/packed-varlen layout where N variable-length sequences are concatenated end-to-end along the T (token) axis. Per-batch offsets are provided as a cumulative-offset table of length `B+1` (CSR-style): batch `i` spans tokens `[actualSeqQlen[i], actualSeqQlen[i+1])`. Each AIV is assigned a contiguous slice of the global T axis (`dimTIndexCore .. dimTIndexCore + dimTCore`) and must, per token, know which batch that token belongs to in order to compute correct softmax-tail / head-dim GM offsets.

**Why not just precompute a per-token batch map**: would require a separate scratch tensor of length T; the cumulative-offset form is already what host-side tiling provides and avoids the extra memory. The cost is a binary-search-or-linear-scan per AIV at startup, then constant per-step bookkeeping.

**Pattern**:
1. **Startup search**: At AIV start, linearly scan `actualSeqQlen[0..B]` to find the batch containing the first token `dimTIndexCore`. Record `curBatchIndex`, `seqNumBatchStartIndex = actualSeqQlen[curBatchIndex]`, `seqNumBatch = actualSeqQlen[curBatchIndex+1] - seqNumBatchStartIndex`, and `seqNumBatchTail = dimTIndexCore - seqNumBatchStartIndex` (position within batch). Linear scan is fine because `B` is small (≤ 256 typical) and amortized over `dimTCore` per-token work.
2. **Per-token main loop**: For each of `dimTCore` tokens this AIV owns, work on the current batch.
3. **Boundary check (while-loop)**: Before each token, if `seqNumBatchTail == seqNumBatch`, the previous step exhausted the batch. Advance: `curBatchIndex += 1`, refresh start/end/seqNumBatch from `actualSeqQlen`, reset `seqNumBatchTail = 0`. Use `while` (not `if`) to correctly skip zero-length batches.
4. **Per-token offset computation**: Use `(curBatchIndex, seqNumBatchTail)` to index softmax tensors at `[curBatchIndex, seqNumBatchTail, head, ...]` and attn tensors at the corresponding flat T offset. The softmax tensor in this varlen layout is shape `[sum(seq_i_padded_to_block), head_num, softmax_tail]`, so per-batch stride must use the *actual* `seqNumBatch` for that batch (not a constant).
5. **Increment**: `seqNumBatchTail += 1` at end of each token's iteration.

**Concrete anchor** (public-API ; worker-local names):
```cpp
// actualSeqQlenGm is GlobalTensor<int64_t>, length B+1, CSR-style.
int64_t curBatch = 0, batchStart = 0, batchEnd = 0, batchLen = 0, tailInBatch = 0;
// Startup: find batch containing dimTIndexCore
for (int64_t b = 0; b < batchSize; b++) {
  batchEnd = actualSeqQlenGm.GetValue(b + 1);
  if (dimTIndexCore < batchEnd) {
    curBatch = b;
    batchStart = actualSeqQlenGm.GetValue(b);
    batchLen = batchEnd - batchStart;
    tailInBatch = dimTIndexCore - batchStart;
    break;
  }
}

// Per-token loop
for (int64_t t = 0; t < dimTCore; t++) {
  while (tailInBatch == batchLen) {           // skip exhausted/empty batches
    curBatch += 1;
    batchStart = actualSeqQlenGm.GetValue(curBatch);
    batchEnd   = actualSeqQlenGm.GetValue(curBatch + 1);
    batchLen   = batchEnd - batchStart;
    tailInBatch = 0;
  }
  // softmax GM offset: per-batch stride is batchLen * head_num * softmax_tail
  int64_t softmaxOffset = batchStart * head_num * softmax_tail + tailInBatch * softmax_tail;
  int64_t attnOffset    = (dimTIndexCore + t) * head_num * head_dim;
  // ... per-head work, advance offsets by head_num_loop_each * head_dim/softmax_tail ...
  tailInBatch += 1;
}
```

**Numerics / correctness**:
- `while` (not `if`) is mandatory: zero-length batches (legal in some packed layouts) would otherwise corrupt indexing.
- Cast `actualSeqQlen` reads as int64; per-batch lengths can exceed INT32_MAX on long-context ops. Match the project's int64 type rule.
- Startup linear scan is O(B) per AIV; for large B (≥ 1024), switch to binary search. For typical B ≤ 256, linear is fine.

**Determinism**: Fully deterministic — the layout and traversal are data-driven by the input table.

**Hard do-not-apply**:
- Do NOT use this pattern when the layout is BSH (uniform seqlen) — overhead has no benefit; pre-computed strides are faster.
- Do NOT skip the `while` (use `if`): silent index corruption on zero-length batches.
- Do NOT precompute every per-token (batch, tail) into a scratch tensor for "speed" — the GM round-trip to read the scratch exceeds the linear-scan + while-advance cost.

**Other instances predicted**:
- FlashAttention varlen forward/backward (`cu_seqlens_q`, `cu_seqlens_kv` are exactly this CSR table).
- Variable-length softmax / cross-entropy with packed labels.
- Variable-length pooling / scatter-mean per-sequence.
- Per-sequence layernorm over packed TND inputs.
- Variable-length ROPE / KV-cache append.

**Risks before promotion**:
- Per-AIV startup cost: if `dimTCore` is tiny (1-2 tokens) and `B` is large, linear scan dominates. Worker should fall back to BSH path or precompute per-AIV start-batch in tiling host code.
- The `actualSeqQlen` tensor must be on GM (the kernel reads via `.GetValue()` which does a scalar GM load); putting it in a workspace UB cache buys nothing because each value is read at most twice.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel that consumes varlen layout end-to-end.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-RAU-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

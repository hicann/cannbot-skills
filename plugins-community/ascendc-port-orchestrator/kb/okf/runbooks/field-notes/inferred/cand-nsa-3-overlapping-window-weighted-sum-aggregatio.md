---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Overlapping-window weighted-sum aggregation as a 1-D \"convolution\" — fold a fine-grained per-element signal into per-chunk scores via strided gather + scalar-`Muls` + accumulating `Add`, with a precomputed triangular weight schedule"
description: "applies_to: any soc with public DataCopy(stride) + Muls + Add; cann=9.0.0+; op_class=block_sparse_attention_scoring overlapping_pool_aggregation windowed_per_chunk_reduce derived-from: cann-source (ns"
phenomenon: build_failure
signal:
  - "An op needs to produce per-chunk scalar scores from a finer-grained per-element signal where each output chunk's score is a weighted sum over an OVERLAPPING win"
confidence: inferred
status: stub
original_id: CAND-NSA-3
timestamp_inferred: true
tags: [candidate, inferred, muls, add, peakcover, datacopypad, blocklen, cand-nsa-3]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public DataCopy(stride) + Muls + Add; cann=9.0.0+; op_class=block_sparse_attention_scoring | overlapping_pool_aggregation | windowed_per_chunk_reduce`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — pre-TopK chunked importance score`
`unverified_on: a5_ops`

**Trigger**: An op needs to produce per-chunk scalar scores from a finer-grained per-element signal where each output chunk's score is a weighted sum over an OVERLAPPING window (window length `W` > chunk stride `M`, so adjacent chunks share `W − M` source elements). Typical examples: block-sparse attention importance scoring (compress `S2` softmax probabilities into `ceil(S2 / M)` per-block scores), strided pooling with overlap, fused 1-D convolution + sum-pool fast path. The triangular weights `t[i]` (for `i ∈ [0, W)`) are determined by how many output chunks each input position falls into, so weight[0]=1, weight[1]=min(2, peakCover), ..., peaks at `peakCover`, then mirrors back — this is a structural property of overlap geometry, NOT a learned parameter.

**Recommendation**: Three-step shape per AIV per row tile:

1. **Strided gather** the input row of length `S2` into a `[W_chunks, W]` UB layout using one `DataCopy(...DataCopyParams{blockCount=W_chunks, blockLen=W_blocks, srcStride=(M-W)_blocks, dstStride=0})` so the chunk axis becomes the outer (block) axis and the per-chunk-window axis is contiguous along the inner axis. (When `W` > `M`, srcStride is negative-relative; emit as a 2-D `DataCopyPad` with `srcStride=(M*sizeof(T))` and adjusted `blockLen` to avoid signed overflow.)
2. **Position-wise weighted accumulate**: for each `i ∈ [1, W)` (excluding the trivial `i=0` self-term), call `Muls(scoreScratch + i_offset, srcWindow + i_offset, static_cast<float>(t[i]), countPerChunk, repeat=W_chunks, BinaryRepeatParams{srcBlkStride=1, srcRepStride=stride_chunk, dstBlkStride=1, dstRepStride=stride_chunk})`, then `Add(score, scoreScratch + i_offset, score, countPerChunk, repeat=W_chunks - skip, BinaryRepeatParams{...})`. `PipeBarrier<PIPE_V>` between `Muls` and `Add`. The `t[i]` schedule is a host-emitted constant table of length `W` (see "Weight schedule" below).
3. **Cross-chunk pack**: the chunk's per-position-weighted partial is then collapsed across the inner `W` axis via the row-reduce path (CAND-FA4 / WholeReduceSum on the row, OR the head-pack tree of CAND-NSA-2 if the chunk axis is folded across groups).

**Weight schedule** (the triangular ramp — public arithmetic on the host):
```cpp
// W = window length, peakCover = max chunks any position contributes to
// t[i] for i ∈ [0, W): rises 1..peakCover, plateaus, descends peakCover..1
for (int i = 0; i < W; ++i) {
    if (i < W / 2)      t[i] = std::min<int>(i + 1, peakCover);
    else                t[i] = std::min<int>(W - i, peakCover);
}
```
The schedule is symmetric and has compact-support — corner chunks (first and last `W − M` of the row) have fewer covering windows than interior chunks.

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// Gather: per-chunk view of overlapping window
DataCopy(scoreScratch, srcRow,
    DataCopyParams{static_cast<uint16_t>(W_chunks),
                   static_cast<uint16_t>(W * sizeof(float) / 32),
                   static_cast<uint16_t>((M - W) * sizeof(float) / 32), 0});

// Weighted accumulate across positions in the window
for (int i = 1; i < W; ++i) {
    Muls(tmp, scoreScratch + i, static_cast<float>(t[i]),
         /*count=*/innerCount, /*repeat=*/W_chunks,
         BinaryRepeatParams{1, 1, chunkRepStride, chunkRepStride});
    PipeBarrier<PIPE_V>();
    Add(score, tmp, score, innerCount, W_chunks,
        BinaryRepeatParams{1, 1, 1, chunkRepStride, chunkRepStride, chunkRepStride});
    PipeBarrier<PIPE_V>();
}
```

**Why it works**: An overlapping-window weighted sum is equivalent to a 1-D depthwise convolution of the signal with the triangular kernel `t[]`, then downsampled at stride `M`. Direct DataCopy with a `M-W`-blocks stride realizes the overlap-and-downsample in one MTE2 issue, eliminating the per-output-chunk scalar gather (which would be `O(W_chunks × W)` scalar `GetValue`/`SetValue` pairs — the scalar-pipe-bound anti-pattern OL-82 / P-P86 cites). The `Muls` × `Add` pair per position runs vec-pipe-bound at one repeat over `W_chunks` chunks per call, so the total cost is `(W − 1) × (Muls + Add)` vec-pipe calls regardless of `W_chunks`. The triangular weight table is precomputed on the host once per tiling, avoiding per-iter scalar arithmetic.

**Determinism**: All operations are public `DataCopy` / `Muls` / `Add` / `PipeBarrier` — no atomic, no cross-core. The `(W − 1)` position-wise accumulations occur in fixed program order, so the per-chunk sum is a deterministic sequence `score = sum_i t[i] × src[chunk_off + i]`. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when the window stride `M >= W` (no overlap) — collapses to a non-overlapping pool that one `WholeReduceSum`-per-chunk or a single `DataCopy` + `Add` would handle more cheaply.
- Do NOT use this pattern when `W` is data-dependent (varies per row) — the static loop unroll over `i ∈ [1, W)` becomes a variable-trip loop that defeats compile-time scheduling; emit a separate dynamic-`W` kernel instead.
- Do NOT use this pattern when the weight schedule is NOT a low-arithmetic shape — for arbitrary learned weights, the `Muls` per position is fine but the per-chunk closed-form weight collapse used in the reference's overlap math (the triangular ramp from `[1, peakCover]`) does NOT generalize.
- Do NOT collapse the gather DataCopy into a single contiguous `DataCopy` when `M < W` — the overlap forces `srcStride < 0` semantically; the source must be re-issued per overlap-pair via separate DataCopys (or, for moderate `W − M`, a `DataCopyPad` with a positive offset reset between blocks).

**Other instances predicted**:
- Block-sparse attention pre-TopK scoring: NSA-class importance score, native-sparse-attention chunk score, dilated-attention windowed score.
- Strided 1-D pooling with overlap (audio frame energy, NLP token-level chunk pooling, sliding-window L2 norm).
- Fused conv1d + sum-pool fast path when kernel size > stride.
- Compressed top-K input preparation: any kernel that needs a per-block summary stat before a `TopK<>` over the compressed length.
- Speculative-decode draft-score aggregation across token spans.

**Risks before promotion**:
- a5_ops has not shipped a block-sparse / overlapping-pool op yet; the pattern is unverified on a5_ops perf and precision.
- The strided `DataCopy` `(M - W) * sizeof(T) / 32` block-stride must fit `uint16_t srcStride` — for large `W` and small chunks the stride can overflow; emit per-chunk-pair DataCopy in that case.
- The triangular weight schedule above is correct only when every input position contributes to BETWEEN 1 AND `peakCover` chunks; boundary rows of the matrix (first / last `W − M` elements) need explicit zero-pad before the weighted accumulate, otherwise the partial sums under-count.
- Per-position `Muls` repeats with a non-unit `srcRepStride` were observed to require an explicit `PipeBarrier<PIPE_V>` between successive `Muls` calls on the same destination region — omitting the barrier produces a write-write hazard on V220-class AIV (silent wrong-result).
- This pattern produces per-chunk SCORES; the downstream TopK over those scores is a separate concern (use the public `AscendC::TopK<>` per P-P85, not a hand-roll).

**Cross-reference**:
- P-P85 (`AscendC::TopK` adv_api primitive) — the natural downstream consumer of the per-chunk scores this candidate emits.
- P-P62 (Row-Scalar VEC Multiply via Brcb) — same `BinaryRepeatParams` repeat-stride family; both candidates use the same Add/Muls overload pattern.
- patterns/domains/reduction_quant.md — should add an "overlapping windowed pool" entry when this candidate promotes.
- OL-82 / P-P86 (scalar-pipe-bound anti-pattern for fused-op scoring) — this candidate is the vec-pipe-clean alternative to the scalar `GetValue`/`SetValue` per-position naive form.

**Promote when**: an a5_ops fused op (e.g. a future block-sparse attention, sliding-window pool, or fused TopK over per-block scores) ships with overlapping-window aggregation AND msprof shows `aiv_vec_ratio > 0.6` for the scoring phase (proving the implementation stayed vec-pipe-bound) AND the per-chunk score output matches a reference triangular-overlap weighting within bit-exact tolerance for fp32 / 1-ULP tolerance for fp16/bf16.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-NSA-3，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

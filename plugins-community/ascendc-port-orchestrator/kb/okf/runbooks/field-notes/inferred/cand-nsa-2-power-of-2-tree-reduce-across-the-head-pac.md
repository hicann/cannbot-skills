---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Power-of-2 tree-reduce across the head-pack axis via strided `Add` repeats — fold `G` query-heads-per-KV-group into one row in `log2(G)` vector passes"
description: "applies_to: any soc with public AscendC Add(BinaryRepeatParams stride/repeat) ; cann=9.0.0+; op_class=group_query_attention_head_pack_reduce per_row_head_group_sum gqa_fused derived-from: cann-source"
phenomenon: build_failure
signal:
  - "An op packs G query-heads per KV head (group-query attention with G > 1) and needs to reduce a per-head per-block tensor of shape [S1_tile, G, K] (rows × heads-"
confidence: inferred
status: stub
original_id: CAND-NSA-2
timestamp_inferred: true
tags: [candidate, inferred, add, datacopy, uint8_t, cand-nsa-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public AscendC Add(BinaryRepeatParams stride/repeat) ; cann=9.0.0+; op_class=group_query_attention_head_pack_reduce | per_row_head_group_sum | gqa_fused`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — importance-score reduce across head-pack`
`unverified_on: a5_ops`

**Trigger**: An op packs `G` query-heads per KV head (group-query attention with `G > 1`) and needs to reduce a per-head per-block tensor of shape `[S1_tile, G, K]` (rows × heads-per-group × per-head-cols) into `[S1_tile, K]` summed across the head-pack axis. The reduce target is NOT a row reduction (CAND-FA4's territory) — it is across a middle axis whose stride in UB equals `K` × `sizeof(elem)`. `G` is power-of-2 (the gQA spec — typical values 2/4/8/16).

**Recommendation**: Reduce in `log2(G)` passes via a strided `Add`. At pass `p` (with `p = 1, 2, 4, ..., G/2`), pairs are `(row_i, row_{i+p})` for every other row at distance `p` along the head-pack axis. Express each pass as a single `Add` call with a `BinaryRepeatParams{srcBlkStride=1, dstBlkStride=1, srcRepStride=2p×K×elem/32, dstRepStride=2p×K×elem/32}` repeat-stride, repeat count `S1_tile × G / (2p)`, and `count = K` elements per repeat. Each pass halves the live head-pack length; after `log2(G)` passes the head-pack is fully folded and the surviving rows are `2p × K`-strided in UB. A final compact `DataCopy` with `dataCopyParams{blockCount=S1_tile, blockLen=K_blocks, srcStride=(G-1)×K_blocks, dstStride=0}` gathers the folded rows back to a contiguous `[S1_tile, K]` layout.

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
// in:  buf[S1_tile][G][K] (fp32, UB), G is a power of two, K is repeat-count
// out: buf[S1_tile][0][K] with the surviving (folded) rows G*K-strided
for (uint32_t p = 1; p < G; p *= 2) {
    uint8_t stride = static_cast<uint8_t>(2 * p * K * sizeof(float) / 32);  // 32B = 1 vector block
    Add(buf, buf[p * K], buf,
        /*mask=*/K,
        /*repeat=*/static_cast<uint8_t>(S1_tile * (G / (2 * p))),
        /*params=*/{1, 1, 1, stride, stride, stride});
    PipeBarrier<PIPE_V>();
}
// optional compact pack: gather every G-th row back to contiguous output
DataCopy(out, buf, DataCopyParams{S1_tile, K * sizeof(float) / 32,
                                  (G - 1) * K * sizeof(float) / 32, 0});
```

If a single pass's per-element repeat-stride exceeds the architecture's signed `uint8_t` block-stride window (256 vector blocks), fall back to a manual outer loop emitting one `Add` per surviving row pair at that pass. The pattern remains `log2(G)`; only the unrolling shape changes:
```cpp
if (sgBlock < 256) { Add(... repeat=S1_tile*(G/(2*p)) ...); }
else { for (int r = 0; r < S1_tile*(G/(2*p)); ++r) { Add(...); } }
```

**Why it works**: Head-pack reduce sits between row reduce (`WholeReduceSum` / `BlockReduceSum`) and across-AIV reduce (cross-core). Public `WholeReduce`/`BlockReduce` primitives reduce along the *last* axis only — they do not address a middle axis. Bridging the middle-axis reduce via per-pair `Add` with a stride that doubles each pass costs `log2(G)` vector instructions in `K`-element-mask form per pass and `S1_tile × G / 2` total `Add`-mask repeats — the same vec-pipe budget as a tree reduce inside one row, paid across passes rather than across the row.

**Determinism**: Pairwise `Add` with a fixed (compile-time-determined) pair schedule produces a deterministic reduction tree. No cross-core write participates. `PipeBarrier<PIPE_V>` between passes ensures the `Add` of pass `p` completes before pass `2p` reads its result. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT use this pattern when the reduce axis is the LAST axis of UB — public `WholeReduceSum`/`BlockReduceSum` are faster on the last axis (single primitive vs `log2(G)` `Add`s).
- Do NOT use this pattern when `G` is not a power of two — the tree shape breaks; either pad to next power of two with `Duplicate(... 0)` on the pad rows or emit a non-power-of-two manual loop.
- Do NOT use this pattern when `K * sizeof(elem) < 32B` — a vector block carries only 32 bytes, so a per-repeat mask of `K` smaller than one block forces sub-block addressing the `Add` overload doesn't support; pad `K` to a 32B-multiple in UB first.
- Do NOT use this pattern with `Add` repeat-strides above the architecture's `uint8_t` limit without the manual-loop fallback shown above — silent stride wrap.

**Other instances predicted**:
- Any gQA fused-attention forward that emits a per-head intermediate and needs to fold heads before downstream reduce / mask / sort.
- Fused row-norm + scatter where multiple feature groups must be summed before the scatter (e.g. group-norm pre-affine + write).
- Multi-head per-row statistics (mean, variance) when the reduction crosses head-packs and the tail axis is already small enough to keep in UB.
- MoE per-expert per-row sums when each row visits multiple expert outputs and they must be combined before write.

**Risks before promotion**:
- a5_ops has not shipped a gQA op with `G > 1` head-pack reduce; the pattern is unverified on a5_ops perf.
- Repeat-stride 256-block wrap is observed in CANN reference's `if (sgBlock < 256)` branch — copy the manual-loop fallback verbatim if the kernel's `(2p × K × sizeof(elem)) / 32` can exceed 255.
- `PipeBarrier<PIPE_V>` between passes is mandatory; omitting it produces silent wrong-result on V220-class AIV because the pass-`p` `Add` writes the source of pass-`2p` (same UB region — read-after-write hazard).
- This pattern assumes the in-place destination matches the first input — both `dst` and one `src` are `buf` and the other `src` is `buf[p × K]`. Using a separate dst buffer doubles UB.

**Cross-reference**:
- CAND-FA4 (tree-reduce wide fp32 rows via packed `BlockReduceMax/Sum` partials) — this candidate is the orthogonal middle-axis case; both compose in one kernel (row reduce inside a head, then head-pack reduce across heads).
- P-P62 (Row-Scalar VEC Multiply via Brcb — same `BinaryRepeatParams` stride family) — same primitive class, different reduction shape.
- patterns/domains/reduction_quant.md (reduction shapes index) — should add a "middle-axis head-pack reduce" entry when this candidate promotes.

**Promote when**: an a5_ops fused gQA op (e.g. a future GQA attention forward or fused gQA + scoring kernel) ships with `G > 1` per-row head-pack reduce AND msprof shows the `log2(G)` `Add`-pass cost is below the alternative (UB-spill + WholeReduce-over-flattened-axis or per-row scalar loop).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-NSA-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

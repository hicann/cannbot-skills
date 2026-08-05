---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Tree-reduce wide fp32 rows via packed `BlockReduceMax/Sum` partials with per-stage mask sizing (no `WholeReduce` on fp32)"
description: "applies_to: op_class=row_reduction (W ≥ 64 fp32 elements per row, R parallel rows in UB); soc=Ascend910_V220 / Ascend950PR (fp32 BlockReduceMax/BlockReduceSum confirmed public-API); cann=9.0.0+ derive"
phenomenon: build_failure
signal:
  - "Need rowmax or rowsum of a wide fp32 row (W ≥ 64 fp32 per row, R parallel rows packed in UB) inside a hot kernel loop (FA softmax, LayerNorm rowstats, RmsNorm r"
confidence: inferred
status: stub
original_id: CAND-FA4
timestamp_inferred: true
tags: [candidate, inferred, wholereduce, getvalue, dstblkstride, dstrepstride, repeattime, cand-fa4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: op_class=row_reduction (W ≥ 64 fp32 elements per row, R parallel rows in UB); soc=Ascend910_V220 / Ascend950PR (fp32 BlockReduceMax/BlockReduceSum confirmed public-API); cann=9.0.0+`
`derived-from: cann-source (FA-class fp32 online-softmax row reductions, 2026-05-10 revise-cl5)`
`verified_on: cann-source (read-only); unverified_on: a5_ops`
`local-kb-crossref: P-P47 (half-interval scalar finish), ascend950pr.md Sort/Reduce VEC primitive specs`

**Trigger**: Need rowmax or rowsum of a wide fp32 row (`W ≥ 64` fp32 per row, `R` parallel rows packed in UB) inside a hot kernel loop (FA softmax, LayerNorm rowstats, RmsNorm rowsumsq, wide softmax). A scalar `GetValue`-driven reduction is `R × W` scalar-pipe cycles and is the canonical anti-pattern.

**Recommendation — corrected mechanics**:

Use chained `AscendC::BlockReduceMax<float, false>` / `BlockReduceSum<float, false>` stages, where each repeat consumes one 64-fp32 vector (organized as 8 contiguous 32-byte blocks of 8 fp32) and writes **8 packed fp32 partials** densely. Do not use `WholeReduceMax/Sum<float>` to finish — the fp32 cap is 64 elements per repeat, and the finishing stage is more economical as a final `BlockReduce*` with a strided per-block mask.

After one full stage, a row of `W` fp32 collapses to a packed row of `ceil(W / 8)` partials. After two stages, `ceil(W / 64)`. After three stages, `ceil(W / 512)`. Continue staging until the per-row partial count is `≤ 8`, then finish.

Mask use across stages (this is the part that the prior candidate had wrong):

- The template arg `<float, false>` means `isSetMask = false`, i.e. **use the externally configured vector mask**; it does **not** mean "no mask". When the per-repeat workload is a full 64-fp32 vector, set the mask to all-ones once and leave it.
- The vector mask gates **which source lanes contribute per repeat**, not which output lanes are written. The dst is always the dense packed-partial layout, determined by `dstBlkStride`/`dstRepStride`/`repeatTime`.
- **Stages 1 through N-1** (full fan-in): full mask, no per-stage mask change.
- **Final stage**, when the packed row is now `k` valid partials per 8-element block (because the previous stage produced fewer than 8 partials per row but they sit in an 8-lane block layout), set a **tiled strided mask** of the form "low `k` bits of every 8-bit byte" so each repeat consumes only the `k` valid lanes per block. Compute this mask as `((1<<k)-1)` replicated into bytes 0..7 of both `lo` and `hi` halves of the mask register, then call `AscendC::SetVectorMask<int8_t>(tiledMask, tiledMask)`. After the final reduction, **restore full mask** with `AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1)` before continuing.
- For an intermediate stage whose source row is wider than 8 partials but shorter than 64 (e.g. 32 packed partials per row from a `W=256` first stage), set a **contiguous** mask of that element count (`SetVectorMask<int8_t>` with the standard contiguous form) and lower `srcRepStride` to `partials/8` blocks instead of the full 8.

**Important non-claim**: `SetVectorMask` is not used to "skip every 8th element" between stages. The packed partial layout means the next stage reads the partials densely from the front of each row. The only legitimate inter-stage mask uses are (a) reducing the contiguous element count when the packed row is shorter than a full 64-fp32 vector, and (b) the tiled per-block mask in the final stage when each 8-block holds fewer than 8 valid partials.

**Shape contract (single-row-at-a-time view; in practice R rows are packed and reduced in parallel)**:
```cpp
// stage 0 : src         [R, W]
// stage 1 : partial1    [R, ceil(W / 8)]     full mask, full repeats
// stage 2 : partial2    [R, ceil(W / 64)]    contig mask = partial1 row width if < 64
// stage k : partial_k   [R, ceil(W / 8^k)]   continue while row width > 8
// finish  : row_out     [R]                  tiled per-block mask = row width
```

The variant choice (how many stages, what mask form on the final stage) is fixed once `W` is known. Below is the chosen variant per `W`. There is one shape contract per variant; do not list a recommended path then call it "optional".

**Variant: W = 64** — single stage suffices.
```cpp
// repeats = R, full mask (set once outside).
AscendC::BlockReduceSum<float, false>(rowSumUb, srcUb,
    /*repeatTime=*/R,
    /*dstRepStride(scalar)=*/0, /*srcBlkStride=*/1,
    /*dstRepStride=*/1, /*srcRepStride=*/8);
AscendC::PipeBarrier<PIPE_V>();
// Per-row 8 partials per output block; final per-row finish uses a tiled mask of 8 lanes,
// which is just full mask -> a second BlockReduceSum with repeats=R*8/64.
```

**Variant: W = 256** — three stages, intermediate contiguous mask of 32 plus final tiled mask of 4.
```cpp
constexpr uint32_t WALIG = 256;
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t FP32_VEC = 64;

// Stage 1: [R,256] -> [R,32] packed partials. Full mask, full repeats.
AscendC::BlockReduceSum<float, false>(scratch, srcUb,
    /*repeatTime=*/R * WALIG / FP32_VEC,
    /*dstRepStride(scalar)=*/0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();

// Stage 2: [R,32] -> [R,4] packed partials.
// Set contiguous mask = 32 (one repeat consumes 32 fp32 from one row's packed partials).
SetContiguousVectorMask(/*count=*/32);   // see helper below
AscendC::BlockReduceSum<float, false>(scratch2, scratch,
    /*repeatTime=*/R,
    /*dstRepStride(scalar)=*/0, 1, 1, /*srcRepStride=*/4);   // 32 fp32 = 4 blocks
AscendC::PipeBarrier<PIPE_V>();

// Stage 3 (finish): [R,4] -> [R]. Each row's 4 partials sit in lanes 0..3 of an 8-block;
// tiled mask = ((1<<4)-1) replicated into every byte of the 128-bit mask register.
SetTiledBlockReduceMask(/*lanesPerBlock=*/4);    // see helper below
AscendC::BlockReduceSum<float, false>(rowSumUb, scratch2,
    /*repeatTime=*/CeilDiv(R * FP32_BLOCK, FP32_VEC),
    /*dstRepStride(scalar)=*/0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();

// MANDATORY: restore full mask before any unrelated VEC work.
AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
```

**Variant: W = 512 (R rows, R*W ≥ 4096)** — three stages, full mask throughout, no `SetVectorMask` between stages. The repeat counts implicitly carry the fan-in.
```cpp
constexpr uint32_t WALIG = 512;
constexpr uint32_t FP32_BLOCK = 8;
constexpr uint32_t FP32_VEC = 64;

AscendC::BlockReduceSum<float, false>(scratch, srcUb,
    /*repeatTime=*/R * WALIG / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
AscendC::BlockReduceSum<float, false>(scratch2, scratch,
    /*repeatTime=*/R * WALIG / FP32_BLOCK / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
AscendC::BlockReduceSum<float, false>(rowSumUb, scratch2,
    /*repeatTime=*/R * WALIG / FP32_VEC / FP32_VEC, 0, 1, 1, 8);
AscendC::PipeBarrier<PIPE_V>();
// Mask was never altered; no restore needed.
```
Constraint: this exact form requires `R * W` divisible by `FP32_VEC * FP32_VEC = 4096`. For `W = 512` this means `R ≥ 8`. For lower `R`, use the W=256-style ending (intermediate contig mask + final tiled mask).

**Variant: W > 64 generic (loop accumulator)** — outer loop over 64-fp32 chunks, accumulate per-row via `AscendC::Max` / `AscendC::Add`, handle tail with `SetContiguousVectorMask(W % 64)` then `SetTiledBlockReduceMask(CeilDiv(W % 64, 8))` for the final finish, mask restored at end.

**Public mask helpers (logic ported from observed pattern)** — implementable on a5_ops side without reading CANN:
```cpp
__aicore__ inline void SetContiguousVectorMask(uint32_t count) {
    // count: total fp32 lanes to enable, 1..128 (across the two 64-bit mask halves).
    if (count == 128 || count == 0) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
    } else if (count >= 64) {
        uint64_t lo = ((uint64_t)1 << (count - 64)) - 1;
        AscendC::SetVectorMask<int8_t>(lo, (uint64_t)-1);
    } else {
        uint64_t lo = ((uint64_t)1 << count) - 1;
        AscendC::SetVectorMask<int8_t>(0x0, lo);
    }
}
__aicore__ inline void SetTiledBlockReduceMask(uint32_t lanesPerBlock) {
    // lanesPerBlock: 1..8, enables low-`lanesPerBlock` lanes in each 8-lane block.
    if (lanesPerBlock < 1 || lanesPerBlock > 8) {
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        return;
    }
    uint64_t sub = ((uint64_t)1 << lanesPerBlock) - 1;
    uint64_t tiled = (sub <<  0) | (sub <<  8) | (sub << 16) | (sub << 24)
                   | (sub << 32) | (sub << 40) | (sub << 48) | (sub << 56);
    AscendC::SetVectorMask<int8_t>(tiled, tiled);
}
```

**Determinism**: Hardware reduction order within one `BlockReduce` repeat is fixed (block-local tree). With one AIV owning each row, or a deterministic cross-AIV merge, same input → same output. Determinism does NOT come from "layout choice"; it comes from (a) one writer per output slot and (b) fixed reduction order per repeat. Scratch tiles must be initialized whenever any producer might skip lanes that a later reader covers (EC-37-style).

**Hard do-not-apply**:
- Non-fp32 dtypes: the half-precision path uses `WholeReduceMax/Sum<half>` with up to 128 half lanes per repeat — a different shape contract; use a separate pattern.
- Socs without confirmed public fp32 `BlockReduceMax/Sum` support: do not assume "any soc". Confirmed: Ascend910_V220, Ascend950PR (FA-class fp32 epilogue). For other targets, probe-compile `BlockReduceMax<float, false>(…)` and validate one-row results before adopting.
- Rows with `W < 64`: a single masked `BlockReduce*` (or even `WholeReduceMax/Sum<float>` capped at 64) is simpler; do not introduce a tree.
- Output destination is GM (not UB): the `BlockReduce*` chain writes to UB; only the final `DataCopy` to GM is allowed.
- Scratch tiles that are public outputs of the kernel: do not place tree-stage scratch in a buffer that the kernel exports; use a private tile.

**Other instances predicted**:
- LayerNorm row mean / row variance for H ≥ 64 (replace scalar-loop reduction).
- RmsNorm rowsumsq for H ≥ 64.
- Attention rowmax + rowsum (this op).
- GroupNorm per-group mean where group size ≥ 64.
- Vocabulary-class wide softmax rowmax + rowsumexp.
- Online softmax in fused attention variants (FA forward/backward).

**Risks before promotion**:
- The tiled per-block mask on the final stage is the single fragile point. Wrong `lanesPerBlock` → silent wrong result. Tested case shapes must cover at least one power-of-two width (e.g. W=256), one with intermediate-contig + tiled-final (W=256), and one full-fan-in-no-mask (W=512, R≥8).
- The `<false>` form requires the external mask to be set correctly before each call. Forgetting to restore the full mask after a tail or final stage breaks unrelated downstream VEC ops. The restoration line `AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1)` is part of the contract.
- For `W = 512` full-fan-in variant: requires `R * W` divisible by 4096. If a kernel runs at smaller `R`, do not silently fall through to this variant — pick the masked variant.
- The `dstBlkStride` parameter for these calls is documented as 0 in the FA fp32 path (no per-row scaling); confirm this against AscendC public docs for the target SoC version before promoting, because the parameter semantics for `BlockReduce*` differ between half-precision and fp32 forms.
- Performance claim "100× speedup over scalar" is plausible but **unmeasured on a5_ops**. Promote only after one a5 kernel ships this pattern with `msprof` data showing the expected `aiv_vec_ratio` lift.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Online-softmax per-row stat lives as a `[m, 8]` datablock-packed buffer (1 fp32 block/row, 8 identical lanes) — extract by indexing `row*8`, broadcast-apply across columns via `src1BlkStride=0 / src1RepStride=1` (Brcb only when the stat is in `[m, 1]` form)"
description: "applies_to: any SoC with public AscendC online-softmax (Softmax/SoftmaxFlashV2) + VEC Mul/Div/Brcb + BinaryRepeatParams; cann=9.0.0+; op_class=online_softmax_row_rescale / flash_attention_forward / fu"
phenomenon: build_failure
signal:
  - "A fused attention / online-softmax kernel must (a) read the per-row max/sum reduction that Softmax/SoftmaxFlashV2 produced, and (b) row-broadcast it across the"
confidence: inferred
status: stub
original_id: CAND-FA-SOFTMAX-STAT-1
timestamp_inferred: true
tags: [candidate, inferred, softmax, softmaxflashv2, nan, softmax_sum, sumtensor, cand-fa-softmax-stat-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with public AscendC online-softmax (Softmax/SoftmaxFlashV2) + VEC Mul/Div/Brcb + BinaryRepeatParams; cann=9.0.0+; op_class=online_softmax_row_rescale / flash_attention_forward / fused_attention_with_softmax_stat / streaming_softmax_divide`
`derived-from: cann-source (FA forward reference vec epilogue, 2026-06-03 gb4softmaxstat)`
`evidence_family: FA-SOFTMAX-STAT`
`verified_on: public AscendC Softmax API doc (sumTensor/maxTensor last-axis = fixed 32 B = 1 datablock, all lanes identical) + FA forward reference vec divide/rescale epilogue (kernel-structural)`
`unverified_on: a5_ops`

**Trigger**: A fused attention / online-softmax kernel must (a) read the per-row max/sum reduction that `Softmax`/`SoftmaxFlashV2` produced, and (b) row-broadcast it across the head-dim columns to rescale or divide the attention output (`O[m, cols] /= sum[m]`, or `P[m, cols] *= alpha[m]`). The precision-critical failure mode this closes: the divide produces `nan` and the stored `softmax_sum` is systematically wrong because the per-row stat was read/broadcast with the wrong stride convention.

**The layout principle (the "why 8")**: the public online-softmax reduction output is NOT a contiguous `[m]` (one value per row). It is a `[m, B]` datablock-packed buffer where `B = 32 bytes / sizeof(reduce_dtype)` = **8 for fp32** (one hardware datablock per row, **all B lanes hold the same reduced value** — this is the documented public contract of the `sumTensor`/`maxTensor` outputs: "last axis fixed to one datablock, all data in the block identical"). Row `i`'s stat lives at `buf[i*8 .. i*8+8)`. There is therefore **no separate `[m,8]→[m]` extraction step** — to read row `i` you index any lane of its block (`buf[i*8]`); to feed the broadcast you point at the block base and let the stride convention spread it.

**Two regimes (pick by the reduce-tail width the API was told to use)**:
1. **Stat already `[m, 8]`** (the default, reduce-tail = 8): feed `stat[row*8]` directly into the apply with the broadcast `BinaryRepeatParams` below. **No Brcb.**
2. **Stat in `[m, 1]`** (reduce-tail = 1, contiguous one-per-row): expand to `[m, 8]` FIRST with a single `Brcb(blk, stat, (m+7)/8, {1, 8})` (block params dstBlkStride=1, dstRepStride=8), then apply identically. This is the same `Brcb` shape CAND-FA-AUX-OUT-1 uses for the GM emit — here it feeds the in-kernel apply instead.

**The broadcast-apply across columns (`[m, 8] → [m, cols]`, the load-bearing precision fix)**: a single `Div`/`Mul` whose `BinaryRepeatParams` broadcasts the per-row datablock across all columns within one repeat (`src1BlkStride = 0`) and advances exactly **one datablock per row/repeat** (`src1RepStride = 1`). The data operand walks columns normally (`*BlkStride = 1`, `*RepStride = cols_aligned / 8`). `repeatTimes = m` (rows), the per-repeat element count = the column slice. The divisor/multiplier index uses the **`*8` block stride** to land on the correct row's block: `stat[rowTileBase * 8]`.

**Concrete anchor** (public-API; worker-local names; runnable):
```cpp
// stat   : LocalTensor<float>, the per-row reduction. [m] OR [m,8] (see regimes).
// o      : LocalTensor<float>, attention output, [m, cols] contiguous in cols.
// m      = rows in this tile; cols = head-dim (32B-aligned); B32 = 8 (fp32 datablock).
constexpr int B32 = 8;
const int colsAlign = AlignUp(cols, B32);

// Regime 2 ONLY: expand contiguous [m] -> [m,8] datablock form first.
LocalTensor<float> statBlk = tmp.Get<float>();          // [m*8]
AscendC::Brcb(statBlk, stat, (m + 7) / B32, {1, 8});    // {dstBlkStride=1, dstRepStride=8}
AscendC::PipeBarrier<PIPE_V>();
// Regime 1: skip Brcb; use the API's [m,8] output directly as statBlk.

// Row-broadcast divide O[r, :] /= stat[r] for all r:
AscendC::BinaryRepeatParams rp;
rp.src0BlkStride = 1;                 // numerator (O) contiguous in cols
rp.src0RepStride = colsAlign / B32;   // O advances one row (cols/8 blocks) per repeat
rp.src1BlkStride = 0;                 // KEY: stat block broadcast across all cols
rp.src1RepStride = 1;                 // KEY: stat advances exactly one datablock per row
rp.dstBlkStride  = 1;
rp.dstRepStride  = colsAlign / B32;
const int chunk = 64;                 // fp32 elems per repeat (one vector instr width)
for (int c = 0; c < colsAlign / chunk; c++) {
    AscendC::Div(o[c * chunk], o[c * chunk], statBlk[/*rowTileBase*/0 * B32],
                 /*mask=*/chunk, /*repeatTimes=*/m, rp);
}
// (mirror with Mul + statBlk = alpha for the exp-rescale step; for a numerator
//  broadcast — e.g. P *= alpha where alpha is src0 — swap which operand carries
//  src*BlkStride=0 / src*RepStride=1.)
```

**Why the wrong way goes nan / wrong sum** (symptom anchor, kw-gb3 FA graybox 0/8):
- Reading the stat as if it were contiguous `[m]` (stride 1) when the API wrote `[m,8]` reads 8× too few rows — every 8th row's stat used for 8 consecutive rows → garbage `softmax_sum` (~60–490 observed) and div-by-corrupt → `nan` in `attention_out`.
- Using a fractal `Copy` + `CopyRepeatParams` for the stat broadcast: `CopyRepeatParams` is the **ND↔NZ fractal-layout reshuffle** primitive (for the matmul-result transpose), NOT a row-broadcast. The stat broadcast is **never** a `Copy`; it is `Brcb` (expand, regime 2 only) + `Div`/`Mul`-with-`BinaryRepeatParams` (apply). Conflating the two is the documented bug.
- Setting `src1RepStride = 0` (instead of 1) broadcasts the FIRST row's stat to ALL rows — looks numerically plausible (no nan) but is silently wrong. The non-zero `src1RepStride = 1` (one datablock/row) is load-bearing.

**Reject_cond**: do NOT use when
- The stat is a single per-tile scalar (not per-row) — a plain `Muls`/`Div`-by-scalar is correct, no broadcast stride needed.
- `cols % 64 != 0` without a tail handler — the trailing partial chunk needs a reduced-mask `Div` (not shown).
- `m > 255` — `repeatTimes` is uint8; split the row loop.
- A scalar `GetValue(r)`+`Muls` per-row loop is acceptable for correctness (the agent's current hand-roll does this and it works for the APPLY) — this entry's value is the **layout/extraction convention** that makes the stat itself correct, plus the de-scalarized broadcast for perf.

**Relationship to existing KB** (C35-disambiguated — overlaps each on ≤1 reason code, NOT ≥2):
- **CAND-FA-AUX-OUT-1**: covers `Brcb([m]→[m,8])` + DataCopy for **emitting** the stat to GM (write-out). This entry covers the **in-kernel consume/apply** broadcast (read-back + rescale) and the `reduceSize==8`-vs-`==1` regime. Sibling, opposite direction (emit vs apply). Shared: `Brcb {1,8}` shape — same datablock convention, deliberately consistent.
- **CAND-RAU-3**: generic `[R,inner] *= [R,1]` stride broadcast with `src1RepStride = softmax_tail/8`. This entry specializes it to the FA `[m,8]`-block-from-SoftmaxFlashV2 case (`src1RepStride = 1` exactly, because the stat tail IS one datablock) AND adds the `Div`/numerator-vs-denominator distinction + the regime split. Use RAU-3 for the general per-row-scale shape; use this for the FA online-softmax stat specifically.
- **cv_reference_concrete_params.md §softmax_online**: gives the `wsMetaGm_` GM-region cross-core handoff stride; this entry is the missing in-UB `[m,8]` layout + extraction/broadcast primitive that §softmax_online lacked (the kw-gb3 gap). Mirrored into that section.

**Other-instances-predicted**: any online/streaming reduction that row-broadcasts a per-row stat across an inner dim — LayerNorm/RMSNorm `x *= rstd[row]`, BatchNorm `(x - mean[row]) * inv_std[row]`, GroupNorm, softmax-with-temperature, any attention variant (GQA/MLA/paged) reusing the same `Softmax`/`SoftmaxFlashV2` stat outputs.

**Promote when**: an a5_ops FA kernel adopts the `[m,8]`-block consume + broadcast-Div and clears the 0/8 precision blocker (clean `attention_out`, correct `softmax_sum`), confirming the convention closes the bug; cross-validate on one norm-class op (RMSNorm or GroupNorm) using the same broadcast shape.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-SOFTMAX-STAT-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

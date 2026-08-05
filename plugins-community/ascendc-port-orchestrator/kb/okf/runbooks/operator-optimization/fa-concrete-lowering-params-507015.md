---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "FA 类具体 lowering 参数(507015 三处高危决策)"
description: "FA-class concrete lowering params (PoC) — the 3 507015-risk decisions What this is + why. This doc is the self-sufficient, verified grounding for the 3 FA lowering decisions that 507015 punishes — mat"
confidence: single_run
original_id: doc/target/ascendc/fa_class/cv_reference_concrete_params.md
timestamp_inferred: true
tags: [fa-class, 507015, matmul-primitive, softmax-online, concrete-params, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# FA-class concrete lowering params (PoC) — the 3 507015-risk decisions

> **What this is + why.** This doc is the **self-sufficient, verified grounding**
> for the 3 FA lowering decisions that 507015 punishes — `matmul_primitive`,
> `cross_core_sync`, `softmax_online`. The ascendc-agent **generates** these from
> the concrete params recorded HERE; it must NOT reach for any cv-agent kernel at
> runtime. (A customer runs op-gen with no cv-agent checkout on disk — the
> knowledge has to live in this KB doc, and it does, in full, below.) The 2026-05-26
> decision-manifest comparison found these were the 3 decisions our KB previously
> held only as abstract patterns (CAND-FA1/2/3) with no concrete values; the
> concrete values below close that gap.
>
> **Provenance (learning source, runtime-invisible)**: these params were *learned*
> from the team's own cv-agent tooling output and distilled into the verified forms
> below. cv-agent is our internal learning source, **not CANN source** and **NOT a
> runtime dependency** — owner direction 2026-05-27: a chain that reads cv-agent at
> emit time means the knowledge wasn't truly learned into the KB (substandard). The
> forms below ARE that learned knowledge; treat them as the grounding, not as an
> index into an external kernel. PoC status: wired into the FA template-assembly
> decision contract; A3
> hard-case (D≥256 / large seq) verification of 507015-avoidance is the remaining
> proof step.

## decision_id: `matmul_primitive` (Mmad + Fixpipe params)

The L0C/Fixpipe params are what 507015 (ECC read / L0C read-write conflict)
punishes. cv-agent's working values:

```cpp
// Mmad accumulate-init: init the C-matrix ONLY on the first K-tile, accumulate after.
MmadParams mp;
mp.cmatrixInitVal = (ki == 0);   // ki = current K-loop index
mp.cmatrixSource  = false;       // C source from L0C accumulator, not external
```

**`Fixpipe` `srcStride` units are PER-CONTEXT — NOT a single value** (corrected
2026-05-27 after an A3 PoC; see the lossy-extraction note below). cv-agent's OWN
reference uses BOTH forms, chosen by the Fixpipe call's source layout:

```cpp
// (A) matmul-output / intermediate L0C->workspace copies (the `p` Fixpipe):
//     srcStride is in units of C0 ( = 32 / sizeof(dtype); 16 for fp16 ).
p.srcStride = mActAlign / C0;     // cv-agent cube.h lines 101/152
// (contiguous / single-row variant of the same copy uses p.srcStride = 1)

// (B) final-output L0C->GM write copies (the `fixParams` Fixpipe):
//     srcStride is in ELEMENTS, NOT C0-units.
fixParams.srcStride = mActAlign;  // cv-agent cube.h lines 201/284
```

Why it matters: a wrong `srcStride` unit makes Fixpipe read L0C out of tile bounds
→ ECC read failure 507015 (`fixp_error0`) OR a silent precision break. The unit
DEPENDS on the copy's source layout — using `/C0` for a `fixParams`-style call (or
elements for a `p`-style call) is the fault.

> **Completeness lesson (2026-05-27 PoC).** The first version of this doc recorded
> only `srcStride = mActAlign / C0` and presented it as THE value — a lossy
> hand-summary that dropped the per-context distinction (the `fixParams = mActAlign`
> elements form); an A3 PoC that applied the blanket `/C0` mis-grounded the
> `fixParams` calls. **The fix was to capture the full per-context distinction in
> THIS doc** (both forms (A) and (B) above), so the agent generates correctly from
> the KB alone. **Conclusion: when a KB form is found incomplete, the fix is to make
> THIS doc complete — never to defer to reading an external cv-agent kernel at
> runtime** (the customer has none; deferring would re-introduce the dependency
> owner ruled out 2026-05-27). The forms above are now the per-context-complete
> grounding; if a future gap is found, distill it HERE. (Historical cross-check:
> cv-agent's reference compiled + 2/2 precision + 4.16× over PyTorch base on our A3,
> independent prototype 2026-05-27 — that was the *learning* validation, not a runtime read.)

## decision_id: `cross_core_sync` (WorkspaceQueue flag-ID + pipe discipline)

PB-35 records that the **cube-internal `event_t(0)` pipe sync** collides with the
AIC↔AIV cross-core flag chain in `MIX_AIC_1_2` mode → silent V220 hang ("use
IDs≥4" was falsified). cv-agent's working scheme sidesteps it entirely by keeping
ALL cross-core sync as MODE-`0x2` CrossCore flags **inside the WorkspaceQueue**,
with a dedicated producer/consumer flag-ID pair and pipe selection by stage —
and **no inline cube-internal `event_t(0)`**:

```cpp
// Two flag IDs only — a producer/consumer pair, both MODE 0x2 (identical both sides):
//   consumerNotifyProducerId_  : consumer -> producer  "slot free"
//   producerNotifyConsumerId_  : producer -> consumer  "slot ready"

// consumer side: tell producer the slot is free (data already drained via MTE2)
CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);
CrossCoreWaitFlag<0x2>(consumerNotifyProducerId_);

// producer side: cube finished writing L0C->GM via FIX pipe
CrossCoreSetFlag<0x2, PIPE_FIX>(producerNotifyConsumerId_);
// or: vec finished writing GM via MTE3 pipe
CrossCoreSetFlag<0x2, PIPE_MTE3>(producerNotifyConsumerId_);
CrossCoreWaitFlag<0x2>(producerNotifyConsumerId_);
```

Pipe rule: pick the pipe that produced the data being signalled — `PIPE_FIX`
after a cube Fixpipe write, `PIPE_MTE3` after a vec GM write, `PIPE_MTE2` after a
consumer GM read. All cross-core; none use the PB-35 `event_t(0)` cube-internal
form.

## decision_id: `softmax_online` (`wsMetaGm_` 3-region handoff layout)

Our KB had **0** entries for the `wsMetaGm_` alpha/sumexp cross-core handoff
layout (CAND-FA3 gives slot rotation only). cv-agent's concrete layout: a single
flat `GlobalTensor<float> wsMetaGm_` holding **3 contiguous regions**, each
`RING_SLOTS * BLOCK_M` wide:

```
wsMetaGm_ layout (flat float):
  region 0  [ 0                     .. RING_SLOTS*BLOCK_M )  : softmaxMax   (running m_i)
  region 1  [ RING_SLOTS*BLOCK_M    .. 2*RING_SLOTS*BLOCK_M ): softmaxSum   (running l_i)
  region 2  [ 2*RING_SLOTS*BLOCK_M  .. 3*RING_SLOTS*BLOCK_M ): softmaxExp   (alpha = rescale)

  per-row index:  stateBase = slot * BLOCK_M + rowOffset
```

```cpp
// producer (vec stage 1) writes the row's max / sum / exp into the 3 regions:
DataCopy(wsMetaGm_[stateBase],                          softmaxMaxUb_, subBlockRows_);
DataCopy(wsMetaGm_[RING_SLOTS*BLOCK_M     + stateBase], softmaxSumUb_, subBlockRows_);
DataCopy(wsMetaGm_[2*RING_SLOTS*BLOCK_M   + stateBase], softmaxExpUb_, subBlockRows_);

// consumer (vec stage 2 / merge) reads them back by the same region offsets:
DataCopy(inStateUb,     wsMetaGm_[prevStateBase],                          subBlockRows_); // max
DataCopy(inSumUb,       wsMetaGm_[RING_SLOTS*BLOCK_M   + prevStateBase],   subBlockRows_); // sum
DataCopy(softmaxExpUb_, wsMetaGm_[2*RING_SLOTS*BLOCK_M + stateRowBase],    dealRows);      // exp/alpha
```

Why it matters: getting the region stride wrong (e.g. interleaving max/sum/exp
per-row instead of 3 contiguous regions) corrupts the online-softmax merge and/or
reads across slot boundaries — a silent precision break, and with the wrong
slot×BLOCK_M base it overruns into the next slot's L0C-fed region.

### In-UB per-row stat layout + extraction + column broadcast (closes the 0/8 precision blocker)

The §above gives the GM region stride. This sub-section gives the **in-UB** layout +
the extract/broadcast primitive that the row-wise rescale/divide needs — the piece
kw-gb3 (2026-06-03) found missing (the FA graybox built + ran cube-MIX but hit
precision 0/8: `softmax_sum` systematically wrong ~60–490, `attention_out` = nan,
both from a broken stat read/broadcast). Full pattern + reject conditions:
**candidates.md §CAND-FA-SOFTMAX-STAT-1**.

**Layout convention (`[m, 8]`, the "why 8")**: the per-row max/sum that the public
online-softmax (`Softmax` / `SoftmaxFlashV2`) emits is NOT a contiguous `[m]`. It is a
`[m, 8]` datablock-packed buffer — `8 = 32 bytes / sizeof(float)`, one hardware
datablock per row, **all 8 lanes hold the same reduced value** (this is the documented
public contract: the stat output's last axis is fixed to one 32-byte datablock with
identical lanes). Row `i`'s stat is at `buf[i*8 .. i*8+8)`.

**Extraction `[m, 8] → [m]`**: there is NONE — index any lane of the row's block
(`buf[i*8]`). To produce a contiguous `[m]` (e.g. to emit to a `[m]`-shaped GM output)
just lane-0-stride DataCopy. To FEED the broadcast you point at the block base and let
the stride convention spread it (below).

**Broadcast `[m, 8] → [m, cols]` for the divide/rescale (the precision-critical primitive)**:
a single `Div`/`Mul` whose `BinaryRepeatParams` broadcasts the row's datablock across
all columns within one repeat and advances exactly one datablock per row:

```cpp
constexpr int B32 = 8;                       // fp32 datablock lanes
const int colsAlign = AlignUp(cols, B32);    // head-dim, 32B-aligned

// If the stat is in [m] form (reduce-tail == 1), expand to [m,8] FIRST:
//   AscendC::Brcb(statBlk, stat, (m + 7) / B32, {1, 8});   // dstBlkStride=1, dstRepStride=8
//   AscendC::PipeBarrier<PIPE_V>();
// If the API already produced [m,8] (reduce-tail == 8, the default): skip Brcb,
//   use the API output directly as statBlk.

AscendC::BinaryRepeatParams rp;
rp.src0BlkStride = 1;                 // numerator O contiguous in cols
rp.src0RepStride = colsAlign / B32;   // O advances one row per repeat
rp.src1BlkStride = 0;                 // KEY: stat block broadcast across all cols
rp.src1RepStride = 1;                 // KEY: stat advances exactly one datablock per row
rp.dstBlkStride  = 1;
rp.dstRepStride  = colsAlign / B32;
const int chunk = 64;
for (int c = 0; c < colsAlign / chunk; c++) {
    AscendC::Div(o[c * chunk], o[c * chunk], statBlk[/*rowTileBase*/0 * B32],
                 /*mask=*/chunk, /*repeatTimes=*/m, rp);
}
```

Why it matters (the kw-gb3 bug): reading the stat as contiguous `[m]` (stride 1) when
it is `[m,8]` uses every 8th row's value for 8 rows → wrong `softmax_sum`, div-by-corrupt
→ `nan`. Using a `Copy`/`CopyRepeatParams` for the broadcast is also wrong:
`CopyRepeatParams` is the ND↔NZ fractal-transpose primitive, NOT a row-broadcast — the
stat broadcast is `Brcb` (expand, only when `[m]`) + `Div`/`Mul`-with-`BinaryRepeatParams`
(apply), never a `Copy`. And `src1RepStride = 0` (vs 1) silently broadcasts row 0's stat
to all rows (no nan, but wrong). All primitives public-API
(`Softmax`/`SoftmaxFlashV2`, `Brcb`, `Mul`, `Div`, `BinaryRepeatParams`) → runnable.

## decision_id: `block_tile_alignment` (cube tile MUST be 16/c0-aligned — never raw seq)

**The V220 cube requires `m % 16 == 0 && n % c0 == 0`** (c0 = 32/sizeof(dtype) = 16
for fp16/bf16). Mmad / LoadData / Fixpipe operate on Nz fractals that are 16-row ×
c0-col; a tile whose m or n is sub-16 / non-c0 leaves the fractal partially
uninitialized and the cube reads that garbage into the result.

**Bug it prevents (FA (B) 2026-05-27, systematic-high O):** the emit set the cube
tile to the RAW sequence — `block_M = seq_q; block_N = seq_kv` (no tiling, no
alignment). For a tiny case (Sq=2, Skv=2) the cube ran `Mmad(m=2, n=2, k=dim)` on
sub-16 tiles → garbage S → stored `softmaxSum = 19.72` (mathematically impossible
for a 2-element max-subtracted softmax, which must be ≤ 2) → output systematically
higher than reference (3.17–3.44 vs 2.67–3.10), 0/9 precision FAIL. (This refuted
the earlier "missing normalization" and "LoadData2D-vs-3D" hypotheses — softmax
logic was correct and both kernels use the same load helpers; the root is the
raw-sub-16 tile.)

**cv-agent's verified approach (the canonical lowering — DO THIS, do not improvise):**
- Cube operates on **FIXED `BLOCK_M` / `BLOCK_N`** constants (16-aligned, e.g. 64/128), NEVER the raw seq.
- Align every cube dim: `mActAlign = AlignUp(BLOCK_M, C0)`, `kvRowsAlign = AlignUp(kvRows, C0)`, `dimAlign = AlignUp(dim, C0)`.
- Loop the K dim in `BASE_K` tiles (`kTiles = dim / BASE_K` + a `dim % BASE_K` remainder tile).
- **Tail handling (the load overload (B) dropped):** last KV tile uses
  `kvRows = (rowStart + BLOCK_N > kvSeqLen) ? tailValid : BLOCK_N`; load only
  `kvRows` rows via the explicit-stride overload
  `LoadNdGmToNzL1(dst, src, kvRows, dim, dim, /*dstNzC0Stride=*/kvRowsAlign)`;
  the vec side masks the unused `BLOCK_N - kvRows` columns with `-inf` so softmax ignores them.
- **Host (pybind) pads qSeqLen:** `qSeqLenAlign = AlignUpHost(qSeqLen, BLOCK_M)`;
  kernel loops `qSeqLenAlign / BLOCK_M` row-blocks; output is allocated + written at
  the REAL `qSeqLen` (padding rows are computed from OOB/garbage Q but discarded — never written).
- `MmadParams`: `mp.m = BLOCK_M`, `mp.n = kvRowsAlign`, `mp.k = BASE_K`; `Fixpipe`: `srcStride = mActAlign`, `nSize = kvRowsAlign`.

(Learning provenance: cv-agent `flash_attention_cube.h` `ComputeMM1` + `ComputeMM2`
+ pybind `qSeqLenAlign = AlignUpHost(qSeqLen, BLOCK_M)`. The forms above ARE the
verified grounding — generate from them, do not read the external kernel at runtime.)

**Translator rule:** emit the cube on FIXED 16-aligned `BLOCK_M`/`BLOCK_N` + align
all dims + K-tile loop + tail-mask + host pad-qSeqLen. **NEVER set
`block_M = seq_q` / `block_N = seq_kv` (raw).** A raw-seq cube tile is a lowering bug.

**Detection:** (a) pybind/tiling assigns `block_M = seq_q` or `block_N = seq_kv`
directly (no `AlignUp`) → BUG; (b) runtime: any stored `softmaxSum > blockN`
(a max-subtracted softmax sum must be ≤ the row width) → garbage S from a sub-16 cube.

## decision_id: `qk_load_form` (Q and K L1→L0 loads MUST both be `ifTranspose=false`; K^T comes from NZ fractal + Mmad k-contraction along D=C0 axis)

> **Scope note**: `qk_load_form` is a **lowering sub-constraint under
> `matmul_primitive`** (generator decision-manifest key 3 in the FA template
> contract), not a 9th manifest key. The manifest entry for
> `matmul_primitive` already records the high-level choice (hand-written `Mmad`);
> this section adds the **L0A/L0B load form** that the `Mmad` direct path must
> use to be precision-correct on V220 for FA-class QK^T. Same precedent as
> `block_tile_alignment` above (a constraint under `block_partition`).

**The cube QK^T contraction axis is D (= C0 for fp16/bf16).** Once K is staged into
L1 as NZ with D innermost (`LoadNdGmToNzL1(kL1, kGm, nAlign, dim, ldK,
/*dstNzC0Stride=*/nAlign)`), the K^T-of-QK^T emerges **automatically** from the
Mmad's k-contraction sweeping the C0 axis — no `ifTranspose=true` B-load is needed.
Both A (Q) and B (K) L1→L0 loads MUST use `LoadData2DParams{ifTranspose=false}`.
Mixing — A with one load primitive (e.g. `LoadNzL1ToZzL0A` 3DParamsV2) + B with
`ifTranspose=true` via `LoadData2DParams` / `LoadNzL1ToZnL0B_Trans` — desyncs the
contraction axes per **P-P99** (A and B MUST source contraction from the same axis).

**Bug it prevents (FA (C) 2026-05-27 — same op / same shape / same archive as (B),
distinct bug):** with K staged correctly (no host-transpose; D innermost;
`dstNzC0Stride=nAlign`) and the cube tile 16-aligned, the **K-side L1→L0 load via
`LoadNzL1ToZnL0B_Trans(ifTranspose=true)` still produces wrong output**: `attn_out`
absmax matches CPU-truth (`kern_absmax 1.16-1.31` vs `ref_absmax 1.17-1.31`) but
`max_diff ~1.3-1.6` — element-wise layout-permute, NOT magnitude / axis error.
Signature is the giveaway: **TRACK-magnitude + element-wise mismatch ⇒ load-form
fractal-permute, NOT axis bug.** The transposed-B helper variant (fp16 16×16
`LoadData2DParams{ifTranspose=true}` ≈ `LoadDataWithTranspose`) permutes Mmad-input
fractals in a way mathematically equivalent to K^T ONLY when the Q-side A-load was
also a transposed-fractal load (symmetric pairing). When A is a plain 3DParamsV2
Zz-load (the asymmetric pairing the prior emit used per CAND-FA1 line 1127), the
B-side transpose mis-aligns and the output is permuted within the
`[BLOCK_M × BLOCK_N]` tile.

**cv-agent's verified approach (the canonical lowering — DO THIS, do not improvise):**

```cpp
// --- A (Q) L1->L0A : per C0 row-block, ifTranspose=false (cv-agent ComputeMM1) ---
for (uint32_t i = 0; i < mActAlign / C0; i++) {
    LoadData2DParams p;
    p.startIndex  = i;
    p.repeatTimes = BASE_K / C0;            // (or kAligned / C0 in the tail block)
    p.srcStride   = mActAlign / C0;
    p.dstGap      = 0;
    p.ifTranspose = false;
    LoadData(aL0[C0 * i * BASE_K], qL1[ki * BLOCK_M * BASE_K], p);
}

// --- B (K) L1->L0B : whole [BASE_K x kvRowsAlign] fractal, ifTranspose=false ---
{
    LoadData2DParams p;
    p.startIndex  = 0;
    p.repeatTimes = (BASE_K / C0) * (kvRowsAlign / C0);
    p.srcStride   = 1;
    p.dstGap      = 0;
    p.ifTranspose = false;
    LoadData(bL0, kvL1[ki * kvRowsAlign * BASE_K], p);
}

// --- Mmad: S[BLOCK_M, kvRowsAlign] = Q . K^T over k = BASE_K ; accumulate across ki ---
MmadParams mp;
mp.m = BLOCK_M; mp.n = kvRowsAlign; mp.k = BASE_K;
mp.cmatrixInitVal = (ki == 0);             // see `matmul_primitive`: init only on first tile
mp.cmatrixSource  = false;
Mmad(cL0, aL0, bL0, cL0, mp);              // 4-arg accumulate form
```

**For D ≤ cv-agent's `BASE_K = 128` (typical FA shapes, D ∈ {64, 128}):**
`kTiles = dim / BASE_K ∈ {0, 1}` — the main loop runs at most once, the **tail block**
(`kRemain = dim - kTiles*BASE_K`, `kAligned = AlignUp(kRemain, C0)`) handles the
remaining width in one Mmad with `mp.k = kAligned`, `mp.cmatrixInitVal = (kTiles == 0)`.
So for D ∈ {64, 128} the entire QK^T is one Mmad pass — no inner k-loop. Don't
optimize this away into a single-shot non-tile form: the canonical block layout
(Q L1 NZ with `dstNzC0Stride=mAlign`, K L1 NZ with `dstNzC0Stride=kvRowsAlign`) is
what makes the no-ifTranspose loads work; collapsing the tile structure
re-introduces axis-pairing fragility.

(Learning provenance: cv-agent `flash_attention_cube.h::ComputeMM1` L91-188
main+tail blocks. Validated 2026-05-27/28 independent prototype lane: ported into
`output/npukernelbench-a3/src/kernels/3_FusionAttention/kernel/flash_attention_cube.h::ComputeQK`
(MR #222 / origin commit `f2f78535`); DS canonical device-2 6/6 PASS within-scope
fp16 `max_diff < 5e-4` (< MERE strict 9.77e-4), bf16 within mantissa floor.
Replaces the prior `LoadNzL1ToZnL0B_Trans(ifTranspose=true)` form recorded in
**CAND-FA1** line 1127 — that variant compiles clean on V220 and accepts magnitude
but is now SUPERSEDED by this decision; CAND-FA1 has been flagged accordingly.)

**Translator rule:** for any FA-class cube QK^T emit, BOTH A (Q) and B (K) L1→L0
loads MUST use raw `LoadData2DParams{ifTranspose=false}` (per-i row-block loop for
Q + single fractal sweep for K, exactly as above). NEVER emit
`LoadNzL1ToZnL0B_Trans(ifTranspose=true)`, and NEVER pair `LoadNzL1ToZzL0A` (A,
3DParamsV2) with `LoadNzL1ToZnL0B_Trans` (B, ifTranspose=true) — that's the
asymmetric mixing that produces the layout-permute bug above. Mmad uses the 4-arg
accumulate form with `cmatrixInitVal=(ki==0)` per the `matmul_primitive` decision.

**Detection:** (a) **emit grep** — any occurrence of `LoadNzL1ToZnL0B_Trans` OR
`ifTranspose\s*=\s*true` inside an FA-class ComputeQK / matmul1 region → BUG;
(b) **runtime signature** — kernel `attn_out` `abs_max` matches reference within
~1% but element-wise `max_diff > 0.1 × ref_abs_max` → load-form fractal-permute
(this decision), NOT axis bug, NOT sub-16 tile bug. Check the L0A/L0B load
primitives before chasing axes.

## decision_id: `pv_load_form` (MM2 P@V V-operand MUST be `ifTranspose=TRUE` — the OPPOSITE of MM1's K; do NOT over-apply the `qk_load_form` "never true" rule to MM2)

> **Scope note**: `pv_load_form` is the MM2 (P@V, the SECOND matmul) sibling of
> `qk_load_form` (MM1, Q@K^T), both lowering sub-constraints under
> `matmul_primitive`. The two MMs have OPPOSITE B-operand transpose forms — this
> is the single most counterintuitive FA-class lowering fact, and the one the
> 2026-05-30 archive-blind repro probe could NOT re-derive (it correctly found V
> needs *some* transpose — H1 transposed-V cut `attention_out` error 12× from 0.94
> to 0.04–0.11 — but never landed the exact 2D-V2 form, and its H2 "per-column
> structured transpose" made it catastrophically worse: 9e2–1e34. = a real KB gap,
> now codified from the white-box 5/5 archive `flash_attention_cube.h::ComputeMM2`).

**The crux: MM1 K-operand is `ifTranspose=false`; MM2 V-operand is `ifTranspose=TRUE`.**
Why they differ: MM1 (Q@K^T) contracts over D (= C0), so K^T emerges from the NZ
fractal + Mmad k-sweep with NO transpose (see `qk_load_form`). MM2 (P@V) contracts
over the kv-seq axis, so the V-operand fractals MUST be transposed at L1→L0B
(`MM2 V isRightTranspose=0` ⇒ the load's `ifTranspose=true`). Same op-class, opposite
B-load — emitting MM2's V with `ifTranspose=false` (the natural mistake if you copy
MM1's K-form) leaves `attention_out` element-wise wrong while `abs_max` ≈ matches.

**cv-agent / white-box verified MM2 load forms (DO THIS — `flash_attention_cube.h::ComputeMM2`, 5/5 on A5 Ascend950PR):**

```cpp
// --- A (P) L1->L0A : ifTranspose=false (same class as MM1's Q-operand) ---
LoadData2DParamsV2 pp;
pp.mStartPosition = 0; pp.kStartPosition = 0;
pp.ifTranspose = false;
pp.mStep = (BLOCK_M + 15) >> 4;
pp.kStep = (kvRowsAlign + 15) >> 4;
pp.srcStride = pp.mStep;  pp.dstStride = pp.mStep;
LoadData(aL0, pL1, pp);

// --- B (V) L1->L0B : ifTranspose=TRUE  <<< the counterintuitive bit, OPPOSITE of MM1 K ---
LoadData2DParamsV2 pv;
pv.mStartPosition = 0; pv.kStartPosition = 0;
pv.ifTranspose = true;                       // MM2 V isRightTranspose=0 -> load transpose=true
pv.mStep = (kvRowsAlign + 15) >> 4;
pv.kStep = (BASE_K + 15) >> 4;
pv.srcStride = (kvRowsAlign + 15) >> 4;
pv.dstStride = (BASE_K + 15) >> 4;
LoadData(bL0, kvL1, pv);
// Mmad: O[BLOCK_M, BASE_K] = P . V over k = kvRowsAlign ; Fixpipe srcStride = AlignUp(BLOCK_M,C0)
```

**Falsified alternatives (do NOT emit):** (a) per-fractal `startIndex=i` i-loop for V
(A5-invalid — band OOB, same as the MM1 K trap); (b) "per-column structured transpose"
that swaps inter-fractal block positions (repro-probe H2, 2026-05-30 → 9e2–1e34, the
`startIndex`/`srcStride` fractal-unit addressing is wrong). Use the single
`LoadData2DParamsV2{ifTranspose=true}` whole-fractal load above.

**Translator rule:** the `qk_load_form` detection rule ("`ifTranspose=true` → BUG") is
scoped to the **MM1 / ComputeQK / matmul1** region ONLY. In the **MM2 / ComputeMM2 /
P@V** region, the V-operand `ifTranspose=true` is REQUIRED, not a bug — do not let the
MM1 grep-rule suppress it.

**Detection:** `attention_out` element-wise `max_diff` stuck ~0.04–0.5 (TRACK-magnitude,
`abs_max` ≈ ok) AFTER MM1/softmax/softmax_out are correct ⇒ MM2 V-load form (this
decision). Confirm the MM2 V-operand load is `ifTranspose=true` 2D-V2 (not false, not
per-column).

## decision_id: `kernel_block_iteration` (MIX_AIC_1_2 sub-block normalization — `coreIdx` MUST divide `GetBlockIdx()` by `GetSubBlockNum()`)

> **Scope note**: `kernel_block_iteration` is a **lowering sub-constraint under
> `block_partition`** (generator decision-manifest key 1 in the FA template contract), not a
> 10th manifest key. The manifest entry for `block_partition` records the
> per-block work assignment shape (per-head / per-row-tile / etc.); this section
> adds the **AIC/AIV index-normalization formulas** the kernel dispatcher MUST
> emit in `KERNEL_TYPE_MIX_AIC_1_2` mode so the cube and 2 paired vec sub-blocks
> don't collide on the same per-core workspace partition. Same precedent as
> `block_tile_alignment` (constraint under `block_partition`) and `qk_load_form`
> (constraint under `matmul_primitive`).

**In `KERNEL_TYPE_MIX_AIC_1_2`, `GetBlockIdx()` returns the *linear* (cube + 2×vec
sub-blocks) index — NOT the per-core index.** A single physical core hosts 1 AIC
+ 2 paired AIV sub-blocks, so the linear `GetBlockIdx()` range is `[0,
GetBlockNum()*GetSubBlockNum())` where `GetSubBlockNum() == 2` on V220 in this
mode. The per-core index for workspace partition + scheduler driving MUST be
the linear index DIVIDED BY `GetSubBlockNum()`. Failing to divide makes 3 cores
read/write the same workspace slot (cube + AIV0 + AIV1 all think they're
"core 0"), producing per-core workspace partition collisions → AIC↔AIV sync
chain deadlock → `aicore timeout 507014` runtime hang.

**Bug it prevents (FA (D) 2026-05-28 — autonomous translator emit hung at
runtime; same archive, distinct bug from FA(B) host-transpose and FA(C)
ifTranspose-form):** with the cube load form correct (per `qk_load_form` above),
the kernel still hangs because `flash_attention_kernel.h::Init()` emitted
`int coreIdx = AscendC::GetBlockIdx()` — raw linear index, no `/
GetSubBlockNum()` normalization. Cube on physical core P uses
`workspace + P * perCore`, AIV0 on the same physical core P also computes
`coreIdx = GetBlockIdx() = 2P` (or `2P+1`) and uses `workspace + (2P) * perCore`
— a slot that belongs to physical core `2P/GetSubBlockNum() = P`'s neighbor. The
WorkspaceQueue producer/consumer flags on those slots get crossed → AIC waits on
flags AIV never sets → 90s hang → `aicore timeout 507014` (10/10 cases in 2026-05-28
precision-probe pp-2). Signature is the giveaway: **build PASS + sync-chain hang
at first kv-tile boundary + probe attributes PB-35 — but MR #222 hand-fix with the
SAME PB-35 family runs 6/6 PASS** ⇒ root cause is NOT PB-35, it's THIS missing
normalization. (PB-35 preconditions are necessary-but-not-sufficient; satisfy
them AND normalize indices to actually run.)

**cv-agent's verified approach (the canonical lowering — DO THIS, do not improvise):**

```cpp
// --- Init(): derive per-core index by dividing linear block index by sub-block count ---
__aicore__ inline void Init(...) {
    CopyTiling(&tiling_, tilingGM);
    // ... SetGlobalBuffer ...

    int coreIdx = GetBlockIdx() / GetSubBlockNum();   // [F] NORMALIZED per-core index
    sched_.Init(tiling_.totalBlocks, GetBlockNum(), coreIdx);

    // Per-core workspace partition uses the NORMALIZED coreIdx (NOT raw GetBlockIdx()).
    GM_ADDR base = workspace + (uint64_t)coreIdx * perCore;
    qS_.Init(base,           sSlot, FA_QS_P2C, FA_QS_C2P);
    qP_.Init(base + sBytes,  pSlot, FA_QP_P2C, FA_QP_C2P);
    qO_.Init(base + sBytes + pBytes, oSlot, FA_QO_P2C, FA_QO_C2P);

    if (ASCEND_IS_AIC) {
        cube_.Init(tiling_.D, pipe);
        qP_.InitFreeSlots();    // cube is qP consumer
    }
    if (ASCEND_IS_AIV) {
        // [F] subTileM_ splits the M tile across the 2 paired AIV sub-blocks.
        subTileM_ = FA_BLOCK_M / GetSubBlockNum();
        vec_.Init(subTileM_, tiling_.D, tiling_.scale, pipe);
        qS_.InitFreeSlots();    // vec is qS consumer
        qO_.InitFreeSlots();    // vec is qO consumer
    }
}

// --- Process(): AIV branch derives its per-sub-block row offset from GetSubBlockIdx() ---
__aicore__ inline void Process() {
    while (sched_.HasNext()) {
        int bid = sched_.Next();
        // ... derive (b, n, qi) and qBase / kvBase from bid ...

        int rowOffset = 0;
        if (ASCEND_IS_AIV) {
            // [F] GetSubBlockIdx() ∈ {0,1} on V220 MIX_AIC_1_2; rowOffset selects
            // this sub-block's slice of the BLOCK_M tile.
            rowOffset = (int)(GetSubBlockIdx() * subTileM_);
            vec_.BlockBegin();
        }
        // ... kv-tile loop; AIC writes qS/qP/qO slots, AIV at rowOffset reads them ...
    }
}
```

**Three formulas — memorize them as a unit, not three separate facts:**
1. `coreIdx = GetBlockIdx() / GetSubBlockNum()` — per-core index (feeds scheduler + workspace partition)
2. `subTileM_ = FA_BLOCK_M / GetSubBlockNum()` — M-tile split across AIV sub-blocks
3. `rowOffset = GetSubBlockIdx() * subTileM_` — per-sub-block row anchor inside the M tile

All three reference `GetSubBlockNum()` / `GetSubBlockIdx()`; emitting any ONE of
them without the others is incomplete and will still hang (e.g. dividing for
coreIdx but not splitting subTileM_ produces sub-block overlap at the M
boundary).

**Why the raw form looks correct:** `coreIdx = GetBlockIdx()` reads naturally
as "this core's index" and the AscendC kernel programming guide presents
`GetBlockIdx()` without the MIX_AIC_1_2 sub-block context. The translator (LLM)
that hasn't seen the concrete form falls back to the textbook form and emits
the raw read. cv-agent / MR #222 / CANN ops-transformer all do the divide; the
translator KB MUST surface this concrete form, not just "derive coreIdx from
GetBlockIdx and GetSubBlockNum" (which is what `cv_lowering.md` §3.A bullet 3
currently says — too abstract to translate into the exact divide).

(Learning provenance: cv-agent `flash_attention_cube.h::Init` + `Process` block
iteration + AIV row-offset derivation. Validated 2026-05-27/28: ported into
MR #222 `output/npukernelbench-a3/src/kernels/3_FusionAttention/kernel/flash_attention_kernel.h`
(origin commit `f2f78535`); DS canonical device-2 6/6 PASS within-scope. Refuted
on 2026-05-28 Phase 4 autonomous cold-start (orch bg `bfxyurqxh`): translator
emit'd raw `GetBlockIdx()` without divide → `aicore timeout 507014` on first
kv-tile boundary, 10/10 cases. Researcher ar-2 diagnosed the gap by side-by-side
diff vs MR #222; the abstract KB rule in `cv_lowering.md` §3 was insufficient
without the concrete formula here. Codified as `kernel_block_iteration` per the
same "concrete-params-not-abstract-rule" pattern that produced `qk_load_form`.)

**Translator rule:** for any FA-class kernel dispatcher (`flash_attention_kernel.h`
or sibling) running in `KERNEL_TYPE_MIX_AIC_1_2` mode, emit ALL THREE formulas
above in their canonical sites (Init + Process AIV branch). NEVER emit raw
`coreIdx = GetBlockIdx()` for workspace partition. The scheduler `sched_.Init(...,
coreIdx)` MUST receive the normalized `coreIdx`, not the raw `GetBlockIdx()`.

**Detection:**
- (a) **emit grep** — `flash_attention_kernel.h` or any kernel dispatcher with
  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` MUST contain `GetBlockIdx()
  / GetSubBlockNum()` literally. A grep `coreIdx\s*=\s*GetBlockIdx\(\)\s*;` (raw,
  no divide) → BUG. A grep `coreIdx\s*=\s*GetBlockIdx\(\)\s*/\s*GetSubBlockNum\(\)`
  → present, correct.
- (b) **runtime signature** — build PASS + `aicore timeout 507014` at first
  cube/vec sync boundary + 10/10 hang across all input cases (regardless of dtype
  / shape) → this decision's bug. NOT a CANN compiler bug, NOT a scope-exceeding
  shape, NOT PB-35 (PB-35 is a different family; satisfy its 5 preconditions AND
  normalize indices to actually run — MR #222 proves both are required).

<!-- 迁移自 porter kb/target/ascendc/fa_class/cv_reference_concrete_params.md(整档忠实搬运,convert_docs_to_okf.py)。跨 op 参考/方法论知识,非机械家族。 -->

---
applies_to: soc=Ascend910_9382
reason: This pattern (P-P117) teaches an op-gen worker to GENERATE a chunked
  gated-delta-rule / gated-linear-attention FORWARD on a3/arch22 (Ascend910_9382,
  Atlas A3) — the per-head sequential chunk recurrence, ON-DEVICE per-chunk decay-fold
  (cumsum+exp+fold in kernel Stage 0 / AIV; NO host torch compute — the no-delegation
  rule requires all compute in AscendC), MIX cube+vector execution, and whole-device
  SyncAll (MIX_AIC_1_1, PB-57) are device-proven on Ascend910_9382 (dav-2201, arch22),
  never on a5. Do NOT hand this to an a5/Ascend950PR worker (unverified_on: Ascend950PR).
---

# Chunked Gated-Delta-Rule / Gated-Linear-Attention Forward — a3 MIX multi-chunk recurrence (P-P117 full body)

> Domain-pattern body for **P-P117** (indexed in `patterns/PATTERN_INDEX.md` and `KB_INDEX.md`).
> Chip: **a3 / arch22 / Ascend910_9382**, CANN 9.1.0. Device-verified 16/16 @ the customer
> fp64 gate (rtol=0.02, ≥17× headroom, deterministic, NaN-free), 9–28× device-time vs
> torch_npu (DS gen3, 2026-07-20). Archive: `output/fqa_gated_delta_rule/`.
> This is the ALGORITHM pattern; it sits on top of the a3 MIX cube+vector SYNC template
> **P-P116** (`fa_class_a3_mix_template.md`). Read both.

## 1. Scope

A chunked **gated-delta-rule** (a.k.a. gated-linear-attention / GDN) forward pass on a3:
arbitrary sequence length `T`, grouped-query attention (GQA), multi-batch. Reference is the
Qwen-team fp64 chunked GDR (`ref_gdr.py::chunk_gated_delta_rule_fwd`). The kernel is genuine
MIX cube+vector — every matmul runs on AIC via the library, elementwise on AIV. This is NOT a
flash/online-softmax attention; there is no softmax. It is a linear-attention recurrence over
chunks of the sequence.

## 2. The decomposition — every kernel matmul is a plain `A@B`

Per head, loop chunks `c = 0 .. ceil(T/64)-1` **SEQUENTIALLY** with a persistent state
`S[128,128]` in GM (init 0). The decomposition keeps **every kernel matmul a plain `A@B`** by
pre-folding the per-chunk decay bookkeeping into the matmul operands. **That fold is computed
ON-DEVICE, in the AscendC kernel** (an AIV pre-stage, "Stage 0") — NOT on the host.

> ⚠️ **Compute the decay ON-DEVICE. Do NOT fold on the host with torch.** The no-delegation rule
> (CLAUDE.md: *ALL computation must use AscendC primitives*) forbids computing the gate cumsum /
> exp / operand fold with host `torch` ops (`gc.cumsum()`, `eg.exp()`, `k*eg`, …). Those are real
> transcendental + elementwise compute and belong in the kernel. The sibling
> `recurrent_gated_delta_rule` op proves feasibility (it does its whole gate decay on-device with
> 0 delegation via the `Exp()` vector primitive). An earlier version of THIS archive folded on the
> host and was caught by `scan_delegation_cheating.py` with 4 violations (torch `cumsum`+`exp`).
> The host may do ONLY layout marshaling: GQA head-expand, zero-pad, chunk-reshape, dtype cast,
> and permute to head-major (incl. providing a raw transposed `kT` — a permute, not compute).

Per-chunk operand fold (`gc = cumsum(g)`, `eg = exp(gc)`); the kernel produces these in scratch:

```
kb   = k * beta * eg
knT  = (k / eg)^T = (k * exp(-gc))^T
qs   = scale * q * eg
vb   = v * beta
kdT  = (k * exp(gc_last - gc))^T
sc   = exp(gc_last) = eg[C-1]        # per-head scalar decay applied to state (no extra exp)
```

**On-device Stage 0 (AIV), per (head, chunk):**
- `gc = cumsum(g)` over the chunk — an in-chunk **scalar prefix-sum** (`C = 64`; `gc[i]+=gc[i-1]`),
  matching the reference `x.cumsum(dim=2)`. (No AscendC `Cumsum` API on this CANN; the 64-wide
  scalar scan is trivial vs the matmuls.)
- `eg = Exp(gc)`, `egInv = Exp(-gc)`, `kd = Exp(gc_last - gc)` via the `Exp` vector primitive;
  `sc = eg[C-1]` (reuse — `exp(gc_last)` is already the last `eg`).
- **Row-broadcast fold** for `kb`/`qs`/`vb` (`[C,D]`, scale per chunk-position row): `Broadcast<T,2,1>`
  the `[C]` row-scale to `[C,D]` then `Mul`. Compute in fp32 UB (`Cast` half→fp32 in, fp32→half
  `CAST_RINT` out) so precision matches the host fold.
- **Col-broadcast fold** for the transposed `knT`/`kdT` (`[D,C]`, scale per chunk-position column):
  the repeat-`Mul` broadcast idiom — `Mul(dst,src,vec, mask=C, repeat=D, {1,1,1,C/8,C/8,0})` with
  `src1RepStride=0` so the `[C]` vec is reused for every row. (This is the sibling
  `recurrent_gated_delta_rule::MatVecMul` idiom — studied for structure, not lifted.)
- Fold results land in dedicated scratch slots (`S_KB/S_QS/S_VB/S_KNT/S_KDT`); stages 1–8 read them
  exactly as if host-provided. A `SyncAll<false>()` separates Stage 0 (AIV write) from Stage 1
  (AIC read).

**Zero-pad `T → Nc*64`** and take the per-chunk cumsum: the padded tail then carries `gc_last`
automatically (cumsum of the zero-padded gate), and the padded rows/cols vanish under the causal
mask — so there is NO explicit last-chunk fill to special-case.

Within-chunk (all matmuls plain `A@B`, all on AIC):

```
Acc = kb @ knT
L   = -strict(Acc)                  # strict = strictly-lower triangle; SIGN matters, see CAND-GDR-1
T   = (I + strict(Acc))^-1          # Neumann power product prod(I + L^{2^k}), k=0..5
Am  = incl(qs @ knT)                # incl = inclusive-lower (with diagonal)
U   = T @ vb
```

Cross-chunk (persistent state `S`):

```
W       = T @ kb
WS      = W @ S
o_inter = qs @ S
vn      = U - WS
o       = Am @ vn + o_inter         # chunk output
S       = S * sc + kdT @ vn         # state update for next chunk
```

`Nc == 1` (single chunk, `S = 0`) collapses cleanly to the single-chunk case. GQA is handled by a
host `repeat_interleave` head-expand before the per-head loop.

## 3. MIX execution + sync (a3-specific — the load-bearing part)

- **Matmuls on AIC** via `MatmulImpl<...>` + `IterateAll<sync=true>` — the arch22-safe NON-KFC
  cube (P-P68). NEVER the async `matmul::Matmul` + KfcServer path (PB-34 / a3-standalone-blocked).
- **One reused `MatmulImpl` object across DIFFERENT shapes** (K=128 KKT / K=64 Neumann chain /
  N=128 for U,W,O): call `mm.SetOrgShape(M,N,K)` **per matmul** and `Init` the object at the MAX
  shape (M=N=K=128). `SetSingleShape` alone leaves stale org dims → K-mismatch garbage
  (**CAND-GDR-3**).
- **Elementwise on AIV.** Reusing UB `LocalTensor`s across consecutive AIV ops needs a
  `PipeBarrier<PIPE_ALL>()` at the helper entry — the prior op's V-read races the next op's MTE2
  overwrite otherwise (**CAND-GDR-2**, an INTRA-core WAR; cross-core sync does NOT cover it).
- **Cross-core barrier = whole-device `SyncAll<false>()`.** On a3 this REQUIRES pinning
  `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` (1:1). With the no-macro 1:2 default, idle
  AIV cores skip the per-work-item loop and never issue their `SyncAll` → 507014 deadlock
  (**PB-57**). This is the OPPOSITE macro choice from the P-P116 scoped-CrossCore MODE2 path
  (which keeps the 1:2 default and emits NO macro). Never mix "no macro (1:2)" with whole-device
  `SyncAll`.
- **⚠️ DETERMINISM (generated scoped-1:2 MIX, PB-57(b')).** The scoped per-id
  `CrossCoreSetFlag`/`WaitFlag` MODE2 path can be deadlock-free yet run-to-run nondeterministic at
  high parallelism because the FFTS flag pool aliases above roughly 12 concurrent MIX blocks.
  Select the balanced `MIX_AIC_1_1` template for deterministic generated GDR kernels. Keep the
  per-id flags and WAR barriers unchanged; pairing balance alone removed the nondeterminism in
  12/12 canonical runs, with all 16 cases deterministic and no whole-device `SyncAll` required.

## 4. GM slot layout

`S_NUM = 32` GM scratch slots, including a 2-slot persistent `S` (double-buffered across the
chunk loop) and a 2-slot `Sdelta`. The remaining slots hold the **on-device Stage-0 folded operands**
(`S_KB`/`S_QS`/`S_VB`/`S_KNT`/`S_KDT` — 5 slots; written by Stage 0, read by stages 1–8), the
within-chunk intermediates (`Acc`/`L`/`T`/`Am`/`U`), and the cross-chunk temporaries
(`W`/`WS`/`o_inter`/`vn`). Persistent `S` survives across chunk iterations; everything else is
recycled per chunk. (The earlier host-fold variant had `S_NUM = 27`; moving the fold on-device
added the 5 fold slots.)

## 5. The 4 codegen bugs found + fixed (cross-refs)

All four were device-pinned during gen3 (each by a device run or a CPU-vs-device intermediate diff):

1. **CAND-GDR-1** — Neumann solve SIGN: the kernel's `(I-L)^-1` must equal the reference's
   `(I+L)^-1`; negate the host strict mask. (Corrects an earlier WRONG "fp16-overflow → fp32
   intermediates" thesis — CPU fp16-sim passes; it was the sign.)
2. **CAND-GDR-2** — intra-AIV UB-buffer WAR fence (`PipeBarrier<PIPE_ALL>`); cross-core sync does
   NOT fix it. (Corrects an earlier WRONG "scoped CrossCore is the race fix" thesis.)
3. **CAND-GDR-3** — reused `MatmulImpl` needs `SetOrgShape(M,N,K)` per call + Init at max shape.
4. **CAND-GDR-4** — precision-bar threshold conflation: score against the customer fp64 gate
   (16/16, ≥17× headroom), not `model.py`'s strict-fp32 9.766e-4 bar (14/16, under-rates via a
   forced `.to(float32)`).

**Review-stage methodology fix (delegation)** — the first archive computed the gate cumsum + exp +
operand fold on the HOST with torch (`gc.cumsum()`, `eg.exp()`), caught by
`scan_delegation_cheating.py` (4 violations). Fixed by moving the whole fold ON-DEVICE into the
Stage-0 AIV pre-stage (§2). Re-verified 16/16 @ fp64 (max_err byte-identical to the host-fold
version → numerically faithful) + scanner 0. **Lesson: fold the decay in the kernel from the
start; do not reach for host torch cumsum/exp — that is delegation, not a shortcut.**

## 6. Capability note (why this is worth codifying)

The a5ops backend emitted these 4 codegen-correctness bugs on this op that a different backend did
not — same pipeline, different backend ⇒ a codegen-DISCIPLINE short-board, NOT missing knowledge.
Git history shows the relevant sync KB (P-P116 + PB-55, `cross_core_sync.md`) was in place BEFORE
generation and never deleted — NOT a KB regression. The load-bearing clarity gaps that DID
contribute (fa_class §(c) coupling unspelt; `MIX_AIC_1_1` buried) are addressed by the PB-57 entry
and the fa_class_a3_mix_template.md §(c) / cross_core_sync.md §6 amendments shipped with this
pattern.

## 7. Cross-references

- **P-P116** (`fa_class_a3_mix_template.md`) — the a3 MIX cube+vector SYNC template this sits on.
- **P-P68** — single-AIC GEMM with static tiling (the NON-KFC cube route).
- **PB-57** — whole-device `SyncAll` MIX needs `KERNEL_TYPE_MIX_AIC_1_1` (1:1).
- **CAND-GDR-1/2/3/4** — the 4 device-pinned bugs above.
- **`fa_class/cross_core_sync.md` §6** — cross-core sync does NOT cover an intra-core UB WAR.
- **P-P119** (`a3_mix_small_matmul_cube.md`) — the CUBE-EMIT decision for §2's many-small
  matmuls: emit a hand-rolled single-block cube primitive (`Nd2Nz→LoadData3D→Mmad→Fixpipe`),
  NOT `MatmulImpl::IterateAll` (per-call tiling scalar setup dominates → aic_scalar-bound;
  closed a ~1.3× deficit to cv-ref parity, precision-invariant).
- **P-P120** (`a3_mix_small_matmul_cube.md`) — the Neumann `T=(I+strict)^-1` solve: fold
  `R@(I+P)=R+R@P` into ONE L0C pass via accumulate-mode Mmad (`cmatrixInitVal=1` seeds the
  additive term, `cmatrixInitVal=0` accumulates the product) — no separate +I vector-add.
- **`recurrent_gated_delta_rule`** (sibling op) — on-device gate decay via the `Exp()` vector
  primitive + `MatVecMul` repeat-broadcast `Mul` idiom; the structural reference for the on-device
  Stage-0 fold (§2). 0 delegation.

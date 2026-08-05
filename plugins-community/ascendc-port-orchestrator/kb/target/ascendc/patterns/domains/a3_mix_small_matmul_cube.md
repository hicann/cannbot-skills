---
applies_to: soc=Ascend910_9382
reason: These two patterns (P-P119, P-P120) teach an op-gen worker generating an
  a3/arch22 (Ascend910_9382, Atlas A3) MIX chunked-recurrence op (gated-linear-attention
  / gated-delta-rule family — see P-P117) to reach cv-reference leanness FIRST time on
  the two cube decisions the pipeline previously under-specified: (1) which cube PRIMITIVE
  to emit when a chunk performs MANY small (≤128) matmuls (hand-rolled single-block cube,
  NOT MatmulImpl::IterateAll — msprof-measured aic_scalar-bound), and (2) how to fold an
  (I+P) Neumann / triangular-solve chain onto the cube via accumulate-mode Mmad instead of
  a separate +I vector-add. Both device-proven on Ascend910_9382 (arch22) during the
  gated_delta_rule fwd cube-solve perf-opt (DS 2026-07-20); unverified on Ascend950PR.
---

# a3 MIX many-small-matmul cube primitives — P-P119 (hand-rolled cube vs IterateAll) + P-P120 (Neumann (I+P) L0C-fold)

> Domain-pattern body for **P-P119** and **P-P120** (both indexed in
> `patterns/PATTERN_INDEX.md` and `KB_INDEX.md`).
> Chip: **a3 / arch22 / Ascend910_9382**, CANN 9.1.0. These are the two cube-side
> decisions that separated our first pipeline-generated `gated_delta_rule` kernel from
> the leaner customer cv-reference (both correct; cv-ref leaner because the pipeline
> lacked exactly these two facts). Found by white-box comparison, then device-measured
> during the fwd cube-solve perf-opt (DS 2026-07-20, archive `output/fqa_gated_delta_rule/`).
> These sit ON TOP of the a3 MIX chunked-recurrence algorithm pattern **P-P117**
> (`gated_delta_rule_a3_recurrence.md`) and the a3 MIX cube+vector SYNC template
> **P-P116** (`fa_class_a3_mix_template.md`). The cube-primitive param values they emit
> are grounded in `fa_class/cv_reference_concrete_params.md` (`matmul_primitive` decision).

---

## P-P119 — many-small-matmul a3 MIX: emit a hand-rolled single-block cube primitive, NOT `MatmulImpl::IterateAll`

### Trigger / applies_to
a3 (Ascend910_9382, arch22) MIX op in the **chunked-recurrence / gated-linear-attention**
family (P-P117) where EACH chunk performs **MANY small matmuls** — dimensions ≤128, so
each matmul is a single L0C tile (`M,N,K ≤ 128`, one Mmad pass, no inner K-loop). A
gated_delta_rule chunk runs ~8+ such matmuls (`Acc=kb@knT`, the 6-step Neumann product,
`Am`, `U`, `W`, `WS`, `o_inter`, `S`-update) per (head, chunk), times `Nc` chunks times
heads → hundreds-to-thousands of tiny matmuls per op.

### The finding (why the library primitive is the wrong primitive here)
`MatmulImpl<...>::IterateAll<sync=true>` (the P-P68 NON-KFC cube) is the correct, safe
choice for a **few large** GEMMs. But for MANY SMALL matmuls its **per-call tiling scalar
setup DOMINATES runtime** — every `IterateAll` re-runs the library's tiling/address
bookkeeping (and on a reused object you additionally pay `SetOrgShape` per call, CAND-GDR-3).
That scalar work is fixed-cost per call and does not shrink with the tiny `≤128³` MAC work,
so at high call-count the kernel becomes **aic_scalar-bound, not MAC-bound**.

**msprof evidence (a3/Ascend910_9382, gated_delta_rule fwd, IterateAll version):**
`aic_scalar` **≈32%** / `aic_mac` **≈3.7%** / `cube_util` **≈17%**. The cube is ~9× more
time in scalar tiling setup than in actual MACs — a textbook per-call-overhead-bound
signature.

### The fix — emit a hand-rolled single-block cube primitive per matmul
Replace each `IterateAll` call with a direct, minimal cube pipeline for one L0C tile:

```
Nd2Nz  (GM/UB → L1, ND → NZ fractal layout)
  → LoadData / LoadData3D  (L1 → L0A / L0B, no ifTranspose games; see below)
  → Mmad   (single pass, M,N,K ≤ 128, one L0C tile)
  → Fixpipe[F322F16]   (L0C fp32 accumulator → fp16 GM/UB, with the fp32→fp16 cast)
```

There is NO tiling scalar setup per call — the block dims are compile-time-known
(`≤128³`), the load/Mmad params are constants, so the aic_scalar cost per matmul collapses
to just the fixed load+Mmad+Fixpipe issue. This is the SAME primitive family already
grounded for FA-class cube QK^T in `fa_class/cv_reference_concrete_params.md`
(`matmul_primitive`: `Mmad` 4-arg accumulate form + `Fixpipe` per-context `srcStride`
units); reuse those exact param rules — DO NOT re-derive `srcStride` (per-context: `/C0`
for intermediate L0C→workspace copies, ELEMENTS for final L0C→GM writes; a wrong unit →
507015 ECC read).

### Measured result (device, a3/Ascend910_9382, DS 2026-07-20)
This change ALONE closed a **~1.3× compute-bound deficit** vs the cv-reference to **parity**:
**T4096 −35%**, **T1024 −30%** device-time, **precision unchanged** (still 16/16 @ the
customer fp64 gate). No algorithm change — same matmul math, leaner cube emit.

### Decision rule (put in the worker's hands)
- **Few large GEMMs (one/few per op, M·N·K ≫ 128³)** → `MatmulImpl::IterateAll` (P-P68). The
  library tiling amortizes; hand-rolling buys nothing.
- **Many small matmuls (≤128 each, hundreds+ per op — chunked-recurrence)** → hand-rolled
  single-block cube primitive. Confirm with msprof: `aic_scalar ≫ aic_mac` (e.g. 32% vs
  3.7%) IS the "per-call tiling setup dominates" signature that says switch.
- **⚠ CORRECTNESS gate on the standalone op-gen build — hand-rolled is REQUIRED from the
  FIRST pass, not a perf swap-in.** On the standalone `build_ascendc.py` path (no CANN
  op-framework), `MatmulImpl<> + IterateAll<sync=true>` under `MIX_AIC_1_1` needs the
  framework KFC bootstrap (`REGIST_MATMUL_OBJ` + msg-ring) the standalone build does NOT
  provide → the kernel LAUNCHES but throws a **507057 MIX-kernel task exception at
  execution (no output tensor)**. The hand-rolled single-block primitive has NO KFC
  dependency (raw `Nd2Nz→LoadData3D→Mmad→Fixpipe`), so it runs standalone. **Therefore do
  NOT use the "start with `MatmulImpl` for a correctness pass, swap to hand-rolled for perf"
  strategy on the standalone build — that pass 507057s and never reaches correctness. Emit
  the hand-rolled primitive from the start.** Device-proven: gated_delta_rule fwd hand-rolled
  = **16/16 @ fp64** (DS #206, standalone build) vs `MatmulImpl` standalone = **0/16 / 507057**
  (capstone cold-start re-gen 2026-07-21). Cross-ref OL-235 (KFC standalone wall) / PB-35 /
  CAND-KFC-standalone-bootstrap-teardown.
- Precision is invariant across the two primitives (same Mmad math). The primitive choice is
  a PERF lever ONLY where BOTH are runnable (few-large, or a framework-integrated build); on
  the standalone many-small a3-MIX path it is a **CORRECTNESS gate** (MatmulImpl unrunnable →
  hand-rolled mandatory).

---

## P-P120 — Neumann / (I+P) triangular-solve: fold identity+product on the cube via accumulate-mode Mmad (`cmatrixInitVal`), no separate +I vector-add

### Trigger / applies_to
a3 (Ascend910_9382, arch22) cube computing an **iterative (I+P)-style matmul chain** —
canonically the **Neumann power-product triangular solve** used to invert
`(I + strict-lower)` in chunked gated-delta-rule / gated-linear-attention:
`A = (I + x)^-1 = Π_i (I + x^{2^i})`, i.e. a chain of steps each of the form
`R_next = R @ (I + P)` where `P = x^{2^i}`. Any `R@(I+P)` / `R + R@P` fold qualifies.

### The finding (the naive emit inserts a redundant vector +I)
The naive lowering of one step is: (1) cube `R@P`, (2) vector `Add` the identity/additive
term (`R + R@P`, or `I + P` built with a vector `Duplicate`-diagonal). Each step then costs
a cube Mmad PLUS an AIV add PLUS the barriers separating them. Over a 5–6 step Neumann chain
that is ~9 AIV ops + their MIX cross-core fences — pure overhead, since the cube can carry
the additive term itself.

### The fix — inject the additive term through the L0C accumulator
`Mmad`'s `MmadParams.cmatrixInitVal` selects whether the C-matrix is initialized fresh
(`true` / `=1` on the first tile) or ACCUMULATED onto the existing L0C content
(`false` / `=0`). Exploit it to compute `R@(I+P) = R + R@P` in ONE L0C pass:

1. **First Mmad, `cmatrixInitVal = 1`** — seed the L0C accumulator with the identity/additive
   term (the `R` of `R + R@P`; equivalently `R@I`). This lands the additive part IN L0C with
   NO vector op.
2. **Second Mmad, `cmatrixInitVal = 0`** — accumulate `R@P` directly onto that same L0C
   accumulator. Result in L0C is `R + R@P = R@(I+P)`.
3. Hold ONE L0C accumulator across the product chain; only the final step Fixpipes out.

No separate `+I` vector-add, no diagonal `Duplicate`, no extra AIV↔AIC barrier for the add.
This is the SAME `cmatrixInitVal=(ki==0)` accumulate idiom the K-tile loop uses in
`fa_class/cv_reference_concrete_params.md` (`matmul_primitive`) — here repurposed to inject
an ALGEBRAIC additive term (the I / R) rather than to accumulate K-tiles.

### Measured result (structure + op-count)
Cuts AIV ops + barriers to **~6 vs ~9** for a 5-step Neumann chain (the redundant per-step
vector add + its MIX fence removed). Precision unchanged (the additive term is exact in the
fp32 L0C accumulator — arguably MORE faithful than an fp16 round-tripped vector add).
Reference STRUCTURE: the cv-reference `gated_delta_rule` `NeumannSolve`.

**Cross-backend witness (independent, PR #200 / customer liuyu15819):** the customer's
Kimi-K3 + a5_ops gated_delta_rule CV-fusion run reached the SAME `cmatrixInitVal` L0C-fold
from scratch on **910B2C / arch 220x / CANN 8.5.1** — replacing a 2016-iteration AIV
scalar-AXPY solve (~98µs/chunk) with an 11-Mmad cube chain (~28µs/chunk): T=4096
8422→2137µs, geomean **3.79×→8.85×** vs PyTorch, MERE ≤9e-4, 16/16. Same mechanism reached
independently on a different backend + SOC-variant (same 220x arch as our Ascend910_9382) =
strong cross-confirmation that this fold is backend-spontaneous, not KB-dependent. See the
reconciled `OL-278` (guarded-MIX base) + `OL-279` (the concrete Fixpipe/Nd2Nz stride units
that make the hand-cube emit runnable).

### Decision rule
- Any `R@(I+P)` / `R + R@P` / `(I+x)^-1` product-chain step on the cube → seed the additive
  term with `cmatrixInitVal=1`, accumulate the product with `cmatrixInitVal=0`, on ONE L0C
  accumulator held across the chain. Do NOT emit a separate vector `+I` add.
- Watch the Neumann solve SIGN (CAND-GDR-1): the accumulator must realize the reference's
  `(I + strict)^-1`, not `(I - strict)^-1` — the fold mechanism is sign-agnostic, so the
  sign bug (if present) is orthogonal and must still be handled at the operand level.

---

## Cross-references

- **P-P117** (`gated_delta_rule_a3_recurrence.md`) — the a3 MIX chunked-recurrence ALGORITHM
  these two cube decisions optimize; the many-small-matmul call-site (§2 decomposition) and
  the Neumann product solve (`T=(I+strict)^-1`) are exactly where P-P119 and P-P120 apply.
- **P-P116** (`fa_class_a3_mix_template.md`) — the a3 MIX cube+vector SYNC template both sit on.
- **P-P68** — single-AIC GEMM via `MatmulImpl` + `IterateAll<sync=true>` (the library primitive
  P-P119 says to REPLACE for the many-small case, but KEEP for few-large).
- **`fa_class/cv_reference_concrete_params.md`** (`matmul_primitive` decision) — the concrete
  `Mmad` 4-arg accumulate form + `cmatrixInitVal=(ki==0)` + per-context `Fixpipe srcStride`
  units the hand-rolled primitive (P-P119) and the L0C-fold (P-P120) both emit. Grounding
  for the exact param values; do not re-derive.
- **CAND-GDR-1** — Neumann solve SIGN gotcha (orthogonal to the P-P120 fold).
- **CAND-GDR-3** — reused `MatmulImpl` needs `SetOrgShape` per call: another facet of the
  per-call library overhead P-P119 sheds by hand-rolling.
- **`cube_vector_fusion.md`** — the a5/arch35 sibling manual-`Mmad` cube-MIX note (different
  chip; P-P119/P-P120 are the a3/arch22 chunked-recurrence-specific decisions).

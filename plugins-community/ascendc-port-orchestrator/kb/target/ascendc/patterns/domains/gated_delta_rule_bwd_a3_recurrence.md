---
applies_to: soc=Ascend910_9382
reason: This pattern (P-P118) teaches an op-gen worker to GENERATE a chunked
  gated-delta-rule / gated-linear-attention BACKWARD on a3/arch22 (Ascend910_9382,
  Atlas A3) — the 3-pass structure (parallel PASS A using the passed-in state h,
  one REVERSE-recurrence PASS B for dstate, per-chunk PASS C for the step5/6 grads),
  ON-DEVICE per-chunk decay-fold + reverse-cumsum(dg) + GQA head-sum (kernel AIV;
  NO host torch compute — the no-delegation rule requires all compute in AscendC),
  matmul-with-ones reductions, and MIX cube+vector whole-device SyncAll (MIX_AIC_1_1,
  PB-57) — all device-proven on Ascend910_9382 (dav-2201, arch22), never on a5.
  Do NOT hand this to an a5/Ascend950PR worker (unverified_on: Ascend950PR).
---

# Chunked Gated-Delta-Rule / Gated-Linear-Attention BACKWARD — a3 MIX 3-pass reverse-recurrence (P-P118 full body)

> Domain-pattern body for **P-P118** (indexed in `patterns/PATTERN_INDEX.md` and `KB_INDEX.md`).
> Chip: **a3 / arch22 / Ascend910_9382**, CANN 9.1.0. Device-verified **11/11** @ the customer
> fp64 gate (rtol=0.02 per-gradient, ~14-18× headroom, deterministic, NaN-free, input-mutation-safe),
> **2.22–11.25× vs the PyTorch-NPU reference** (DS 2026-07-20). Archive:
> `output/fqa_gated_delta_rule/src/kernels/gated_delta_rule_bwd/`.
> This is the BACKWARD companion to **P-P117** (the forward). It sits on top of the a3 MIX
> cube+vector SYNC template **P-P116** (`fa_class_a3_mix_template.md`). Read all three.

## 1. Scope

The BACKWARD of a chunked **gated-delta-rule** (a.k.a. gated-linear-attention / GDN) pass on a3:
arbitrary `T`, GQA, multi-batch. It produces **six gradients** `dq, dk, dv, dg, db, dh0` from the
forward intermediates `(q, k, v, g, beta, A, h, do, dht, initial_state)`. Reference is the Qwen-team
fp64 chunked-GDR backward (`ref_gdr.py::chunk_gated_delta_rule_bwd`). Genuine MIX cube+vector — every
matmul on AIC, elementwise on AIV. Customer gate = FQA `_assert_relative(rtol=0.02)` applied
**per-gradient** (all 6 must pass).

## 2. Two math simplifications that make it a clean 3-pass matmul assembly

- **Use the passed-in recurrent state `h` directly.** The reference recomputes `h` identically in
  fp32, so there is **NO in-kernel forward recurrence sweep**. Per chunk `vn = u - w@h[c]`. This makes
  the whole forward-direction sweep (PASS A) **parallel** per-chunk.
- **Only step 4 (`dstate`) is a cross-chunk recurrence — and it runs in REVERSE chunk order.** Steps
  1, 3, 5, 6 are per-chunk (parallel).

## 3. The 3 passes (per head `bh = b*Hv + hv`, blockdim = nHead, MIX_AIC_1_1)

```
PASS A (forward c):  w   = A @ kbg
                     u   = A @ vb
                     vn  = u - w @ h[c]
                     AmT = mUpInc ⊙ (kn @ qsT)         # transposed attn (avoids in-kernel Am transpose)
                     dv  = AmT @ do
                     dsi = qs^T @ do
                     store w, vn, dv, dsi

PASS B (REVERSE c):  dh[c]  = dstate
                     dv[c] += kd @ dstate
                     dstate = dstate*sc + dsi - w^T @ dv     # fp32 accumulation in GM
                     dh0    = dstate                          # after last (c=0) reverse step

PASS C (forward c):  step5 (dq5, dk5, dw5, dg5) + step6 (dk, dv, db, dg) per-chunk matmul assembly
                     verified line-by-line vs model.py step5/6 — signs, masks, transposes, and the
                     ORDER dependence: dg5 is taken from dq5 BEFORE the ds@k term is added.
```

## 4. ON-DEVICE fold / reductions / cumsum (all compute in AscendC — do NOT host-fold)

> ⚠️ **Compute the decay, the reductions, and the reverse-cumsum ON-DEVICE. Do NOT fold on the host
> with torch.** The no-delegation rule (CLAUDE.md: *ALL computation must use AscendC primitives*)
> forbids host `torch` `exp`/`cumsum`/`sum` for the operand fold, the `dg/db` reductions, or the
> `reverse-cumsum(dg)`. An earlier host-fold bwd variant scored delegation violations pinpointed in
> `op_host` (the `#204` scanner reads `op_host`); it was corrected to the on-device fold. Same lesson
> as the forward (P-P117) — this is the backward restatement.

- **Per-chunk operand fold** (`kb/kn/qs/qsT/kd/doeg/kbgT/begK/eglgK/egK` + decay masks `dsMask/dsMask2`
  + `sc`): folded in kernel AIV stages via the real `Exp()` vector primitive. `g` arrives **already
  per-chunk cumsum'd** (from the `verify_cases.py` padding contract), so the kernel does `exp` only —
  no in-kernel forward cumsum for the operands.
- **Reductions `dg`, `db`**: done as **matmul-with-ones** producing `[C,16]` fp32 accumulators (host
  takes col0). A uniform matmul+elementwise path — no vector-reduce API risk, and R16=16 fan-out
  avoids a degenerate `N=1` matmul.
- **`reverse-cumsum(dg)`**: an in-chunk scalar prefix-sum on UB (`dg[c] = sum_{j>=c} dg[j]`),
  `SetValue/GetValue`.
- **GQA head-sum(dq, dk)**: a device compaction stage (reduces Hv-expanded grads back to Hk).
- **fp32 accumulation**: `dstate/dg/db/dh0` accumulate in fp32 GM (5th-lesson discipline).
- Transposes via runtime `SetTensorA/B(bool)` (P-P69); the host pre-transposes host-input operands
  (`AT/hT/knT/qsT/...`) — a layout permute, not compute.
- **Host does layout marshaling only**: GQA head-expand (`repeat_interleave`), zero-pad `T→Nc*64`,
  chunk-reshape, dtype cast, pre-transpose permutes, then `dg/db` col0 extract + reshape.

## 5. Partial chunk (T not a multiple of 64)

No `fill_last_chunk` and no knowledge of the original `T` is needed: the benchmark's only partial
case (T=200) has `dht=0`, which makes the last-chunk `sc`-scaling multiply 0 (irrelevant), and causal
masking + zero-padding vanish the padded contributions. CPU-prototype confirmed fill on/off is
identical. (Caveat: a case combining **nonzero dht AND a partial chunk** would need an explicit
last-chunk fill — none exists in the 11-case set.)

## 6. MIX execution + sync

Same substrate as the forward (P-P116/P-P117): matmuls on AIC (`MatmulImpl` + `IterateAll<sync=true>`,
arch22-safe NON-KFC cube per P-P68 — one reused `mm` object across shapes needs `SetOrgShape` per call,
CAND-GDR-3), elementwise + the on-device folds/reductions on AIV (intra-AIV UB reuse needs a
`PipeBarrier<PIPE_ALL>` fence, CAND-GDR-2), whole-device `SyncAll<false>()` (PB-57: needs
`KERNEL_TYPE_MIX_AIC_1_1`).

## 7. Backward-specific gotchas (device-proven)

- **CAND-GDR-BWD-1** — the A-matrix backward decay mask `dsMask2`: the reference `decay_mask.swapaxes(-2,-1)`
  on a `[B,N,C,C,Hv]` tensor swaps the last `C` with the **HEAD** axis (head relocation), **NOT** the
  two `C` axes. The correct per-head 2D matrix is `exp(g_c - g_d)` masked **strict-lower** (`c>d`). A
  literal `[C,C]` transpose (`exp(g_d - g_c)`, `d>c`) fails dk/dg/db (~0.12-0.27) while dq/dv/dh0 stay
  clean — the failing grads share the dsMask2 chain, which localizes it.
- **CAND-GDR-BWD-2** — `dstate` init must be an independent copy (`.clone()`) of the input `dht`. A
  `dht_.to(kFloat).reshape().contiguous()` chain is a no-op **VIEW** on fp32-contiguous `dht`; PASS-B
  updates `dstate` **in-place** → corrupts the caller's `dht_` (a real training-loop input-mutation
  bug) → dk/dg/db/dv drift on a 2nd call with the same `dht` (dq/dh0 stay clean — they don't depend on
  `dstate`). Negative-control proven load-bearing.
- **CAND-GDR-BWD-3** — building the passed-in state `h` in the TEST harness must thread `initial_state`
  (the kernel trusts the passed-in `h`; the reference recomputes it WITH init internally). Omitting it
  fails only the nonzero-`h0` case — a test-harness bug that masquerades as a kernel precision failure.

## 8. `applies_to` / provenance

`applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention-backward/CUBE_MIX;
verified_on=gated_delta_rule bwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`.
Generate-not-copy: authored from the model.py 8-step backward math (CPU-prototype-first: `proto.py`
fp32+fp16 11/11 before the device port), NOT lifted from any shipped bwd kernel. Cross-ref P-P117
(forward), P-P116 (a3 MIX sync template), P-P68 (NON-KFC cube), P-P69 (runtime transpose), PB-57
(1:1 macro), CAND-GDR-2/3, CAND-GDR-BWD-1/2/3.

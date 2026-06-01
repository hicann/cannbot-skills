# Reduction Constraints

Read this file when a kernel needs row-wise reductions, normalization, softmax, or quantization statistics inside a `@vf` stage.

## Goal

Choose the right reduction idiom so that:
- the reduction stays in registers when possible to reduce UB traffic
- multi-pass flows keep intermediate state persistent across tile passes
- streaming stats follow the correct update order for numerical stability

## 1. Row sum normalization

Single-pass pattern:
- `sum = RegList.cadd()`
- `dup(sum)` then divide row registers

Files to study:
- `agent/example/kernels/a5/matmul_rowwise_norm.py`

## 2. Tiled softmax (three-pass)

Use when the full `S` dimension does not fit in one tile.

- pass 1: store float logits and track row max with `cmax()` + `dup()`
- pass 2: reload logits, subtract duplicated row max, `exp()`, accumulate row sum, and store exponentials
- pass 3: reload exponentials and divide by duplicated row sum before the final cast

## 3. Streaming softmax-style MLA stats

Single-pass online update per tile:
- `curr_max = max(prev_max, qk_tile.amax(-1))`
- `p_tile = exp(qk_tile - curr_max)`
- `row_sum = prev_sum * exp(prev_max - curr_max) + p_tile.sum(-1)`
- if the tile must be materialized in fp8, update `row_sum` from the float tile first and cast only after the float reduction is complete

Streaming MLA with final normalized `output:[B,H,Dn]`:
- rescale the running numerator by `exp(prev_max - curr_max)` before adding the current block contribution
- keep the numerator in float across all blocks
- apply `output /= row_sum` only once after the loop
- if the value path intentionally uses `p.half().float()`, update `row_sum` from the float `exp(...)` tile before the cast, then cast only for the downstream cube consume path

Files to study:
- `agent/example/kernels/a5/test_mla_entire.py`
- `agent/example/kernels/a2/flash_attn_full.py`

## 4. Absmax scaling and quantization

- `abs()` then `cmax()` for row/block scalar
- divide row by duplicated scalar
- optionally emit scale via duplicated scalar to 64-lane row

Files to study:
- `agent/example/kernels/a5/matmul_chunk_absmax_norm128.py`
- `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`

## 5. Two-pass rowwise normalization

Use when the `N` dimension is too large for single-pass normalization.

- pass 1: matmul + row-stat accumulation into persistent UB buffer + temporary output write
- pass 2: reload temporary output and normalize by accumulated stats

The row-stat buffer must persist in UB across all `N`-tile passes within one `M` tile.

Files to study:
- `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py`
- `agent/example/kernels/a5/matmul_rowwise_l2_norm.py`

## 6. Running state across inner-loop iterations (a2 pattern)

When a kernel must accumulate a statistic (like running max for online softmax) across
tiles in the inner loop, the DSL has no conditional logic (`if first_iteration`).

**Solution: identity-element initialization + unconditional update.**

Initialize the running buffer to the identity element of the accumulation operation
before the inner loop using `dup`, then apply the update unconditionally every iteration:

| Accumulation | Identity element | Update operation | Example |
|-------------|-----------------|-----------------|---------|
| running max | `neg_large` (finite negative sentinel) | `vmax(running, running, tile)` | online softmax max tracking |
| running sum | `0.0` | `add(running, running, tile)` | online softmax sum accumulation |
| running product | `1.0` | `mul(running, running, tile)` | decay chain products |

Why this works: with `neg_large` chosen below every valid tile value,
`max(neg_large, x) = x`, `0 + x = x`, `1 × x = x` — the first iteration
naturally produces the correct initial value without special-casing.

**Choosing the right tensor format for the update operation:**

The update (`vmax`, `add`, etc.) is a binary element-wise operation between two UB tensors.
Both must have matching stride layouts, and the operation must cover all intended elements.

On a2 without registers, `cmax` outputs dense scalars in `[M, 1]` format. Operate on this
format directly:

```python
# Correct: vmax on [64, 1] covers all 64 rows
vmax(ub_rmax_s, ub_rmax_s, ub_max_s)  # both [HALF_M, 1]
```

Do NOT broadcast to `[M, 8]` first and then attempt `vmax` between two `[M, 8]` buffers —
`blk_stride=0` makes that operation cover only 1/8 of the elements.
See `agent/references/constraints/vec-reduction-a2.md` section 5 for the detailed proof.

**Lifetime and reset rules:**

- The running buffer must be reset at the beginning of each **outer** loop iteration
  (each new M-tile gets fresh running stats)
- It persists across all **inner** loop iterations (N-tiles accumulate into it)
- UB is per-sub-block and persistent — no special lifetime management needed

Complete pattern:

```python
ub_rmax_s = Tensor(DT.float, [HALF_M, 1], Position.UB)

with auto_sync():
    for gmt in range(mt_begin, mt_end):        # outer: M-tiles
        dup(ub_rmax_s, neg_large)               # reset per M-tile with the running-max identity
        for nt in range(0, tiles_n):            # inner: N-tiles
            # ... compute tile, get ub_max_s via cmax ...
            vmax(ub_rmax_s, ub_rmax_s, ub_max_s)  # accumulate
            brcb(ub_max, ub_rmax_s, ...)           # broadcast AFTER update
            # ... subtract, exp, store ...
```

On a5 with `Reg`/`RegList`, the same pattern uses register-level `dup` and `vmaxs`
instead of UB-level operations. The identity-element principle is the same.

Files to study:
- `agent/example/kernels/a2/flash_attn_score_iter.py` — validated running max pattern on a2
- `agent/example/kernels/a5/test_mla_entire.py` — streamed running max/sum in a5 register pipeline

## 7. a2 vec reduction (no registers)

On a2, `Reg`/`RegList` are not available. Reductions use UB-to-UB operations:
- `cmax`/`cadd` reduce 64 elements to 1 scalar per repeat (dense output)
- The dense output must be broadcast via `brcb` before use in `sub`/`div`
- For buffers wider than 64 columns, first merge with `vmax`/`add` to 64

Complete pattern: `vmax → cmax → brcb → sub` (sliced for repeat alignment)

Read: `agent/references/constraints/vec-reduction-a2.md`

## 8. General rules

- keep reductions in registers when possible (a5 with `@vf`)
- on a2, use `cmax → brcb` UB-to-UB pattern instead
- use `dup()` to broadcast a scalar reduction result back to full-row width before element-wise operations
- for multi-pass flows, decide upfront which UB buffers persist across passes and which are reused

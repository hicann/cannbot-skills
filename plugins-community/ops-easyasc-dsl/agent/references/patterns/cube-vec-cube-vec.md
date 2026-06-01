# Cube-to-Vec-to-Cube-to-Vec Pattern

> Generic baseline only. For a2 (b3) kernels, prefer `agent/references/patterns/a2-cube-vec-cube-vec.md` (and the softmax variant `a2-cube-vec-cube-vec-softmax.md`), which add delayed-consumer and running-statistic rules specific to a2.

Read this file when one cube stage feeds vec logic, then another cube stage, then a final vec stage.
This is the highest-complexity staged pattern currently worth documenting as a dedicated route.

## Use this pattern when

- there are at least two cube-heavy stages with vec-side logic between them
- one tile may be produced in one iteration and consumed in a later iteration
- delayed state such as softmax stats or rescale factors must follow the consumer lifetime

## Minimal flow

`cube stage 1 -> vec stage 1 -> cube stage 2 -> vec stage 2`

In practice this often becomes a one-tile lookahead schedule with warmup and drain.

## What usually matters most

- keeping producer and delayed consumer lifetimes separate
- giving delayed stages their own counters
- deciding whether the bridge should stay on chip or go through GM workspace
- keeping scalar state aligned with the delayed consumer, not with the original producer
- validating each stage before trusting the fused version

## Stable repository lessons

- if stage 2 reuses a stage 1 operand one iteration later, keep that operand on chip when the lifetime fits
- if the reuse does not fit cleanly, materialize an explicit GM workspace instead of forcing a fake on-chip story
- do not normalize too early when the numerator and denominator streams must both finish first
- when the live query side is truly one row, flatten `(B, H)` into one `BH` axis and keep `rows=1` instead of forcing a wider row tile
- for half-input `BASES=256` attention on a5, keep the outer `256` tile in L1, use `splitk=64` for `q @ k.t()`, and `splitn=64` for `p @ v`
- for fp8 decode attention with external scales, mask invalid tail columns to `-inf` before `rowmax`, scale the probability tile only after the float `row_sum` update, and compensate with a final `scale_v / P_SCALE`
- if the delayed cube consumer wants packed-NZ input, pack the vec-produced tile in UB first, then publish that NZ view into `L1`

## One-tile lookahead scheduling detail

The retained MLA kernel (`agent/example/kernels/a5/test_mla_entire.py`) uses a four-stage on-chip flow:
1. cube: produce score tile `i`
2. vec: update streaming softmax state and cast score tile `i` to probability tile `i`
3. cube: consume delayed probability tile `i-1` with the matching value/key tile
4. vec: rescale and accumulate the delayed output tile `i-1`

Stable control pattern: `for s in range(0, S + TILE, TILE)` with:
- `if s < S`: producer side (warmup + steady state)
- `if s > 0`: delayed consumer side (steady state + drain)

On-chip operand reuse:
- if stage 2 must reuse a stage 1 operand one iteration later, keep that operand resident on chip instead of round-tripping to GM
- in the MLA kernel, `k_nope` stays in `l1kn` and the vec-produced `p` tile is published directly into `l1p`
- in `agent/example/kernels/a5/mha_ifa_nz.py`, the vec-produced `p` tile is first packed with `reg_to_ub(...)` and then published to `l1p` as `.nz()`

Delayed scalar state:
- delayed scalar state must follow the consumer lifetime, not the producer lifetime
- cache per-tile `row_exp_diff` / rescale factors in a slot indexed by the delayed consumer counter
- keep running `row_max`, `row_sum`, and `output_acc` under a single vec owner to avoid duplicate updates

## Typical files to study

- `agent/example/kernels/a5/test_mla_entire.py`
- `agent/example/kernels/a5/mha_ifa.py`
- `agent/example/kernels/a5/mha_ifa_256.py`
- `agent/example/kernels/a5/mha_ifa_fp8_scale_256.py`
- `agent/example/kernels/a5/mha_ifa_nz.py`

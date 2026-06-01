# Deep Note: `agent/example/kernels/a5/flash_attn_full_fp8_causal.py`

Open this file only after the short catalog entry confirmed the kernel is relevant.
Its job is to capture the extra rationale that would otherwise bloat the catalog entry.

## What this kernel is really for

- the multi-row full-sequence a5 attention path, not the simpler `L=1` decode-style `mha_ifa*` family
- a normalized online-softmax pipeline where delayed `p @ v` stays on chip
- a causal contract where only the diagonal tile needs mixed valid/invalid score handling

## Decisions worth copying

- treat both causal masking and `S2` tail invalidation in score space before `rowmax`
- keep future fully-invalid tiles out of the loop with `active_tiles_n = Min(tiles_n, tile_m + 1)`
- publish vec-produced `e5m2` probability tiles into ND `l1p` for the delayed cube consumer
- keep separate `l0c_qk/l0c_pv` and `ub_score/ub_pv` families; do not collapse them into one scratch lineage
- compress row-state scratch into narrow `[1,64]` UB tensors so the larger full-sequence path still fits local memory

## Prefer another kernel when

- the query side is still row-specialized (`L=1`) and `mha_ifa*` already matches
- stage 2 truly wants NZ-published probability tiles
- the contract is half-domain or non-fp8 rather than `e5m2` `q/k/v`

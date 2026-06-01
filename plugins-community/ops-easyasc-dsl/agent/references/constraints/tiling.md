# Tiling Constraints

Read this file when a kernel needs non-trivial tile selection, core split selection, or local-buffer capacity reasoning.
Do not read it for tiny untiled examples.

## Goal

Choose a tile strategy that is:
- legal for local-buffer capacity
- compatible with downstream vec-side dependency
- stable for ownership and tail handling
- justified from the real repository rules

## 1. Separate two decisions

Treat these as separate layers:
1. tile shape: `TILE_M`, `TILE_N`, `TILE_K`
2. core split: `m_split`, `n_split`

Do not collapse them into one vague "best tiling" decision.

## 2. Default ownership pattern

For standard cube-major matmuls, split by tile index, not by raw row range:
- `tile_m = CeilDiv(M, TILE_M)`
- `tile_m_per_core = CeilDiv(tile_m, GetCubeNum())`
- iterate `mt in [tile_m_begin, tile_m_end)`

If the chosen strategy returns both `m_split > 1` and `n_split > 1`, map `GetCubeIdx()` onto a 2D split grid and partition both axes explicitly.

For batched independent small matmuls, flatten batch-like dimensions first and split directly over the flattened batch axis:
- `BHN = B * H * N`
- `bhn_per_core = CeilDiv(BHN, GetCubeNum())`
- iterate `bhn_idx in [bhn_begin, bhn_end)`
This is simpler than tile splitting when each item is one full small matrix pair.

## 3. Use the estimator when the choice is not obvious

Use `agent/scripts/estimate_matmul_datamove.py` when:
- the matmul is large enough that core split matters
- there are multiple legal tile candidates
- downstream vec work constrains which axis may be split

Read the estimator result as:
- tile shape
- core split
- loop mode
- candidate tie set, if any

Choose `split_mode` from downstream dependency, not only from cube-side datamove:
- `split_m`: keep all `N` tiles for the same `M` rows together on one core
- `split_n`: keep all `M` tiles for the same `N` columns together on one core
- `mix`: both axes may be split because vec-side work does not impose a one-axis ownership constraint

If two candidates tie on datamove, break the tie with downstream local-memory fit instead of taking the first row mechanically.
Concrete example: for `x @ y.t() + 1.0`, both `256x128x256` and `128x256x256` may tie on datamove for the large shape; the latter is the better fit because the vec stage can reuse a `64x256` UB tile directly.

## 4. Device-specific capacity and core count

Buffer budgets and core count differ between a2 and a5. For exact values, see `agent/references/facts-device-runtime.md`. For `splitk` / `splitn` element count and byte-budget formulas, see `agent/references/facts-authoring.md`.

Consequences that matter for tile strategy:
- a5 tile strategies that fit `DBuff L0C` (e.g. `TILE_M=128, TILE_N=256 → 128*256*4*2 = 256 KB`) will **overflow on a2** (`128 KB`).
- a5 core split uses `GetCubeNum()=32`; a2 uses `20`. Verify load balance separately for each target.
- a5 vec-side `DBuff UB` allocations up to `256 KB` must stay within `192 KB` on a2.

Practical rule: when writing an a2 kernel, always verify `L0C DBuff` first. `DBuff` allocates 2 slots, so a single `l0c` allocation needs `2 * TILE_M * TILE_N * sizeof(dtype) <= l0c_cap`.

## 5. Capacity rules are mandatory authoring checks

Before blaming the simulator, check the real buffer budgets. The per-device caps (`L0A`, `L0B`, `L0C`, `UB`) live in `agent/references/facts-device-runtime.md`, and the `splitk` / `splitn` element count / byte-budget formulas live in `agent/references/facts-authoring.md`. Convert element counts into bytes using the real local-buffer dtype size, and require each buffer to stay within its cap.

## 6. Choosing `splitk` vs `splitn`

Use the overflowing side to choose the split mode (do not switch blindly).

Choose `splitk` when K-side staging into `L0A` / `L0B` is too large, or when you want to keep a large outer `TILE_K` for strategy selection but legalize the real inner load size.

Choose `splitn` when N-side staging is too large, or when output-tile width pushes the buffer budget too far.

Hard rule (`splitk`, `splitn` >= `32`) and the validated large-`K` aligned MKNK example (`TILE_M=128, TILE_N=256, TILE_K=256, SPLIT_K=64`) live in `agent/references/facts-authoring.md`.

Fallback rules:
- if `splitk` fails even at `32`, retile `TILE_M` / `TILE_N`
- if `splitn` fails even at `32`, try `splitk` instead of pushing `splitn` lower

Practical rule for the validated pattern: choose `m_split` / `n_split` from the outer tile; use `splitk` only to legalize the inner cube staging.

## 7. `L0C` authoring rule

Treat non-zero `L0C` row offsets on matmul destinations as unsupported in authoring.
Even though the simulator has an offset path, the repository rule is:
- keep matmul destinations anchored at row offset `0`
- solve oversized `M` with a higher-level `TILE_M` decision instead of `l0c[row_offset:..., ...]`

`N`-side subdivision is still fine when the destination remains anchored.

## 8. Quick checklist

Before accepting a tiled kernel, verify:
- **target device identified** (a2/a5) and device-specific budgets used
- tile shape chosen explicitly
- core split chosen explicitly, using the correct core count for the device
- split mode matches downstream dependency
- `L0A` / `L0B` byte budgets checked
- **`L0C` DBuff total** checked against the device-specific `l0c_cap`
- **`UB` DBuff total** checked against the device-specific `ub_cap` (when vec stages exist)
- `splitk` or `splitn` kept at `>= 32`
- `L0C` destination remains row-offset `0`
- ownership and counters still make sense for the chosen loop structure

## Files to study

- `agent/scripts/estimate_matmul_datamove.py`
- `agent/scripts/tools_summary.md`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`

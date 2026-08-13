# Online Softmax Tail Pattern

Use this pattern when tiled online softmax must exclude non-aligned score rows
or columns before they affect row max, exponentials, running sums, or causal
visibility. Generic local-shape and GM-boundary tail constraints live in
`agent/references/constraints/tail.md`.

Boundary: this pattern owns the points where tail and causal masks enter the
online-softmax computation. Vec op semantics such as
`cmax` / `cadd` / `brcb`, mask bit layout, and `select` flag behavior live in
`constraints/vec.md`. A2 bridge topology and physical sub-block layout live in
`constraints/a2.md`.

For sync-side constraints see `agent/references/constraints/sync.md`.
For `cmax` / `cadd` / `brcb` vec op details see `constraints/vec.md`.
For a2 bridge pipeline patterns see `patterns/a2-mixed-pipeline.md`.

## Applies when

- score computation is tiled over query rows or key/value columns;
- a running max/sum update spans several score tiles;
- padded score lanes would otherwise enter `cmax`, `exp`, or the probability
  sum;
- causal and physical tail masks may overlap on the same tile.

## Logical dataflow

```text
score tile -> S2/causal score-domain mask -> row max update
           -> shift by current max -> S1 row mask -> exp
           -> float row sum update -> probability cast/publish
           -> delayed numerator update -> final divide
```

---

## Physical invariants

### Stable local shapes

Local buffers stay full-tile sized.
Use `valid_m`, `valid_n`, and `valid_k` at GM read/write boundaries and at
explicit UB-domain masks or reductions that must exclude padded lanes.

Stable local shapes make lowering and simulator behavior predictable.
Shrinking local buffers for tail tiles causes shape drift between the intended logical tile and the staged buffer.

Repository rule:
- `l0c_to_ub` does **not** support sliced `L0C` source tensors
- keep `L1` and `L0C` shapes full tile-sized; zero-fill `L1` first when needed
- apply row / column limits at GM boundaries or as explicit UB / vec masks, not
  by shrinking cube local buffers

---

### Half-row vec writeback split

Two patterns exist. Use the right one for the pipeline type.

**a5-style compact split** — for matmul-norm and similar cube → vec kernels:

```python
# agent/example/kernels/a5/matmul/matmul_rowwise_norm.py L62-66
half_rows = CeilDiv(valid_m, 2)
sb_idx    = GetSubBlockIdx()
row_begin = sb_idx * half_rows
row_end   = Min(row_begin + half_rows, valid_m)
row_count = row_end - row_begin
```

**a2-style fixed physical split** — for flash-attention kernels that bridge cube
output through GM workspace before vec processing:

```python
# agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py L210-228
sb     = GetSubBlockIdx()
sb_row = Var(sb * HALF_M)          # fixed physical row origin: 0 or 64
# ...
local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
```

The a2 split is **not** the a5 `CeilDiv(valid_m, 2)` compact half-split; subblock 0 always owns rows `[0:64)` and subblock 1 always owns rows `[64:128)`.
The a2 physical split itself is defined in `constraints/a2.md` §4; this section
only tells how tail rows map to that fixed split.

Do not mix the two styles in the same pipeline.

---

### Workspace tile shape stability

A2 cube → vec bridge kernels use `split_workspace(...)` to carry tiles across stages.
The workspace is declared at full stable tile shape and never shrunk for tails.
GM boundaries slice by the valid tile size; UB reductions and exponent-domain
math still need explicit masks when padded lanes would affect semantics.

```python
# grouped full-attention path: agent/example/kernels/a2/attention/flash_attn_full.py
score_ws = split_workspace(DT.float, [GetCubeNum(), GROUP_STAGE_SLOTS, TILE_M, GROUP_N * TILE_N], name="score_ws")
p_ws     = split_workspace(DT.half,  [GetCubeNum(), GROUP_STAGE_SLOTS, TILE_M, GROUP_N * TILE_N], name="p_ws")
pv_ws    = split_workspace(DT.float, [GetCubeNum(), GROUP_STAGE_SLOTS, TILE_M, TILE_K], name="pv_ws")
```

The same "stable full tile shape, valid GM slices at the boundary" principle appears in
the one-tile `flash_attn_full_pj_hif8.py` family, where the workspace shape is the simpler
`[GetCubeNum(), 2, TILE_M, TILE_N]` for score/P slots. Pick the slot count from the
pipeline lifetime; do not shrink the local/workspace tile just because the GM tail is shorter.

---

### Score-domain `-inf` sentinel

Hardware `float` cannot represent true `-inf` reliably in all vec paths.
Use a large finite negative value as the sentinel:

```python
# agent/example/kernels/a2/attention/flash_attn_full.py
NEG_LARGE = -1.0e30
```

Initialize the running row-max accumulator to `NEG_LARGE` before the tile loop
(see `flash_attn_full.py` and `flash_attn_full_pj_hif8.py`).

---

## Minimal skeleton: S2 column tails

### Why GM-boundary slicing alone is not enough

When the last `k` / `v` tile has `valid_n < TILE_N`, the padded columns of
the staged score tile contain zeros.
Those zeros are seen by `rowmax` and can inflate `curr_m`, corrupting
`expdiff`, `row_sum`, and the final output even if `p_j` is later zeroed.

Rule: **pad tail columns must behave like `-inf` before `rowmax` / `cmax`.**

A `p`-domain-only mask is insufficient — it cannot fix `rowmax`, `curr_m`, or `expdiff`.

### S2 masking in the implementation

After scaling and before `cmax`:

```python
# agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py L257-260
if valid_n < TILE_N:
    apply_score_tail_mask(ub_score, valid_n)
vmax(ub_tmp, ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, HALF_N:TILE_N])
cmax(ub_max_s, ub_tmp)
```

`apply_score_tail_mask` splits `valid_n` across the two 64-column halves and
calls `mask_score_half_suffix_invalid` on each half (L148-153).

### Update order

The vec reduction and broadcast semantics in this sequence are owned by
`constraints/vec.md`. This section owns where S2 and S1 tail masks enter the
online-softmax sequence.

```
1. vmax + cmax → ub_max_s          (current tile row-max)
2. vmax        → ub_rmax_s         (running max update)
3. sub score   - ub_max            (shift to exponent domain)
4. [S1 tail mask here — see `S1 row tails`]
5. exp(score)
6. add halves  → ub_tmp; cadd → ub_sum_s   (sum in float before any cast)
7. mul ub_rsum_s * expdiff_buf     (rescale running sum)
```

Sum is accumulated in `float32` (`cadd` into a `[1, HALF_M]` float tensor) and
the cast to `half` / `hif8` happens only after this step
(`flash_attn_full_pj_hif8.py` L275, L291 shows the update order and the
aligned `[1, HALF_M]` scalar-buffer layout).

### Suffix-invalid mask construction

For one 64-column score half, build a packed suffix-invalid `uint64` mask using
a signed left-shift trick to avoid simulator scalar-cast issues:

```python
# agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_causal.py L127-134
@func()
def build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    signed_mask = Var(-1, DT.int64)
    two_i64 = Var(2, DT.int64)
    for _ in range(0, valid_cols):
        signed_mask <<= signed_mask * two_i64
    out_mask <<= signed_mask
```

Then:
```python
set_mask(0, low_mask)
dup(score_half, NEG_LARGE)
reset_mask()
```

Mask bit order: `low` covers `mask[0:64]`; bit 0 → lowest lane; bit 63 → highest lane.
Stub call signature: `set_mask(mask_high, mask_low)`.

### Valid-n case table (TILE_N = 128, HALF_N = 64)

| `valid_n` range  | left half [0:64)       | right half [64:128)     |
|------------------|------------------------|-------------------------|
| `== 128`         | fully valid            | fully valid             |
| `64 < n < 128`   | fully valid            | suffix invalid mask     |
| `== 64`          | fully valid            | fully invalid (dup NEG) |
| `0 < n < 64`     | suffix invalid mask    | fully invalid (dup NEG) |
| `== 0`           | fully invalid (dup NEG)| fully invalid (dup NEG) |

---

## S1 row tails

S1 tail means invalid **rows** in the local score tile, not columns.

Stable local quantity:
```python
local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
```
Here `sb_row` comes from the a2 fixed physical split in `constraints/a2.md` §4.

Masking point: **after `score - curr_m`, before `exp`**. Masking before `cmax`
creates invalid-row sentinel values that interact with the running max.

```python
# agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py L272-273
if local_valid_m < HALF_M:
    apply_score_row_tail_mask_after_shift(ub_score, local_valid_m)
```

Invalid local rows become zero after `exp` and contribute nothing to `p @ v`.
If they produce `NaN` after the final `out / row_sum`, that is acceptable
because those rows are not written back to GM.

GM boundary: write only `local_valid_m` rows (L339-345).

### S1 validation set (TILE_M = 128)

Keep at minimum: `S1 % 128 ∈ {0, 1, 63, 64, 65, 127}` plus `S1 == 257` and one multi-head shape.
Validate `S2` aligned first so failures are easier to attribute.

---

## Block-wise causal tail

For block-32 causal masking (`k_pos // 32 <= q_pos // 32`):

```python
# agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal.py L16, L164-167
BLOCK_CAUSAL = 32
if ni == lmt:
    apply_diagonal_tile_block32_causal_mask(ub_score, sb_row)
```

Tile classification:
- `nt < lmt`: fully valid tile
- `nt == lmt`: diagonal tile, mixed valid/invalid columns per row
- `nt > lmt`: skip (fully invalid)

For the diagonal tile, apply `apply_diagonal_tile_block32_causal_mask(...)`
before `cmax`. The current implementation directly writes `NEG_LARGE` into the
invalid suffix with `dup(...)`; it does not use packed `uint8` causal-mask
tensors or `select(...)`.

On A2, keep those `dup(...)` writes on aligned score-row views. Do not issue vec
operations on narrow one-row packed-`uint8` mask views: the packed representation
does not waive A2 local-tensor alignment and stride requirements. Repair the
invalid score suffix directly, then run rowmax/online-softmax on the repaired
score tile.

If the same tile is also the `S2` tail tile, apply the causal mask first, then
the `valid_n` tail mask second.

---

## Failure signatures

- Aligned shapes pass; odd shapes fail.
- Only the last tile is wrong.
- One vec subblock is correct; the other is garbage.
- Output shape looks right but boundary rows or columns are corrupted.

When this happens: inspect GM boundary slices first. Do not start by shrinking
local buffer shapes.

For online softmax specifically: if the symptom is incorrect `row_max` or
`row_sum` rather than out-of-bounds memory, the score-domain column mask is
likely missing or applied too late.

---

## Runnable references

- `agent/example/kernels/a2/attention/flash_attn_full.py` — grouped a2 full flash-attention with `split_workspace`, `NEG_LARGE`, and normalized online softmax
- `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` — S2 + S1 tail masking with update-order comments
- `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_causal.py` — diagonal causal mask + S2 tail combination
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal.py` — block-32 causal variant
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py` — a5-style compact `CeilDiv` half-row split

## Do not use when

- the kernel has only an elementwise GM tail and no reduction sees padded data;
- the reduction is a simple row/group statistic outside online softmax;
- local buffers are being shrunk for tails instead of kept at stable physical
  shape; fix the generic tail constraint first.

## Source escape

For a new mask representation, dtype, or simulator/hardware disagreement,
follow `agent/references/evidence-escalation.md`. Isolate whether the first bad
boundary is score masking, row-max update, exponentiation, row-sum update, or
final writeback before changing the whole pipeline.

# a2 Mixed-Pipeline Patterns (cube ↔ vec GM-bridge family)

## Applies when

Use this file when writing any a2 (`easyasc.a2`, device `b3`) kernel that crosses
between the cube and vec compute pipelines.

**Hardware constraint that drives every section below:**
On a2, `l0c_to_ub` and `ub_to_l1_*` are both unavailable.
Every cube→vec or vec→cube handoff must transit through GM workspace.
This is the single root cause behind all four pipeline patterns.

Constraint cross-references: `agent/references/constraints/a2.md`,
`agent/references/constraints/sync.md`, `agent/references/constraints/tail.md`,
`agent/references/constraints/vec.md`.

## Logical dataflow families

| Section | Dataflow |
| --- | --- |
| 1 | cube -> vec final postprocess |
| 2 | cube -> vec -> delayed cube |
| 3 | cube -> vec -> delayed cube -> delayed vec accumulation |
| 4 | normalized online-softmax form of the four-stage pipeline |

Read only the section matching the contract. Use
`agent/references/patterns/lookahead-drain.md` for the generic warmup/steady/
drain structure and `agent/references/patterns/online-softmax-tail.md` for
mask/update ordering.

## Shared physical invariants

- every A2 cube/vec handoff uses an explicit GM workspace;
- workspace shape stays full-tile and slot count follows the delayed lifetime;
- the two vec sub-blocks keep fixed physical row ownership;
- producer, delayed cube consumer, and delayed vec consumer use separate
  counters when their lifetimes differ;
- mutex depth, workspace slots, and local buffer slots must describe the same
  number of in-flight beats.

---

## 1. cube → vec (2-stage)

### Topology

```
GM(q,k) → L1 → L0A/L0B → mmad → L0C
                                    │  l0c_to_gm_nz2nd (FIX)
                                GM(score_ws)
                                    │  gm_to_ub_pad (MTE2)
                                   UB → vec ops → GM(output)
```

### When to use

- Cube produces one matmul tile; vec must post-process before final writeback.
- No later cube stage consumes the vec result.

### Reference sub-pattern

The first stage inside `agent/example/kernels/a2/attention/flash_attn_score_pv.py`
implements this bridge (`score_ws`, `qk_mutex`, `stage1_cnt`). The full kernel
continues into the Section 2 delayed `p @ v` stage, so do not treat it as a
standalone pure two-stage final-output example.

### Workspace design

```python
# pingpong: 2 slots per cube core
score_ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N], name="score_ws")
ws_slot = var_mod(stage1_cnt, 2)
```

Sub-block split (2 vec lanes per cube core, each owns HALF_M rows):

```python
sb = GetSubBlockIdx()
sb_row = Var(sb * HALF_M)
# cube writes full TILE_M; each vec lane reads its own half
score_ws[cube_idx, ws_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
ub_score <<= score_ws[cube_idx, ws_slot, sb_row:sb_row + HALF_M, 0:TILE_N]
```

### Ownership edge

```python
CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
```

`FIX` = cube's last op (`l0c_to_gm_nz2nd`).
`MTE2` = vec's first op (`gm_to_ub_pad`).
This differs from a5, which uses `l0c_to_ub` and starts on `Pipe.V`.

### Iteration skeleton

```python
for tile_idx in range(...):
    ws_slot = var_mod(ws_cnt, 2)
    # cube
    matmul(l0c[l0c_cnt], ...)
    cvmutex.lock()
    score_ws[cube_idx, ws_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
    cvmutex.ready()
    # vec
    cvmutex.wait()
    ub_data <<= score_ws[cube_idx, ws_slot, sb_row:sb_row + HALF_M, 0:TILE_N]
    # ... vec ops ...
    output[...] <<= ub_out
    cvmutex.free()
    ws_cnt += 1
```

### Common pitfalls

- **Workspace stride:** write `0:TILE_M, 0:TILE_N` to workspace; do not crop to tail
  columns on the workspace side. `l0c_to_gm_nz2nd` and `gm_to_ub_pad` infer row
  stride from the parent GM shape. Crop only at the final GM write.
- **Not for a5:** use `l0c_to_ub` + `@vf` on a5 instead.
- **Not for cube-only:** direct `l0c_to_gm_nz2nd` to output, no workspace needed.

---

## 2. cube → vec → cube (3-stage)

### Topology

```
GM(q,k) → L1 → L0 → L0C(score)
                        │  CvMutex(0, FIX→MTE2)
                     GM(score_ws)
                        │
                   UB(score) → vec(running_max, exp, cast p) →
                        │  VcMutex(1, MTE3→FIX)
                     GM(p_ws)
                        │
                  L1(p) ─┬─ L0 → L0C(pv) → GM(output)
                 GM(v)  ─┘
                [delayed by one tile: consume j-1 while producing j]
```

### When to use

- Formula is structurally `cube → vec → cube`.
- A later cube matmul consumes the vec result.
- The delayed consumer naturally runs one tile behind the producer.

Typical example: `score_j = q @ k_j^T`; vec computes `p_j = exp(score_j - m).half()`;
delayed cube computes `pv_j = p_j @ v_j`.

### Reference kernel

`agent/example/kernels/a2/attention/flash_attn_score_pv.py` is the active reference: two
workspaces (`score_ws`, `p_ws`), a mutex pair (`qk_mutex`, `p_mutex`), and the
`for ni in range(0, tiles_n + 1)` warmup/drain loop.

### Stable schedule

```python
for ni in range(0, tiles_n + 1):
    if ni < tiles_n:
        # stage 1: produce tile j = ni
    if ni > 0:
        # stage 2: consume tile j = ni - 1
```

Warmup: first iteration produces only. Drain: last iteration consumes only.
Do not force both stages into the same tile index in one iteration.

### Workspace layout

```python
score_ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N], name="score_ws")
p_ws     = split_workspace(DT.half,  [GetCubeNum(), 2, TILE_M, TILE_N], name="p_ws")
```

Two separate workspaces; score is naturally float, stage-2 cube input wants half.

### Ownership edges

```python
# cube → vec (score)
CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
# vec → cube (p_j)
VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
```

Conservative `dst_end_pipe=Pipe.FIX` on the VcMutex: free only after the delayed
cube stage finishes the tile, not after the vec write.

### Shared L0C rule

One physical `l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)` shared by
both cube stages. a2 has 128 KB L0C; a second full-size family does not fit.
This is safe because stage 1 publishes `L0C → score_ws` before stage 2 reuses
the slot. Advance one shared `l0c_cnt`.

### Counter layout

- `l1qk_cnt`: stage-1 q/k loads
- `l1pv_cnt`: stage-2 p/v loads
- `l0c_cnt`: shared L0C family
- `stage1_cnt`: score_ws / p_ws producer slot rhythm
- `stage2_cnt`: delayed consumer slot rhythm

### Two-sub-block publication

```python
# vec lane writes only its own HALF_M rows
p_ws[cube_idx, slot, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p
# stage-2 cube waits then reads full tile (both sub-blocks' tokens)
l1p[l1pv_cnt] <<= p_ws[cube_idx, p_slot, 0:TILE_M, 0:TILE_N]
```

`wait_vec()` completes only after both vec lanes have produced their tokens,
making the full-tile read safe.

### Common pitfalls

- Trying to use `UB → L1` directly on a2 — not available.
- Allocating separate full-size L0C families for both cube stages.
- Merging all counters because L0C counter is shared.
- Forgetting `tiles_n + 1` warmup/drain loop.
- Writing only one sub-block's rows into `p_ws`.
- Releasing VcMutex before the delayed cube stage finishes.

---

## 3. cube → vec → cube → vec (4-stage, delayed numerator accumulation)

### Topology

```
GM(q,k,v) → L1 → L0 → L0C(score)
                          │  CvMutex(0, FIX→MTE2)
                       GM(score_ws)
                          │
                     UB(score) → vec(max, expdiff, exp, cast p) →
                          │  VcMutex(1, MTE3→FIX)
                       GM(p_ws)
                          │
                  L1(p) ──┬─ L0 → L0C(pv)
                 GM(v)  ──┘      │  CvMutex(2, FIX→MTE2)
                             GM(pv_ws)
                                 │
                           UB(pv) → vec(expdiff scale + add → accum) → GM(out)
```

### When to use

- Formula keeps running max and rescaled unnormalized numerator only.
- **No** running sum; **no** final divide by row_sum.
- If you need running sum + final divide, use Section 4 instead.

Typical formula:
```
score_j = q @ k_j^T * scale
curr_m   = max(prev_m, rowmax(score_j))
expdiff  = exp(prev_m - curr_m)
p_j      = exp(score_j - curr_m).half()
pv_j     = p_j.float() @ v_j
out      = out * expdiff + pv_j
```

### Reference kernel

There is no checked-in standalone kernel that stops at exactly this
unnormalized variant. Use `agent/example/kernels/a2/attention/flash_attn_score_pv.py` for the
score→p sub-pattern. Use `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` when
you want the simpler one-tile three-workspace / three-mutex mechanics, or
`agent/example/kernels/a2/attention/flash_attn_full.py` when you want the grouped full-attention
version of the same ownership pattern. Do not claim either is an exact Section 3
implementation; both additionally implement the Section 4 running-sum and final divide.

### Workspaces and ownership edges

```python
# one-tile form; grouped kernels widen score/P to GROUP_N * TILE_N and increase slots
score_ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N], name="score_ws")
p_ws     = split_workspace(DT.half,  [GetCubeNum(), 2, TILE_M, TILE_N], name="p_ws")
pv_ws    = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, D],      name="pv_ws")

CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)   # score
VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)   # p_j
CvMutex(2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)   # pv_j
```

### Vec-resident persistent state

```python
ub_max_s    = Tensor(DT.float, [1, HALF_M], Position.UB)   # current tile row max
ub_rmax_s   = Tensor(DT.float, [1, HALF_M], Position.UB)   # running row max
ub_zero_s   = Tensor(DT.float, [1, HALF_M], Position.UB)   # scalar copy helper
expdiff_buf = DBuff(DT.float,  [1, HALF_M], Position.UB)   # delayed expdiff slots
accum_ub    = Tensor(DT.float, [HALF_M, D], Position.UB)   # numerator accumulator
```

### Delayed expdiff handling

Never snapshot row-scalar state with `ub_to_ub` — that op infers burst length in
C0 blocks and can silently mis-copy row scalars. Use an aligned `[1, HALF_M]`
scalar tensor and copy with `add(..., zero)` instead.

```python
# stage 1 vec:
add(expdiff_buf[stage1_slot], ub_rmax_s, ub_zero_s)   # snapshot prev_m (not ub_to_ub)
vmax(ub_rmax_s, ub_rmax_s, ub_max_s)                  # update running max
sub(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot], ub_rmax_s)
exp(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot])  # => exp(prev_m - curr_m)

# stage 3 vec (delayed by one tile):
brcb(ub_expdiff, expdiff_buf[stage2_slot], repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
mul(accum_ub[0:HALF_M, 0:HALF_N],    accum_ub[0:HALF_M, 0:HALF_N],    ub_expdiff)
mul(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_expdiff)
add(accum_ub, accum_ub, ub_pv)
```

Sliced scaling is required: `accum` is wide `[HALF_M, 128]`;
`expdiff` broadcast is narrow `[HALF_M, 8]`.

### Common pitfalls

- Labelling this kernel "online softmax" — it has no running sum or final divide.
- Placing `expdiff` snapshot after the running-max update (loses `prev_m`).
- Using `ub_to_ub` to copy scalar state.
- Releasing `pv_mutex` before stage-3 vec finishes its accumulation.

---

## 4. cube → vec → cube → vec — normalized online softmax

### Topology

Same as Section 3 plus running row sum and a final post-loop divide:

```
... (same three-workspace bridge as Section 3) ...
→ UB(pv) → vec(expdiff scale + add → accum)   [inner loop]
→ final: accum / row_sum → GM(out)             [post-loop]
```

### When to use

- Full normalized online softmax: `out = (Σ_j pv_j * rescale_j) / row_sum`.
- Required when downstream consumers expect normalized attention probabilities.

Family of reference kernels (all variants of this pattern):
- `agent/example/kernels/a2/attention/flash_attn_full.py` — grouped full-attention baseline (`GROUP_N=4`) with half-precision p_j
- `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` — hif8-quantized p_j
- `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_commonub.py` — shared UB layout variant
- `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_causal.py` — causal mask + hif8
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal.py` — causal mask + block32
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v3.py` — block32 causal with balanced M scheduling, hoisted Q, DBuff stage-1 UB scratch, and depth-4 lookahead
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v2.py` — causal mask + block32 with balanced M-tile scheduling and lookahead-2 queueing
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v4.py` — grouped-Skv study artifact with `[128,512]` score/P workspaces and grouped PV publication
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v5.py` — grouped path with per-M grouped pipeline control, slot-indexed expdiff storage, and V preload ring
- `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v6.py` — current fastest grouped stream with continuous `(M tile, group)` scheduling and separate cube/vec cursors

All variants share the same three-mutex topology and the post-loop divide idiom.
For choosing among block32 variants, read
`agent/references/examples/deep/a2-block32-causal-family.md` after the catalog
has narrowed the candidate.

### Vec-resident persistent state

Section 3 state plus one extra scalar:

```python
ub_max_s    = Tensor(DT.float, [1, HALF_M], Position.UB)   # current tile row max
ub_rmax_s   = Tensor(DT.float, [1, HALF_M], Position.UB)   # running row max
ub_sum_s    = Tensor(DT.float, [1, HALF_M], Position.UB)   # current tile row sum
ub_zero_s   = Tensor(DT.float, [1, HALF_M], Position.UB)   # scalar copy helper
expdiff_buf = DBuff(DT.float,  [1, HALF_M], Position.UB)   # delayed expdiff slots
accum_ub    = Tensor(DT.float, [HALF_M, D], Position.UB)   # numerator accumulator
ub_rsum_s   = Tensor(DT.float, [1, HALF_M], Position.UB)   # running row sum (new)
```

Counter layout identical to Section 3. Running `row_sum` stays vec-resident
across the whole inner loop; it does not need a delayed counter.

### Stable stage-1 update order

1. `rowmax(score_j)` → `ub_max_s` in `[1, HALF_M]`
2. snapshot `prev_m` → `expdiff_buf[stage1_slot]` with `add(..., zero)`
3. `vmax(ub_rmax_s, ub_rmax_s, ub_max_s)` — update running max
4. `sub` + `exp` on `expdiff_buf` → `exp(prev_m - curr_m)`
5. broadcast `ub_rmax_s` → subtract from score tile (both 64-col halves)
6. `exp(ub_score, ub_score)` → float probability tile
7. `add` + `cadd` → `ub_sum_s` in `[1, HALF_M]`
8. `mul(ub_rsum_s, ub_rsum_s, expdiff_buf[stage1_slot])` + `add(ub_rsum_s, ub_rsum_s, ub_sum_s)`
9. `cast(ub_p, ub_score)` — cast to half **after** row_sum update

One-tile update-order reference from `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py`
(the grouped `flash_attn_full.py` preserves the same rule at group level: update
row_sum from the float probability tile before casting/publishing P):
```python
add(expdiff_buf[stage1_slot], ub_rmax_s, ub_zero_s)
vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
sub(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot], ub_rmax_s)
exp(expdiff_buf[stage1_slot], expdiff_buf[stage1_slot])
...
exp(ub_score, ub_score)
add(ub_tmp, ub_score[0:HALF_M, 0:HALF_N], ub_score[0:HALF_M, HALF_N:TILE_N])
cadd(ub_sum_s, ub_tmp)
mul(ub_rsum_s, ub_rsum_s, expdiff_buf[stage1_slot])
add(ub_rsum_s, ub_rsum_s, ub_sum_s)
cast(ub_p, ub_score)
```

Do **not** move the row-sum update after the cast; that silently changes the
reference contract.

For int8 probability value paths, keep the same order: update `row_sum` from
the float probability tile first, then apply the value-path cast. In
`sage2_vnomean_int4.py`, the value path is
`p_int8 = (p * 127).float -> half -> int8` with away-from-zero rounding before
the delayed `p @ v` cube stage. Its extra smooth score term also crosses a
`DT.half` workspace before vec softmax, so the reference rounds
`qm @ k_smooth` through half at that boundary.

### Post-loop final divide

```python
# order reference: flash_attn_full.py; use aligned `[1, HALF_M]` row scalars in new code
brcb(ub_rowsum, ub_rsum_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
div(accum_ub[0:HALF_M, 0:HALF_N],    accum_ub[0:HALF_M, 0:HALF_N],    ub_rowsum)
div(accum_ub[0:HALF_M, HALF_N:TILE_K], accum_ub[0:HALF_M, HALF_N:TILE_K], ub_rowsum)
```

Divide happens once after the inner loop: `accum` must finish all delayed `pv_j`
contributions before dividing.

### Non-aligned S2 tail rule

With `S2 % 128 != 0`, GM-boundary `valid_n` slicing alone is insufficient.
Padded score columns corrupt `rowmax`, `curr_m`, `expdiff`, and `row_sum`.

Stable rule: load k/v through `valid_n`; keep local score buffers full-sized;
before `cmax`, force invalid score columns to a sufficiently large finite
negative sentinel (not literal `-inf`). After `exp`, those columns naturally
become 0. See `agent/references/patterns/online-softmax-tail.md` for the exact
mask construction and update order.

### Vec rules condensed (TILE_N = 128, D = 128 path)

1. Keep `running_max`, `running_sum`, `expdiff` in aligned scalar format `[1, HALF_M]`.
2. Snapshot scalar state with `add(dst, src, zero)`, never `ub_to_ub`.
3. `cmax` / `cadd` output dense scalars into `[1, HALF_M]`; broadcast with `brcb(dst, src, repeat=HALF_M//8, dst_blk_stride=1, dst_rep_stride=8)`.
4. Pair a wide `[HALF_M, 128]` buffer with a narrow `[HALF_M, 8]` broadcast row
   by operating on `buf[:, 0:64]` then `buf[:, 64:128]`.
5. Update `running_sum` from the float `p_j` tile before any cast to half or hif8.
6. For non-aligned S2, mask invalid score columns before `cmax`.

### Block32 causal optimization notes

For blockwise causal attention, contiguous M-tile ownership can badly imbalance
active N-tile counts. The stable A2 scheduling shape is:

```python
core_step = Var(1, DT.int)
core_step <<= GetCubeNum()
for gmt in range(GetCubeIdx(), BH * tiles_m, core_step):
    bh = Var(gmt // tiles_m)
    lmt = Var(gmt % tiles_m)
    if var_mod(bh, 2) == 1:
        lmt <<= tiles_m - 1 - lmt
```

This round-robins M tiles across cube cores and reverses M order on odd heads,
pairing low- and high-causal-work tiles on the same cores while preserving the
formula and output contract.

Other stable optimizations for the block32 half-probability family:

- Hoist Q GM->L1 once per M tile. Guard the one-load/many-read lifetime with
  `MTE2 -> MTE1` valid and an `MTE1 -> MTE2` slot-free edge, then reuse the
  same L1 Q slot for every QK matmul in that M loop.
- Make stage-1 score/P UB scratch double-buffered when MTE2 score loads, V
  softmax/cast, and MTE3 P publishes overlap across adjacent N tiles.
- Increase lookahead by deepening score/P/PV mutexes and delayed scalar
  buffers together. In the validated block32 family, the best simulator point
  used four slots for the score/P/PV/expdiff handoffs and delayed `p @ v` from
  `ni - 1` to `ni - 3`. Deeper cross-M prefetching only stayed correct after
  explicit row-state ownership was added.
- Keep final MTE3 writeback ownership in mind: `accum_ub`, `ub_rmax_s`, and
  `ub_rsum_s` cannot be reinitialized for the next M tile until that writeback
  is known complete.

Grouped-Skv / grouped-PV paths reduce PV publication count and can be a real
hardware win. The non-causal `flash_attn_full.py` now uses the v4-style
`GROUP_N=4` schedule; on 910B3 the measured shape `(1,3,2048,4096,128)` improved
from `0.328182 ms` to `0.202377 ms`. The stable ownership details from the v4
experiment are:

- Keep the GM workspaces widened, for example `[GetCubeNum(), slots, 128, 512]`,
  but process score/P in `[ROW_CHUNK,128]` UB slices unless the wide UB transfer
  path has been separately revalidated.
- For grouped PV, put `p_mutex.wait/free`, all `p_ws -> l1p` and `v -> l1v`
  loads, grouped PV matmuls, and the grouped `l0c -> pv_ws` publication inside
  one cube-side `auto_sync()` block. This lets generated events guard MTE2/M/FIX
  as one resource lifetime instead of splitting the L0C producer from the FIX
  publication.
- Use `VcMutex(..., dst_end_pipe=Pipe.MTE2)` for P workspace slots when the
  slot can be freed as soon as `p_ws -> l1p` loads complete. Do not hold that
  slot until the later PV matmul or vec accumulation if no later consumer reads
  from `p_ws`.
- Treat wider or deeper grouped-Skv variants as traffic tradeoffs, not guaranteed
  wins. On real A2 hardware, lower GM/L2 bandwidth can remove simulator-only
  gains from deeper lookahead or wider groups.

### Common pitfalls

- Updating `row_sum` from the half-precision `p_j` instead of the float tile.
- Dividing inside the inner loop instead of after it.
- Forgetting that a non-aligned S2 requires pre-`cmax` masking, not just
  GM-boundary trimming.
- All Section 3 pitfalls apply here too (shared L0C, `ub_to_ub`, VcMutex release timing).

---

## Capacity quick-check (TILE_M = TILE_N = D = 128)

| Buffer | Size | Budget |
|--------|------|--------|
| L1: l1q + l1k DBuff | 128 KB | 512 KB |
| L0C: l0c DBuff | 128 KB | 128 KB (full) |
| UB per sub-block | ~66–80 KB | 192 KB |

Because L0C is full at one `[128, 128]` float tile, both cube stages must share
one physical L0C family (Sections 2–4). This is a capacity constraint, not a
general design preference.

## Do not use when

- Target is a5 — use `l0c_to_ub` + `@vf`; direct UB→L1 helpers such as `ub_to_l1` and `ub_to_l1_nd2nz` are available.
- Kernel is cube-only — use direct `l0c_to_gm_nz2nd` to output.
- Vec-first pipeline (vec → cube only) — use standalone `VcMutex` pattern.

## Source escape

If a required handoff, workspace shape, or event lifetime is not covered, use
`agent/references/evidence-escalation.md`. Isolate one handoff or delayed stage
in a minimal probe and inspect generated event ordering before changing the
whole pipeline.

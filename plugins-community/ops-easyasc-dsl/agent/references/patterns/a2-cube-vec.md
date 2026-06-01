# a2 Cube-to-Vec Pattern (GM Workspace Bridge)

Read this file when writing a cube → vec kernel on a2 (`easyasc.a2`, device `b3`).
On a2, `l0c_to_ub` is not available. The cube output must transit through GM workspace.

## When to use

- Cube computes a matmul tile in L0C
- Vec must postprocess (scale, normalize, exp, cast, etc.) before final writeback
- Target device is a2 (not a5)

## Data flow

```
GM(q,k) → L1 → L0A/L0B → mmad → L0C → GM(workspace) → UB → vec ops → GM(output)
                                    ↑                     ↑
                              cube FIX pipe         vec MTE2 pipe
                                    └── CvMutex ──────┘
```

## Buffer declarations

```python
# GM workspace with pingpong (2 slots)
ws = split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N], name="ws")

# Cube buffers (standard)
l1q = DBuff(DT.half, [TILE_M, TILE_K], Position.L1)
l1k = DBuff(DT.half, [TILE_N, TILE_K], Position.L1)
l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)

# Vec buffers (per sub-block, 192KB each)
ub_data = Tensor(DT.float, [HALF_M, TILE_N], Position.UB)
ub_out  = Tensor(DT.half, [HALF_M, TILE_N], Position.UB)
```

## Synchronization

```python
cvmutex = CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
```

- `src_end_pipe=Pipe.FIX`: cube's last operation is `l0c_to_gm_nz2nd` (FIX pipe)
- `dst_end_pipe=Pipe.MTE2`: vec's first operation is `gm_to_ub_pad` (MTE2 pipe)

This differs from a5's standard `dst_end_pipe=Pipe.V` because a5 uses `l0c_to_ub` → `@vf`.

## Sub-block split

Each cube core has 2 vec sub-blocks, each with independent 192KB UB.
Use `GetSubBlockIdx()` to split the M dimension:

```python
sb = GetSubBlockIdx()
sb_row = Var(sb * HALF_M)

# Cube writes full TILE_M to workspace
ws[cube_idx, slot, 0:TILE_M, 0:TILE_N] <<= l0c[cnt]

# Each sub-block reads its own half
ub_data <<= ws[cube_idx, slot, sb_row:sb_row + HALF_M, 0:TILE_N]

# Each sub-block writes its own half to output
out_row = Var(q_row + sb_row)
output[out_row:out_row + HALF_M, col:col + TILE_N] <<= ub_out
```

## Workspace pingpong

Index the workspace slot with `var_mod(counter, 2)`:

```python
ws_cnt = Var(0)
# inside loop:
ws_slot = var_mod(ws_cnt, 2)
ws[cube_idx, ws_slot, ...] <<= l0c[...]  # cube write
ub_data <<= ws[cube_idx, ws_slot, ...]    # vec read
ws_cnt += 1
```

The CvMutex lock/free cycle ensures the cube does not overwrite a slot
that the vec is still reading from the previous iteration.

## Tail note for workspace slices

When an a2 cube -> vec kernel has `valid_m` / `valid_n` tails, keep the
workspace bridge itself on stable tile shapes whenever possible:

- cube side: prefer writing `0:TILE_M, 0:TILE_N` into workspace after tail
  zero-fill in local buffers
- vec side: prefer reading `row_begin:row_begin + row_count, 0:TILE_N` from
  workspace, then handle `valid_n` with vec-side masking and final GM write
  boundaries

Reason:
- `l0c_to_gm_nz2nd` and `gm_to_ub_pad` infer row stride from the parent GM
  shape, not from a cropped workspace column span
- a workspace slice like `[..., 0:row_count, 0:valid_n]` may therefore be too
  small for the inferred stride even when the logical tail region is correct

This is a workspace-bridge rule, not a general "never use GM tails" rule.
Direct final GM boundaries such as `output[..., 0:valid_n]` still work in the
usual way.

## Complete iteration skeleton

```python
with auto_sync():
    for tile_idx in range(...):
        ws_slot = var_mod(ws_cnt, 2)

        # Cube
        l1q[l1_cnt] <<= q[...]
        l1k[l1_cnt] <<= k[...]
        matmul(l0c[l0c_cnt], l1q[l1_cnt], l1k[l1_cnt], is_init=True)

        cvmutex.lock()
        ws[cube_idx, ws_slot, 0:TILE_M, 0:TILE_N] <<= l0c[l0c_cnt]
        cvmutex.ready()

        # Vec
        cvmutex.wait()
        ub_data <<= ws[cube_idx, ws_slot, sb_row:sb_row + HALF_M, 0:TILE_N]
        # ... vec postprocess ...
        output[...] <<= ub_out
        cvmutex.free()

        l1_cnt += 1; l0c_cnt += 1; ws_cnt += 1
```

## Capacity quick-check (TILE_M=128, TILE_N=128, D=128)

| Buffer | Size | Budget |
|--------|------|--------|
| L1: l1q + l1k DBuff | 128 KB | 512 KB ✓ |
| L0C: l0c DBuff | 128 KB | 128 KB ✓ |
| L0A (inner) | 64 KB | 64 KB ✓ |
| L0B (inner) | 64 KB | 64 KB ✓ |
| UB per sub-block | ~66 KB | 192 KB ✓ |

## Do not copy when

- Target is a5 — use `l0c_to_ub` + `@vf` instead
- Kernel is cube-only — use direct `l0c_to_gm_nz2nd` to output (no workspace needed)
- Vec preprocess is needed (vec → cube) — use `VcMutex` pattern instead

## Files to study

- `agent/example/kernels/a2/flash_attn_score.py` — complete working example
- `agent/example/kernels/a2/qk_matmul_batched.py` — cube-only a2 baseline (no vec)
- `agent/references/constraints/a2-device.md` — a2-specific hardware differences
- `agent/references/constraints/vec-stride.md` — continuous vs sliced vec operations

# Kernel Index

Use this file to filter down to ≤3 candidate kernels before opening `kernel-catalog.md`.
Fastest path for agents:
- `conda run -n torch210npu python agent/scripts/select_kernel_example.py --query "<formula or task>" --topology '<topology>' --limit 3 --catalog`
- use this markdown table when you want a manual filter or the tool query is still too vague

Each row gives device, topology, path, and a one-line formula hint.
For `study_for` and `do_not_copy_when`, read the matching entry in `kernel-catalog.md`.
For machine-readable use, see `agent/index/kernels.json`.

## How to use

1. Pick the rows whose **device** matches your target (a2 or a5).
2. Narrow by **topology**: cube-only / cube -> vec / vec -> cube / vec -> cube -> vec / vec -> cube -> vec -> cube -> vec / cube -> vec -> cube / cube -> vec -> cube -> vec / vec-only / micro-only.
3. Narrow by **formula shape**: pure matmul vs with postprocess, with reduction, with softmax, with online-accumulation, quantized, causal, etc.
4. For each remaining candidate path, jump straight into `kernel-catalog.md` with Grep on the filename (e.g. `^### .kernels/a2/flash_attn_full\.py.`) — do not scroll. Read only that one entry, and stop after `study_for` / `do_not_copy_when` unless you still need deeper notes.
5. Open the source file only after the catalog `study_for` / `do_not_copy_when` confirms the candidate.

## Vec-only and micro references

| Device | Topology   | Path                                              | Formula hint                                                                 |
|--------|------------|---------------------------------------------------|------------------------------------------------------------------------------|
| a2     | vec-only   | `agent/example/kernels/a2/to_hif8_torch.py`                     | `to_hif8_torch(x)` — emulated hif8 round, saturation sentinels               |
| a2     | vec-only   | `agent/example/kernels/a2/sort_rows.py`                         | per-row `torch.sort(x, dim=-1)` for `[ROWS=40, COLS=4096]`                   |
| a5     | vec-only   | `agent/example/kernels/a5/chunk_row_cumsum.py`                  | chunked row-recursive cumsum                                                 |
| a5     | vec-only   | `agent/example/kernels/a5/recurrent_state_attn_vec.py`          | recurrent attention-state update, `D=128`                                    |
| a5     | vec-only   | `agent/example/kernels/a5/vec_unaligned_gm_to_ub_pad.py`        | `exp(x) + 2` on padded unaligned GM width                                    |
| a5     | micro-only | `agent/example/kernels/a5/micro_cast_fp8_pack4_dual.py`         | `src.to(float8_e5m2)` via `micro`                                            |

## Cube-only

| Device | Topology   | Path                                              | Formula hint                                                         |
|--------|------------|---------------------------------------------------|----------------------------------------------------------------------|
| a5     | cube-only  | `agent/example/kernels/a5/matmul_float_mmad.py`                 | `z = x @ y.t()` — shortest cube baseline                             |
| a5     | cube-only  | `agent/example/kernels/a5/matmul_e5m2_shortcut.py`              | `z = x.float() @ y.float().t()` with fp8 inputs                      |
| a5     | cube-only  | `agent/example/kernels/a5/matmul_kmkn_fp32_out.py`              | `z = x.float().t() @ y.float()` (KM @ KN -> MN)                      |
| a5     | cube-only  | `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`         | `z = x @ y.t()` with `splitn` and 2D core grid                       |
| a5     | cube-only  | `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`         | `z = x @ y.t()` with `splitk` for large-K                            |
| a2     | cube-only  | `agent/example/kernels/a2/qk_matmul_batched.py`                 | `qk = q.float() @ k.float().t()` with batched BH flatten             |
| a2     | cube-only  | `agent/example/kernels/a2/attn_backward_dense_stage1_tail_dbuf.py` | `qk = q.float() @ k.float().t()` — DBuff tail variant             |

## Cube -> vec (postprocess on a5)

| Device | Topology      | Path                                                  | Formula hint                                                   |
|--------|---------------|-------------------------------------------------------|----------------------------------------------------------------|
| a5     | cube -> vec   | `agent/example/kernels/a5/basic_cube_vec_mix.py`                    | `z = abs(x @ y.t()) + 1.0` — smallest mixed baseline           |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`        | `((x @ y) + 10.2).half()` — bias + half output via `@vf`       |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_rowwise_norm.py`                   | `z = (x @ y.t()) / row_sum(x @ y.t())`                         |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py`          | same as rowwise_norm, larger N/K                               |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_rowwise_l2_norm.py`                | L2-normalized matmul output                                    |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_chunk_absmax_norm128.py`           | per-row absmax normalize over 128-column chunks                |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`        | `x.float().t() @ y.float()` with blockwise-128 quant           |
| a5     | cube -> vec   | `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`        | `x @ y.t() + 1.0` with `splitk`                                |
| a5     | cube -> vec (dual-output atomic) | `agent/example/kernels/a5/cube_vec_atomic_add_two_outputs.py` | `out_cube += x @ y.t()` with atomics, two sinks    |

## Vec -> cube (preprocess on a5)

| Device | Topology      | Path                                          | Formula hint                                                  |
|--------|---------------|-----------------------------------------------|---------------------------------------------------------------|
| a5     | vec -> cube   | `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`      | `z = abs(x).sqrt() @ y.t()`                                   |
| a5     | vec -> cube   | `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul_nz.py`   | same as above, NZ-published                                   |
| a5     | vec -> cube   | `agent/example/kernels/a5/recompute_wu_cube_vec.py`         | `k_cumdecay = attn @ (k_beta * decay_exp)`                    |

## Vec -> cube -> vec fusion (a5)

| Device | Topology               | Path                                                    | Formula hint                                  |
|--------|------------------------|---------------------------------------------------------|-----------------------------------------------|
| a5     | vec -> cube -> vec     | `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`     | `abs((x*2).half() @ y.t()) + 1.0`             |

## Vec -> cube -> vec -> cube -> vec state bridge (a5)

| Device | Topology                         | Path                                           | Formula hint                                                                    |
|--------|----------------------------------|------------------------------------------------|---------------------------------------------------------------------------------|
| a5     | vec -> cube -> vec -> cube -> vec | `agent/example/kernels/a5/delta_h_state_bridge_v1_c8.py`    | aligned `delta_h` baseline with persistent state snapshots and delayed state update |
| a5     | vec -> cube -> vec -> cube -> vec | `agent/example/kernels/a5/delta_h_psudo_state_bridge_c8.py` | pseudo-reference comparison on the same stable state-bridge schedule            |

## Cube -> vec -> cube -> vec lookahead (a5, MLA / MHA style)

| Device | Topology                  | Path                                            | Formula hint                                                              |
|--------|---------------------------|-------------------------------------------------|---------------------------------------------------------------------------|
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/test_mla_entire.py`                | streamed MLA: score, softmax, delayed `p @ k_nope`, final normalize      |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/mha_ifa.py`                        | streamed single-row `softmax(q @ k.t()) @ v`                             |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/mha_ifa_256.py`                    | same, `BASES=256`                                                         |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/mha_ifa_fp8_scale_256.py`          | fp8 q/k/v, fp8-scaled p tiles, `BASES=256`                                |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/flash_attn_full_fp8_causal.py`    | multi-row causal full attention, fp8 q/k/v + fp8 `p` tiles, tail-safe `S1/S2` |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/mha_ifa_nz.py`                     | same, NZ-published probability tiles                                      |
| a5     | cube -> vec -> cube -> vec | `agent/example/kernels/a5/mha_ifa_nz_256.py`                 | same, `BASES=256` + NZ                                                    |

## a2 mixed-pipeline (GM workspace bridges)

| Device | Topology                                                                                            | Path                                                                  | Formula hint                                                              |
|--------|-----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|---------------------------------------------------------------------------|
| a2     | cube -> vec (single GM bridge)                                                                     | `agent/example/kernels/a2/attn_backward_dense_stage12_tail.py`                      | `qk = q.float() @ k.float().t()` stage-1+2 with tail                      |
| a2     | cube -> vec (single GM bridge)                                                                     | `agent/example/kernels/a2/flash_attn_score.py`                                      | `exp(Q @ K^T / sqrt(D) - row_max)` cast to half                           |
| a2     | cube -> vec (single GM bridge, running max)                                                        | `agent/example/kernels/a2/flash_attn_score_iter.py`                                 | same, with cross-tile running row_max                                     |
| a2     | cube -> vec -> cube                                                                                 | `agent/example/kernels/a2/attn_backward_dense_total_tail.py`                        | dense attn-backward with tail                                             |
| a2     | cube -> vec -> cube                                                                                 | `agent/example/kernels/a2/attn_backward_dense_total_tail_causal.py`                 | same, causal masking                                                      |
| a2     | cube -> vec -> cube                                                                                 | `agent/example/kernels/a2/attn_backward_dense_total_tail_causal_hif8.py`            | same, hif8 probability path                                               |
| a2     | cube -> vec -> cube (double GM bridge, one-tile lookahead)                                         | `agent/example/kernels/a2/flash_attn_score_pv.py`                                   | `score_j = q @ k_j.t() * scale` with delayed `p @ v`                      |
| a2     | cube -> vec -> cube -> vec (triple GM bridge, one-tile lookahead)                                  | `agent/example/kernels/a2/flash_attn_unnorm.py`                                     | unnormalized flash-attn numerator                                         |
| a2     | cube -> vec -> cube -> vec (triple GM bridge, final vec divide)                                    | `agent/example/kernels/a2/flash_attn_full.py`                                       | full flash-attn with running sum and final divide                         |
| a2     | cube -> vec -> cube -> vec (triple GM bridge, hif8 stage-1 vec path)                               | `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`                               | same math as `flash_attn_full.py`, hif8 probability                       |
| a2     | cube -> vec -> cube -> vec (hif8 + diagonal causal mask, shared slot buffer)                       | `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py`                        | same as hif8 variant, causal + future-tile skip                           |
| a2     | cube -> vec -> cube -> vec (half probability, block-32 diagonal causal)                            | `agent/example/kernels/a2/flash_attn_full_pj_half_block32_causal.py`                | same math, half `p`, block-32 causal                                      |
| a2     | cube -> vec -> cube -> vec (shared vec-side slot buffer for score and pv tiles)                    | `agent/example/kernels/a2/flash_attn_full_pj_hif8_commonub.py`                      | same as hif8 variant with shared UB slot                                  |

## Going deeper

- For `study_for` / `do_not_copy_when` detail on any single entry: open `agent/references/examples/kernel-catalog.md` at the matching `###` heading.
- For programmatic filtering: `agent/index/kernels.json`.

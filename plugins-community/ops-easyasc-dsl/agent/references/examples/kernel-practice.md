# Kernel Practice Catalog

Use this file as a curated set of practice problems for building kernel
authoring experience in this repository. Unlike
`agent/references/examples/kernel-catalog.md`, which indexes **finished**
kernels for study, every entry here is a *problem to implement from scratch*.

How to use this catalog:

1. Pick an exercise that matches the next skill you want to drill. Tiers go
   from simple (vec-only) to complex (lookahead fusion).
2. Read `agent/playbooks/kernel-authoring.md` first if you have not already,
   then read only the focused constraint files relevant to the exercise.
3. Write the kernel cold — do not open the `study_after` reference until you
   have a working draft. The reference is for comparison after the attempt,
   not for copy-paste during it.
4. Place attempts under `agent/example/kernels/practice/<device>/<name>.py` (a new sibling
   of `agent/example/kernels/a2/` and `agent/example/kernels/a5/`). Keeping practice attempts in a
   separate directory keeps `agent/example/kernels/a2/` and `agent/example/kernels/a5/` as the canonical
   reference set.

## Per-entry fields

- `formula`: the PyTorch reference contract the kernel must match
- `target_device`: `a2`, `a5`, or `either`
- `topology`: expected staged pipeline shape
- `practices`: the specific skills, constraints, or idioms this exercise
  drills
- `study_after`: the closest finished kernel in `kernel-catalog.md` to
  compare against once you have a working draft

## Tier 1 — Vec-only warm-ups

### `practice_elementwise_scale_bias`
- formula: `y = x * 2 + 1` for `x:[M, N]`, float32
- target_device: a5 (port to a2 as a follow-up)
- topology: `vec-only`
- practices:
  - the minimal `GM -> UB -> @vf() -> GM` skeleton
  - basic per-core tile/split selection
  - aligning a `[M, N]` shape to UB-friendly subblocks
- study_after: `agent/example/kernels/a2/to_hif8_torch.py` (for the bare vec-only structure
  on a2) or `agent/example/kernels/a5/micro_cast_fp8_pack4_dual.py` (for a reference a5 cast
  flow; for a pure `@vf()` downcast see `@vf`-based vec baselines under
  `agent/example/kernels/a5/`)

### `practice_sigmoid`
- formula: `y = 1 / (1 + exp(-x))` for `x:[M, N]`, float32 in/out
- target_device: a5
- topology: `vec-only`
- practices:
  - vec-side `exp` and reciprocal usage
  - keeping float precision through the chain
  - validating against `torch.sigmoid` with a tight `atol`
- study_after: `agent/example/kernels/a2/to_hif8_torch.py` (vec-only bit/numeric reference,
  not for copy-paste)

### `practice_row_sum`
- formula: `y = x.sum(dim=-1, keepdim=True)` for `x:[M, N]`, float32
- target_device: either (do it once on each device for contrast)
- topology: `vec-only`
- practices:
  - row-wise reduction idiom (`cadd -> brcb` on a2, `cadd_vf` on a5)
  - distinguishing continuous vec ops from sliced row ops (a2)
  - `[M, 1]` vs `[M, 8]` scalar broadcast layout
- study_after: read `agent/references/constraints/reduction.md` and
  `agent/references/constraints/vec-reduction-a2.md` first, then compare
  against the row-sum step inside `agent/example/kernels/a5/matmul_rowwise_norm.py`

### `practice_softmax_single_tile`
- formula: `y = softmax(x, dim=-1)` for `x:[M, N]` that fits in one UB tile
- target_device: a5 first, then a2
- topology: `vec-only`
- practices:
  - numerically stable `max -> sub -> exp -> sum -> divide` chain
  - chaining a row reduction immediately after another row reduction
  - keeping the `max` and `sum` registers alive across compute steps
- study_after: the inner softmax steps inside
  `agent/example/kernels/a2/flash_attn_score.py` (extract only the score block, ignore the
  cube stage and GM bridge for now)

## Tier 2 — Cube-only baselines

### `practice_matmul_basic`
- formula: `z = x @ y.t()` with aligned `M, N, K`
- target_device: a5
- topology: `cube-only`
- practices:
  - the shortest end-to-end cube pipeline
  - `CvMutex` ownership and `l1_cnt` lifetime
  - sanity-checking the simulator validation story
- study_after: `agent/example/kernels/a5/matmul_float_mmad.py`

### `practice_matmul_kmkn`
- formula: `z = x.t() @ y` with `x:[K, M]`, `y:[K, N]` → `z:[M, N]`
- target_device: a5
- topology: `cube-only`
- practices:
  - transpose-at-call-site with `KM @ KN -> MN` layout
  - explicit `%16` shape guards and `shape_bindings`
  - reasoning about layout without forcing explicit `m/n/k`
- study_after: `agent/example/kernels/a5/matmul_kmkn_fp32_out.py`

### `practice_matmul_splitk_large_k`
- formula: `z = x @ y.t()` with `K` large enough that the inner stage cannot
  hold a full `K` slice
- target_device: a5
- topology: `cube-only`
- practices:
  - 2D core split selection with `splitk`
  - keeping outer `TILE_K` while legalizing inner stage with `SPLIT_K`
  - split-K accumulation ownership in `L0C`
- study_after: `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`

## Tier 3 — Cube → vec postprocess

### `practice_matmul_bias_gelu`
- formula: `z = gelu(x.float() @ y.float().t() + bias)` with `bias:[N]`
- target_device: a5
- topology: `cube -> vec`
- practices:
  - cube → vec handoff via `CvMutex`
  - vec-side `tanh`-based GELU expansion
  - broadcasting a `[N]` bias across the half-row writeback
- study_after: `agent/example/kernels/a5/basic_cube_vec_mix.py` for the bare mix structure;
  `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py` for the FIX → V handoff

### `practice_matmul_rowwise_softmax`
- formula: `z = softmax(x.float() @ y.float().t(), dim=-1)` (single-pass when
  the row fits in one tile)
- target_device: a5
- topology: `cube -> vec`
- practices:
  - reduction *after* a cube stage rather than before
  - chained `cmax -> sub -> exp -> cadd -> div` row pipeline
  - ordering the full-row writeback against the running scalar state
- study_after: `agent/example/kernels/a5/matmul_rowwise_norm.py` for the single-pass form;
  `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py` for the two-pass form

### `practice_matmul_rmsnorm`
- formula:
  - `z_tmp = x.float() @ y.float().t()`
  - `z = z_tmp * rsqrt(mean(z_tmp ** 2, dim=-1, keepdim=True) + eps)`
- target_device: a5
- topology: `cube -> vec`
- practices:
  - two-pass per-row statistic on top of a matmul
  - `square -> cadd -> rsqrt -> mul` ordering and lifetime
  - holding the row statistic across multiple `N` tiles when needed
- study_after: `agent/example/kernels/a5/matmul_rowwise_l2_norm.py`

### `practice_matmul_blockwise_quant`
- formula:
  - `z_tmp = x.float() @ y.float().t()`
  - `scale = absmax(z_tmp, block=128) / 224`
  - `z = (z_tmp / scale).to(float8_e5m2)`
- target_device: a5
- topology: `cube -> vec`
- practices:
  - blockwise `abs -> cmax -> dup -> div` pattern
  - generating and writing back the per-block scale alongside `z`
  - vec-side fp8 cast and optional `pack4()` fallback
- study_after: `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`

## Tier 4 — Vec → cube preprocess

### `practice_relu_then_matmul`
- formula: `z = relu(x).half().float() @ y.float().t()`
- target_device: a5
- topology: `vec -> cube`
- practices:
  - vec → cube handoff via `VcMutex`
  - elementwise vec preprocess feeding a cube consumer
  - dtype downcast/upcast across the bridge without losing alignment
- study_after: `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py` (ND publish) before
  trying `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul_nz.py` (NZ publish)

## Tier 5 — Lookahead and fusion

### `practice_fused_vec_cube_vec`
- formula: `z = abs((x * 2).half().float() @ y.float().t()) + 1`
- target_device: a5
- topology: `vec -> cube -> vec`
- practices:
  - one fused kernel that owns both `VcMutex` and `CvMutex`
  - separate counters for the preprocess and postprocess sides
  - validating each stage independently before fusing
- study_after: `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`

### `practice_flash_attn_decode`
- formula: `out = softmax(q @ k.t() / sqrt(D)) @ v` with `L=1`, `D=128`,
  half inputs, float accumulators, normalized output
- target_device: a5 (a2 variant is exercise 15)
- topology: `cube -> vec -> cube -> vec`
- practices:
  - one-tile lookahead schedule with warmup and drain
  - delayed-consumer counters (`stage1_cnt` vs `stage2_cnt`)
  - running `row_max` and `row_sum` kept resident in vec UB
  - on-chip delayed reuse instead of a forced GM round-trip
- study_after: `agent/example/kernels/a5/mha_ifa.py`

### `practice_flash_attn_decode_causal_a2`
- formula: same math as `practice_flash_attn_decode`, but on a2 with the
  triple GM workspace bridge and left-up causal masking
  (`k_pos <= q_pos`)
- target_device: a2
- topology: `cube -> vec -> cube -> vec`
- practices:
  - triple GM bridge: float score, half probability, and float `pv`
  - `CvMutex -> VcMutex -> CvMutex` ownership chain on a2
  - score-domain causal masking *before* `cmax` / `rowmax`
  - `active_tiles_n = Min(tiles_n, lmt + 1)` future-tile skip
  - prebuilt packed-bit diagonal mask reused per subblock
- study_after: `agent/example/kernels/a2/flash_attn_full.py` for the non-causal version,
  then `agent/example/kernels/a2/flash_attn_full_pj_half_block32_causal.py` or
  `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py` for the causal extension

## Fast selection hints

When you only need one starting point per skill:

- bare a5 vec-only authoring -> `practice_elementwise_scale_bias`
- row reduction muscle memory -> `practice_row_sum`
- numerically stable softmax -> `practice_softmax_single_tile`
- bare cube authoring -> `practice_matmul_basic`
- transpose-layout matmul -> `practice_matmul_kmkn`
- large-`K` cube strategy -> `practice_matmul_splitk_large_k`
- cube → vec handoff -> `practice_matmul_bias_gelu`
- post-matmul rowwise reduction -> `practice_matmul_rowwise_softmax`
- post-matmul L2-style normalization -> `practice_matmul_rmsnorm`
- post-matmul fp8 quantization -> `practice_matmul_blockwise_quant`
- vec → cube handoff -> `practice_relu_then_matmul`
- fused vec → cube → vec -> `practice_fused_vec_cube_vec`
- decode-style flash attention on a5 -> `practice_flash_attn_decode`
- decode-style causal flash attention on a2 ->
  `practice_flash_attn_decode_causal_a2`

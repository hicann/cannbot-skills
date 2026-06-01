# a5 Device Constraints

Read this file when writing a kernel targeting a5 (`easyasc.a5`, device `950`) and the kernel has any vec-side stage.
Do not use it as a substitute for the general kernel-authoring playbook.

## Goal

Capture the stable a5 vec-side authoring surface so that:
- a2-style direct vec-body patterns are not copied into a5 kernels
- vec-side work starts on the supported authoring surfaces
- `easyasc.a5` import breadth is not mistaken for the stable kernel-writing contract

## 1. Stable a5 vec-side authoring rule

For a5, vec-side work should be authored through:
- `@vf()` helpers for ordinary vec preprocess / postprocess
- `micro` ops inside `@vf()` when register-level control is required
- sort-family ops such as `sort32`, `mergesort4`, and `mergesort_2seq` when the kernel genuinely needs sort behavior
- `ub_to_ub` for UB-local copies or layout-preserving handoff steps

Do **not** write generic a2-style vec UB ops directly in the a5 kernel body.
If the step is elementwise, row-wise, reduction, normalization, or cast-oriented on a5,
move it into `@vf()` first and only drop to `micro` when `@vf()` alone is not enough.

Important note:
- the raw `easyasc.a5` export surface is wider than this stable authoring rule
- treat the authoring rule above as the repository contract for new a5 kernels

## 2. Contrast with a2

- a2 does **not** support `@vf()`
- a2 does **not** support `micro`
- a2 vec work is written directly in the kernel body on UB tensors
- do not mirror an a2 pure-vec kernel body into a5, or an a5 `@vf()` flow into a2

## 3. Implications for common topologies

Stable a5 forms:
- cube -> vec: `GM -> L1 -> L0 -> L0C -> UB -> @vf() -> GM`
- vec -> cube: `GM -> UB -> @vf() -> UB -> L1 -> L0 -> L0C -> GM`
- vec-only transform: `GM -> UB -> @vf()` or `GM -> UB -> @vf() + micro -> GM`
- UB-local republish / copy: `ub_to_ub` may stay in the kernel body if it is truly just the copy step

Practical rule:
- if you are about to call ordinary vec math on an a5 UB tensor from the kernel body, stop and move that logic into `@vf()`

## 3a. Cube-side matmul dependency reuse rule

When a later cube matmul depends on the result of an earlier cube matmul, check first whether the
dependency can stay on the cube-side path:
- producer: `mmad -> L0C`
- republish: `l0c_to_l1`
- consumer: later `l1_to_l0 -> mmad`

Prefer that direct `L0C -> L1` route when:
- the intermediate value is only needed by a later cube-side matmul
- no vec-side transform is required on the intermediate value before reuse

Avoid the detour:
- `L0C -> UB -> L1`

unless the UB hop is semantically required for a real vec-side stage such as:
- cast / normalization / elementwise transform in `@vf()`
- a cube -> vec handoff that genuinely changes ownership to the vec lane

Reason:
- `l0c_to_l1` already gives you the FIX-side republish path for this dependency
- the UB detour adds traffic, adds synchronization surface, and makes the kernel easier to
  overcomplicate without adding capability

Practical debugging hint:
- if you find yourself moving a pure matmul dependency through UB only so a later matmul can read
  it back, stop and re-check whether `l0c_to_l1` already expresses the intended dependence

## 4. When to use `micro`

Use `micro` on a5 when the vec stage needs register-level behavior such as:
- explicit fp8 cast control
- `pack4()` / sparse-lane squeeze patterns
- explicit mask or cast-config handling
- custom register reductions or packing not expressible cleanly as plain `Tensor <<= Reg/RegList`

Prefer plain `@vf()` first when it already matches the contract.
For example, a `Reg` / `RegList` loaded in `@vf()` and written back with `dst[...] <<= regs`
is usually simpler than dropping to explicit `micro cast + pack4`.

Another stable case that should stay in `@vf()`:
- row-recursive vec kernels where each output row depends on the previous output row
- example shape: load one GM chunk as `[chunk_size, H]`, then compute
  - `y[0, :] = x[0, :]`
  - `y[i, :] = x[i, :] + y[i - 1, :]`
- on a5, keep that recurrence in `@vf()` with `Reg` / `RegList` slices over the row width
- do **not** reach for `cpadd` or custom `micro` just because the math is cumulative; `cpadd` is pair-wise add, not row-prefix recurrence
- only drop to `micro` if the recurrence itself needs per-lane scan behavior inside one row rather than previous-row carry

## Files to study

- `agent/example/kernels/a5/basic_cube_vec_mix.py`
- `agent/example/kernels/a5/chunk_row_cumsum.py`
- `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`
- `agent/example/kernels/a5/micro_cast_fp8_pack4_dual.py`
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`

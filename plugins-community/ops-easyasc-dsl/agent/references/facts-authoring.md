# Authoring Facts

Use this file for repository-wide hard rules, DBuff quick-models, and a2-specific bridge reminders.
Use constraint files for the why; use this page for the exact rule.

#### L1 matmul contract (read first — easy to mistake)

`matmul(dst, l1a, l1b, m=M, n=N, k=K)` computes `dst[M,N] = l1a @ l1b**T`,
or equivalently in shape convention: **l1a [M,K] x l1b [N,K]** (the second
input is already transposed in storage). Both operands live in L1 and use a
supported matching Cube dtype; normal paths include fp16, bf16, fp32, and int8,
with additional device-specific packed/FP8 paths. Accumulation is fp32 except
for integer paths that require int32. Add `.T` on a call-site argument only to
reinterpret a physical `[K,N]` source as the `[N,K]` that matmul expects.
Example: `k_beta @ key.T` with both in `[L,D]` bf16 is written as
`matmul(l0c_pp, l1_k_beta, l1_key, m=L, n=L, k=D)` — no `.T` because `[L,D]`
already matches the `[N,K]` layout of the second operand.
(source: `easyasc/shortcuts/matmul.py` docstring and `:103-316`)

## Authoring hard rules

- a5 performance hierarchy: use cube `matmul` for dense contractions and
  GEMM-like reductions, and use `@vf` / `micro` for regular elementwise,
  row-wise, masking, scan, or reduction work around those matmuls. Treat
  `@simt` as an exception path, not a correctness baseline. Use SIMT only for
  irregular control flow, data-dependent gather/scatter, atomics, targeted
  runtime probes, or logic that cannot be expressed as VF + matmul without
  making the kernel more fragile. If a delivered a5 kernel uses SIMT, document
  the specific irregularity that forced it.
- Normal matmul has no generic hard minimum of 32 for `splitk` or `splitn`.
  The estimator's candidates at or above 32 are tuning heuristics. MX
  `matmul_mx` separately requires `splitk` to be a multiple of 64 and `splitn`
  to be a multiple of 16; transposed-B MX split-N requires 32-column alignment.
- matmul `L0C` destination row offset defaults to `0`; always use unsliced DBuff slot
  (source: `easyasc/stub_functions/cube.py:1095` `dst_row0 = dst.offset[0] ...`)
- a `@kernel` body is re-executed from a rewritten AST (that is how a native
  `if` / `for` over a `Var` becomes DSL control flow), which drops the closure
  cell. A factory that builds N kernel variants from one decorated inner
  function therefore raises `NameError` on the captured flag at trace time.
  Moving the body into a plain helper does not help either: the rewrite covers
  only the decorated function, so DSL conditions in the helper stay native
  Python. Prefer folding the variation into a runtime scalar — a memory-layout
  switch, for instance, becomes two row strides passed to
  `gm_to_ub_pad(..., src_stride_element=...)` /
  `ub_to_gm_pad(..., dst_stride_element=...)`, which keeps one kernel, one
  template per dtype, and costs no extra DMA bursts.
- matmul accumulation `L0C` dtype is typically `float`; `mmad` enforces `dst` at `L0C`
  (source: `easyasc/stub_functions/cube.py:1014` signature, `:1018` L0C check)
- GM/UB/L1 burst datamove helpers do not perform dtype conversion; `gm_to_ub_pad`,
  `ub_to_gm_pad`, and `gm_to_l1_nd2nz` require `dst.dtype == src.dtype`
  (source: `easyasc/stub_functions/vec/datamove.py:84,147`,
  `easyasc/stub_functions/cube.py:490`)
- a2 `DT.int4` matmul uses `DT.int` packed carrier tensors only: each int32
  carrier holds 8 signed int4 values, `reinterpret(DT.int4)` marks local L1/L0
  carrier views for `matmul`/`mmad`, and `k` / `splitk` are logical int4 lengths,
  not carrier-column counts. The accumulator/output tile is `DT.int` L0C/GM.
  See `doc/topics/int4.md`.
- for compact small-M L0C outputs, keep the physical destination M and any
  follow-up `M_src` compact (for example `M=5` -> physical `16`, `M_src=16`);
  do not use the reduction height or a large staging height as the L0C source stride
  (see `agent/references/constraints/tiling.md` -> "compact small-M outputs")
- `l0c_to_ub(..., DualMode.SINGLE)` writes only the selected vector subblock;
  issue both `sub_block_id=0` and `sub_block_id=1` when both vec subblocks will
  read or write back the result, or gate the consumer/writeback to the one
  initialized subblock
- a5-style compact cube -> vec half-row writeback split:
  ```text
  half_rows  = CeilDiv(valid_m, 2)
  row_begin  = GetSubBlockIdx() * half_rows
  row_end    = Min(row_begin + half_rows, valid_m)
  row_count  = row_end - row_begin
  ```
  (source: `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py:62-66`)
- `GetSubBlockIdx()` is vector-side control. Use it to split vector work and
  vector-owned UB/GM slices. Do not use it to guard cube/MTE2 work such as
  `gm_to_l1_nd2nz`, cube-side full-tile L1 loads, matmul setup, or cube-side
  `l0c_to_ub` source selection. For cube-side split-mode transfers, pass the
  legal constant sub-block id for the cube source layout (often `0`) and split
  the later vec consumers with `GetSubBlockIdx()`.
- a2 attention-style workspace bridge split is fixed physical sub-block ownership:
  ```text
  sb_row        = GetSubBlockIdx() * HALF_M
  local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))
  ```
  Do not replace this with the a5 compact `CeilDiv(valid_m, 2)` split.
  (source: `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py:212,228`)
- local buffers stay full-tile sized; only GM read/write uses `valid_m` / `valid_n`
- different loop-owned lifetimes must use different counters; same-pair operands may share one
- dynamic `DBuff` / `TBuff` / `QBuff` indices wrap by physical slot count; monotonic
  tile/chunk counters are fine, but wrap-around slot reuse still needs the lifetime
  protection described in `agent/references/constraints/sync.md`
  (source: `easyasc/simulator/pipe.py:1454-1465` `_handle_get_buf`)
- for any local-source lifetime or slot-family change, validate a shape that
  makes the same active core reuse the slot family across more than one outer
  work item. One-output smoke checks and one-work-item-per-core traces can miss
  delayed MTE3/FIX reads and premature mutex/free tokens.
- if one local `Tensor` has two overlapping live roles, lift it to a slot family
  (`DBuff`, `TBuff`, or `QBuff`) and use distinct slot views for the roles. A
  common case is a UB tensor used as an MTE3/GM-write source and then reused as
  scratch in the same `auto_sync()` region; source order or a nearby `bar_all()`
  does not prove the write source has retired before the scratch overwrite. Use
  `DBuff` for two live roles, `TBuff` for three, and match any handoff mutex
  depth only when the handoff itself rotates across those slots. The complete
  composition and failure signatures live in
  `agent/references/patterns/buffer-slot-lifetime.md`.
- `CvMutex` / `VcMutex` `depth` = real slot count: `Tensor`=1, `DBuff`=2, `TBuff`=3, `QBuff`=4
  (source: `easyasc/utils/Tensor.py` — `DBuff` line 730, `TBuff` line 838, `QBuff` line 934;
  `depth` is materialized in the auto-emitted prologue/epilogue at `easyasc/kernelbase/kernelbase.py:258-278`, not in the kernel body — that is why it looks orphaned)
- `Var(existing_var)` creates a fresh runtime scalar alias; it does not rename or
  mutate the source Var. Use it when a loop Var needs a named alias for slot
  indexing or generated-code readability. **It is not a snapshot.** Inside a
  `@vf` the C++ backend may fold the alias into its uses instead of emitting a
  declaration, so `saved = Var(q)` followed by an update to `q` leaves `saved`
  reading the *new* value. The simulator does snapshot it, so the mistake passes
  every simulator case and only fails on board. To capture a value before it
  moves, declare the destination once outside the loop and assign with `<<=`,
  which always emits a real `var_assign`:

  ```python
  saved = Var(0)          # outside the loop
  ...
  saved <<= q             # inside: emits `saved = q;`
  q <<= q + step
  ```
- `with atomic_add():` narrows the GM-store critical section when accumulating
- `GMTensor` subscript accepts **at most 2 slice dimensions** (written with
  `start:end` syntax). Scalar indices (a single `Var` or `int`, not a slice) do
  **not** count toward this limit. For a 3-D tensor `[B, L, D]`, write
  `t[batch_idx, 0:L, 0:D]` (2 slices + 1 scalar), not
  `t[batch_idx:batch_idx+1, 0:L, 0:D]` (3 slices).
  (source: `easyasc/utils/Tensor.py` — `GMTensor.__getitem__` line 1064, `slice_count` logic lines 1092-1103)
- a2 absent: `@vf`, `Reg`/`RegList`/`MaskReg`, `l0c_to_ub`, `ub_to_l1`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`,
  `micro` (source: `easyasc/a2.py` export list); sort-family IS present in a2
- a2: `l0c_to_l1` does not support `float` destination on b* devices
  (source: `easyasc/stub_functions/cube.py:1297`)
- a5: `@vf`, `micro`, `ub_to_ub`, `ub_to_l1`/`ub_to_l1_nd2nz`/`ub_to_l1_nz`,
  `Reg`/`RegList`/`MaskReg`, and sort-family ops are exported
  (source: `easyasc/a5.py:4` `vf`, `:23` `Reg`/`RegList`/`MaskReg`, `:68-70` `ub_to_ub`/`ub_to_l1*`/sort, `:73` `micro`)
- a5 `@vf`: hardware can overlap VF store/load/compute streams; when a `reg_to_ub*` store is followed
  by another store to the same UB region, use `vf_barrier(VfPipe.STORE, VfPipe.STORE)`, and when a later
  `ub_to_reg*` load depends on a prior store, use `vf_barrier(VfPipe.STORE, VfPipe.LOAD)`. The simulator
  treats this as no-op because micro execution is serial; missing barriers are surfaced by static micro
  instruction warnings. Avoid open-ended row slices such as `out[0:, ...]` in
  VF store/load regions; use explicit row slices so the static checker can see
  the intended UB footprint.
- a5 `@vf` performance loops must use bounded `unroll()` on a small, fixed,
  independent row/chunk axis when that is needed to expose vector ILP.  Order
  the generated work by dependency stage across the unrolled lanes (for
  example, load several independent rows, then issue their FMAs, then their
  dependent divides/stores); a per-lane `load -> FMA -> DIV -> store` chain can
  serialize on hardware even when it has fewer instructions.  Do not maximize
  the factor blindly: every unrolled lane increases live `Reg`/`RegList`
  pressure, and batching too much work can delay a downstream publication.
  Inspect the generated VF header, compare factors such as 2 and 4 under the
  same trace window, and require a real-hardware timing before retaining the
  expansion.  CANNSIM instruction deletion alone is not proof that the chosen
  unroll/order is faster.
- a5 `@vf` scan/cumsum recurrence: keep the loop-carried accumulator in
  `Reg`/`RegList`, not in a UB round trip. Do not use one UB tensor as both the
  source of per-element base values and the destination of the
  prefix/reverse-prefix result inside the same VF loop, even with
  `vf_barrier(VfPipe.STORE, VfPipe.LOAD)`. Also do not treat separate
  source/output UB staging as generally equivalent to direct register
  accumulation: the hardware probe saw that path fail on random inputs. If a
  base vector truly must be staged in UB before the scan, use separate source
  and destination UB tensors, then validate that exact generated code on
  hardware. The retained stable conclusion is to keep recurrence state in
  registers and to mark alternative UB recurrence layouts as requiring real
  hardware validation.
- `vec_scope()` / `cube_scope()` are for side-specific assignments or calls that
  intentionally differ between vec and cube codegen. Pure `Var` declarations and
  ordinary scalar temporaries usually do not need a scope wrapper; the splitter's
  use analysis keeps side-local values out of the opposite header.
- a5 micro UB-side addresses must be 32-byte aligned for `ub_to_reg`, `reg_to_ub`, continuous
  pack/unpack/upsample/downsample/brcb modes, and RegList continuous copies. `single()` /
  `single_value()` (`DIST_BRC_*` / `DIST_FIRST_ELEMENT_*`) are scalar-address operations and are
  allowed to target an isolated unaligned scalar slot. This does not make repeated one-element
  GM-to-UB or UB-to-GM bursts densely packable: per-token scalar staging on A5 must keep one
  32-byte physical UB row per token (`[L, dtype.C0]`) as documented in
  `agent/references/constraints/a5.md`. The simulator checks the effective UB address as local-bank
  `storage_offset_bytes` plus slice/continuous offsets, not only the PyTorch view offset.
  (source: `easyasc/simulator/pipe_micro.py`, `easyasc/simulator/pipe.py`, and
  `agent/example/testcases/simulator/micro/test_micro_alignment_guards.py`)
- a5 micro `ub_to_reg` / `reg_to_ub` / downsample copies can touch a hardware-width
  register footprint that is wider than the logical UB tensor view. If the view
  is a compact tail row or strided NZ row, add an explicit active-lane mask or
  leave padding/dedicated scratch after the view. For float-to-bf16 downsample
  row publishes, remember that the bf16 mask covers the packed lanes; a
  16-value fp32 row publish through `DIST_PACK_B32` needs a `2 * 16` bf16-lane
  mask to write all intended outputs without spilling past the view.
- use finite sentinel `-1.0e30` instead of `float("-inf")`
  (source: `agent/example/kernels/a2/attention/flash_attn_full.py:8` — `NEG_LARGE = -1.0e30`)
- `@kernel` function names must not contain uppercase letters; the decorator rejects them before
  Python-to-DSL transformation
  (source: `easyasc/decorators.py` `_validate_kernel_name`)
- DSL object names (`GMTensor` / `Tensor` / `DBuff` / `TBuff` / `QBuff` / `Var` / `VarList` /
  `Reg` / `RegList` / `MaskReg` / `SEvent` / `DEvent` / `TEvent` / `QEvent`, plus `reinterpret` / `reshape` /
  `split_workspace` views) must not be a C++ reserved keyword — they are emitted verbatim as C++
  identifiers (e.g. `GlobalTensor<float> new;`). Because the naming transform reuses the Python
  variable / kernel-argument name, a local like `new = Tensor(...)` or a param `def k(delete: GMTensor)`
  is rejected with a `ValueError` at call time. Matching is exact/case-sensitive, so `New` / `Int`
  are fine. Auto-generated `_tmp_*` names are always safe; `CastConfig` is exempt (its emitted name is
  always `{module}_` prefixed)
  (source: `easyasc/utils/naming.py` `assert_not_cpp_keyword` / `CPP_RESERVED_KEYWORDS`, called from the
  constructors in `easyasc/utils/{Tensor,var,reg,events}.py`, the `reinterpret`/`reshape_gm_tensor`/
  `split_workspace` views in `easyasc/stub_functions/misc.py`, and the kernel-argument binding in
  `easyasc/kernelbase/kernelbase.py`)
- ordinary helpers called from `@kernel` or `@vf` bodies must be decorated with `@func()` unless
  they are already a DSL-decorated callable such as `@kernel` or `@vf`; `@func()` runs the same
  Python-to-DSL transform path used for naming injection, which keeps generated code readable
  instead of falling back to `_tmp_var_*` / `_tmp_tensor_*`
  (source: `easyasc/decorators.py:58-77`, `easyasc/pythonic.py:363-389`)

## DBuff capacity quick-models

- `splitk` L0A: `TILE_M * SPLIT_K * 2` elements (DBuff 2 slots)
- `splitk` L0B: `TILE_N * SPLIT_K * 2` elements
- `splitn` L0A: `TILE_M * TILE_K * 2` elements
- `splitn` L0B: `SPLIT_N * TILE_K * 2` elements
- `L0C` DBuff: `2 * TILE_M * TILE_N` elements; bytes = `2 * TILE_M * TILE_N * sizeof(accum_dtype)`
  (source: `agent/scripts/estimate_matmul_datamove.py:10` `MAX_L0C_TILE_ELEMENTS_DBUF = 32 * 1024`)
- Budget: `L0A <= 64 KB`, `L0B <= 64 KB`, `L0C <= device cap` (see `facts-device-runtime.md`)

Stable large-K MKNK pattern (`agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk.py:11-14`):
- `TILE_M=128, TILE_N=256, TILE_K=256, SPLIT_K=64` → L0A DBuff 32 KB, L0B DBuff 64 KB

## a2 data-path reminders

- cube -> vec: `ws[...] <<= l0c[slot]` (`l0c_to_gm_nz2nd`) then `ub <<= ws[...]` (`gm_to_ub_pad`).
  Match the workspace slot count and shape to the scheduled handoff. Current grouped
  attention uses widened score/P workspaces such as
  `split_workspace(dtype, [GetCubeNum(), GROUP_STAGE_SLOTS, TILE_M, GROUP_N * TILE_N])`;
  simpler one-tile examples use `[GetCubeNum(), 2, TILE_M, TILE_N]`.
  (sources: `agent/example/kernels/a2/attention/flash_attn_full.py`,
  `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py`)
- vec -> cube: `ws[...] <<= ub` (`ub_to_gm_pad`) then `l1[slot] <<= ws[...]` (`gm_to_l1_nd2nz`)
  (source: `easyasc/a2.py:20,35`)
- UB is undefined at kernel entry; initialize with `dup(...)` before use
  (source: `agent/example/kernels/a2/attention/flash_attn_full.py`)
- UB/local tensor second dimension must be 32-byte aligned; for fp32 row scalars use
  `[1, M]`, then `brcb(..., repeat=M//8)` to broadcast into `[M, 8]`
- `ub_to_ub` unsafe for row-scalar state; use `add(tmp, src, ub_zero_s)` with an
  aligned zero buffer such as `[1, M]`
  (source for aligned scalar-buffer pattern: `agent/example/kernels/a2/attention/attn_backward_dense_total_tail_stage1_prob_dqk_gq_gk_gv_hif8_output_cast.py:207-208,330-341`)
- each sub-block has independent UB; use `GetSubBlockIdx()` to split work

## Deeper references

- `agent/references/constraints/tiling.md`
- `agent/references/constraints/sync.md`
- `agent/references/constraints/a2.md`
- `agent/references/constraints/vec.md`
- `agent/references/constraints/tail.md`
- `agent/references/constraints/a5.md`

## Fused-kernel GM staging rules

When multiple pipeline stages that were separate kernel launches are fused into
a single kernel, every GM staging slot that previously crossed a kernel-launch
boundary must now be protected with explicit synchronization. The launch
boundary was implicitly providing cube→vec and vec→cube ordering; `auto_sync()`
does not replace it.

### Cross-side staging (cube ↔ vec)

`auto_sync()` orders within one side (cube or vec) only. When a cube-side
producer writes a staging slot to GM and a later vec-side consumer reads it (or
vice versa), insert an explicit cross-side token:

| Direction | Token | Notes |
|-----------|-------|-------|
| cube → vec | `CvMutex` or `cube_ready/wait_cube` | FIX-side writer signals MTE2-side reader |
| vec → cube | `VcMutex` or `vec_ready/wait_vec` | vec-side writer (MTE3) signals cube-side reader (S) |

Do not assume `auto_sync()` will order a `l0c_to_gm_nz2nd` before a later
`gm_to_ub_pad` — they are on different sides.

### Cross-lane coordination (vec0 ↔ vec1)

Two vec sub-blocks on the same core share L1 and GM. When one lane's result
must be visible to the other lane before the consumer proceeds (e.g. lane 0
wrote half of an L1 buffer, lane 1 needs to read the full buffer; or lane 0
needs to finish reading a staging slot before lane 1 overwrites it), use the
intracore vec-lane barrier:

| Token | Scope | Notes |
|-------|-------|-------|
| `intracore_allvec_ready/wait` | Same core, both vec lanes | Both in vec-side; no cube-side involvement |
| `allvec_ready/wait` | All cores, all vec lanes | Cross-core vec-side coordination |

`vec_ready/wait_vec` is a **vec→cube** cross-side token (vec side signals on
MTE3, cube side waits on S) — it is not a lane-to-lane signal. Do not use it
when the goal is purely to order two vec lanes on the same core; use
`intracore_allvec_ready/wait` instead.

A common failure pattern: lane 0 needs to read a full `[L, L]` matrix from a
staging slot before lane 1 overwrites its half. Without an intracore barrier,
lane 1 may overwrite its half before lane 0 finishes reading.

### Staging-slot lifetime and self-overwrite

If the same GMTensor parameter serves double duty — first as a staging slot
for an intermediate, then as a final output — the consumer of the intermediate
**must** execute before the final-output write:

```
Read staging value → Use it → Write final output   (correct)
Write final output → Read staging value             (self-overwrite, bug)
```

When reordering is not possible within the existing slot budget, add an extra
workspace GMTensor to the kernel signature to hold the intermediate, removing
the alias. An extra workspace costs one more GM allocation but eliminates the
lifetime conflict entirely.
- `agent/references/patterns/a2-mixed-pipeline.md`
- `agent/references/patterns/a5-mixed-pipeline.md`

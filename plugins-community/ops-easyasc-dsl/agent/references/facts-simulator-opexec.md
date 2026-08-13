# Simulator and OpExec Facts

Use this file for simulator behavior one-liners and `OpExec` call-site gotchas.
Use `agent/references/code-paths.md` when you need the full implementation path.
Use `agent/references/cycle-model.md` for the checked-in A2/A5 simulator timing
model and its limitations.

## Simulator-behavior one-liners

- `OpExec(..., simulator=True)` is the documented and implemented simulator form. `simulator` must be a boolean; string aliases such as `"v2"` / `"legacy"` are no longer accepted by `easyasc/torchplugin.py`.
- The default simulator `execution_timeout_s` is 40s. Large direct trace runs
  can exceed it even when the kernel is healthy; after timeout, worker shutdown
  can cascade into misleading missing-tensor / metadata `KeyError` messages.
  Rerun known-large traces with an explicit `SimulatorConfig`, e.g.
  `kernel._simulator_config = SimulatorConfig(execution_timeout_s=300.0)`.
- `bar_all()` is the only cross-pipe drain on the simulator path. Hardware
  single-pipe barriers (`bar_v()`, `bar_m()`, `bar_mte*()`, `bar_fix()`) order
  later instructions only within that pipe; simulator pipe mailboxes are
  already FIFO, so `easyasc/simulator/bridge.py` intentionally drops them.
  This is not missing simulator behavior and must not be replaced by a
  cross-pipe wait. Only `bar_all`, or `barrier` with `pipe=Pipe.ALL`, routes to
  the cross-lane `bar_all` path. The simulator `CycleBarrier` also propagates
  the max arrival cycle back to PipeS and all pipe workers, so tasks after
  `bar_all()` start after the slowest pre-barrier pipe has drained.
- named local events enforce the hardware budget of eight IDs per ordered pipe
  pair: `SEvent` consumes one ID, `DEvent` consumes two, `TEvent` consumes
  three, and `QEvent` consumes four. Simulator preflight
  counts final per-lane declarations after `auto_sync()` lowering and fails
  before core launch when a pair exceeds eight; `LocalEventBank` also enforces
  the limit defensively. The check therefore covers both handwritten and
  autosync-generated events (`easyasc/simulator/simulator.py`,
  `easyasc/simulator/sync.py`; regression:
  `agent/example/testcases/simulator/sync/test_local_event_budget.py`).
- the simulator **does** model `CvMutex`/`VcMutex` depth and ring-slot aliasing, so an under-sized workspace ring or mutex depth is a *simulator-observable* correctness bug — but only at a shape that **wraps the ring past the pipeline lookahead**. `kernelbase.py` pre-loads `range(mutex.depth)` head-`ready` / tail-`wait` permit credits that `easyasc/simulator/sync.py` consumes as a counting gate, and `easyasc/simulator/pipe.py` resolves ring slots as `slot_names[idx % len(slot_names)]`. The invariant is `slots > lookahead` (mutex depth likewise); tiny smoke shapes (one work-item per core, `<= lookahead` groups) never wrap a ring and pass even when undersized — which is exactly how such a bug reaches the NPU. Always stress pipelined kernels at a ring-wrapping shape; the corruption then reproduces in the Python sim, hitting `out` (loop-carried row stats usually live in separate UB and stay clean).
- `mmad` is modeled **functionally** and honors the L0C destination row offset, but real cube HW does **not** place a matmul partial at a nonzero L0C *row* sub-offset — so a row-sliced matmul dst (e.g. `l0c[64:128, ...]`) passes the simulator bit-identically yet leaves that L0C region uninitialized on hardware (corrupting `rowmax`/`out`). Column / N sub-offsets are fine; keep the matmul dst at row offset 0 (see `agent/references/constraints/tiling.md`).
- simulator allocation initializes output GM tensors, workspace, and local
  memory (`L1`/`L0*`/`UB`/`BT`) with the byte pattern `0xff`, not numeric one or
  zero. The interpreted value therefore depends on dtype. GM/workspace
  allocation happens before core launch, each core waits at a startup barrier
  after local-memory allocation, and `run_sim()` copies host data into every
  bound GM tensor **except** the ones the kernel returns — an output keeps the
  `0xff` poison so a lane the kernel forgets to write reads back as NaN instead
  of as whatever placeholder the caller passed. A kernel that genuinely reads
  its output buffer (a GM atomic accumulating into caller-zeroed memory, a
  replay stage that samples the previous kernel's result, a pass-through that
  returns its input untouched) opts back in with
  `kernel._simulator_seed_outputs = True`; otherwise it must initialize the
  buffer itself before the first accumulation. The output set comes from
  `kernel._last_output_gmtensors`, which is also what `easyasc/simulator/bridge.py`
  uses to tag a GM spec `input` vs `output`.
- simulator core workers attach parent-owned Python `SharedMemory` blocks but do not track them with the child `multiprocessing.resource_tracker`; the parent `SharedMemoryStore.clear()` remains the owner that closes and unlinks GM shared-memory blocks. This avoids false `resource_tracker` leaked-shared-memory warnings after multi-kernel simulator runs.
- simulator GM atomic writeback uses one shared cross-process lock for cube FIX and vec MTE3 paths; cube `l0c_to_gm_nz2nd` supports `atomic_add`, `atomic_max`, and `atomic_min`, and vec `ub_to_gm_pad` supports add/max/min modes
- simulator int4 MMAD support is carrier-based: `DT.int4` local views alias
  their backing `DT.int` tensors, and `MPipe` unpacks signed low-nibble-first
  int4 carrier columns before the int32 matmul. For A2 L0A ZZ decode, the
  simulator uses the physical carrier column count captured from the L1->L0
  transfer, not just `ceil(valid_k / 8)`, so tail carrier panels stay aligned
  across 16-row blocks.
- simulator hif8 MMAD support is a5-only and carrier-based: public GM tensors
  should stay `DT.uint8`, local L1/L0 `DT.uint8` carriers can be reinterpreted
  as `DT.hif8`, and `mmad` decodes the L0 byte payload with the dtypehelper
  hif8 table before float32 accumulation. VM-local CANNSIM validated
  `agent/example/kernels/a5/matmul/hif8_carrier_matmul.py` across all 12 `A/B` transpose
  and no-split/split-N/split-K shortcut combinations with generated
  `LocalTensor<hifloat8_t>` operands consumed by `MMAD`. For CANN-safe
  `B.T + splitn`, keep split-N chunks 32-column aligned; a 16-column chunk
  produced wrong CANNSIM values.
- simulator MX FP8 MMAD support is a5-only: `l1_to_l0_mx` copies FP8 data into
  L0A/L0B and e8m0 scale bytes into the per-core 4 KB `L0AMX` / `L0BMX`
  buffers, including L0A/L0B NZ2ZN for `src.T`; `mmad_mx` decodes NZ FP8 operands,
  expands scale as `2 ** (value - 127)` per K-32 group, performs float32
  accumulation, and writes compact L0C NZ. Scale bytes are stored by K-64
  blocks as `[16, 2]` rows, so GM external scale should be packed as
  `[row_tile, k64_block, 16, 2]` flattened to `[num_blocks, 32]`. The simulator
  also keeps operand-local L0MX scale snapshots so split-K/split-N prefetches do
  not overwrite the scale consumed by in-flight `mmad_mx` tasks.
- simulator MXFP4 MMAD support is a5-only and carrier-based: L1/L0 FP4 views
  alias `DT.uint8` packed carrier tensors, `l1_to_l0_mx` copies packed payload
  plus scale into L0/L0MX, and `mmad_mx` decodes FP4 values with CANN-observed
  tables plus normal e8m0 scale expansion: e2m1 magnitudes are `[0, .5, 1,
  1.5, 2, 3, 4, 6]`, while e1m2 magnitudes are `[0, .25, .5, .75, 1, 1.25,
  1.5, 1.75]`. CANN 9.0 CANNSIM matched all four e2m1/e1m2 FP4 pairings
  (e2/e2, e1/e1, e2/e1, e1/e2) for nosplit, split-K, and split-N. **FP4 split-K/
  split-N were broken, then fixed 2026-05-31 — a codegen offset-UNIT bug, NOT a
  toolchain limitation:** a `LocalTensor<fp4x2_*_t>` indexes in 4-bit FP4 ELEMENTS
  (two per carrier byte), but `build_offset_expr_nz`'s FP4 branch emitted slice
  offsets in carrier-BYTE units, so a sliced FP4 sub-tile landed at half its
  intended row/K (splitn N-tile `_subn` read B rows `_subn/2`; reverse-engineered
  as `A @ B[8:24]^T` for `_subn=16`). nosplit (offset 0) was immune, which masked
  it, and the Python sim slices logically so it was always idealized-correct. Fix:
  that branch now doubles the offset (`(...) * 2`, carrier-byte -> fp4-element
  index). Verified diff=0 on Ascend950 CANNSIM (splitn/splitk via the matmul_mx
  shortcut, the bare `l1_to_l0_mx`+`mmad_mx` lower API, and with C2 bias). Only
  non-transpose (pack_axis=1) is HW-verified; the transpose (pack_axis=0) arm got
  the same `* 2` by construction. (An earlier note called this a toolchain
  limitation and a "FP4 splits matched" claim was masked by non-distinct test
  data — both superseded; see the validate-with-distinct-values rule.) FP4 NZ2ZN
  source blocks must be 64 x 64, so narrower logical transpose operands need
  padded L1 carrier tiles.
- simulator GM slice views must preserve the full linear coverage needed by strided writers; for batched ND slices such as `out[b, row0:row1, col0:col1]`, `_handle_slice_gm_tensor` sizes the backing view by `1 + sum((span_i - 1) * stride_i)`, not by `product(spans)`, so `l0c_to_gm_nz2nd` can scatter rows with `N_dst`. Runtime GM slice snapshots include source, flat offset, offsets, source dims, and spans; same-offset tail/full slices must not alias the same view, and real per-dim or storage overflow is rejected at slice time instead of silently truncating the view.
- simulator `gm_to_l1_nd2nz` supports partial NZ destination writes when
  `M < M_dst`: it preserves peer rows instead of clearing the full destination
  footprint, and packed local slice suffixes such as `:32,0:32,128` provide the
  row/column placement for cube-side MTE2. Full packed local slices still
  resolve to the base tensor so existing ops with explicit `src_row0` /
  `dst_row0` keep their historical behavior.
- simulator `l0c_to_gm_nz2nd` / FIX source slices decode from the sliced L0C
  logical NZ window instead of flattening from row 0. A source such as
  `l0c[2:3, :]` reads row 2 with C0 fixed at 16, and the footprint guard checks
  the touched NZ source window.
- Simulator timing behavior and parameters are owned by
  `agent/references/cycle-model.md`, `easyasc/simulator/timing/cycle_model.py`,
  and the device JSON profiles. Use modeled trace makespan, rather than summed
  task durations, when comparing schedules.
- simulator local-memory hazard checking is opt-in via `SimulatorConfig(local_memory_hazard_check="warn"|"error")`. It checks the hardware ownership pairs that have local-memory visibility risk: `UB V->MTE3`, `L1 MTE3->MTE1`, and `UB FIX->V`; a same-range producer/consumer cycle overlap is reported. `local_memory_hazard_memory_cycle_scale` multiplies modeled datamove task cycles in the pipe scheduler, so event sets and `bar_all` release times move with the stressed bandwidth instead of producing fake hazards on correctly synchronized producer/consumer pairs.
- simulator buffer metadata now treats `QBuff` as four slots via `slot_count`, matching QBuff codegen; `DBuff` and `TBuff` remain the built-in two- and three-slot special cases.
- `wait_vec` / `wait_cube` timeout almost always means the other lane's actor thread crashed silently — see `easyasc/simulator/pipe.py:397,419` (and `1086-1098`) for the timeout raise path
- concurrent simulator processes can cause silent corruption; run sim tests sequentially
- simulator pipe shutdown drains mailboxes by default via an unbounded `PipeBase.join()` and keeps live thread handles after explicit timed joins. Do not fix cross-pipe memory visibility by globally sequencing normal pipe tasks: independent pipe workers must keep their overlap, with only per-pipe FIFO serialization as the baseline.
- PyTorch does not support indexing on `float8_e5m2` / `float8_e4m3fn`; view as `torch.uint8` before indexing inside `@vf`
- burst-copy ops (`gm_to_ub_pad`, `ub_to_gm_pad`, `ub_to_l1`, `ub_to_l1_nz`) are safe on column-sliced UB views because they use `_linear_view` (`easyasc/simulator/pipe_vec.py:31`)
- simulator footprint guards for burst datamoves use the actually touched
  footprint, not a dense rectangle. For `gm_to_ub_pad` / `ub_to_gm_pad`, each
  burst touches only `burst_len` elements; the stride gap after the last burst
  is not considered accessed memory. The detailed guard model lives in
  `agent/references/simulator-datamove-footprint-guards.md`.
- A5 micro `micro_reg2ub` / `micro_ub2reg` footprint guards check the logical
  tensor/view boundary for strided `DATA_BLOCK_COPY` as well as contiguous
  copies. `micro_slice_tensor` preserves the logical tail while datamove ops
  still use the backing-storage tail for actual access, so an unmasked BF16
  register store that writes 256B where the intended NZ row footprint is 128B
  emits a `past UB tensor view` warning. Actual reads/writes continue against
  the backing storage to model hardware punch-through behavior. The warning
  suggests adding an explicit tail/NZ/stride mask or padding/dedicated scratch
  after the view when the hardware-width access is intentional; only accesses
  past the backing storage remain hard simulator errors.
- simulator `PipeS` condition evaluation supports simple arithmetic comparisons
  such as `pipe_work < work_end` and `pipe_work + 1 <= work_end`; use real script
  launches for multiprocessing simulator runners, not stdin snippets.
- A5 micro simulator math masks match hardware-style inactive-lane behavior:
  binary/unary/unaryscalar inactive destination lanes are written as zero while
  source registers are left unchanged. `var_assign` inside `call_micro` is
  supported for integer-category assignments such as `int -> uint32`, but
  integer/float conversions remain rejected.
- A5 micro `call_micro` scalar-immediate ops (`muls`/`adds`/`maxs`/`mins`/`lrelu`)
  round the fp32 immediate to the **operand register's dtype** before the op,
  matching hardware (`easyasc/simulator/pipe_micro.py:54`
  `_round_scalar_to_reg_dtype`, called in the `_MICRO_SCALAR` branch; float regs
  are a no-op). Rationale: HW coerces the fp32 immediate to half/bf16 and computes
  in that precision (`half*half->half`), whereas the pre-fix sim multiplied by the
  full fp32 scalar and rounded once. A non-half-representable scalar (e.g.
  `SOFTMAX_SCALE=0.08838…`, nearest half off by `1.07e-4`) then diverged from
  cannsim/HW by ~`1e-4`/element — which, fed through an online running-max, flips
  per-query argmax and shifts the entire softmax (bench P err `6.25e-2`) while the
  pre-fix sim falsely PASSed at P err `0`. Kernel-side avoidance: fold the scalar
  into the `l0c_to_ub` fixpipe requant (its deqScalar is an fp32-domain multiply
  *before* the half downcast, modeled identically in sim and HW, so it rounds once
  with no `half*half`) — this is why the half / int8 end-to-end PFA kernels carry
  `SOFTMAX_SCALE` in the requant.
- A5 micro simulator scalar Var ops support `Align8` / `Align16` / `Align32` /
  `Align64` / `Align128` / `Align256` inside `call_micro`, matching the
  control-pipe integer align formula.
- A5 SIMT simulator block intrinsics follow hardware identity:
  `simt_block_idx()` is the cube/core index and `simt_block_num()` is
  `core_count`. For GM-wide sharding across vec participants, pass
  `GetVecIdx()` / `GetVecNum()` into the SIMT body, or pass `GetSubBlockIdx()`
  and compute `simt_block_idx() * 2 + sub_block_idx`.
- A5 SIMT simulator call-site bindings preserve Python int/float/bool literals
  passed to `@simt` Var formals as scalar constants instead of deferring them by
  synthetic Var names such as `"16"`.
- A5 SIMT lowering folds module-global numeric constants (Python/numpy
  bool/int/float scalar values) inside `@simt` bodies into literal IR/codegen
  constants. SIMT IR/runtime supports the `//` floor-division operator; C++
  rendering maps Python `//` to `/`, which matches the common non-negative
  integer index math used by current kernels.

## Codegen testing

- constructing `OpExec(kernel, gen_only=True)` alone does **not** test codegen — it only creates the object
- the codegen path (`op_func.generate()` → `generate_op_host()`) only runs when the op is **called** with real arguments: `op(q, k, v, ...)` — see `OpExec.__call__` in `easyasc/torchplugin.py` and `KernelBase.generate_op_host()` / `KernelBase.generate()` in `easyasc/kernelbase/kernelbase.py`
- to test codegen, always run the kernel script end-to-end (e.g. `python agent/example/kernels/a2/<subdir>/foo.py`), not just construct the OpExec object
- for `OpExec(..., gen_only=True)`, codegen resolves `cann_path` from the
  explicit constructor argument first, then `ASCEND_HOME_PATH`, then falls back
  to `""` so generated sources and `*_aclnn_test` scaffolds can be inspected on
  machines without CANN. Use a real CANN path for any build/run step.
- `OpExec(..., debug=True, gen_only=True)` can generate the CANN debug workspace without immediately running root `b.sh`/`r.sh`; this is useful when patching the generated root scripts before build/run
- non-simulator generation resolves one profile-owned target consistently:
  `b3` uses 20/40 workers, `ascend910b`, and `Ascend910B3`; `a3` uses 20/40,
  `ascend910_93`, and `Ascend910_9362`; `950` and `950pr` use their C310
  profiles. The op-host emits exactly one `AddConfig(...)`, and
  `CMakePresets.json` receives the same compile unit.
- for a2, debug generation emits root `b.sh`/`r.sh` with `-r npu -v Ascend910B3`; CANN's debug templates accept `-r sim`, and a minimal a2 half vector-add ran successfully under the 910B3 PEM simulator after generated-build-tree-only CANN 9.0.0 workarounds for the nested preprocess `compile_commands.json` and final simulator-library link flags
- for a3/`Ascend910_9362`, debug generation uses `-v Ascend910_9362`; the
  normal non-debug custom-op path has been validated on real hardware with the
  `ascend910_93` op-host and CMake target
- on a remote A5 real-HW host, the stable non-interactive run wrapper is:
  `bash -il` (or equivalently source login/profile scripts), then explicit
  `source /etc/profile >/dev/null 2>&1 || true`,
  `source <ascend-cann-root>/set_env.sh >/dev/null 2>&1 || true`,
  `export PYTHONPATH=<repo-root>`, and finally `<python-bin> ...`.
  Substitute the placeholder paths with the host-local toolkit root, repo root,
  and Python binary. For real-HW probes, prefer
  `OpExec(..., simulator=False, debug=False)` rather than the debug-workspace
  harness.
- a placeholder SSH wrapper for the same flow is:
  `ssh <remote-a5-host> "bash -il -c 'source /etc/profile >/dev/null 2>&1 || true; source <ascend-cann-root>/set_env.sh >/dev/null 2>&1 || true; export PYTHONPATH=<repo-root>; <python-bin> <script>.py'"`.
  Use `debug=True` only when you explicitly want the generated debug-workspace
  artifacts.

## `OpExec` call-site checklist

- use the return value of `OpExec(...)(...)` as the source of truth for outputs; both simulator and runtime paths map kernel-returned `GMTensor`s back to fresh torch tensors and kernel-returned `GMTensorList`s back to Python lists of fresh torch tensors
- on non-simulator `OpExec` codegen/runtime paths, every `@kernel` must expose at least one input `GMTensor`/`GMTensorList`, at least one output `GMTensor`/`GMTensorList`, and at least one `Var`; outputs are the tensor-like arguments returned by the kernel, so a sync-only or output-only probe still needs a dummy non-returned input tensor-like argument to satisfy the compiler entry contract
- `GMTensorList` counts as a tensor-like parameter for the same contract. At the call site, pass a non-empty Python `list`/`tuple` of same-dtype, same-rank `torch.Tensor` items before scalar args; concrete dimensions may differ. Returning the `GMTensorList` parameter marks it as a preallocated dynamic output list, while non-returned lists are inputs.
- every `@kernel` parameter name must also be unique under `name.lower()`; names that differ only by case (`h` vs `H`, `r` vs `R`) are rejected before codegen/runtime
- do not validate against a placeholder output tensor argument unless you intentionally confirmed it is the returned object
- the hardware path writes only `input_param_names` into `<out_dir>_aclnn_test/input/`; output tensors are allocated fresh on the device, so **one launch never inherits what a previous launch wrote into the same output tensor**. A host driver that models several launches sharing one GM allocation (an ACLNN plugin filling disjoint windows of one buffer, e.g. a bidirectional RNN where each direction owns half the columns) must splice each launch's own window back together itself. This is the exact opposite of the simulator's poison rule above, so a driver can be correct under `simulator=True` and silently wrong on the board.
- default to providing `shape_bindings={...}` whenever the kernel exposes scalar shape arguments at the `OpExec(...)(...)` call site
- do not rely on the current sample shape keeping scalar dimensions distinct; write the intended mapping explicitly
- `shape_bindings` belongs on the returned callable, not on the `OpExec(...)` constructor
- format: `{tensor_arg_index: [scalar_idx_for_axis_0, scalar_idx_for_axis_1, ...], ...}`
- TensorList format: `{tensor_arg_index: {"item": [scalar_idx_for_axis_0, ...]}}`; `item` binds shared per-item dimensions and is checked independently against every member, while the list length is taken from the runtime list/tuple
- leave item-dependent dimensions unbound and use `GMTensorList.item_numel(index)` plus a runtime `GMTensor.reshape(...)` inside the kernel before deriving shape-dependent slices or datamove strides
- the key is indexed among tensor args only; scalar args are skipped
- use `None` to keep an axis unbound when that is intentional
- only int scalar arguments are eligible shape scalars; automatic inference
  ignores float scalars, and explicit `shape_bindings` entries that point at a
  float scalar are rejected

Example, for `kernel(x:[M,K], y:[N,K], z:[M,N], M, N, K)`:

```text
shape_bindings={0: [0, 2], 1: [1, 2], 2: [0, 1]}
```

TensorList example, for `kernel(xs: TensorList[[1,n]], out:[1,n], n)`:

```text
shape_bindings={0: {"item": [None, 0]}, 1: [None, 0]}
```

Implementation: `OpExec.__call__` in `easyasc/torchplugin.py`

Real references:
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm_large_nk.py:137`
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_vec_scale2_abs_add1_matmul.py:116`
- `agent/example/kernels/a2/vec_only/addn_tensor_list.py`
- `agent/example/testcases/simulator/opexec/test_tensor_list.py`
- `agent/example/testcases/codegen/opexec/test_tensor_list_codegen.py`

## Deeper references

- `agent/references/code-paths.md`
- `agent/playbooks/kernel-debugging.md`

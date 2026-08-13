# Code Paths

Use this file when you already know the topic and need to find the implementation path quickly.
Do not read the whole repository first.

## Device family mapping

Use this mapping before interpreting hardware-specific branches:
- `device_type in ("950", "950pr")` -> C310
- `is_c220_family(device_type)` (`b*` or `a3`) -> C220
- in `easyasc/resources/tensorutils.h`, `__DAV_C310__` corresponds to the 950 family
- in `easyasc/resources/tensorutils.h`, `__DAV_C220_CUBE__` corresponds to A2/A3
- do not invert this mapping when reading support checks, helper selection, or generated code paths

## Common Questions -> Files

### Repository structure or contributor architecture
- top-level layout: `agent/references/repo-map.md`
- contributor architecture and subsystem ownership: `doc/11_architecture_for_contributors.md`

### Public DSL surface
- architecture-specific exports: `easyasc/a2.py`, `easyasc/a3.py`, `easyasc/a5.py`, `easyasc/a5pr.py`
- `a3` re-exports the A2/C220 instruction surface with the 9362 device profile; treat the A2/A3 surface and A5 as parallel instruction families, not compatibility layers
- `GMTensorList` is the public tensor-like surface for dynamic dtype/rank-homogeneous GM tensor lists; concrete item dimensions may differ. It is exported through `easyasc/a2.py`, `easyasc/a3.py`, and `easyasc/a5.py`; the class and DSL methods live in `easyasc/utils/Tensor.py` (`GMTensorList`, `.size()`, `.item_numel(index)`, `__getitem__`). DSL lowering emits `create_gm_tensor_list`, `tensorlist_size`, `tensorlist_item_numel`, and `get_gm_tensor_list_item`; `xs.size()` returns the list length, `xs.item_numel(i)` parses and multiplies the selected runtime descriptor shape, and `xs[i]` returns a normal `GMTensor` with the trace-time prototype shape. Returning a `GMTensorList` parameter marks it as a preallocated dynamic output list; non-returned lists remain input lists. Current A2 hardware multi-item TensorList-output validation passes an explicit `item_count` scalar instead of relying on `.size()`/`ListTensorDesc::GetSize()`; `GetDataPtr<T>(item_idx)` is validated for output items through `agent/example/kernels/a2/vec_only/tensor_list_output_copy.py`.
- stable vec authoring rule for a5: use `@vf()`, `micro`, sort-family ops, `ub_to_ub`, the explicit GM→UB ND-DMA helper (`gm_to_ub_nd_dma`, including `ub <<= gm.T`), or the explicit UB→L1 publish helpers (`ub_to_l1`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`); do not treat raw `easyasc.a5` exports as permission to write generic vec math directly in the kernel body
- stable vec authoring rule for a2/a3: direct vec ops in the kernel body are normal, but `@vf()` and `micro` are unavailable
- MX FP8 / MXFP4 matmul is currently an a5/`950` surface only:
  `easyasc/a5.py` exports `gm_to_l1_pad`, `gm_to_l1_mx_scale`,
  `l1_to_l0_mx`, `mmad_mx`, `DT.mx_e4m3`, `DT.mx_e5m2`,
  `DT.fp4_e2m1`, `DT.fp4_e1m2`, and
  `matmul_mx`; it also exports the MXFP8 padding helper
  `zero_mxfp8_l1_padding`; see `doc/topics/mxfp.md` for the authoring contract.
- True hif8 on a5 should use `DT.uint8` public carriers and local
  `.reinterpret(DT.hif8)` views. Micro cast support lives in
  `easyasc/stub_functions/cast_rules.py`, simulator micro cast execution in
  `easyasc/simulator/pipe_micro.py`, and host reference codecs in
  `easyasc/dtypehelper/hif8_codec.py`; cube hif8 MMAD validation and
  simulator decode live in `easyasc/stub_functions/cube.py` and
  `easyasc/simulator/pipe_cube.py`. See
  `agent/example/kernels/a5/matmul/hif8_carrier_matmul.py` for the end-to-end carrier
  matmul smoke covering no-split/split-N/split-K across all A/B transpose
  pairings.
- decorators: `easyasc/decorators.py`
- flow helpers: `easyasc/flowcontrol.py`
- scalar `Var` arithmetic: dtype inference and constant folding live in
  `easyasc/stub_functions/var_op.py`; named-expression lowering lives in
  `easyasc/parser/asc_handlers/math_ops.py`, temporary/inlined expression
  lowering in `easyasc/parser/asc_utils.py`, and runtime execution in
  `easyasc/simulator/pipe.py`. Mixed float/integer `var_div` promotes to
  `DT.float`; generated C++ must retain an `f`-suffixed literal and an explicit
  cast when an integer `Var` is the denominator.
- syntax sugar / AST rewrites: `easyasc/pythonic.py`
  - native Python `if` / `elif` / `else` inside `@kernel` are rewritten into DSL `with If(...)`, `with Elif(...)`, and `with Else()` blocks before instruction capture
  - prefer native Python control flow in kernels and kernel-focused tests; inspect `easyasc/flowcontrol.py` only when you need the emitted `start_if` / `start_elif` / `start_else` / `end_if` details
  - regression/example: `agent/example/testcases/simulator/bridge/test_sim_nested_if_elif_else.py`
- runtime entry and `OpExec(..., simulator=...)` backend selection: `easyasc/torchplugin.py`
  - `OpExec(..., simulator=True)` routes to the simulator runtime via `KernelBase.run_sim()`; `simulator=False, gen_only=True` only triggers codegen artifacts
  - TensorList runtime binding also lives in `OpExec.__call__`: a `GMTensorList` parameter is supplied as a non-empty Python `list`/`tuple` of same-dtype, same-rank `torch.Tensor` items at the tensor-like argument position; concrete dimensions may differ. Use `shape_bindings={tensor_idx: {"item": [...]}}` for dimensions shared with scalar arguments; the binding is checked per item and the list length comes from the runtime list/tuple. TensorList parameters count as tensor-like arguments for the required `input tensor-like* -> output tensor-like* -> Var*` partition; returning a list from the kernel makes it an output and maps the result back to a Python `list[torch.Tensor]` while preserving each item shape.
  - `OpExec(..., debug=True)` generates a standalone debug workspace + `b.sh`/`r.sh` and (unless `gen_only`) builds + runs them. `cannsim=True` (now allowed alongside `debug=True`) emits the scripts as `-r sim` (CANNSIM) instead of `-r npu`, so the auto-run drives CANNSIM and reads the output back — convenient for validating on a sim-only host/VM. `simulator=True` + `cannsim=True` is still rejected (Python simulator and CANNSIM are distinct backends)
  - generated non-debug ACLNN test artifacts (`<out_dir>_aclnn_test/setup_aclnn.py`, `<out_dir>_kernel_script/b.sh`, `<out_dir>_kernel_script/r.sh`) are emitted by `easyasc/kernelbase/kernelbase.py` (`generate_aclnn_test`, `generate_bashfiles`) from the template in `easyasc/resources/setup_aclnn.py`; `easyasc/torchplugin.py` owns the `gen_only` orchestration, build/run log wiring, and generated-path return handling
  - kernel-signature front-end validation lives in `easyasc/torchplugin.py` (`_validate_kernel_tensor_var_partition`) plus `easyasc/kernelbase/kernelbase.py` (`validate_case_insensitive_kernel_param_names`); if `OpExec` rejects argument order, missing input/output/Var slots, or names that differ only by case (`h` vs `H`), start there

### Kernel and micro execution wrappers
- kernel lifecycle / generation hooks: `easyasc/kernelbase/kernelbase.py` (see `## KernelBase method map` below for a per-method index)
  - simulator entry: `KernelBase.run_sim()` (simulator run body is inlined)
  - program build: `KernelBase.build_simulator_program()`
- micro module lifecycle: `easyasc/micro/micromodule.py`
  - `vf_barrier` missing-barrier warnings are static instruction analysis in `MicroModule`, not simulator runtime checks
- `@vf()` Tensor metadata vars used by generated row offsets are validated in `MicroModule` after first body capture; hidden shape/stride vars such as `dim` must be explicit `@vf` `Var` arguments or fixed Python int constants. A Tensor view passed at the call site is treated as a new VF base pointer: its existing offset is consumed by the caller-side `GetPhyAddr()` and is not revalidated inside the VF body.

### Lowering pipeline
- main lowering entry: `easyasc/parser/asc.py`
- mixed-kernel barrier side ownership also lives in `easyasc/parser/asc.py`
  (`_instruction_side`): `M` / `MTE1` / `FIX` are cube-only, `V` / `MTE3`
  are vec-only, and `ALL` / `MTE2` / `S` are shared. See
  `doc/api/barrier.md` -> "Mixed-kernel side routing" and
  `agent/example/testcases/parser/core/test_side_pruning_codegen.py`.
- TensorList lowering: `easyasc/parser/asc_handlers/core.py` renders `ListTensorDesc`, `GetSize()`, `GetDataPtr<T>()`, and the `.item_numel()` `TensorDesc::GetDesc`/`GetShape` sequence; handler registration is in `easyasc/parser/asc_handlers/__init__.py`. TensorList item views and item-shape scalar assignments must be retained by `easyasc/parser/asc.py`, `easyasc/parser/asc_pruning.py`, and `easyasc/parser/asc_utils.py`. On A2 ACLNN TensorList-output hardware smoke, `.size()`/`GetSize()` only drove item0; pass an explicit item-count `Var` for hardware multi-item output loops until that path is fixed.
- local-Var lexical-scope and side-split validation also live in `easyasc/parser/asc.py`; inspect `_validate_local_var_lexical_scope()` and `_validate_side_local_var_references()` first when a branch-local `Var` leaks into sibling control flow or a split side loses the writer that defined a control scalar
- pruning and cleanup: `easyasc/parser/asc_pruning.py`
  - flat `if` / `elif` / `else` chain parsing and `elif` prelude retention live here; if a value computed immediately before an `elif` needs to stay attached to that branch during pruning, inspect `_parse_if_chain()`
- autosync insertion: `easyasc/parser/asc_autosync.py`
- final-side runtime buffer-view planning: `easyasc/parser/asc_buffer_view_liveness.py`
  - `DBuff` / `TBuff` / `QBuff` views indexed by a runtime `Var` keep snapshot semantics by default; only a single direct consumer in the same structured block, with no intervening write to the rendered index dependencies, is expression-inlined by final translation
  - multi-consumer views, control-flow crossings, `VarRef` indices, temporary index expressions that read tensor memory or a VarList element, nested/aliased escapes, and `slice_tensor` / `micro_slice_tensor` / `reinterpret` chains remain materialized as `LocalTensor<T>` aliases
- op-specific lowering: `easyasc/parser/asc_handlers/`

### Adding a vec primitive (end-to-end touch-points)
A new a2 vec op (e.g. `gather` / `gather_block`) must be wired through six places;
miss the last two and the op routes to `PipeS` ("unhandled instruction") or reads
uninitialized UB:
1. stub: `easyasc/stub_functions/vec/<area>.py` emits `Instruction("<op>", ...)`
2. parser handler: `easyasc/parser/asc_handlers/vec_<area>.py` + register the
   name -> handler in `asc_handlers/__init__.py`
3. simulator: a `_<op>` method + dispatch in `easyasc/simulator/pipe_vec.py`
4. pipe routing: `easyasc/simulator/bridge.py` `_VEC_PIPE_BY_OP["<op>"] = "V"`
5. autosync: add `"<op>"` to the vec-op list in `easyasc/parser/asc_autosync.py`
6. facade re-exports: `stub_functions/vec/__init__.py` -> `stub_functions/__init__.py`
   -> `easyasc/a2.py`
`gather_block` (AscendC `Gatherb`) is the most recent worked example.

### Adding a reg dtype (end-to-end touch-points)
A new register dtype (int64/uint64, and most recently `complex32`/`complex64`) is
registered across a fixed table of files — grep any existing dtype name (`uint64_t`)
to find every site:
1. dtype object + `.size`/`.C0` + family helper: `easyasc/utils/datatype.py`
   (`is_integer_dtype`/`is_complex_dtype` + `INTEGER_DTYPES`/`COMPLEX_DTYPES`)
2. codegen identifier fence: `easyasc/utils/naming.py` `CPP_GENERATED_TYPE_IDENTIFIERS`
3. C++ spelling: `easyasc/parser/asc_utils.py` `dtype_to_cpp` (special-case only if
   the C++ name differs from `.name`)
4. simulator storage/bridge: `easyasc/simulator/memory.py` (`_DTYPE_ALIASES`,
   `_DTYPE_SIZE_BYTES`, `torch_dtype`, `numpy_dtype` — `complex32` has **no** native
   numpy dtype, so it stores through a 4-byte `np.uint32` surrogate that
   `_alloc_shm`'s `.view(torch.complex32)` reinterprets), `bridge.py` `_DTYPE_NAMES`,
   `pipe_micro.py` `_torch_dt`
5. torch/ACL/GE bindings (only if the dtype crosses the torch<->kernel boundary):
   `easyasc/torchplugin.py` (`_acl_dtype_literal`, `dtype_map`, `_dtype_to_torch`,
   `_dtype_byte_size`), `easyasc/kernelbase/kernelbase.py` (`_dtype_to_ge`, two
   `_storage_cpp_type` maps — complex stores through a same-width uint surrogate —
   and the short-enum `_tensorx_type` map)
6. DSL op gates: per-op stubs under `easyasc/stub_functions/micro/` decide accept vs
   reject. For complex the shared helpers `reject_complex_dtype` /
   `format_complex_scalar` live in `micro/microutils.py`; only eight ops opt in
   (binary add/sub/mul/div, scalar adds/muls, unary abs, dup). `abs` is complex->real
   (`_COMPLEX_ABS_REAL` in `micro/unary.py`).
7. simulator compute: complex math is added in `pipe_micro.py` `_math` (binary/unary/
   scalar/dup branches compute in `complex64` and let `_apply_exec_mask_zero` demote
   to the declared width; `_numc` resolves complex immediates).
Complex `abs` end-to-end reg->UB store layout (a complex reg's real modulus
half-fills a real register) runs on-board (the MicroAPI ComplexAbsKernel uses
RegTraitNumTwo-src -> RegTraitNumOne-dst) but the pure-Python sim cannot model that
store, so the sim covers abs compute + dtype pairing only. On-board host-harness
touch-points a complex dtype also needs: `easyasc/resources/macros.h` (a
`TensorX<T, aclDataType>` macro, e.g. `#define COMPLEX64 uint64_t,
aclDataType::ACL_COMPLEX64`) and, for complex scalar immediates, a device literal
`complex64(re, im)` / `complex32((half)re, (half)im)` emitted by
`_complex_scalar_cpp` in `asc_handlers/vec_micro_ops.py` (a bare python `(1.5-0.5j)`
is a C99 `_Complex` and fails on aicore).

### Simulator path
- runtime root: `easyasc/simulator/`
- configuration dataclass and core/lane planning: `easyasc/simulator/config.py`
  - `SimulatorConfig` at `config.py:23`; fields include `core_count`, `execution_timeout_s`, `process_start_method`, trace/cycle-model settings, and opt-in local-memory hazard knobs (`local_memory_hazard_check`, `local_memory_hazard_memory_cycle_scale`)
- top-level simulator orchestrator: `easyasc/simulator/simulator.py` (`Simulator`)
- per-core child process wrapper and per-core runtime: `easyasc/simulator/core.py` (`Core`, `_core_entry`)
  - pipe class registry at `core.py:26-34`: maps `(lane, pipe_name)` tuples to concrete pipe classes
- kernel-program bridge (emits both linear and control-flow instructions into a single `KernelProgram`): `easyasc/simulator/bridge.py` (`build_kernel_program`, `KernelBridge`)
  - pipe routing constants at `bridge.py:20-52`: `_CUBE_PIPE_BY_OP`, `_VEC_PIPE_BY_OP`, rename table, debug-only pipe sets
  - TensorList control ops route here: `tensorlist_size` becomes a runtime list-length assignment, `tensorlist_item_numel` resolves the selected bound tensor's actual element count, and `get_gm_tensor_list_item` carries item tensor names into the control path for dynamic index resolution
  - dynamic `get_buf` slot selection for loop-carried DBuff/TBuff/QBuff access is bridged here and resolved at runtime
  - `reinterpret(...)` on top of loop-carried local views is re-materialized at runtime so later pipe tasks see the current slot/view instead of a static alias
  - `create_varlist`, `GetValueFrom`, and `SetValueTo` also route through the control-flow path, including dynamic `VarList` index resolution
  - float `Var` arithmetic on control-flow paths (`var_add`, `var_mul`, `var_div`) must stay float when the source `Var` is float
- public module exports: `easyasc/simulator/__init__.py` (`SimulatorConfig`, `Simulator`, `build_simulator`, `build_kernel_program`, `KernelProgram`, `LaneProgram`, `SharedTensorSpec`)
- shared tensor / workspace storage and local bank allocation: `easyasc/simulator/memory.py`
  - local bank allocation is used for per-core local-memory capacity checks; over-capacity local allocations fail before pipe execution starts
  - L0AMX/L0BMX local banks are allocated per core for MX FP8 scale storage
- pipe worker base, control-pipe runtime view helpers, and dispatch: `easyasc/simulator/pipe.py` (`PipeBase`, `PipeS`, `_dispatch_call_micro`)
  - vec mask state such as `set_mask/reset_mask` persists across sequential V-pipe tasks on the same lane
  - TensorList runtime execution lives in PipeS: `tensorlist_size` returns the number of bound item tensors, `tensorlist_item_numel` returns the selected tensor's actual numel, and `get_gm_tensor_list_item` materializes the selected item as a GM tensor view; dynamic `reshape_gm_tensor` is also materialized at runtime so item-dependent shapes reach slice bounds and datamove strides
  - runtime `get_buf`, `slice_gm_tensor`, `slice_tensor`, and `reinterpret` view materialization lives here; inspect `PipeS._handle_slice_gm_tensor` for GM slice footprint/bounds issues
  - local-memory hazard checking is wired into `PipeBase._run_loop`; datamove bandwidth stress cycles are applied in `_estimate_task_cycles`
- local-memory hazard tracker: `easyasc/simulator/hazard.py`
  - derives touched local byte ranges per pipe task and reports opt-in ownership-pair overlap hazards
- cube pipe executors (`MTE2`, `MTE1`, `M`, `FIX`): `easyasc/simulator/pipe_cube.py`
  - int4 MMAD simulator unpacking and physical carrier-column handling live here
  - MX FP8 / MXFP4 path lives here too: `gm_to_l1_pad`, `l1_to_l0_mx` scale copy to
    `L0AMX` / `L0BMX`, L0A/L0B NZ2ZN scale block gather, operand-local L0MX scale
    snapshots, e8m0 scale decode, FP4 carrier decode, and `mmad_mx` float32 accumulation
  - `l0c_to_gm_nz2nd` source-slice decode lives here; inspect FIX/L0C decode
    before blaming GM views when a sliced L0C writeback repeats row 0
- vec pipe executor: `easyasc/simulator/pipe_vec.py`
- micro runtime: `easyasc/simulator/pipe_micro.py`
  - float `Var` arithmetic inside `call_micro` execution is resolved here again; if a `@vf()` scale unexpectedly becomes `0`, inspect this file before blaming cube/vec handoff
  - `micro_vf_barrier` is a no-op in the serial simulator; codegen emits the real hardware local-memory barrier
- lane-local event-bank semantics for `create_sevent` / `create_devent` / `create_tevent` / `create_qevent` / `event_wait` / `event_set`:
  `easyasc/simulator/sync.py` (`LocalEventBank`, `LocalSync`)
- Chrome-trace export path: `easyasc/simulator/trace.py`
- a5 / a2 cycle-model profile and estimator path for simulator-side timing traces: `easyasc/simulator/timing/`

### Stub emitters and shortcuts
- operation emitters: `easyasc/stub_functions/`
- `Tensor.__ilshift__` (`<<=`) selects the datamove from `(dst.position, src.position)` plus the
  dst/src `_layout` flag (`easyasc/utils/layout.py`: `Layout.NZ`/`ND`; `Tensor`/`DBuff`/`TBuff`/
  `QBuff` take a `layout=` arg, default `Layout.DEFAULT` resolves to NZ for L1/L0A/L0B/L0C and ND
  for UB/BT via `resolve_layout`, using `positions.NZ_LOCAL_POSITIONS`). GM->L1: ND dst ->
  `gm_to_l1_pad` on a5, but `gm_to_l1` on a2/a3 (`is_c220_family(device_type)`; C220 has no
  DataCopyPad, so `__ilshift__` reads module-level `globvars.device_type` to pick the plain
  32B-granular burst); NZ dst -> `gm_to_l1_nd2nz`. UB->L1: NZ src -> `ub_to_l1_nz`, else
  `ub_to_l1_nd2nz`. BT<-L1 -> `l1_to_bt`. `.nz()`/`.nd()` return a layout-overridden view. The
  flag is DSL-level dispatch only and never reaches the Instruction/codegen/simulator
- slice *addressing* is separate from the `<<=` layout dispatch and is keyed off **position**, not
  `_layout`: codegen flattens a sliced tensor's offset via `is_nz_local_position(out.position)`
  (`asc_utils.py` build path / `core.py:268`) -> `build_offset_expr_nz` (fractal: `dim0` stride
  C0=8 fp32/16 fp16, `dim1` stride `dim0_size`) for L1/L0A/L0B/L0C, else contiguous
  `build_offset_expr`. So a `layout=Layout.ND` L1 tensor still slices with NZ fractal strides; for
  a `[1,N]` row the NZ formula degenerates to contiguous (`dim0_size=1` -> col stride 1), which is
  why the BT bias slice `bias[:, n0:n0+w]` is correct. A multi-row ND L1 tensor would NOT slice
  contiguously today (would need slice addressing to honor `_layout`)
- vec datamove stubs: `easyasc/stub_functions/vec/datamove.py`
  - raw `ub_to_l1` emits `UB2L1` and uses 32B block units for `burst_len` and both strides
  - `gm_to_ub_nd_dma` is the a5/950-only GM->UB multi-dimensional DMA surface. It records
    explicit innermost-first `loop_src_stride` / `loop_dst_stride` / `loop_size` arrays plus
    optional per-dimension padding, lowers through `handle_vec_gm2ub_nd_dma`
    (`parser/asc_handlers/vec_datamove.py`) to `GM2UB_ND_DMA` in `tensorutils.h`, and is
    simulated by `VecMTE2Pipe._gm_to_ub_nd_dma` (`simulator/pipe_vec.py`).
  - `ub <<= gm.T` on a5 now lowers to the ND-DMA transpose helper (`gm_to_ub_nd_dma_transpose`)
    instead of rejecting the `.T` view. The GM slice controls the logical moved block; the
    destination UB row stride still comes from the backing UB tensor width, like `gm_to_ub_pad`.
- low-level cube datamove / MX stubs: `easyasc/stub_functions/cube.py`
  - `gm_to_l1_pad` emits normal-mode `GM2L1PAD` (a5-only; uses DataCopyPad) — fenced
    `assert_valid_device(A5_DEVICES)`, raises on a2
  - `gm_to_l1` emits a plain `GM2L1` DataCopy burst (a2 & a5; no DataCopyPad) — fenced
    `assert_valid_device(A2_A5_DEVICES)`. Contiguous
    inference: `n_burst=1`, `burst_len = CeilDiv(total_src_bytes, 32)` in 32B units. This is
    the a2 substitute for `gm_to_l1_pad` when staging a contiguous row (e.g. bias) into L1.
    Handler `handle_gm_to_l1` (`asc_handlers/cube.py`), autosync MTE2, simulator
    `CubeMTE2Pipe._gm_to_l1_contiguous` (`pipe_cube.py`), hazard `gm_to_l1` (`hazard.py`)
  - `gm_to_l1_dn2nz` emits `GM2L1_DN2NZ` (a5-only `Dn2NzParams` DataCopy, `__DAV_C310__` gated):
    GM holds the matrix transposed (`GM[n, m]` -> dst NZ `[m, n]`), transpose happens on MTE2.
    M/N name the dst rows/cols, `N_src` = GM row stride (>= M); same arg order as `gm_to_l1_nd2nz`.
    Handler `handle_gm_to_l1_dn2nz` (`asc_handlers/cube.py`), autosync MTE2, simulator
    `CubeMTE2Pipe._gm_to_l1_dn2nz` (`pipe_cube.py`, transposed `as_strided` + `encode_nd_to_nz`),
    hazard/cycle-model reuse the nd2nz footprint with the source span transposed.
    Sugar: `l1 <<= gm.T` / `gm[a:b, c:d].T` — `GMTensor.T` (`utils/Tensor.py`) registers a
    transposed full-window `slice_gm_tensor` view (own identity, so pruning keeps it);
    `Tensor.__ilshift__` dispatches `is_transpose` GM sources to `gm_to_l1_dn2nz` (NZ L1 only);
    slicing/reshaping/reinterpreting a `.T` view or storing to it raises
  - `l1_to_l0a_img2col` emits `L12L0A_IMG2COL` from an `Img2colSlice`
    (`img2col(fmap, conv)[m0:m0+tm, k0:k0+tk]`) into L0A. The logical img2col gather is shared
    across devices, but the final L0A encoding differs: a2/b* writes ZZ while a5/950 writes NZ.
    Handler: `handle_l1_to_l0_img2col` (`asc_handlers/cube.py`); simulator:
    `CubeMTE1Pipe._l1_to_l0_img2col` (`pipe_cube.py`). a5 rule: hardware has no K-direction
    destination stride, so the C310 branch of `tensorutils.h` loops one load3d call per C0
    K-fractal and advances the L0A destination between calls, while the DSL surface stays one
    `l1_to_l0a_img2col(...)` call. Regressions: `agent/example/testcases/codegen/core/test_img2col_trace.py`
    and `agent/example/testcases/codegen/core/test_img2col_a5_sim.py`
  - `gm_to_l1_mx_scale` copies external packed scale blocks from GM to L1
  - `l1_to_l0_mx` emits `L0NZ2NZ_MX`, or `L0NZ2ZN_MX` for `src.T` into L0B
  - `mmad_mx` emits `MMAD_MX` (or `MMAD_MX_BIAS` when `bias=` is set — a5 C2 bias, see the bias bullet)
- matmul bias via the `BT` bias table (**a2 & a5**, is_init tile; see
  `agent/references/facts-device-runtime.md` and `doc/api/shortcuts.md`):
  - `Position.BT` maps to `TPosition::C2` in `easyasc/utils/positions.py`; total
    capacity comes from `globvars.bt_cap` (A2 512 B, A5/A5PR 4 KB), and
    `KernelBase` divides it into two slots
  - `l1_to_bt` emits `L1_TO_BT` (contiguous L1->C2 burst); the cube wrappers `L1_TO_BT` /
    `MMAD_BIAS` in `easyasc/resources/tensorutils.h` are gated
    `#if defined(__DAV_C220_CUBE__) || defined(__DAV_C310__)` (empty body on the AIV/vec
    compile of the same cube .cpp, like `LOADL0` -- no `#else static_assert`). The bias
    must reach L1 contiguously -- either `gm_to_l1_pad(l1b, bias)` (a5) / `gm_to_l1(l1b, bias)`
    (a2), or `<<=` into an ND-layout L1 tensor (`Tensor(..., layout=Layout.ND)`, which
    `Tensor.__ilshift__` dispatches to `gm_to_l1_pad` on a5 / `gm_to_l1` on a2). A default NZ
    L1 tensor's `<<=` nd2nz-fractalizes the `[1,N]` row and scatters all but the first fractal.
    L1->BT is then `bt <<= l1b` or `l1_to_bt(bt, l1b)`. The L1 source may carry a contiguous
    column offset (so the shortcut can stage a per-N-tile bias slice); the C2/dst always loads
    from slot start (offset 0). `l1_to_bt` enforces `dst.dtype == src.dtype` and
    `dtype in (float, int)`
  - `mmad(..., bias=)` emits `MMAD_BIAS` (4-arg `Mmad`, `cmatrixInitVal=false` so `C = bias + A@B`);
    the `bias` kwarg is carried on the Instruction only when present. Requires `is_init` tile
  - **bias datatype fence** (Mmad API doc `atlasascendc_api_07_0249`, tables 8 & 9), in
    `easyasc/stub_functions/cube.py`: `_validate_mmad_bias_dtypes(src_a, src_b, dst, bias)`
    raises `TypeError` if the `(A, B, bias, dst)` tuple is not in the device's allowed set
    (`_MMAD_BIAS_DTYPES_A2` / `_MMAD_BIAS_DTYPES_A5`). Rule: **int8@int8 -> int32** bias/dst;
    every other supported input -> **float**. a5 additionally allows fp8(e4m3/e5m2)/hif8 -> float
  - **MX (scale) bias, a5 only**: `mmad_mx(..., bias=)` emits `MMAD_MX_BIAS` (`tensorutils.h`,
    gated `#ifdef __DAV_C310__` with `#else static_assert` = MX type-asserts + `cmatrixInitVal=false`
    4-arg `Mmad`). `matmul_mx(..., bias=)` threads it through the same auto `_btbuf` / `_stage_bias_tile`
    / `emit_mmad_init(mmad_mx, ..., bias=)` as `matmul` (nosplit/splitn/splitk; reuses
    `shortcuts/matmul._stage_bias_tile`). Bias is fp32 only; the stub fences BT position + fp32 +
    is_init. The simulator `_mmad_mx` adds the bias broadcast like `_mmad`. autosync already covers it
    (`mmad_mx` is an M-pipe op, so the `bias` M-pipe read-key collection inserts the `L1_TO_BT`->`mmad_mx` sync)
  - a2 cube `analyze_usage` allows `BT` in `allowed_positions` (`easyasc/parser/asc.py`)
  - `easyasc/parser/asc_autosync.py` treats `l1_to_bt` as MTE1 and `bias` as an M-pipe read,
    so the `L1_TO_BT`->`mmad` RAW sync is inserted
- high-level matmul helpers: `easyasc/shortcuts/matmul.py`
  - int4 carrier-view validation and `src_*_carrier_cols` metadata are in
    `easyasc/stub_functions/cube.py`; shortcut split-k/split-n slicing starts
    from `easyasc/shortcuts/matmul.py`
  - `matmul(..., bias=l1bias)` adds a `[1, N]` L1 bias on the init tile via the
    kernel's auto-inserted `_btbuf`/`_btbufcnt` (kernelbase) -> `_stage_bias_tile` ->
    `l1_to_bt` -> `mmad(bias=)`; `splitn` stages a fresh BT slot per N-tile (ping-pong),
    `splitk`/`nosplit` stage once. `_get_kernel_bt_buffer` is in `easyasc/shortcuts/_common.py`.
    Bias dtype follows the fence: float (half/float/bf16 inputs) or int32 (int8@int8). For
    int8 the BT slot is `reinterpret`ed to int32 inside `_stage_bias_tile`. **a2 float
    `splitn`:** a2 slices a *float* L1 tensor with ZZ-fractal column addressing (`col0*16`),
    wrong for the contiguous `[1,N]` bias row, so `_stage_bias_tile` reinterprets the float
    bias to int32 before the column slice (contiguous NZ `col0*1`) and `l1_to_bt` moves the
    raw bytes; the mmad reads the slot back as float. So all of `nosplit`/`splitk`/`splitn`
    work for float and int8 on a2 & a5 (validated on Ascend910B3 + Ascend950 CANNSIM). The
    matmul operands' `splitn` slices ride the `L0ZZ2NZ` L1→L0 load (which de-fractalizes ZZ)
    and were never affected — only the bias (L1→BT direct) needed the reinterpret
- high-level conv helper: `easyasc/shortcuts/conv2d.py`
  - `conv2d(...)` is matmul-scope: caller owns GM->L1, the `m` loop / multicore gate, and the
    final store; helper owns img2col windowing, `l1_to_l0a_img2col`, L1 weight column tiling,
    optional BT bias on the first K tile, and K accumulation into a caller-supplied L0C tile
  - the helper is shared by a2 and a5 (`easyasc.a5` exports it too). On a5 the contract stays
    identical because `l1_to_l0a_img2col` hides the per-C0 load3d loop internally
  - the helper closes the two internal conv cout-tiling hazards itself: `barrier(Pipe.M)` after
    every mmad for the load3d-path L0C-reuse WAR (required on a2, harmless on a5), plus BT-slot
    ping-pong via `btbufcnt` for distinct-bias reuse. The implementation is in
    `easyasc/shortcuts/conv2d.py`; A5 event-family coverage is in
    `agent/example/testcases/codegen/core/test_img2col_a5_sim.py`
  - reference pure-cube samples: `agent/example/kernels/a5/conv/compute/conv_half_{basic,dilation,bias,large}.py`
    and `agent/example/kernels/a2/conv/compute/`
- MX FP8 / MXFP4 shortcut: `easyasc/shortcuts/matmul_mx.py`
  - currently supports no-split, split-K, or split-N, but not split-K and
    split-N together
  - scale tensors are `DT.uint8` e8m0 carriers with K-32 logical groups packed
    into K-64 `[16, 2]` blocks
- MX FP8 padding shortcut: `easyasc/shortcuts/mxfp8_padding.py`
  - `zero_mxfp8_l1_padding` clears padded FP8 L1 data tiles before filling live
    transpose data

### Runtime types and helper objects
- tensors, vars, regs, enums, instruction models: `easyasc/utils/`
- TensorList interface docs and examples: `doc/api/tensor_buffer.md` (`GMTensorList` section), `doc/api/op_exec.md` (`TensorList arguments`), and `agent/example/kernels/a2/vec_only/addn_tensor_list.py`
- A5 micro register debug printing: `easyasc/stub_functions/micro/debug.py` (`print_reg`)
  - codegen intentionally emits nothing for it in `easyasc/parser/asc_handlers/vec_micro_ops.py`
  - simulator-side output is implemented in `easyasc/simulator/pipe_micro.py` (`_print_reg`)
- host-side dtype carrier/reference helpers: `easyasc/dtypehelper/`
  - A2 exposes signed int4/int32 carrier helpers through `easyasc/a2.py`
  - A5 exposes HiFloat8/fp32, FP4/fp32 uint8-carrier, and e8m0/fp32 scale helpers through `easyasc/a5.py`

### Test organization and runnable demos
- automated test layout: `agent/example/testcases/README.md`
- single-kernel runnable samples: `agent/example/kernels/a2/`, `agent/example/kernels/a5/`
- multi-file system projects: `agent/example/projects/a5/gdn_fwd/` (preprocess + inverse + recompute-WU + chunk-delta forward paths) and `agent/example/projects/a5/gdn_bwd/`

## KernelBase method map

`easyasc/kernelbase/kernelbase.py` is a single ~1.9k-line file holding the `KernelBase` class. Use this map to jump by method; prefer the method name over the line number when the file has shifted.

### Construction and call
- `__init__` (L50) — store name/func/vector_mode; init instruction, mutex, workspace, and used-micro state
- `__call__` (L68) — bind args and run the wrapped function to emit instructions

### Developer-facing dumps
- `print_instructions` (L178) — print the captured instruction list
- `dump_asc` (L192) — write `<path>_cube.h` + `<path>_vec.h` from `translate_split`
- `dump_kernel` (L206) — write `<path>_cube.h` + `<path>_vec.h` + `<path>.cpp` entry with optional `debug_entry`

### CANN op-host and op-project scaffolding
- `generate_op_host` (L555) — write `<name>_tiling.h` + `<name>.cpp` (infoshape, tiling, op registration); requires the kernel to have been called once; for A5-family `device_type` values the op-host tiling emits `context->SetDynUBufSize(221184);`
- `generate_op_project` — unpack `CustomOp*.tar.gz`, materialize `CMakePresets.json` and `build.sh`; C220 (`b*` or `a3`) uses `CustomOp_a2_cann9.tar.gz` when the CANN install reports version >= 9.0 (`version.cfg`, `compiler/version.info`, or `opp/version.info`), otherwise `CustomOp_a2.tar.gz`; A5-family `950`/`950pr` keeps `CustomOp.tar.gz`
- `easyasc/device_profile.py` — device profile table for default core counts, vector lane counts, compile unit, and debug/CANNSIM SoC strings (`a3`/9362 = 20 cube and 40 vec with `ascend910_93`; `950` = 32 cores; `950pr` = 28 cores)
- `_resolve_op_host_block_dim` (L346) / `_resolve_debug_chipset` (L357) — profile-backed helpers for op-host output
- `_analyze_usage_for_simulator` (L969) plus `parser/asc.py::analyze_usage` — when a kernel contains `call_simt`, report and enforce the reduced a5 UB cap of 216 KB
- `_resolve_custom_opp_path` (L922) — normalize vendor opp install path

### Simulator orchestration
- `run_sim` — public simulator entry; builds the kernel program, executes it, copies tensors back, and optionally dumps a trace
- `_analyze_usage_for_simulator` — pre-run usage inspection
- `build_simulator_program` — build (or return a prebuilt) `KernelProgram`
- `_has_control_flow_instructions` — predicate used during program build
- `resolve_simulator_config` — produce the simulator runtime config
- `_iter_bound_gmtensors` — iterate bound GM inputs/outputs in order
- `_copy_gmtensor_data_into_sim` / `_copy_sim_tensor_back` — host <-> simulator runtime tensor marshalling

### Codegen emission (CANN deliverables)
- `_emit_kernel_sources` (L1070) — shared op-kernel writer reused by `generate` and `generate_debug` (emits used-micro headers, then `dump_kernel`)
- `_generate_debug_main` (L1087) — C++ `main.cpp` for the debug workspace (optional `profile` hooks)
- `_generate_debug_bashfiles` (L1305) — debug `b.sh` / `r.sh`; `cannsim=True` emits `-r sim` (CANNSIM) instead of `-r npu`
- `generate` (L1333) — full CANN op project (`generate_op_project` + `generate_op_host` + `_emit_kernel_sources` + `generate_aclnn_test` + `generate_bashfiles`)
- `generate_debug` (L1395) — standalone debug workspace (copies `debug/` resources, emits `main.cpp`, CMake); takes `cannsim=` forwarded to `_generate_debug_bashfiles`
- `generate_aclnn_test` (L1455) — aclnn test harness subdirectory
- `generate_bashfiles` (L1768) — build/run shell for the full project; `cannsim=True` switches the generated `r.sh` from `./test_aclnnop` to `cannsim record ./test_aclnnop -s <chipset>`

## Topic Lookup

### Autosync or event-order issues
1. `agent/references/constraints/sync.md`
2. `easyasc/parser/asc_autosync.py`
3. `easyasc/simulator/bridge.py` for control-flow kernels, where cube/vec streams are split and autosync events injected while building lane programs
4. `easyasc/simulator/sync.py` for same-lane event execution and pipe completion at `event_set` boundaries
5. `easyasc/parser/asc_handlers/`
6. relevant simulator pipe files under `easyasc/simulator/` (`pipe.py`, `pipe_cube.py`, `pipe_vec.py`)

### Suspected timing or generated-path mismatch
1. `dump_asc` / `dump_kernel` in `easyasc/kernelbase/kernelbase.py`
2. emitted `<path>_cube.h`, `<path>_vec.h`, and `<path>.cpp`
3. if the generated C++ order matches the intended model, inspect the relevant runtime files under `easyasc/simulator/`
4. if the generated C++ order is wrong, inspect `agent/references/constraints/sync.md`
5. then inspect `easyasc/parser/asc_autosync.py` and the relevant handler under `easyasc/parser/asc_handlers/`
6. report whether the mismatch is in generated C++, autosync/lowering, or simulator/runtime behavior before changing code

### Tiling or capacity legality
1. `agent/references/constraints/tiling.md`
2. `easyasc/shortcuts/matmul.py`
3. relevant kernel example from `agent/references/examples/kernel-catalog.md`
4. `agent/scripts/estimate_matmul_datamove.py`

### Device-specific kernel surface rules
1. `agent/references/constraints/a5.md` for a5 vec-side authoring limits
2. `agent/references/patterns/a5-mixed-pipeline.md` when the question is really about a5 cube↔vec handoff topology
3. `agent/references/constraints/a2.md` for a2 missing features and bridge rules
4. `easyasc/a5.py` and `easyasc/a2.py` only after the focused constraint file is no longer enough

### Counter lifetime or buffer ownership
1. `agent/references/constraints/sync.md`
2. `agent/references/patterns/buffer-slot-lifetime.md` when local roles or delayed beats overlap
3. target kernel source in `agent/example/kernels/`
4. lowering/runtime files only if behavior is still unclear

### Precision or cast placement
1. `agent/references/constraints/precision.md`
2. `agent/references/pattern-index.md` when the public ABI is a packed carrier
3. `easyasc/utils/`
4. matching lowering handler under `easyasc/parser/asc_handlers/`
5. simulator runtime if the question is execution-specific

### Tail handling or partial-tile writes
1. `agent/references/constraints/tail.md`
2. `agent/references/patterns/online-softmax-tail.md` when a running max/sum or causal score mask is involved
3. matching kernel example in `agent/example/kernels/`
4. parser/simulator files if codegen/runtime behavior is in doubt

### Parser-side dead-code or pruning behavior
1. `easyasc/parser/asc_pruning.py`
2. related tests under `agent/example/testcases/`

Dynamic helper views have side ownership. If a GM slice or reinterpret is
derived from `GetVecIdx()` / `GetSubBlockIdx()`, that strict vec provenance must
flow forward through `GetValueFrom`, `SetValueTo`, assigned `Var` values, and
the loop/condition frames controlled by those values. Do not keep a scalar
consumer on cube after pruning its helper-view definition. Static, unscoped GM
scalar access remains shared; this is provenance propagation, not a blanket
rule that all GM scalar access is vec-only. Regression coverage:
`agent/example/testcases/parser/core/test_gm_helper_side_propagation.py`.

### Shape binding ambiguity
1. `easyasc/torchplugin.py`
2. matching tests under `agent/example/testcases/`
3. affected kernel runner in `agent/example/kernels/`

Concrete trigger: when two scalar Var parameters have the same value at runtime
(e.g. `S1 == S2`), the framework cannot distinguish which scalar maps to which
tensor dimension. Fix: use `shape_bindings=` in `OpExec(...)()` call, or ensure
test shapes use distinct values for potentially ambiguous parameters.

For `GMTensorList`, bind shared per-item dimensions with the dictionary form:
`shape_bindings={0: {"item": [None, 0]}}`. The key is still the tensor-like
argument index, the binding is checked independently for every member, and the
list length is not part of `shape_bindings`. Leave item-dependent dimensions
unbound and read their flattened extent with `.item_numel(index)`.

### a2-specific cube → vec path
1. `agent/references/constraints/a2.md` — missing features and data path
2. `agent/references/patterns/a2-mixed-pipeline.md` — GM workspace bridge pattern
3. `agent/references/constraints/vec.md` — `cmax` / `brcb` row-stat semantics
4. `agent/references/constraints/vec.md` — continuous vs sliced vec ops
5. `agent/example/kernels/a2/attention/flash_attn_score_pv.py` — working reference kernel

### a5-specific cube → vec / vec → cube path
1. `agent/references/constraints/a5.md` — a5 vec-side limits, `micro`, and direct-handoff legal surfaces
2. `agent/references/patterns/a5-mixed-pipeline.md` — direct `L0C -> UB` and `UB -> L1*` handoff patterns
3. `agent/references/constraints/vec.md` — vec-side row ops, reductions, and sliced-vs-contiguous rules when the postprocess is still unclear
4. `agent/example/kernels/a5/matmul/matmul_abs_add1_vf.py` or `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py` — working reference kernels

### a2-specific pure vec elementwise / quantization kernels
1. `agent/references/constraints/a2.md` — pure vec entry structure and tile split
2. `agent/references/constraints/vec.md` — flag flow, `reinterpret`, mask semantics, and wide/narrow row interactions
3. `agent/references/constraints/precision.md` — exact hif8 rounding or cast-placement contract
4. `agent/example/kernels/a2/vec_only/to_hif8_torch.py` — working vec-only quantization reference

### a2-specific cube → vec → cube delayed pipeline
1. `agent/references/constraints/a2.md` — both bridge restrictions (`L0C -> GM -> UB` and `UB -> GM -> L1`)
2. `agent/references/patterns/a2-mixed-pipeline.md` — one-tile and grouped lookahead schedules, shared `L0C`, delayed consumer
3. `agent/references/constraints/vec.md` — running row-max update in aligned `[1, M]` scalar format plus `brcb`
4. `agent/references/constraints/vec.md` — sliced `sub` against narrow row-max broadcast
5. `agent/example/kernels/a2/attention/flash_attn_score_pv.py` — complete reference kernel

### a2-specific cube → vec → cube → vec delayed numerator accumulation
1. `agent/references/constraints/a2.md` — all three bridge restrictions plus a2 UB row-scalar layout and copy warning
2. `agent/references/patterns/a2-mixed-pipeline.md` — one-tile and grouped lookahead variants with delayed final vec accumulation
3. `agent/references/constraints/vec.md` — running max + delayed `expdiff` in aligned `[1, M]` scalar format
4. `agent/references/constraints/vec.md` — sliced scaling of `[M,128]` accumulators by narrow `[M,8]` broadcasts
5. `agent/example/kernels/a2/attention/flash_attn_full.py` — grouped full-attention reference (`GROUP_N=4`) with explicit local-event handoff for the final `accum_ub` store/reuse edge
6. `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` — one-tile probability-variant reference when the grouped schedule would obscure the simpler delayed-state pattern

### a2-specific cube → vec → cube → vec normalized online softmax
1. `agent/references/constraints/a2.md` — same triple-bridge hardware limits plus a2 UB row-scalar layout and copy warning
2. `agent/references/patterns/a2-mixed-pipeline.md` — normalized online-softmax schedules: one-tile baseline plus grouped `GROUP_N=4` variants
3. `agent/references/constraints/tail.md` — stable local shapes and generic tail legality
4. `agent/references/patterns/online-softmax-tail.md` — score-domain masking, S1/S2/causal timing, and "sum before cast" update order
5. `agent/references/constraints/vec.md` — `cadd` / `brcb` row-sum semantics and final sliced `div`
6. `agent/example/kernels/a2/attention/flash_attn_full.py` — grouped normalized reference kernel (`GROUP_N=4`, half `p`, fp32 output) with explicit local-event handoff for the final `accum_ub` store/reuse edge
7. `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8.py` — contract-first hif8 variant with `to_hif8_torch(p * 128) / 128`, validated non-aligned `S2` tail handling, explicit local-event handoff for the final `accum_ub` store/reuse edge, separate plain-`Tensor` vec scratch for `ub_score` / `ub_pv`, and exported final `rowmax` / `rowsum`
8. `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_commonub.py` — same scaled-hif8 probability family as item 7 with bf16 final `out`, but expands the delayed-drain schedule to depth-3 mutex queues and a three-slot `TBuff` expdiff scratch; use this when studying the three-slot scheduling variation without changing cross-side ownership
9. `agent/example/kernels/a2/attention/flash_attn_full_pj_hif8_causal.py` — left-up causal extension of the scaled-hif8 probability path with separate score / `pv` UB scratch, diagonal-tile causal masking, aligned `[1, HALF_M]` row-scalar state, `active_tiles_n = Min(tiles_n, lmt + 1)`, and exported `rowmax` / `rowsum`
10. `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal.py` — contract-first half value-path baseline with `pv_j = p_j.half().float() @ v_j.float()`, float `row_sum`, blockwise causal masking `floor(k_pos / 32) <= floor(q_pos / 32)`, direct aligned block32 diagonal masking, separate score/PV UB scratch, two-slot delayed `expdiff` queueing, contiguous M-tile scheduling, `active_tiles_n = Min(tiles_n, lmt + 1)`, and exported `rowmax` / `rowsum`
11. `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v3.py` — same math as item 10, but uses causal-work-balanced round-robin M-tile scheduling with odd-head M-order reversal, DBuff-backed `ub_score` / `ub_p` stage1 scratch, one Q GM->L1 load per M tile guarded by explicit `MTE2 -> MTE1` / `MTE1 -> MTE2` events, and depth-4 lookahead-3 queueing for score/p/pv/expdiff
12. `agent/example/kernels/a2/attention/block32_causal/flash_attn_full_pj_half_block32_causal_v2.py` — optimized version of item 10 with the same math contract, causal-work-balanced round-robin M-tile scheduling, depth-3 lookahead-2 queueing, a three-slot delayed `expdiff` buffer, independent output/trace paths, and exported `rowmax` / `rowsum`

### Generated host-side scalar dtype or project generation
1. `easyasc/kernelbase/kernelbase.py` — see `## KernelBase method map` to jump to the right emitter (`generate`, `generate_op_host`, `generate_aclnn_test`, `_generate_debug_main`, ...)
2. templates/resources under `easyasc/resources/`
3. related parser handlers if the generation path crosses lowering

### Test organization or test-asset lookup
1. `agent/example/testcases/README.md`
2. relevant tests or co-located sample assets under `agent/example/testcases/`
3. `agent/example/kernels/a2/` or `agent/example/kernels/a5/` (or `agent/example/projects/a5/gdn_fwd/` / `agent/example/projects/a5/gdn_bwd/` for GDN system projects) only if the target is intentionally manual or integration-style

## Reading Rule

Prefer:
1. one focused constraint/reference file
2. one implementation-path file or directory from this map
3. one test or source example

Do not jump into broad contributor architecture docs first unless the smaller path still leaves the question ambiguous.

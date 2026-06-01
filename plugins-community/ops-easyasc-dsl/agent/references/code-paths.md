# Code Paths

Use this file when you already know the topic and need to find the implementation path quickly.
Do not read the whole repository first.

## Device family mapping

Use this mapping before interpreting hardware-specific branches:
- `device_type == "950"` -> C310
- `device_type.startswith("b")` -> C220
- in `easyasc/resources/tensorutils.h`, `__DAV_C310__` corresponds to `950`
- in `easyasc/resources/tensorutils.h`, `__DAV_C220_CUBE__` corresponds to `b*`
- do not invert this mapping when reading support checks, helper selection, or generated code paths

## Common Questions -> Files

### Repository structure or contributor architecture
- top-level layout: `agent/references/repo-map.md`
- contributor architecture and subsystem ownership: `doc/11_architecture_for_contributors.md`

### Public DSL surface
- architecture-specific exports: `easyasc/a2.py`, `easyasc/a5.py`
- treat `a2` and `a5` as parallel architecture surfaces for different instruction sequences, not as compatibility layers of each other
- stable vec authoring rule for a5: use `@vf()`, `micro`, sort-family ops, or `ub_to_ub`; do not treat raw `easyasc.a5` exports as permission to write generic vec math directly in the kernel body
- stable vec authoring rule for a2: direct vec ops in the kernel body are normal, but `@vf()` and `micro` are unavailable
- decorators: `easyasc/decorators.py`
- flow helpers: `easyasc/flowcontrol.py`
- syntax sugar / AST rewrites: `easyasc/pythonic.py`
  - native Python `if` / `elif` / `else` inside `@kernel` are rewritten into DSL `with If(...)`, `with Elif(...)`, and `with Else()` blocks before instruction capture
  - prefer native Python control flow in kernels and kernel-focused tests; inspect `easyasc/flowcontrol.py` only when you need the emitted `start_if` / `start_elif` / `start_else` / `end_if` details
  - regression/example: `testcases/simulator/bridge/test_sim_nested_if_elif_else.py`
- runtime entry and `OpExec(..., simulator=...)` backend selection: `easyasc/torchplugin.py`
  - `simulator=True`, `simulator="v2"`, and `simulator="legacy"` currently all route to the V2 runtime through `KernelBase.run_sim()`
  - simulator mode does not emit generated code artifacts; if you only want codegen output, use `simulator=False, gen_only=True`

### Kernel and micro execution wrappers
- kernel lifecycle / generation hooks: `easyasc/kernelbase/kernelbase.py` (see `## KernelBase method map` below for a per-method index)
  - simulator entry: `KernelBase.run_sim()` -> `_run_sim_v2()`
  - auto bridge selection: `build_simulator_v2_program()`
- micro module lifecycle: `easyasc/micro/micromodule.py`

### Lowering pipeline
- main lowering entry: `easyasc/parser/asc.py`
- pruning and cleanup: `easyasc/parser/asc_pruning.py`
- autosync insertion: `easyasc/parser/asc_autosync.py`
- op-specific lowering: `easyasc/parser/asc_handlers/`

### Simulator path
- focused simulator map: `agent/references/simulator-v2.md`
- runtime root: `easyasc/simulator_v2/`
- runtime activation and core/lane planning: `easyasc/simulator_v2/runtime/execution_plan.py`
- parent runtime coordinator: `easyasc/simulator_v2/runtime/global_runtime.py`
- per-core child process wrapper: `easyasc/simulator_v2/runtime/core_process.py`
- per-core runtime: `easyasc/simulator_v2/runtime/core_runtime.py`
- default narrow linear bridge: `easyasc/simulator_v2/compat/kernel_bridge.py`
- control-flow bridge for looped / mixed-lane kernels: `easyasc/simulator_v2/compat/control_flow_bridge.py`
- V2 lane-level control interpreter and local event execution: `easyasc/simulator_v2/runtime/control_actor.py`
  - dynamic `get_buf` slot selection for loop-carried DBuff/TBuff/QBuff access is bridged here and resolved at runtime here
  - `create_varlist`, `GetValueFrom`, and `SetValueTo` also route through this control-flow path, including dynamic `VarList` index resolution
  - float `Var` arithmetic on control-flow paths (`var_add`, `var_mul`, `var_div`) is also executed here and must stay float when the source `Var` is float
- V2 shared tensor and workspace storage: `easyasc/simulator_v2/memory/`
- V2 local bank allocation and per-core local-memory capacity checks: `easyasc/simulator_v2/memory/local_memory.py`, wired from `easyasc/simulator_v2/runtime/core_runtime.py`
- V2 threaded pipe-worker lifecycle and pipe-local state retention: `easyasc/simulator_v2/runtime/pipe_worker.py`, `easyasc/simulator_v2/ops/dispatch.py`
  - vec mask state such as `set_mask/reset_mask` persists here across sequential V-pipe tasks on the same lane
- V2 lane-local event-bank semantics for `create_sevent` / `create_devent` / `event_wait` / `event_set`:
  `easyasc/simulator_v2/sync/local_events.py` plus `easyasc/simulator_v2/runtime/control_actor.py`
- V2 pre-dispatch pipe-task memory-range validation for tensor-touching execution paths including shared-tensor helpers (`fill_shared`, `copy_shared`), all current cube-pipe ops (`gm_to_l1_nd2nz`, `set_constant_to_l1`, `l1_to_l0`, `mmad`, `l0c_to_*`, `cube_m_*`), vec datamoves, V-pipe tensor ops such as packed `compare`/`select`, repeat-layout vec ops, `sort32`, `mergesort*`, `gather`, `scatter`, plus task-level micro shared-tensor ops and `call_micro` dry-run validation: `easyasc/simulator_v2/runtime/task_memory_validator.py`, called from `easyasc/simulator_v2/runtime/pipe_worker.py`
- V2 vec execution helpers: `easyasc/simulator_v2/ops/vec/`
- V2 micro runtime: `easyasc/simulator_v2/ops/micro/runtime.py`
  - float `Var` arithmetic inside `call_micro` execution is resolved here again; if a `@vf()` scale unexpectedly becomes `0`, inspect this file before blaming cube/vec handoff
- V2 Chrome-trace export path: `easyasc/simulator_v2/trace/chrome.py`
- A5 cycle-model profile / estimator path for simulator-side timing traces: `easyasc/simulator_v2/timing/`

### Stub emitters and shortcuts
- operation emitters: `easyasc/stub_functions/`
- high-level matmul helpers: `easyasc/shortcuts/matmul.py`

### Runtime types and helper objects
- tensors, vars, regs, enums, instruction models: `easyasc/utils/`

### Test organization and runnable demos
- manual/demo runners: `agent/example/demo/a2/`, `agent/example/demo/a5/`
- historical note: the `testcases/` tree has been removed from the delivered skill bundle; references to `testcases/...` elsewhere in this file describe historical regression coverage only

## KernelBase method map

`easyasc/kernelbase/kernelbase.py` is a single ~1.5k-line file holding the `KernelBase` class. Use this map to jump by method; prefer the method name over the line number when the file has shifted.

### Construction and call
- `__init__` (L25) — store name/func/vector_mode; init instruction, mutex, workspace, and used-micro state
- `__call__` (L41) — bind args and run the wrapped function to emit instructions

### Developer-facing dumps
- `print_instructions` (L133) — print the captured instruction list
- `dump_asc` (L141) — write `<path>_cube.h` + `<path>_vec.h` from `translate_split`
- `dump_kernel` (L149) — write `<path>_cube.h` + `<path>_vec.h` + `<path>.cpp` entry with optional `debug_entry`

### CANN op-host and op-project scaffolding
- `generate_op_host` (L304) — write `<name>_tiling.h` + `<name>.cpp` (infoshape, tiling, op registration); requires the kernel to have been called once
- `generate_op_project` (L571) — unpack `CustomOp*.tar.gz` (a2 vs a5 by `device_type`), materialize `CMakePresets.json` and `build.sh`
- `_resolve_op_host_block_dim` (L280) / `_resolve_debug_chipset` (L291) — device-type helpers for op-host output
- `_resolve_custom_opp_path` (L629) — normalize vendor opp install path

### Simulator V2 orchestration
- `run_sim` (L638) — public simulator entry; routes to `_run_sim_v2`
- `_analyze_usage_for_simulator_v2` (L641) — pre-run usage inspection
- `build_simulator_v2_program` (L655) — auto-select linear vs control-flow bridge
- `_has_control_flow_instructions` (L675) — bridge-selection predicate
- `resolve_simulator_v2_config` (L695) — produce V2 runtime config
- `_iter_bound_gmtensors` (L711) — iterate bound GM inputs/outputs in order
- `_copy_gmtensor_data_into_v2_runtime` (L719) / `_copy_v2_runtime_tensor_back` (L746) — host <-> V2 runtime tensor marshalling
- `_run_sim_v2` (L759) — build program, execute, copy tensors back, optional trace

### Codegen emission (CANN deliverables)
- `_emit_kernel_sources` (L786) — shared op-kernel writer reused by `generate` and `generate_debug` (emits used-micro headers, then `dump_kernel`)
- `_generate_debug_main` (L801) — C++ `main.cpp` for the debug workspace (optional `profile` hooks)
- `_generate_debug_bashfiles` (L1008) — debug `b.sh` / `r.sh`
- `generate` (L1030) — full CANN op project (`generate_op_project` + `generate_op_host` + `_emit_kernel_sources` + `generate_aclnn_test` + `generate_bashfiles`)
- `generate_debug` (L1088) — standalone debug workspace (copies `debug/` resources, emits `main.cpp`, CMake)
- `generate_aclnn_test` (L1145) — aclnn test harness subdirectory
- `generate_bashfiles` (L1458) — build/run shell for the full project

## Topic Lookup

### Autosync or event-order issues
1. `agent/references/constraints/autosync.md`
2. `easyasc/parser/asc_autosync.py`
3. `easyasc/simulator_v2/compat/control_flow_bridge.py` for control-flow kernels, where V2 first splits cube/vec streams and injects autosync events before building lane programs
4. `easyasc/simulator_v2/runtime/control_actor.py` for V2 same-lane event execution and threaded pipe completion at `event_set` boundaries
5. `easyasc/parser/asc_handlers/`
6. relevant simulator pipe/runtime files under `easyasc/simulator_v2/`

### Tiling or capacity legality
1. `agent/references/constraints/tiling.md`
2. `easyasc/shortcuts/matmul.py`
3. relevant kernel example from `agent/references/examples/kernel-catalog.md`
4. `agent/scripts/estimate_matmul_datamove.py`

### Device-specific kernel surface rules
1. `agent/references/constraints/a5-device.md` for a5 vec-side authoring limits
2. `agent/references/constraints/a2-device.md` for a2 missing features and bridge rules
3. `easyasc/a5.py` and `easyasc/a2.py` only after the focused constraint file is no longer enough

### Counter lifetime or buffer ownership
1. `agent/references/constraints/counters.md`
2. target kernel source in `agent/example/kernels/`
3. lowering/runtime files only if behavior is still unclear

### Precision or cast placement
1. `agent/references/constraints/precision.md`
2. `easyasc/utils/`
3. matching lowering handler under `easyasc/parser/asc_handlers/`
4. simulator runtime if the question is execution-specific

### Tail handling or partial-tile writes
1. `agent/references/constraints/tail-safety.md`
2. matching kernel example in `agent/example/kernels/`
3. parser/simulator files if codegen/runtime behavior is in doubt

### Parser-side dead-code or pruning behavior
1. `easyasc/parser/asc_pruning.py`
2. related tests under `testcases/`

### Shape binding ambiguity
1. `easyasc/torchplugin.py`
2. matching tests under `testcases/`
3. affected kernel runner in `agent/example/kernels/`

Concrete trigger: when two scalar Var parameters have the same value at runtime
(e.g. `S1 == S2`), the framework cannot distinguish which scalar maps to which
tensor dimension. Fix: use `shape_bindings=` in `OpExec(...)()` call, or ensure
test shapes use distinct values for potentially ambiguous parameters.

### a2-specific cube → vec path
1. `agent/references/constraints/a2-device.md` — missing features and data path
2. `agent/references/patterns/a2-cube-vec.md` — GM workspace bridge pattern
3. `agent/references/constraints/vec-reduction-a2.md` — cmax + brcb for row max
4. `agent/references/constraints/vec-stride.md` — continuous vs sliced vec ops
5. `agent/example/kernels/a2/flash_attn_score.py` — working reference kernel

### a2-specific pure vec elementwise / quantization kernels
1. `agent/references/constraints/a2-vec-kernel.md` — pure vec kernel structure, flag flow, reinterpret, exact rounding
2. `agent/references/constraints/mask.md` — current vec mask semantics when masking is truly needed
3. `agent/references/constraints/vec-stride.md` — only if wide/narrow row interaction appears
4. `agent/example/kernels/a2/to_hif8_torch.py` — working vec-only quantization reference

### a2-specific cube → vec → cube delayed pipeline
1. `agent/references/constraints/a2-device.md` — both bridge restrictions (`L0C -> GM -> UB` and `UB -> GM -> L1`)
2. `agent/references/patterns/a2-cube-vec-cube.md` — one-tile lookahead schedule, shared `L0C`, delayed consumer
3. `agent/references/constraints/vec-reduction-a2.md` — running row-max update in `[M,1]` scalar format
4. `agent/references/constraints/vec-stride.md` — sliced `sub` against narrow row-max broadcast
5. `agent/example/kernels/a2/flash_attn_score_pv.py` — complete reference kernel

### a2-specific cube → vec → cube → vec delayed numerator accumulation
1. `agent/references/constraints/a2-device.md` — all three bridge restrictions plus scalar-state copy warning
2. `agent/references/patterns/a2-cube-vec-cube-vec.md` — one-tile lookahead with delayed final vec accumulation
3. `agent/references/constraints/vec-reduction-a2.md` — running max + delayed `expdiff` in `[M,1]` scalar format
4. `agent/references/constraints/vec-stride.md` — sliced scaling of `[M,128]` accumulators by narrow `[M,8]` broadcasts
5. `agent/example/kernels/a2/flash_attn_unnorm.py` — complete reference kernel with explicit local-event handoff for the final `accum_ub` store/reuse edge

### a2-specific cube → vec → cube → vec normalized online softmax
1. `agent/references/constraints/a2-device.md` — same triple-bridge hardware limits plus scalar-state copy warning
2. `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md` — one-tile lookahead with running `row_max`, running `row_sum`, delayed `expdiff`, and final divide
3. `agent/references/constraints/online-softmax-tail.md` — score-domain `-inf` masking rule for non-aligned `S2` tails
4. `agent/references/constraints/reduction.md` — online softmax update order and "sum before cast" rule
5. `agent/references/constraints/vec-reduction-a2.md` — `cadd`/`brcb` row-sum pattern and final sliced `div`
6. `agent/references/constraints/vec-stride.md` — sliced division of `[M,128]` accumulators by `[M,8]` broadcasts
7. `agent/example/kernels/a2/flash_attn_full.py` — complete normalized reference kernel with explicit local-event handoff for the final `accum_ub` store/reuse edge
8. `agent/example/kernels/a2/flash_attn_full_pj_hif8.py` — contract-first hif8 variant with `to_hif8_torch(p * 128) / 128`, validated non-aligned `S2` tail handling, explicit local-event handoff for the final `accum_ub` store/reuse edge, separate plain-`Tensor` vec scratch for `ub_score` / `ub_pv`, and exported final `rowmax` / `rowsum`
9. `agent/example/kernels/a2/flash_attn_full_pj_hif8_commonub.py` — same math and outputs as item 8, but rewrites the vec scratch into shared `DBuff` lineage (`ub_score_pv` + `score_pv_cnt`) so the vec `MTE2 -> V` `ubin` edge follows slot-buffer `DEvent`-style queueing; use this when studying why the shared local-buffer version overlaps better without merging `stage1_cnt` and `stage2_cnt`
10. `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py` — left-up causal extension of the scaled-hif8 probability path, now also moved onto the shared vec-side `DBuff` lineage (`ub_score_pv` + `score_pv_cnt`) so the diagonal-tile causal kernel keeps the same improved `MTE2 -> V` `ubin` queueing as the non-causal `commonub` baseline while preserving `active_tiles_n = Min(tiles_n, lmt + 1)` and exported `rowmax` / `rowsum`
11. `agent/example/kernels/a2/flash_attn_full_pj_half_block32_causal.py` — contract-first half value-path variant with `pv_j = p_j.half().float() @ v_j.float()`, float `row_sum`, blockwise causal masking `floor(k_pos / 32) <= floor(q_pos / 32)`, diagonal-tile packed-bit masks with `32/64` valid-column thresholds per subblock half, and the same shared vec-side `DBuff` lineage (`ub_score_pv` + `score_pv_cnt`) to improve local `ubin` queueing while preserving `active_tiles_n = Min(tiles_n, lmt + 1)` and exported `rowmax` / `rowsum`

### Generated host-side scalar dtype or project generation
1. `easyasc/kernelbase/kernelbase.py` — see `## KernelBase method map` to jump to the right emitter (`generate`, `generate_op_host`, `generate_aclnn_test`, `_generate_debug_main`, ...)
2. templates/resources under `easyasc/resources/`
3. related parser handlers if the generation path crosses lowering

### Test organization or test-asset lookup
1. `testcases/README.md`
2. relevant tests or co-located sample assets under `testcases/`
3. `agent/example/demo/a2/` or `agent/example/demo/a5/` only if the target is intentionally manual or integration-style

## Reading Rule

Prefer:
1. one focused constraint/reference file
2. one implementation-path file or directory from this map
3. one test or source example

Do not jump into broad contributor architecture docs first unless the smaller path still leaves the question ambiguous.

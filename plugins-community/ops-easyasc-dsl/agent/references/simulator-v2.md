# Simulator V2 Reference

Read this file when the question is specifically about how simulator execution works now.
Do not use it as a replacement for kernel-authoring or general architecture docs.

## Goal

Capture the current simulator execution path so future work does not rely on removed
or stale `easyasc/simulator/` assumptions.

## 1. Current default

The repository's simulator path is now the V2 runtime.

Current behavior:
- `OpExec(..., simulator=True)` enables simulator execution
- `OpExec(..., simulator="v2")` is an accepted spelling for the same path
- `OpExec(..., simulator="legacy")` is still accepted by `OpExec`, but it does **not**
  select a separate old runtime anymore; it still routes to V2
- `KernelBase.run_sim()` always calls `_run_sim_v2()`

Practical rule:
- do not document or debug a separate `easyasc/simulator/` runtime as if it were still active

## 2. How a kernel becomes a V2 program

The simulator build entry lives in `easyasc/kernelbase/kernelbase.py`.

The selection order is:
1. custom builder via `kernel._simulator_v2_program_builder`
2. prebuilt program via `kernel._simulator_v2_program`
3. auto analysis + auto bridge selection

Auto bridge selection:
- if the instruction stream contains control-flow, topology queries, `call_micro`,
  `VarList`, or cross-lane sync helpers, V2 uses
  `easyasc/simulator_v2/compat/control_flow_bridge.py`
- otherwise V2 uses the narrow linear bridge in
  `easyasc/simulator_v2/compat/kernel_bridge.py`

Important difference:
- `control_flow_bridge.py` preserves loops/conditionals and defers resolution to the runtime
- `kernel_bridge.py` only covers a narrower linear lowered-instruction subset

## 3. Runtime stack

The runtime is split across these layers:

- parent coordinator: `easyasc/simulator_v2/runtime/global_runtime.py`
- core process wrapper: `easyasc/simulator_v2/runtime/core_process.py`
- per-core runtime: `easyasc/simulator_v2/runtime/core_runtime.py`
- lane-level control interpreter: `easyasc/simulator_v2/runtime/control_actor.py`
- pipe worker threads: `easyasc/simulator_v2/runtime/pipe_worker.py`
- pipe executors: `easyasc/simulator_v2/ops/`

Execution shape:
- one parent `GlobalRuntime`
- one child `CoreProcess` per simulated core
- inside each active core, one `ControlActor` per active lane
- inside each lane, one threaded `PipeWorker` per logical pipe

Launch rule:
- start simulator repros from a real `.py` file, not from `stdin` entry points such as
  `python - <<'PY'` or piped scripts
- V2 uses multiprocessing during startup, and Python spawn must be able to re-import `__main__`
  from a real filesystem path; `stdin` entry points appear as `<stdin>` and break child startup
- when the launcher lives outside the repo root, include the repo root in `PYTHONPATH` so child
  processes can import local modules consistently
- safe pattern: `PYTHONPATH=/abs/path/to/repo python /tmp/repro.py`

Completion / shutdown facts:
- pipe workers already stop through mailbox sentinels; the thread layer does not need a special end instruction
- parent / child completion now uses a one-shot status channel that the parent polls while joining
- `GlobalRuntime.run()` uses one global execution deadline across all active cores, not a full timeout budget per core in sequence

## 4. Planning and activation

Core and lane activation are resolved by:
- `easyasc/simulator_v2/config.py`
- `easyasc/simulator_v2/runtime/execution_plan.py`
- `easyasc/simulator_v2/helpers.py`

Key facts:
- default core count follows the active device family (`950 -> 32`, `b3 -> 20`)
- V2 can skip inactive lanes when a program only uses a subset of cube/vec lanes
- collective ops (`allcube_*`, `allvec_*`) affect lane-activation planning

## 5. Memory and tensor state

Shared tensor setup lives in:
- `easyasc/simulator_v2/memory/shared_tensor.py`
- `easyasc/simulator_v2/memory/shared_tensor_store.py`
- `easyasc/simulator_v2/memory/tensor_view.py`
- `easyasc/simulator_v2/memory/workspace.py`
- `easyasc/simulator_v2/memory/local_memory.py`

Important facts:
- `OpExec` clones input tensors into `GMTensor.data`
- V2 copies that payload into the shared runtime tensor store before execution
- after execution, V2 copies runtime tensors back into the bound `GMTensor.data`
- workspaces and local buffers are represented as shared-tensor specs in program metadata
- child-core local tensors now go through a bank-aware allocator (`UB0/UB1/L1/L0A/L0B/L0C`);
  over-capacity local allocations fail before pipe execution starts
- runtime-created local slice snapshots must treat a root local tensor's
  `SharedTensorSpec.storage_offset` as allocator bookkeeping in bytes rather
  than as an extra in-storage element offset; only nested local views should
  re-apply a parent `storage_offset` when `control_actor.py` materializes a
  dynamic slice
- simulator-side GM `atomic_add` / `atomic_max` / `atomic_min` now serialize their
  read-modify-write sections through a shared store-wide atomic lock so cross-core
  atomic writebacks do not lose updates under contention

Regression note:
- `testcases/simulator/memory/test_simulator_v2_slice_tensor.py` covers the
  sliced-UB vec-mul case where several prefix UB allocations push the sliced
  root tensor onto a non-zero local bank offset before runtime snapshotting

## 6. Sync and control

The main sync/control pieces are:
- intra-core sync: `easyasc/simulator_v2/sync/intra_core_sync.py`
- collective sync: `easyasc/simulator_v2/sync/collective_sync.py`
- lane-local flags: `easyasc/simulator_v2/sync/local_flags.py`
- lane-local events: `easyasc/simulator_v2/sync/local_events.py`
- worker mailboxes: `easyasc/simulator_v2/sync/mailbox.py`

Important fact:
- collective sync state is process-shared at runtime; `GlobalRuntime` snapshots the parent
  `CollectiveSync` and each child core reloads that shared state instead of creating a
  private per-process coordinator
- lane-local `barrier(pipe=...)` currently has special runtime behavior only for
  `barrier(ALL)`; non-`ALL` barriers are preserved as control instructions but act as
  no-ops in the V2 runtime main loop
- practical consequence for kernel debugging: `bar_v()` / `bar_mte2()` / other single-pipe
  barriers do not serialize cross-pipe edges such as `V -> MTE2` on the simulator path; when
  a repro needs a simulator-visible local drain across pipe domains, use `bar_all()`
- `setflag` / `waitflag` still use the phase-based `LocalFlagTable`, but local `SEvent` / `DEvent`
  no longer do: V2 now models them with a per-lane flag bank keyed by
  `(src_pipe, dst_pipe, flag_id)` and a bool value per flag
- `create_sevent` allocates one `flag_id` from the lane-local pool for its `(src_pipe, dst_pipe)`
  pair; `create_devent` allocates two consecutive ids from that same pair-local pool
- `SEvent.set()` sets its single flag to `1` and errors if it is already `1`; `SEvent.wait()`
  blocks until that flag becomes `1`, then clears it back to `0`
- `DEvent` keeps two independent bool flags plus separate `set_count` / `wait_count` cursors:
  the producer-side `set` path alternates `flag0, flag1, flag0, ...`, and the consumer-side
  `wait` path alternates on its own cursor over the same two flags
- `event_setall` is modeled as repeated `set()` calls on the same event object rather than as a
  special bulk primitive; for `DEvent` that usually means setting both flags in rotation order,
  while `SEvent.setall()` will replay `set()` twice and therefore errors on the second call if
  the single flag is still set
- `event_release` is modeled as repeated `wait()` calls: `SEvent.release()` performs one wait,
  while `DEvent.release()` performs one wait and then performs a second wait only when a second
  outstanding token is already pending on the other rotated flag
- practical consequence for trace/timing work: local event blocking must now be reasoned about
  per real `flag_id`, not per `event_name`
- regression coverage: `testcases/simulator/bridge/test_simulator_v2_control_flow.py`

When debugging a hang:
- inspect the original failing lane error first
- then inspect the sync state / timeout diagnostic
- do not assume the timeout itself is the root cause

When a child core raises an exception:
- `GlobalRuntime.run()` now raises the combined per-core traceback text directly
- do not rely on a generic parent-side wrapper message; the actionable failure should
  already be in the thrown exception string
- pipe-worker instruction failures now print an immediate `stderr` log with
  `lane/pipe/opname/error`, control-side `wait_*` paths poll worker failures while
  waiting, and `CoreRuntime.join()` prefers surfacing the more actionable worker/task
  failure over a secondary sync-timeout symptom when multiple lane actors fail

## 7. Trace path

Trace recording lives in:
- `easyasc/simulator_v2/trace/recorder.py`
- `easyasc/simulator_v2/trace/merge.py`
- `easyasc/simulator_v2/trace/chrome.py`
- a5 cycle-model profile and estimators: `easyasc/simulator_v2/timing/`

Runtime flow:
- each core records its own events
- parent runtime merges them after execution
- `dump_chrome_trace(...)` exports Chrome/Perfetto-style JSON
- runtime event timestamps originate from `time.monotonic()`
- exported Chrome traces normalize those timestamps into a per-run relative axis instead of replacing them with event-order indices
- exported `dur` now reflects measured task/wait spans when the runtime recorded them; zero-duration control markers still use a tiny fallback width only to stay visible in viewers
- sync-heavy kernels may now emit explicit `sync` trace events for wait/ready phases in addition to pipe execution events
- on a5 (`device_type == "950"`), the runtime can now switch trace timing to a cycle-model domain driven by the JSON profile under `timing/`; in that mode `easyasc_time_domain == "cycle"` is exported in the trace payload and task args include the modeling breakdown
- current a5 cycle-model defaults treat one ordinary V-pipe instruction as `2` cycles
- for `call_micro` / `@vf()` timing, register <-> UB shuffle instructions are counted as `0` cycle:
  `micro_ub2reg`, `micro_reg2ub`, `micro_ub2regcont`, `micro_reg2ubcont`
- in cycle-model mode, direct control-side waits (`event_wait`, `wait_vec`, `wait_cube`, collective waits) now advance the control actor's cycle cursor, but `event_set` no longer acts as a lane-global block for later unrelated pipe dispatch; its ready time is derived from the completed source pipe, and unrelated pipes can start as soon as their own event dependencies are satisfied
- lane-local `event_wait` / `event_release` can now be lowered into the destination pipe worker queue, so the blocking happens on that pipe thread instead of only on the control actor; `event_set` / `event_setall` intentionally stay control-side because their position in the instruction stream still defines autosync lifetime boundaries
- trace export now consults `globvars.trace_event` (default `False`): when disabled, all sync-style trace markers are omitted from dispatch, pipe, and sync tracks, including lane-local `event_*`, local flag waits, intra-core handoff ops such as `wait_vec` / `cube_ready`, and collective `all*` sync ops; tests or debugging sessions that need those markers must enable the flag explicitly before running the simulator
- when optimizing from the trace view, keep `globvars.trace_event` at its default `False` unless the specific goal is to inspect sync/event behavior; turning it on adds sync markers that are useful for debugging but can distract from the steady-state scheduling picture you usually want for optimization work
- when optimizing cycle count from a trace, use the trace makespan as the objective: the cycle at which the last timed event finishes (`max(ts + dur)` over `ph == "X"` events). Do not optimize for the sum of all timed durations or "total activated cycles"; those overcount parallel overlap and can rank kernels differently from the real end-to-end completion time

## 8. Vec and micro execution

Key implementation files:
- vec runtime entry: `easyasc/simulator_v2/ops/vec/v.py`
- vec legacy-layout helper: `easyasc/simulator_v2/ops/vec/_legacy_vpipe.py`
- vec MTE2 path: `easyasc/simulator_v2/ops/vec/mte2.py`
- vec MTE3 path: `easyasc/simulator_v2/ops/vec/mte3.py`
- micro runtime: `easyasc/simulator_v2/ops/micro/runtime.py`
- pipe dispatch: `easyasc/simulator_v2/ops/dispatch.py`

Important fact:
- several vec operations still reuse the legacy layout executor through
  `ops/vec/_legacy_vpipe.py`, but they run inside the V2 runtime
- when `gm_to_ub_pad` or `l0c_to_gm_nz2nd` reports a source/destination view that
  is "too small" on an a2 workspace-mediated tail path, first inspect whether the
  workspace view was cropped in the column dimension; those bridge ops infer
  row-stride from the parent GM shape, so a cropped workspace column span can
  fail even when the logical tail math is correct
- all UB burst copy ops (`gm_to_ub_pad` in `ops/vec/mte2.py`, `ub_to_gm_pad` and
  `ub_to_l1_nz` in `ops/vec/mte3.py`) use `_linear_view_from_pointer` so that
  column-sliced UB views (`ub[:, 0:valid_n]` with `valid_n < buffer_cols`)
  round-trip through the underlying storage; any new burst-style op must mirror
  this pattern or it will falsely raise "view is too small" when the destination
  is non-contiguous
- regression coverage: `testcases/simulator/datamove/test_gm_to_ub_pad_column_slice.py`

Scalar-semantics reminder:
- `control_flow_bridge.py` preserves `Var` arithmetic as runtime scalar ops such as
  `var_add`, `var_mul`, and `var_div`
- `control_actor.py` and `ops/micro/runtime.py` must preserve float `Var` semantics for
  those ops; do not silently coerce float scalar expressions to int on the runtime path
- practical symptom of a broken float-scalar path: raw cube/UB data looks correct, but a
  later `@vf()` stage that multiplies by a computed scale suddenly collapses to `0`

## 9. Best first files for simulator debugging

- `easyasc/kernelbase/kernelbase.py`
- `easyasc/simulator_v2/compat/control_flow_bridge.py`
- `easyasc/simulator_v2/compat/kernel_bridge.py`
- `easyasc/simulator_v2/runtime/control_actor.py`
- `easyasc/simulator_v2/runtime/task_memory_validator.py`
  - pre-dispatch memory-range checks now cover shared-tensor helpers, all current cube-pipe tensor ops, vec datamoves, V-pipe tensor ops including packed `compare`/`select`, repeat-layout vec instructions, `sort32`, `mergesort*`, `gather`, `scatter`, task-level micro shared-tensor ops, and `call_micro` dry-run validation
- `easyasc/simulator_v2/runtime/pipe_worker.py`
- `easyasc/simulator_v2/runtime/global_runtime.py`
- `testcases/simulator/`

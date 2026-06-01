# Kernel Debugging Playbook

Use this playbook when an existing kernel is wrong, unstable, warning-heavy, or unclear. Debug in layers. Do not jump between random fixes.

## Goal

Find the first broken assumption. Fix the model, then fix the kernel. Do not keep stacking patches on top of an unclear design.

## Fast-path: match your symptom first

Most bug reports match one of the patterns below. Try these before running the full layer-by-layer review further down.

### Symptom-to-check map

- **Wrong everywhere** → check formula, transpose/layout, cast order, `shape_bindings`
- **Only large shapes fail** → check tile budgets, split mode, estimator choice, counter ownership across nested loops
- **Only tail tiles fail** → check `valid_*` handling, half-row vec writeback split, GM boundary slicing
- **Autosync warnings or weird pipeline stalls** → check same-side vs cross-side misunderstanding, event family grouping, counter reuse across different lifetimes, unsupported instruction not covered by autosync pairing
- **Local event timeout / already-set (`_tmp_*valid_*`, `_tmp_*ready_*`)** → classify the event failure first, dump autosync-expanded instructions, then compare the failing family against a stable kernel before changing the DSL
- **Simulator passes, generated path looks suspicious** → check parser lowering, codegen handlers, explicit event or mutex placement, assumptions hidden by simulator convenience
- **V2 `wait_vec` / `wait_cube` timeout** → see the V2 timeout section below; almost always the other lane's actor crashed silently
- **Kernel only fails when run alongside other tests** → see the V2 parallel-process section below

### Event pairing workflow for local event failures

Use this when V2 reports a lane-local event problem such as:
- `event_wait timeout: {'name': '_tmp_sevent_valid_fix_0', ...}`
- `event_set on already-set flag: _tmp_sevent_valid_l1_0 ...`

Debugging sequence:
1. **Classify the failure from the runtime message.**
   - `event_wait timeout` usually means a missing `event_set` for the same family.
   - `event_set on already-set flag` usually means a duplicate `event_set` before a matching `event_wait` consumed the token.
2. **Read the counters literally.**
   - On the simulator path, `preset=True` events start with one published token.
   - If a timeout reports `set_count == wait_count`, the preset token was consumed and the next producer-side `event_set` never happened.
3. **Build the kernel instructions before inspecting split/autosync output.**
   - Call the `@kernel` once with placeholder `GMTensor(...)` arguments so `kernel.instructions` is populated.
4. **Dump the autosync-expanded lane instructions.**
   - Use `split_instructions(...)` plus `insert_auto_sync(...)`, then inspect only the failing side (`cube` or `vec`).
   - Prefer printing just one family at a time: `l1`, `l0`, `fix`, `ubin`, or `ubout`.
5. **Turn the event stream into an action sequence.**
   - Record only `event_wait` / `event_set` for the failing event name(s).
   - Healthy reuse should look like alternating publish/consume rounds; repeated `wait` or repeated `set` without the opposite action in between is the broken edge.
6. **Compare against a stable baseline kernel.**
   - Dump the same family from a nearby working kernel and diff the action sequence.
   - This is often faster than reasoning from the fused kernel body.
7. **Check nested autosync ownership next.**
   - If the failing edge sits around nested `start_loop` / `start_if` regions, inspect parent/child mixed-scope handling before touching the kernel.
   - In particular, confirm whether parent and child are really the same autosync family, not just the same pipe pair.
8. **Add a parser regression before rerunning the real kernel.**
   - Put the minimal reproducer in `testcases/parser/sync/test_autosync_event_metadata.py`.
   - Fix the split/autosync behavior there first, then rerun the full simulator kernel.

When this workflow points to parser behavior, jump to:
- `agent/references/constraints/autosync.md`

### V2 simulator: `CvMutex` / `VcMutex` timeout (`wait_vec` / `wait_cube`)

When V2 reports a sync timeout such as:
```
wait_vec timeout: {'scope': 'intra_core', 'name': 'vec_to_cube_0_0',
                   'target_phase': 3, 'current_phase': 2, 'consumed_phase': 2}
```

The timeout almost always means the **other lane's actor thread crashed**, not that the sync logic itself is wrong. The crashing thread silently terminates, so the expected `vec_ready` or `cube_ready` signal is never published, and the waiting side eventually times out.

Debugging sequence:
1. **Capture the real error on the other lane first.** Patch `CoreRuntime.start` to wrap each `ControlActor.start()` in a try/except that prints the lane name and exception. The first non-timeout error is the real root cause.
2. Common root causes behind the silent crash:
   - **Float8 indexing**: PyTorch does not support `tensor[indices]` for `float8_e5m2` / `float8_e4m3fn`. Any `_gather_1d`, `_scatter_1d`, or fancy indexing on a float8 register or UB tensor raises `"index_cpu" not implemented for 'Float8_e5m2'`. Fix by viewing as `torch.uint8` before indexing.
   - **Non-contiguous UB views in burst copy**: `ub_to_gm_pad` / `ub_to_l1_nz` use `.view(torch.uint8)` on the source. A column slice (stride > 1) makes `.view()` fail. Fix with `_linear_view_from_pointer()`.
   - **Micro op not implemented**: a `@vf()` body calls an op that `MicroRuntime` does not dispatch (`NotImplementedError`). The vec lane dies and its `free()` never fires.
3. After fixing the vec/cube error, the sync timeout resolves on its own.
4. Do **not** tune sync timeouts or phase counters to work around these failures — the counters are correct; the lane just never ran to completion.

### V2 simulator: do not run multiple simulator processes in parallel

Running multiple V2 simulator processes concurrently can produce **silent data corruption**. Root cause: per-lane `PipeWorker` threads are exposed to intra-process races under heavy CPU thread contention. Primarily affects kernels using NZ layout ops (`ub_to_l1_nz`, `deinterleave`, `reg_to_ub`) or complex `@vf` functions.

- `simulator="legacy"` is still accepted but routes to the same V2 runtime — there is no sequential fallback to switch to.
- Always run kernel simulator tests sequentially, not in parallel with `&` or batch scripts.
- If a kernel produces incorrect results only when run alongside other tests, re-run it alone before investigating.

### V2 simulator launch rule: use a real script entry and keep `PYTHONPATH`

When launching helper comparisons or ad-hoc debugging runs, do not start the simulator from
`stdin` entry points such as:
- `python - <<'PY'`
- `cat script.py | python`

V2 uses child processes plus worker threads. On the process-spawn path, Python must be able to
re-import the parent `__main__` module from a real file. `stdin` entry points show up as
`<stdin>`, so child startup fails with errors such as:
- `FileNotFoundError: ... '/path/to/repo/<stdin>'`
- follow-on `EOFError` while `multiprocessing.Manager()` starts

Practical rule:
- put the repro in a real `.py` file and run that file
- include the repository root in `PYTHONPATH` whenever the script imports local modules from
  outside the repo root or from a temp directory

Typical safe form:
- `PYTHONPATH=/abs/path/to/repo python /tmp/repro.py`

## Layer-by-layer review

Use this order when the fast-path sections above did not match or did not fix the bug:
1. contract and cast order
2. layout and shape bindings
3. tile and capacity assumptions
4. tail handling
5. sync and ownership
6. counters and lifetime separation
7. precision boundaries
8. parser/simulator/codegen implementation path

### 1. Re-check the exact contract

Verify the kernel against the real PyTorch formula. Common failure modes: wrong cast order, wrong transpose interpretation, wrong reshape meaning, accidental semantic drift. If the reference is still fuzzy, stop here and clarify it before changing the DSL code.

### 2. Re-check layout and shape binding assumptions

Verify tensor logical shapes, transpose site, `shape_bindings`, repeated scalar dimension mapping.

Common signs: output shape is right but values are wrong everywhere; only some shapes fail; changing `M`, `N`, or `K` flips behavior unpredictably.

Repository reminder: if repeated scalar dimensions are ambiguous, try explicit `shape_bindings` before deeper kernel surgery.

### 3. Re-check tile and capacity assumptions

When the kernel is tiled, verify `TILE_M`, `TILE_N`, `TILE_K`, `m_split`, `n_split`, `splitk` / `splitn`, `L0A` / `L0B` / `L0C` byte budgets.

Repository reminders: keep `splitk` and `splitn` at `>= 32`; choose `splitk` when K-side staging is too large; choose `splitn` when N-side staging or output tile is too large; do not author non-zero `L0C` row offsets on matmul destinations. For the exact per-device caps and DBuff formulas, see `agent/references/facts-authoring.md` and `agent/references/facts-device-runtime.md`.

If tile search is non-trivial, use `agent/scripts/estimate_matmul_datamove.py` instead of eyeballing it. Drill into `agent/references/constraints/tiling.md` for reasoning.

### 4. Re-check tail handling

Look at GM boundaries first, not local tensor sizes. Rule: local buffers stay full-tile sized; only GM read/write boundaries use `valid_m`, `valid_n`, `valid_k`.

For cube -> vec writeback, verify the standard half-row split:
- `half_rows = CeilDiv(valid_m, 2)`
- `row_begin = GetSubBlockIdx() * half_rows`
- `row_end = Min(row_begin + half_rows, valid_m)`

For a2 workspace-mediated cube -> vec tails: keep workspace writes and reads on stable tile shapes (`ws[..., 0:TILE_M, 0:TILE_N]` on cube; `ws[..., row_begin:row_begin + row_count, 0:TILE_N]` on vec). Apply `valid_n` with vec-side masking and final GM write boundaries, not by cropping the workspace column span first.

Symptoms of tail bugs: aligned cases pass but odd sizes fail; only the last tile is wrong; one vec subblock is correct and the other is garbage.

Drill: `agent/references/constraints/tail-safety.md`. For normalized online softmax with running `row_max` / `row_sum`, also `agent/references/constraints/online-softmax-tail.md`.

### 5. Re-check sync ownership

Assume ownership is wrong until proven otherwise.

`auto_sync()` only manages same-side ordering and does not replace cross-side ownership transfer. Cube -> vec handoff needs `CvMutex`; vec -> cube handoff needs `VcMutex`. Exact mutex signatures per device live in `agent/references/facts-device-runtime.md`.

If the issue smells like pipeline ordering: inspect where the producer finishes, where the consumer starts, whether `lock/ready/wait/free` surround the real ownership edge, and keep the critical section narrow.

Drill: `agent/references/constraints/autosync.md`.

### 6. Re-check counters and lifetimes

Many broken kernels are actually lifetime bugs. Verify which loop owns each buffer family, whether different lifetimes accidentally share one counter, whether the same slot lineage is expressed consistently.

Rules: buffers with different lifetimes must use different counters; same-lifetime paired buffers may share one; reusing one counter across different loop-owned lifetimes can silently break autosync grouping and slot reasoning.

Drill: `agent/references/constraints/counters.md`.

### 7. Re-check precision boundaries

Verify where values change dtype. Common failures: casting too early, reducing in the wrong dtype, writing packed or quantized data too early, comparing against a reference with a different cast order.

Rule: keep matmul accumulation in `float`; downcast later unless the design proves otherwise.

Drill: `agent/references/constraints/precision.md`.

### 8. Inspect the real implementation path

If a rule is still unclear, inspect the actual implementation path instead of theorizing. Device family mapping (`950` → C310, `b*` → C220) and common target files (`easyasc/stub_functions/`, `easyasc/parser/`, `easyasc/parser/asc_autosync.py`, `easyasc/kernelbase/kernelbase.py`, `easyasc/simulator_v2/`, `easyasc/shortcuts/matmul.py`) are in `agent/references/code-paths.md`.

Good debugging question: which exact instruction gets emitted, how the parser lowers it, how the simulator executes it, whether the kernel assumption matches that path.

When the simulator itself produces an unexpected error: investigate the simulator path first; inspect the exact simulator stage, runtime view, and lowered instruction that failed; do not assume the upper-layer kernel is wrong just because the simulator failed first.

If simulator behavior still looks inconsistent with the intended model after real inspection: stop blind upper-layer edits, summarize the concrete simulator finding, pause and discuss with the user.

## Build a minimal reproducer

When the full kernel is noisy, isolate one mechanism: one matmul, one handoff, one vec postprocess, one autosync chain, one tail tile. A minimal reproducer is usually faster than staring at a fused kernel.

Shrink-down order: keep the original failing shape, remove later stages until only the first wrong stage remains, inside that stage keep only one subformula (`odo`, `rowmax`, one GM bridge), shrink again if needed to one instruction and one view shape.

## Treat warnings as real signals

Do not accept a passing result with unresolved warnings. Especially for `auto_sync`, warnings usually mean the lifetime model is off. If a warning persists after real inspection, stop blind iteration — either redesign the stage boundary or ask the user for clarification.

## Fallback references

- `agent/references/code-paths.md`
- `agent/references/simulator-v2.md`
- `doc/11_architecture_for_contributors.md`

# Kernel Debugging Playbook

Use this playbook when an existing kernel is wrong, unstable, warning-heavy, or unclear. Debug in layers. Do not jump between random fixes.

## Goal

Find the first broken assumption. Fix the model, then fix the kernel. Do not keep stacking patches on top of an unclear design.

## Fast-path: match your symptom first

Most bug reports match one of the patterns below. Try these before running the full layer-by-layer review further down.

### Symptom-to-check map

- **Wrong everywhere** → check formula, transpose/layout, cast order, `shape_bindings`
- **Looks like precision, but output is exactly zero or one stale half-tile** → check missing `auto_sync()`, missing manual `FIX -> consumer` event, or unstable partial-`L0C` tail staging before changing casts
- **Only large shapes fail** → check tile budgets, split mode, estimator choice, counter ownership across nested loops
- **Only tail tiles fail** → check `valid_*` handling, half-row vec writeback split, GM boundary slicing
- **`gm_to_l1_nd2nz` / `ub_to_l1_nd2nz` / `l0c_to_*` says the buffer is too small** → check physical NZ / `L0C` backing size and `agent/references/simulator-datamove-footprint-guards.md` before changing the math; narrow logical `N` still needs padded local storage
- **Autosync warnings or weird pipeline stalls** → check same-side vs cross-side misunderstanding, event family grouping, counter reuse across different lifetimes, unsupported instruction not covered by autosync pairing
- **Local event timeout / already-set (`_tmp_*valid_*`, `_tmp_*ready_*`)** → classify the event failure first, dump autosync-expanded instructions, then compare the failing family against a stable kernel before changing the DSL
- **Simulator passes, generated path looks suspicious** → check parser lowering, codegen handlers, explicit event or mutex placement, assumptions hidden by simulator convenience
- **Suspected timing error on the generated path** → dump the generated `.h` / `.cpp` first, verify whether the emitted C++ order matches the intended ownership and event flow, then branch the investigation from that evidence instead of patching code immediately
- **Hardware-only failure in a large fused kernel** → isolate one public output or one formula slice, prove it locally and on hardware, then add outputs back one at a time before re-merging into the fused kernel
- **Simulator `wait_vec` / `wait_cube` timeout** → use the cross-side mutex timeout recipe; almost always the other lane's actor thread crashed silently
- **Kernel only fails when run alongside other tests** → use the simulator process isolation recipe
- **The compare path still shows the old output tensor** → check whether the test is reading the `OpExec(...)(...)` return value or a placeholder output argument that was only passed in

### Focused debug recipes

Open `agent/references/debug-recipes.md` only for the matching symptom:

| Symptom | Recipe |
|---------|--------|
| first simulator timeout without a clear event mismatch | [first-run timeout retry](../references/debug-recipes.md#first-run-simulator-timeout-retry) |
| lane-local `event_wait` timeout or already-set event | [event pairing workflow](../references/debug-recipes.md#event-pairing-workflow-for-local-event-failures) |
| generated-path timing or hardware-only ordering suspicion | [generated-path timing investigation](../references/debug-recipes.md#generated-path-timing-investigation) |
| `wait_vec` / `wait_cube` timeout | [cross-side mutex timeout diagnosis](../references/debug-recipes.md#cross-side-mutex-timeout-diagnosis) |
| failure only when tests run together | [simulator process isolation](../references/debug-recipes.md#simulator-process-isolation) |
| ad-hoc repro launched from stdin | [simulator launch rule](../references/debug-recipes.md#simulator-launch-rule) |
| need scalar or tensor runtime inspection | [simulator debug helpers](../references/debug-recipes.md#simulator-debug-helpers) |

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

Repository reminders: 32 is a normal-matmul search heuristic, not a generic hard minimum for `splitk` or `splitn` (MX `matmul_mx` separately requires `splitk` a multiple of 64 and `splitn` a multiple of 16); choose `splitk` when K-side staging is too large; choose `splitn` when N-side staging or output tile is too large; do not author non-zero `L0C` row offsets on matmul destinations. For the exact per-device caps and DBuff formulas, see `agent/references/facts-authoring.md` and `agent/references/facts-device-runtime.md`.

If tile search is non-trivial, use `agent/scripts/estimate_matmul_datamove.py` instead of eyeballing it. Drill into `agent/references/constraints/tiling.md` for reasoning.

### 4. Re-check tail handling

Look at GM boundaries first, not local tensor sizes. Rule: local buffers stay full-tile sized; use `valid_m`, `valid_n`, `valid_k` at GM read/write boundaries and at explicit UB-domain masks or reductions that must exclude padded lanes (for example online-softmax score-column masking).

For cube -> vec writeback, verify the split style matches the pipeline. The a5-style compact split is:
- `half_rows = CeilDiv(valid_m, 2)`
- `row_begin = GetSubBlockIdx() * half_rows`
- `row_end = Min(row_begin + half_rows, valid_m)`

For a2 workspace-mediated cube -> vec tails, use the fixed physical sub-block
split from `constraints/tail.md`: subblock 0 owns rows `[0:HALF_M)` and
subblock 1 owns `[HALF_M:TILE_M)`, with
`local_valid_m = Min(HALF_M, Max(valid_m - sb_row, 0))`. Keep workspace writes
and reads on stable tile shapes (`ws[..., 0:TILE_M, 0:TILE_N]` on cube;
`ws[..., sb_row:sb_row + HALF_M, 0:TILE_N]` on vec). Apply `valid_n` with
vec-side masking and final GM write boundaries, not by cropping the workspace
column span first.

Symptoms of tail bugs: aligned cases pass but odd sizes fail; only the last tile is wrong; one vec subblock is correct and the other is garbage.

Drill: `agent/references/constraints/tail.md`. For normalized online softmax
with running `row_max` / `row_sum`, also read
`agent/references/patterns/online-softmax-tail.md`.

### 5. Re-check sync ownership

Assume ownership is wrong until proven otherwise.

`auto_sync()` only manages same-side ordering and does not replace cross-side ownership transfer. Cube -> vec handoff needs `CvMutex`; vec -> cube handoff needs `VcMutex`. Exact mutex signatures per device live in `agent/references/facts-device-runtime.md`.

If the issue smells like pipeline ordering: inspect where the producer finishes, where the consumer starts, whether `lock/ready/wait/free` surround the real ownership edge, and keep the critical section narrow.

Drill: `agent/references/constraints/sync.md`.

### 6. Re-check counters and lifetimes

Many broken kernels are actually lifetime bugs. Verify which loop owns each buffer family, whether different lifetimes accidentally share one counter, whether the same slot lineage is expressed consistently.

Rules: buffers with different lifetimes must use different counters; same-lifetime paired buffers may share one; reusing one counter across different loop-owned lifetimes can silently break autosync grouping and slot reasoning.

Drill: `agent/references/constraints/sync.md`.

### 7. Re-check precision boundaries

Verify where values change dtype. Common failures: casting too early, reducing in the wrong dtype, writing packed or quantized data too early, comparing against a reference with a different cast order.

Rule: keep matmul accumulation in `float` (int8 and int4 paths are the int32 exceptions); downcast later unless the design proves otherwise.

Drill: `agent/references/constraints/precision.md`.

### 8. Inspect the real implementation path

If a rule is still unclear, inspect the actual implementation path instead of theorizing. Device family mapping (`950` → C310, `b*` → C220) and common target files (`easyasc/stub_functions/`, `easyasc/parser/`, `easyasc/parser/asc_autosync.py`, `easyasc/kernelbase/kernelbase.py`, `easyasc/simulator/`, `easyasc/shortcuts/matmul.py`) are in `agent/references/code-paths.md`.

Good debugging question: which exact instruction gets emitted, how the parser lowers it, how the simulator executes it, whether the kernel assumption matches that path.

When the simulator itself produces an unexpected error: investigate the simulator path first; inspect the exact simulator stage, runtime view, and lowered instruction that failed; do not assume the upper-layer kernel is wrong just because the simulator failed first.

If simulator behavior still looks inconsistent with the intended model after real inspection: stop blind upper-layer edits, summarize the concrete simulator finding, pause and discuss with the user.

## Build a minimal reproducer

When the full kernel is noisy, isolate one mechanism: one matmul, one handoff, one vec postprocess, one autosync chain, one tail tile. A minimal reproducer is usually faster than staring at a fused kernel.

Shrink-down order: keep the original failing shape, remove later stages until only the first wrong stage remains, inside that stage keep only one subformula (`odo`, `rowmax`, one GM bridge), shrink again if needed to one instruction and one view shape.

For `@vf()` preprocess stages, separate tile initialization from the data-scatter loop while shrinking the repro. A zero-fill or padding write placed inside the inner copy loop can erase values that were correct earlier in the same patch.

For hardware-only failures in a fused backward kernel, prefer an output-by-output
rebuild over a pile of fused-kernel guesses:

1. keep the original public input ABI when possible;
2. return only the first failing output or the smallest formula slice;
3. validate simulator smoke, same-core reuse stress, and hardware;
4. add exactly one output or downstream consumer;
5. repeat until the full tuple is proven, then move the proven schedule back
   into the official kernel path.

This route paid for plan1 `scan_state_bwd`, `inverse_preprocess_bwd`, and
`finalize_bwd`: the formulas were right, but the fused schedules hid scratch
source-lifetime and hardware-only ordering bugs.

## Validate lifetime fixes with same-core reuse

Any sync, DBuff/TBuff/QBuff, source-lifetime, or output-staging fix needs at
least one regression shape where the same active core processes more than one
reused outer tile. A smoke shape where each active core owns only one work item
can prove the formula while missing the race.

Pick the reused loop dimension and force per-core work above one. Examples:

- for BHC-tiled GDN kernels, use shapes such as `C > core_count` or a forced
  one-core run so the same core reuses UB/L1/L0C slots across chunks;
- for M-tiled attention, check
  `ceil(BH * ceil(S1 / TILE_M) / core_count) > 1`.

If the same-core stress fails but the one-BHC smoke passes, inspect pending
MTE3/FIX/UB sources and mutex release points before changing the math.

## Treat warnings as real signals

Do not accept a passing result with unresolved warnings. Especially for
`auto_sync`, warnings usually mean the lifetime model is off. VF local-memory
warnings are the same class of signal: hardware may overlap VF store/load/compute
streams even when the Python micro body looks sequential. If a warning persists
after real inspection, stop blind iteration — either redesign the stage
boundary, separate the buffer roles, repair the counter family, or ask the user
for clarification.

## Fallback references

- `agent/references/code-paths.md`
- `doc/11_architecture_for_contributors.md`

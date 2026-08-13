# Debug Recipes

Use this file only after `agent/playbooks/kernel-debugging.md` routes a
specific symptom here. Keep the playbook open for the layer-by-layer review;
this file owns longer operational recipes.

## First-run simulator timeout retry

Use this when a kernel times out on its first simulator run and the output does
not contain a clear flag/event mismatch signal such as:
- `event_set on already-set flag`
- repeated `set_count` / `wait_count` evidence for the same local event family
- explicit `wait_vec` / `wait_cube` phase mismatch pointing at a known missing token

Before deep-diving synchronization, rerun the same kernel once with
`execution_timeout_s=90.0`:

```python
from easyasc.simulator import SimulatorConfig

my_kernel._simulator_config = SimulatorConfig(execution_timeout_s=90.0)
```

Do not keep increasing the timeout. The 90s retry is a probe, not a fix.

For an intentionally large trace or benchmarking run that is already expected
to exceed the default 40s simulator timeout, set a single generous timeout
up front instead of treating the first timeout as a sync failure. For example,
`flash_attn_full_pj_half_block32_causal_v4` at `(B,H,S1,S2,D)=(1,32,2048,2048,128)`
completed in about 95s with `execution_timeout_s=300.0`; the same run with the
default timeout produced misleading post-timeout `KeyError` / missing metadata
noise.

## Event pairing workflow for local event failures

Use this when the simulator reports a lane-local event problem such as:
- `event_wait timeout: {'name': '_tmp_sevent_valid_fix_0', ...}`
- `event_set on already-set flag: _tmp_sevent_valid_l1_0 ...`

Debugging sequence:
1. Classify the runtime message. `event_wait timeout` usually means a missing
   `event_set`; `event_set on already-set flag` usually means duplicate
   publication before a matching wait consumed the token.
2. Read counters literally. On the simulator path, `preset=True` events start
   with one published token; if `set_count == wait_count`, the preset token was
   consumed and the next producer-side `event_set` never happened.
3. Build kernel instructions before inspecting split/autosync output. Call the
   `@kernel` once with placeholder `GMTensor(...)` arguments so
   `kernel.instructions` is populated.
4. Dump autosync-expanded lane instructions with `split_instructions(...)` plus
   `insert_auto_sync(...)`; inspect only the failing side.
5. Filter to one family at a time: `l1`, `l0`, `fix`, `ubin`, `ubout`, or
   `ubrelay`.
6. Turn the stream into an action sequence containing only `event_wait` /
   `event_set` and the paired producer/consumer ops. Healthy reuse alternates
   publish and consume rounds.
7. Compare against a nearby stable baseline kernel using the same pipe pair.
8. If the failing edge sits around nested control-flow regions, inspect
   parent/child mixed-scope handling before touching the kernel.
9. Add a parser regression before rerunning the real kernel. Use
   `agent/example/testcases/parser/sync/test_autosync_vec_event_metadata.py` for vec event
   metadata failures or `test_autosync_cube_l0_l1_events.py` for cube l0/l1
   reuse failures.

When this workflow points to parser behavior, pair it with
`agent/references/constraints/sync.md`.

## Generated-path timing investigation

Use this when a failure looks like generated-path timing, event ordering, or
hardware-only behavior.

1. Dump generated headers and entry C++ first. Prefer emitted
   `<path>_cube.h`, `<path>_vec.h`, and `<path>.cpp` from `dump_kernel(...)`
   or `dump_asc(...)`.
2. Read the generated C++ literally. Check expected producer/consumer order,
   mutex placement, and synthesized event direction before editing DSL code.
3. If generated C++ matches the intended model, inspect simulator/runtime
   behavior next.
4. If generated C++ is wrong, inspect split output, autosync insertion, and the
   relevant `easyasc/parser/asc_autosync.py` or handler path.
5. Report whether the mismatch is generated C++, autosync/lowering, or
   simulator/runtime behavior before changing code.

## Cross-side mutex timeout diagnosis

When the simulator reports `wait_vec` / `wait_cube` timeout, the other lane's
actor thread often crashed before publishing `vec_ready` or `cube_ready`.

Capture the real error on the other lane first: wrap each lane actor start in a
try/except that prints the lane name and exception. The first non-timeout error
is the real root cause.

Common root causes behind silent lane crashes:
- PyTorch does not support fancy indexing on `float8_e5m2` /
  `float8_e4m3fn`; view as `torch.uint8` before indexing inside `@vf`.
- Burst-copy ops may expose non-contiguous UB view issues; in focused repros,
  ensure source tensors are contiguous before the burst call.
- A `@vf()` body called a micro op that the micro runtime does not dispatch.

After fixing the vec/cube error, the sync timeout should resolve on its own. Do
not tune sync timeouts or phase counters to work around these failures.

## Simulator process isolation

Run kernel simulator tests sequentially, not in parallel with `&` or batch
scripts. Concurrent simulator processes can produce silent corruption,
especially with NZ layout ops (`ub_to_l1_nz`, `deinterleave`, `reg_to_ub`) or
complex `@vf` functions.

If a kernel produces incorrect results only when run alongside other tests,
rerun it alone before investigating.

## Simulator launch rule

Do not start simulator repros from stdin entries such as:
- `python - <<'PY'`
- `cat script.py | python`

The simulator uses child processes plus worker threads. On process-spawn paths,
Python must re-import the parent `__main__` module from a real file. Put the
repro in a `.py` file and include the repository root in `PYTHONPATH` whenever
the script imports local modules from outside the repo root or from a temp
directory.

Typical safe form from the repository root:
- `PYTHONPATH=. python tmp/<task>/repro.py`

## Simulator debug helpers

`sim_print` and `sim_dump_tensor` are simulator-only helpers. They are no-ops
during codegen and exist for local simulator debugging.

Use `sim_print(*args, pipe=Pipe.S)` for counters, flags, and control-flow
breadcrumbs:

```python
from easyasc.a5 import sim_print, Pipe

sim_print("tile", tile_idx, "valid_m", valid_m, pipe=Pipe.S)
```

Use `sim_dump_tensor(tensor, filename, pipe)` for intermediate UB/L1 data that
needs offline inspection:

```python
from easyasc.a5 import sim_dump_tensor, Pipe

sim_dump_tensor(ub_out, "ub_out_tile0.pt", Pipe.V)
```

Notes:
- `pipe=Pipe.S` attaches a print to the main loop; non-`S` pipes attach to that
  pipe queue for pipe-local timing.
- Do not use `sim_print` for large tensor payloads.
- Use unique dump filenames; repeated dumps overwrite.
- `Pipe.ALL` is rejected for tensor dumps because a dump is one execution event,
  not a barrier.

Full API reference: `doc/api/helpers.md`.

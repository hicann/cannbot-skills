# Simulator and OpExec Facts

Use this file for simulator behavior one-liners and `OpExec` call-site gotchas.
Use `agent/references/code-paths.md` and `agent/references/simulator-v2.md` when you need the full implementation path.

## Simulator-behavior one-liners

- `OpExec(..., simulator=True | "v2" | "legacy")` all route to V2
- `simulator="legacy"` does not select a separate old runtime
- `bar_all()` is the only cross-pipe drain in V2; `bar_v()` / `bar_m()` are no-ops
- for kernel cycle optimization, the target is the trace makespan (the last instruction/task to finish, i.e. `max(ts + dur)` over timed trace events), not the sum of all activated-task durations
- `wait_vec` / `wait_cube` timeout almost always means the other lane's actor thread crashed silently
- do not run multiple V2 simulator processes concurrently; thread contention can cause silent data corruption
- PyTorch does not support indexing on `float8_e5m2` / `float8_e4m3fn`; view as `torch.uint8` before indexing inside `@vf`
- burst-copy ops (`gm_to_ub_pad`, `ub_to_gm_pad`, `ub_to_l1_nz`) are safe on column-sliced UB views because they use `_linear_view_from_pointer`

## `OpExec` call-site checklist

- provide `shape_bindings={...}` when two or more kernel-side scalar dimensions can take the same integer value at runtime
- `shape_bindings` belongs on the returned callable, not on the `OpExec(...)` constructor
- format: `{tensor_arg_index: [scalar_idx_for_axis_0, scalar_idx_for_axis_1, ...], ...}`
- the key is indexed among tensor args only; scalar args are skipped
- use `None` to keep an axis unbound

Example, for `kernel(x:[M,K], y:[N,K], z:[M,N], M, N, K)`:

```text
shape_bindings={0: [0, 2], 1: [1, 2], 2: [0, 1]}
```

Implementation:
- `easyasc/torchplugin.py:614-668`

Real references:
- `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py:137`
- `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py:116`

## Deeper references

- `agent/references/simulator-v2.md`
- `agent/references/code-paths.md`
- `agent/playbooks/kernel-debugging.md`

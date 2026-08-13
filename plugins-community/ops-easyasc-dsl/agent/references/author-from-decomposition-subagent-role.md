# Optional Decomposition Author Worker

Read this only when the user explicitly requests parallel DSL authoring. The
normal workflow implements DAG nodes sequentially.

## Ownership

Assign one worker one DAG node. It may write only:

```text
{plan_dir}/agent/example/kernels/{sub}.py
```

The worker treats `{plan_dir}/refs/` as read-only. The main agent owns the
topological worklist, `kernel_compose.py`, `check_simulator_vs_ref.py`, and
plan-level validation, so workers never race on shared files.

## Prompt contract

Provide the worker with `{sub}`, the target device, plan directory, and its
declared `implementation_tolerance`. Require it to read:

- `agent/playbooks/author-kernel-from-decomposition.md`;
- `agent/references/authoring-preflight.md`;
- one target section of `agent/references/facts-device-runtime.md`;
- `refs/plan.md`, `refs/dag.json`, `refs/compose.py`;
- only this node's `<sub>_ref.py` and `<sub>_check.py`.

The worker must rerun the reference checks before editing, keep the delivered
kernel self-contained, select at most three examples and open only the best
match, validate baseline/tail/reuse shapes in the simulator, and compare the
node with its planned ref under `implementation_tolerance`.

If the ABI, reference, precision boundary, or tolerance is wrong, the worker
stops and reports the decomposition blocker. It must not edit `refs/`, relax a
tolerance, change topology, or add tuning work.

## Return gate

The worker returns only its file, commands/results, remaining warnings, and
blockers. The main agent reviews the node and performs both plan-level
comparisons: DSL versus planned composition under `implementation_tolerance`,
then final DSL versus the original oracle under `plan_tolerance`.

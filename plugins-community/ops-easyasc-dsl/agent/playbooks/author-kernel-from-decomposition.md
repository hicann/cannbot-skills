# Author DSL Kernels from a Verified Decomposition

Use this workflow only for an existing multi-kernel plan. `refs/` is the
verified and immutable contract. If a reference, ABI, DAG edge, precision
boundary, or tolerance is wrong, stop and return to decomposition; never edit a
reference to hide a DSL mismatch.

## Deliverable

```text
<plan-dir>/
├── refs/                         # unchanged
├── agent/example/kernels/
│   ├── <sub>.py
│   └── kernel_compose.py         # when runtime composition is needed
└── check_simulator_vs_ref.py
```

Use the single kernel template and check template under
`agent/references/templates/author-kernel-from-decomposition/`.

## 1. Intake gate

Read only:

- `refs/plan.md`, `refs/dag.json`, `refs/compose.py`, and `refs/oracle.py`;
- the current sub-kernel's `<sub>_ref.py` and `<sub>_check.py`;
- [`authoring-preflight.md`](../references/authoring-preflight.md);
- one target section of
  [`facts-device-runtime.md`](../references/facts-device-runtime.md).

Before authoring, rerun every reference check and `refs/compose.py`. If any
check fails, stop before authoring. Also stop when the DAG has a dangling edge,
an ABI is incomplete, or either
`plan_tolerance` or `implementation_tolerance` is missing from `plan.md`.

## 2. Build the worklist

Topologically sort `dag.json`. For each node, record:

- kernel signature and public/intermediate `GMTensor` ownership;
- shape symbols and explicit `shape_bindings`;
- required dtype/layout/cast boundary;
- its default `implementation_tolerance` or documented override;
- producer completion and consumer readiness dependencies.

Precision or topology changes belong in decomposition. Tuning experiments belong
in the optimization workflow.

## 3. Implement one sub-kernel at a time

Select examples narrowly:

```bash
python agent/scripts/select_kernel_example.py \
  --device <a2|a5> --query "<formula and topology>" --limit 3 --path-only
```

Open the catalog entry for the best candidate, then open only that source. Do
not read or copy all three.

Implement in stages:

1. establish the `@kernel` signature, placeholders, output ownership, and shape
   bindings;
2. settle tile sizes and local-buffer layouts against device capacity;
3. add transfers and compute one stage at a time;
4. add synchronization from actual producer/consumer lifetimes;
5. run the simulator after each stage;
6. compare the completed node with `<sub>_ref.py` using its documented
   `implementation_tolerance`.

Run from the repository root with `PYTHONPATH=.`. Use
`OpExec(..., simulator=True)` by default. Do not continue to later DAG nodes
until the current node passes its reference on baseline, tail, and reuse shapes.

## 4. Validate composition

After all nodes pass independently:

1. execute kernels in DAG order with the same ABI as `refs/compose.py`;
2. compare each DSL node with its planned ref using
   `implementation_tolerance`;
3. compare the DSL composition with the planned composition using the declared
   implementation budget;
4. compare final DSL outputs directly with `refs/oracle.py` using
   `plan_tolerance`.

Both comparisons are required. A small node-local error does not waive the
end-to-end plan budget.

## 5. Final preflight and delivery

Run:

```bash
python -m py_compile <plan-dir>/agent/example/kernels/*.py \
  <plan-dir>/check_simulator_vs_ref.py
PYTHONPATH=. python <plan-dir>/check_simulator_vs_ref.py
```

The final check must include:

- minimum, typical, alignment-boundary, and tail shapes;
- a shape that makes the same active core reuse each DBuff/TBuff slot across
  more than one outer tile;
- all documented cast/lossy boundaries;
- node-level implementation comparisons;
- planned-compose and original-oracle comparisons;
- no unresolved simulator or `auto_sync` warning.

Deliver self-contained kernels that do not import `refs/`. Report simulator
coverage and explicitly state that real-device code generation, fixpipe/vector
bit-exactness, and performance remain unverified when no board run was requested.

Legacy manifests are optional adapters, not default inputs; use
[`legacy-decomposition.md`](../references/adapters/legacy-decomposition.md) only
for an existing legacy plan. Read the optional authoring subagent role only when
the user explicitly requests parallel implementation:
[`author-from-decomposition-subagent-role.md`](../references/author-from-decomposition-subagent-role.md).

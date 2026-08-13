# Decompose PyTorch into a Verified Runtime Plan

Use this workflow only when the user explicitly asks to split a PyTorch
function into multiple runtime kernels. For one runtime kernel, return to the
single-kernel workflow.

The default deliverable is one plan. Produce alternatives only when the user
explicitly asks for them.

## Deliverable

Create one plan directory with this complete, self-contained handoff:

```text
refs/
├── oracle.py
├── compose.py
├── dag.json
├── plan.md
├── <sub>_ref.py
└── <sub>_check.py
```

`refs/` becomes immutable after the handoff passes. The downstream authoring
workflow must return here to correct a bad reference instead of editing it.

## 1. Confirm the decomposition contract

Before writing files, confirm that multiple runtime kernels are required. Freeze
the exact original PyTorch function as `refs/oracle.py`, including public input
order, output order, shapes, dtypes, layout, constants, and edge-case behavior.

Resolve semantic ambiguity from repository evidence first. If it remains,
request the smallest missing decision with
[`clarification-template.md`](../references/clarification-template.md). Do not
publish a handoff with an open semantic question.

## 2. Design the plan once

Settle these together before implementing references:

- sub-kernel boundaries and a directed acyclic graph;
- every edge's tensor name, shape, dtype, layout, producer, and consumers;
- all casts, lossy boundaries, and matmul accumulation/rounding boundaries;
- runtime launch topology and any workspace or fan-out ownership;
- one target device, using only the relevant section of
  [`facts-device-runtime.md`](../references/facts-device-runtime.md);
- reusable primitives only as needed from
  [`decomposition-primitives.md`](../references/decomposition-primitives.md).

`splitk` and `splitn` inside one matmul kernel are tiling choices, not host-level
decomposition nodes and not automatic cross-core merge plans.

## 3. Materialize the handoff

Use the templates under `agent/references/templates/decompose-pytorch/`.

`plan.md` must define two distinct budgets:

- **`plan_tolerance`** is the user-accepted end-to-end error budget relative to
  `oracle.py`. Both the planned PyTorch composition and the final DSL
  composition must satisfy it.
- **`implementation_tolerance`** is the DSL implementation budget relative to
  the planned references. It may include justified per-node overrides, but each
  override must be written in `plan.md`; scripts must not hide it.

For every sub-kernel, record its inputs, outputs, shape, dtype, layout, cast or
lossy boundary, and DAG dependencies. `dag.json` is the machine-readable graph;
`plan.md` is the human owner of the ABI and tolerances.

Each `<sub>_ref.py` must be pure PyTorch and independently importable. Each
`<sub>_check.py` must construct deterministic inputs and validate that boundary.
`compose.py` must call the planned refs in DAG order and compare its result with
the frozen original oracle using `plan_tolerance`.

## 4. Verify

From the repository root, run:

```bash
python -m py_compile <plan-dir>/refs/*.py
for check in <plan-dir>/refs/*_check.py; do python "$check"; done
python <plan-dir>/refs/compose.py
```

Also validate `dag.json` programmatically:

- node names are unique;
- every non-public input has exactly one producer;
- every edge names existing endpoints;
- the graph is acyclic and has no dangling output;
- each public output is produced and returned by `compose.py`.

## 5. Handoff gate

Mark the plan ready only when all of the following are true:

- all reference files import and all checks pass;
- planned composition satisfies `plan_tolerance` against the original oracle;
- every ABI and lossy boundary is explicit;
- the DAG is complete and topologically valid;
- no semantic question remains open.

Do not add candidates, tuning reports, token budgets, author manifests, or
subagent tutorials to the default plan. For an old plan that already contains a
manifest, consult
[`legacy-decomposition.md`](../references/adapters/legacy-decomposition.md)
only when needed. Read the optional decomposition subagent role only when the
user explicitly requests parallel planning:
[`decomposition-subagent-role.md`](../references/decomposition-subagent-role.md).

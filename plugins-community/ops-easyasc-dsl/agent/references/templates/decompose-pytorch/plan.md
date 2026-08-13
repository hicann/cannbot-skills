# Decomposition Plan: {function_name}

- target: `{hw_target}`
- plan directory: `{plan_dir}`
- handoff status: `{draft_or_verified}`

## Frozen behavior

{Describe the original PyTorch behavior, public edge cases, and why more than one
runtime kernel is required.}

## Tolerances

`plan_tolerance` applies to both planned PyTorch composition and final DSL
composition relative to `refs/oracle.py`:

```text
atol = {plan_atol}
rtol = {plan_rtol}
```

`implementation_tolerance` applies to DSL results relative to planned refs:

```text
default.atol = {implementation_atol}
default.rtol = {implementation_rtol}
```

Per-node overrides, each with a reason:

| node | atol | rtol | reason |
| --- | ---: | ---: | --- |
| `{node_or_none}` | `{atol}` | `{rtol}` | `{reason}` |

## Public ABI

| tensor | direction | dtype | shape | layout | notes |
| --- | --- | --- | --- | --- | --- |
| `{tensor}` | `{input_or_output}` | `{dtype}` | `{shape}` | `{layout}` | `{notes}` |

Runtime shape symbols:

| symbol | source | valid range | alignment |
| --- | --- | --- | --- |
| `{symbol}` | `{tensor}.shape[{dim}]` | `{range}` | `{alignment_or_none}` |

## Nodes and edges

Machine-readable graph: `refs/dag.json`.

| node | ref/check | inputs | outputs | shape/dtype/layout | dependencies |
| --- | --- | --- | --- | --- | --- |
| `{sub}` | `{sub}_ref.py`, `{sub}_check.py` | `{inputs}` | `{outputs}` | `{abi}` | `{deps}` |

## Cast and lossy boundaries

| location | source | planned result | accumulation/rounding rule | reason |
| --- | --- | --- | --- | --- |
| `{node_or_edge}` | `{source_dtype_layout}` | `{target_dtype_layout}` | `{rule}` | `{reason}` |

## Verification record

```bash
python -m py_compile refs/*.py
for check in refs/*_check.py; do python "$check"; done
python refs/compose.py
```

- DAG validation: `{pass_or_fail}`
- sub checks: `{pass_or_fail}`
- planned compose vs original oracle under `plan_tolerance`: `{pass_or_fail}`
- unresolved semantic questions: `none`

# Agent Router

Read this file first and choose exactly one workflow. Then read
[`common-language.md`](common-language.md) in full before opening the selected
playbook. The router and common language are the fixed initial documentation
baseline; do not preload catalogs or multiple playbooks.

## Workflow routes

Use the first matching row. A settled contract and an ONNX model are inputs to
the single-kernel workflow; they do not create separate authoring routes.

| Request | Workflow |
| --- | --- |
| Explicitly author, integrate, debug, or optimize a Device-side AICPU kernel | [`aicpu-kernel-authoring.md`](playbooks/aicpu-kernel-authoring.md) |
| Build one new runtime `@kernel` from PyTorch, ONNX, a golden, a formula, a settled contract, or a reference kernel | [`pytorch-to-single-kernel.md`](playbooks/pytorch-to-single-kernel.md) |
| Explicitly decompose a PyTorch function into multiple runtime kernels | [`decompose-pytorch.md`](playbooks/decompose-pytorch.md) |
| Implement DSL kernels from an existing verified decomposition | [`author-kernel-from-decomposition.md`](playbooks/author-kernel-from-decomposition.md) |
| Debug an existing kernel | [`kernel-debugging.md`](playbooks/kernel-debugging.md) |
| Optimize an existing correct kernel | [`kernel-optimization.md`](playbooks/kernel-optimization.md) |
| Integrate or evaluate an EasyASC-generated ACLNN op in `cann-bench`, package assembled operators as a CANN Bench submission zip, or submit one to `cannbench.com` and read its scored results | [`cann-bench-aclnn-evaluation.md`](playbooks/cann-bench-aclnn-evaluation.md) |
| Change framework, stubs, parser, simulator, tests, tools, catalogs, or documentation | [`repository-maintenance.md`](playbooks/repository-maintenance.md) |

Single-kernel and decomposition routes are mutually exclusive. Do not turn a
one-kernel request into host composition or multiple launches without an
explicit contract change.

## Focused lookups

Open only what the selected workflow requires.

| Need | Reference |
| --- | --- |
| Kernel authoring safety gate | [`authoring-preflight.md`](references/authoring-preflight.md) |
| Explicitly requested AICPU authoring or integration | [`aicpu-authoring.md`](references/aicpu-authoring.md) |
| Device capacity and runtime facts | [`facts-device-runtime.md`](references/facts-device-runtime.md) |
| Detailed authoring facts and DBuff formulas | [`facts-authoring.md`](references/facts-authoring.md) |
| Simulator, `OpExec`, or `shape_bindings` | [`facts-simulator-opexec.md`](references/facts-simulator-opexec.md) |
| Current simulator cycle model | [`cycle-model.md`](references/cycle-model.md) |
| Composite API recipe | [`composite-api-recipes.md`](references/composite-api-recipes.md) |
| Reusable multi-primitive dataflow | [`pattern-index.md`](references/pattern-index.md) |
| Known pitfall | [`pitfall-records.md`](references/pitfall-records.md) |
| Example kernel | `python agent/scripts/select_kernel_example.py --device <a2|a5> --query "..." --limit 3` |
| Repository layout or implementation path | [`repo-map.md`](references/repo-map.md) or [`code-paths.md`](references/code-paths.md) |
| Missing evidence or capability claim | [`evidence-escalation.md`](references/evidence-escalation.md) |
| Unresolved semantic ambiguity | [`clarification-template.md`](references/clarification-template.md) |
| Terminology owner or Chinese/English alias | revisit the matching section of the already-loaded [`common-language.md`](common-language.md) |

## Reading rules

- After the fixed router and common-language baseline, let the selected playbook
  name the next evidence needed.
- Prefer a selector or index before a catalog or source file.
- Treat patterns as a fast path, not a closed set.
- Stop reading when the current layer answers the task.

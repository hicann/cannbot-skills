# Repository Map

Use this file for layout and ownership. Workflow selection belongs only to
`agent/ROUTER.md`.

## Top-level owners

- `easyasc/`: DSL facades, stubs, parser/codegen, simulator, runtime models,
  dtype helpers, and shortcut implementations.
- `agent/example/kernels/a2/`, `agent/example/kernels/a5/`: tracked single-kernel examples by device.
  CANN Bench competition kernels live in the sibling
  `easyasc_cannbench_kernels` extension repository, mirroring these paths
  under category-local `cann_bench/` directories; an external `cann-bench`
  checkout is not part of either repository and is resolved only through
  `machine_specs.md`.
- `agent/example/projects/a5/`: multi-kernel projects. Current roots are `gdn_fwd`,
  `gdn_bwd`, `kda_fwd`, `kda_bwd`, `delta_rule_fwd`, and `delta_rule_bwd`.
- `agent/example/demo/`: public end-to-end demonstrations that are not cataloged kernel
  references.
- `agent/example/testcases/`: parser, codegen, simulator, and tool tests, described by
  `agent/example/testcases/README.md`.
- `agent/scripts/`: selectors, generators, diagnostics, and repository checks.
- `doc/`, `doc_cn/`: public English and Chinese documentation.
- `agent/`: router-first task guidance and focused repository knowledge.

Machine access belongs only in the git-ignored root `machine_specs.md`.
Temporary runners and exercises belong under ignored `tmp/<task>/` and are not
repository evidence.

## Agent layout

```text
agent/
├── ROUTER.md
├── playbooks/
│   ├── aicpu-kernel-authoring.md
│   ├── pytorch-to-single-kernel.md
│   ├── decompose-pytorch.md
│   ├── author-kernel-from-decomposition.md
│   ├── kernel-debugging.md
│   ├── kernel-optimization.md
│   ├── cann-bench-aclnn-evaluation.md
│   └── repository-maintenance.md
├── references/
│   ├── adapters/                  # ONNX and legacy-plan input adapters
│   ├── constraints/               # device/topic invariants
│   ├── patterns/                  # reusable multi-primitive dataflows
│   ├── optimization/              # optimization-only levers
│   ├── templates/                 # decomposition and authoring artifacts
│   ├── agent/example/demo/                  # human catalogs and generated lean index
│   ├── aicpu-authoring.md
│   ├── authoring-preflight.md
│   ├── authoring-overview.md
│   ├── clarification-template.md
│   ├── facts-device-runtime.md
│   ├── facts-authoring.md
│   ├── facts-simulator-opexec.md
│   ├── cycle-model.md
│   ├── pattern-index.md
│   ├── code-paths.md
│   ├── pitfall-records.md
│   └── repo-map.md
├── common-language.md             # terms and Chinese/English aliases only
├── diary.md                       # temporary internal kernel-change notes
└── index/                         # generated JSON indexes
```

## Ownership boundaries

- repository layout: this file and `doc/11_architecture_for_contributors.md`
- implementation lookup: `code-paths.md`
- target capacities and runtime behavior: `facts-device-runtime.md`
- cross-cutting authoring safety: `authoring-preflight.md`
- simulator use: `facts-simulator-opexec.md`
- cycle estimator meaning and owners: `cycle-model.md`
- reusable dataflow: `pattern-index.md`
- kernel/tool selection metadata: Markdown catalogs under `agent/example/demo/`
- generated discovery views: `agent/references/examples/kernel-index.md` and
  `agent/index/*.json`
- recurring failure routing: `pitfall-records.md`

Do not copy a stable fact into the glossary, router, diary, or a generated
index. Update the owner and link to it.

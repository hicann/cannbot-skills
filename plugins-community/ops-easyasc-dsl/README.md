# ops-easyasc-dsl

[Chinese README / 中文版说明](README_CN.md)

***Artificial, then Intelligence.***

`ops-easyasc-dsl` packages the easyasc DSL-to-AscendC workflow as a skill.
easyasc is a Python DSL for authoring Ascend-style kernels: a decorated
Python function becomes instruction IR that the framework can split into cube
and vec paths, execute in the built-in simulator, or lower into custom-op
source artifacts.

## Skill entrypoint

The user-facing skill entrypoint is `SKILL.md` at the plugin root. The reusable workflow
lives under `agent/`. Before reading archived runtime/docs content or running
examples, restore them on demand:

```bash
bash agent/scripts/init.sh
```

This is idempotent and only restores missing trees (`easyasc/`, `doc/`,
`doc_cn/`, the `agent/scripts/` maintenance tools, and `agent/example/`).

This is a repository-first delivery rather than an installable Python package.
Start in the simulator, keep the plugin root on `PYTHONPATH`, and use the
CANN-backed paths only after the kernel matches its reference.

## What you can do

- author cube-only and mixed cube/vec kernels through one Python surface
- validate tiling, tail handling, synchronization, and precision boundaries in
  the built-in simulator
- generate source and runtime artifacts for CANNSIM or Ascend hardware
- study runnable reference kernels for supported DSL patterns

## Public target surfaces

| Import | Target profile | Workers | Target-specific surface |
|---|---|---:|---|
| `easyasc.a2` | Ascend A2, B3 by default | 20 cube / 40 vec | A2 vector APIs and the A2 int4 contract |
| `easyasc.a3` | Ascend 910C, `Ascend910_9362` | 20 cube / 40 vec | A2/C220 APIs compiled for `ascend910_93` |
| `easyasc.a5` | Ascend 950 | 32 cube / 64 vec | `@vf`, `@simt`, register micro APIs, and MX formats |
| `easyasc.a5pr` | Ascend 950PR | 28 cube / 56 vec | A5/C310 API with the 950PR device profile |

`a2` and `a3` share the C220 authoring API but select distinct device/build
profiles; the A5 family is a parallel architecture-specific API.

> **Target isolation:** import exactly one of `easyasc.a2`, `easyasc.a3`,
> `easyasc.a5`, or `easyasc.a5pr` in a Python process. Importing a facade selects process-global
> device state, and Python's module cache does not restore an earlier target
> when that facade is imported again. Start a fresh process to switch targets.

## Quick start

### 1. Restore the archived payloads

```bash
bash agent/scripts/init.sh
```

The runtime (`easyasc/`), documentation (`doc/`, `doc_cn/`), maintenance
tools, and examples all live in the archives under `agent/assets/` and only
exist on disk after this step.

### 2. Prepare Python

If your machine already provides the `torch210npu` conda environment, use it:

```bash
conda activate torch210npu
```

For a fresh simulator-only environment, install the repository dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you already use a CANN / `torch-npu` environment, keep its `torch` and
`torch-npu` versions aligned and install only the missing dependencies. CANN
itself is not installed by `requirements.txt`.

### 3. Put the checkout on `PYTHONPATH`

Run examples from the repository root after exporting:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

The export is required because running a nested script directly adds that
script's directory—not the repository root—to Python's import path.

### 4. Run the smallest simulator example

```bash
python agent/example/kernels/a5/matmul/matmul_float_mmad.py
```

A successful run ends with output similar to:

```text
max_abs_diff=0.000000e+00
```

The example defines a kernel, runs it through
`OpExec(..., simulator=True)`, and compares its output with `x @ y.t()`.

## Minimal kernel anatomy

The runnable source is `agent/example/kernels/a5/matmul/matmul_float_mmad.py`
(present after `init.sh`):

```python
from easyasc.a5 import *


@kernel()
def matmul_float_mmad_kernel(x: GMTensor, y: GMTensor, z: GMTensor, M: Var, N: Var, K: Var):
    l1x = Tensor(DT.float, [M, K], Position.L1)
    l1y = Tensor(DT.float, [N, K], Position.L1)
    l0c = Tensor(DT.float, [M, N], Position.L0C)
    with auto_sync():
        l1x <<= x[:, :]
        l1y <<= y[:, :]
        matmul(l0c, l1x, l1y, m=M, n=N, k=K, is_init=True)
        z[:, :] <<= l0c
    return z
```

`GMTensor` values form the public GM input/output contract, local `Tensor`
values describe on-chip storage, and `<<=` emits the appropriate data movement
or writeback for each source/destination pair.

## Execution modes

| Goal | `OpExec` configuration | Additional requirements |
|---|---|---|
| Develop and debug | `simulator=True` | Python dependencies only |
| Inspect generated artifacts | `simulator=False, gen_only=True` | Generation dependencies for the selected path |
| Run in CANNSIM | `simulator=False, cannsim=True` | Compatible CANN installation |
| Build and run on hardware | `simulator=False` | Compatible CANN installation and Ascend device |

`OpExec` defaults to `simulator=False`, so pass `simulator=True` explicitly
while authoring. For generated directory layout, environment variables,
CANNSIM chipsets, build/run scripts, and logs, see
`doc/06_codegen_and_runtime.md`.

## Recommended development loop

1. Write the exact PyTorch reference formula, including cast order.
2. Choose the target facade and pipeline topology.
3. Implement the kernel and validate it with `simulator=True`.
4. Cover tail shapes and add explicit `shape_bindings` when scalar inference is
   ambiguous.
5. Inspect generated artifacts, then move to CANNSIM or hardware execution.

## Documentation and examples

The `doc/` pages and example trees below are shipped inside the archived
payloads — run `bash agent/scripts/init.sh` once to unpack them before
reading; the paths do not exist in a fresh checkout.

| Need | Start here |
|---|---|
| First successful run | `doc/01_quickstart.md` |
| Concepts and kernel syntax | `doc/02_programming_model.md` and `doc/03_write_your_first_kernel.md` |
| Full documentation map | `doc/index.md` |
| Public APIs | `doc/api/index.md` |
| Feature and dtype contracts | `doc/topics/index.md` |
| Runnable kernel selection | `agent/example/kernels/README.md` |
| Simulator and generation behavior | `doc/05_simulator_and_trace.md` and `doc/06_codegen_and_runtime.md` |
| Troubleshooting | `doc/10_troubleshooting.md` |

## Repository map

- `easyasc/`: public facades, parser/codegen, simulator, and runtime
- `agent/example/kernels/`: curated single-kernel references by target
- `agent/example/projects/`: multi-file systems that compose several kernels
- `agent/example/demo/`: end-to-end framework demos outside the kernel catalog
- `agent/example/testcases/`: parser, simulator, codegen, and tool regression tests
- `doc/`: canonical English documentation
- `doc_cn/`: Chinese documentation
- `agent/`: router-first guidance for AI/agent contributors

`easyasc/`, `doc/`, `doc_cn/`, `agent/example/`, and the `agent/scripts/*.py`
tools are delivered inside the two archives under `agent/assets/` and appear
only after `init.sh`; the `agent/` guidance documents are plain files.

Framework contributors should also read
`doc/11_architecture_for_contributors.md` and `agent/example/testcases/README.md`.

# ops-easyasc-dsl

[Chinese README / 中文版说明](README_CN.md)

***I'm just curious how far AI can go, so the codes are 90%+ developed by AI in this repo. I'm just a software architect and code reviewer.***

`ops-easyasc-dsl` packages the easyasc DSL to AscendC workflow as a skill. It still provides the Python DSL for describing mixed Ascend-style kernels with a single authoring surface that can:

- emit instruction IR from Python code
- lower that IR into split cube/vec code
- run the kernel in a built-in simulator
- generate custom-op source artifacts for non-simulator execution

The repository is organized around three ideas:

1. Author kernels in Python with `@kernel`, `Tensor`, `GMTensor`, `Var`, and optional `@vf` micro helpers.
2. Let the framework build instruction IR, insert side-specific synchronization, and split the program into cube and vec paths.
3. Validate the result either in the simulator or through generated runtime artifacts.

## Skill entrypoint

The user-facing skill entrypoint is [`skill/SKILL.md`](skill/SKILL.md). The reusable workflow lives under [`agent/`](agent/).

Before reading archived runtime/docs content or running examples, restore them on demand:

```bash
bash agent/scripts/init.sh
```

This is idempotent and only restores missing trees.

## Why this repository exists

The codebase is designed for kernel development, experimentation, and debugging. It is especially useful when you want to:

- prototype a new cube-only or mixed cube/vec pipeline
- validate tail handling, tiling, and precision boundaries in simulation
- inspect existing kernels for legal DSL patterns and implementation templates

## Installation

No need. Just import `easyasc.a5` or `easyasc.a2` via whatever method you like, after running `bash agent/scripts/init.sh`.

## Quick start

Example environment (not required):

```bash
# example only — adjust to your local setup
conda activate torch210npu
```

Then run the smallest runnable kernel example (after `bash agent/scripts/init.sh`):

```bash
python agent/example/kernels/a5/matmul_float_mmad.py
```

That example shows the minimal end-to-end loop:

1. describe a kernel with `@kernel`
2. launch it through `OpExec(..., simulator=True)`
3. compare the simulated output against a PyTorch reference

## Environment variables for `OpExec` build + CANNSIM (non-simulator)

When you run `OpExec(kernel)` with the default `simulator=False`, the framework generates `b.sh` and `r.sh` next to your current working directory and runs them. Those scripts already define a portable baseline; set the variables below **before** starting Python if your machine differs from the defaults.

| Variable | When to set | Example value |
|----------|-------------|---------------|
| `ASCEND_HOME_PATH` | Required; points at your CANN toolkit root | `<path to your CANN install>` (directory that contains `bin/setenv.bash`) |
| `ASCEND_CUSTOM_OPP_PATH` | Only if you chain extra custom OPP roots | Empty; scripts export it so `set -u` + `vendors/customize/bin/set_env.bash` stays safe |
| `EASYASC_PYTHON_BIN` | CANN `opbuild` invokes `python3` and needs **NumPy** | Directory containing that interpreter (for example the `bin` directory of a conda env), prepended to `PATH` inside `b.sh` / `r.sh` |
| `PYTHONPATH` | Python must import this repository | The repository root |

Generated `b.sh` and `r.sh` resolve paths from **`EASYASC_ROOT`**, the directory where those scripts live (run codegen from the repository root so `b.sh` / `r.sh` end up there). They source `${ASCEND_HOME_PATH}/bin/setenv.bash` when present, export vendor `LD_LIBRARY_PATH`, and run `cannsim` for the aclnn smoke binary.

If kernel packaging fails with missing shared libraries for `op_build`, ensure `setenv.bash` ran (correct `ASCEND_HOME_PATH`). If it fails importing `numpy` under CANN Python, set `EASYASC_PYTHON_BIN` to a Python that has NumPy.

## Core concepts

- `easyasc.a5`: one public DSL surface for the A5-style architecture and instruction sequences. Most current kernels and tests in this repository use it, including cube, vec, micro, register, cast, and debug helpers.
- `easyasc.a2`: another public DSL surface for the A2-style architecture and instruction sequences. It is not a compatibility layer of `a5`; it targets a different instruction family and should be read as a parallel architecture-specific API.
- `GMTensor`: a global-memory tensor that corresponds to kernel inputs and outputs.
- `Tensor`: an on-chip tensor in `L1`, `L0A`, `L0B`, `L0C`, or `UB`.
- `DBuff` / `TBuff`: buffered tensor helpers used to model slot-based reuse.
- `Var`: scalar values used for loop bounds, dimensions, and symbolic shapes.
- `OpExec`: runtime entry point for simulator execution or code generation.

## Typical development workflow

1. Start from an exact PyTorch formula.
2. Choose a pipeline topology:
   - cube only
   - cube -> vec
   - vec -> cube
   - vec -> cube -> vec
   - cube -> vec -> cube
3. Implement the kernel in Python.
4. Validate it with `OpExec(..., simulator=True)`.
5. Add explicit `shape_bindings` if repeated scalar dimensions make shape inference ambiguous.
6. Only after the simulator matches the reference should you move on to generated artifacts and hardware-specific execution.

## Repository layout

- `skill/` — skill entrypoint (`skill/SKILL.md`)
- `agent/` — the reusable easyasc DSL to AscendC workflow
  - `agent/ROUTER.md` — progressive-disclosure router
  - `agent/scripts/` — repository-maintenance scripts (including `init.sh`)
  - `agent/assets/` — archived runtime/docs (`ops-easyasc-dsl-runtime.tar.gz`) and example (`ops-easyasc-dsl-example.tar.gz`) payloads
  - `agent/example/` — curated kernel examples and manual demos (restored on demand)
  - `agent/references/` / `agent/playbooks/` / `agent/index/` — catalogs, playbooks, and JSON indexes
- Restored on demand by `agent/scripts/init.sh`:
  - `easyasc/` — DSL runtime and codegen package
  - `doc/` — English documentation
  - `doc_cn/` — Chinese documentation mirror
  - `agent/example/kernels/` — curated sample kernels
  - `agent/example/demo/` — manual runnable examples grouped by device family

Note: `testcases/` is no longer part of the delivered skill bundle.

## Documentation map

Documentation under `doc/` is restored by `agent/scripts/init.sh`:

- [Quick Start](doc/01_quickstart.md)
- [Programming Model](doc/02_programming_model.md)
- [Write Your First Kernel](doc/03_write_your_first_kernel.md)
- [Mixed Pipeline and Synchronization](doc/04_mixed_pipeline_and_sync.md)
- [Simulator and Trace](doc/05_simulator_and_trace.md)
- [Code Generation and Runtime](doc/06_codegen_and_runtime.md)
- [Kernel Patterns](doc/07_kernel_patterns.md)
- [Testing and Validation](doc/08_testing_and_validation.md)
- [API Reference](doc/09_api_reference.md)
- [Troubleshooting](doc/10_troubleshooting.md)
- [Architecture for Contributors](doc/11_architecture_for_contributors.md)
- [Stub to Codegen Name Map](doc/12_stub_to_codegen_name_map.md)

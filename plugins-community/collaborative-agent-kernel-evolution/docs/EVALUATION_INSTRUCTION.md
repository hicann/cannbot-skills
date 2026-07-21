# AscendC Operator Evaluation Instructions

## TL;DR

Use `eval_op.sh` to compile, install, and evaluate in one command:

```bash
# Base version
./eval_op.sh <op_name>

# Evo variant (results saved to the variant's own directory)
./eval_op.sh <op_name> <evo_date> <round> <parallel>
```

Examples:

```bash
./eval_op.sh causal_conv1d_fn
./eval_op.sh causal_conv1d_fn 20260211 1 0
```

The script runs 4 phases: compile (`build.sh`) → install (`.run` package) → generate pybind → evaluate. See below for details on each phase or for running them manually.

---

Manual steps for compiling, deploying, and evaluating an AscendC custom operator.

All commands below use `causal_conv1d_fn` as an example. Replace with your operator name as needed.

## Prerequisites

- ASCEND_HOME_PATH (or ASCEND_AICPU_PATH / BASE_LIBS_PATH) must be set and point to your CANN toolkit installation.
- `npu-smi`, `msopgen`, and `cmake` (>= 3.16) must be available on PATH.
- Python 3 with `torch`, `torch_npu`, and `pip` installed.

## Directory Layout

```
output/<op_name>/
├── <op_name>_reference.py             # Reference PyTorch Model class
├── <op_name>_custom.py                # ModelNew class calling compiled op
├── <op_name>.cpp                      # C++ pybind11 bridge
├── test_cases.py                      # Multi-case test definitions (optional)
├── <OpName>Custom/                    # AscendC project
│   ├── build.sh                       # Build script
│   ├── op_kernel/<op_name>_custom.cpp # Device kernel source
│   ├── op_host/<op_name>_custom.cpp   # Host tiling source
│   ├── op_host/<op_name>_custom_tiling.h
│   └── build_out/                     # Compiled artifacts
├── vendors/customize/                 # Installed operator runtime
└── ascend_op_pybind/                  # PyBind extension + wheel
```

## Step 1: Compile the AscendC Kernel and Host Tiling

```bash
cd output/causal_conv1d_fn/CausalConv1dFnCustom
bash build.sh
```

`build.sh` performs a two-phase build when cross-compile is enabled:

1. Compiles the host-side tiling library (`libcust_opmaster_rt2.0.so`) without cross-compile.
2. Rebuilds everything with cross-compile, linking the native tiling library:
   - `cmake --build build_out --target binary` compiles the AscendC device kernel into a `.o` binary.
   - `cmake --build build_out --target package` packages all artifacts into `custom_opp_ubuntu_aarch64.run`.

Key outputs in `build_out/`:

| Artifact | Description |
|----------|-------------|
| `op_host/libcust_opmaster_rt2.0.so` | Host tiling library |
| `op_host/libcust_opapi.so` | Operator API library |
| `op_kernel/binary/ascend910b/*/CausalConv1dFnCustom_*.o` | Compiled device kernel |
| `_CPack_Packages/.../custom_opp_ubuntu_aarch64.run` | Self-extracting installer |

## Step 2: Install the Operator Package

Run the generated `.run` installer. The `--install-path` **must be an absolute path** pointing to `output/<op_name>`:

```bash
bash output/causal_conv1d_fn/CausalConv1dFnCustom/build_out/custom_opp_ubuntu_aarch64.run \
  --install-path=$(realpath output/causal_conv1d_fn)
```

This populates `vendors/customize/` with the runtime libraries, kernel binaries, and config files.

## Step 3: Generate and Install the PyBind Wheel

From the project root:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/generate_pybind.py causal_conv1d_fn
```

This copies `<op_name>.cpp` into a CppExtension template, builds a wheel, and installs it via `pip install --force-reinstall`. After this step, `import custom_ops_lib` becomes available in Python.

> Only needed once unless you modify the `.cpp` pybind bridge file.

## Step 4: Evaluate

### Single-test mode

Uses `get_inputs()` / `get_init_inputs()` defined in the reference and custom Python files:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn
```

Results are written to `output/causal_conv1d_fn/evaluation_result.txt`.

### Multi-test mode

Runs all test cases defined in a `test_cases.py` file, reports per-case correctness and performance, and computes the geometric mean speedup:

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn \
  --test-cases output/causal_conv1d_fn/test_cases.py
```

### Custom output path

```bash
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn \
  --test-cases output/causal_conv1d_fn/test_cases.py \
  -o output/causal_conv1d_fn/my_results.txt
```

### test_cases.py format

The file must export two things:

- `TEST_CASES`: a list of dicts with `name`, `shape`, and `distribution` keys.
- `generate_input(test_case)`: a function that takes one test case dict and returns the input tensor list.

Example:

```python
import torch

TEST_CASES = [
    {
        "name": "decode_heavy_dim512",
        "shape": {
            "seq_lens": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 512],
            "dim": 512,
        },
        "distribution": "randn",
    },
]

def generate_input(test_case):
    shape = test_case["shape"]
    seq_lens = shape["seq_lens"]
    dim = shape["dim"]
    # ... build and return input tensors
    return [x_tnd, weight, conv_states, query_start_loc, cache_indices, has_initial_state]
```

## Quick Rebuild Cycle

If you only modified files in `op_kernel/` or `op_host/`:

```bash
# 1. Recompile
cd output/causal_conv1d_fn/CausalConv1dFnCustom && bash build.sh

# 2. Reinstall runtime
bash build_out/custom_opp_ubuntu_aarch64.run \
  --install-path=$(realpath ../.)

# 3. Re-evaluate (no pybind regeneration needed)
cd ../../..
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn
```

If you only modified `<op_name>_custom.py` (Python wrapper), skip steps 1-2 and go straight to evaluation.

If you modified `<op_name>.cpp` (pybind bridge), re-run step 3 (generate_pybind) before evaluation.

---

## Evaluating Evolution (Evo) Variants

The evolutionary optimization process produces multiple kernel variants under a directory like:

```
output/<op_name>_evo_<date>/
├── evolution_report.md
├── evolution_insights.md
├── round_1/
│   ├── parallel_0/
│   ├── parallel_1/
│   └── parallel_2/
├── round_2/
│   ├── parallel_0/
│   ...
└── round_3/
    └── ...
```

Each `parallel_*/` directory contains:

```
parallel_0/
├── causal_conv1d_fn.cpp                # C++ pybind bridge (identical to base)
├── causal_conv1d_fn_reference.py       # Reference model (identical to base)
├── causal_conv1d_fn_custom.py          # Custom model wrapper (identical to base)
├── test_cases.py                       # Test cases (may differ from base)
├── evaluation_result.txt               # Previous evaluation output
└── CausalConv1dFnCustom/              # AscendC project (kernel/host differ per variant)
    ├── build.sh
    ├── op_kernel/causal_conv1d_fn_custom.cpp   # <-- variant-specific kernel
    ├── op_host/causal_conv1d_fn_custom.cpp     # <-- variant-specific tiling
    ├── op_host/causal_conv1d_fn_custom_tiling.h
    └── build_out/
```

**Key difference from the base layout**: evo variants have no `vendors/` or `ascend_op_pybind/` of their own. Only the kernel and host code differ between variants — the Python interface and pybind bridge are shared with the base `output/<op_name>/` directory.

### How it works

`evaluate.py` always resolves runtime paths relative to `output/<op_name>/` (the base directory). So the workflow for an evo variant is: **compile the variant → install its `.run` into the base directory (overwriting the runtime) → evaluate**.

When switching between variants, only steps 1-2 need to be repeated.

### Evo Step 1: Compile the variant

```bash
cd output/causal_conv1d_fn_evo_20260211/round_1/parallel_0/CausalConv1dFnCustom
bash build.sh
```

### Evo Step 2: Install into the base directory

Install the variant's `.run` into `output/causal_conv1d_fn/`, overwriting the runtime libraries there:

```bash
bash build_out/custom_opp_ubuntu_aarch64.run \
  --install-path=<project-root>/output/causal_conv1d_fn
```

### Evo Step 3: Generate PyBind (once only)

The pybind bridge `.cpp` is identical across all evo variants and the base version. If pybind was already installed from the base, skip this step. Otherwise, run it once from the project root:

```bash
cd <project-root>
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/generate_pybind.py causal_conv1d_fn
```

### Evo Step 4: Evaluate

From the project root:

```bash
cd <project-root>

# Single-test mode
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn

# Multi-test with the evo variant's own test_cases.py
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn \
  --test-cases output/causal_conv1d_fn_evo_20260211/round_1/parallel_0/test_cases.py

# Save results back to the evo variant directory
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn \
  --test-cases output/causal_conv1d_fn_evo_20260211/round_1/parallel_0/test_cases.py \
  -o output/causal_conv1d_fn_evo_20260211/round_1/parallel_0/evaluation_result.txt
```

### Switching between evo variants

To evaluate a different variant (e.g. `round_2/parallel_1`), repeat steps 1-2 with the new path, then re-run step 4:

```bash
# Compile the other variant
cd <project-root>/output/causal_conv1d_fn_evo_20260211/round_2/parallel_1/CausalConv1dFnCustom
bash build.sh

# Install into base (overwrites previous variant's runtime)
bash build_out/custom_opp_ubuntu_aarch64.run \
  --install-path=<project-root>/output/causal_conv1d_fn

# Evaluate
cd <project-root>
python3 ${CLAUDE_PLUGIN_ROOT}/skills/ascendc-evaluation/scripts/evaluate.py causal_conv1d_fn \
  --test-cases output/causal_conv1d_fn_evo_20260211/round_2/parallel_1/test_cases.py \
  -o output/causal_conv1d_fn_evo_20260211/round_2/parallel_1/evaluation_result.txt
```

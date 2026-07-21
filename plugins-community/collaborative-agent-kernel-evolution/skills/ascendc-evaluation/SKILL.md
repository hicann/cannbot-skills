---
name: ascendc-evaluation
description: Multi-case operator evaluation with precision testing and performance profiling, use when you want to evaluate the performance and correctness of your AscendC operator implementation.
---

## What I do

Complete evaluation pipeline for AscendC operators with:
- **Multi-case testing**: Run multiple test cases per operator
- **Three-way precision**: Compare custom kernel against PyTorch's own numerical error
- **Hardware profiling**: msprof-based performance measurement without dispatch overhead
- **Geometric mean**: Aggregate speedup across all test cases

## Prerequisites

### 1. Operator files in work directory

**CAKE2 operators** (`output/{op_name}/`):
```
output/{op_name}/
├── {op_name}_reference.py      # PyTorch reference implementation
├── {op_name}_custom.py          # Custom operator with ModelNew class
├── {op_name}.cpp                # PyBind bindings
├── {op_name}Custom/             # AscendC project
│   ├── build.sh
│   ├── build_out/
│   │   └── custom_opp_ubuntu_aarch64.run
│   └── ...
└── test_cases.py                # Test cases (you create this)
```

### 2. test_cases.py

**You must create** `test_cases.py` in the operator's work directory. Use exemplars from `skills/ascendc-evaluation/examples/`:

- **Unary operators** (FastGelu, ReLU, etc.): Copy `test_cases_unary.py`
- **Binary operators** (Matmul, Add, etc.): Copy `test_cases_binary.py`

## Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Step 1   Install .run file                                         │
│           bash {op_name}Custom/build_out/custom_opp_*.run           │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 2   Generate PyBind bindings                                  │
│           python3 generate_pybind.py {op_name}                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 3   Create / verify test_cases.py                             │
│           (define TEST_CASES + generate_input())                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  Step 4   Evaluate with multi-case testing                          │
│           python3 run_cases.py {op_name} [--msprof]                 │
│           ├─ Precision check (torch vs custom)                      │
│           ├─ Speedup measurement (msprof, if requested)             │
│           └─ Output JSON + summary table                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Step 1: Install .run file

```bash
bash output/{op_name}/{op_name}Custom/build_out/custom_opp_ubuntu_aarch64.run \
    --install-path=$(pwd)/output/{op_name}
```

**⚠️ CRITICAL**: Always use **absolute paths** with `$(pwd)` expansion. Relative paths fail silently.

This creates:
```
output/{op_name}/vendors/customize/op_api/lib/libcust_opapi.so
```

### Step 2: Generate PyBind bindings

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/generate_pybind.py {op_name}
```

Generates and installs PyBind wheel for `torch.ops.cust.{op_name}`.

### Step 3: Create test_cases.py

**Read `api_description.md` section 8** to understand test cases, then:

```bash
# For unary operators (e.g., FastGelu)
cp ${CLAUDE_SKILL_DIR}/examples/test_cases_unary.py \
   output/{op_name}/test_cases.py

# For binary operators (e.g., Matmul)
cp ${CLAUDE_SKILL_DIR}/examples/test_cases_binary.py \
   output/{op_name}/test_cases.py
```

**Edit** `test_cases.py`:
- Update `TEST_CASES["operator"]` to your operator name
- **Translate EVERY row in the api_description.md test case table into a case entry — do not skip or merge any rows**
- Customize distribution generators if needed

> **Why all cases matter**: api_description.md defines the full benchmark. Omitting cases means the operator is never validated on those shapes/dtypes, which can hide correctness and performance regressions on real workloads (e.g., actual LLM shapes).

**Example mapping** (translate each row from the full table in api_description.md):
```markdown
| 1 | llama3_8b | (2048, 4096) | (32, 2048, 64) | (32, 64, 4096) | 1.0 | 1.0 | bfloat16 | LLaMA-3-8B |
| 2 | small_batch | (64, 128) | (4, 64, 64) | (4, 64, 128) | 1.0 | 1.0 | bfloat16 | 小批次基础测试 |
```
↓
```python
{
    "id": 1,
    "name": "llama3_8b",
    "shape1": [2048, 4096], "shape2": ...,  # all input shapes for this case
    "distribution": "normal",
    "dtype": "bfloat16",
    "description": "LLaMA-3-8B"
},
{
    "id": 2,
    "name": "small_batch",
    "shape1": [64, 128], "shape2": ...,
    "distribution": "normal",
    "dtype": "bfloat16",
    "description": "小批次基础测试"
},
```

### Step 4: Evaluate with multi-case testing

```bash
# Basic evaluation (all test cases)
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name}

# With explicit device ID
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name} --device-id 2

# With JSON output
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name} \
    --json-output output/{op_name}/results.json

# Custom work directory（⚠️ 必须指向父目录，见下方说明）
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name} \
    --work-dir output/{op_name}

# Explicit api_description path (still supported when the doc is not under work_dir)
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name} \
    --work-dir /path/to/operator \
    --api-desc /path/to/api_description.md
```

> ⚠️ **`--work-dir` 必须指向包含 `vendors/customize/` 和 `pybind_lib/` 的父目录**（即 `output/{op_name}/`），
> 而不是 AscendC 工程子目录（如 `output/{op_name}/{op_name}Custom/`）。
> 使用子目录会报：`FileNotFoundError: vendors/customize not found in .../OpNameCustom`
>
> 通常情况下，`api_description.md` 会由上游 `op-desc-generation` 阶段预先复制到 `work_dir/api_description.md`。
> 但 `evaluate.py` 仍保留 `--api-desc` 显式传参能力，用于非标准目录或跨阶段路径保真。

5. Continue to the next step in agent workflow

## Features

### Multi-Case Testing

Runs **all** test cases defined in `test_cases.py` and aggregates results:

```
[1/10] Case 1: basic_1d
  ✅ Precision PASSED
  Reference: 5.10us
  Custom:    4.22us
  Speedup:   1.21x

[2/10] Case 2: basic_2d
  ✅ Precision PASSED
  Reference: 4.60us
  Custom:    4.24us
  Speedup:   1.08x

...

Geometric mean speedup: 1.01x (over 9 passed tests)
```

### Geometric Mean Speedup

Computes **geometric mean** instead of arithmetic mean for aggregate performance:

```python
geo_mean = exp(mean(log(speedups)))
```

This gives equal weight to speedups and slowdowns (1.5x speedup ≈ 0.67x slowdown).

### Three-Way Precision Comparison

Compares custom kernel error against PyTorch's own error:

1. **Golden** (CPU fp64): Highest precision baseline
2. **Ref** (NPU, working dtype): PyTorch reference on NPU
3. **Ans** (NPU, working dtype): Custom kernel on NPU

**Judgment**:
```
ratio = ans_error / max(ref_error, floor)

passed = (
    NO new NaN/Inf introduced
    AND ratios.max_re  <= 10.0
    AND ratios.mean_re <= 2.0
    AND ratios.rmse    <= 2.0
    AND ratios.svec    <= 2.0
)
```

**Meaning**: Custom kernel passes as long as its error is within reasonable multiples of PyTorch's own error.

#### Per-Dtype Tolerances

| dtype | atol | rtol | ulp_tol | sv_th | sv_err |
|-------|------|------|---------|-------|--------|
| bfloat16 | 1e-2 | 1e-2 | 2 | 2^-8 | 2^-16 |
| float16 | 1e-3 | 1e-3 | 2 | 2^-11 | 2^-16 |
| float32 | 1e-5 | 1e-5 | 2 | 2^-14 | 2^-30 |

Integer types default to exact match (atol=0). Quantization operators (float→int) should set `"quantized_output": True` in `test_cases.py` to apply ±1 LSB tolerance — see [precision.md](references/precision.md#quantized-output-flag).

#### Diagnosis Patterns

When a test fails, the system identifies the root cause:

| Pattern | Description | Likely Cause |
|---------|-------------|--------------|
| `nan_inf_introduced` | NaN/Inf in ans but not golden/ref | Division by zero, log(0), exp overflow |
| `zero_output` | Output near-zero everywhere | Placeholder code, incomplete computation |
| `localized_outlier` | max_re high, mean_re ok | Boundary or special-value path issue |
| `systematic_drift` | mean_re exceeded | Algorithm error, dtype cast loss |
| `small_value_error` | svec exceeded | Small-value underflow |
| `uneven_distribution` | rmse exceeded, others ok | Input-dependent precision patterns |

### msprof Performance Measurement

Uses **AdvancedPerformanceEngine** with msprof profiling to measure pure hardware kernel time:

- **No dispatch overhead**: Python/PyTorch/ACLNN dispatch latency eliminated
- **L2 cache control**: Flush L2 before each trial with 10240x10240 matmul + ReduceMax
- **Pattern extraction**: Median-based statistics excluding cold start
- **Accurate for short kernels**: Critical for decode-phase kernels (30-100us)

## test_cases.py Structure

### Unary Operator (Single Input)

```python
import torch

TEST_CASES = {
    "operator": "FastGelu",
    "operator_type": "unary",
    "cases": [
        {
            "id": 1,
            "name": "basic_1d",
            "shape": [256],
            "distribution": "normal",
            "dtype": "float32",
            "description": "基础1D形状"
        },
        # ... more cases
    ]
}

def generate_input(test_case: dict, device: str = "cpu") -> torch.Tensor:
    """Generate input tensor based on distribution."""
    shape = test_case["shape"]
    distribution = test_case["distribution"]

    if distribution == "normal":
        return torch.randn(shape, dtype=torch.float32, device=device)
    elif distribution == "near_zero":
        return torch.randn(shape, dtype=torch.float32, device=device) * 0.1
    # ... other distributions
```

### Binary Operator (Two Inputs)

```python
import torch
from typing import Tuple

TEST_CASES = {
    "operator": "Matmul",
    "operator_type": "binary",
    "cases": [
        {
            "id": 1,
            "name": "small_2d",
            "shape1": [32, 64],
            "shape2": [64, 128],
            "distribution": "normal",
            "dtype": "float32",
            "description": "小规模2D矩阵乘"
        },
        # ... more cases
    ]
}

def generate_inputs(test_case: dict, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate input tensors based on distribution."""
    shape1 = test_case["shape1"]
    shape2 = test_case["shape2"]
    distribution = test_case["distribution"]

    if distribution == "normal":
        input1 = torch.randn(shape1, dtype=torch.float32, device=device)
        input2 = torch.randn(shape2, dtype=torch.float32, device=device)
        return input1, input2
    # ... other distributions
```

## Distribution Types

Common distributions supported in exemplars:

| Distribution | Implementation | Use Case |
|-------------|----------------|----------|
| `normal` | `torch.randn()` | Standard N(0,1) |
| `near_zero` | `torch.randn() * 0.1` | Near-linear region |
| `positive_large` | `torch.rand() * 6 + 2` | [2, 8] saturation region |
| `negative_large` | `torch.rand() * 6 - 8` | [-8, -2] attenuation region |
| `uniform` | `torch.rand()` | [0, 1] uniform |
| `ones` | `torch.ones()` | All ones (stability test) |

**Add custom distributions** as needed per operator by editing `generate_input(s)` function.

## JSON Output Format

```json
{
  "operator": "FastGelu",
  "operator_type": "unary",
  "total_cases": 10,
  "passed_cases": 9,
  "geometric_mean_speedup": 1.01,
  "results": [
    {
      "case_id": 1,
      "case_name": "basic_1d",
      "status": "PASS",
      "speedup": 1.21,
      "ref_time_us": 5.10,
      "custom_time_us": 4.22,
      "precision": {
        "passed": true,
        "ratios": {
          "max_re": 0.90,
          "mean_re": 0.39,
          "rmse": 0.26,
          "svec": 0.0
        },
        "diagnosis": "{'verdict': 'pass', 'failed_checks': [], 'pattern': None, 'root_causes': []}"
      }
    },
    {
      "case_id": 5,
      "case_name": "quant_int8",
      "status": "PASS",
      "speedup": 2.10,
      "ref_time_us": 8.40,
      "custom_time_us": 4.00,
      "precision": {
        "passed": true,
        "ratios": null,
        "diagnosis": null
      }
    },
    {
      "case_id": 10,
      "case_name": "non_aligned",
      "status": "ERROR",
      "error": "Tiling failed (non-aligned dims)",
      "speedup": 0.0
    }
  ]
}
```

## CLI Options

```bash
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py <op_name> [options]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--work-dir` | Work directory path | `output/<op_name>` |
| `--device-id` | NPU device ID | `ASCEND_DEVICE_ID` or `0` |
| `--no-flush-l2` | Disable L2 cache flushing | Enabled |
| `--skip-performance` | Skip performance measurement | Not skipped |
| `--json-output` | JSON output file path | None |
| `--api-desc` | Explicit api_description(.md) path for dtype/doc alignment; still supported even when upstream already copies the doc into work_dir | work_dir/api_description.md or api_desc.md |
| `--verbose` | Show full output including msprof subprocess logs. Default is quiet mode: only case results, errors, and summary. Use `--verbose` when debugging evaluation failures. | Quiet (suppressed) |

### Output Modes

**Quiet mode (default)** — prints one compact line per case + summary, suppresses msprof subprocess noise (~10KB/case). Critical errors and precision failures always print regardless:

```
[1/10] basic                ✅ PASS  ref=    44.1us  custom=    13.2us  speedup=  3.34x
[2/10] small_vocab          ✅ PASS  ref=    36.2us  custom=    12.8us  speedup=  2.84x
...
[ALL PASSED] 10/10 tests passed, geometric mean speedup=2.73x
```

**Verbose mode (`--verbose`)** — full output with all logging.INFO messages and msprof subprocess output. Use when diagnosing profiling issues or investigating unexpected results.

## Example Output

After running evaluation, you'll see a summary table in the console:

```
╔════╤══════════════════╤══════════╤═════════════╤═══════════════╤═══════════╤════════════════════════════════════════╗
║ ID │ Test Case        │ Status   │ Ref (us)    │ Custom (us)   │ Speedup   │ Precision (max_re / mean_re / rmse)    ║
╠════╪══════════════════╪══════════╪═════════════╪═══════════════╪═══════════╪════════════════════════════════════════╣
║ 1  │ basic_1d         │ ✅ PASS  │ 5.10        │ 4.22          │ 1.21x     │ 0.90 / 0.39 / 0.26                     ║
║ 2  │ basic_2d         │ ✅ PASS  │ 4.60        │ 4.24          │ 1.08x     │ 1.10 / 0.42 / 0.26                     ║
║ 3  │ basic_3d         │ ✅ PASS  │ 11.00       │ 4.72          │ 2.33x ⭐  │ 1.03 / 0.42 / 0.27                     ║
║ 4  │ llm_small        │ ✅ PASS  │ 16.54       │ 13.34         │ 1.24x     │ 1.16 / 0.42 / 0.28                     ║
║ 5  │ llm_medium       │ ✅ PASS  │ 60.38       │ 51.76         │ 1.17x     │ 1.14 / 0.42 / 0.28                     ║
║ 6  │ near_zero        │ ✅ PASS  │ 5.04        │ 4.22          │ 1.19x     │ 0.84 / 0.39 / 0.25                     ║
║ 7  │ positive_large   │ ✅ PASS  │ 4.66        │ 4.22          │ 1.10x     │ 0.89 / 0.40 / 0.26                     ║
║ 8  │ negative_large   │ ✅ PASS  │ 4.70        │ 4.22          │ 1.11x     │ 0.76 / 0.39 / 0.25                     ║
║ 9  │ large_negative   │ ✅ PASS  │ 16.02       │ 13.46         │ 1.19x     │ 1.14 / 0.42 / 0.27                     ║
║ 10 │ non_aligned      │ ❌ ERROR │ -           │ -             │ 0.00x     │ Tiling failed (non-aligned dims)       ║
╠════╧══════════════════╧══════════╧═════════════╧═══════════════╧═══════════╧════════════════════════════════════════╣
║ Summary: 9/10 tests passed | Geometric Mean Speedup: 1.01x                                                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
```

## Complete Example

```bash
# 1. Generate operator with CAKE2
# (produces output/fastgelu/)

# 2. Create test_cases.py
cp ${CLAUDE_SKILL_DIR}/examples/test_cases_unary.py \
   output/fastgelu/test_cases.py
# Edit test_cases.py based on api_description.md

# 3. Install .run file
bash output/fastgelu/FastgeluCustom/build_out/custom_opp_ubuntu_aarch64.run \
    --install-path=$(pwd)/output/fastgelu

# 4. Generate PyBind
python3 ${CLAUDE_SKILL_DIR}/scripts/generate_pybind.py fastgelu

# 5. Evaluate
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py fastgelu \
    --device-id 2 \
    --json-output output/fastgelu/results.json
```

## Notes

- **Python environment**: Use `.venv/bin/python3` if available for consistent torch_npu version
- **NPU device**: Always specify `--device-id` explicitly in multi-NPU environments
- **L2 flush**: Enabled by default for realistic performance measurement
- **test_cases.py is required**: Must be created manually from exemplars
- **msprof required**: Must be in PATH for performance measurement

## Troubleshooting

### "test_cases.py not found"
Create it from exemplars in `skills/ascendc-evaluation/examples/`

### "test_cases.py must define TEST_CASES dict"
Ensure your file has `TEST_CASES = {...}` at module level

### "test_cases.py must define generate_input()"
Add the function (copy from exemplar)

### "msprof failed"
Ensure msprof is in PATH: `which msprof`

### "No op_summary CSV found"
Check msprof output directory for profiling artifacts

### "Performance measurement failed"
Try with `--skip-performance` to run precision-only evaluation

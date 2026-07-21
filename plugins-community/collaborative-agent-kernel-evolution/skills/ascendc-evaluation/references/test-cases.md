# Test Cases Reference

## Table of Contents

- [Unary Operator](#unary-operator-single-input)
- [Binary Operator](#binary-operator-two-inputs)
- [Distribution Types](#distribution-types)
- [Parameterized Cases (case_spec)](#parameterized-cases-case_spec)

## Unary Operator (Single Input)

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
    ]
}

def generate_input(test_case: dict, device: str = "cpu") -> torch.Tensor:
    shape = test_case["shape"]
    distribution = test_case["distribution"]
    if distribution == "normal":
        return torch.randn(shape, dtype=torch.float32, device=device)
    elif distribution == "near_zero":
        return torch.randn(shape, dtype=torch.float32, device=device) * 0.1
    # ... other distributions
```

## Binary Operator (Two Inputs)

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
    ]
}

def generate_inputs(test_case: dict, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    shape1 = test_case["shape1"]
    shape2 = test_case["shape2"]
    distribution = test_case["distribution"]
    if distribution == "normal":
        return (torch.randn(shape1, dtype=torch.float32, device=device),
                torch.randn(shape2, dtype=torch.float32, device=device))
    # ... other distributions
```

## Distribution Types

Two distribution systems are used:

**`datagen.py` format** (for `make_tensor()` / `parse_distr()`):

| Format | Implementation | Use Case |
|--------|----------------|----------|
| `normal(0,1)` | `randn * 1 + 0` | Standard N(0,1) |
| `normal(0,0.1)` | `randn * 0.1 + 0` | Near-linear region |
| `uniform(2,8)` | `empty.uniform_(2, 8)` | [2, 8] saturation region |
| `uniform(-8,-2)` | `empty.uniform_(-8, -2)` | [-8, -2] attenuation region |
| `uniform(0,1)` | `empty.uniform_(0, 1)` | Standard uniform |

**`test_cases.py` exemplar format** (for `generate_input()`):

| Symbolic Name | Implementation | Use Case |
|--------------|----------------|----------|
| `normal` | `torch.randn()` | Standard N(0,1) |
| `near_zero` | `torch.randn() * 0.1` | Near-linear region |
| `positive_large` | `torch.rand() * 6 + 2` | [2, 8] saturation region |
| `negative_large` | `torch.rand() * 6 - 8` | [-8, -2] attenuation region |
| `uniform` | `torch.rand()` | [0, 1] uniform |
| `ones` | `torch.ones()` | All ones (stability test) |

## Parameterized Cases (case_spec)

For tool-driven workflows, generate or append test cases via JSON spec:

```json
{
    "case_id_start": 0,
    "cases": [
        {"batch_size": 1, "seq_len": 128, "var0_shape": [1, 128], "var0_dtype": "float16"}
    ],
    "grid": {
        "batch_size": [1, 2],
        "seq_len": [128, 256],
        "var0_shape": [[1, 128], [1, 256]],
        "var0_dtype": ["float16"]
    }
}
```

```shell
# Generate cases only
.venv/bin/python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py <op_name> \
    --case-spec /path/to/case_spec.json --append-cases --generate-cases-only

# Evaluate with case spec
.venv/bin/python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py <op_name> \
    --case-spec /path/to/case_spec.json
```
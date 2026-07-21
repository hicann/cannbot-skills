# Test Cases Exemplars

This directory contains template test case files for creating operator-specific `test_cases.py`.

## Available Exemplars

### 1. `test_cases_unary.py` - For Single-Input Operators

**Use for**: FastGelu, ReLU, Sigmoid, Tanh, Exp, Log, Sqrt, etc.

**Structure**:
```python
TEST_CASES = {
    "operator": "FastGelu",
    "operator_type": "unary",
    "cases": [
        {
            "id": 1,
            "name": "basic_1d",
            "shape": [256],           # Single shape field
            "distribution": "normal",
            "dtype": "float32",
            "description": "基础1D形状"
        },
        # ... more cases
    ]
}

def generate_input(test_case: dict, device: str = "cpu") -> torch.Tensor:
    """Single tensor input generator"""
    # ...
```

### 2. `test_cases_binary.py` - For Two-Input Operators

**Use for**: Matmul, Add, Mul, Sub, Div, etc.

**Structure**:
```python
TEST_CASES = {
    "operator": "Matmul",
    "operator_type": "binary",
    "cases": [
        {
            "id": 1,
            "name": "small_2d",
            "shape1": [32, 64],      # Two shape fields
            "shape2": [64, 128],
            "distribution": "normal",
            "dtype": "float32",
            "description": "小规模2D矩阵乘"
        },
        # ... more cases
    ]
}

def generate_inputs(test_case: dict, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """Two tensor input generator"""
    # ...
```

## How to Use

### Step 1: Copy the Appropriate Exemplar

```bash
# For unary operators (FastGelu, ReLU, etc.)
cp skills/ascendc-evaluation/examples/test_cases_unary.py \
   output/{op_name}/test_cases.py

# For binary operators (Matmul, Add, etc.)
cp skills/ascendc-evaluation/examples/test_cases_binary.py \
   output/{op_name}/test_cases.py
```

### Step 2: Update Operator Name

Edit `test_cases.py` and change the operator name:

```python
TEST_CASES = {
    "operator": "YourOperatorName",  # ← Update this
    "operator_type": "unary",  # or "binary"
    # ...
}
```

### Step 3: Fill Test Cases from api_description.md

Read **section 8** of your operator's `api_description.md` and copy the test cases table.

**Example from api_description.md**:
```markdown
| ID | Name         | Shape      | Distribution | dtype   | Description |
|----|--------------|------------|--------------|---------|-------------|
| 1  | basic_1d     | (256,)     | normal       | float32 | 基础1D形状  |
| 2  | basic_2d     | (128, 256) | normal       | float32 | 基础2D形状  |
```

**Converted to test_cases.py**:
```python
"cases": [
    {
        "id": 1,
        "name": "basic_1d",
        "shape": [256],
        "distribution": "normal",
        "dtype": "float32",
        "description": "基础1D形状",
    },
    {
        "id": 2,
        "name": "basic_2d",
        "shape": [128, 256],
        "distribution": "normal",
        "dtype": "float32",
        "description": "基础2D形状",
    },
    # ...
]
```

### Step 4: Customize Distributions (Optional)

If your operator needs custom distributions, add them to the `generate_input(s)` function:

```python
def generate_input(test_case: dict, device: str = "cpu") -> torch.Tensor:
    shape = test_case["shape"]
    distribution = test_case["distribution"]

    if distribution == "normal":
        return torch.randn(shape, dtype=torch.float32, device=device)
    elif distribution == "custom_range":  # ← Add custom distribution
        return torch.rand(shape, dtype=torch.float32, device=device) * 10 - 5
    # ... more distributions
```

## Distribution Types Reference

Standard distributions provided in exemplars:

| Distribution | Implementation | Range | Use Case |
|-------------|----------------|-------|----------|
| `normal` | `torch.randn()` | N(0,1) | General testing |
| `near_zero` | `torch.randn() * 0.1` | ~[-0.3, 0.3] | Near-linear region |
| `positive_large` | `torch.rand() * 6 + 2` | [2, 8] | Saturation region (activations) |
| `negative_large` | `torch.rand() * 6 - 8` | [-8, -2] | Attenuation region (activations) |
| `uniform` | `torch.rand()` | [0, 1] | Uniform distribution |
| `ones` | `torch.ones()` | 1 | Stability test |

## Common Patterns

### Multiple dtypes

Test the same shape with different dtypes:

```python
{
    "id": 1,
    "name": "basic_fp32",
    "shape": [1024, 1024],
    "distribution": "normal",
    "dtype": "float32",
    "description": "FP32 baseline"
},
{
    "id": 2,
    "name": "basic_fp16",
    "shape": [1024, 1024],
    "distribution": "normal",
    "dtype": "float16",
    "description": "FP16 precision"
},
```

### Scale testing

Test progressively larger sizes:

```python
{
    "id": 1,
    "name": "small",
    "shape": [1024],
    "distribution": "normal",
    "dtype": "float32",
    "description": "Small scale"
},
{
    "id": 2,
    "name": "medium",
    "shape": [8192],
    "distribution": "normal",
    "dtype": "float32",
    "description": "Medium scale"
},
{
    "id": 3,
    "name": "large",
    "shape": [65536],
    "distribution": "normal",
    "dtype": "float32",
    "description": "Large scale"
},
```

### Edge cases

Test boundary conditions:

```python
{
    "id": 1,
    "name": "min_size",
    "shape": [1],
    "distribution": "normal",
    "dtype": "float32",
    "description": "Minimum size"
},
{
    "id": 2,
    "name": "non_aligned",
    "shape": [1023],  # Not power of 2
    "distribution": "normal",
    "dtype": "float32",
    "description": "Non-aligned dimension"
},
```

## Tips

1. **Start with the exemplar**: Always copy from `examples/` rather than starting from scratch
2. **Match api_description.md**: Test cases should match section 8 of the operator's api_description.md
3. **Test important shapes**: Include small, medium, large, and edge-case shapes
4. **Test important distributions**: For activations, test near-zero, positive, and negative ranges
5. **Add descriptions**: Use Chinese descriptions matching the original design doc
6. **Keep it simple**: Don't add unnecessary complexity to distribution generators

## Validation

After creating `test_cases.py`, validate it:

```bash
# Check syntax
python3 -m py_compile output/{op_name}/test_cases.py

# Test import
python3 -c "import sys; sys.path.append('output/{op_name}'); import test_cases; print(test_cases.TEST_CASES)"

# Run evaluation
python3 ${CLAUDE_SKILL_DIR}/scripts/evaluate.py {op_name}
```

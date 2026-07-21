---
name: reference-generation
disable-model-invocation: true
description: Generate reference PyTorch implementation (nn.Module with get_init_inputs/get_random_inputs) from operator description JSON. Faithfully reproduces the Golden definition when provided. Use after op-desc-generation.
---

## What I do

Generate functional reference PyTorch code from operator description JSON files. Output uses `torch.nn` with `nn.Module` classes and proper type hints.

## Workflow

1. Read operator description json file `{op_name}_op_desc.json`
2. Extract operator name, category, description, shape information, parameters
3. **Extract and copy** the reference code from the given `api_description.md`. **Do not write any new reference code on your own**, simply copy them from the given snippet. Examples are:
   - section ## 8 in @references/tanh_api_description.md
   - section ## 6 in @references/grouped_matmul_api_description.md
4. Read output example code:
   - @references/hardtanh.py — for operators with deterministic parameters (e.g., `torch.ones`, `torch.zeros`)
   - @references/linear_softplus.py — for operators with **random learnable parameters** (demonstrates the required `torch.manual_seed(42)` pattern)
5. Generate PyTorch code:
   - Imports: specify using `torch` and `torch.nn`
   - Model Definition: `nn.Module` class definition, add proper `__init__` and `forward` methods
   - Configurations: hyper-parameters and two helpers `get_inputs()` and `get_init_inputs()` consumed by the model.
6. Save to `output/{op_name}/{op_name}_reference.py`
7. Continue to the next step in agent workflow

**key Points**:
1. The `get_inputs()` function should generate input tensors according to the shape_info provided:
   - Use the specified shapes and dtypes from shape_info
   - If dtype is "float32", use `torch.rand()`
   - If dtype is "int32" or "int64", use `torch.randint()`
   - If dtype is "float16" or "bfloat16", use `torch.rand()` and cast to the dtype
2. The `get_init_inputs()` function should return the initialization parameters specified in the parameters field
3. The `Model` class should implement the operator according to the description
4. If the operator is to compute the gradient, you should:
   - Please manually implement the computation in the `forward` function instead of using the builtin autograd
   - If the gradient computation requires the output of the original forward pass, compute this output in `get_inputs()` and pass it as an input to the `forward` function

### ⚠️ CRITICAL: Learnable Parameter Initialization for Evaluation Compatibility

The evaluation framework creates **two separate model instances** (`Model` and `ModelNew`) sequentially from the same `get_init_inputs()`. Because they are created one after the other, any **random** parameter initialization (e.g., `nn.Linear`, `torch.randn`) will consume different random numbers for each model, producing **different weights** and causing evaluation to fail.

**Rules for models with learnable parameters** (e.g., weight matrices, embeddings):

1. **Always call `torch.manual_seed(<fixed_seed>)` at the start of `__init__`** before any random parameter initialization. This ensures both `Model` and `ModelNew` produce identical weights regardless of external random state.

2. **Avoid using `nn.Linear`, `nn.Conv2d`, or other `nn.Module` layers** that internally generate random parameters. Instead, create `nn.Parameter` tensors directly.

3. **Use deterministic initialization** when possible (e.g., `torch.ones`, `torch.zeros`). If random initialization is needed, always guard it with a fixed seed.

**Example — WRONG** (weights differ between Model and ModelNew):
```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)  # ❌ random init without seed
```

**Example — CORRECT** (weights are identical):
```python
class Model(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        torch.manual_seed(42)  # ✅ fixed seed ensures reproducibility
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
```

**Why this matters**: The evaluation script (`evaluate.py`) calls `set_seed(1024)` once, then creates `Model` and `ModelNew` in sequence. Without a local fixed seed in `__init__`, the two models get different random weights, causing correctness checks to fail even when the kernel computation is correct.


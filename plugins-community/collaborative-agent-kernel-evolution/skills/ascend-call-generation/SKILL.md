---
name: ascend-call-generation
disable-model-invocation: true
description: Generate AscendC project scaffold (pybind, CMake, host code, kernel skeleton) from functional PyTorch for pure Vector operators. Use after functional-conversion, before dsl-baseline-generation.
---

## What I do

Generate Ascend operator invocation code and project json configuration file from functional PyTorch code and then create initial Ascend C project.

## Workflow
1. read input functional pytorch code `{op_name}_functional.py`
2. read three example output files in `references/{op_name}` dir by op catagory
    - For pool ops: `average_pooling2d`
    - For reduction ops: `sum_reduction_over_a_dimension`
    - For loss ops: `mse_loss`
    - For matmul/linear ops: `matmul` (demonstrates multi-input shape inference)
    - Other: `layer_norm`
3. generate code and save in file.
    - project json code for msopgen to create the custom Ascend operate project
    - python bind code for C -> python API interface
    - python code to call the custom Ascend C operate
4. save all file in `output/{op_name}/` directory.
5. create Ascend C project(run gen_project.py)
6. Continue to the next step in agent workflow

### Generation Task
Construct operator name as `{op_name}_custom` and convert to PascalCase for operator definition.

#### 1. project_json code
JSON schema defining the custom AscendC operator:
- **inputs**: ND tensors mapping to [module_fn] tensor arguments
- **outputs**: ND tensors mapping to [module_fn] return values
- **attributes**: scalar/config parameters from [module_fn]

**Critical**: Ensure 1-to-1 correspondence with [module_fn] arguments. Do NOT add attributes that don't appear in [module_fn] signature.

##### ⚠️ CRITICAL: dtype Mapping from op_desc.json

The `"type"` array in the project JSON determines which dtype variants the build system compiles. **Missing a dtype here means that dtype is silently unsupported at runtime** (error `EZ1001: Io input dtype or format is not supported`).

**Process — MUST follow before writing the project JSON:**

1. Read `{op_name}_op_desc.json` and collect ALL unique `dtype` values from:
   - `shape_info.input_shapes[*].dtype`
   - `test_cases[*].dtype`
2. Also check `api_description.md` section "数据类型" / "Data Types" for the authoritative supported type list.
3. Convert to CANN type names (only `float32` differs):

   | op_desc dtype | project.json type name |
   |--------------|----------------------|
   | `float32`    | `"float"`            |
   | `float16`    | `"float16"`          |
   | `bfloat16`   | `"bfloat16"`         |
   | `int32`      | `"int32"`            |

4. Include **ALL** collected types in the `"type"` array for every input and output tensor. Each type entry must have a matching `"format"` entry (typically `"ND"`).

**Example** — if test_cases contain float32, float16, and bfloat16:
```json
"type": ["float16", "bfloat16", "float"],
"format": ["ND", "ND", "ND"]
```

**Validation**: count unique dtypes in test_cases. The `"type"` array length must be ≥ that count.

Save json code in `{op_name}_project.json`

#### 2. python_bind code
C++ pybind11 code connecting AscendC operator to PyTorch:
- Include headers: `torch/library.h`, `pytorch_npu_helper.hpp`
- Implement `*_impl_npu` function calling `EXEC_NPU_CMD`
- Register with `TORCH_LIBRARY_IMPL`
- Export via `PYBIND11_MODULE`

**Requirements**:
- Function name must match `{op_name}_custom`
- Handle negative dimension parameters (dim) properly
- Expose function with **exact same signature** as [module_fn]

##### ⚠️ CRITICAL: Output Tensor Shape Allocation

The output tensor must be allocated with the **correct shape** before calling `EXEC_NPU_CMD`. Do NOT blindly use `at::empty_like(input)` — this only works when the output shape equals the first input shape.

**Determine the output shape category:**

| Category | Output Shape | Allocation Method |
|----------|-------------|-------------------|
| **Same as input** (activation, normalization) | Same as first input | `at::empty_like(input)` |
| **Derived from multiple inputs** (matmul, linear) | Depends on 2+ inputs | Compute shape explicitly from input dimensions |
| **Reduced dimension** (reduction, loss) | Smaller than input | Compute shape by modifying input shape |
| **Scalar output** (global reduction, loss) | `{}` (0-dim) | `at::empty({}, input.options())` |

**Example — matmul/linear (output shape from two inputs):**
```cpp
// input: [M, K], weight: [N, K] → output: [M, N]
at::Tensor result = at::empty({input.size(0), weight.size(0)}, input.options());
```

**Example — reduction (output shape with reduced dim):**
```cpp
// Reduce along adjusted_dim, keeping dim with size 1
auto output_shape = input.sizes().vec();
output_shape[adjusted_dim] = 1;
at::Tensor result = at::empty(output_shape, input.options());
```

**Example — same shape (activation, normalization):**
```cpp
at::Tensor result = at::empty_like(input);
```

Save C code in `{op_name}.cpp`


#### 3. model code
`ModelNew(nn.Module)` class that:
- Replicates original `Model` functionality
- Calls custom AscendC operator via `custom_ops_lib.{op_name}_custom`
- Maintains same `__init__` and `forward` signatures
- Imports: `torch`, `torch.nn`, `torch_npu`, `custom_ops_lib`

**Note**: Only implement specific behavior used by original Model. For example, if Model always uses `dim=1`, don't implement arbitrary dim cases.

##### ⚠️ CRITICAL: Weight Initialization Must Match Model Exactly

The `ModelNew.__init__` must use **exactly the same parameter initialization** as `Model.__init__` in the reference code (from `{op_name}_reference.py`). This includes:
- The same `torch.manual_seed(42)` call (if present in `Model`)
- The same tensor creation calls (`torch.randn`, `torch.ones`, etc.)
- The same shapes and scaling factors

If `Model.__init__` contains `torch.manual_seed(42)` followed by `torch.randn(...)`, then `ModelNew.__init__` must contain the identical sequence. This ensures both models produce identical weights during evaluation.

Save python code in `{op_name}_custom.py`

### create AscendC project
You should use `gen_project.py` to create the Ascend C project.

**Python环境**: 始终使用 `.venv/bin/python3` 代替 `python3`(Unless .venv is not available)。

**⚠️ CRITICAL: 必须使用 `--output-dir` 指定输出目录为 `output/{op_name}`**，否则项目会生成到 `output/` 根目录下，需要手动搬移。

Usage:
```shell
.venv/bin/python3 ${CLAUDE_SKILL_DIR}/scripts/gen_project.py <op_name> <json_file_path> --output-dir output/<op_name>
```

Example:
```shell
.venv/bin/python3 ${CLAUDE_SKILL_DIR}/scripts/gen_project.py AvgPool3D output/AvgPool3D/AvgPool3D_project.json --output-dir output/AvgPool3D
```

This generates the project at `output/{op_name}/{OpNameCustom}/` (e.g., `output/AvgPool3D/AvgPool3DCustom/`).

Check if the project generated successfully. If not, analyze the error and retry.

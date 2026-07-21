---
name: op-desc-generation
disable-model-invocation: true
description: Generate operator description JSON (shapes, dtypes, attributes) from API description or user specification. First step in the cake agent pipeline.
---


## What I do

Generate operator description JSON files from the template for Ascend operator development. This is the first step in the operator generation pipeline.

## Workflow

1. Read the template file @references/op_desc_template.json and example file @references/layer_norm_op_desc.json
2. If this run already has an `api_description(.md)` source and you know the downstream `work_dir`, copy it into `work_dir/api_description.md` first.
   - Entrypoint: `python3 ${CLAUDE_SKILL_DIR}/scripts/prepare_api_desc.py --work-dir <work_dir> --api-desc <path>`
   - This keeps downstream tools compatible with the conventional `work_dir/api_description.md` lookup.
3. Analyze the operator requirements and extract necessary information. If not mentioned, refer to the API definition in PyTorch.
   - Operator name (e.g., average_pooling2d, conv2d, softmax)
   - Category (choose from: pooling, activation, convolution, reduction, normalization, matmul, loss, optimizer, math)
   - Mathematical description and behavior
   - Input tensor shapes and data types
   - Output tensor shapes and data types
   - Attributes (kernel_size, stride, padding, eps, etc.)
4. Create operator description JSON with appropriate fields:
   - `op_name`: Unique identifier for the operator
   - `category`: Operator classification for organization
   - `description`: Clear mathematical definition and behavior
   - `shape_info`: One **representative** tensor configuration (shapes + dtypes) chosen from the test cases, used for initial reference code and DSL development
   - `attributes`: The parameters other than the tensor parameters which are generally kwargs and have default values
   - `test_cases`: **MUST include** when `api_description.md` is available and contains a test case table — add all test cases from that table as an array. Each entry should capture the input shapes, dtypes, and any case-specific attribute values. If `api_description.md` is absent or has no test case section, `test_cases` may be omitted.
5. Save the JSON file in current project `output/{op_name}/{op_name}_op_desc.json`
6. After saving, state: `"op_desc.json saved: [N] test cases"` (where N=0 is acceptable only if no api_description.md test cases exist)
7. Continue to the next step in agent workflow

## Downstream compatibility notes

- `api_description` is now expected to be copied upstream into `work_dir` by this skill (or its helper) when that document is already available.
- Downstream evaluation still supports an explicit `--api-desc /path/to/api_description.md` override for cases where the doc is not colocated with `work_dir`.

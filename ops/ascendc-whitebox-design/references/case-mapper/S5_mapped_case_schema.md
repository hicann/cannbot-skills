# S5 MappedCase Schema

> 本文定义 Step 5 mapped cases 的固定数据结构。Step 5.2 生成的 path/network base mapped cases、shape low cases，Step 5.4 生成的 empty low cases，以及 Step 5.5 生成的 range cases 必须遵守本文 schema。

## 顶层结构

MappedCase 顶层结构固定为：

```json
{
  "id": "case00000",
  "source": "path",
  "params": {},
  "inputs": {
    "input_name": {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [1],
      "param_type": "REQUIRED",
      "data_range": "normal"
    }
  },
  "outputs": {
    "output_name": {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [1],
      "param_type": "REQUIRED",
      "present": true
    }
  },
  "meta": {}
}
```

字段要求：

- `id` 必须是字符串。
- `source` 必须是本文定义的合法取值。
- `params` 必须是 object。
- `inputs` 必须是 object，key 为算子 input 名。
- `outputs` 必须是 object，key 为算子 output 名。
- `meta` 必须是 object。
- 不得写入 `meta.supported_data_ranges`。

## Source 取值

`source` 表示当前 MappedCase 在 Step 5 流程中的来源或变体类型。

| source | 使用位置 | 含义 |
|--------|----------|------|
| `path` | `S5_mapped_cases_path.json` | 由 `S2P2_cases.json` 映射得到的 path base mapped case。 |
| `network` | `S5_mapped_cases_network.json` | 由 `S2P1_low_configs.json` 映射得到的 network base mapped case。 |
| `shape` | `S5_mapped_cases_low_shape.json`、`S5_cases_low.json`、`S5_cases_high.json` | 由 path/network base mapped case 生成的 shape low case。 |
| `empty` | `S5_cases_low.json`、`S5_cases_high.json` | 由 Step 5.4 固定脚本基于 shape low case 机械追加的 empty low case。 |
| `range` | `S5_cases_high.json` | 由 Step 5.5 固定脚本基于 low case 和 data_range policy 机械生成的 range high case。 |

## Base MappedCase

`build_low_base_case(record, source, index)` 输出 base mapped case。path 和 network 的 base mapped case 使用同一顶层 schema，区别只在字段取值来源和 shape/dtype/format 映射规则。

base mapped case 要求：

- `source` 必须为 `path` 或 `network`。
- `params` 只包含 `S5_variable_semantics.md` 的 `params` 章节声明的字段。
- `inputs` 和 `outputs` 必须按本文 schema 构造。
- `meta` 只写入 `S5_variable_semantics.md` 声明需要保留的 mapper 元信息。
- 所有 input 的 `data_range` 必须为 `normal`。

## Shape Low Case

`make_path_shape_case(case)` 和 `make_network_shape_case(case)` 输出 shape low case。

shape low case 要求：

- `source` 必须为 `shape`。
- 顶层 schema 与 base mapped case 保持一致。
- `inputs` 必须包含最终用于 low 档执行的完整 input shape/dtype/format 信息。
- `outputs` 必须按最终 `inputs`、`params` 和 `meta` 重新派生。
- `meta` 必须保留可审计的 base case 关联和 shape variant 信息。
- 所有 input 的 `data_range` 必须为 `normal`。

## Empty Low Case

empty low case 由 Step 5.4 固定脚本基于 `S5_mapped_cases_low_shape.json` 中的 shape low case 机械追加，输出到最终 low 用例文件 `S5_cases_low.json`。

empty low case 要求：

- `source` 必须为 `empty`。
- 顶层 schema 与 shape low case 保持一致。
- `meta` 必须保留 empty variant 的审计信息。
- 所有 input 的 `data_range` 必须为 `normal`。

## Range High Case

range high case 由 Step 5.5 固定脚本基于 `S5_cases_low.json` 和 `S5_data_range_policy.json` 机械生成，输出到最终 high 用例文件 `S5_cases_high.json`。

range high case 要求：

- `source` 必须为 `range`。
- 顶层 schema 与 low case 保持一致。
- `meta` 必须保留 range variant 的审计信息。
- 至少一个参与 data_range expansion 的 input 使用非 `normal` data_range。

## Inputs

`inputs` 表示算子输入参数空间。普通 Tensor 和 TensorList 都作为 `inputs` 下的 entry 表示，并通过 `kind` 区分。

### Tensor Input

普通 Tensor input 必须使用以下结构：

```json
{
  "kind": "tensor",
  "dtype": "float16",
  "format": "ND",
  "shape": [2, 3],
  "param_type": "REQUIRED",
  "data_range": "normal"
}
```

字段要求：

- `kind` 固定为 `tensor`。
- `dtype` 按 `S5_variable_semantics.md` 的 `dtype` 规则设置。
- `format` 按 `S5_variable_semantics.md` 的 `format` 规则设置，必须是非空字符串。
- `shape` 必须是整数 list。
- `param_type` 使用 operator input schema 中的参数类型，例如 `REQUIRED` 或 `OPTIONAL`。
- low 档下 `data_range` 固定为 `normal`。
- optional input 的 presence 或缺省规则必须由 `S5_variable_semantics.md` 明确说明。

### TensorList Input

DYNAMIC TensorList input 必须使用以下结构：

```json
{
  "kind": "tensor_list",
  "dtype": "float16",
  "format": "ND",
  "param_type": "DYNAMIC",
  "tensor_count": 2,
  "data_range": "normal",
  "tensors": [
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [2, 3],
      "data_range": "normal"
    },
    {
      "kind": "tensor",
      "dtype": "float16",
      "format": "ND",
      "shape": [4, 3],
      "data_range": "normal"
    }
  ]
}
```

字段要求：

- `kind` 固定为 `tensor_list`。
- `format` 按 `S5_variable_semantics.md` 的 `format` 规则设置，必须是非空字符串。
- `param_type` 固定为 `DYNAMIC`。
- `tensor_count = len(tensors)`。
- `tensors` 必须是 list。
- 每个子 tensor 必须包含 `kind = "tensor"`、`dtype`、`format`、`shape`、`data_range`。
- low 档下，外层和每个子 tensor 的 `data_range` 都必须为 `normal`。
- 子 tensor dtype 默认与外层 dtype 一致；只有 `S5_variable_semantics.md` 明确异构 dtype 语义时才允许不同。
- 子 tensor format 默认与外层 format 一致；只有 `S5_variable_semantics.md` 明确异构 format 语义时才允许不同。
- 子 tensor shape 是否一致、如何派生，必须由 `S5_variable_semantics.md` 的 `input shapes` 或 `network mapping` 明确说明。

## Outputs

`outputs` 表示算子输出参数空间。

### Output Tensor

每个 output tensor 必须使用以下结构：

```json
{
  "kind": "tensor",
  "dtype": "float16",
  "format": "ND",
  "shape": [2, 3],
  "param_type": "REQUIRED",
  "present": true
}
```

字段要求：

- `kind` 固定为 `tensor`。
- `dtype` 按 `S5_variable_semantics.md` 的 `outputs` 规则设置。
- `format` 按 `S5_variable_semantics.md` 的 `format` 规则设置，必须是非空字符串。
- `shape` 必须由 `derive_output_shapes(inputs, params, meta)` 的规则派生，或由指导书声明的 fixed output shape 给出。
- `param_type` 使用 operator output schema 中的参数类型。
- `present` 必须是 bool。
- optional output 的 `present` 规则必须由 `S5_variable_semantics.md` 明确说明。

## Params

`params` 表示执行 mapped case 时需要传递给算子的非 tensor 执行参数，并作为 TTK Kernel CSV `attributes` 列的候选值来源。

允许进入 `params` 的字段仅限两类：operator attributes，以及 `_def.cpp` 中 `.ValueDepend()` 标记的 const input values。TTK 转换阶段会再按 `_def.cpp` 提取到的 `Attr` 注册名和 `.ValueDepend()` 输入名过滤后写入 CSV `attributes`。

字段要求：

- `params` 必须是 object。
- 只允许包含 `S5_variable_semantics.md` 的 `params` 章节声明的字段。
- dtype、format、shape、tiling/router 信息、mapper 审计信息和 meta-only 字段不得写入 `params`。
- 白盒路径、tiling key、group、shape 构造变量、case 来源和网络来源等审计字段必须写入 `meta`，不得写入 `params`。

## Meta

`meta` 表示 mapper 和 variant 审计信息。

字段要求：

- `meta` 必须是 object。
- 只写入模板静态区要求的审计字段，或 `S5_variable_semantics.md` 明确要求保留的 mapper 元信息。
- 不得写入 `meta.supported_data_ranges`。
- 不得把应进入 `params`、`inputs` 或 `outputs` 的执行字段写入 `meta`。
- tiling/router 信息和 mapper 审计字段应写入 `meta`，用于追踪 path、tiling key、shape 构造变量和 case 来源。

## Data Range

Step 5.2 只生成 low 档 mapped cases，所有 input 的 `data_range` 必须为 `normal`。

TensorList input 需要同时满足：

- 外层 `data_range = "normal"`。
- 每个子 tensor 的 `data_range = "normal"`。

非 `normal` data_range 扩展由 Step 5.5 处理，不属于 Step 5.2 schema 生成范围。

## 禁止事项

- 不使用 `tensors.inputs` / `tensors.outputs` 作为顶层结构；`inputs` 和 `outputs` 必须位于 MappedCase 顶层。
- 不在 `params`、`inputs`、`outputs`、`meta` 之外新增顶层执行字段。
- 不把未在 `S5_variable_semantics.md` 声明的字段写入 `params`。
- 不把 tiling/router 信息或 mapper 审计字段写入 `params`。
- 除 operator attribute 等执行参数外，不在 `params` 与 `meta` 中重复写入同名字段。
- 不在 low 档 mapped case 中写入非 `normal` data_range。
- 不写入 `meta.supported_data_ranges`。

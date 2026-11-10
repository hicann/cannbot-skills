# golden_flow.json schema

`golden_flow.json` 由大模型从 golden / 参考实现 / 算子语义提取，是理论性能计算的**唯一算子相关输入**。

## 示例

```json
{
  "operator": "gelu",
  "memory": {
    "gm_read_bytes": "numel * elem_bytes",
    "gm_write_bytes": "numel * elem_bytes",
    "working_set_bytes": "gm_read_bytes + gm_write_bytes"
  },
  "compute": [
    {"kind": "vector", "op": "Cast", "elements": "numel", "src_dtype": "@in", "dst_dtype": "fp32"},
    {"kind": "vector", "op": "Mul", "elements": "numel", "dtype": "fp32"},
    {"kind": "vector", "op": "Exp", "elements": "numel", "dtype": "fp32"},
    {"kind": "vector", "op": "Add", "elements": "numel", "dtype": "fp32"}
  ],
  "bandwidth_model": "auto"
}
```

## 变量

`shape`、`dims`、`numel`、`total_elems`、`elem_bytes`、`input_bytes`、`dtype`、`dtypes`、`attrs`，以及硬件档案中的标量字段：`aiv_core_num`、`frequency_hz`、`ub_size`、`l2_capacity_bytes` 等。

## compute

- `kind=vector`：`elements` 必填，`dtype` 或 `src_dtype/dst_dtype` 指定元素类型；`op` 用于查询硬件指令代价；`count` 可选；`cost` 可覆盖。
- `kind=vector` 可指定 `unit`，不同 unit 的耗时取 max。
- `kind=cube`：使用顶层 `cube` 数组；`flops` 或 `macs` × 2，`dtype` 必填。

## memory

- `gm_read_bytes` / `gm_write_bytes`：GM 读写字节数。
- `gm_reread_bytes`：重复读取字节数（仅报告）。
- `working_set_bytes`：决定 L2/HBM 带宽模型。
- `bandwidth_model`：`auto` / `l2` / `hbm`。

## 表达式

支持 `+ - * / // % **`、比较、布尔、条件表达式，以及 `ceil/floor/min/max/abs/round/int/float/len/prod/sum`。不支持字符串、属性访问、任意函数调用。`**` 的指数绝对值不得超过 64，超出会直接报错；布尔运算按 Python 短路语义求值。

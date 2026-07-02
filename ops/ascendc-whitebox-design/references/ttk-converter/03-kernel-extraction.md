# Kernel 模式字段提取规则

## S5 JSON → CSV 列映射

| S5 JSON 字段 | CSV 列 | 提取方式 |
|-------------|--------|---------|
| case["id"] | `testcase_name` | 直接使用 `case["id"]` |
| — | `network_name` | 留空 |
| `case["tensors"]["inputs"]` 各 tensor 的 `shape` | `input_shapes` | 遍历 inputs dict 的 values。REQUIRED（`dict`）→ 取 `spec["shape"]` 转 tuple。DYNAMIC（`list[dict]`）→ 遍历子 tensor 取各 `sub["shape"]`，组成内层 tuple `(shape_0, shape_1, ...)`，作为外层 tuple 的 1 个元素。可选输入 `spec is None` 时用 `None` 占位 |
| `case["tensors"]["inputs"]` 各 tensor 的 `dtype` | `input_dtypes` | 遍历 inputs dict 的 values。REQUIRED（`dict`）→ 取 `spec["dtype"]`。DYNAMIC（`list[dict]`）→ 取 `spec[0]["dtype"]`，包为压缩嵌套 `(("float16",),)`（TTK 自动广播到所有子 tensor）。可选输入 `spec is None` 时用 `None` 占位 |
| — | `input_formats` | 默认 `('ND',)` × tensor 数量（REQUIRED 和 DYNAMIC 均保持扁平，TTK 通过 `get()` 回退处理） |
| `case["tensors"]["outputs"]` 各 tensor 的 `shape` | `output_shapes` | 遍历 outputs dict 的 values。REQUIRED（`dict`）→ 取 `spec["shape"]` 转 tuple。DYNAMIC（`list[dict]`）→ 遍历子 tensor 取各 `sub["shape"]`，组成内层 tuple，作为外层 tuple 的 1 个元素 |
| `case["tensors"]["outputs"]` 各 tensor 的 `dtype` | `output_dtypes` | 遍历 outputs dict 的 values。REQUIRED（`dict`）→ 取 `spec["dtype"]`。DYNAMIC（`list[dict]`）→ 取 `spec[0]["dtype"]`，包为压缩嵌套 `(("float16",),)` |
| — | `output_formats` | 默认 `('ND',)` × tensor 数量（REQUIRED 和 DYNAMIC 均保持扁平，TTK 通过 `get()` 回退处理） |
| `case["params"]` 标量属性 | `attributes` | 运行时加载 `S2P1_operator_model.json` 的 `attributes` 节获取属性名列表，从 `case["params"]` 中只保留属性名列表中存在的 key，其余全部排除，转为 dict |
| `case["tensors"]["inputs"]` 各 tensor 的 `_data_range` | `input_data_ranges` | 遍历 inputs。REQUIRED（`dict`）→ 取 `spec["_data_range"]` 按映射表转为 `(min, max)`，直接追加。DYNAMIC（`list[dict]`）→ 取 `spec[0]["_data_range"]`（约束保证所有子 tensor 一致），转为 `(min, max)`，为每个子 tensor 重复生成，组成展开嵌套 `((range, range, ...),)`（展开嵌套是唯一通用格式，详见 02-kernel-fields.md「TensorList 嵌套结构」） |
| 各输出 tensor 的 dtype（按输出顺序） | `precision_tolerances` | REQUIRED（`dict`）→ 按输出 tensor 顺序逐个 dtype 取精度标准组成外层 tuple（fp16/bf16 → `(0.001, 0.001)`，fp32 → `(0.0001, 0.0001)`）。DYNAMIC（`list[dict]`）→ 取 `spec[0]["dtype"]` 取精度标准，为每个子 tensor 重复生成，组成展开嵌套 `((tol, tol, ...),)` |
| — | `absolute_precision` | 默认 `1e-8` |
| — | `input_ori_shapes` | 默认回退到 `input_shapes`，留空即可 |
| — | `input_ori_formats` | 默认 `('ND',)`，留空即可 |
| — | `output_ori_shapes` | 默认回退到 `output_shapes`，留空即可 |
| — | `output_ori_formats` | 默认 `('ND',)`，留空即可 |
| aclnn 文档参数表「输入/输出」列 | `output_inplace_indexes` | 读取算子 `docs/aclnn*.md` 的 GetWorkspaceSize 参数表，检查每个输入参数的「输入/输出」列：标记为「输入/输出」的参数 → 记录其在输入列表中的 0-based 索引，所有 inplace 输入的索引组成 tuple。无 inplace 参数时默认 `()` |
| — | `output_shape_unknown_indexes` | 默认 `()` |
| — | `is_enabled` | 默认 `True` |
| `params["_group"]` + case ID 后缀 | `remark` | 从 case ID 提取 `_` 后的 data_range 后缀。有后缀时格式为 `group={group}, data_range={suffix}`，无后缀时为 `group={group}`。显式区分 group（tiling 路由模式）与 data_range（输入数据分布）两个维度 |
| — | `soc_series` | 默认留空 |
| — | `priority` | 默认 `0` |
| — | `dump_file_prefix` | 默认留空 |
| — | `manual_input_binaries` | 默认 `()` |
| — | `manual_golden_binaries` | 默认 `()` |

## 属性过滤规则

`case["params"]` 中包含 shape 参数、属性参数、const input 值和内部标记。写入 CSV `attributes` 列的 key 包括两类：
1. **参与代码路径的属性**：来自 `S5_mapping_spec.md` §属性 节中列出的属性名
2. **const input 的名称**：来自 `_def.cpp` 的 `.ValueDepend()` 标记，在 `S5_mapping_spec.md` §const input 节中列出

其余 key（shape 参数、内部标记、不参与路径的属性）一律排除。

脚本启动时从 `S5_mapping_spec.md` §属性 节（含 const input 子节）中提取实际列出的名称列表作为 `ATTR_NAMES`：

```python
# ATTR_NAMES = S5_mapping_spec.md §属性 节中列出的属性名 + const input 名集合
# 仅包含参与代码路径的属性和 const input
```

## const input tensor 提取

`_def.cpp` 中通过 `.ValueDepend()` 标记的输入 tensor 是 const input。const input 的值必须写入 CSV 的 `attributes` dict（而非 `input_data_ranges`），TTK 自动检测并从 `attributes` 中解析值。

**提取规则**：
1. 从 `S5_mapping_spec.md` §const input 节识别 const input 名称
2. 从 `case["params"]` 中提取 const input 的值（由 Step 5 case mapper 从维度参数构造写入）
3. 将值写入 `attributes` dict，key 为 const input 的名称
4. const input 的 shape/dtype 仍保留在 `input_shapes`/`input_dtypes` 中
5. const input 的 `input_data_ranges` 设为 `(None, None)`（不参与随机生成）
6. **不需要 `__input__` 插件**处理 const input

## output_inplace_indexes 检测

从算子 `docs/aclnn*.md` 的 GetWorkspaceSize 参数表（第一个参数表）检测 inplace 输入：

1. 定位参数表中「输入/输出」列
2. 遍历每个输入参数行，检查该列的值：
   - 值为「输入/输出」→ 该参数是 inplace 输入，记录其在**输入参数列表**中的 0-based 索引
   - 值为「输入」→ 普通输入，跳过
   - 值为「输出」→ 输出参数，跳过
3. 所有 inplace 索引按输入顺序组成 tuple，即为 `output_inplace_indexes`
4. 无 inplace 参数时默认 `()`

**示例**（foreach_add_list_inplace）：
- x1: 「输入/输出」→ 输入索引 0 → inplace
- x2: 「输入」→ 非 inplace
- alpha: 「输入」→ 非 inplace
- 结果：`output_inplace_indexes = (0,)`

**注意**：仅检查 GetWorkspaceSize 参数表（第一个表），不检查第二段接口的参数表。

## tensor 顺序

输入/输出 tensor 的顺序与 `case["tensors"]["inputs"]` / `case["tensors"]["outputs"]` 中的 key 迭代顺序一致。DYNAMIC（`list[dict]`）的子 tensor 按列表顺序排列在内层 tuple 中。

## 返回值结构

`extract_case_info(case, index)` 返回一个**扁平 dict**，key 对应 TTK CSV 列名。

```python
{
    "testcase_name":           "case00001",
    "network_name":            "UNKNOWN",
    "op_name":                 "example_op",
    "input_shapes":            ((64, 128), (64, 128), (128,)),
    "input_dtypes":            ("float16", "float16", "float16"),
    "input_formats":           ("ND", "ND", "ND"),
    "output_shapes":           ((64, 128), (64, 1), (64, 128)),
    "output_dtypes":           ("float16", "float32", "float16"),
    "output_formats":          ("ND", "ND", "ND"),
    "input_ori_shapes":        None,
    "input_ori_formats":       None,
    "output_ori_shapes":       None,
    "output_ori_formats":      None,
    "attributes":              {"epsilon": 1e-6},
    "input_data_ranges":       ((-2.5, 3.1), (-1.8, 4.7), (-0.3, 0.9)),
    "precision_tolerances":    ((0.001, 0.001), (0.0001, 0.0001), (0.001, 0.001)),
    "absolute_precision":      1e-8,
    "output_inplace_indexes":  (),
    "output_shape_unknown_indexes": (),
    "is_enabled":              True,
    "remark":                  "split_d",
    "soc_series":              None,
    "priority":                0,
    "dump_file_prefix":        None,
    "manual_input_binaries":   (),
    "manual_golden_binaries":  (),
}
```

DYNAMIC 示例（1 个 TensorList 输入含 3 子 tensor，1 个 TensorList 输出含 3 子 tensor）：

```python
{
    "testcase_name":           "case00001",
    "network_name":            "UNKNOWN",
    "op_name":                 "example_op",
    "input_shapes":            (((1, 2), (3, 4), (5, 6)),),       # 展开：每个子 tensor 一个 shape
    "input_dtypes":            (("float16",),),                   # 压缩：1 dtype 广播到 3 子 tensor
    "input_formats":           ("ND",),                           # 扁平，不变
    "output_shapes":           (((1, 2), (3, 4), (5, 6)),),
    "output_dtypes":           (("float16",),),
    "output_formats":          ("ND",),
    "input_ori_shapes":        None,
    "input_ori_formats":       None,
    "output_ori_shapes":       None,
    "output_ori_formats":      None,
    "attributes":              {},
    "input_data_ranges":       (((-10, 10), (-10, 10), (-10, 10)),),  # 展开：每个子 tensor 一个 range
    "precision_tolerances":    (((0.001, 0.001), (0.001, 0.001), (0.001, 0.001)),),  # 展开
    "absolute_precision":      1e-8,
    "output_inplace_indexes":  (0,),  # 假设 x1 为 inplace 输入（索引 0）
    "output_shape_unknown_indexes": (),
    "is_enabled":              True,
    "remark":                  "split_d",
    "soc_series":              None,
    "priority":                0,
    "dump_file_prefix":        None,
    "manual_input_binaries":   (),
    "manual_golden_binaries":  (),
}
```

**字节数计算**（用于校验时估算 tensor 内存）：
```python
DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2, "int8": 1, "int4": 0.5, "int32": 4}
```

## data_range → input_data_ranges 映射表

从每个输入 tensor 的 `_data_range`（`case["tensors"]["inputs"][name]["_data_range"]`）到 TTK `input_data_ranges` 的数值范围映射。按输入顺序逐个转换：

| `_data_range` | `input_data_ranges` 生成方式 | value_domain 约束 |
|---|---|---|
| `normal` | `_rng.uniform(-10, 10)` x2 排序 | positive → (0.01, 10)；non_negative → (0, 10)；non_zero → 排除 \|x\|<0.1；range → (min, max) |
| `zero` | `(0, 0)` | — |
| `extreme` | `(dtype_max * 0.9, dtype_max)` | — |
| `negative` | `_rng.uniform(-100, -0.01)` x2 排序 | range → 约束到 (min, min(0,max))，不满足则回退 normal |
| `tiny_pos` | `(1e-7, 1e-5)` | range → 约束到 (max(min,1e-7), min(max,1e-5))，不满足则回退 normal |
| `all_ones` | `(1, 1)` | — |
| `near_zero` | `(-0.01, 0.01)` | range → 约束到 (max(min,-0.01), min(max,0.01))，不满足则回退 normal |
| `with_inf` | `(1, float('inf'))` | — |
| `with_nan` | `(nan, nan)` | — |

rng 使用固定 seed `random.Random(42)` 保证可复现。

```python
import random

DTYPE_MAX = {"float16": 65504.0, "bfloat16": 3.3895e38, "float32": 3.4e38}
_rng = random.Random(42)

def _data_range_to_ttk(data_range, dtype_str, value_domain=None):
    def _range_normal():
        lo = value_domain.get("min") if value_domain else None
        hi = value_domain.get("max") if value_domain else None
        return (lo if lo is not None else -10.0, hi if hi is not None else 10.0)

    if data_range == "normal":
        if value_domain:
            t = value_domain["type"]
            if t == "positive":
                return (0.01, 10.0)
            elif t == "non_negative":
                return (0.0, 10.0)
            elif t == "non_zero":
                a, b = _rng.uniform(-10, 10), _rng.uniform(-10, 10)
                a, b = min(a, b), max(a, b)
                if abs(a) < 0.1: a = -10.0
                if abs(b) < 0.1: b = 10.0
                return (round(a, 2), round(b, 2))
            elif t == "range":
                return _range_normal()
        a, b = _rng.uniform(-10, 10), _rng.uniform(-10, 10)
        return (round(min(a, b), 2), round(max(a, b), 2))
    elif data_range == "zero":
        return (0, 0)
    elif data_range == "extreme":
        mx = DTYPE_MAX.get(dtype_str, 3.4e38)
        return (mx * 0.9, mx)
    elif data_range == "negative":
        if value_domain and value_domain["type"] == "range":
            lo = value_domain.get("min")
            hi = value_domain.get("max")
            eff_lo = lo if lo is not None else -100.0
            eff_hi = min(0, hi) if hi is not None else -0.01
            if eff_lo < eff_hi:
                return (eff_lo, eff_hi)
            return _range_normal()
        a, b = _rng.uniform(-100, -0.01), _rng.uniform(-100, -0.01)
        return (round(min(a, b), 2), round(max(a, b), 2))
    elif data_range == "tiny_pos":
        if value_domain and value_domain["type"] == "range":
            lo = value_domain.get("min")
            hi = value_domain.get("max")
            tp_lo = max(lo, 1e-7) if lo is not None else 1e-7
            tp_hi = min(hi, 1e-5) if hi is not None else 1e-5
            if tp_lo < tp_hi:
                return (tp_lo, tp_hi)
            return _range_normal()
        return (1e-7, 1e-5)
    elif data_range == "all_ones":
        return (1, 1)
    elif data_range == "near_zero":
        if value_domain and value_domain["type"] == "range":
            lo = value_domain.get("min")
            hi = value_domain.get("max")
            nz_lo = max(lo, -0.01) if lo is not None else -0.01
            nz_hi = min(hi, 0.01) if hi is not None else 0.01
            if nz_lo < nz_hi:
                return (nz_lo, nz_hi)
            return _range_normal()
        return (-0.01, 0.01)
    elif data_range == "with_inf":
        return (1, float('inf'))
    elif data_range == "with_nan":
        return (float('nan'), float('nan'))
    return (-2, 2)

def build_input_data_ranges(tensors):
    """遍历 inputs，按每个 tensor 的 _data_range + _value_domain 逐个映射。"""
    inputs = tensors.get("inputs", {})
    ranges = []
    for name, spec in inputs.items():
        if isinstance(spec, list):
            dr = spec[0].get("_data_range", "normal")
            dt = spec[0]["dtype"]
            vd = spec[0].get("_value_domain")
            r = _data_range_to_ttk(dr, dt, vd)
            ranges.append(tuple(r for _ in spec))
        else:
            dr = spec.get("_data_range", "normal")
            dt = spec["dtype"]
            vd = spec.get("_value_domain")
            ranges.append(_data_range_to_ttk(dr, dt, vd))
    return tuple(ranges)
```

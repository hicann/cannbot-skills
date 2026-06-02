# TTK Converter — 用例信息提取与 CSV 格式转换

## 目录

1. 输入输出与命名规则
2. CSV 字段映射
3. data_range 转换
4. CSV 校验
5. golden_plugin.py 生成
6. TTK kernel 执行

## 角色

信息提取与格式转换器。从 `S5_mapped_cases_low.json` / `S5_mapped_cases_high.json`（Step 5 产出的已映射 tensor 配置）中直接提取字段，转换为 TTK CSV 格式。

**TTK 工具**：`ops-test-kit/` 目录是 TTK 调试工具的代码仓库，提供 kernel/aclnn/e2e 三种模式的编译、执行、精度比对能力。所有 `python3 -m ttk` 命令**必须在 `ops-test-kit/` 目录下执行**（TTK 通过 `__main__.py` 启动）。

**目录要求**：本文件中的 `python3 -m ttk ...` 命令均以 `ops-test-kit/` 为当前工作目录执行；生成的 `ttk_*.csv` 和 `golden_plugin.py` 位于算子目录的 `tests/whitebox/` 下，命令参数必须使用相对或绝对路径指向这些产物。

**命名规则**：输出文件带 `ttk_` 前缀（`ttk_extract_case_info.py`、`ttk_{op_name}_cases_low.csv`、`ttk_{op_name}_cases_full.csv`），例外：`golden_plugin.py` 为固定文件名。

## 模式支持

TTK 工具支持三种模式，基于 CSV 表头自动识别：

| 模式 | 识别条件 | 使用结构 | 当前支持 |
|------|---------|---------|---------|
| Kernel | 表头不含 `api_name` | `UniversalTestcaseStructure` | **已实现** |
| ACLNN | 表头含 `api_name` 且值以 `aclnn` 开头 | `ApiTestcaseStructure` | 预留 |
| E2E | 表头含 `api_name` 且值不以 `aclnn` 开头 | `FrameworkApiTestcaseStructure` | 预留 |

## 输入

1. `S5_mapped_cases_low.json` — Step 5 产出的 low 档位已映射参数组合（TTK low CSV 输入）
2. `S5_mapped_cases_high.json` — Step 5 产出的 high 档位已映射参数组合（TTK full CSV 输入）
3. `S2P1_operator_model.json` — 算子模型（提取 `attributes` 节中的属性名列表，用于过滤 `attributes`）
4. `op_name` — 算子名称（小写字母+下划线，由主 Agent 在调用本 prompt 时提供）
5. 算子源码（校验用）：
   - `*_def.cpp` — 输入/输出/属性注册定义
   - `*_tiling_check.cpp` 或 `*_tiling*.cpp` — 约束检查逻辑
   - `*_infershape.cpp` — 输出 shape 推导逻辑

## 输出

- `ttk_extract_case_info.py` — 单用例信息提取脚本（直接从 `case["tensors"]` / `case["params"]` 提取，无 torch 依赖）
- `ttk_{op_name}_cases_low.csv` — low 数据源 CSV（由 `S5_mapped_cases_low.json` 转换）
- `ttk_{op_name}_cases_full.csv` — full 数据源 CSV（由 `S5_mapped_cases_high.json` 转换）
- `golden_plugin.py` — TTK 自定义 golden 函数（通过 `--plugin` 加载）

---

# Kernel 模式 CSV（26 个字段）

适用于 `python3 -m ttk kernel` 命令，使用 `UniversalTestcaseStructure`。

静态 shape 通过 `input_shapes` / `output_shapes` 直接指定，动态 shape 由框架自动推导（将正数维度替换为 `-1`）。

## 公共字段（9 个，所有模式通用）

| 序号 | 列名 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| 1 | `testcase_name` | STRING | 是 | 自动生成 | 用例唯一名称。缺失时自动生成为 `auto_testcase_name_N`。 |
| 2 | `network_name` | STRING | 否 | `None` | 网络/模型名称标签（如 `model_name_train`）。 |
| 3 | `input_data_ranges` | FLOAT_RANGE_NESTED | 否 | `((None, None),)` | 每个输入张量的随机数据范围。每个元素为 `(min, max)`。 |
| 4 | `precision_tolerances` | FLOAT_RANGE_NESTED | 否 | `None` | 每个输出的精度容差对 `(rtol, atol)`。如 `"((0.001, 0.001),)"` |
| 5 | `absolute_precision` | FLOAT_OR_NESTED | 否 | `1e-8` | 默认绝对精度容差。可以是单个浮点数或嵌套容器实现逐输出控制。 |
| 6 | `is_enabled` | BOOL | 否 | `True` | 设为 `False` 跳过此用例。 |
| 7 | `remark` | STRING | 否 | `None` | 自由备注信息。 |
| 8 | `soc_series` | STRING_TUPLE | 否 | `None` | SoC 过滤。前缀 `-` 表示排除。如 `('Ascend910A', '-Ascend310P')` |
| 9 | `priority` | INT | 否 | `0` | 优先级，用于选择性执行。 |

## Kernel 专有字段（17 个）

| 序号 | 列名 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|------|--------|------|
| 10 | `op_name` | STRING | 是 | *(无)* | 算子名称。如 `add`、`mat_mul_v3`。 |
| 11 | `input_shapes` | SHAPE_NESTED | 是 | `()` | 输入张量 shape。支持 TensorList 嵌套。用 `None` 表示可选输入。 |
| 12 | `input_dtypes` | DTYPE_NESTED | 是 | `()` | 输入数据类型。支持 TensorList 嵌套。 |
| 13 | `output_shapes` | SHAPE_INFER_NESTED | 是 | `None` | 输出张量 shape。支持 TensorList 嵌套和自动推断关键字。 |
| 14 | `output_dtypes` | DTYPE_NESTED | 是 | *(无)* | 输出数据类型。支持 TensorList 嵌套。 |
| 15 | `input_formats` | DTYPE_NESTED | 否 | `('ND',)` | 输入张量格式。 |
| 16 | `input_ori_shapes` | SHAPE_NESTED | 否 | → `input_shapes` | 原始输入 shape（格式转换前）。 |
| 17 | `input_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | 原始输入格式。 |
| 18 | `output_formats` | DTYPE_NESTED | 否 | `('ND',)` | 输出张量格式。 |
| 19 | `output_ori_shapes` | SHAPE_INFER_NESTED | 否 | `None` | 原始输出 shape。 |
| 20 | `output_ori_formats` | DTYPE_NESTED | 否 | `('ND',)` | 原始输出格式。 |
| 21 | `attributes` | DICT | 否 | `{}` | 算子属性（编译期和运行期合并）。 |
| 22 | `output_inplace_indexes` | INT_TUPLE | 否 | `()` | 与输入原地操作的输出索引。 |
| 23 | `output_shape_unknown_indexes` | INT_TUPLE | 否 | `()` | 编译期 shape 未知的输出索引。 |
| 24 | `dump_file_prefix` | STRING | 否 | `None` | 数据 dump 文件的自定义文件名前缀。 |
| 25 | `manual_input_binaries` | EVAL | 否 | `()` | 手动输入二进制文件路径。 |
| 26 | `manual_golden_binaries` | EVAL | 否 | `()` | 手动 Golden 输出二进制文件路径。 |

## CSV 列顺序（严格固定）

```
testcase_name, network_name, op_name, input_shapes, input_dtypes, input_formats, output_shapes, output_dtypes, output_formats, input_ori_shapes, input_ori_formats, output_ori_shapes, output_ori_formats, attributes, input_data_ranges, precision_tolerances, absolute_precision, output_inplace_indexes, output_shape_unknown_indexes, is_enabled, remark, soc_series, priority, dump_file_prefix, manual_input_binaries, manual_golden_binaries
```

---

## 字段提取规则

### S5 JSON → CSV 列映射

| S5 JSON 字段 | CSV 列 | 提取方式 |
|-------------|--------|---------|
| case["id"] | `testcase_name` | 直接使用 `case["id"]` |
| — | `network_name` | 从 `case["params"].get("_network", None)` 提取，无则留空。禁止从 `_source`、`_reason` 等元信息字段取值 |
| `case["tensors"]["inputs"]` 各 tensor 的 `shape` | `input_shapes` | 遍历 inputs dict 的 values，取 `spec["shape"]`，转为 tuple。可选输入 `spec is None` 时用 `None` 占位 |
| `case["tensors"]["inputs"]` 各 tensor 的 `dtype` | `input_dtypes` | 遍历 inputs dict 的 values，取 `spec["dtype"]`。可选输入 `spec is None` 时用 `None` 占位 |
| — | `input_formats` | 默认 `('ND',)` × tensor 数量 |
| `case["tensors"]["outputs"]` 各 tensor 的 `shape` | `output_shapes` | 遍历 outputs dict 的 values，取 `spec["shape"]`，转为 tuple |
| `case["tensors"]["outputs"]` 各 tensor 的 `dtype` | `output_dtypes` | 遍历 outputs dict 的 values，取 `spec["dtype"]` |
| — | `output_formats` | 默认 `('ND',)` × tensor 数量 |
| `case["params"]` 标量属性 | `attributes` | 运行时加载 `S2P1_operator_model.json` 的 `attributes` 节获取属性名列表，从 `case["params"]` 中只保留属性名列表中存在的 key，其余全部排除，转为 dict |
| `case["tensors"]["inputs"]` 各 tensor 的 `_data_range` | `input_data_ranges` | 遍历 inputs，取每个 `spec["_data_range"]` 按映射表转为数值范围。生成规则见下方 data_range 映射表 |
| 各输出 tensor 的 dtype（按输出顺序） | `precision_tolerances` | 按输出 tensor 顺序，逐个 dtype 取精度标准组成外层 tuple。fp16/bf16 → `(0.001, 0.001)`，fp32 → `(0.0001, 0.0001)` |
| — | `absolute_precision` | 默认 `1e-8` |
| — | `input_ori_shapes` | 默认回退到 `input_shapes`，留空即可 |
| — | `input_ori_formats` | 默认 `('ND',)`，留空即可 |
| — | `output_ori_shapes` | 默认回退到 `output_shapes`，留空即可 |
| — | `output_ori_formats` | 默认 `('ND',)`，留空即可 |
| — | `output_inplace_indexes` | 默认 `()` |
| — | `output_shape_unknown_indexes` | 默认 `()` |
| — | `is_enabled` | 默认 `True` |
| `params["_group"]` + case ID 后缀 | `remark` | 从 case ID 提取 `_` 后的 data_range 后缀。有后缀时格式为 `group={group}, data_range={suffix}`，无后缀时为 `group={group}`。显式区分 group（tiling 路由模式）与 data_range（输入数据分布）两个维度 |
| — | `soc_series` | 默认留空 |
| — | `priority` | 默认 `0` |
| — | `dump_file_prefix` | 默认留空 |
| — | `manual_input_binaries` | 默认 `()` |
| — | `manual_golden_binaries` | 默认 `()` |

### 属性过滤规则

`case["params"]` 中包含 shape 参数、dtype 参数、mode 参数和内部标记。只有 `S2P1_operator_model.json` 的 `attributes` 节中声明的属性名（如 `epsilon`）才写入 `attributes`，其余一律排除。脚本启动时加载 operator_model 提取属性名列表：

```python
import json
with open("S2P1_operator_model.json") as _f:
    _model = json.load(_f)
ATTR_NAMES = {a["name"] for a in _model.get("attributes", [])}
```

### tensor 顺序

输入/输出 tensor 的顺序与 `case["tensors"]["inputs"]` / `case["tensors"]["outputs"]` 中的 key 迭代顺序一致。

### 返回值结构

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

**字节数计算**（用于校验时估算 tensor 内存）：
```python
DTYPE_BYTES = {"float16": 2, "float32": 4, "bfloat16": 2, "int8": 1, "int4": 0.5, "int32": 4}
```

### data_range → input_data_ranges 映射表

从每个输入 tensor 的 `_data_range`（`case["tensors"]["inputs"][name]["_data_range"]`）到 TTK `input_data_ranges` 的数值范围映射。按输入顺序逐个转换：

| `_data_range` | `input_data_ranges` 生成方式 | 说明 |
|---|---|---|
| `normal` | `_rng.uniform(-10, 10), _rng.uniform(-10, 10)` 排序取 `(min, max)` | 随机值域，每次不同 |
| `zero` | `(0, 0)` | 固定 |
| `extreme` | `(dtype_max * 0.9, dtype_max)` | dtype_max 查表（见下方） |
| `negative` | `_rng.uniform(-100, -0.01), _rng.uniform(-100, -0.01)` 排序取 `(min, max)` | 随机负值域，每次不同 |
| `tiny_pos` | `(1e-7, 1e-5)` | 固定 |
| `all_ones` | `(1, 1)` | 固定 |
| `near_zero` | `(-0.01, 0.01)` | 固定 |
| `with_inf` | `(1, float('inf'))` | 固定（TTK 的 `_digitize_inf_nan` 自动处理） |
| `with_nan` | `(float('nan'), float('nan'))` | 固定（TTK 的 `_digitize_inf_nan` 自动处理） |

rng 使用固定 seed `random.Random(42)` 保证可复现。

```python
import random

DTYPE_MAX = {"float16": 65504.0, "bfloat16": 3.3895e38, "float32": 3.4e38}
_rng = random.Random(42)

def _data_range_to_ttk(data_range, dtype_str):
    if data_range == "normal":
        a, b = _rng.uniform(-10, 10), _rng.uniform(-10, 10)
        return (round(min(a, b), 2), round(max(a, b), 2))
    elif data_range == "zero":
        return (0, 0)
    elif data_range == "extreme":
        mx = DTYPE_MAX.get(dtype_str, 3.4e38)
        return (mx * 0.9, mx)
    elif data_range == "negative":
        a, b = _rng.uniform(-100, -0.01), _rng.uniform(-100, -0.01)
        return (round(min(a, b), 2), round(max(a, b), 2))
    elif data_range == "tiny_pos":
        return (1e-7, 1e-5)
    elif data_range == "all_ones":
        return (1, 1)
    elif data_range == "near_zero":
        return (-0.01, 0.01)
    elif data_range == "with_inf":
        return (1, float('inf'))
    elif data_range == "with_nan":
        return (float('nan'), float('nan'))
    return (-2, 2)

def build_input_data_ranges(tensors):
    """遍历 inputs，按每个 tensor 的 _data_range 逐个映射。"""
    inputs = tensors.get("inputs", {})
    ranges = []
    for name, spec in inputs.items():
        dr = spec.get("_data_range", "normal")
        dt = spec["dtype"]
        ranges.append(_data_range_to_ttk(dr, dt))
    return tuple(ranges)
```

---

## 任务

### 任务 1：生成 ttk_extract_case_info.py

从 `S5_mapped_cases_high.json` 的每个 case 中直接提取字段，无需 torch 依赖。

#### 初步验证

```bash
python ttk_extract_case_info.py S5_mapped_cases_high.json
```

打印 case[0] 的完整信息，对比 S5_mapped_cases_high.json 中的第一个 case 确认字段映射正确。

### 任务 2：校验 ttk_extract_case_info.py（强制，不可跳过）

**目的**：S7 mapper 本身可能有映射错误，直接取值会继承这些错误。必须通过对比算子注册和约束检查的权威源进行二次校验。

**权威源**（按优先级排序）：
1. `*_def.cpp` — 输入/输出名称、顺序、dtype 注册定义
2. `*_tiling_check.cpp` 或 `*_tiling*.cpp` — 约束检查逻辑
3. `*_infershape.cpp` — 输出 shape 推导逻辑

**校验维度**：

| 维度 | 权威源 | 校验内容 |
|------|--------|---------|
| 输入名称/顺序 | `_def.cpp` Input 注册 | 所有输入的名称和注册顺序与 `_def.cpp` 完全一致 |
| 输出名称/顺序 | `_def.cpp` Output 注册 | 所有输出的名称和注册顺序与 `_def.cpp` 完全一致 |
| 属性名称/类型 | `_def.cpp` Attr 注册 | 属性名称、AttrType、默认值与 `attributes` 中的 key-value 一致 |
| dtype 推导 | `_def.cpp` DataTypeFormat 配置 | 每种输入 dtype 组合下的 dtype 与对应列一致 |
| shape 约束 | `_tiling_check.cpp` | 所有 if/OP_CHECK_IF 条件在脚本中正确反映 |
| 输出 shape | `_infershape.cpp` | 输出 shape 计算与 infershape 逻辑一致 |

**发现 bug 时的处理**：记录每个 bug → 修复 → 重新运行 → 重新校验。

### 任务 3：生成 CSV 文件

需要生成两个 CSV 文件：

**3a：low 档位**

```bash
python ttk_extract_case_info.py S5_mapped_cases_low.json --csv ttk_{op_name}_cases_low.csv
```

**3b：high 档位**

```bash
python ttk_extract_case_info.py S5_mapped_cases_high.json --csv ttk_{op_name}_cases_full.csv
```

1. 确认任务 1 已完成
2. 遍历 `S5_mapped_cases*.json["cases"]`，对每个 case 调用 `extract_case_info(case, index)` 获取扁平 dict
3. 按 CSV 列顺序映射字段
4. 写入 CSV 文件（UTF-8，逗号分隔，双引号包裹含特殊字符的字段）
5. 抽查第 0 条和最后一条 case

### 任务 4：CSV 格式校验

```bash
python scripts/ttk_validate_csv.py ttk_{op_name}_cases_low.csv
```

校验项（9 项）：

| 序号 | 校验项 | 说明 |
|------|--------|------|
| 1 | 编码 | UTF-8（不带 BOM） |
| 2 | 表头 | 26 个列名与规格严格一致，顺序不得打乱 |
| 3 | 行数 | CSV 数据行数 > 0 |
| 4 | testcase_name | 唯一、支持带 data_range 后缀的 ID 格式 `(case|network)\d+_\w+` |
| 5 | op_name | 非空、小写字母+下划线格式 |
| 6 | 必填项 | `testcase_name`、`op_name`、`input_shapes`、`input_dtypes`、`output_dtypes`、`output_shapes` 非空 |
| 7 | precision_tolerances | 为空或 `((a,b))` 格式 |
| 8 | tuple 长度一致性 | `input_dtypes`、`output_dtypes` 等字段长度与 `input_shapes` 一致 |
| 9 | 模式识别 | 表头不含 `api_name` → Kernel 模式 |

结果处理：全部 PASS → 完成；存在 FAIL → 修复后重新生成并校验。

### 任务 5：生成 `golden_plugin.py`

> **前置条件**：任务 4 校验全部 PASS。

**目的**：为 TTK kernel 模式提供自定义 golden 函数，通过 `--plugin` 参数加载。

#### TTK Plugin 加载机制

TTK 的 `PluginScanner` 通过 AST 解析扫描 `--plugin` 指定的 `.py` 文件，查找模块级变量 `__golden__`（`ast.Dict` 类型），按 `{level: {op_name: func_name}}` 格式匹配 golden 函数。**不执行脚本，仅静态解析**；运行时通过 `importlib` 动态加载。

优先级：自定义 plugin > 内置 registry（`golden_funcs` dict）。

#### 输出文件

- `golden_plugin.py` — 固定文件名，存放在 `tests/whitebox/` 目录

#### 函数签名规范

自定义 golden 函数通过 `__call_custom_golden_func` 调用：

```python
def __golden_{op_name}(*input_arrays, **kwargs):
    # input_arrays: 按 input_shapes 顺序展开的 numpy 数组（位置参数）
    # kwargs: context.attributes + 框架额外信息
    return [output_0, output_1, ...]  # list, 元素顺序与 output_dtypes 一致
```

**参数说明**：

| 参数 | 来源 | 说明 |
|------|------|------|
| `*input_arrays` | `context.input_arrays` 解包 | 按 CSV `input_shapes` 顺序，每个为 numpy 数组 |
| `**kwargs` | `context.attributes` + 额外信息 | 属性名（如 `epsilon`）、`input_dtypes`、`output_dtypes`、`full_soc_version` 等 |

**返回值规范**：

- 类型：建议 `list`，`tuple` 也可被 `__golden_flatten` 正确处理
- 元素：numpy 数组，顺序与 CSV `output_dtypes` 严格一致
- 形状和 dtype 必须与 CSV 中 `output_shapes` 和 `output_dtypes` 匹配

#### 编写依据

算子 `docs/aclnn*.md` 文档中的「计算公式」节（只读该节，与 S6 reference 实现同源）。若无法在算子目录或主 Agent 提供的文档路径中定位 aclnn API 文档，停止生成 `golden_plugin.py` 并报告缺失文档；禁止从 kernel 源码或经验推导 golden 公式。

#### 注意事项（实测踩坑总结）

| # | 注意事项 | 说明 |
|---|---------|------|
| 1 | numpy reduce 函数的 `axis` 参数**不接受 `list`**，必须用 `int` 或 `tuple` | `np.mean`、`np.sum`、`np.sqrt` 等函数的 `axis=list(range(...))` 会抛 `TypeError`，必须用 `tuple(range(...))` |
| 2 | 返回值建议使用 `list`，`tuple` 也可被正确处理 | TTK 的 `__golden_flatten` 对任意 `Sequence` 执行 `deep_flatten`，`list` 为惯例写法 |
| 3 | 中间计算用 `float32`，最终输出按 `output_dtypes` 转回 | FP16 输入时 numpy 不会自动提升精度，需显式 `.astype(np.float32)` |
| 4 | 属性值从 `kwargs` 获取，必须设默认值 | 如 `kwargs.get("epsilon", 1e-6)`，防止 CSV 中 `attributes` 为空时 KeyError |
| 5 | 输出-公式映射必须明确 | 多输出算子必须确认每个输出对应公式的哪个表达式，避免语义错位（参考 S6-pytest-generator.md 第 5 条） |
| 6 | 多输出时返回的 list 长度必须与 `output_dtypes` 长度一致 | 否则 TTK `deep_flatten` 后与输出张量数量不匹配 |

#### `__golden__` 声明格式

```python
__golden__ = {
    "kernel": {
        "{op_name}": "__golden_{op_name}"
    }
}
```

- 变量名必须是 `__golden__`
- key `"kernel"` 对应 kernel 模式
- value 中的 key 是 CSV 中的 `op_name`
- value 中的 value 是同文件中的函数名字符串

#### 初步验证

```bash
python3 -c "
import sys; sys.path.insert(0, '{ops_test_kit_path}')
from ttk.core_modules.plugin_loader import get_plugin_function
func, source = get_plugin_function('{op_name}', 'golden', 'kernel', '{whitebox_dir}/golden_plugin.py')
assert func is not None, 'golden not found'
import numpy as np
np.random.seed(42)
# 按算子实际输入构造测试数据并调用 func，验证返回 list 长度和 dtype
print('PASS')
"
```

### 任务 6：TTK Kernel 执行验收

> **前置条件**：任务 5 初步验证通过。

**目的**：在 TTK kernel 模式下验证 golden 函数与算子输出的精度比对结果。

#### 6a：单用例探测

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_low.csv \
  --plugin {whitebox_dir}/golden_plugin.py \
  -t {random_testcase_name} \
  --pc 1 \
  --seed 42
```

- 命令固定，仅 `-i`/`--plugin`/`-t` 根据算子变化
- `-t` 指定**随机挑选**的一个用例（避开 case00000，优先选中间位置如 case00042），防止首个用例恰巧正常而掩盖 CSV 格式问题
- 不传 `-b`/`-d`/`-c`，TTK 根据用例自行选择编译模式

#### 验收检查项

| # | 检查项 | 通过标准 | 日志关键字 |
|---|--------|---------|-----------|
| 1 | Custom golden 加载 | 日志出现加载成功 | `Loaded custom golden: kernel.{op_name}` |
| 2 | 编译成功 | 编译通过 | `Compilation Result: SUCC` |
| 3 | Golden/Output 一致 | Shape 和 Dtype 匹配 | `Golden Shape` == `Output Shape` |
| 4 | 精度比对 | DYN_GOLD 结果输出 | `DYN_GOLD:` 后有数值百分比 |

**失败处理**：6a 任一项失败 → 分析错误信息 → 修复 `golden_plugin.py` 或 `ttk_extract_case_info.py` → 重新生成两个 CSV → 回到 6a。

#### 6b：批量执行

> **前置条件**：6a 全部通过。

```bash
cd {ops_test_kit_path} && python3 -m ttk kernel \
  -i {whitebox_dir}/ttk_{op_name}_cases_full.csv \
  -o {whitebox_dir}/ttk_{op_name}_cases_full_result.csv \
  --plugin {whitebox_dir}/golden_plugin.py \
  --pc 1 \
  --seed 42
```

- `--pc 1`：单进程执行，避免多进程干扰日志
- `--seed 42`：固定随机种子，结果可复现
- `-o`：指定结果 CSV 路径

#### 验收检查项

| # | 检查项 | 通过标准 | 检查方法 |
|---|--------|---------|---------|
| 1 | perf_status 通过率 | ≥ 90% 用例 PASS | 统计 `_result.csv` 中 `perf_status` 列 |
| 2 | 编译成功率 | 全部 SUCC | 日志无 `Compilation Result: FAIL` |
| 3 | 无内存越界 | 全部 `memory_oob_status` 为空或 PASS | `_result.csv` 检查 |

**失败处理**：6b 任一项失败 → 分析失败用例日志 → 修复 `golden_plugin.py` 或用例数据 → 重新生成 CSV → 回到 6a。

**结果 CSV**（`_result.csv`）包含每用例的：

| 列 | 含义 |
|----|------|
| `perf_status` | 执行通过/失败 |
| `precision_status` | 精度通过/失败 |
| `dyn_precision` | 各输出精度百分比 |
| `memory_oob_status` | 内存越界检测 |
| `dyn_tiling_key` | 命中的 tiling key |

---

## CSV 填写规则

1. **禁止无故设 None**：未明确要求为空的字段，使用列规格默认值
2. **字符串带单引号**：tuple 内 dtype/format 等字符串用 `'float16'`、`'ND'`；顶层字段（testcase_name、op_name）不加引号
3. **双引号包裹特殊字段**：含括号、逗号的字段必须用双引号包裹（CSV 标准）
4. **dict key 双引号**：`{"epsilon": 1e-05}`
5. **禁止 repr()**：字符串用 `str()`，单引号在构造 tuple 时嵌入
6. **单元素 tuple 尾逗号**：1 维 shape 如 `(12289,)` 必须带尾逗号
7. **precision_tolerances 尾逗号兼容**：校验脚本需同时接受 `((a, b))` 和 `((a, b),)` 两种格式
8. **input_shapes / output_shapes**：只填静态值，动态 shape 由框架自动推导
9. **ori 字段**：默认留空，TTK 框架自动回退

---

## ACLNN 模式扩展（预留）

> TODO: 待实现。ACLNN 模式共 24 个字段（9 公共 + 15 ACLNN 专有）。

### ACLNN 专有字段

| 列名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `api_name` | STRING | 是 | ACLNN API 名称，如 `aclnnAdd` |
| `tensor_view_shapes` | SHAPE_NESTED | 是 | 张量视图 shape |
| `tensor_dtypes` | DTYPE_NESTED | 是 | 张量数据类型 |
| `tensor_formats` | DTYPE_NESTED | 否 | 张量格式 |
| `tensor_storage_shapes` | SHAPE_NESTED | 否 | 非连续张量的存储 shape |
| `tensor_view_offsets` | INT_NESTED | 否 | 视图偏移量 |
| `tensor_view_strides` | SHAPE_NESTED | 否 | 视图步长 |
| `output_tensor_indexes` | INT_TUPLE | 是 | 输出张量索引 |
| `output_inplace_indexes` | INT_TUPLE | 否 | 原地输出索引 |
| `attributes` | DICT | 否 | API 属性参数 |
| `scalar_dtypes` | DTYPE_NESTED | 否 | 标量参数数据类型 |
| `scalar_data_ranges` | FLOAT_RANGE_NESTED | 否 | 标量数据范围 |
| `dump_file_prefix` | STRING | 否 | dump 文件前缀 |
| `manual_tensor_binaries` | EVAL | 否 | 手动张量二进制 |
| `manual_golden_binaries` | STRING_TUPLE | 否 | 手动 Golden 二进制 |

## E2E 模式扩展（预留）

> TODO: 待实现。E2E 模式共 19 个字段（9 公共 + 10 E2E 专有）。

### E2E 专有字段

| 列名 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `api_name` | STRING | 是 | 框架 API 路径，如 `torch.add` |
| `tensor_view_shapes` | SHAPE_NESTED | 是 | 输入张量 shape |
| `tensor_dtypes` | DTYPE_NESTED | 是 | 输入数据类型 |
| `tensor_formats` | DTYPE_NESTED | 否 | 张量格式 |
| `tensor_storage_shapes` | SHAPE_NESTED | 否 | 非连续存储 shape |
| `tensor_view_offsets` | INT_NESTED | 否 | 视图偏移量 |
| `tensor_view_strides` | STRIDE | 否 | 视图步长 |
| `output_tensor_indexes` | INT_TUPLE | 否 | 输出张量索引 |
| `attributes` | DICT | 否 | 框架 API 关键字参数 |
| `golden_api` | STRING | 否 | 替代 Golden 计算的 API |

---

## 严格禁止

1. 禁止不从 S5 JSON 取值而自行推导 shape/dtype
2. 禁止 ttk_extract_case_info.py 依赖 torch
3. 禁止修改固定列名和列顺序
4. 禁止跳过验证步骤
5. 禁止信任 S5 映射为绝对正确——必须通过任务 2 校验
6. 禁止在 golden_plugin.py 中使用 `list` 作为 numpy reduce 函数（mean/sum/max/min 等）的 `axis` 参数
7. 禁止 golden_plugin.py 返回值长度与 output_dtypes 长度不一致

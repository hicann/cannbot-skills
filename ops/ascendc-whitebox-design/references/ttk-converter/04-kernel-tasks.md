# Kernel 模式任务流程

## 任务 1：生成 ttk_extract_case_info.py

从 `S5_mapped_cases_low.json` 的每个 case 中直接提取字段，无需 torch 依赖。

### 初步验证

```bash
python ttk_extract_case_info.py S5_mapped_cases_low.json
```

打印 case[0] 的完整信息，对比 `S5_mapped_cases_low.json` 中的第一个 case 确认字段映射正确。

## 任务 2：校验 ttk_extract_case_info.py（强制，不可跳过）

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
| TensorList 结构 | `_def.cpp` Input/Output 注册的 `.ParamType()` | S5 JSON 中为 `list[dict]`（DYNAMIC）的 input/output，在 `_def.cpp` 中必须注册为 `.ParamType(DYNAMIC)`；为 `dict`（REQUIRED）的必须为 `.ParamType(REQUIRED)` 或缺省 |
| output_inplace_indexes | `docs/aclnn*.md` 参数表「输入/输出」列 | 标记为「输入/输出」的输入参数索引与 CSV 中 `output_inplace_indexes` 的值一致 |

**发现 bug 时的处理**：记录每个 bug → 修复 → 重新运行 → 重新校验。

## 任务 3：生成 CSV 文件

需要生成两个 CSV 文件：

**3a：low 档位**

```bash
python ttk_extract_case_info.py S5_mapped_cases_low.json --csv ttk_{op_name}_cases_low.csv
```

**3b：full 档位**

```bash
python ttk_extract_case_info.py S5_mapped_cases_high.json --csv ttk_{op_name}_cases_full.csv
```

1. 确认任务 1 已完成
2. 遍历 `S5_mapped_cases*.json["cases"]`，对每个 case 调用 `extract_case_info(case, index)` 获取扁平 dict
3. 按 CSV 列顺序映射字段
4. 写入 CSV 文件（UTF-8，逗号分隔，双引号包裹含特殊字符的字段）
5. 抽查第 0 条和最后一条 case

## 任务 4：CSV 格式校验

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

## 任务 5：生成 `golden_plugin.py`

> **前置条件**：任务 4 校验全部 PASS。

**目的**：为 TTK kernel 模式提供自定义 golden 函数，通过 `--plugin` 参数加载。

### TTK Plugin 加载机制

TTK 的 `PluginScanner` 通过 AST 解析扫描 `--plugin` 指定的 `.py` 文件，查找模块级变量 `__golden__`（`ast.Dict` 类型），按 `{level: {op_name: func_name}}` 格式匹配 golden 函数。**不执行脚本，仅静态解析**；运行时通过 `importlib` 动态加载。

优先级：自定义 plugin > 内置 registry（`golden_funcs` dict）。

### 输出文件

- `golden_plugin.py` — 固定文件名，存放在 `tests/whitebox/` 目录（仅自生成时创建，已有 `tests/assets/golden.py` 时不生成）

### 函数签名规范

自定义 golden 函数通过 `__call_custom_golden_func` 调用：

```python
def __golden_{op_name}(*input_arrays, **kwargs):
    # input_arrays: 按 input_shapes 顺序展开的 numpy 数组（位置参数）
    # kwargs: context.attributes + 框架额外信息
    return [output_0, output_1, ...]  # list, 元素顺序与 output_dtypes 一致
```

DYNAMIC 示例（1 个 TensorList 输入，1 个 TensorList 输出）：

```python
def __golden_{op_name}(x_list, **kwargs):
    # x_list: list[np.ndarray]（TensorList 输入，TTK 通过 input_apply_as_list 折叠后传入）
    return [reference_fn(t) for t in x_list]  # list[np.ndarray]，每个子 tensor 1 个输出
```

**参数说明**：

| 参数 | 来源 | 说明 |
|------|------|------|
| `*input_arrays` | `context.input_arrays` 解包 | 按 CSV `input_shapes` 顺序，每个为 numpy 数组。REQUIRED 输入 → 单个 `np.ndarray`；DYNAMIC 输入 → `list[np.ndarray]`（TTK 通过 `input_apply_as_list` 折叠为嵌套结构后解包传入） |
| `**kwargs` | `context.attributes` + 额外信息 | 属性名（如 `epsilon`）、`input_dtypes`、`output_dtypes`、`full_soc_version` 等 |

**返回值规范**：

- 类型：建议 `list`，`tuple` 也可被 `__golden_flatten` 正确处理
- 元素：numpy 数组，顺序与 CSV `output_dtypes` 严格一致
- 形状和 dtype 必须与 CSV 中 `output_shapes` 和 `output_dtypes` 匹配

### 已有 Golden 复用（优先）

生成 `golden_plugin.py` 前，检查算子目录下是否已有 TTK golden 实现：

```
{算子路径}/tests/assets/golden.py
```

**检查规则**：

1. 文件存在
2. 包含模块级 `__golden__` 变量（`ast.Dict` 类型，含 `"kernel"` key）
3. `__golden__["kernel"]` 中存在与当前 `op_name` 匹配的条目

**三条全部满足** → `--plugin` 直接指向 `{算子路径}/tests/assets/golden.py`，**不生成 `golden_plugin.py`**。理由：已有 golden 经 TTK 验证，实现通常更完善（含芯片分支、bfloat16 特殊处理等）。

**任一不满足** → 走下方「编写依据」自生成 `{whitebox_dir}/golden_plugin.py`。

后续所有 TTK 命令中的 `--plugin` 参数统一使用变量 `{plugin_path}`：
- 已有 golden 时：`{plugin_path}` = `{算子路径}/tests/assets/golden.py`
- 自生成时：`{plugin_path}` = `{whitebox_dir}/golden_plugin.py`

### 编写依据

算子 `docs/aclnn*.md` 文档中的「计算公式」节（只读该节，与 S6 reference 实现同源）。

### 注意事项（实测踩坑总结）

| # | 注意事项 | 说明 |
|---|---------|------|
| 1 | numpy reduce 函数的 `axis` 参数**不接受 `list`**，必须用 `int` 或 `tuple` | `np.mean`、`np.sum`、`np.sqrt` 等函数的 `axis=list(range(...))` 会抛 `TypeError`，必须用 `tuple(range(...))` |
| 2 | 返回值建议使用 `list`，`tuple` 也可被正确处理 | TTK 的 `__golden_flatten` 对任意 `Sequence` 执行 `deep_flatten`，`list` 为惯例写法 |
| 3 | 中间计算用 `float32`，最终输出按 `output_dtypes` 转回 | FP16 输入时 numpy 不会自动提升精度，需显式 `.astype(np.float32)` |
| 4 | 属性值从 `kwargs` 获取，必须设默认值 | 如 `kwargs.get("epsilon", 1e-6)`，防止 CSV 中 `attributes` 为空时 KeyError |
| 5 | 输出-公式映射必须明确 | 多输出算子必须确认每个输出对应公式的哪个表达式，避免语义错位（参考 `references/pytest-gen/03-pitfalls.md`「输出-公式映射表」） |
| 6 | 多输出时返回的 list 长度必须与 `output_dtypes` 长度一致 | 否则 TTK `deep_flatten` 后与输出张量数量不匹配 |
| 7 | const input 不需要 `__input__` 插件 | TTK 自动检测 const input（通过 `_def.cpp` 中的 `.ValueDepend()` 标记），从 `attributes` 中解析值传给算子。`__input__` 插件仅用于非 const 的特殊输入 |
| 8 | DYNAMIC 输入：golden 函数接收 TensorList 为单个位置参数 `list[np.ndarray]` | TTK 通过 `input_apply_as_list` 将扁平数组折叠为嵌套结构。1 个 TensorList 输入 → 1 个位置参数（list of arrays），非 N 个位置参数 |
| 9 | DYNAMIC 输出：golden 函数返回 `list[np.ndarray]`（1 个 per 子 tensor） | TTK `deep_flatten` 自动展平嵌套返回值。返回 `[reference_fn(t) for t in x_list]` 即可，无需手动嵌套 |

### `__golden__` 声明格式

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

### 初步验证

```bash
python3 -c "
import sys; sys.path.insert(0, '{ops_test_kit_path}')
from ttk.core_modules.plugin_loader import get_plugin_function
func, source = get_plugin_function('{op_name}', 'golden', 'kernel', '{plugin_path}')
assert func is not None, 'golden not found'
import numpy as np
np.random.seed(42)
# 按算子实际输入构造测试数据并调用 func，验证返回 list 长度和 dtype
print('PASS')
"
```

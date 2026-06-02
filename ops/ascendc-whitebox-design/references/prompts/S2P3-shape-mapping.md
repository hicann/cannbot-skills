# Task E：shape_mapping 生成

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 详细规则见本文件后续章节，执行顺序节未覆盖的细节以各章节为准。

1. 读 shape_mapping_schema.md → 完整字段定义
   前置：无
2. 读 S2P1_operator_model.json + S2P2_param_def.json
   前置：无
3. 生成 dtype 子节
   前置：Step 2 完成
4. 生成 shape_params 子节（role 判断）
   前置：Step 2 完成
5. 生成 ndim 子节
   前置：Step 2 完成
6. 生成 inputs 子节（rule 判断）
   前置：Step 4 完成
7. 生成 outputs 子节
   前置：Step 2 完成
8. 生成 attrs 子节
   前置：Step 2 完成
9. 写回 S2P1_operator_model.json（仅添加 shape_mapping 字段）
   前置：Step 3-8 完成
10. 执行 9 项校验
   前置：Step 9 完成

**完成标志**：S2P1_operator_model.json 已更新含 shape_mapping 字段，9 项校验全部通过

## 角色

你是 shape 映射专家。基于接口模型 + 参数定义 + 路径分析结果，为算子生成完整的 `shape_mapping` 配置，写入 `S2P1_operator_model.json`。

## 输入

你从主 agent 处获得以下参数：
- `S2P1_operator_model.json` 路径（Task B 产出，含 inputs/outputs/attributes/caller_options）
- `S2P2_param_def.json` 路径（Task D 产出，含 params 定义 + groups + constraints + S5 字段名）
- `S2P1_path_list.json` 路径（Task A/D 产出，辅助理解 tiling 语义，可选）
- 算子路径（可能需要读 infershape.cpp / proto.h 源码确认细节）

## 首要步骤

1. **读取 schema 文档**：Read `{skill_base}/references/shape_mapping_schema.md` 全文，这是 shape_mapping 的**完整字段定义**，后续生成必须严格遵循
2. **读取 S2P1_operator_model.json**：理解 inputs/outputs/attributes 的 dtype/shape 规则
3. **读取 S2P2_param_def.json**：获取 S5 字段名（`params` 的 key 名）、分组信息、约束条件
4. **读取 S2P1_path_list.json**（可选）：辅助理解 tiling 语义（哪些参数参与 shape 计算）

## 生成逻辑

逐子节生成 `shape_mapping`，最终写入 `S2P1_operator_model.json`。

### 1. dtype — 数据类型映射

| 字段 | 填写方法 |
|------|---------|
| `param` | 从 `S2P2_param_def.json` 的 `params` 中找 dtype 类参数的 **key 名**。该参数的 values 应为 dtype 字符串列表 |
| `values` | 从 `S2P1_operator_model.json` 的第一个 `inputs[*].dtype.values` 获取 |
| `default`（可选） | 通常不需要，除非算子有默认 dtype |

验证：`shape_mapping.dtype.values` 必须与 `operator_model.inputs` 中独立 dtype 的 values 一致。

### 2. shape_params — shape 相关参数

| 字段 | 填写方法 |
|------|---------|
| key 名 | 从 `S2P2_param_def.json` 的 `params` 中找参与 shape 构造的参数 key 名 |
| `.role` | 分析该参数的语义角色（见下表） |
| `.type` | 从 `S2P2_param_def.json` 的 `params[param].type` 获取 |
| `.default` | 从算子行为推断的合理默认值（如 leading_product 默认 128） |
| `.group_defaults`（可选） | 某些 group 下该参数的特殊默认值（如某 group 下该参数为固定值） |
| `.values`（可选） | 仅 mode_switch 类参数需要，列出合法值 |

**role 判断方法**：

| role | 判断依据 |
|------|---------|
| `leading_product` | 该参数的值是多个维度的乘积（如 batch×seq），用于 `_balanced_decompose` 分解 |
| `trailing_size` | 该参数的值直接作为 shape 的尾部维度大小（如 hidden_size） |
| `ndim` | 该参数直接指定 tensor 的维度数 |
| `mode_switch` | 该参数控制算子的行为模式分支（如 norm_mode），影响 shape 约束 |
| `other` | 不属于以上任何角色 |

判断依据来源：tiling 源码中的 shape 计算逻辑（S2P1_path_list.json 的 source 字段指向的代码行）、infershape.cpp 中的 shape 推导。

### 3. ndim — 维度数规则

| 字段 | 填写方法 |
|------|---------|
| `source` | `"random"`（随机生成）或 `"from_param"`（从 S5 参数获取）。若算子有显式的 ndim 参数用 `from_param`，否则用 `random` |
| `param` | `source=from_param` 时，从 `S2P2_param_def.json` 找到 ndim 参数的 key 名 |
| `range` | 从 `operator_model.inputs` 的 `rank.min` 和 `rank.max` 获取（取所有输入的交集范围） |
| `fallback_range`（可选） | `from_param` 缺失时的兜底范围 |
| `tensor_constraints` | 从 infershape 源码中的条件约束推导。格式：`{tensor_name: [{when, min/max}, ...]}` |

**tensor_constraints 推导**：

从 infershape.cpp 或 aclnn 接口源码中提取 per-tensor 的 ndim 约束，如：
- "输入 B dim num <= 输入 A dim num" → `{"input_a": [{"when": "...", "min": ...}]}`
- "mode=1 时 dim num <= 2" → `{"input_c": [{"when": "mode_param == 1", "max": 2}]}`

`when` 表达式中的参数名**必须使用 S2P2_param_def.json 的 key 名**。

### 4. inputs — 输入 tensor shape 规则

每个 key 必须与 `operator_model.inputs[*].name` 一致。

**rule 判断**：

| rule | 判断依据 |
|------|---------|
| `decompose` | 主输入 tensor，shape 由多个参数分解而来 |
| `sync_with` | `operator_model.inputs[*].shape.sync_with` 非空，shape 与另一 tensor 完全相同 |
| `fixed` | shape 固定（如 bias 恒为 `[1024]`） |
| `optional` | 该 tensor 可能不存在，由某个参数控制 |

**decompose 子字段**：

| 字段 | 填写方法 |
|------|---------|
| `leading.param` | 从 `shape_params` 中 role=`leading_product` 的参数 key 名 |
| `leading.strategy` | 通常 `"balanced"`；当因子不能为 1 时用 `"balanced_nontrivial"` |
| `trailing[].param` | 从 `shape_params` 中 role=`trailing_size` 的参数 key 名 |
| `parts_expr` | 通常 `"ndim - len(trailing)"` |

**align_trailing_with**（可选）：当该 tensor 的尾部维度需与另一 tensor 对齐时填写。

**ndim_fallback**（可选）：仅配合 `balanced_nontrivial` 使用，分解失败时的回退 ndim。

### 5. outputs — 输出 tensor shape 规则

每个 key 必须与 `operator_model.outputs[*].name` 一致。

**rule 判断**：

| rule | 判断依据 |
|------|---------|
| `same_as` | `operator_model.outputs[*].shape.rule` 为 `same_as_x1` 或 `same_as_input:{name}` |
| `derived` | `operator_model.outputs[*].shape.rule` 为 `derived`，需从 `expr` 转换为 shape_mapping 的 `derived.expr` 格式 |
| `fixed` | 输出 shape 固定 |

**same_as 映射**：

| operator_model.shape.rule | shape_mapping.rule |
|--------------------------|-------------------|
| `same_as_x1` | `{"rule": "same_as", "same_as": "input_a"}` |
| `same_as_input:input_b` | `{"rule": "same_as", "same_as": "input_b"}` |

**derived 映射**：

将 `operator_model.outputs[*].shape.expr` 转换为 shape_mapping 的 `derived.expr` 格式：
- `{tensor_a}.shape[:-{tensor_b}_ndim]` → `{tensor_a}.shape[:-{tensor_b}_ndim]`（`{tensor_b}_dim` → `{tensor_b}_ndim`，表示 `len({tensor_b}.shape)`）
- 中间量用 tensor 名称 + `_ndim` 表示

当输出 dtype 与输入不同时（如 rstd 固定 float32），填写 `dtype_override`。

### 6. attrs — 标量属性

| 字段 | 填写方法 |
|------|---------|
| key 名 | 属性在 map_case 中的引用名（通常与 `operator_model.attributes[*].name` 一致） |
| `.param` | 从 `S2P2_param_def.json` 的 `params` 中找该属性对应的 key 名 |
| `.type` | 从 `operator_model.attributes[*].type` 获取（`"float"` / `"int"` / `"str"` / `"bool"`） |
| `.coerce` | type 为 `float` 时填 `"float"`，其他类型可省略 |

## 输出

将生成的 `shape_mapping` 写入 `S2P1_operator_model.json`：

```python
import json

path = "{S2P1_operator_model.json 路径}"
with open(path) as f:
    data = json.load(f)

data["shape_mapping"] = { ... }  # 完整的 shape_mapping 对象

with open(path, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
```

**只添加 `shape_mapping` 字段，不修改已有的 `inputs`/`outputs`/`attributes`/`caller_options` 字段。**

## 验证

生成后执行以下校验（全部通过才算完成）：

1. **key 一致性**：`shape_mapping.inputs` 的所有 key 存在于 `operator_model.inputs[*].name` 中
2. **key 一致性**：`shape_mapping.outputs` 的所有 key 存在于 `operator_model.outputs[*].name` 中
3. **param 一致性**：`shape_mapping.dtype.param` 存在于 `S2P2_param_def.json` 的 `params` key 中
4. **param 一致性**：`shape_mapping.shape_params` 的所有 key 存在于 `S2P2_param_def.json` 的 `params` key 中
5. **param 一致性**：`shape_mapping.attrs` 的每个 `param` 值存在于 `S2P2_param_def.json` 的 `params` key 中
6. **completeness**：`operator_model` 中声明的所有 input tensor 都在 `shape_mapping.inputs` 中有对应条目
7. **completeness**：`operator_model` 中声明的所有 output tensor 都在 `shape_mapping.outputs` 中有对应条目
8. **decompose 引用**：`decompose.leading.param` 和 `decompose.trailing[].param` 引用的参数名都是 `shape_params` 中的 key
9. **when 表达式**：`tensor_constraints` 和 `group_defaults` 中引用的参数名都是 `S2P2_param_def.json` 中的 key 名

## 关键规则

1. **schema 唯一真相来源**：所有字段定义、结构、语义以 `shape_mapping_schema.md` 为准，不自行发明字段
2. **字段名对齐 S2P2**：所有 `param` 引用和 `shape_params` 的 key 必须使用 `S2P2_param_def.json` 中的 key 名，不自行编造
3. **tensor 名对齐 operator_model**：`inputs`/`outputs` 的 key 必须与 `operator_model` 中的 `name` 一致
4. **以源码为准**：shape 规则、约束条件从 infershape.cpp / proto.h / aclnn 接口源码提取，不猜测
5. **参考完整示例**：`shape_mapping_schema.md` 末尾有完整的 shape_mapping 示例（虚构），生成时可参考其整体结构

## 严格禁止

1. 禁止省略 `shape_mapping` — 必填节，必须生成
2. 禁止修改 `S2P1_operator_model.json` 中已有的 `inputs`/`outputs`/`attributes`/`caller_options` 字段
3. 禁止使用不在 `S2P2_param_def.json` 中的字段名作为 `param` 值
4. 禁止凭直觉推断 shape 规则 — 必须从源码分析得出
5. 禁止跳过验证步骤 — 生成后必须执行 9 项校验

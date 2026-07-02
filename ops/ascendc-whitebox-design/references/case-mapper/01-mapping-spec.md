# Step 5a-pre：映射规格生成 → S5_mapping_spec.md

> **职责**：指导子 agent 从 `S2P1_operator_model.json` + `S2P2_param_def.json` 推导映射逻辑，按 §3 模板生成纯散文的 S5_mapping_spec.md。该文档是 S5_case_mapper.py 和 S5_verify_mapper.py 在映射层面唯一的指导源——它描述目标（参数怎么映射到 shape），02-step5a-mapper.md 负责根据目标生成实现代码。

## 1. 执行流程

1. Read `S2P0_file_manifest.json` 获取 tiling 源码文件路径
2. Read tiling 入口文件（P0 优先级），提取各 shape 参数的实际赋值表达式（如 `{param} = {expr}`），确定参数与维度的对应关系
3. Read `S2P1_operator_model.json` + `S2P2_param_def.json` + `S2P2_cases.json`（采样前 10 条确认字段名）
4. 按 §2 推导来源表确定映射关系
5. 按 §3 模板生成 S5_mapping_spec.md（全散文），写入 `{whitebox}/`
6. 按 `02-step5a-mapper.md` 根据该规格生成实现代码

---

## 2. 推导来源

| 需要确定的信息 | 数据来源 |
|--------------|---------|
| dtype 的控制参数名 + 合法值 | `S2P2_param_def.json` → `dtype_tensors[0].param`；`S2P1_operator_model.json` → `inputs[*].dtype.values` |
| shape 构造参数的 key 名 + 默认值 | `S2P2_param_def.json` → 两级结构（见 §2.1 参数来源层级） |
| 各 shape 参数的默认值 | 算子行为推断（合理默认值） |
| ndim 范围 | `S2P1_operator_model.json` → 所有 `inputs[*].rank.min/max` 的交集 |
| per-tensor ndim 约束 | `S2P1_operator_model.json` → `inputs[*].shape.constraints` |
| 各输入 tensor 的 shape 规则（decompose / sync_with / fixed / optional） | `S2P1_operator_model.json` → `inputs[*]` + tiling 源码（参数赋值表达式，通过 `S2P0_file_manifest.json` 定位） |
| 各输出 tensor 的 shape 规则（same_as / derived / fixed） | `S2P1_operator_model.json` → `outputs[*]` + infershape.cpp（derived.expr 表达式） |
| 各 shape 参数控制哪些 tensor 的哪些维度 | tiling 源码参数赋值表达式（通过 `S2P0_file_manifest.json` 定位） |
| 属性的注入策略（直接取 / 采样） | `S2P1_operator_model.json` → `attributes[*]` |
| DYNAMIC TensorList 的 tensor count | `S2P1_operator_model.json` → `inputs[*].tensor_count`（`{param, min, max}` 或 `{derived_from}`） |
| DYNAMIC 子 tensor 的 dtype/rank | `S2P1_operator_model.json` → `inputs[*].dtype.values` + `inputs[*].rank.min/max` |
| DYNAMIC 子 tensor 的 shape 约束 | `S2P1_operator_model.json` → `inputs[*].shape.constraints`（结构化对象数组，如 `{"type": "same_dtype_within_list"}`） |

### 2.1 参数来源层级

S2P2_param_def.json v3 schema 下，shape 构造参数分布在两个层级：per_dtype 层（路由参数 + 单 key 维度 + compound 维度）和 group 级层（非路由形状维度）。这两层的所有值最终都通过 gen_cases.py 采样后以**标量字段**形式写入 S2P2_cases.json 的每条 case。

case-mapper 生成 S5_mapping_spec.md 时必须识别 case dict 中哪些 key 是 shape 构造参数：

#### 收集方法

```python
# per_dtype 层所有 case key（采样后的标量字段）
per_dtype_keys = set()
for group in param_def["groups"]:
    for dtype, entries in group["per_dtype"].items():
        for entry in entries:
            for k, v in entry.items():
                if k in ("path", "key"):
                    continue
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    # compound 维度：展开为子维度 key（case.json 实际 key）
                    per_dtype_keys |= set(v[0].keys())
                else:
                    # 非 compound：key 本身即 case key
                    per_dtype_keys.add(k)

# group 级层所有 case key（采样后的标量字段）
group_keys = set()
for group in param_def["groups"]:
    group_keys |= set(group.keys()) - {"id", "mode", "per_dtype", "constraint_note"}

# shape 构造参数候选集（与 tiling 源码匹配后确定）
shape_params = per_dtype_keys | group_keys
```

#### 字段分类表

| 层级 | param_def.json 中的位置 | 字段格式 | 在 case dict 中的形态 | mapper 的处理策略 |
|------|--------------------------|---------|---------------------|-----------------|
| **per_dtype 路由参数** | `groups[*].per_dtype[{dtype}][*].{route_param}` | flat value array（≥1 元素，string/int） | `case[{route_param}]` = 标量值（单值） | `_resolve_param(case, "{route_param}", default)` |
| **per_dtype compound 维度** | `groups[*].per_dtype[{dtype}][*].{compound_dim}` | multi-key dict list（5 个 dict，每个含多个子维度键值对） | `case[{sub_a}]`、`case[{sub_b}]` = 标量值（compound 的子维度被展开为独立字段） | `_resolve_param(case, "{sub_a}", default)` 逐个读取 |
| **per_dtype 单 key 维度** | `groups[*].per_dtype[{dtype}][*].{dim_name}` | flat value array（通常 5 元素，退化 1 元素，标量） | `case[{dim_name}]` = 标量值（dtype-dependent 或跨 dtype 同值） | `_resolve_param(case, "{dim_name}", default)` |
| **group 级层** | `groups[*].{dim}` | flat value array（10 个标量值） | `case[{dim}]` = 标量值（影响 tensor shape 的非路由维度） | `_resolve_param(case, "{dim}", default)` |

#### 关键：采样后统一为标量

gen_cases.py 在 case 生成阶段对所有维度执行采样：
- **路由参数**：从 `[V1, V2, ...]` 中采样得到 case 中的单值
- **compound 维度**：从 multi-key dict list 中采样得到 1 个 dict，其所有子维度键值对被**展开**写入 case（如 compound 字段 `{compound_dim}` 展开为 `case["{sub_a}"]` + `case["{sub_b}"]`）
- **单 key 维度**：从 flat value array 中采样得到 case 中的单值
- **group 级层维度**：从 flat value array 中采样得到 case 中的单值

所以 case dict 中**全部是标量字段**，mapper 统一用 `_resolve_param` 读取即可。生成 S5_mapping_spec.md 散文时，需要明确声明每参数的来源和默认值选择依据。

#### 与 tiling 源码的匹配

收集到 case key 候选集后，必须与 tiling 入口文件（P0 优先级）中各 shape 参数的实际赋值表达式匹配，确定：
- 每个 case key 对应哪个 tiling 参数
- 每个 tiling 参数控制哪个 tensor 的哪些维度
- 维度索引表达式（如 `{param} = {tensor_shape}.{expr}`）

匹配结果写入 S5_mapping_spec.md §shape 构造参数 段落。

---

## 3. S5_mapping_spec.md 模板

子 agent 按以下章节结构生成。`{...}` 为占位符，填入具体值。全文为自然语言散文，**禁止建字段表、JSON 示例或代码块**。

```
# {op_name} 参数映射

## 概述
{算子的输入输出一句话描述}

## dtype
所有 tensor 的数据类型由 case 参数 `{param}` 控制，合法取值为 {values}。

## shape 构造参数

（以下三小节为 mapper 生成提供完整语义描述）

### 总览

参数按层级分类如下。`case key` 表示该参数在 case dict 中的字段名，`控制关系` 表示它影响哪个 tensor 的哪个维度。

- {compound_param}（compound）：case key 为 {sub_a}、{sub_b}，分别控制 {tensor_a}.dim_{x} 和 {tensor_b}.dim_{y}
- {single_key_param}（单 key 维度）：case key 为 {key}，控制 {tensor}.dim_{z}
- {group_level_param}（group 级维度）：case key 为 {key}，控制 {tensor}.dim_{z}
- {route_param}（路由参数，可选）：case key 为 {key}，不直接决定 dim 数值，但影响某些 dim 的索引选择

### 各参数语义详解

**{compound_param}**（compound 维度）：由 case 中的多个标量字段表示，每个字段控制一个 tensor 的某个维度。例：compound 字段 `{compound_dim}` 展开为 case 中的 `{sub_a}` 和 `{sub_b}`，分别控制不同 tensor 的对应维度。默认值 `{default}`（用于属性注入）。

**{single_key_param}**（单 key 维度）：由 case 中的单个标量字段表示，控制某 tensor 的特定维度。例：`{dim_name}` 控制 `{tensor}` 的某维度（dim 索引可能受路由参数影响）。默认值 `{default}`。

**{group_level_param}**（group 级维度）：由 case 中的单个标量字段表示，通常是影响 tensor 总体空间但不影响路径选择的维度。例：`{dim}` 控制输入 `{tensor}` 的某维度（所有路径共享）。默认值 `{default}`。

**{route_param}**（路由参数，可选）：控制代码分支选择的参数，可能影响 dim 索引而非 dim 数值。例：`{route_param}={route_val_A}` 时 `{tensor}` 的 dim 顺序为 `{dim_seq_A}`；`{route_param}={route_val_B}` 时 dim 顺序为 `{dim_seq_B}`。mapper 根据此参数决定后续 shape 各维的语义映射。

### 映射规则汇总

- 若某 dim 受 compound 控制 → 直接取 case 中的对应子维度字段（已展开为标量）
- 若某 dim 受单 key 维度控制 → 直接取 sampled 值，必要时做分解（decompose）
- 若某 dim 受 group 级维度控制 → 直接取 sampled 值，必要时做分解
- 若某 dim 索引受路由参数影响 → 根据 route 值切换 dim 索引选择
- 所有参数在 case 中已确定（采样完成），mapper 直接读取，不需要做额外随机

## 维度数（ndim）
主要由各 tensor 的合法 ndim 范围及相互约束决定（见输入/输出 tensor 各节）。{如有全局规则在此描述}

## 输入 tensor

按算子定义中的入参顺序逐一描述。每个 tensor 根据 param_type 选择对应模板。

> **标注规则（强制）**：每个 tensor 的 header 必须显式标注 param_type —— `### {name}（REQUIRED）` 或 `### {name}（DYNAMIC）`，不可省略。Step 6（pytest-gen）依赖此标注进行 param_type 派发，缺失标注将导致派发失败。

### {name}（REQUIRED）
ndim 合法范围 [{lo}, {hi}]。{如有与其他 tensor 的 ndim 关系}

特殊的轴约束：
- dim {N}：{约束描述}（无为"无"）

{如果 dtype/rank/shape 含 sync_with：它的 dtype/rank/shape 与 {target} 完全相同}
{如果 shape.constraints 含字符串约束：逐条列出}
{如果 rule=fixed：固定为 {shape}}

### {name}（DYNAMIC）
tensor count：{从 operator_model 的 tensor_count 字段读取}
- 若 tensor_count 为 {param, min, max}：count 范围 [{min}, {max}]，由 case 参数 {param} 控制
- 若 tensor_count 为 {derived_from}：count 与 {target} 相同

dtype 合法值：{values}。
ndim 合法范围 [{min}, {max}]。

shape 约束（来自 operator_model shape.constraints）：
- {逐条列出 constraints 中的结构化对象，如 {"type": "same_dtype_within_list"} → 所有子 tensor dtype 相同}

子 tensor shape 构造规则：
- 各 shape 参数来源与 REQUIRED tensor 相同（来自 §shape 构造参数）
- {如有特殊规则在此描述}

### {name}（...）
...（按入参顺序，每个输入 tensor 一段，标注 REQUIRED 或 DYNAMIC，不可省略）

## 输出 tensor

按算子定义中的输出顺序逐一描述。每个 tensor 根据 param_type 选择对应模板。

> **标注规则（强制）**：每个 tensor 的 header 必须显式标注 param_type —— `### {name}（REQUIRED）` 或 `### {name}（DYNAMIC）`，不可省略。Step 6（pytest-gen）依赖此标注进行 param_type 派发，缺失标注将导致派发失败。

### {name}（REQUIRED）
它的 shape {rule=same_as: 与 {target} 完全相同 / rule=derived: 由表达式 {expr} 推导 / rule=fixed: 固定为 {shape}}。
dtype {来源}。

### {name}（DYNAMIC）
tensor count：{derived_from: 与输入 {target} count 相同}。

子 tensor 推导规则：
- shape：{从 operator_model shape.rule 读取}
- dtype：{从 operator_model dtype.source 读取}
- rank：{从 operator_model rank.source 读取}

### {name}（...）
...（按输出顺序，每个输出 tensor 一段，标注 REQUIRED 或 DYNAMIC）

## 属性

### 属性参与性判定（强制）

并非所有属性都需要枚举不同值。只有**参与代码路径**的属性才写入本节并在测试中体现；不参与的属性忽略（不写入 `params`，让算子使用默认行为）。

**判定方法**：对 `S2P1_operator_model.json` 的每个属性，通过 `S2P1_tiling_glossary.md` 找到对应的 tiling 变量名，然后检查以下三个来源：

1. `S2P1_path_list.json` 各 path 的 `conditions` 字段
2. tiling 源码（`S2P0_file_manifest.json` P0 文件）的分支条件（`if`/`switch`/三元表达式）
3. kernel dispatch 文件（`*_apt.cpp`）的 `TILING_KEY_IS` 块内条件

**任一来源出现该变量 → 参与**（按 group/路径分支中已有的值取用，生成 §属性.{name} 段落）；**全部未出现 → 不参与**（忽略，不生成段落，不在 `params` 中出现）。

### {name}
来自 case["{key}"]（{type}）。缺失时按 {strategy}({参数}) 采样。

### const input tensor 处理（强制）

`_def.cpp` 中通过 `.ValueDepend()` 标记的输入 tensor 是 const input。const input 的值在编译期确定，影响输出 shape 推导，必须参与维度组合。

**识别方法**：
- **源码标记**：`_def.cpp` 中 `.Input("{name}").ValueDepend(OPTIONAL)` 或 `.ValueDepend(REQUIRED)`
- **结构化记录**：`S2P1_operator_model.json` 的 `inputs[*].shape.constraints` 中包含 `ValueDepend` 关键字

**shape 确定**：从 `S2P1_operator_model.json` 的 `inputs[*].rank` 和 `shape.constraints` 读取。const input 的 shape 通常固定（由 `_def.cpp` 注册决定）。

**值确定**：从 `_infershape.cpp` 和 tiling 源码追溯 const input 各元素与维度参数的映射关系。映射结果写入 §const input.{name} 段落，供 case mapper 消费。

**处理规则**：
1. const input 的**值**写入 `attributes` dict（key 为 const input 名称），而非作为随机数据输入
2. const input 的 **shape/dtype** 仍保留在 `input_shapes`/`input_dtypes` 中（TTK 需要知道 tensor 元信息）
3. const input 的值由 §shape 构造参数 中的维度参数构造，在 §const input.{name} 中描述映射关系
4. TTK 自动检测 const input（通过算子元信息中的 `valueDepend` 标记），从 `attributes` 中解析值传给算子，无需 `__input__` 插件

### const input.{name}
- shape: `{shape}`（固定）
- dtype: `{dtype}`
- 值映射：`{name}[0]` → `{dim_0}`，`{name}[1]` → `{dim_1}`，...
- 构造方式：`{name} = [case["{dim_0}"], case["{dim_1}"], ...]`
- 写入 `params["{name}"]`

## 验证规则（指导 S5_verify_mapper.py 生成）

| 层 | 检查内容 | 推导来源 |
|----|---------|---------|
| L1 | 字段级动态校验（ndim 范围、dtype 合法值、tensor 对齐等） | 本文档 §shape 构造参数~§输出 tensor |
| L2 | operator_model 交叉验证（outputs key、dtype/rank 合法范围） | S2P1_operator_model.json |
| L3 | source_constraints 交叉验证（shape 语义约束） | operator_model.shape.constraints |
| L4 | NPU e2e（调用算子 API 检查 output shape） | 运行时 |
```

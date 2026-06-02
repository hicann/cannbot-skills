# shape_mapping Schema 参考文档

`shape_mapping` 是 `S2P1_operator_model.json` 的一个**必选节**，描述如何将 S2P2_cases.json 中的抽象参数映射为具体的 tensor shape。它是 map_case 函数的**唯一驱动源**。

**生成者**：Step 2 Phase 3c Task E。消费 `S2P1_operator_model.json`（接口模型）+ `S2P2_param_def.json`（参数定义 + S5 字段名）+ `S2P1_path_list.json`（路径分析，辅助理解 tiling 语义），生成完整的 `shape_mapping` 并写回 `S2P1_operator_model.json`。

```json
{
  "shape_mapping": {
    "dtype":        { ... },
    "shape_params": { ... },
    "ndim":         { ... },
    "inputs":       { ... },
    "outputs":      { ... },
    "attrs":        { ... }
  }
}
```

---

## 1. dtype — 输入数据类型

声明 S2P2_cases.json 中哪个字段是 dtype 参数，以及合法的 dtype 值。

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `param` | string | 是 | S5 中 dtype 参数的字段名（如 `"{input_name}_dtype"`） |
| `values` | string[] | 是 | 合法 dtype 值列表，用于生成 DTYPE_MAP |
| `default` | string | 否 | 参数缺失时的默认 dtype |

```json
"dtype": {
  "param": "{input_name}_dtype",
  "values": ["float16", "bfloat16", "float32"]
}
```

map_case 用法：`dtype = DTYPE_MAP[case[sm["dtype"]["param"]]]`

---

## 2. shape_params — shape 相关参数声明

声明 S2P2_cases.json 中哪些参数参与 shape 构造，及其语义角色。

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `<param_name>` | object | — | key 为 S5 中参数字段名 |
| `.role` | string | 是 | 参数语义角色（见下表） |
| `.type` | string | 是 | 值类型：`"int"` / `"float"` / `"str"` |
| `.default` | any | 否 | 缺失时的默认值 |
| `.group_defaults` | object | 否 | 按 `_group` 的特殊默认值 |
| `.values` | string[] | 否 | 合法取值列表（mode_switch 类常用） |

### role 枚举

| role | 含义 | map_case 用法 |
|------|------|--------------|
| `leading_product` | 多个维度的乘积 | `_balanced_decompose` 分解为多个因子 |
| `trailing_size` | 尾部单个维度大小 | 直接作为 shape 最后几维 |
| `ndim` | 显式指定的维度数 | 直接使用，配合 `ndim.source: "from_param"` |
| `mode_switch` | 行为模式开关 | 条件分支判断 |

不支持自定义 role。新增 role 前必须先在本 schema 和 S5 mapper 中定义消费语义。

```json
"shape_params": {
  "dim_front": {
    "role": "leading_product",
    "type": "int",
    "default": 128,
    "group_defaults": {"special_group": 1}
  },
  "dim_back": {
    "role": "trailing_size",
    "type": "int"
  }
}
```

参数取值优先级：`S5 case[param]` > `group_defaults[group]` > `default`

---

## 3. ndim — 维度数确定规则

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `source` | string | 是 | `"random"` 或 `"from_param"` |
| `param` | string | 条件 | `from_param` 时的 S5 字段名 |
| `range` | [int, int] | 是 | 随机范围或合法范围 |
| `fallback_range` | [int, int] | 否 | `from_param` 缺失时的兜底 |
| `tensor_constraints` | object | 否 | per-tensor ndim 条件约束 |

### source

| 值 | map_case 行为 |
|----|--------------|
| `"random"` | `ndim = rng.choice(range(min, max+1))` |
| `"from_param"` | `ndim = case[param]` |

### tensor_constraints

对特定 tensor 施加的 ndim 条件约束。多个约束取交集。

| 字段 | 类型 | 含义 |
|------|------|------|
| `when` | string | 触发条件，引用 S5 case 参数值 |
| `min` | int | ndim 下限 |
| `max` | int | ndim 上限 |

`when` 仅允许以下结构：
- 标识符：S5 case 中的字段名，正则 `[A-Za-z_][A-Za-z0-9_]*`
- 字面量：字符串、整数、浮点数、布尔值
- 比较：`==`, `!=`, `<`, `<=`, `>`, `>=`
- 逻辑：`and`, `or`, `not`
- 括号分组

禁止函数调用、属性访问、下标访问、导入、算术表达式和任意 Python `eval`。消费方必须用 AST 白名单求值。

```json
"ndim": {
  "source": "random",
  "range": [1, 8],
  "tensor_constraints": {
    "input_a": [{"when": "op_mode != 'mode_a'", "min": 2}],
    "weight": [
      {"when": "op_mode == 'mode_a'", "max": 1},
      {"when": "op_mode != 'mode_a'", "max": 2}
    ]
  }
}
```

构造各 tensor 时，对 base ndim 范围应用该 tensor 的 constraints 过滤，得到该 tensor 的合法 ndim 列表，再从中选择。

---

## 4. inputs — 输入 tensor shape 规则

每个 key 对应一个输入 tensor 名称。

### 顶层字段

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `rule` | string | 是 | 规则类型 |
| `align_trailing_with` | string | 否 | 尾部维对齐目标（仅 decompose） |
| `ndim_fallback` | int | 否 | nontrivial 失败回退 ndim（仅 decompose） |

### rule 枚举

| rule | 含义 | 适用场景 |
|------|------|---------|
| `decompose` | 分解参数为多维 shape | 主输入 tensor |
| `sync_with` | shape + dtype 与另一 tensor 相同 | x2=f(x1) |
| `fixed` | 固定 shape | 特殊 tensor |
| `optional` | 条件性存在 | 可选 tensor |

---

### rule: decompose

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `decompose.leading` | object | 否 | 前导维度配置（无此字段则无前导维） |
| `decompose.trailing` | array | 否 | 尾部维度配置列表 |
| `decompose.parts_expr` | string | 否 | 前导因子个数表达式，通常 `"ndim - len(trailing)"` |

#### leading 对象

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `param` | string | 是 | 乘积来自哪个 shape_param |
| `strategy` | string | 是 | `"balanced"` / `"balanced_nontrivial"` / `"none"` |
| `default` | int | 否 | 覆盖 shape_params 中的 default |
| `group_defaults` | object | 否 | 覆盖 shape_params 中的 group_defaults |

#### trailing 数组元素

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `param` | string | 是 | 该维来自哪个 shape_param |
| `strategy` | string | 否 | 分解策略（默认 `"none"`，直接取值） |

#### 分解策略

| strategy | 含义 | 失败行为 |
|----------|------|---------|
| `balanced` | **乘积分解**：各因子尽量接近，满足 `∏factors == param_value`，允许因子=1 | 无失败（允许因子=1） |
| `balanced_nontrivial` | **乘积分解**：每个因子 > 1，满足 `∏factors == param_value` | 质因数不足时返回含 1 结果，调用方按 ndim_fallback 回退 |
| `none` | 不分解，直接取值 | 无失败 |

#### decompose 通用流程

```
1. trailing_shape = 对每个 trailing 元素：取 param 值，按 strategy 处理
2. leading_dims_count = parts_expr 求值（通常 = ndim - len(trailing)）
3. 若 leading_dims_count == 0：leading_shape = ()
4. 若 leading_dims_count > 0：
   a. product = _resolve_param(case, leading.param, shape_params)
   b. 按 leading.strategy 将 product 乘积分解为 leading_dims_count 个因子
   c. 若 strategy=balanced_nontrivial 且分解失败（存在因子<=1）：
      - 按 ndim_fallback 回退：trailing 合并为单维，重新分解 leading
5. shape = leading_shape + trailing_shape
```

#### 示例：有 leading

```json
"input_a": {
  "rule": "decompose",
  "decompose": {
    "leading": {"param": "dim_front", "strategy": "balanced"},
    "trailing": [{"param": "dim_back"}],
    "parts_expr": "ndim - len(trailing)"
  }
}
```

推导：`dim_front=128, dim_back=1999, ndim=2` → `leading_dims_count=1` → `leading=(128,)` → `trailing=(1999,)` → `shape=(128, 1999)`

#### 示例：无 leading

```json
"weight": {
  "rule": "decompose",
  "decompose": {"trailing": [{"param": "dim_back"}]},
  "align_trailing_with": "input_a",
  "ndim_fallback": 1
}
```

推导：`dim_back=1999, ndim=1` → `leading_dims_count=0` → `leading=()` → `trailing=(1999,)` → `shape=(1999,)`

---

### rule: sync_with

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `sync_with` | string | 是 | 目标 tensor 名称 |

```json
"input_b": {"rule": "sync_with", "sync_with": "input_a"}
```

用法：`cfg.inputs["input_b"] = copy(cfg.inputs["input_a"])`

---

### rule: fixed

| 字段 | 类型 | 含义 |
|------|------|------|
| `shape` | int[] | 固定 shape |
| `dtype` | string | 固定 dtype（可选，默认同主输入） |

```json
"bias": {"rule": "fixed", "shape": [1024], "dtype": "float32"}
```

---

### rule: optional

| 字段 | 类型 | 含义 |
|------|------|------|
| `optional_param` | string | S5 中控制存在性的参数名 |
| `shape_when_present` | object | 存在时的 shape 规则（decompose / fixed） |

```json
"mask": {
  "rule": "optional",
  "optional_param": "has_mask",
  "shape_when_present": {"rule": "decompose", ...}
}
```

用法：`case[optional_param]` 为真时构造 tensor，否则置 None。

---

### align_trailing_with

仅用于 decompose。语义：`this.shape == target.shape[-len(this.shape):]`

```json
"align_trailing_with": "{target_tensor}"
```

validate 校验：`src.shape == target.shape[-len(src.shape):]` 且 `src.ndim ≤ target.ndim`

### ndim_fallback

仅用于含 `balanced_nontrivial` 的 decompose。分解失败时回退到指定 ndim：
1. trailing 合并为单维（取 product）
2. 重新计算 `leading_dims_count = ndim - ndim_fallback`
3. 用 `balanced` 重新分解 leading

---

## 5. outputs — 输出 tensor shape 规则

| rule | 含义 | 适用场景 |
|------|------|---------|
| `same_as` | shape + dtype 与指定 tensor 相同 | 残差、element-wise |
| `derived` | 从输入 shape 推导 | 中间结果 |
| `fixed` | 固定 shape | 特殊输出 |

### rule: same_as

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `same_as` | string | 是 | 目标 tensor 名称 |

```json
"output_y": {"rule": "same_as", "same_as": "input_a"}
```

### rule: derived

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `expr` | string | 是 | shape 推导伪代码 |
| `dtype_override` | string | 否 | 输出 dtype 覆盖 |

expr 中可引用：
- 输入 tensor shape：`{tensor_name}.shape`、`{weight_name}.shape`
- 中间量：`{weight_name}_ndim` = `len({weight_name}.shape)`
- Python 语法：切片、拼接、元组乘法

`derived.expr` 仅允许引用已构造 tensor 的 `.shape`、`len(name.shape)`、切片、元组/列表拼接、整数常量和元组乘法。禁止函数调用（除 `len`）、属性链展开、文件/模块访问和依赖 case 之外的全局变量。

```json
"output_stat": {
  "rule": "derived",
  "derived": {
    "expr": "input_a.shape[:-weight_ndim] + (1,) * weight_ndim",
    "dtype_override": "float32"
  }
}
```

---

## 6. attrs — 标量属性

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `param` | string | 是 | S5 中对应字段名 |
| `type` | string | 是 | `"float"` / `"int"` / `"str"` / `"bool"` |
| `coerce` | string | 否 | 强制类型转换函数名 |

```json
"attrs": {
  "scalar_attr": {"param": "scalar_attr", "type": "float", "coerce": "float"}
}
```

用法：`val = case[spec["param"]]; val = float(val) if coerce else val`

### 6a. sampling — 非路由属性的注入策略

attrs 中未在 S5 param_def 出现的属性（如 `epsilon`），由 mapper 自动采样注入：

| 字段 | 类型 | 必填 | 含义 |
|------|------|------|------|
| `sampling.strategy` | string | 否 | `"log_uniform"` / `"uniform"` / `"choice"` / `"default"` |
| `sampling.range` | [float, float] | 按策略 | log_uniform / uniform 的采样区间 |
| `sampling.values` | [any] | 按策略 | choice 的候选值列表 |

示例：
```json
"attrs": {
  "epsilon": {
    "param": "epsilon",
    "type": "float",
    "coerce": "float",
    "sampling": {
      "strategy": "log_uniform",
      "range": [1e-12, 1e-04]
    }
  }
}
```

---

## 完整示例（虚构，仅示意各节如何互相配合）

```json
{
  "shape_mapping": {
    "dtype": {
      "param": "input_a_dtype",
      "values": ["float16", "bfloat16", "float32"]
    },
    "shape_params": {
      "dim_front": {
        "role": "leading_product",
        "type": "int",
        "default": 128,
        "group_defaults": {"special_group": 1}
      },
      "dim_back": {
        "role": "trailing_size",
        "type": "int"
      },
      "op_mode": {
        "role": "mode_switch",
        "type": "str",
        "values": ["mode_a", "mode_b", "mode_c"]
      }
    },
    "ndim": {
      "source": "random",
      "range": [1, 8],
      "tensor_constraints": {
        "input_a": [{"when": "op_mode != 'mode_a'", "min": 2}],
        "weight": [
          {"when": "op_mode == 'mode_a'", "max": 1},
          {"when": "op_mode != 'mode_a'", "max": 2}
        ]
      }
    },
    "inputs": {
      "input_a": {
        "rule": "decompose",
        "decompose": {
          "leading": {"param": "dim_front", "strategy": "balanced"},
          "trailing": [{"param": "dim_back"}],
          "parts_expr": "ndim - len(trailing)"
        }
      },
      "input_b": {"rule": "sync_with", "sync_with": "input_a"},
      "weight": {
        "rule": "decompose",
        "decompose": {"trailing": [{"param": "dim_back"}]},
        "align_trailing_with": "input_a",
        "ndim_fallback": 1
      }
    },
    "outputs": {
      "output_y": {"rule": "same_as", "same_as": "input_a"},
      "output_stat": {
        "rule": "derived",
        "derived": {
          "expr": "input_a.shape[:-weight_ndim] + (1,) * weight_ndim",
          "dtype_override": "float32"
        }
      },
      "output_x": {"rule": "same_as", "same_as": "input_a"}
    },
    "attrs": {
      "scalar_attr": {"param": "scalar_attr", "type": "float", "coerce": "float"}
    }
  }
}
```

---

## 能力边界

| 场景 | 支持 | 说明 |
|------|------|------|
| 单输入 element-wise | ✅ | x: decompose, y: same_as |
| norm 类（leading + trailing + 权重对齐） | ✅ | 已验证 |
| 多输入不同 shape | ✅ | 每个 input 独立 decompose |
| 可选 tensor | ✅ | rule: optional |
| ndim 来自 S5 参数 | ✅ | ndim.source: from_param |
| 输出 shape 推导 | ✅ | derived.expr |
| 输出 dtype 覆盖 | ✅ | derived.dtype_override |
| trailing 多维分解 | ✅ | trailing[].strategy |
| conv 类独立维度（N,C,H,W） | ❌ | 需新增 dims rule（未来扩展） |
| 动态 shape（-1 维度） | ❌ | 当前 Step 1-6 主流程不支持；如启用 TTK 模块，仅在 TTK CSV 转换阶段作为工具格式字段处理，不作为主流程 shape_mapping 能力 |

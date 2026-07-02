# Task B 产出验证

校验 `S2P1_operator_model.json` 中对算子接口的声明是否真实。辅助源码 `_def.cpp` + `_infershape.cpp` 为权威真值。

## 源码优先级规则（强制）

`_def.cpp` 是算子 inputs、outputs、attributes 的**唯一权威来源**。`_infershape.cpp` 仅作为 shape 推导逻辑和辅助验证的补充。

| 场景 | 处理方式 |
|------|---------|
| `_def.cpp` 中声明的 dtype/format/属性 | 最终真值，operator_model 必须与之完全一致 |
| `_infershape.cpp` 中出现但 `_def.cpp` 中不存在的信息（如额外 dtype） | **不计入** operator_model，不视为 operator_model 缺失 |
| `_def.cpp` 与 `_infershape.cpp` 冲突 | 以 `_def.cpp` 为准 |
| `_infershape.cpp` 中的 shape 推导逻辑（`SetDim`、`*y_shape = *x_shape` 等） | 作为 B1.5 outputs.shape 的验证依据 |

---

## B1.1 inputs

Read `_def.cpp` 的 `.Input("name")...DataType(...)` 段，逐条比对 operator_model `inputs[]`：

- **name**：必须与 `_def.cpp` 的 `Input()` 注册名一致 → **fail** 不一致
- **dtype**，按 `_def.cpp` 中 DataType 参数形式分两种处理：
  - 直接列表（`.DataType({DT_A, DT_B, ...})`）：operator_model `values` 须为同一集合 → **fail**
  - 共享数组引用（`.DataType(valueDataTypeX)`）：Grep 该数组变量取值列表；operator_model `values` 须为其去重子集 → **fail** 遗漏或多出

---

## B1.2 outputs

Read `_def.cpp` 的 `.Output("name")...DataType(...)` 段，逐条比对 operator_model `outputs[]`：

- **name**：必须与 `_def.cpp` 的 `Output()` 注册名一致 → **fail**
- **dtype**，按 operator_model 写法分三类：

| operator_model 写法 | 验证规则 |
|--------------------|---------|
| `{values: [...]}` | 同 B1.1 dtype 规则 |
| `{sync_with: "{x}"}` | (1) `{x}` 在 `_def.cpp` 中存在；(2) `_infershape` 中 `InferDtype*` 函数以 `{x}`.dtype 推导 output dtype；(3) 推导逻辑无附加约束 → **pass**；有附加约束（如 dtype 属性联合限制）→ **warn** |
| `{fixed: "{type}"}` | `InferDtype*` 中 `SetOutputDataType` 固定为 `DT_{TYPE}` 且无条件分支 → **fail** 不一致 |

---

## B1.3 attributes

Read `_def.cpp` 的 `.Attr("name")` 注册段，逐条比对 operator_model `attributes[]`：

- 属性名一致（顺序无关）
- 类型一致（`.Bool()`/`.Int()`/`.ListFloat()`/`.Float()`/`.String()` ↔ `type` 字段）
- 默认值一致
- operator_model 多出的属性（如 aclnn 独有）→ **warn**
- 属性名/类型/默认值不一致 → **fail**

---

## B1.4 dtype 约束验证

两层 Grep 提取算子支持的 dtype 枚举集：

**Step A**：Grep `_def.cpp` 中 `OP_CHECK_DTYPE_NOT_SUPPORT` / `OP_CHECK_DTYPE_NOT_MATCH`
- 命中 → 提取枚举集，与 operator_model 各 input/output 的 dtype 比对

**Step B**（fallback）：Read `_infershape.cpp` 中 `InferDtype*` / `InferDataType*` 函数，识别：
- `OP_CHECK_IF(... == ge::DT_* || ...)` → dtype 枚举集
- `{ge::DT_*, ge::DT_*, ...}` → dtype 数组
- `SetOutputDataType(...)` → 输出 dtype

提取约束后与各 input/output 的 dtype 比对 → **fail** 不一致；两层都无命中 → **warn**

---

## B1.5 outputs.shape

当 `outputs[*].shape.rule == "derived"` 时，对 `expr` 执行 4 点语义检查（Read `shape.source` 行号处 ±5 行的 `_infershape.cpp` 代码）：

1. **维度保持**：expr 声称"与 x 相同"的维度 → `_infershape` 中有 `*y_shape = *x_shape` 或 `SetDim(idx, x->GetDim(idx))`
2. **维度替换**：expr 声称"由其他输入推导"的维度（如 `H→size[0]`）→ `SetDim(h_idx, out_info.output_h)` 的赋值来源匹配
3. **format 索引**：expr 提到 format-dependent 索引 → `_infershape` 中有对应分支（如 `format == NHWC ? 1 : 2`）
4. **行号**：`shape.source` 行号 ±5 行须包含相关推导逻辑 → **fail** 行号不对

1-3 满足 + 行号有效 → **pass**；行号错误 → **fail**；语义基本一致但措辞差异 → **warn**

`rule == "same_as_input"` 时，验证 `_infershape` 中 `*y_shape = *x_shape`（其中 `x` 为 `shape.input` 字段引用的输入 tensor）（无 `SetDim` 覆写）→ **fail** 不一致

---

## B1.6 platform

Read `_def.cpp` 的 `.AICore().AddConfig(...)` 段，提取平台列表 + capability flags（`DynamicShapeSupportFlag`、`DynamicRankSupportFlag` 等）。

- operator_model 的 `platform` 字段描述的架构与 AddConfig 平台列表一致 → **fail** 完全不匹配
- operator_model 有 `aiconfig` 字段时比对 capability flags → **fail** 不一致；无该字段 → **warn**

---

## B1.7 结构完整性

| 字段 | 验证规则 | 判定 |
|------|---------|:----:|
| `inputs[*].rank` | 每个 input 有 rank 字段；`_infershape` 中有 `GetDimNum()` 校验时比对值 | **warn** 缺失；**fail** 值不一致 |
| `inputs[*].shape.constraints` | 每条约束与 `_infershape` 的 `OP_CHECK_IF` 核对一致性 | **warn** |
| `torch_npu_api_exposure` | 字段存在（可以是 `null`） | **warn** 缺失 |

---

## B1 整体判定

任一子项 fail → **fail**；仅 B1.2 sync_with / B1.6 / B1.7 warn → **pass_with_warnings**；全 pass 无 warn → **pass**。

## 输出格式

```
| ID | 状态 | verified/total | 备注 |
|----|------|---------------|------|
| B1.1 | pass/fail | v/t | |
| B1.2 | pass/fail/warn | v/t | sync_with warn: N |
| B1.3 | pass/fail/warn | v/t | aclnn独有 warn: N |
| B1.4 | pass/fail/warn | v/t | |
| B1.5 | pass/fail/warn | v/t | |
| B1.6 | pass/warn | —/— | |
| B1.7 | pass/warn | —/— | |
```

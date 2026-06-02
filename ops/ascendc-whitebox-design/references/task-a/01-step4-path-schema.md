# path-schema：路径清单与变量分类

对应 Task A 执行步骤 4：构建路径清单 + 变量三分类。

---

## 路径清单 JSON Schema

结构化 JSON 列表。每条路径 schema：

```json
{
  "id": "P1",
  "name": "描述性名称（涉及输入/输出/属性信息时须在属性值前写明参数名）",
  "conditions": [
    {"var": "参数名_属性", "op": "运算符", "value": "值"}
  ],
  "input_variables": ["对应算子输入参数/属性的变量"],
  "caller_options": ["调用者通过 API 调用方式控制的选项"],
  "internal_variables": ["路径内部的派生量"],
  "key_instructions": ["该路径使用的关键指令或函数"],
  "source": "tiling 文件:行号 → kernel 文件:行号"
}
```

Task A 不指定 group 归属。Group 划分由 Task D 在 Phase 2 完成。

---

## Conditions Schema

数组格式，固定格式，不允许自由文本：

| 条件类型 | 格式 |
|----------|------|
| 等于/不等于 | `{"var": "x", "op": "==", "value": 0}` |
| 范围 | `{"var": "x", "op": "range", "min": N, "max": M}` |
| 大于/小于等 | `{"var": "x", "op": ">", "value": 8}` |
| 枚举 | `{"var": "x", "op": "in", "value": [...]}` |
| 整除 | `{"var": "x", "op": "mod_eq", "divisor": 32, "remainder": 0}` |
| 变量比较 | `{"var": "a", "op": "==", "ref": "b"}` |
| 变量不等式 | `{"var": "a", "op": "<=", "ref": "b"}` |
| 派生表达式 | `{"expr": "a % b", "op": "==", "value": 0}` |

区分规则：比较常量用 `value`，比较另一个变量用 `ref`，多变量运算用 `expr`。

---

## 命名规则

**变量命名：`{参数名}_{属性}`**。凡涉及输入/输出 tensor 或属性的属性字段，必须标注来源参数名（如 `x1_dtype`）。即使只有一个来源也必须标注前缀，禁止裸属性名。

**path name**：涉及输入/输出/属性信息时，必须在属性值前写明参数名。格式如 `{mode}_{参数名}_{dtype}`。

---

## 变量三分类

- `input_variables`：用户可控制的输入属性。包括直接设置（tensor dtype、标量属性）和 shape/dtype 派生（由输入 tensor 的 shape/dtype/rank 经算术运算得出的量，用户可通过构造 tensor 间接控制）。

- `caller_options`：调用者通过选择 API 调用方式控制的执行路径选项（非 tensor shape/dtype/属性）。caller_options 是调用者控制的抽象，不是 tiling 内部编码。

- `internal_variables`：路径内部的派生量，计算链不经过任何用户可控属性（仅依赖平台常量、编译时常量、其他 internal_variable）。包括框架内部信号（aclnn 层决策在 tiling 层的编码）。internals 仅在分支树中记录以保持完整性，不映射为 S2P2_param_def.json 的维度。

只有 input_variables 和 caller_options 会映射为 S2P2_param_def.json 的维度。

---

## 判定流程

对 conditions 中出现的每个变量 v：

1. v 的值由用户通过 tensor shape/dtype/rank 或标量属性直接设置 → `input_variable`
2. v 反映调用者控制的抽象选项 → `caller_option`
3. v 反映框架编码信号 → `internal_variable`
4. v 的值在 tiling/kernel 内部计算得出 → **必须追溯计算链**：
   a. 计算链经过 tensor 的 shape/dtype/rank 或标量属性 → 将用户可控属性归为 `input_variable`
   b. 计算链不经过任何用户可控量 → `internal_variable`
   c. **内部变量边界检查**：当内部变量的边界检查（如 `内部变量 == 边界值`）导致不同代码路径（不同 kernel dispatch、不同 mode_key 赋值）时：

      i. 识别内部变量边界检查 → 记录条件表达式和源码位置

      ii. 写出内部变量的完整计算链，格式：
          ```
          内部变量 = 表达式
            ├─ 常量A = 值（源码行号或 constexpr 定义）
            ├─ 常量B = 值（源码行号或 constexpr 定义）
            └─ 中间变量 = 子表达式（源码行号）
          ```

      iii. 将边界检查条件记录到该路径的 `conditions` 中，格式为：
           `{"boundary_check": "内部变量 == 边界值"}`

      iv. 边界值的数学反推（解方程、代入平台常量、对齐校正等）由 Task D 完成，此处仅记录原始条件。

   d. **降级路径独立**：当 tiling 代码中存在"先进入某 mode 分支，再因内部变量边界检查回退到另一个 mode 或使用不同 kernel"的模式时，降级后的路径必须作为独立路径条目创建。

      判定标准（满足任一即需独立）：
      1. 内部变量边界检查导致 key 分量被重新赋值为其他值
      2. 内部变量边界检查导致实际执行的 kernel 指令与该分支主路径不同（不同的 `key_instructions`）

      执行规则：
      1. 降级路径 `id` 在主路径之后递增分配（如 P5 → P5d1）
      2. conditions 包含：降级前的入口条件 + 降级触发条件
      3. key_instructions 填写降级后实际执行的 kernel（不是降级前的）
      4. input_variables 和 caller_options 与主路径相同
      5. internal_variables 额外记录导致降级的内部变量
      6. source 引用降级回退后的赋值代码行号

      禁止合并降级路径到降级目标 mode 的路径中；禁止省略降级路径（即使它与已有路径使用相同 kernel）。

步骤 3 和 4a 的关系：caller_option 是调用者的抽象选择，internal_variable 中的框架信号是该选择在 tiling 层的编码。两者描述同一件事的不同层面，但分类不同。

---

## S2P1_path_list.json 校验规则

1. `input_variables` 中的变量名必须是算子接口层的参数名，不能是内部派生量或框架信号
2. `caller_options` 中的变量名必须是调用者 API 层面的抽象选项名，不能是 tiling 内部编码变量
3. 每条路径的 `conditions` 不为空
4. 每条路径有 `source` 行号引用

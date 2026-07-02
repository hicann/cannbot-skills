# path-schema：路径清单与变量分类

对应 Task A 执行步骤 4：构建路径清单 + 变量三分类。

---

## 路径清单 JSON Schema

结构化 JSON 列表。每条路径 schema：

```json
{
  "id": "T1K1",
  "name": "描述性名称（涉及输入/输出/属性信息时须在属性值前写明参数名）",
  "conditions": [
    {"var": "tiling源码变量名", "op": "运算符", "value": "值"}
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

### 条件来源约束

每条路径的 `conditions` 只包含该路径**自身分支**的条件，来源限定为以下两类：

1. **当前 IsMatch 函数的正向条件**：该路径对应的策略匹配函数（如 `IsMatchX()`）内部使函数 return true 的所有条件
2. **当前 DoTiling / kernel dispatch 块内部的条件判断**：DoTiling 函数或 kernel dispatch 块中的 if-else 分支条件。同一 tiling key 内可能存在多条 kernel dispatch 分支（如不同模板参数、不同 submode），每条分支构成独立路径，其 conditions 需包含 kernel 侧的区分条件

**禁止**将同一 if-else 链中**前置兄弟分支**的内部条件（或其否定形式）加入当前路径的 conditions。原因：前置 IsMatch 函数 return false 只需其内部**任一条件**不满足（各条件间为 OR 关系），将所有条件取反后 AND 到当前路径会引入不存在的约束，导致路径被误判为 dead。

示例：
- **if-else 策略链**：对于 `if (IsMatchA()) { ... } else if (IsMatchB()) { ... }` 形式的策略链，IsMatchB 路径的 conditions 只提取 `IsMatchB()` 内部的 return true 条件，不提取 `IsMatchA()` 内部任何条件的否定
- **同一 tiling key 多 kernel 分支**：同一 IsMatch 函数命中后，kernel 侧按属性值、尺寸比较等条件 dispatch 到不同模板实例，每条 dispatch 分支作为独立路径，conditions = IsMatch 正向条件 + kernel dispatch 区分条件

### 路径 ID 命名规则（T*K*）

路径 `id` 采用 `T{t}K{k}` 格式，编码 tiling/kernel 两级分支结构：

| 路径类型 | ID 格式 | 说明 |
|----------|---------|------|
| 常规路径 | `T{t}K{k}` | t = tiling 分支序号（按 IsMatch 函数在策略链中的出现顺序从 1 递增），k = kernel dispatch 分支序号（同一 tiling 分支内从 1 递增） |
| 降级路径 | `T{t}K{k}d{d}` | 继承父路径的 T{t}K{k}，d = 降级序号（从 1 递增） |
| 孤儿 Dispatch | `D{N}` | 无对应 tiling 分支，N 从 1 递增 |

**命名约束**：
- 同一 tiling 分支（同一 IsMatch 函数命中）下的所有路径共享相同 `T{t}` 前缀
- 同一 tiling 分支内的不同 kernel dispatch 分支通过 `K{k}` 区分
- T 序号按 tiling 策略选择函数中分支的出现顺序分配
- K 序号按 kernel dispatch 块中分支的出现顺序分配

**分组含义**：共享相同 `T{t}` 前缀的 reachable 路径自动归入同一 group（详见 `02-step2-group.md`）。

---

## 命名规则

**路径 ID**：见上方 §路径 ID 命名规则（T*K*）。

**变量命名：tiling 源码变量名**。conditions 中 `var`/`ref`/`expr` 字段的变量名必须使用 tiling 源码中的原始变量名（如 `srcDim_`、`dtypeX_`），禁止自行发明语义化名称。
**例外**：当 dispatch 变量通过框架 API 管理（如 `SetTilingKey()` + `TILING_KEY_IS()`），不存在对应的源码局部变量名时，允许使用描述性名称（如 `tilingKey`），但须在 `S2P1_tiling_glossary.md` 中记录其框架 API 来源（`source` 列标注为 `framework_api`）。
变量含义表的格式和规则 → 步骤 4 时 Read `{skill_base}/references/task-a/05-step4-glossary.md`。

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
      1. 降级路径 `id` 在父路径 ID 后追加 `d{d}` 后缀（如 T5K1 → T5K1d1）
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
5. `S2P1_tiling_glossary.md` 已生成，且三分类列表中所有变量均有对应记录

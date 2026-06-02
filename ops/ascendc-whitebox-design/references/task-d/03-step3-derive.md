# Step 3：参数推导

> **前置条件**：Step 2 已完成，groups 已确定

读取 S2P1_path_list.json，根据 Step 2 写入的 `group` 字段获取每个 group 下的路径及其 conditions。**一次读 tiling 源码，完成以下全部子步骤**。

## 3.1 参数范围确定

为每个 group 的每个输入维度确定取值范围的约束条件。输入维度来自 `input_variables` 中影响 tensor 空间的参数——包含出现在 conditions 中的路由维度和不出现在 conditions 但映射到 tensor 元素的非路由维度（3.2 表格第 4 行）。维度名必须使用 tiling 源码中的变量名，禁止使用抽象分解名（如 `hidden_size`、`batch_size`）。

1. **下界**：来自 `source_constraints`，无显式下界则默认为 1
2. **上界**：来自 `source_constraints` 的显式上限。无显式上限的维度设实操上限（如 ≤ 65536），避免海量用例
3. **路由阈值**：该 group 触发/退出所对应的参数空间边界值。不同 group 间以该值为分界线
4. **对齐约束**：若源码对维度有对齐要求（必须为 N 的倍数，或不可为 N 的倍数），在此声明
5. **跨 group 互斥**：互补 group 的同类维度边界不重叠，min/max 衔接对齐。此规则已在 Step 2 规则 3 确立，此处为范围确定时的具体实施
6. **影响 tensor shape 的非路由维度**：识别属于 `input_variables`、映射到 tensor 元素的维度，但并非该 group 的路由维度（不决定 group 选择）。此类维度不改变 group 选择，但改变输入/输出空间，必须枚举测试。其值域可能因 routing 条件而异，需在取值时体现。

如何将上述范围条件转换为具体的取值列表写入 per_dtype，见 3.3；如何选取 group 级共享维度取值，见 3.3 步骤 6-8。

## 3.2 维度分类与取值路由

按维度类型分类，确定写入位置（per_dtype / group 级 `{dim}` / 默认值）。分类依据：是否出现在 conditions 中、是否影响 tensor 空间。

| 判定 | 条件 | 处理 |
|------|------|------|
| 路由维度 | 出现于 conditions | 纳入 per_dtype |
| 非路由维度 | 属于 `input_variables` 但不出现在 `conditions` 且不直接映射到 tensor 元素（如 epsilon 等标量属性） | 不纳入。在 S2P2_gen_cases.py 中统一设默认值 |
| 退化维度 | 出现在 conditions 但取值恒为单一常量 | 不纳入。在 `constraint_note` 中标注 |
| 影响 tensor shape 的非路由维度 | 属于 `input_variables`、映射到 tensor 元素，但并非该 group 的路由维度（不决定 group 选择） | 提升为 group 级 `{dim}` 字段，按 3.3 步骤 6-8 选取 10 个值 |

注：分类后的"处理"列仅指引写入位置，不表示维度被忽略。所有影响 tensor 空间的维度最终都会写入 JSON（per_dtype 或 group 级字段）。

**float 属性特殊值**：若为非路由维度，在 S2P2_gen_cases.py 中设默认值。若为路由维度，选取 `0.0`（零值边界）、负值（若源码允许）、极小正数（精度边界）。

**禁止**：将 `internal_variables` 中的纯计算中间变量作为 param 名。param 命名规则见 3.1。

**constraint_note 编写规则**：

- 只能引用当前 group 的 param 维度名（如 `{dim}`、`{dtype}`）和具体数值
- 禁止引用内部/中间变量名（如 tiling 源码中的 `{internal_var}`）
- 若约束来源于内部变量条件，必须将内部变量→param 的等价过程写入 S2P2_traceability.md 的「内部变量 → params 等价推导」表

## 3.3 取值推导

基于 3.2 的维度分类结果，分两阶段取值：

**阶段一（步骤 1-5）**：处理路由维度。对出现在 conditions 中且属于 input_variables 的维度，选取 per_dtype 取值列表写入。

**阶段二（步骤 6-8）**：处理影响 tensor shape 的非路由维度。对不出现在 conditions 但映射到 tensor 元素的维度，选取 group 级 `{dim}` 字段取值写入。

---

对每个 group，汇总该 group 内所有路径的 conditions（同一 group 内不同路径仅 dtype 不同，其余 conditions 共用）：

1. 提取 `conditions` 中引用的变量 → 确定该 group 的 params
2. 对 `internal_variables` 中的条件，读 tiling 源码回溯到 `input_variable`：
   - 识别内部变量 → 找到 tiling 源码中的赋值/计算位置
   - 追溯计算链（如 `{internal_var} = {formula}({input_var}, {core_count})` → `{input_var} ≤ {N}`）
   - 代入当前目标平台的常量计算等价边界值
   - 若计算链不经过任何 input_variable，跳过
3. 确认 3.1 确定的下界和上界，范围外的值不纳入取值列表
4. 根据 3.1 确定的范围约束，为每个 per_dtype 条目生成 5 个值的取值列表：[lo, mid₁, mid₂, mid₃, hi]。
   a. lo 取 group 内该维度的下界（含对齐约束）；hi 取 group 内该维度的上界（路由阈值或 source_constraints 上限）。
   b. mid₁~mid₃ 在 (lo, hi) 内跨量级均匀选取，优先 Po2 或对齐友好值。
5. 取值列表按 dtype 区分，写入 `per_dtype[{dtype}].{param}`
6. 对影响 tensor shape 的非路由维度（3.1 第 6 项），按 Step 2 的 group 划分逻辑确定其在各 group 中的取值区间（区间信息来自 constraint_note 或 tiling 源码中的隐式边界）。按以下规则写入：
   a. 取值区间互斥的 group → 各自独立选取 10 个值，区间不重叠。
   b. 取值区间相同或不受限的 group → 每个 group 也独立选取 10 个值，写入各自的 group 级字段。组间值可不同，增加多样性。不受限区间（如 lo 明确但 hi 未定）的上限按以下方式选取（取优先级较高者的较小值）：
    1) `source_constraints` 中该维度的显式上限；
    2) 所有 group 中最严格的路由阈值上界的 10 倍（如某 group 路由阈值为 {boundary}，则取 `{boundary} * 10`）；
    3) 3.1 第 2 项的实操上限（如 65536）。
    上限应尽可能小于实操上限，使 10 个值在 [lo, hi] 内有足够粒度。

7. 每 group 选取恰好 10 个值，固定比例 3:4:3（Po2 : Composite : Prime）。
   a. 先取区间边界值 lo 和 hi，按其数值类型（Po2/Composite/Prime）扣减对应类别的配额。
      （若 lo = hi，区间退化为单点，等同 3.2 表格退化维度。该值仅扣减一次配额，10 个值全部取 lo，不执行步骤 7b。）
   b. 剩余配额按 Po2 → Composite → Prime 的顺序逐类填满：
      i. 值在 [lo, hi] 区间内跨量级均匀分布，优先覆盖区间两端量级，剩余填入中间量级。
      ii. 若该类别配额 > 区间的量级数，允许同一量级内放置多个值；若配额 < 量级数，保证两端量级至少各有一个该类别值。
      iii. 禁止使用连续整数（如 1,2,3,4,5）。
   c. 所有值必须在 `source_constraints` 合法范围内。

8. 写入各自 group 的 `{dim}` 字段（不在 per_dtype 中）。各组独立取值，不做 group 间共享。

**示例**（通用推导过程）：
- conditions：`{dim} ≤ {threshold}`、`{internal_var} == {value}`、`{platform_var} ≠ {platform}`
- 回溯 `{internal_var} == {value}`：L 行号 `{internal_var} = {formula}` → 等价于 `{input_var} ≤ {boundary}`
- 回溯 `{platform_var} ≠ {platform}`：代入目标平台常量 → 恒满足 / 恒不满足 / 条件简化
- per_dtype 取值列表：每种 dtype 取 lo、mid₁~mid₃、hi，共 5 个值

## 3.4 计算 tiling_keys

从 tiling 源码中定位 `SetTilingKey(...)` 调用处，找到 key 编码公式。

1. 对每个 group 内各 dtype 的路径，根据其 conditions 变量值代回公式计算 key 值
2. **顶层 `tiling_keys`**：所有 reachable 路径的 key 去重排序
3. **per_dtype**：每个 dtype 条目的 key 值写入 `per_dtype[{dtype}].key`

多条路径映射到相同 key 值 → 去重。同一 key 可能出现在多个 group。计算完成后与 Step 2 的 key 覆盖校验结果交叉核对，不一致则回到对应 Step 修正。

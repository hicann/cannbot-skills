# 输出定义与禁止行为

> 本文件定义 S3_verification_report.md 的 Markdown 模板，规范各节的最低内容要求，并列出全文档禁止行为。

---

## 输出：S3_verification_report.md

完成所有检查后，将结果汇总写入 `S3_verification_report.md`。

### 完整模板

```markdown
---
op_name: {op_name}
platform: {platform}
status: pass | fail | pass_with_warnings
checks_total: 5
checks_pass: {N}
checks_fail: {N}
checks_warn: {N}
---

# 交叉验证报告：{op_name}

## 总览

| 项目 | 值 |
|------|---|
| 算子 | {op_name} |
| 平台 | {platform} |
| 全局状态 | **{status}** |
| 检查项总数 | 5 |
| pass 项数 | {N} |
| fail 项数 | {N} |
| warn 项数 | {N} |

## 检查结果

| ID | 检查名 | 状态 | 验证通过率 |
|----|--------|:----:|:----------:|
| A1 | source_reference_validity | ✅ pass | {v}/{t} |
| B1 | interface_factual_check   | ✅ pass_with_warnings | {v}/{t} |
| D1 | traceability_factual_check | ✅ pass | {v}/{t} |
| D2 | schema_compliance         | ✅ pass | — |
| D3 | gen_cases_script_semantic | ✅ pass | — |


## 各项详情

### A1: source_reference_validity — ✅ pass

**验证通过率**：{v}/{t}

{detail，满足 detail 最低内容要求}

---

...（依次 B1/D1/D2/D3，每项一节）

---

## Issues（共 {N} 项）

### ❌ Fail（{N} 项）

#### F1: {check_id}

- **问题**：{description}
- **期望值**：{expected}
- **实际值**：{found}
- **位置**：{location}

### ⚠️ Warn（{N} 项）

#### W1: {check_id}

- **问题**：{description}
- **调查发现**（可选）：{investigation_findings}
```

### YAML front matter 规范

| 字段 | 类型 | 必填 | 说明 |
|------|------|:----:|------|
| `op_name` | string | 是 | 算子名 |
| `platform` | string | 是 | 平台描述（如 `DAV_3510 (Ascend950PR)`） |
| `status` | string | 是 | `pass` / `fail` / `pass_with_warnings` |
| `checks_total` | int | 是 | 检查项总数（固定 5） |
| `checks_pass` | int | 是 | pass 项数 |
| `checks_fail` | int | 是 | fail 项数 |
| `checks_warn` | int | 是 | warn 项数（含 pass_with_warnings 中的 warn） |

> `status` 字段供 Step 2 回填/Step 4 用户确认等下游流程解析，是整个报告的机读入口。

---

## 检查结果表字段规范

| 列 | 适用检查项 | 必填 | 说明 |
|----|-----------|:----:|------|
| ID | 全部 | 是 | 检查项 ID（见 00-execution-order.md 检查项总览表） |
| 检查名 | 全部 | 是 | 英文 ID 或中文名均可 |
| 状态 | 全部 | 是 | `✅ pass` / `❌ fail` / `⚠️ warn` |
| 验证通过率 | 仅真实性类检查项 | 是 | 格式 `{verified_count}/{total_count}`；结构性检查写 `—` |

**真实性类检查项**（需要读源码做计数验证）：A1, B1, D1 — 写验证通过率

**结构性检查**（不存在"验证 N/M 项"的概念）：D2, D3 — 验证通过率列写 `—`

---

## 各项详情最低内容要求

按 Task 顺序排列。

| 检查 ID | detail 必须包含 |
|---------|---------------|
| **Task A** | |
| A1 `source_reference_validity` | A1.1 source 行号通过率 + A1.2 conditions 通过率（含 boundary_check warn 数）+ A1.3 key_instructions 通过率 + A1.4 source_constraints 通过率（含 variables warn 数） |
| **Task B** | |
| B1 `interface_factual_check` | B1.1 inputs 匹配数/总数 + B1.2 outputs 匹配数/总数（含 sync_with warn 数）+ B1.3 attributes 匹配数/总数（含 aclnn 独有属性 warn 数）+ B1.4 dtype 约束验证比对结果 + B1.5 outputs.shape 语义检查通过率 + B1.6 platform AddConfig 比对结论 + B1.7 结构完整性 warn 汇总 |
| **Task D** | |
| D1 `traceability_factual_check` | 触发条件表验证行数 + 推导链表验证行数；有行号偏差时注明偏差范围 |
| D2 `schema_compliance` | pass 时一句话；fail 时列出缺失/多余的字段名和位置 |
| D3 `gen_cases_script_semantic` | D3.1 JSON 加载 + `_a` 计算 + `_default_cap` 公式验证 + D3.2 extract 函数正确性 + D3.3 工具函数正确性 + D3.4 生成循环完整性 + seed 唯一性 + D3.5 case dict 结构 + D3.6 路径覆盖可达性 |


---

## Issues 表格式规范

Issues 按严重程度分为两个子节，**先 ❌ Fail，后 ⚠️ Warn**，每个 issue 用带编号的四级标题。

### Fail issue 必填字段

| 字段 | 格式 | 说明 |
|------|------|------|
| 编号 | `F1`、`F2`... | Fail issue 顺序编号 |
| check_id | 写在标题中 | 如 `#### F1: gen_cases_script_semantic` |
| 问题 | bullet | 一句话描述 |
| 期望值 | bullet | S2 产物声称的内容 |
| 实际值 | bullet | 源码中实际发现的内容（只写事实，不附加定性用语；如做了调查，可写调查发现的事实） |
| 位置 | bullet | 源码位置 `文件:行号`；无法定位时写 `未定位` |

> `⚠️ Warn` issue 只需 `问题` 一项，其余字段省略。

---

## 全局 status 判定规则

| 条件 | `status` 值 |
|------|------------|
| 所有检查项状态均为 pass | `pass` |
| 至少一个检查项状态为 fail | `fail` |
| 无 fail，至少一个检查项状态为 warn | `pass_with_warnings` |

---

## ✅/❌ 判断示例

代表性判例，占位符 `{...}` 对应实际算子中的具体标识符。

### Task A

- **A1 源码引用**
  - ✅ pass: 路径 `{path_id}` source `{file}:{line}` → 该处为 `{branch_expr}`，与路径模式一致
  - ❌ fail: 路径 `{path_id}` source `{file}:{line}` → 该行号为空行/注释，未找到分支逻辑
  - ❌ fail: source_constraints `{C_id}` source_location `{file}:{line}` 处实际表达式为 `{actual_expr}`，与 source_expr `{claimed_expr}` 不一致（operator 差异）

### Task D

- **D1 推导链**
  - ✅ pass: traceability 引用 `{file}:{line}` → 源码该处确实包含推导链声称的变量名和赋值逻辑
  - ❌ fail: traceability 引用 `{file}:{line}` 的运算符为 `>`，源码实际为 `>=`（运算符不一致）

- **D3 gen_cases.py**
  - ✅ D3.1 pass: py_compile 返回 0，且 `import random` + `random.seed(42)` 均存在
   - ❌ D3.4 fail: Section 4 中 group=`{group_A}` 和 group=`{group_B}` 的 shuffled_pool 调用均使用 seed=`{seed}`，不同 group 使用了相同 seed 值



---

## 禁止行为（全文档适用）

以下行为在整个 Step 3 执行期间严格禁止，违反任一条视为检查无效：

1. **禁止跳过真实性验证** — A1, B1, D1 中任一项不得因"源码太长"或"路径太多"而跳过
2. **禁止只看 S2 产物不看源码** — 真实性类检查项的核心就是读源码查证，禁止仅依赖产物内部一致性判断
3. **禁止将"语义近似"判定为 pass** — 如果推导链中的表达式与源码存在运算差异（如运算符方向不同、运算符精度不同），必须 fail
4. **禁止跳过 _def.cpp 核实** — B1 必须读 _def.cpp 文件，不得仅凭 operator_model 字段完整就判定 pass
5. **禁止通过运行 gen_cases.py 来验证 D3** — D3 必须 Read 脚本源码做静态语义匹配；运行脚本产出 cases.json 属于动态执行，会因 sampling/seed 的不确定性而无法断言。D3 的唯一合法验证方式是逐字节理解脚本逻辑，再与 param_def.json + path_list.json 做集合比对和语义匹配
6. **禁止对 D3.6 仅凭 cases.json 的存在就断言路径可达** — D3.6 必须分析脚本逻辑上是否有某个 group 的 dtype 条目引用目标 path ID；cases.json 中的 case 数量受压缩和采样策略影响，不能作为静态覆盖完整性证据
7. **禁止酌情处理** — 发现差异时，可以读取其他相关源码调查原因（包括检查项规则本身定义的分步追加读取，如 B1.4 第 5 步），并将调查发现的事实写入报告；但不得基于调查发现改变检查项规则定义的 pass/fail/warn 判定结果，不得用"语义等价"等理由合理化差异。判定结果必须由规则决定，不由 Step 3 酌情决定。详见 `00-execution-order.md` 核心原则节
8. **禁止 Step 3 添加定性判断用语** — 报告差异和调查发现时，只描述事实（"S2 声称 X，源码是 Y，调查发现 L{line} 处有 Z"），禁止附加 Step 3 自创的定性用语（如"运行时扩展"、"废弃常量"、"预留路径"、"动态注册"等）。以下两类用语不在此限：(1) 检查项规则本身的判定用语（如"语义一致"、"分类不一致"）；(2) 差异事实分类用语（如"运算符不一致"、"数值不匹配"）

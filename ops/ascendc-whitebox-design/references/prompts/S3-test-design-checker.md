# Test Design Checker — S2P2_param_def.json 交叉验证

> **执行顺序（最高优先级）**
> 严格按照以下步骤编号顺序执行。前置条件未满足禁止启动该步骤。
> 详细规则见本文件后续章节，执行顺序节未覆盖的细节以各章节为准。

1. 读 S2P2_param_def.json + S2P3_test_design.md + S2P1_path_list.json
   前置：无
2. 独立读 tiling 源码（关键分支/常量/模式函数）
   前置：Step 1 完成
3. 执行 14 项检查（values_traceable → schema_compliance） → checks 数组
   前置：Step 1-2 完成
4. 汇总 status（pass/fail/pass_with_warnings）+ issues
   前置：Step 3 完成
5. 写入 S3_verification_report.json
   前置：Step 4 完成

**完成标志**：S3_verification_report.json 已写入，含 status/checks/issues

## 角色

你是独立的交叉验证员。你的任务是检查 S2P2_param_def.json 和 S2P3_test_design.md 是否与算子源码一致。你没有参与 S2P2_param_def.json 的生成，需要从源码独立验证。

## CRITICAL: 不要信任分析者的报告

分析者可能遗漏分支、过度归纳、或误读源码。你必须独立验证每一项。

**不要：**
- 因为 S2P3_test_design.md 写得详细就默认它是对的
- 因为 S2P2_param_def.json 格式正确就跳过内容检查
- 因为 constraint_note 写得详细就跳过内容检查

**要做：**
- 自己读源码中的关键分支和常量
- 逐项对照 S2P2_param_def.json 的内容与源码
- 找出分析者遗漏的分支或约束

## 输入

1. `S2P2_param_def.json` — 待验证的参数定义
2. `S2P3_test_design.md` — 待验证的测试设计文档
3. `S2P1_path_list.json` — Agent A 的路径清单 + 源码约束表
4. `S2P2_traceability.md` — Task D 的内部变量等价推导报告（取值→源码行号的溯源桥梁）
5. 算子源码路径 — 用于独立验证

## Gate Function

```
各检查项必须全部通过才能整体 pass。
任一项 fail → 整体 fail → 回 Step 2 修正。
任一项 warn → 整体 pass_with_warnings → 可以继续但需记录。
```

## 检查清单

### 1. 取值可追溯性（values_traceable）
- 对 S2P2_param_def.json 中每个 group 的 `per_dtype` 取值列表：每个值能否追溯到 tiling 源码的边界条件或合法范围
- 边界值（如 min、max、阈值两侧）必须在 tiling 源码中有对应的分支判断或常量定义
- 取值列表的区间（min/max）是否在 `source_constraints` 定义的合法范围内

### 2. 测试关注点覆盖（groups_coverage）
- 源码中是否存在重要的分支/路由点没有被任何 group 覆盖
- 检查方法：读源码中的主要 if/switch/策略选择逻辑，看 S2P2_param_def.json 的 groups 是否覆盖了这些分支的关键条件
- 不要求每个 group 必须对应一条代码路径，但主要的风险场景应被覆盖

### 3. 维度值完整性（dimension_values_complete）
- 取值列表（`per_dtype.{dtype}.{dim}`）中的值是否覆盖了源码和 `source_constraints` 定义的有效值范围
- 是否有源码中的合法值被遗漏

### 4. constraint_note 正确性（constraint_note_correct）
- 每个 group 的 `constraint_note` 中引用的变量名是否全部是该 group 的 param 维度名（`per_dtype` 内的维度名和 group 级维度字段名），无内部/中间变量名
- `constraint_note` 中引用的具体数值与 `per_dtype` 取值列表的范围是否一致
- `constraint_note` 描述的约束逻辑与 tiling 源码的分支条件是否一致：不能比源码更严（排除 tiling 可达的合法值）、不能比源码更松（放过 tiling 不可达的非法值）

### 5. 取值范围匹配（value_ranges_match）
- 从每个 group 的 `per_dtype.{dtype}.{dim}` 取值列表推算该维度的有效区间（min 和 max），检查是否在 tiling 源码和 `source_constraints` 定义的合法范围内
- 对齐要求（如取值必须/不可为某数的倍数）是否与 tiling 源码一致
- 注意维度换算关系（取值列表中的值是否对应源码中的派生变量或输入维度）是否正确

### 6. 平台一致性（platform_consistency）
- `platform` 值是否与 S2P0_file_manifest.json 的 `platform.soc_version` 一致
- `platform_cores` 值是否与 S2P0_file_manifest.json 的 `platform.core_count` 一致
- 所有维度值是否来自该平台的源码
- dtype 组合是否属于该平台的 binary 注册
- 取值列表中的数值是否为目标平台的值

### 7. 执行模式覆盖（execution_mode_coverage）
- S2P3_test_design.md 中是否有"执行模式分析"节，包含轴映射表
- 分核轴是否有 `platform_cores` 值作为取值边界值
- 分核轴的范围是否覆盖 < coreNum、= coreNum、> coreNum
- UB 切分轴是否有足够大的 max（能触发多轮 UB loop）
- 指令对齐轴是否有向量宽度（BLOCK_ELEM / VL）作为对齐要求
- `axis_role` 标注是否与源码的实际分核/UB/指令逻辑一致

### 8. path_dimension_coverage

读取 `S2P1_path_list.json`，对每条路径：

- 取路径的 `group` 字段，找到 `S2P2_param_def.json` 中对应的 group
- 取路径的 `input_variables` 和 `caller_options` 列表
- 检查该 group 的 `per_dtype` 各 dtype 下定义的维度名和 group 级维度字段名是否覆盖了所有 `input_variables` 和 `caller_options`

**判定**：
- 路径的某个 input_variable 或 caller_option 在对应 group 中没有维度 → **fail**，列出缺失的变量和路径 id
- 所有路径的 input_variables 和 caller_options 都被覆盖 → **pass**

**关键**：你必须独立执行此检查。即使 Agent D 的一致性检查已经做过，你也要重做一遍，作为独立复核。

### 9. completeness_check

读取 `S2P1_path_list.json` 的 `completeness_checklist` 字段：

- 逐项检查 api_variants、format_variants、mode_variants、quant_variants、optional_input_combos、dispatch_coverage
- 对每个 status=missing 的项，检查 `S2P2_param_def.json` 中是否有对应的 group 或参数处理了该缺失
- 对 dispatch_coverage：如果 status=missing，还需确认 paths 数组中未分配 group 的 reachable 路径是否被 S2P2_param_def.json 的 groups 覆盖（正交维度作为同一 group 内参数差异处理的，视为已覆盖）
- 对 tiling_analysis：如果 status=delegated，确认 S2P3_test_design.md 第 1 节"tiling 代码"行未标注"是"

**判定**：
- status=missing 且 S2P2_param_def.json 未对应处理 → **fail**，列出缺失项和 evidence
- dispatch_coverage status=missing 且对应 kernel 调度分支未被任何 group 覆盖 → **fail**，列出未覆盖的 dispatch 条目
- 所有项 status=covered 或 na，或 missing 已被 S2P2_param_def.json 处理 → **pass**

### 10. cross_file_consistency

对 S2P3_test_design.md、S2P1_path_list.json、S2P2_param_def.json 三个文件做数值级交叉比对：

1. **路径数**：S2P3_test_design.md 第 3 节声明的路径总数 == S2P1_path_list.json paths 数组长度
2. **group 数**：S2P3_test_design.md 第 5 节的 group 数 == S2P2_param_def.json groups 数组长度
3. **group ID**：S2P3_test_design.md 中的 group id 集合 == S2P2_param_def.json groups 中的 id 集合
4. **路径覆盖**：S2P3_test_design.md 第 3 节列出的路径 ID 集合 ⊇ S2P1_path_list.json paths 中的 id 集合（允许 S2P3_test_design.md 额外标注 disputed 路径）
5. **关键参数值**：S2P3_test_design.md 第 2 节（事实摘要）和第 5 节（groups）中的 dtype、epsilon、ndim 等取值与 S2P2_param_def.json 对应 group 的 per_dtype 一致

**判定**：
- 路径数或 group 数不匹配 → **fail**，列出各文件的数值
- group ID 集合不一致 → **fail**，列出差异的 ID
- 路径 ID 有遗漏（S2P1_path_list.json 中的路径在 S2P3_test_design.md 中未出现）→ **fail**，列出缺失的路径 ID
- 关键参数值不一致（如 dtype 列表、epsilon 列表在两个文件间无交集）→ **fail**，列出具体差异
- 全部一致 → **pass**

**关键**：此检查确保 S2P3_test_design.md 是 S2P1_path_list.json 和 S2P2_param_def.json 的忠实摘要，而非主 Agent 自行简化或改写后的产物。

### 11. 公共函数溯源验证（public_function_traced）

检查 S2P1_path_list.json 中引用的公共函数（在当前算子目录下无定义的函数调用）是否溯源到了实现代码。

**检查方法**：
1. 从 S2P1_path_list.json 的 paths 中收集所有 `key_instructions` 和 `conditions` 中引用的函数名
2. 对每个在当前算子目录下无定义的函数，检查其 `source` 字段是否引用了实现文件（而非仅引用调用点）
3. 抽查至少 2 个平台判断函数或工具函数，独立验证其返回值是否与分析者的结论一致

**判定**：
- 存在未溯源的公共函数（source 仅引用调用文件，无实现文件引用）→ **fail**，列出未溯源的函数名
- 溯源结论错误（如某函数对目标平台的返回值与实现代码不符）→ **fail**，列出正确的返回值和源码证据
- 所有引用的公共函数均有实现文件溯源 → **pass**

**示例**：
```
✅ pass: IsRegbaseSocVersion 的 source 标注为 "tiling.cpp:415 → common/inc/op_host/tiling_util.h:24-29"，实现中 regbaseNpuArchs={DAV_3510,DAV_5102}，目标平台 DAV_2201 不在集合中，结论为返回 false，与路径 conditions 一致
❌ fail: IsRegbaseSocVersion 的 source 仅标注 "tiling.cpp:415"，未引用实现文件，且路径 conditions 假设其返回 true，未经验证
```

### 12. 路径可触发性验证（path_triggerability）

读取 S2P1_path_list.json 中所有 reachable 路径，对每条路径验证 S2P2_param_def.json 的取值列表是否能产生满足该路径 input_variable 和 caller_option 条件的值。

**检查步骤：**

1. 遍历 S2P1_path_list.json 中 `reachability == "reachable"` 的所有路径
2. 对每条路径，从 conditions 中筛选 **`var` 属于 `input_variables` 或 `caller_options`** 的条件（跳过 `var` 属于 `internal_variables` 的条件）
3. 对含比较运算符（`>`、`>=`、`<`、`<=`）的条件：
   - 根据路径的 `group` 字段，找到 S2P2_param_def.json 中对应 group 的 `per_dtype`，取出该维度的取值列表
   - 按路径的 dtype 条件确定对应的 dtype key（如 `DT_FLOAT16`），从 `per_dtype` 中取该 dtype 下的取值列表
   - 检查取值列表中**是否存在满足条件的值**
4. 对含枚举匹配的条件（如 `{dtype_param} == DT_FLOAT16`）：
   - 检查 S2P2_param_def.json 对应 group 的 `per_dtype` 的 key 集合中是否包含该枚举值

**跳过规则：**
- `var` 属于 `internal_variables` 的条件 → 跳过（内部变量不直接作为 param 维度）
- `var` 不属于该 group `per_dtype` 中定义的维度名 → 跳过（非测试维度，由 group 分组隐式覆盖）
- 路径无 input_variable 和 caller_option 相关条件 → 跳过该路径

**判定：**
- 路径的所有 input_variable 和 caller_option 条件都能被 per_dtype 取值列表满足 → **pass**
- 某个条件无法满足（取值列表中无满足条件的值）→ **fail**，列出路径 ID、不可满足的条件、缺失的值建议

#### 12b. dtype 调度一致性验证

在 12（路径可触发性）基础上，额外检查每条路径的 dtype 条件与源码中该路径对应的 tiling 调度逻辑是否一致。tiling 源码中某些 mode 的分支条件可能**隐式限制**了 dtype（如 `if (dataType == DT_FLOAT16)` 写死在 mode 判断中），但 Task A 在提取 conditions 时可能遗漏了这个限制。

**检查步骤：**

1. 对每条 reachable 路径，从路径的 `source` 字段定位到 tiling 源码中的对应分支
2. 读取该分支的调度条件代码，检查是否存在对 dtype 的**显式限制**（如 `dataType == ge::DT_FLOAT16`）
3. 将源码中的 dtype 限制与路径 conditions 中的 dtype 条件对比：
   - 如果源码仅允许特定 dtype（如仅 FP16），但路径 conditions 中包含其他 dtype（如 BF16）→ 该路径对非 FP16 dtype 不可达
   - 如果源码对 dtype 无额外限制 → 无问题

**判定：**
- 路径的 dtype 条件与源码调度逻辑一致 → **pass**
- 路径的 conditions 包含源码调度逻辑不支持的 dtype → **fail**，列出路径 ID、源码限制、建议将路径标记为 `unreachable` 或修正 conditions

**注意：** 此检查需要 agent 读 tiling 源码中负责 mode/key 分支调度的函数（由 Phase 0 侦察或代码分析阶段识别），验证每个 mode_key 分支的 dtype 限制。读取范围仅限 Phase 0/1 已确认的平台可达 tiling 入口、mode/key 调度函数和其直接依赖的分支判断函数。不得重新打开 Phase 0 已排除的平台不可达文件；如需读取未列入范围的符号，记录为未确认项并回 Step 2 补充清单。

### 13. 参数空间覆盖完整性（coverage_gaps）

检查 S2P2_param_def.json 的所有 groups 的约束联合后，是否存在未被任何 group 覆盖的参数空间区域（覆盖空洞）。

**检查步骤**：

1. 根据每个 group 的 `per_dtype` 取值列表范围和 `constraint_note` 描述，推断该 group 覆盖的参数空间区域
2. 识别所有 group 共有的维度（如 {dtype_param}、{dim_a}、{dim_b}）
3. 对每个维度，找出所有 group 覆盖的区域，检查是否存在间隙：
   - 离散维度（dtype 枚举）：检查所有合法值的并集是否等于源码约束定义的合法域
   - 连续维度（取值列表）：检查所有 group 的 min-max 区间是否存在未覆盖的间隙
4. 特别关注：降级边界值附近是否被至少一个 group 覆盖

**判定**：
- 存在未被任何 group 覆盖的合法参数区域 → **warn**，列出间隙的维度和范围，建议补充 group 或调整取值
- 特别关注降级边界：如果 S2P1_path_list.json 中存在 `boundary_check` 字段但对应的降级区间未被任何 group 覆盖 → **fail**
- 所有合法参数区域均被覆盖 → **pass**

### 14. schema 合规性（schema_compliance）

验证 `S2P2_param_def.json` 的结构是否符合定义。

**检查步骤：**

1. 读取 S2P2_param_def.json 的所有顶层 keys
2. 检查**必填顶层字段**是否都存在：
   - `platform`（string）：目标平台标识
   - `platform_cores`（int）：目标平台核数
   - `tiling_keys`（int[]）：所有 reachable 路径的 tiling key（去重排序）
   - `groups`（array）：分组参数定义
3. 对每个 group，检查**必填 group 字段**：
   - `id`（string）：group 标识
   - `mode`（string）：触发模式描述
   - `per_dtype`（dict）：按 dtype 绑定 path/key/取值
   - `constraint_note`（string）：约束说明
4. 对每个 `per_dtype` 的 value，检查**必填字段**：`path`（string）、`key`（int）
5. 检查禁止字段。禁止字段列表以 `references/task-d/04-step4-output.md` 为唯一来源。Step 3 只检查 S2P2_param_def.json 是否违反该文件当前维护的列表，不在本 prompt 内复制字段名。

**判定：**
- 缺少任一必填字段 → **fail**，列出缺失的字段名和位置
- 存在禁止字段 → **fail**，列出多余的字段名和位置
- 全部合规 → **pass**

**判定示例：**
```
✅ pass: 顶层 keys = {"platform", "platform_cores", "tiling_keys", "groups"}，group 内字段合规，无禁止字段
❌ fail: 缺少必填顶层字段 "tiling_keys"
❌ fail: group "split_d" 存在禁止字段 "group_tilingkeys"
```

## ✅/❌ 判断示例

```
✅ pass: 取值列表 {dim}=[{VALUE_A}, {VALUE_B}] 覆盖了 {group} 边界值和代表值，与源码阈值常量 {THRESHOLD} 一致
❌ fail: {group_a} group 的 {dtype} 下 {dim} 取值列表中包含 {VALUE}（满足某源码条件的值），根据源码该值应触发 {group_b} 而非 {group_a}
✅ pass: {dtype_param} 取值覆盖所有支持的 dtype（{DTYPE_A}、{DTYPE_B}、{DTYPE_C}），与源码一致
❌ fail: constraint_note 中出现 "{internal_var}"——内部变量名，应替换为参数维度名的具体数值范围
```

## 禁止行为

- 禁止"看起来合理就 pass"——每一项都需要源码证据
- 禁止跳过 constraint_note 检查
- 禁止只看 S2P3_test_design.md 不看 S2P2_param_def.json——两个都要验证

## 输出

写入 `S3_verification_report.json`：

```json
{
  "status": "pass|fail|pass_with_warnings",
  "checks": [
    {
      "id": "values_traceable",
      "status": "pass|fail|warn",
      "detail": "具体发现，包含源码行号引用"
    }
  ],
  "issues": [
    "具体问题描述（仅 fail/warn 时）"
  ]
}
```

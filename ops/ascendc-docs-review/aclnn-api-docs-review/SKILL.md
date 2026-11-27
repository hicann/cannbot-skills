---
name: aclnn-api-docs-review
description: |
  aclnn 算子 API 说明文档质检工具。用于检查算子 API 文档的质量，基于预设规则检测文档中的问题。
  当用户提到以下场景时触发：
  - 检查/质检/审核/验证 aclnn 算子 API 文档质量
  - 检查/质检/审核文档是否符合 aclnn 文档规范
  - aclnn 文档内容与代码一致性检查
  - aclnn 文档规范性、完备性、可读性、正确性检查
  - 检查 aclnn 文档产品标签（npu/id 配对）、芯片差异化描述、Batch一致性说明
  - 审查/检查 aclnnxxx.md 文档
version: 1.1.0
metadata:
  author: Huawei Technologies Co., Ltd.
  category: utility-doc
---
# aclnn 算子 API 说明文档质检 Skill

本 skill 用于对算子 API 说明文档进行质量检查，基于预设规则检测文档中的规范性、完备性、可读性、正确性问题。

## 重要原则（必须严格遵守，防止幻觉）

### 1. 仅使用给定规则

- **只能**使用 `references/aclnn-rules-{1,2,3,4}.md` 中明确定义的规则
- **严禁**创造新规则、修改规则名称或规则级别
- 规则名称必须与规则文件中的完全一致（如"产品型号命名规范"）

### 2. 代码即真相（正确性检查）

- 对于"正确性"类别的检查，**必须**先读取代码文件再进行判断
- **禁止**仅根据文档内容推测代码行为
- 如果无法从代码中找到明确证据，**不要报告该问题**

### 3. 每项检查后追加写入 JSON 文件

- **每项检查完成后**：立即将结果追加写入 JSON 文件
- **质检期间**：禁止输出任何解释性文字、总结、表格
- **最终输出时**：保证输出到文件，无需输出到控制台
- 输出格式**只能**是标准 JSON

### 4. 输出前验证最终质检结果

- **只报告**真实存在问题的检查项
- 无问题，文档与代码一致的检查项**严禁报告**

### 5. 多子项规则独立报告原则

- 当质检规则包含多个子项（如"API 函数原型格式规范"包含8项检查）时，**每个子项的问题必须独立报告**
- **禁止**将同一规则的多个问题合并为一个问题项
- 每个问题位置需要独立的 line_number 和 snippet
- 示例：如果"产品型号命名规范"在文档中有3处产品名称错误，应报告独立的问题项，每个指向不同的 line_number

## 输入

1. **`-doc` 待检查的文档**：用户输入需要检查质量的 API 文档文件（.md 格式）
2. **`-type` 质检类别**：指定需要执行的质检类别
   - `1` 规范性检查
   - `2` 完备性检查
   - `3` 可读性检查
   - `4` 正确性检查
   - `all` 全部检查（默认）
   - 多类别用逗号分隔，如 `-type 1,2,4`

### 示例

```
# 全部检查
/aclnn-api-docs-review -doc "doc.md" -type all
或
/aclnn-api-docs-review -doc "doc.md"

# 仅规范性
/aclnn-api-docs-review -doc "doc.md" -type 1

# 规范性 + 完备性
/aclnn-api-docs-review -doc "doc.md" -type 1,2
```

## 输出

输出两个文件（同目录，位于本 skill 目录下 `evals/results/`，与 `evals/dataset/` 平级；目录不存在则先创建）：

1. **质检结果 JSON 文件**（机器可读）
   - 默认输出目录：本 skill 目录下 `evals/results/` （运行产物与评测数据集同域，不随执行目录散落）
   - 文件命名规则：`{被检查文档名}-qc-result.json`
2. **质检报告 MD 文件**（人读报告）
   - 同目录输出：`{被检查文档名}-qc-report.md`
   - 内容：质检概要、问题分布、按类别分组的问题明细（含行号定位链接与修改建议）

> 运行产物默认不入库：`evals/results/` 及 bench 报告由 `evals/.gitignore` 忽略；首次基线能力数据报告存档于 [evals/baseline/](./evals/baseline/README.md)。

### 示例

- 待检查的文档 -doc：`/Users/aclnn/docs/aclnnAbs.md`
- 输出质检结果JSON文件到：`<skill目录>/evals/results/aclnnAbs-qc-result.json`
- 输出质检报告MD文件到：`<skill目录>/evals/results/aclnnAbs-qc-report.md`

## 执行流程

### 核心原则：按需读取，逐任务执行

**严禁提前批量读取所有文件。** 每个任务在执行时才读取该任务所需的文件，执行完毕后立即写入结果，再进入下一个任务。

### 第 1 步：初始化

1. 打印进度：`[CHECKPOINT: 开始质检]`
2. 读取[文档结构参考](./references/aclnn-doc-structure.md)了解基本结构
3. 读取用户提供的待检查文档文件
4. 在输出目录创建初始 JSON 结果文件：
   - 初始内容：`{"result": "pending", "details": [], "skipped": []}`

### 第 2 步：解析类别并创建动态任务列表

打印进度：`[CHECKPOINT: 前置条件处理完成]`
根据 `-type` 参数解析需要执行的质检类别：

| 类型参数       | 对应任务     |  说明                | 是否需要代码 |
| ------------- | ---------- |--------------------- | ---------- |
| `1` 或包含 `1` | 规范性检查   | 检查文档格式、命名规范等 | 否         |
| `2` 或包含 `2` | 完备性检查   | 检查文档内容是否完整    | 否         |
| `3` 或包含 `3` | 可读性检查   | 检查文档表达是否清晰易懂 | 否         |
| `4` 或包含 `4` | 正确性检查   | 检查文档与代码一致性    | 是         |
| `all`或未提供  | 全部四项检查  | 全部四项检查          | 是         |

**创建动态任务（Todo）列表**：

根据解析结果创建对应的任务项，展示任务列表和进展。

检查项规则级别说明

| 级别 | 说明                             |
| ---- | -------------------------------- |
| 严重 | 必须修复，否则会导致用户使用错误 |
| 一般 | 建议修复，影响文档质量和用户体验 |
| 提示 | 可选修复，属于优化建议           |

### 第 3 步：逐任务执行

**严格按动态任务列表顺序依次执行，每个任务完整执行完毕（含读取文件、检查、写入结果）后，才进入下一个任务。禁止跨任务并行读取文件。**

##### 如果包含类别 1（规范性检查）

- Task #1 规范性检查
  a. 打印进度：`[CHECKPOINT: 开始规范性检查]`
  b. **此时读取**规则文件 `references/aclnn-rules-1.md`，以及规则中引用的参考资料（如 `aclProductList.md` 等，仅在规则需要时读取）。数据类型和数据格式分别按需读取 ops-math 官方[数据类型](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_type.md)和[数据格式](https://gitcode.com/cann/ops-math/blob/master/docs/zh/context/data_format.md)源文件。错误码按需读取 opbase 官方头文件 [op_errno.h](https://gitcode.com/cann/opbase/blob/master/include/nnopbase/opdev/op_errno.h)。
  c. 基于已读取的文档和规则，执行全面规范性检查
  d. 将检查结果追加写入 JSON 文件：读取当前文件内容，合并新发现的问题后重新写入
  e. 打印进度：`[CHECKPOINT: 完成规范性检查]`
- 要求：仅检查文档格式，不需要读代码

##### 如果包含类别 2（完备性检查）

- Task #2 完备性检查
  a. 打印进度：`[CHECKPOINT: 开始完备性检查]`
  b. **此时读取**规则文件 `references/aclnn-rules-2.md`，以及规则中引用的参考文件（仅在规则需要时读取）
  c. 基于已读取的文档和规则，执行全面完备性检查
  d. 将检查结果追加写入 JSON 文件：读取当前文件内容，合并新发现的问题后重新写入
  e. 打印进度：`[CHECKPOINT: 完成完备性检查]`
- 要求：仅检查文档内容完备性，不需要读代码

##### 如果包含类别 3（可读性检查）

- Task #3 可读性检查
  a. 打印进度：`[CHECKPOINT: 开始可读性检查]`
  b. **此时读取**规则文件 `references/aclnn-rules-3.md`，以及规则中引用的参考文件（仅在规则需要时读取）
  c. 基于已读取的文档和规则，执行全面可读性检查
  d. 将检查结果追加写入 JSON 文件：读取当前文件内容，合并新发现的问题后重新写入
  e. 打印进度：`[CHECKPOINT: 完成可读性检查]`
- 要求：仅检查文档表达清晰度，不需要读代码

##### 如果包含类别 4（正确性检查）

- Task #4 正确性检查
  a. 打印进度：`[CHECKPOINT: 开始正确性检查]`
  b. 查找**文档对应算子的代码目录**：仅在文档所属的算子文件夹内查找，不向算子文件夹之外扩大搜索范围。对于 `<算子目录>/docs/` 下的文档（含其子目录），以该 `docs/` 的父目录为算子目录；例如 `math/abs/docs/aclnnAbs.md` 只在 `math/abs/` 内查找 `op_api/`、`op_host/`、`op_kernel/` 等代码目录，并通过文档中的 API 名确认对应关系。无法确定所属算子文件夹或找不到对应代码时，将 `{"category": 4, "reason": "具体跳过原因"}` 追加写入结果文件的 `skipped` 数组，再跳过正确性检查并打印进度结束。
  c. **此时读取**规则文件 `references/aclnn-rules-4.md`
  d. **此时读取**[代码结构参考](./references/aclnn-code-structure.md)
  e. **此时读取**已确认算子根目录下 `op_api/`、`op_host/`、`op_kernel/`、`examples/`、`tests/` 等实际存在目录中的相关代码文件（如 `aclnn_xxx.h`、`aclnn_xxx.cpp`、`test_aclnn_xxx.cpp` 等，按规则需要逐个读取）
  f. 基于已读取的文档和代码，执行全面正确性检查
  g. 将检查结果追加写入 JSON 文件：读取当前文件内容，合并新发现的问题后重新写入
  h. 打印进度：`[CHECKPOINT: 完成正确性检查]`
- 要求：必须有文档对应算子的代码目录才能读取代码文件，从而验证文档与代码的一致性。

### 第 4 步：输出最终质检结果

1. 先根据 `details` 和 `skipped` 更新 JSON 文件中的 `result`：仅当两个数组都为空时为 `pass`，否则为 `fail`，最终结果不得保留 `pending`
2. 基于最终 JSON 结果生成质检报告 MD 文件（格式见"质检报告 MD 格式"），与 JSON 文件同目录
3. 打印进度：`[CHECKPOINT: 质检结束]`

## JSON 文件操作说明

- **准备阶段**：创建空 JSON 文件 `{"result": "pending", "details": [], "skipped": []}`
- **每项检查后**：追加新发现的问题到当前输出文件
  - 同一规则的多个子项问题需分别追加，每个问题作为独立的问题项
- **检查被跳过时**：立即将类别编号（1–4）及具体原因写入 `skipped`；仅记录 `-type` 请求执行但未完成的类别，未请求的类别不记录。跳过信息不写入 `details`。
- **全部检查结束后**：将 `result` 从 `pending` 更新为最终状态：仅当 `details` 和 `skipped` 都为空时为 `pass`，否则为 `fail`；写入文件后打印进度：`[CHECKPOINT: 质检结束]`

### 质检结果统一 JSON 格式

**有问题时输出（单问题）：**

```json
{
  "result": "fail",
  "details":[
    {
      "rule_name": "产品型号命名规范",
      "rule_type": "规范性",
      "file_name": "docs/aclXxx.md",
      "line_number": 10,
      "snippet": "- Ascend 950PR&950DT系列产品：支持",
      "description": "产品名称缺少 term 标记",
      "suggestion": "- <term>Ascend 950PR&950DT系列产品</term>：支持",
      "rule_level": "严重"
    }
  ],
  "skipped": []
}
```

**请求的检查全部完成且无问题时输出：**

```json
{
  "result": "pass",
  "details":[],
  "skipped": []
}
```

**检查被跳过且未发现文档问题时输出：**

```json
{
  "result": "fail",
  "details": [],
  "skipped": [
    {"category": 4, "reason": "未找到文档对应的算子代码"}
  ]
}
```

#### 输出字段说明

| 字段名              | 类型   | 描述                                                     | 要求                     |
| ------------------- | ------ | -------------------------------------------------------- | ------------------------ |
| result              | string | 质检结果：pass（请求的检查全部完成且无问题）/ fail（发现问题或有请求的检查未完成）                   | 必须                     |
| skipped             | array  | 跳过的类别及原因，无跳过时为 [] | 必须 |
| skipped.category    | number | 请求执行但未完成的类别编号（1–4） | 必须 |
| skipped.reason      | string | 该类别未完成的具体原因 | 必须 |
| details             | array  | 问题对象数组，无问题时为空数组 []                        | 必须                     |
| details.rule_name   | string | 规则名称                                                 | 必须与规则文件中完全一致 |
| details.rule_type   | string | 规则类别：规范性、完备性、正确性、可读性                 | 必须与规则文件中一致     |
| details.file_name   | string | 带被测文档路径文件名，从根目录开始，例如：docs/aclXxx.md | 必须准确                 |
| details.line_number | number | 问题所在文档的行号                                       | 必须准确                 |
| details.snippet     | string | 问题在文档中的具体片段                                   | 直接引用文档内容         |
| details.description | string | 问题的具体描述                                           | 清晰说明问题             |
| details.suggestion  | string | 问题的具体修改建议                                       | 直接给出正确形式         |
| details.rule_level  | string | 规则严重等级：提示、一般、严重                           | 必须与规则文件中一致     |

### 质检报告 MD 格式

基于最终 JSON 结果生成 `{被检查文档名}-qc-report.md`，结构如下：

```markdown
# {被检查文档名} 质检报告

- 质检时间：{时间}
- 检查引擎：DomainCheck_aclnn_API
- 质检类别：{实际执行的类别，如 all / 1,2}
- 质检结论：{pass / fail}（共 {N} 个问题，{M} 个类别跳过）

## 1. 问题分布

| 类别 \ 级别 | 严重 | 一般 | 提示 | 小计 |
| --- | ---: | ---: | ---: | ---: |
| 规范性 | x | x | x | x |
| 完备性 | x | x | x | x |
| 可读性 | x | x | x | x |
| 正确性 | x | x | x | x |
| **合计** | **x** | **x** | **x** | **N** |

- 跳过类别：{category: reason 列表；无跳过时写"无"}

## 2. 问题明细

### 规范性（x 项）

| 序号 | 规则名称 | 行号 | 问题片段 | 问题描述 | 修改建议 | 级别 | 定位 |
| --- | --- | ---: | --- | --- | --- | :-: | --- |
| 1 | xxx | 10 | xxx | xxx | xxx | 严重 | [L10](vscode://file/{文档绝对路径}:10) |

（完备性、可读性、正确性各一个小节，格式相同；问题按行号升序排列；无问题的类别写"无"）

## 3. 级别说明

- 严重：必须修复，否则会导致用户使用错误
- 一般：建议修复，影响文档质量和用户体验
- 提示：可选修复，属于优化建议
```

## Bench 验证（能力变更后必须执行）

**触发条件**：任何对**本 skill 能力**的修改，完成后必须提示用户执行 bench 验证，量化确认能力未劣化。修改范围包括：

- `references/aclnn-rules-{1,2,3,4}.md` 规则文件（新增/修改/删除规则、调整规则级别）
- `references/aclnn-doc-structure.md`、`references/aclnn-code-structure.md` 结构参考文件
- 本 `SKILL.md` 的执行流程、输入输出、判定逻辑
- `evals/dataset/` 待测数据集、`evals/aclnn_bench.md` golden 数据

**验证维度**：① golden 问题命中率（skill 是否查得全、查得准）；② skill 执行耗时。

### 数据说明

| 数据 | 路径 | 说明 |
| ---- | ---- | ---- |
| 待测试数据集 | `evals/dataset/` | 缺陷注入样例集。核心样例：`evals/dataset/random/drop_out_v3/docs/aclnnDropoutV3.md`，其算子目录（含 `op_host/`）与 `docs/zh/context/` 参考资料随数据集提供 |
| golden 数据 | `evals/aclnn_bench.md` | 已知问题清单（问题总数与分布见文件开头"问题分布"表），每条问题含：问题类型、位置行号、问题上下文、问题描述、严重程度 |

### 执行步骤

1. **记录开始时间**：打印 `[CHECKPOINT: 开始质检]` 时记录时间戳
2. **按标准质检流程执行**（不改变任何检查逻辑）：
   ```
   /aclnn-api-docs-review -doc "evals/dataset/random/drop_out_v3/docs/aclnnDropoutV3.md" -type all
   ```
   输出质检报告：`evals/results/aclnnDropoutV3-qc-result.json`
3. **记录结束时间**：`[CHECKPOINT: 质检结束]` 时记录时间戳，计算总耗时及各类别耗时（各类别 CHECKPOINT 时间差）
4. **读取 golden 数据**：逐条解析 `evals/aclnn_bench.md` 中每个"问题N"块的问题类型、位置行号、问题上下文、问题描述、严重程度
5. **命中比对**：对每条 golden 问题，在质检报告 `details` 中按"命中判定标准"查找是否存在命中条目
6. **计算指标，输出命中报告**

**顺序约束**：必须先执行质检、后读取 golden 数据，防止针对 golden 调整检查行为；禁止为提高命中率对数据集内容做任何特判或硬编码。

### 命中判定标准

一条 golden 问题记为**命中**，当且仅当质检报告 `details` 中存在一条问题**同时满足**以下三条：

1. **问题类型一致**：`rule_type` 与 golden 的问题类型一致（规范性/完备性/正确性/可读性）
2. **行号一致**：`line_number` 与 golden 的位置行号一致
3. **上下文匹配**：`snippet` 与 golden 的问题上下文在去除首尾空白与缩进后完全相同，或互为子串

任一条不满足即记为**未命中**，并标注未命中原因（无该行号问题/问题类型不符/上下文不符）。`rule_level`（严重程度）不参与命中判定，但单独统计命中条目的级别一致数。

反向统计**误报**：质检报告 `details` 中不能对应任何一条 golden 问题的问题，记为误报（衡量"查得准"）。若某类别被跳过（记录在 `skipped`），该类别 golden 问题记为"类别跳过未命中"，单独标注，不计入检出能力缺陷。

### 指标计算

| 指标 | 公式 | 参考基线 |
| ---- | ---- | ---- |
| 命中率（查全） | 命中 golden 问题数 / golden 总数 | ≥ 90% 通过，低于需逐条分析 misses |
| 精准率（查准） | 命中 golden 问题数 / 质检报告问题总数 | ≥ 70% 通过，误报过多需检查规则误判 |
| 执行耗时 | 总耗时（秒），及规范性/完备性/可读性/正确性各类别耗时 | 与上一次 bench 结果对比，无显著劣化 |

> **附加观察项**（不列入核心指标）：级别一致率 = 命中条目中 rule_level 与 golden 严重程度一致的条数 / 命中条数（≥ 80% 视为健康），衡量定级准确性，不参与命中判定，在报告中以附加观察形式呈现。

### 命中报告

报告输出到 `evals/bench-hit-report.json`，并向用户返回摘要：

```json
{
  "run_at": "2026-11-26 10:00:00",
  "dataset": "evals/dataset/random/drop_out_v3/docs/aclnnDropoutV3.md",
  "golden": "evals/aclnn_bench.md",
  "duration_seconds": 300,
  "category_duration_seconds": {"规范性": 90, "完备性": 60, "可读性": 50, "正确性": 100},
  "golden_total": 52,
  "hit_count": 47,
  "hit_rate": 0.90,
  "report_total": 50,
  "false_positive_count": 3,
  "precision": 0.94,
  "level_consistency": 0.96,
  "hits": [
    {"bench_issue": 1, "rule_type": "规范性", "line_number": 1, "issue": "一级标题不规范", "matched": {"rule_name": "xxx", "line_number": 1, "snippet": "# aclnndropoutV3"}, "level_match": true}
  ],
  "misses": [
    {"bench_issue": 33, "rule_type": "规范性", "line_number": 14, "issue": "产品差异化描述未被产品标签包裹", "miss_reason": "报告中无行号14的问题"}
  ],
  "false_positives": [
    {"rule_name": "xxx", "rule_type": "规范性", "line_number": 100, "description": "xxx"}
  ]
}
```

报告完成后向用户汇报核心指标：**命中率、精准率、总耗时与各类别耗时**；级别一致率作为附加观察项一并呈现（不列入核心指标）。并列出未命中（misses）与误报（false_positives）清单及原因，作为规则修复或流程优化的输入；本次 bench 验证结果（命中率、耗时）需写入变更记录。历史基线存档见 [evals/baseline/](./evals/baseline/README.md)，**命中率不得低于基线、耗时不得显著劣化**方可合入能力变更。**运行产物（`evals/results/`、`evals/bench-hit-report.*`）已由 `evals/.gitignore` 忽略，禁止提交入库**；如需更新基线存档，须先与基线对比并经确认后复制至 `evals/baseline/`。

## 常见幻觉陷阱（避免！）

| 陷阱                 | 错误做法                                       | 正确做法                                             |
| -------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| 未读代码下结论       | 根据文档推测函数原型                           | 读取 `aclnn_xxx.h` 头文件确认                        |
| 规则名称错误         | 使用"函数原型格式"                             | 使用规则文件中的"API 函数原型格式规范"               |
| 创造规则             | 添加规则文件中不存在的检查项                   | 仅使用规则文件中明确定义的规则                       |
| 输出多余内容         | 输出总结、表格、解释                           | 仅输出 JSON 结果文件与质检报告 MD 文件，不在对话中输出总结、表格、解释 |
| 合并多子项问题       | 将同一规则的多处问题合并为一个问题项        | 每个问题位置独立报告，使用独立的行号     |
| 遗漏其他位置的错误   | 只报告第一处错误，忽略同一规则的后续错误       | 全面检查文档，同一规则的所有问题都要独立报告         |
| 针对 bench 特判       | 为提高命中率对数据集内容硬编码、先读 golden 再调整检查行为 | 按标准流程先执行质检后比对 golden，规则保持通用       |

## 注意事项

1. 本 skill 只关注文档内容的质量检查，默认代码是正确的
2. 只有当代码目录存在，才能进行正确性检查
3. 输出报告要具体到文档的行号，便于用户定位和修改
4. **仅使用给定规则文件的范围进行检查，严禁增删改规则条目**
5. **对于任何正确性问题，必须先读取代码验证才能报告**

## 变更记录

- **1.1.0**：规则与评测体系升级。规则：类别1新增"芯片标签规范"（严重）与"芯片差异化描述规范"分档定级（键值对格式=一般，表格/代码段内=严重）；类别2新增"Batch一致性说明缺失"（可选），确定性说明为必选、话术含义类似即可，拦截表字段改为"返回码或者错误码、码值、描述"；类别3固定话术按语义相似度≥70%匹配，`aclrtSetSysParamOpt` 须逐字精确（由 `aclrtCtxSetSysParamOpt` 更正）；规则类别统一为"完备性"。评测：新增"Bench 验证"流程（命中判定=问题类型+行号+上下文；核心指标：命中率/精准率/耗时），golden 重整为 52 条；新增质检报告 MD 输出、`evals/results/` 输出目录与 `evals/baseline/` 基线存档，运行产物经 `.gitignore` 不入库。bench 基线：命中率 100%（52/52）、精准率 77.6%、耗时 584s。
- **1.0.0**：初始版本。

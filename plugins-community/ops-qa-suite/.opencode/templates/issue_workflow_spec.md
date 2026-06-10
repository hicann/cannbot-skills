# Issue 创建流程规范

## 一、目录结构规范

### 1.1 统一目录结构

**按日期+仓库层级组织**，便于追踪历史和区分不同仓库的扫描结果：

```
reports/
├── 20260512/                                     # 日期目录 (YYYYMMDD)
│   ├── ops-cv/                                   # 仓库目录
│   │   ├── repo-docs-scan_report_143058.md       # 文档质量扫描报告
│   │   ├── op-doc-completeness_report_143159.md  # 算子文档完备性报告
│   │   ├── cmake-scan_report_143200.md           # CMake 配置扫描报告
│   │   ├── ut-analysis-guide_report_143230.md    # UT 缺失分析报告
│   │   ├── ut-test-report_report_143300.md       # UT 测试执行报告
│   │   ├── examples-analysis-guide_report_143330.md # Examples 缺失报告
│   │   ├── examples-test-report_report_143400.md # Examples 测试报告
│   │   ├── op-list-validation_report_143430.md   # 算子列表一致性报告
│   │   ├── op-api-list-validation_report_143500.md # 接口列表一致性报告
│   │   ├── issues/                               # Issue 文件目录
│   │   │   ├── readme_missing_issue_143159.md    # README 缺失 Issue
│   │   │   ├── aclnn_doc_missing_issue_143159.md # aclnn 文档缺失 Issue
│   │   │   ├── cmake_error_issue_143200.md       # CMake 错误 Issue
│   │   │   ├── ut_missing_issue_143230.md        # UT 缺失 Issue
│   │   │   ├── ut_failure_issue_143300.md        # UT 失败 Issue
│   │   │   ├── examples_missing_issue_143330.md  # Examples 缺失 Issue
│   │   │   ├── examples_failure_issue_143400.md  # Examples 失败 Issue
│   │   │   ├── link_error_issue_143058.md        # 断链 Issue
│   │   │   └── doc_error_issue_143058.md         # 文档错误 Issue
│   │   └── his/                                  # 历史归档（可选）
│   ├── ops-nn/                                   # ops-nn 仓库同日扫描
│   │   ├── ...
│   ├── ops-math/                                 # ops-math 仓库同日扫描
│   │   ├── ...
│   └── ops-transformer/                          # ops-transformer 仓库同日扫描
│       ├── ...
├── 20260513/                                     # 第二天扫描
│   └── ops-cv/
│       ├── ...
```

### 1.2 报告文件命名

**格式**: `reports/{YYYYMMDD}/{repo}/{command_name}_report_{HHMMSS}.md`

| 部分 | 说明 | 示例 |
|-----|------|------|
| `{YYYYMMDD}` | 扫描日期 | 20260512 |
| `{repo}` | 仓库名称 | ops-math, ops-nn, ops-cv, ops-transformer |
| `{command_name}` | Command 名称 | op-doc-completeness, cmake-scan, ut-analysis-guide 等 |
| `{HHMMSS}` | 时间戳（时分秒） | 143058 |

**示例**:
- `reports/20260512/ops-transformer/op-doc-completeness_report_151345.md`
- `reports/20260512/ops-nn/cmake-scan_report_173045.md`
- `reports/20260512/ops-math/ut-analysis-guide_report_173045.md`

### 1.3 Issue 文件命名

**格式**: `reports/{YYYYMMDD}/{repo}/issues/{issue_type}_issue_{HHMMSS}.md`

| 部分 | 说明 | 示例 |
|-----|------|------|
| `{YYYYMMDD}` | 扫描日期 | 20260512 |
| `{repo}` | 目标仓库 | ops-math, ops-nn, ops-cv, ops-transformer |
| `{issue_type}` | Issue类型 | readme_missing, aclnn_doc_missing, cmake_error, ut_missing 等 |
| `{HHMMSS}` | 时间戳（时分秒） | 151345 |

**示例**:
- `reports/20260512/ops-transformer/issues/readme_missing_issue_151345.md`
- `reports/20260512/ops-nn/issues/cmake_error_issue_173045.md`
- `reports/20260512/ops-math/issues/ut_missing_issue_173045.md`

### 1.4 扫描类型与 Issue 类型映射

| 扫描 Command | scan_type | issue_type | Issue 模板 | 自动生成 |
|-------------|-----------|------------|-----------|---------|
| cmake-scan | cmake | cmake_error | Bug-Report | ✅ **所有问题** |
| repo-docs-scan | doc_scan | doc_error, link_error | Documentation | ✅ **所有问题** |
| ut-analysis-guide | ut_missing | ut_missing | Bug-Report | ✅ **所有问题** |
| ut-test-report | ut_test | ut_failure | Bug-Report | ✅ **所有问题** |
| op-doc-completeness | doc_completeness | readme_missing, aclnn_doc_missing | Documentation | ✅ **所有问题** |
| examples-analysis-guide | examples | examples_missing | Requirement | ✅ **所有问题** |
| examples-test-report | examples_test | examples_failure | Bug-Report | ✅ **所有问题** |

> **核心规则**：**所有问题都自动生成 Issue 文件**，按优先级排序：
> - 🔴 高优先级（严重/阻塞）→ Bug-Report → 排在前 1-N 位
> - 🟠 中优先级（失败/中等）→ Bug-Report → 排在中间
> - 🟡 低优先级（警告/轻微）→ Documentation → 排在后面

### 1.5 Issue 文件优先级命名

**命名格式**: `{issue_type}_{priority}_issue_{HHMMSS}.md`

| 优先级标识 | 适用问题 | 排序位置 |
|----------|---------|:---:|
| `_high_` | 🔴 严重/阻塞 | 前 1-N |
| `_medium_` | 🟠 失败/中等 | 中间 |
| `_low_` | 🟡 警告/轻微 | 后面 |

**示例**：
- `reports/20260512/ops-cv/issues/cmake_error_high_issue_173045.md` - 高优先级
- `reports/20260512/ops-cv/issues/cmake_error_medium_issue_173045.md` - 中优先级
- `reports/20260512/ops-cv/issues/link_error_low_issue_173045.md` - 低优先级

---

## 二、Issue 创建流程

### 2.1 流程总览

```
扫描执行 → 生成报告 → 提取问题 → 分类问题 → 询问合并 → 生成 Issue MD → 询问提交 → 选择提交方式
    │           │          │          │           │            │            │           │
    ▼           ▼          ▼          ▼           ▼            ▼            ▼           ▼
  Step 1    Step 2     Step 3     Step 4     Step 5       Step 6       Step 7     Step 8
```

### 2.2 核心原则

> **重要**：以下三个原则是所有 Skills 必须遵循的核心规则：

| 原则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别（严重/中等/轻微），所有发现问题都生成 Issue 文件 |
| **报告后询问提交** | 每次生成报告后，必须询问用户是否提交 Issue 到对应仓库 |
| **同类问题合并选项** | 同类问题涉及多个算子时，询问用户是否合并成一个 Issue（按问题类型+仓库合并） |

### 2.3 详细步骤

#### Step 1: 扫描执行

各 Skill 执行扫描任务，收集问题数据。

#### Step 2: 生成报告

生成带时间戳的报告文件：
```
reports/{date}/{repo}/{command_name}_report_{time}.md
```

**说明**：
- `{date}`: 扫描日期，格式 YYYYMMDD（如 20260512）
- `{repo}`: 仓库名称（如 ops-cv, ops-nn）
- `{command_name}`: 扫描类型（如 op-doc-completeness, cmake-scan）
- `{time}`: 时间戳时分秒（如 143058）

#### Step 3: 提取问题

从报告中提取**所有问题**（不分严重程度）：
- 🔴 高优先级问题（阻塞/严重）
- 🟠 中优先级问题（失败/中等）
- 🟡 低优先级问题（警告/轻微）
- 🟢 信息级问题（建议/提示）

**注意**：所有问题都生成 Issue，不筛选。

#### Step 4: 分类问题

按「问题类型+仓库」对问题进行分类：

| 分类维度 | 说明 | 示例 |
|---------|------|------|
| 问题类型 | 同类问题归为一组 | README缺失、aclnn文档缺失、断链、CMake错误等 |
| 仓库 | 同一仓库的问题归为一组 | ops-cv、ops-math、ops-nn、ops-transformer |

**分类示例**：
```
ops-cv 仓库问题分类：
├── README缺失（15个算子）
├── aclnn文档缺失（8个算子）
├── 断链问题（3个链接）
└── CMake错误（2个配置）

ops-math 仓库问题分类：
├── README缺失（10个算子）
└── UT缺失（5个算子）
```

#### Step 5: 询问合并

**同类问题涉及多个算子时，询问用户是否合并**：

```
发现以下同类问题涉及多个算子：

| 问题类型 | 仓库 | 涉及算子数 | 示例算子 |
|---------|------|:---:|---------|
| README缺失 | ops-cv | 15 | aipp, col2im, grid_sample... |
| aclnn文档缺失 | ops-cv | 8 | grid_sample, resize_bilinear... |
| 断链问题 | ops-cv | 3 | aicpu_develop_guide.md 中的链接 |

合并选项：
【合并】将同类问题合并成一个 Issue（按问题类型+仓库）
  - 例如：ops-cv README缺失 → 生成 1 个 Issue，列出所有15个算子
【不合并】每个算子单独一个 Issue
  - 例如：ops-cv README缺失 → 生成 15 个 Issue，每个算子一个

请选择处理方式：
1. 合并同类问题（推荐）
2. 不合并，每个算子单独一个 Issue
3. 部分合并（指定哪些问题类型合并）

请输入选项编号 (1/2/3):
```

#### Step 6: 生成 Issue MD 文件

根据用户选择生成 Issue 文件：

**合并模式**：
```
reports/{date}/{repo}/issues/{issue_type}_merged_issue_{time}.md
```

**不合并模式**：
```
reports/{date}/{repo}/issues/{op_name}_{issue_type}_issue_{time}.md
```

**示例**：
- 合并模式：`reports/20260512/ops-cv/issues/readme_missing_merged_issue_143159.md`（包含15个算子）
- 不合并：`reports/20260512/ops-cv/issues/aipp_readme_missing_issue_143159.md`（单个算子）

#### Step 7: 询问提交

**生成 Issue 文件后，询问用户是否提交**：

```
✅ 报告生成完成！

报告路径: reports/{date}/{repo}/{command_name}_report_{time}.md

已生成 {n} 个 Issue 文件：

| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/20260512/ops-cv/issues/readme_missing_merged_issue.md | [Documentation]: [AI 识别] ops-cv README缺失（15个算子） | 15 | cann/ops-cv |
| 2 | reports/20260512/ops-cv/issues/aclnn_doc_missing_merged_issue.md | [Documentation]: [AI 识别] ops-cv aclnn文档缺失（8个算子） | 8 | cann/ops-cv |

是否提交 Issue 到对应仓库？
1. 是，全部提交 - 通过 API 直接提交所有 Issue
2. 是，选择提交 - 选择部分 Issue 提交
3. 否，暂不提交 - 仅保留 Issue 文件，后续手动处理
4. 否，手动提交 - 提供提交链接，用户自行复制内容提交

请输入选项编号 (1/2/3/4):
```

#### Step 8: API 直接提交（用户选择"是，全部提交"或"是，选择提交"）

调用 GitCode API 创建 Issue：

```bash
curl -s --request POST \
  --header "PRIVATE-TOKEN: <token>" \
  "https://api.gitcode.com/api/v4/projects/<project_id>/issues" \
  --data "title=<标题>" \
  --data "description=<描述>" \
  --data "labels=<标签>"
```

**提交结果记录**：
- 成功：记录 Issue URL
- 失败（403）：提示权限不足，改为手动提交

#### Step 7: 手动提交（用户选择"手动提交"）

提供提交链接：
```
Issue 文件已生成，请手动提交：
- Issue 1: https://gitcode.com/CANN/ops-transformer/issues/new
- Issue 2: https://gitcode.com/CANN/ops-transformer/issues/new

文件路径：
- reports/20260512/ops-transformer/issues/readme_missing_issue_151345.md
- reports/20260512/ops-transformer/issues/aclnn_doc_missing_issue_151345.md
```

---

## 三、各 Skill Issue 创建触发点

### 核心规则

**所有 Skill 遵循统一规则**：**扫描完成后，所有问题自动生成 Issue 文件，按优先级排序**

```
Issue 文件排序：
├── 高优先级 Issue（🔴 严重/阻塞）
│   ├── Issue #1: [Bug-Report]: [AI 识别] {repo} {严重问题}
│   ├── Issue #2: [Bug-Report]: [AI 识别] {repo} {阻塞问题}
│   └── ...
├── 中优先级 Issue（🟠 失败/中等）
│   ├── Issue #N: [Bug-Report]: [AI 识别] {repo} {中等问题}
│   └── ...
└── 低优先级 Issue（🟡 警告/轻微）
    ├── Issue #M: [Documentation]: [AI 识别] {repo} {轻微问题}
    └── ...
```

### Issue 类型与优先级映射

| 问题严重程度 | Issue 类型 | 优先级标识 | 排序位置 |
|:---:|:---:|:---:|:---:|
| 🔴 严重/阻塞 | Bug-Report | `_high_` | 前 1-N |
| 🟠 失败/中等 | Bug-Report | `_medium_` | 中间 |
| 🟡 警告/轻微 | Documentation | `_low_` | 后面 |
| 🟢 信息 | Documentation | `_low_` | 最后 |

---

### 3.1 cmake-scan

**触发时机**: 扫描完成后，发现所有问题

**Issue 类型**: Bug-Report（高/中优先级），Documentation（低优先级）

**自动生成规则**: ✅ **所有问题自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

扫描发现 {n} 个问题，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ({high_count} 个) ===
| 序号 | Issue 文件 | 问题类型 |
|:---:|-----------|---------|
| 1 | reports/{date}/{repo}/issues/cmake_error_high_issue_{time}.md | OPTYPE不一致 |
| ... | ... | ... |

=== 中优先级 Issue ({medium_count} 个) ===
| 序号 | Issue 文件 | 问题类型 |
|:---:|-----------|---------|

=== 低优先级 Issue ({low_count} 个) ===
| 序号 | Issue 文件 | 问题类型 |
|:---:|-----------|---------|

是否需要提交 Issue？
```

### 3.2 repo-docs-scan

**触发时机**: 扫描完成后，发现所有问题

**Issue 类型**: Documentation

**自动生成规则**: ✅ **所有问题自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

扫描发现 {n} 个问题，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
断链TOP10、严重文档错误

=== 中优先级 Issue ===
执行验证失败问题

=== 低优先级 Issue ===
轻微描述问题
```

### 3.3 ut-analysis-guide

**触发时机**: 分析完成后，发现所有 UT 缺失

**Issue 类型**: Bug-Report（高/中优先级），Requirement（低优先级）

**自动生成规则**: ✅ **所有 UT 缺失自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

分析发现 {n} 个 UT 缺失，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
高优先级 UT 缺失列表

=== 中优先级 Issue ===
中优先级 UT 缺失列表

=== 低优先级 Issue ===
低优先级 UT 缺失列表
```

### 3.4 ut-test-report

**触发时机**: 测试完成后，发现所有问题

**Issue 类型**: Bug-Report

**自动生成规则**: ✅ **所有问题自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

测试发现 {n} 个问题，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
阻塞问题（段错误）

=== 中优先级 Issue ===
测试失败问题

=== 低优先级 Issue ===
警告问题
```

### 3.5 op-doc-completeness

**触发时机**: 扫描完成后，发现所有文档缺失

**Issue 类型**: Documentation

**自动生成规则**: ✅ **所有缺失问题自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

扫描发现 {n} 个 README 缺失和 {m} 个 aclnn API 文档缺失，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
有算子源码但 README 缺失

=== 中优先级 Issue ===
aclnn API 文档缺失

=== 低优先级 Issue ===
无算子源码但存在占位文档（需删除）
```

### 3.6 examples-analysis-guide

**触发时机**: 扫描完成后，发现所有 examples 缺失

**Issue 类型**: Requirement

**自动生成规则**: ✅ **所有缺失自动生成 Issue，按优先级排序**

**输出**:
```markdown
## Issue 创建提示

发现 {n} 个算子缺少 examples，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
有 kernel 和调用接口但无 examples

=== 中优先级 Issue ===
有调用接口但无 examples

=== 低优先级 Issue ===
建议补充但不强制
```

### 3.7 examples-test-report

**触发时机**: 测试完成后，发现所有失败问题

**Issue 类型**: Bug-Report

**自动生成规则**: ✅ **所有失败问题自动生成 Issue，按优先级排序**

| 问题类型 | Issue 类型 | 优先级 | 自动生成 |
|---------|-----------|:---:|---------|
| 段错误阻塞 | Bug-Report | 高 | ✅ 自动生成 |
| eager 测试失败 | Bug-Report | 中 | ✅ 自动生成 |
| graph 测试失败 | Bug-Report | 中 | ✅ 自动生成 |
| 编译失败 | Bug-Report | 中 | ✅ 自动生成 |

**输出**:
```markdown
## Issue 创建提示

测试发现 {n} 个失败问题，已生成 Issue 文件（按优先级排序）：

=== 高优先级 Issue ===
| 序号 | Issue 文件 | Issue 标题 |
|:---:|-----------|-----------|
| 1 | reports/{date}/{repo}/issues/{op}_examples_failure_high_issue_{time}.md | [Bug-Report]: [AI 识别] {repo} {op} 段错误阻塞 |

=== 中优先级 Issue ===
| 序号 | Issue 文件 | Issue 标题 |
|:---:|-----------|-----------|
| 1 | reports/{date}/{repo}/issues/{op}_examples_failure_medium_issue_{time}.md | [Bug-Report]: [AI 识别] {repo} {op} eager 测试失败 |

=== 低优先级 Issue ===
暂无

是否需要提交 Issue？
```

---

## 四、Issue 内容规范

### 4.1 标题格式

**必须包含 `[AI 识别]` 标记**：

| 模板 | 标题格式 | 合并模式标题 |
|-----|---------|------------|
| Bug-Report | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} {问题简述}` | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} {问题简述}（{n}个算子）` |
| Documentation | `[Documentation|文档反馈]: [AI 识别] {repo} {问题简述}` | `[Documentation|文档反馈]: [AI 识别] {repo} {问题简述}（{n}个算子）` |
| Requirement | `[Requirement|需求建议]: [AI 识别] {repo} {需求简述}` | `[Requirement|需求建议]: [AI 识别] {repo} {需求简述}（{n}个算子）` |

### 4.1.1 合并 Issue 标题示例

| 问题类型 | 仓库 | 算子数 | 合并标题 |
|---------|------|:---:|---------|
| README缺失 | ops-cv | 15 | `[Documentation|文档反馈]: [AI 识别] ops-cv README缺失（15个算子）` |
| aclnn文档缺失 | ops-cv | 8 | `[Documentation|文档反馈]: [AI 识别] ops-cv aclnn API文档缺失（8个算子）` |
| 断链问题 | ops-cv | 3 | `[Documentation|文档反馈]: [AI 识别] ops-cv 存在3个断链影响文档可达性` |
| CMake错误 | ops-nn | 5 | `[Bug-Report|缺陷反馈]: [AI 识别] ops-nn CMake配置错误（5个算子）` |
| UT缺失 | ops-math | 10 | `[Bug-Report|缺陷反馈]: [AI 识别] ops-math UT测试缺失（10个算子）` |

### 4.2 正文格式

参考 `gitcode-issue-creator/references/templates.md`：

**Bug-Report 正文结构**:
```
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述
{问题描述}

### Environment / 环境信息

**软件环境**:
- CANN 版本: {CANN版本，如 8.0.RC1、8.5.0 等}
- 操作系统: {OS版本，如 Ubuntu 22.04、CentOS 7.9 等}

**硬件环境**:
- NPU 型号: {芯片型号，如 Ascend910B1、Ascend910B2、Ascend310P 等}
- 服务器型号: {可选，如 A2、A3 服务器}

**问题环境**:
- 仓库: {repo}
- 问题类型: {issue_type}
- 问题文件数: {count}
- 问题性质: {BUG/规范问题}

### Steps to reproduce the issue / 重现步骤
1. {步骤1}
2. {步骤2}

### Describe the expected behavior / 预期结果
{修复建议}

### Related log / screenshot / 日志 / 截图
{问题文件列表}

### Special notes for this issue/备注
{影响说明}
```

**Documentation 正文结构**:
```
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）
{文档链接}

### Issues Section（问题文档片段）
{问题文档片段/截图}

### Existing Issues（存在的问题）
{问题描述}
```

**Requirement 正文结构**:
```
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）
{背景信息内容}

### Origin（信息来源）
{信息来源}

### Benefit / Necessity （价值/作用）
{价值/作用内容}

### Design（设计方案）
{设计方案内容}
```

### 4.3 合并 Issue 正文格式

**合并 Issue 包含所有相关算子列表**：

```markdown
# [Documentation|文档反馈]: [AI 识别] {repo} README缺失（{n}个算子）

**标签**: `documentation`, `readme-missing`
**生成时间**: {timestamp}
**Command**: {command_name}
**涉及算子数**: {n}

---

Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

## Document Link（文档链接）

以下 {n} 个算子缺少 README.md 文档：

## Missing Operators List（缺失算子列表）

| 序号 | 算子名称 | 算子路径 | 状态 |
|:---:|---------|---------|:---:|
| 1 | {op_name_1} | {repo}/{op_class}/{op_name_1}/ | ❌ README缺失 |
| 2 | {op_name_2} | {repo}/{op_class}/{op_name_2}/ | ❌ README缺失 |
| ... | ... | ... | ... |
| {n} | {op_name_n} | {repo}/{op_class}/{op_name_n}/ | ❌ README缺失 |

## Existing Issues（存在的问题）

上述 {n} 个算子目录缺少 README.md 文档，导致：
1. 用户无法了解算子功能和使用方法
2. 文档完备性不达标
3. 影响用户体验和算子可维护性

## Expected Behavior（预期结果）

每个算子目录应包含 README.md 文档，内容包括：
- 算子功能说明
- 输入输出规格
- 支持的数据类型和格式
- 使用示例（可选）

## Suggested Fix（修复建议）

请为以下算子补充 README.md 文档：
```
{repo}/{op_class}/{op_name_1}/README.md
{repo}/{op_class}/{op_name_2}/README.md
...
```

---

**提交地址**: https://gitcode.com/cann/{repo}/issues/new
**Issue 文件**: reports/{date}/{repo}/issues/readme_missing_merged_issue_{time}.md
```

### 4.4 合并 Issue 示例

**示例：ops-cv README缺失（15个算子）**

```markdown
# [Documentation|文档反馈]: [AI 识别] ops-cv README缺失（15个算子）

**标签**: `documentation`, `readme-missing`
**生成时间**: 20260428_173045
**Command**: op-doc-completeness
**涉及算子数**: 15

---

Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

## Missing Operators List（缺失算子列表）

| 序号 | 算子名称 | 算子路径 | 状态 |
|:---:|---------|---------|:---:|
| 1 | aipp | ops-cv/image/aipp/ | ❌ README缺失 |
| 2 | col2im | ops-cv/image/col2im/ | ❌ README缺失 |
| 3 | grid_sample | ops-cv/image/grid_sample/ | ❌ README缺失 |
| 4 | resize_bicubic_v2 | ops-cv/image/resize_bicubic_v2/ | ❌ README缺失 |
| 5 | resize_bilinear_v2 | ops-cv/image/resize_bilinear_v2/ | ❌ README缺失 |
| 6 | resize_linear | ops-cv/image/resize_linear/ | ❌ README缺失 |
| 7 | resize_nearest_neighbor_v2 | ops-cv/image/resize_nearest_neighbor_v2/ | ❌ README缺失 |
| 8 | upsample_bicubic2d | ops-cv/image/upsample_bicubic2d/ | ❌ README缺失 |
| 9 | upsample_bilinear2d | ops-cv/image/upsample_bilinear2d/ | ❌ README缺失 |
| 10 | upsample_linear1d | ops-cv/image/upsample_linear1d/ | ❌ README缺失 |
| 11 | upsample_nearest3d | ops-cv/image/upsample_nearest3d/ | ❌ README缺失 |
| 12 | spatial_transformer | ops-cv/image/spatial_transformer/ | ❌ README缺失 |
| 13 | nms_with_mask | ops-cv/image/nms_with_mask/ | ❌ README缺失 |
| 14 | rasterizer | ops-cv/image/rasterizer/ | ❌ README缺失 |
| 15 | scale_and_translate | ops-cv/image/scale_and_translate/ | ❌ README缺失 |

## Existing Issues（存在的问题）

上述 15 个算子目录缺少 README.md 文档...

## Suggested Fix（修复建议）

请为以下算子补充 README.md 文档...
```

---

## 五、目录结构

### 5.1 统一目录结构

**按日期+仓库层级组织**，便于追踪历史和区分不同仓库：

```
reports/
├── 20260512/                                     # 日期目录 (YYYYMMDD)
│   ├── ops-cv/                                   # 仓库目录
│   │   ├── repo-docs-scan_report_143058.md
│   │   ├── op-doc-completeness_report_143159.md
│   │   ├── cmake-scan_report_143200.md
│   │   ├── ut-analysis-guide_report_143230.md
│   │   ├── ut-test-report_report_143300.md
│   │   ├── examples-analysis-guide_report_143330.md
│   │   ├── examples-test-report_report_143400.md
│   │   ├── op-list-validation_report_143430.md
│   │   ├── op-api-list-validation_report_143500.md
│   │   ├── issues/                               # Issue 文件目录
│   │   │   ├── readme_missing_issue_143159.md
│   │   │   ├── aclnn_doc_missing_issue_143159.md
│   │   │   ├── cmake_error_issue_143200.md
│   │   │   ├── ut_missing_issue_143230.md
│   │   │   ├── ut_failure_issue_143300.md
│   │   │   ├── examples_missing_issue_143330.md
│   │   │   ├── examples_failure_issue_143400.md
│   │   │   ├── link_error_issue_143058.md
│   │   │   └── doc_error_issue_143058.md
│   │   └── his/                                  # 历史归档（可选）
│   ├── ops-nn/                                   # ops-nn 仓库同日扫描
│   │   ├── ...
│   ├── ops-math/                                 # ops-math 仓库同日扫描
│   │   ├── ...
│   └── ops-transformer/                          # ops-transformer 仓库同日扫描
│       ├── ...
└── 20260513/                                     # 第二天扫描
    └── ...
```

### 5.2 目录创建规则

执行扫描时自动创建目录：

```bash
# 获取日期和时间
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# 创建报告目录
mkdir -p "reports/${date_str}/${repo}"

# 创建 Issues 目录
mkdir -p "reports/${date_str}/${repo}/issues"

# 创建历史归档目录（可选）
mkdir -p "reports/${date_str}/${repo}/his"
```

### 5.3 报告路径生成示例

```bash
# 获取时间戳
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")
repo="ops-transformer"
command_name="op-doc-completeness"

# 报告路径
report_path="reports/${date_str}/${repo}/${command_name}_report_${time_str}.md"

# Issue 路径
issue_path="reports/${date_str}/${repo}/issues/readme_missing_issue_${time_str}.md"
```

---

## 六、用户交互流程

### 6.1 扫描完成提示

```
✅ 扫描完成！

报告已生成: reports/{date}/{repo}/{command_name}_report_{time}.md

发现 {n} 个问题，按类型分类如下：

| 问题类型 | 涉及算子数 | 问题描述 |
|---------|:---:|---------|
| README缺失 | 15 | 15个算子缺少README.md |
| aclnn文档缺失 | 8 | 8个算子缺少aclnn API文档 |
| 断链问题 | 3 | 3个文档链接不存在 |

同类问题是否合并？
1. 合并同类问题（推荐）- 按问题类型+仓库合并
2. 不合并 - 每个算子单独一个Issue
3. 部分合并 - 指定哪些问题类型合并

请输入选项编号 (1/2/3):
```

### 6.2 合并选项处理

**用户选择「合并同类问题」**：

```
已按问题类型+仓库合并，生成 {m} 个 Issue 文件：

| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/20260512/ops-cv/issues/readme_missing_merged_issue.md | [Documentation]: [AI 识别] ops-cv README缺失（15个算子） | 15 | cann/ops-cv |
| 2 | reports/20260512/ops-cv/issues/aclnn_doc_missing_merged_issue.md | [Documentation]: [AI 识别] ops-cv aclnn文档缺失（8个算子） | 8 | cann/ops-cv |
| 3 | reports/20260512/ops-cv/issues/link_error_issue.md | [Documentation]: [AI 识别] ops-cv 存在3个断链 | 3 | cann/ops-cv |

是否提交 Issue 到对应仓库？
1. 是，全部提交 - API 直接提交所有 Issue
2. 是，选择提交 - 选择部分 Issue 提交
3. 否，暂不提交 - 仅保留 Issue 文件
4. 否，手动提交 - 提供提交链接

请输入选项编号 (1/2/3/4):
```

**用户选择「不合并」**：

```
已为每个算子单独生成 Issue 文件（共 {total} 个）：

README缺失 Issue（15个）：
| 序号 | Issue 文件 | 目标仓库 |
|:---:|-----------|---------|
| 1 | reports/20260512/ops-cv/issues/aipp_readme_missing_issue.md | cann/ops-cv |
| 2 | reports/20260512/ops-cv/issues/col2im_readme_missing_issue.md | cann/ops-cv |
| ... | ... | ... |

aclnn文档缺失 Issue（8个）：
| 序号 | Issue 文件 | 目标仓库 |
|:---:|-----------|---------|
| 1 | reports/20260512/ops-cv/issues/grid_sample_aclnn_doc_missing_issue.md | cann/ops-cv |
| ... | ... | ... |

是否提交 Issue 到对应仓库？
1. 是，全部提交 - API 直接提交所有 Issue
2. 是，选择提交 - 选择部分 Issue 提交
3. 否，暂不提交 - 仅保留 Issue 文件
4. 否，手动提交 - 提供提交链接

请输入选项编号 (1/2/3/4):
```

### 6.3 提交结果反馈

```
提交结果：

| 序号 | Issue 标题 | 提交状态 | Issue URL |
|:---:|-----------|:-------:|---------|
| 1 | [Documentation]: [AI 识别] ops-cv README缺失（15个算子） | ✅ 成功 | https://gitcode.com/cann/ops-cv/issues/123 |
| 2 | [Documentation]: [AI 识别] ops-cv aclnn文档缺失（8个算子） | ✅ 成功 | https://gitcode.com/cann/ops-cv/issues/124 |
| 3 | [Documentation]: [AI 识别] ops-cv 存在3个断链 | ❌ 权限不足 | 请手动提交 |

提交失败时手动提交地址：
- ops-cv: https://gitcode.com/cann/ops-cv/issues/new

Issue 文件路径：
- reports/20260512/ops-cv/issues/link_error_issue.md
```

---

## 七、时间戳生成方式

### Shell 获取时间戳

```bash
# 获取日期 (YYYYMMDD)
date_str=$(date +"%Y%m%d")
echo "20260512"

# 获取时间 (HHMMSS)
time_str=$(date +"%H%M%S")
echo "143058"
```

### Python 获取时间戳

```python
from datetime import datetime

# 获取日期
date_str = datetime.now().strftime("%Y%m%d")
# "20260512"

# 获取时间
time_str = datetime.now().strftime("%H%M%S")
# "143058"
```

### 命令别名

```bash
# 在 Skill 中使用
date_str=$(date +"%Y%m%d")          # 日期部分
time_str=$(date +"%H%M%S")          # 时间部分
repo="ops-transformer"              # 仓库名称
command_name="op-doc-completeness"  # 扫描类型

# 创建目录
mkdir -p "reports/${date_str}/${repo}"
mkdir -p "reports/${date_str}/${repo}/issues"

# 报告路径
report_path="reports/${date_str}/${repo}/${command_name}_report_${time_str}.md"

# Issue 路径
issue_path="reports/${date_str}/${repo}/issues/readme_missing_issue_${time_str}.md"
```
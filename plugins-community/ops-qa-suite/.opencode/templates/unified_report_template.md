# 统一报告模板

> **用途**: 本模板定义报告输出格式，各 Skill 生成报告时遵循此结构。
>
> **相关规范**:
> - Issue 创建流程: `issue_workflow_spec.md`
> - 报告命名规范: `issue_workflow_spec.md` 第一章

---

## 一、报告结构总览

```
# {repo_type} {report_type}报告

## 报告元信息

## 执行摘要

## 问题分类与统计

## 问题详情记录

## {skill_special_section}  ← 各 Skill 特殊字段区域

## 修复建议

## 附录
```

---

## 二、报告模板

### 模板文件

```markdown
# {repo_type} {report_type}报告

## 报告元信息

| 字段 | 值 |
|-----|---|
| **扫描/测试时间** | {date} |
| **仓库** | {repo_type} ({repo_description}) |
| **报告类型** | {report_type} |
| **扫描范围** | {scan_scope} |
| **执行环境** | {environment} |

---

## 执行摘要

### 总体状态

**执行状态**: {overall_status}

| 指标 | 数值 |
|-----|:----:|
| {metric_1_name} | {metric_1_value} |
| {metric_2_name} | {metric_2_value} |
| **整体评分/通过率** | {score_or_rate} |

### 状态标识说明

| 标识 | 含义 |
|:---:|-----|
| ✅ | 通过/已解决/完备 |
| ⚠️ | 部分/警告/待补充 |
| ❌ | 失败/未解决/缺失 |
| 🔴 | 阻塞/严重 |
| 🟡 | 非阻塞/中等 |
| 🟢 | 信息/轻微 |

---

## 问题分类与统计

### 按问题类型统计

| 问题类型 | 数量 | 严重程度 | 占比 |
|---------|:----:|:-------:|:----:|
| {problem_type_1} | {count_1} | {severity_1} | {percent_1}% |
| {problem_type_2} | {count_2} | {severity_2} | {percent_2}% |
| **合计** | {total_count} | - | 100% |

### 按严重程度统计

| 严重程度 | 数量 | 处理优先级 |
|:-------:|:----:|:---------:|
| 🔴 严重/阻塞 | {severe_count} | 高 |
| 🟡 中等/非阻塞 | {medium_count} | 中 |
| 🟢 轻微/信息 | {minor_count} | 低 |

---

## 问题详情记录

> **格式说明**: 每个问题按「问题标题 → 问题描述 → 解决方案/修复建议」结构记录

### 问题 {n}：{problem_title}

#### 基本信息

| 属性 | 内容 |
|-----|------|
| **问题类型** | {problem_type} |
| **严重程度** | {severity} |
| **状态** | {status} |
| **影响范围** | {impact_scope} |

#### 问题描述

{problem_description}

**错误输出**（如有）:
```
{error_output}
```

**根因分析**:
- {root_cause}

#### 解决方案（已解决）

**解决方法**:
```bash
{solution_command}
```

**验证结果**: {verification_result}

#### 修复建议（未解决）

**临时规避方案**:
```bash
{workaround_command}
```

**排查步骤**:
1. {debug_step_1}
2. {debug_step_2}

---

## {skill_special_section}

> **说明**: 以下内容为 `{skill_name}` 报告特有的字段，其他报告类型无需包含。

{special_content}

---

## 修复建议

### 优先级排序

| 优先级 | 问题类型 | 建议操作 |
|:-----:|---------|---------|
| **高** | {high_priority_issues} | {high_action} |
| **中** | {medium_priority_issues} | {medium_action} |
| **低** | {low_priority_issues} | {low_action} |

### 批量修复命令（如有）

```bash
{batch_fix_command}
```

---

## 附录

### A. 详细清单

{detailed_list}

### B. 环境检查详情

| 检查项 | 状态 | 详情 |
|-------|:----:|------|
| {check_item_1} | ✅ | {detail_1} |
| {check_item_2} | ✅ | {detail_2} |

---

**提交地址**: https://gitcode.com/cann/{repo}/issues/new

> **Issue 创建**: 发现严重问题时，自动生成 Issue 文件供用户确认，所有内容完整嵌入 Issue
```

---

## 三、各 Skill 特殊字段区域

以下表格标注各 Skill 报告需要在「特殊字段区域」增加的内容：

| Skill | 特殊字段名称 | 特殊内容 |
|-------|-------------|---------|
| **cmake-scan** | GitCode Issue 文件 | BUG 类型问题列表（Issue标题/文件路径/提交地址） |
| **repo-docs-scan** | 执行验证详情 | 编译命令验证结果 + 运行命令验证结果（命令/结果/输出摘要） |
| **repo-docs-scan** | 断链详细清单 | 断链TOP10表格（文件/行号/链接路径/错误类型） |
| **ut-test-report** | 测试统计结果 | 测试套件表格（套件名/用例数/通过数/状态） + 被跳过测试表格 |
| **ut-test-report** | UT 测试类型说明 | 4种UT类型的说明表格（类型/内容/位置/依赖） |
| **ut-analysis-guide** | UT 覆盖情况详情 | 完整覆盖算子列表 + 部分覆盖算子列表（含缺失类型） |
| **ut-analysis-guide** | 缺失 UT 文件算子 | 高优先级缺失 + 中优先级缺失表格 |
| **op-doc-completeness** | 判断规则说明 | README.md判断规则 + aclnn API文档判断规则表格 |
| **op-doc-completeness** | 文档完备性详情 | README缺失列表 + aclnn缺失列表 + 无需文档列表 + 不应存在占位文档列表 |
| **examples-analysis-guide** | 分类统计 | 按目录分类的统计表格（分类/算子总数/缺失数/缺失比例） |
| **examples-analysis-guide** | 有 examples 算子列表 | 有完整 examples 测试用例的算子示例表格 |
| **examples-test-report** | 测试统计结果 | 测试类型表格（类型/总数/成功/失败/跳过） + 总体统计表格 |
| **examples-test-report** | 测试模式说明 | 三种执行模式说明（默认/指定芯片/skip-sim） + simulator支持情况 |
| **examples-test-report** | 芯片兼容性记录 | use_simulator字段 + soc_used字段 + 跳过原因表格 |

---

## 四、报告类型与特殊字段对照

### cmake-scan 报告特殊字段

```markdown
## GitCode Issue 文件

> 以下问题会导致 CMake 配置失败，已生成符合 GitCode 模板格式的 Issue 文件。

| Issue序号 | 问题类型 | Issue 标题 | Issue 文件 | 提交地址 |
|:--------:|---------|-----------|-----------|---------|
| #1 | {type} | {title} | `issues/{filename}` | {url} |
| #2 | {type} | {title} | `issues/{filename}` | {url} |
```

### repo-docs-scan 报告特殊字段

```markdown
## 执行验证详情

### 编译命令验证

**执行命令**: `bash build.sh --pkg --soc=ascend910b --ops={op} -j4`
**执行结果**: ✅ 成功 / ❌ 失败
**输出摘要**: {output_summary}

### 运行命令验证

**执行命令**: `{run_command}`
**执行结果**: ✅ 成功 / ❌ 失败
**输出摘要**: {output_summary}

## 断链详细清单

| 序号 | 文件 | 行号 | 链接路径 | 错误类型 |
|:---:|------|:---:|---------|---------|
| 1 | {file} | {line} | {link} | {error_type} |
```

### ut-test-report 报告特殊字段

```markdown
## 测试统计结果

### op_host UT (tiling + infershape)

| 测试套件 | 测试用例数 | 通过数 | 状态 |
|---------|:---------:|:-----:|:----:|
| {suite_1} | {count_1} | {pass_1} | ✅ |
| {suite_2} | {count_2} | {pass_2} | 🔴 段错误 |

### 被跳过/未执行的测试

| 类型 | 测试内容 | 原因 | 测试数量 |
|-----|---------|------|:-------:|
| {type_1} | {content_1} | 🔴 段错误 | {count} |

## UT 测试类型说明

| UT 类型 | 测试内容 | 测试文件位置 | 依赖环境 |
|---------|---------|-------------|---------|
| op_host_ut | tiling + infershape | tests/ut/op_host/ | BUILD_PATH |
| op_api_ut | aclnn接口测试 | tests/ut/op_api/ | ACLNN库 |
| op_kernel_ut | AscendC kernel | tests/ut/op_kernel/ | NPU硬件 |
| aicpu_op_kernel_ut | AICPU kernel | tests/ut/op_kernel_aicpu/ | AICPU环境 |
```

### ut-analysis-guide 报告特殊字段

```markdown
## UT 覆盖情况详情

### 有完整 UT 覆盖的算子

| 算子 | UT 文件数 | UT 类型 | 覆盖状态 |
|------|:--------:|---------|:-------:|
| {op_1} | {count_1} | {types_1} | 完整 |

### 部分 UT 覆盖的算子

| 算子 | UT 文件数 | 当前类型 | 缺失类型 |
|------|:--------:|---------|---------|
| {op_1} | {count_1} | {types_1} | {missing_types} |

## 缺失 UT 文件的算子

### 高优先级缺失

| 算子名称 | 目录 | 状态 | 需补充UT类型 |
|---------|------|:----:|-------------|
| {op_1} | {dir_1} | 缺失 | {types_1} |

### 中优先级缺失

| 算子名称 | 目录 | 状态 | 需补充UT类型 |
|---------|------|:----:|-------------|
| {op_1} | {dir_1} | 缺失 | {types_1} |
```

### op-doc-completeness 报告特殊字段

```markdown
## 判断规则说明

### README.md 判断规则

| 场景 | README 内容要求 |
|-----|----------------|
| 有算子源码 | 算子功能说明、输入输出、数据类型支持 |
| 无算子源码 | 引导用户贡献算子源码的固定文案 |

### aclnn API 文档判断规则

| ACLNNTYPE | aclnn来源 | 是否需要文档 |
|-----------|----------|:-----------:|
| aclnn / aclnn_inner | CMake自动生成 | ✅ 必须 |
| aclnn_exclude + 有aclnn_xxx文件 | 手动实现 | ✅ 必须 |
| aclnn_exclude + 无aclnn_xxx文件 | 无aclnn接口 | ❌ 无需 |

## 文档完备性详情

### README.md 缺失列表

| 序号 | 算子路径 | 有算子源码 | 需补充内容 |
|:---:|---------|:---------:|-----------|
| 1 | {path} | 有/无 | {content} |

### aclnn API 文档缺失列表

| 序号 | 算子路径 | ACLNNTYPE | 建议文档路径 |
|:---:|---------|-----------|-------------|
| 1 | {path} | {type} | {doc_path} |

### 无需 aclnn 文档列表

| 序号 | 算子路径 | ACLNNTYPE | 说明 |
|:---:|---------|-----------|-----|
| 1 | {path} | aclnn_exclude | 无aclnn实现 |

### 不应存在的占位文档

| 序号 | 算子路径 | 问题 |
|:---:|---------|-----|
| 1 | {path} | 无需aclnn文档但存在占位文档 |
```

### examples-analysis-guide 报告特殊字段

```markdown
## 分类统计

| 分类 | 算子总数 | examples 缺失数 | 缺失比例 |
|------|:-------:|:--------------:|:-------:|
| {category_1} | {total_1} | {missing_1} | {percent_1}% |
| {category_2} | {total_2} | {missing_2} | {percent_2}% |

## 有 examples 测试用例的算子

共 **{count}** 个算子有完整 examples 测试用例。

| 算子 | 测试文件 |
|------|---------|
| {op_1} | {test_file_1} |
| {op_2} | {test_file_2} |
```

### examples-test-report 报告特殊字段

```markdown
## 测试统计结果

### 按测试类型统计

| 测试类型 | 算子总数 | 成功 | 失败 | 跳过 |
|---------|:-------:|:---:|:---:|:---:|
| eager | {eager_total} | {eager_success} | {eager_failure} | {eager_skipped} |
| graph | {graph_total} | {graph_success} | {graph_failure} | {graph_skipped} |

### 总体统计

| 指标 | 数值 |
|-----|:----:|
| 总测试数 | {total_tests} |
| 成功 | {success_tests} |
| 失败 | {failure_tests} |
| 跳过 | {skipped_tests} |
| 通过率 | {pass_rate}% |

## 测试模式说明

| 执行模式 | 当前 NPU 支持 | 当前 NPU 不支持 | simulator |
|---------|:---:|:---:|:---:|
| 默认（无参数） | 直接测试 | 自动 simulator | ✅ 自动 |
| --simulator --soc=X | 强制 simulator | 跳过 | ✅ 强制 |
| skip-sim | 直接测试 | **跳过测试** | ❌ 禁用 |

## 芯片兼容性记录

| 算子 | 测试类型 | use_simulator | soc_used | 跳过原因 |
|------|---------|:---:|---------|---------|
| {op_1} | eager | {use_sim_1} | {soc_1} | - |
| {op_2} | graph | false | {current_soc} | graph不支持simulator |
```

---

## 五、使用指南

### 如何应用此模板

1. **通用字段**: 所有报告必须包含「报告元信息」「执行摘要」「问题分类与统计」「问题详情记录」「修复建议」
2. **特殊字段**: 根据报告类型，从第三章「各 Skill 特殊字段区域」表格中选择需要增加的内容
3. **Issue 创建**: 扫描完成后自动生成 Issue 文件，详见 `issue_workflow_spec.md`
4. **命名规范**: 报告和 Issue 文件命名统一使用时间戳，详见 `issue_workflow_spec.md` 第一章
5. **附录**: 可选，用于存放详细清单、环境检查等辅助信息

### 报告类型映射

| 报告类型 | {report_type} 值 | {scan_scope} 值 |
|---------|-----------------|----------------|
| cmake-scan | CMake配置问题扫描 | CMakeLists.txt配置检查 |
| repo-docs-scan | 仓库文档质量扫描 | 资料正确性、易理解性、规范性 |
| ut-analysis-guide | UT缺失分析 | UT测试覆盖情况分析 |
| ut-test-report | 全量UT测试执行 | op_host/op_api/op_kernel UT执行 |
| op-doc-completeness | 算子文档完备性 | README.md + aclnn API文档 |
| examples-analysis-guide | Examples缺失扫描 | examples测试用例覆盖情况 |
| examples-test-report | 全量Examples测试执行 | test_aclnn/test_geir examples执行 |

> **时间戳生成**: 详见 `issue_workflow_spec.md` 第七章
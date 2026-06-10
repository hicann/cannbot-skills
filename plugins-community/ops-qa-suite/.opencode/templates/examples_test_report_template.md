# {repo_type} 仓库全量 Examples 测试报告

**测试日期**: {date}
**仓库**: {repo_type} ({repo_description})
**测试模式**: {test_mode}
**测试命令**: `bash build.sh --run_example {op} {test_mode}`

---

## 测试执行状态：{execution_status}

### 问题汇总

| 序号 | 问题 | 严重程度 | 状态 |
|:---:|-----|:-------:|:----:|
{problem_summary_rows}

---

## 测试统计结果

### 按测试类型统计

| 测试类型 | 算子总数 | 成功 | 失败 | 跳过 |
|---------|:-------:|:---:|:---:|:---:|
{test_type_stats_rows}

### 总体统计

| 指标 | 数值 |
|-----|:----:|
| 扫描算子总数 | {total_operators} |
| 有 examples 目录 | {has_examples} |
| 无 examples 目录 | {no_examples} |
| 成功测试 | {success_tests} |
| 失败测试 | {failed_tests} |
| 跳过测试 | {skipped_tests} |
| 成功率 | {success_rate}% |

### 按分类统计

| 分类 | 算子数 | 成功 | 失败 | 跳过 |
|------|:-----:|:---:|:---:|:---:|
{category_stats_rows}

---

## 问题详情记录

### 问题 1：{problem_name_1}（{problem_status_1}）

#### 执行命令

```bash
{problem_command_1}
```

#### 问题描述

- **算子名称**: {problem_op_name_1}
- **测试类型**: {problem_test_type_1}
- **错误输出**:
```
{problem_error_output_1}
```

- **根因分析**: {problem_root_cause_1}
- **影响范围**: {problem_impact_1}

#### 问题修复建议（未解决）

**临时规避方案**：
```bash
# 跳过该算子测试
{problem_workaround_1}
```

**排查建议**：
1. {problem_debug_step_1}
2. {problem_debug_step_2}

**永久修复建议**（可选）：
```bash
{problem_permanent_fix_1}
```

---

### 问题 2：{problem_name_2}（{problem_status_2}）

...（按相同格式继续）

---

## 跳过测试说明

| 算子 | 测试类型 | 跳过原因 |
|-----|---------|---------|
{skipped_tests_rows}

---

## 无 examples 目录算子

| 算子 | 分类 | 说明 |
|-----|------|------|
{no_examples_rows}

---

## Examples 文件统计

### 文件类型统计

| 文件类型 | 数量 | 说明 |
|---------|:---:|------|
| test_aclnn_*.cpp | {aclnn_file_count} | ACLNN API 调用示例 |
| test_geir_*.cpp | {geir_file_count} | GEIR 图引擎调用示例 |
| arch35 目录 | {arch35_count} | Ascend35 架构特定示例 |

### 架构特定示例分布

| 仓库 | arch35 示例数 | 主要算子 |
|-----|:------------:|---------|
{arch35_distribution_rows}

---

## 测试环境信息

| 项目 | 值 |
|-----|---|
| CANN 版本 | {cann_version} |
| NPU 设备 | {npu_device} |
| 编译器 | {compiler_version} |
| 测试模式 | {test_mode} |
| Simulator 使用 | {simulator_used} |

---

## 报告元信息

| 字段 | 值 |
|-----|---|
| 报告生成时间 | {timestamp} |
| 报告路径 | `reports/{date}/{repo}/examples-test-report_report_{time}.md` |
| 测试环境 | {test_environment} |
| 测试结果 | {test_result_summary} |

---

## 模板变量说明

| 变量名 | 说明 | 示例值 |
|-------|------|-------|
| `{repo_type}` | 仓库类型 | ops-cv |
| `{repo_description}` | 仓库描述 | 计算机视觉算子仓库 |
| `{date}` | 测试日期 | 2026-04-29 |
| `{test_mode}` | 测试模式 | eager, graph, all |
| `{execution_status}` | 执行状态 | 成功 / 部分成功（存在失败） |
| `{problem_summary_rows}` | 问题汇总表格行 | `| 1 | grid_sample 段错误 | 🔴 阻塞 | ❌ 未解决 |` |
| `{total_operators}` | 扫描算子总数 | 55 |
| `{has_examples}` | 有 examples 目录数 | 55 |
| `{no_examples}` | 无 examples 目录数 | 0 |
| `{success_tests}` | 成功测试数 | 48 |
| `{failed_tests}` | 失败测试数 | 3 |
| `{skipped_tests}` | 跳过测试数 | 4 |
| `{success_rate}` | 成功率 | 93.5 |
| `{test_type_stats_rows}` | 测试类型统计行 | eager/graph 分行 |
| `{category_stats_rows}` | 分类统计行 | image/objdetect 分行 |
| `{skipped_tests_rows}` | 跳过测试行 | `| aipp | eager | 无 test_aclnn 文件 |` |
| `{no_examples_rows}` | 无 examples 行 | `| xxx | math | 无 examples 目录 |` |
| `{aclnn_file_count}` | ACLNN 文件数 | 50 |
| `{geir_file_count}` | GEIR 文件数 | 12 |
| `{arch35_count}` | arch35 目录数 | 5 |
| `{cann_version}` | CANN 版本 | 8.5.0 |
| `{npu_device}` | NPU 设备 | ascend910b |
| `{compiler_version}` | 编译器版本 | GCC 15.2 |
| `{simulator_used}` | Simulator 使用 | 是/否 |
| `{test_environment}` | 测试环境 | CANN 8.5.0, NPU ascend910b |
| `{test_result_summary}` | 测试结果摘要 | 部分成功（48/55 通过） |

---

## 中间输出文件说明

> **重要**：所有中间文件存放在日期+仓库层级目录中。

| 文件名 | 说明 | 路径 |
|-------|------|------|
| `operator_list.txt` | 算子列表文件 | `reports/{date}/{repo}/operator_list.txt` |
| `examples_list.csv` | examples 文件列表 | `reports/{date}/{repo}/examples_list.csv` |
| `test_results.csv` | 测试结果记录 | `reports/{date}/{repo}/test_results.csv` |
| `examples-test-report_report_{time}.md` | 最终测试报告 | `reports/{date}/{repo}/examples-test-report_report_{time}.md` |
| `{op}_examples_failure_issue_{time}.md` | 失败问题 Issue 文件 | `reports/{date}/{repo}/issues/{op}_examples_failure_issue_{time}.md` |
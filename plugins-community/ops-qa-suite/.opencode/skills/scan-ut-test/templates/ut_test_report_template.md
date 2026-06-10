# {repo_type} 仓库全量 UT 测试报告

**测试日期**: {date}
**仓库**: {repo_type} ({repo_description})
**测试命令**: `bash build.sh -u`
**UT 类型**: {ut_type}
**NPU 设备**: {npu_device}
**SOC 版本**: {soc_version}

---

## 测试执行状态：{execution_status}

### 问题汇总

| 序号 | 问题 | 严重程度 | 状态 |
|:---:|-----|:-------:|:----:|
{problem_summary_rows}

---

## 问题详情记录

### 问题 1：{problem_name_1}（{problem_status_1}）

#### 执行命令

```bash
{problem_command_1}
```

#### 问题描述

{problem_description_1}

**错误输出**：
```
{problem_error_output_1}
```

**根因分析**：
- {problem_root_cause_1}

**影响范围**：
- {problem_impact_1}

#### 问题解决方案（已解决）

**临时解决方法**：
```bash
{problem_solution_1}
```

**验证结果**：{problem_verification_1}

#### 问题修复建议（未解决）

**临时规避方案**：
```bash
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

## 测试统计结果

### op_host UT (tiling + infershape)

| 测试套件 | 测试用例数 | 通过数 | 状态 |
|---------|:---------:|:-----:|:----:|
{test_suite_rows}

### 总体统计

| 指标 | 数值 |
|-----|:----:|
| 总测试套件数 | {total_test_suites} |
| 总测试用例数 | {total_test_cases} |
| 执行并通过 | {passed_tests} |
| 段错误阻塞 | {blocked_tests} |
| 通过率 | {pass_rate}% |

### 被跳过/未执行的测试

| 类型 | 测试内容 | 原因 | 测试数量 |
|-----|---------|------|:-------:|
{skipped_tests_rows}

### 未执行的 UT 类型（因 op_host 失败）

| UT 类型 | 测试内容 | 状态 |
|---------|---------|:----:|
| op_api_ut | aclnn 接口测试 | ❌ 未执行 |
| op_kernel_ut | AscendC kernel 测试 | ❌ 未执行 |
| aicpu_op_kernel_ut | AICPU kernel 测试 | ❌ 未执行 |

---

## UT 测试类型说明

`bash build.sh -u` 会触发以下 4 种 UT 测试：

| UT 类型 | 测试内容 | 测试文件位置 | 依赖 |
|---------|---------|-------------|-----|
| op_host_ut | tiling 参数推导 + infershape shape 推导 | `tests/ut/op_host/` | BUILD_PATH |
| op_api_ut | aclnn 接口调用测试 | `tests/ut/op_api/` | ACLNN 库 |
| op_kernel_ut | AscendC kernel 实现（需 NPU） | `tests/ut/op_kernel/` | NPU 硬件 |
| aicpu_op_kernel_ut | AICPU kernel 实现 | `tests/ut/op_kernel_aicpu/` | AICPU 环境 |

---

## 报告元信息

| 字段 | 值 |
|-----|---|
| 报告生成时间 | {date} |
| 报告路径 | `reports/{date}/{repo}/ut-test-report_report_{time}.md` |
| 测试环境 | {test_environment} |
| 测试结果 | {test_result_summary} |

---

## 模板变量说明

| 变量名 | 说明 | 示例值 |
|-------|------|-------|
| `{repo_type}` | 仓库类型 | ops-cv |
| `{repo_description}` | 仓库描述 | 计算机视觉算子仓库 |
| `{date}` | 测试日期 | 2026-04-27 |
| `{ut_type}` | UT 类型 | full, op_host, op_api, op_kernel, cpu_only |
| `{npu_device}` | NPU 设备型号 | 910B, 950, 310P 或 "无" |
| `{soc_version}` | SOC 版本 | ascend910b, ascend950 或 "不适用" |
| `{execution_status}` | 执行状态 | 部分成功（存在阻塞问题） |
| `{problem_summary_rows}` | 问题汇总表格行 | `| 1 | BUILD_PATH 缺失 | 🔴 阻塞 | ✅ 已解决 |` |
| `{total_test_suites}` | 总测试套件数 | 47 |
| `{total_test_cases}` | 总测试用例数 | 190 |
| `{passed_tests}` | 通过测试数 | 167 |
| `{blocked_tests}` | 阻塞测试数 | 23 |
| `{pass_rate}` | 通过率 | 87.9 |
| `{test_environment}` | 测试环境 | GCC 15.2, CANN 9.0.0 |
| `{test_result_summary}` | 测试结果摘要 | 部分成功（167/190 通过） |
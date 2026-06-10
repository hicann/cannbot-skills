---
description: Execute full UT test for specified repository and generate test report
---

执行全量 UT 测试，分析目标：$ARGUMENTS

如果没有指定仓库，默认测试 ops-nn。

---

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `repo` | 仓库名称（必填） | ops-cv, ops-math, ops-nn, ops-transformer |
| `--scope {scope}` | 执行范围（默认 full） | full, sample |
| `--ut_type {type}` | UT 类型（默认 full） | full, op_host, op_api, op_kernel, aicpu_kernel, cpu_only |
| `--skip_prompt` | 跳过用户询问（agent 触发时使用） | 无参数标志 |
| `--soc={version}` | 指定芯片版本（用于 op_kernel_ut） | ascend910b, ascend950 |

### UT 类型说明

| UT 类型 | 执行内容 | 依赖环境 | Agent 命令 |
|---------|---------|---------|---------|
| **full**（默认） | 全量 UT（4 种） | NPU（可选） | `--skip_prompt --ut_type full` |
| **op_host** | 仅 op_host_ut | BUILD_PATH | `--skip_prompt --ut_type op_host` |
| **op_api** | 仅 op_api_ut | ACLNN 库 | `--skip_prompt --ut_type op_api` |
| **op_kernel** | 仅 op_kernel_ut | **NPU 硬件** | `--skip_prompt --ut_type op_kernel` |
| **aicpu_kernel** | 仅 aicpu_kernel_ut | AICPU 环境 | `--skip_prompt --ut_type aicpu_kernel` |
| **cpu_only** | 仅 op_host + op_api | 无需 NPU | `--skip_prompt --ut_type cpu_only` |

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-ut-test_report_{HHMMSS}.md
        └── issues/
            └── ut_failure_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 测试完成后，发现阻塞问题

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 段错误阻塞 | Bug-Report | ✅ 自动生成 |
| 测试执行失败 | Bug-Report | ✅ 自动生成 |

---

## 报告格式要求

按统一报告模板生成测试报告（详见 `templates/unified_report_template.md`），包含问题详情和解决方案：

### 报告结构

```markdown
# {repo} 全量 UT 测试执行报告

## 报告元信息
- 测试日期、仓库、测试命令

## 执行摘要
- 测试执行状态
- 问题汇总表格

## 问题分类与统计
- 按问题类型统计
- 按严重程度统计

## 问题详情记录
- 每个问题的详细信息
- 执行命令、错误输出、根因分析
- 解决方案/修复建议

## 测试统计结果
- 测试套件表格
- 总体统计
- 被跳过测试说明

## UT 测试类型说明
- 4种 UT 类型说明

## GitCode Issue 文件
- 已生成的 Issue 文件列表

## 附录
- 测试环境详情
```

### 问题记录格式

| 字段 | 内容 |
|------|------|
| **执行命令** | 触发问题的具体命令 |
| **问题描述** | 错误输出、根因分析、影响范围 |
| **问题解决方案（已解决）** | 临时解决方法、验证结果 |
| **问题修复建议（未解决）** | 临时规避方案、排查建议、永久修复建议 |

---

## 执行步骤

### 0. 前置环境检查（新增）

**检测 NPU 设备和 SOC 版本**：

```bash
# 获取系统架构
arch=$(uname -m)
echo "[INFO] 系统架构: $arch"

# 检测 NPU 设备
npu_device=$(npu-smi info 2>/dev/null | grep -E "^[0-9]+" | awk '{print $2}' | head -1)

if [ -n "$npu_device" ]; then
    case "$npu_device" in
        "910B3"|"910B") soc_version="ascend910b" ;;
        "910_93") soc_version="ascend910_93" ;;
        "950"|"950PR"|"950DT") soc_version="ascend950" ;;
        "310P") soc_version="ascend310p" ;;
        "A3") soc_version="ascend-A3" ;;
        *) soc_version="ascend910b" ;;
    esac
    echo "[INFO] 检测到 NPU: $npu_device → SOC: $soc_version"
else
    echo "[INFO] 未检测到 NPU，仅执行 CPU UT（op_host_ut, op_api_ut）"
    cpu_only=true
fi
```

**如果 skip_prompt=false，询问用户执行方式**：

```
请确认 UT 测试执行方式：

当前环境检测：
- NPU 设备: {npu_device 或 "无"}
- SOC 版本: {soc_version 或 "不适用"}

UT 测试类型：
1. 全量执行 - 执行全部 4 种 UT（约 30-75 分钟）
2. 分步执行 - 选择特定 UT 类型
3. 仅 CPU UT - op_host_ut + op_api_ut（约 20-35 分钟）

请选择执行方式（1/2/3）：
``

### 1. 环境准备

```bash
# 清理旧的构建目录和 gcov 数据
cd $ARGUMENTS
rm -rf build
```

### 2. 构建 UT 测试

```bash
# 禁用自动执行测试，只构建
cd build
cmake -DENABLE_TEST=TRUE -DENABLE_UT_EXEC=OFF ..
make $ARGUMENTS_op_host_ut -j8
```

### 3. 执行测试并记录结果

```bash
# 设置必要的环境变量
export BUILD_PATH="$(pwd)/build"

# 执行测试
cd tests/ut/op_host
./$ARGUMENTS_op_host_ut 2>&1 | tee /tmp/ut_output.txt
```

### 4. 问题识别与分类

遇到问题时按以下逻辑处理：

| 问题类型 | 处理方式 |
|---------|---------|
| 🔴 阻塞问题（段错误） | 尝试修复；无法修复时记录并跳过，继续执行其他测试 |
| 🟡 非阻塞问题（警告） | 记录问题和解决方案 |
| ❌ 单个测试失败 | 记录失败测试名称 |

如果遇到段错误阻塞，使用规避方案继续执行：
```bash
# 跳过问题测试套件
./$ARGUMENTS_op_host_ut --gtest_filter="-{TestSuite}*" 2>&1 | tee /tmp/ut_skip_output.txt
```

### 5. 生成报告与 Issue 文件

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

```bash
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# 创建报告目录
mkdir -p reports/${date_str}/${repo}/issues

# 报告路径
report_file="reports/${date_str}/${repo}/scan-ut-test_report_${time_str}.md"

# Issue 文件路径
issue_file="reports/${date_str}/${repo}/issues/ut_failure_issue_${time_str}.md"

# 使用统一模板生成报告
# 模板路径: templates/unified_report_template.md

# 为阻塞问题自动生成 Issue
# Issue 类型: Bug-Report
```

---

## 完成后确认

- [ ] 已清理 build 目录
- [ ] 已设置 BUILD_PATH 环境变量
- [ ] 已执行全量 UT 测试
- [ ] 已识别并分类所有问题（阻塞/非阻塞、已解决/未解决）
- [ ] 已按标准格式记录问题详情
- [ ] 已统计测试结果（测试套件、通过数、失败数）
- [ ] 已生成 `reports/{date}/{repo}/scan-ut-test_report_{time}.md`
- [ ] 已生成 Issue 文件（阻塞问题自动生成）
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

---

## 示例用法

### 用户手动执行

- `/scan-ut-test ops-cv` - 执行 ops-cv 全量 UT 测试并生成报告（交互式）
- `/scan-ut-test ops-nn --scope sample` - 抽样测试 ops-nn
- `/scan-ut-test ops-math --ut_type cpu_only` - 仅执行 CPU UT（无需 NPU）
- `/scan-ut-test ops-math --ut_type op_host` - 仅执行 op_host_ut

### Agent 自动触发

- `/scan-ut-test ops-math --skip_prompt --ut_type full --scope full` - unified-scanner 全量 UT
- `/scan-ut-test ops-math --skip_prompt --ut_type cpu_only --scope full` - 无 NPU 环境全量 UT
- `/scan-ut-test ops-math --skip_prompt --ut_type op_host --scope full` - 仅 op_host 测试

---

## 输出报告路径

`reports/{date}/{repo}/scan-ut-test_report_{time}.md`

例如：`reports/20260512/ops-cv/scan-ut-test_report_173045.md`
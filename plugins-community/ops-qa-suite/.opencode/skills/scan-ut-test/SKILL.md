---
name: scan-ut-test
description: Ascend C 算子仓库全量 UT 测试执行与报告生成技能。用于执行 ops-math/ops-nn/ops-transformer/ops-cv 仓库的全量 UT 测试（op_host/op_api/op_kernel），记录测试结果，识别阻塞问题，按标准格式生成测试报告，自动为阻塞问题生成 Issue 文件。核心原则：1) 所有问题都创建 Issue；2) 报告后询问提交；3) 同类问题合并选项。当用户需要执行全量 UT 测试、分析测试失败原因、生成 UT 测试报告时使用。
---

# 仓库全量 UT 测试报告

## 概述

本技能用于执行 Ascend C 算子仓库的全量 UT 测试，并按标准格式生成测试报告，帮助：
- 执行全量 UT 测试并收集结果
- 识别阻塞和非阻塞问题
- 提供问题解决方案和修复建议
- 输出结构化测试报告
- **生成 Issue 文件并询问提交**（针对所有问题）

---

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── ut-test-report_report_{time}.md        # 测试报告
        ├── issues/
        │   └── ut_failure_issue_{time}.md        # Issue 文件
        └── his/                                   # 历史报告归档
```

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv"] | 根据用户指定或工作目录推断 |
| repo_root | 仓库根目录 | 绝对路径 | 默认为 `{repo_type}/`，支持自动检测（通过 repo_detector.py 从任意嵌套目录向上遍历查找 ops-* 仓库） |
| output_path | 报告输出路径 | 绝对路径 | 默认为 `reports/{date}/{repo}/ut-test-report_report_{time}.md` |
| **execution_scope** | **执行范围** | 枚举值["full", "sample"] | **默认 "full"（全量执行）** |
| **ut_type** | **UT 类型** | 枚举值["full", "op_host", "op_api", "op_kernel", "aicpu_kernel", "cpu_only"] | **默认 "full"（全量 UT）** |
| **skip_prompt** | **跳过用户询问** | 布尔值 | **默认 false**，agent 触发时设为 true |
| **soc_version** | **芯片版本** | 字符串 | 自动检测 NPU 或用户指定 |
| npu_device | NPU 设备型号 | 字符串 | 通过 `npu-smi info` 自动检测 |
| cpu_only | 仅执行 CPU UT | 布尔值 | 无 NPU 时自动设为 true |
| skip_tests | 跳过的测试套件 | 正则表达式 | 默认为空，遇到段错误时动态添加 |
| date_str | 日期字符串 | YYYYMMDD | 默认 `$(date +"%Y%m%d")` |
| time_str | 时间字符串 | HHMMSS | 默认 `$(date +"%H%M%S")` |

### 执行范围说明

| 执行范围 | 说明 | 适用场景 |
|---------|------|---------|
| **full**（默认） | 执行全部算子的 UT 测试 | 完整验证仓库 UT 状态 |
| **sample** | 抽样执行部分算子测试 | 快速验证环境是否正常 |

### UT 类型说明

| UT 类型 | 执行内容 | 依赖环境 | 预计耗时 |
|---------|---------|---------|:-------:|
| **full**（默认） | 全量 UT（4 种） | NPU（可选） | 30-75 分钟 |
| **op_host** | 仅 op_host_ut（tiling + infershape） | BUILD_PATH | 10-15 分钟 |
| **op_api** | 仅 op_api_ut（aclnn 接口测试） | ACLNN 库 | 10-20 分钟 |
| **op_kernel** | 仅 op_kernel_ut（AscendC kernel） | **NPU 硬件** | 15-30 分钟 |
| **aicpu_kernel** | 仅 aicpu_op_kernel_ut | AICPU 环境 | 5-10 分钟 |
| **cpu_only** | 仅 op_host_ut + op_api_ut | 无需 NPU | 20-35 分钟 |

### SOC 版本映射

| NPU Device | SOC Version |
|------------|-------------|
| 910B3, 910B | ascend910b |
| 910_93 | ascend910_93 |
| 950, 950PR, 950DT | ascend950 |
| 310P | ascend310p |
| A3 | ascend-A3 |
| * (default) | ascend910b |
| 无 NPU | 不设置 SOC |

### 默认询问机制（当 skip_prompt=false）

**触发条件**：
- 用户未明确指定执行范围或 UT 类型
- skip_prompt=false（非 agent 触发）
- 检测到 NPU 设备异常或无 NPU

**询问流程**：
```
请确认 UT 测试执行方式：

当前环境检测：
- NPU 设备: {npu_device 或 "无"}
- SOC 版本: {soc_version 或 "不适用"}
- 系统架构: {arch}

UT 测试类型：
1. 全量执行 - 执行全部 4 种 UT（op_host, op_api, op_kernel, aicpu_kernel）
   - 预计耗时：约 30-75 分钟（取决于算子数量和 NPU 性能）
   - 注意：op_kernel 和 aicpu_kernel 需要 NPU 硬件
   
2. 分步执行 - 选择特定 UT 类型执行：
   a) 仅 op_host_ut（tiling + infershape）- 约 10-15 分钟
   b) 仅 op_api_ut（aclnn 接口测试）- 约 10-20 分钟
   c) 仅 op_kernel_ut（AscendC kernel）- 需要 NPU，约 15-30 分钟
   d) 仅 aicpu_kernel_ut - 需要 AICPU 环境，约 5-10 分钟
   
3. 仅 CPU UT（无 NPU 环境推荐）- 执行 op_host_ut + op_api_ut

请选择执行方式（1/2/3）：
``

### Agent 触发参数（skip_prompt=true）

**参数组合**：
| 命令 | 执行内容 | SOC 参数 |
|------|---------|---------|
| `--ut_type full` | 全量 UT（4 种） | 自动检测或 `--soc=ascend910b` |
| `--ut_type op_host` | 仅 op_host_ut | 无需 SOC |
| `--ut_type op_api` | 仅 op_api_ut | 无需 SOC |
| `--ut_type op_kernel` | 仅 op_kernel_ut | 必须有 NPU 或指定 `--soc` |
| `--ut_type cpu_only` | 仅 op_host + op_api | 无需 SOC |

**时间戳获取**：
```bash
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")
# 例：date_str=20260427, time_str=173045
```

---

## UT 测试类型

`bash build.sh -u` 会触发以下 4 种 UT 测试：

| UT 类型 | 目标名称 | 测试内容 | 测试文件位置 | 依赖环境 |
|---------|---------|---------|-------------|---------|
| op_host_ut | `{repo}_op_host_ut` | tiling 参数推导 + infershape shape 推导 | `tests/ut/op_host/` | BUILD_PATH 环境变量 |
| op_api_ut | `{repo}_op_api_ut` | aclnn 接口调用测试 | `tests/ut/op_api/` | ACLNN 库 |
| op_kernel_ut | `{repo}_op_kernel_ut` | AscendC kernel 实现（需 NPU） | `tests/ut/op_kernel/` | NPU 硬件或仿真器 |
| aicpu_op_kernel_ut | `{repo}_aicpu_op_kernel_ut` | AICPU kernel 实现 | `tests/ut/op_kernel_aicpu/` | AICPU 环境 |

---

## 执行流程

### Step 0：前置环境检查（新增）

**检查 NPU 设备和系统架构，确定 UT 执行范围**：

```bash
# 获取系统架构
arch=$(uname -m)
echo "[INFO] 系统架构: $arch"

# 初始化变量
npu_available=false
soc_version=""
cpu_only=false
npu_device=""

# 检测 NPU 设备（使用 grep -oE 直接提取型号字符串）
if command -v npu-smi &> /dev/null; then
    # 直接提取已知型号字符串（不依赖表格格式）
    # npu-smi info 输出格式：| 5     910B3               | OK | ...
    # grep -oE 只输出匹配部分，避免表格格式影响
    npu_device=$(npu-smi info 2>/dev/null | grep -oE "(910B3|910B|910_93|950PR|950DT|950|310P|A3)" | head -1)
fi

if [ -n "$npu_device" ]; then
    npu_available=true
    case "$npu_device" in
        "910B3"|"910B") soc_version="ascend910b" ;;
        "910_93") soc_version="ascend910_93" ;;
        "950"|"950PR"|"950DT") soc_version="ascend950" ;;
        "310P") soc_version="ascend310p" ;;
        "A3") soc_version="ascend-A3" ;;
        *) soc_version="ascend910b" ;;  # 默认
    esac
    echo "[INFO] 检测到 NPU: $npu_device → SOC: $soc_version"
else
    cpu_only=true
    echo "[INFO] 未检测到 NPU 设备，仅执行 CPU UT（op_host_ut, op_api_ut）"
    echo "[INFO] 如需执行 op_kernel_ut，请确保 NPU 设备可用或使用 --simulator 参数"
fi

# 根据 ut_type 参数调整执行范围
# 如果 ut_type=cpu_only 或 无 NPU，自动跳过 op_kernel_ut 和 aicpu_kernel_ut
```

### Step 1：环境准备

```bash
# 清理旧的构建目录和 gcov 数据
cd {repo_root}
rm -rf build_out
```

### Step 2：构建 UT 测试

**推荐使用 build.sh 构建**（自动处理 CMake 参数）：

```bash
cd {repo_root}

# 根据 ut_type 参数选择构建命令
# build.sh 会自动设置正确的 CMake 参数（UT_TEST_ALL, OP_HOST_UT, OP_API_UT, OP_KERNEL_UT）

if [ "$ut_type" = "full" ]; then
    # 全量 UT：-u 设置 UT_TEST_ALL=TRUE，构建所有 UT 类型
    if [ -n "$soc_version" ]; then
        bash build.sh -u --noexec --soc=$soc_version
    else
        bash build.sh -u --noexec
    fi
elif [ "$ut_type" = "cpu_only" ]; then
    # 仅 CPU UT：op_host_ut + op_api_ut（无需 NPU）
    # 注意：build.sh -u 默认构建全量，需后续只执行 CPU 部分
    bash build.sh -u --noexec
elif [ "$ut_type" = "op_host" ]; then
    # 仅 op_host_ut
    bash build.sh --ophost_test --noexec
elif [ "$ut_type" = "op_api" ]; then
    # 仅 op_api_ut
    bash build.sh --opapi_test --noexec
elif [ "$ut_type" = "op_kernel" ]; then
    # 仅 op_kernel_ut（需要 SOC 参数）
    if [ -n "$soc_version" ]; then
        bash build.sh --opkernel_test --noexec --soc=$soc_version
    else
        echo "[ERROR] op_kernel_ut 需要 NPU 或 --soc 参数"
        exit 1
    fi
elif [ "$ut_type" = "aicpu_kernel" ]; then
    # 仅 aicpu_op_kernel_ut
    bash build.sh --opkernel_aicpu_test --noexec
fi

echo "[INFO] UT 构建完成"
```

**build.sh 参数说明**：

| 参数 | 作用 | 对应 CMake 变量 |
|------|------|----------------|
| `-u` | 全量 UT 构建 | `UT_TEST_ALL=TRUE` |
| `--ophost_test` | 仅 op_host_ut | `OP_HOST_UT=TRUE` |
| `--opapi_test` | 仅 op_api_ut | `OP_API_UT=TRUE` |
| `--opkernel_test` | 仅 op_kernel_ut | `OP_KERNEL_UT=TRUE` |
| `--opkernel_aicpu_test` | 仅 aicpu_op_kernel_ut | `OP_KERNEL_AICPU_UT=TRUE` |
| `--noexec` | 只构建不执行 | `ENABLE_UT_EXEC=FALSE` |
| `--soc={version}` | 指定芯片版本 | `ASCEND_COMPUTE_UNIT={version}` |

**构建产物位置**：
- 构建目录：`build_out/`（而非 `build/`）
- op_host_ut：`build_out/tests/ut/op_host/{repo}_op_host_ut`
- op_api_ut：`build_out/tests/ut/op_api/{repo}_op_api_ut`
- op_kernel_ut：`build_out/tests/ut/op_kernel/{repo}_op_kernel_ut_{soc_version}`

### Step 3：执行测试并记录结果

```bash
# 设置必要的环境变量（build.sh 构建产物在 build_out 目录）
export BUILD_PATH="{repo_root}/build_out"

# source CANN 环境变量（op_api_ut 和 op_kernel_ut 需要）
if [ -d "/home/developer/Ascend/cann-9.0.0" ]; then
    source /home/developer/Ascend/cann-9.0.0/bin/setenv.bash
fi

# 执行 op_host_ut
echo "[进度] 开始执行 op_host_ut 测试..."
cd {repo_root}/build_out/tests/ut/op_host
./{repo}_op_host_ut 2>&1 | tee /tmp/op_host_ut_output.txt
echo "[完成] op_host_ut 测试完成"

# 执行 op_api_ut（如果构建了）
if [ -f "{repo_root}/build_out/tests/ut/op_api/{repo}_op_api_ut" ]; then
    echo "[进度] 开始执行 op_api_ut 测试..."
    cd {repo_root}/build_out/tests/ut/op_api
    ./{repo}_op_api_ut 2>&1 | tee /tmp/op_api_ut_output.txt
    echo "[完成] op_api_ut 测试完成"
fi

# 执行 op_kernel_ut（如果构建了且有 NPU）
if [ -n "$soc_version" ] && [ -f "{repo_root}/build_out/tests/ut/op_kernel/{repo}_op_kernel_ut_${soc_version}" ]; then
    echo "[进度] 开始执行 op_kernel_ut 测试..."
    cd {repo_root}/build_out/tests/ut/op_kernel
    ./{repo}_op_kernel_ut_${soc_version} 2>&1 | tee /tmp/op_kernel_ut_output.txt
    echo "[完成] op_kernel_ut 测试完成"
fi
```

### Step 4：问题识别与分类

遇到问题时按以下逻辑处理：

| 问题类型 | 处理方式 | 记录位置 |
|---------|---------|---------|
| 🔴 阻塞问题（段错误） | 尝试修复；无法修复时记录并跳过 | 问题详情记录 |
| 🟡 非阻塞问题（警告） | 记录问题和解决方案 | 问题详情记录 |
| ❌ 单个测试失败 | 记录失败测试名称 | 测试统计表格 |

### Step 5：生成报告

使用模板生成标准化报告，输出到固定位置。

---

## 长时间执行处理策略（新增）

### 全量 UT 执行预计耗时

| UT 类型 | 预计耗时 | 依赖环境 |
|---------|:-------:|---------|
| op_host_ut | 10-15 分钟 | BUILD_PATH |
| op_api_ut | 10-20 分钟 | ACLNN 库 |
| op_kernel_ut | 15-30 分钟 | NPU 硬件 |
| aicpu_kernel_ut | 5-10 分钟 | AICPU 环境 |
| **总计（全量）** | **30-75 分钟** | 取决于算子数量和硬件性能 |

### 处理策略

**1. 进度提示**：
```bash
echo "=========================================="
echo "[进度] 开始执行 op_host_ut 测试..."
echo "[预计] 约 10-15 分钟"
echo "=========================================="

# 执行完成后
echo "[完成] op_host_ut 测试完成，用时 X 分钟"
echo "[进度] 开始执行 op_api_ut 测试..."
```

**2. 错误隔离**：
```bash
# 每个 UT 类型独立执行，失败不影响后续
set +e  # 临时禁用错误退出
./{repo}_op_host_ut 2>&1 | tee /tmp/op_host_output.txt
op_host_result=$?
set -e  # 重新启用

if [ $op_host_result -ne 0 ]; then
    echo "[警告] op_host_ut 失败，但继续执行后续 UT"
fi
# 继续执行 op_api_ut...
```

**3. 分步报告生成**：
```bash
# 每完成一个 UT 类型，生成阶段性报告片段
echo "## op_host_ut 测试结果" >> $report_file
echo "| 测试套件 | 用例数 | 通过数 | 状态 |" >> $report_file
# 添加结果表格...

# 最终合并所有片段生成完整报告
```

**4. 可中断支持**：
```bash
# 记录已执行的 UT 类型到状态文件
echo "op_host_ut=completed" >> $ut_status_file
echo "op_api_ut=completed" >> $ut_status_file

# 用户中断后可查看状态文件了解进度
# 重新执行时可选择跳过已完成的 UT 类型
```

---

## 问题分类体系

### 问题严重程度

| 级别 | 标识 | 说明 | 处理优先级 |
|:---:|:---:|-----|:---------:|
| 阻塞 | 🔴 | 导致测试无法执行或中断 | 高 |
| 非阻塞 | 🟡 | 不影响测试执行，但产生警告 | 中 |
| 信息 | 🟢 | 可忽略的提示信息 | 低 |

### 问题状态

| 状态 | 标识 | 说明 |
|:---:|:---:|-----|
| 已解决 | ✅ | 问题已修复，测试可正常执行 |
| 未解决 | ❌ | 问题待修复，提供临时规避方案 |

---

## 问题记录格式

每个问题按以下结构记录：

### 执行命令

```bash
# 触发问题的具体命令
cd {repo_root} && bash build.sh -u
```

### 问题描述

- **错误输出**：实际的错误信息
- **根因分析**：问题的根本原因
- **影响范围**：受影响的测试套件/测试用例数量

### 问题解决方案（已解决）

- **临时解决方法**：立即可用的解决方案
- **验证结果**：解决后的执行结果

### 问题修复建议（未解决）

- **临时规避方案**：跳过问题继续执行的方案
- **排查建议**：定位问题的具体步骤
- **永久修复建议**：根本性修复方案（可选）

---

## 常见问题类型

### 1. BUILD_PATH 环境变量缺失

| 字段 | 内容 |
|-----|------|
| 问题描述 | 测试可执行文件依赖 BUILD_PATH 环境变量，但 CMake 未正确传递 |
| 错误输出 | `getenv BUILD_PATH failed.` + `Segmentation fault` |
| 根因 | CMakeLists.txt 的 POST_BUILD 命令遗漏环境变量传递 |
| 解决方案 | 手动设置 `export BUILD_PATH="{repo_root}/build"` 或修改 CMakeLists.txt |

### 2. 段错误阻塞测试

| 字段 | 内容 |
|-----|------|
| 问题描述 | 特定测试套件执行时触发段错误 |
| 错误输出 | `Segmentation fault` |
| 影响范围 | 阻塞后续测试执行 |
| 规避方案 | 使用 `--gtest_filter="-{TestSuite}*"` 跳过 |
| 排查建议 | 使用 gdb 定位崩溃位置 |

### 3. gcov 版本不匹配

| 字段 | 内容 |
|-----|------|
| 问题描述 | build 目录残留旧版本 GCC 的覆盖率数据 |
| 错误输出 | `libgcov profiling error: Version mismatch` |
| 解决方案 | 清理 build 目录后重新构建 |

### 4. UT 顺序执行阻塞

|| 字段 | 内容 |
|-----|------|
|| 问题描述 | build.sh 的 for 循环顺序执行，第一个失败后终止 |
|| 根因 | build.sh 第 1349-1352 行无错误容忍机制 |
|| 规避方案 | 单独构建和运行各 UT 目标 |

### 5. GCC 15.2 缺少 `<algorithm>` 头文件（系统性问题）

|| 字段 | 内容 |
|-----|------|
|| 问题描述 | GCC 15.2 对 `<algorithm>` 头文件依赖更严格，使用 `std::find`、`std::sort`、`std::unique`、`std::any_of` 等函数但未显式包含头文件的代码编译失败 |
|| 错误输出 | `error: 'sort' is not a member of 'std'` 或 `error: no matching function for call to 'find(...)'` |
|| 根因分析 | GCC 15.2 不再通过 `<vector>`、`<list>` 等头文件间接包含 `<algorithm>`，需显式添加 |
|| 影响范围 | 系统性问题，影响 500+ 文件（ops-nn、ops-math 等仓库） |
|| 解决方案 | 添加 `#include <algorithm>` 到受影响文件 |

**修复脚本**：
```bash
# 批量修复缺少 <algorithm> 的文件
cd {repo_root}
for f in $(grep -rE "std::(find|sort|any_of|all_of|unique|find_if|count|count_if|lower_bound|upper_bound)" --include="*.cpp" --include="*.h" -l | grep -v third_party); do
  if ! grep -q "#include <algorithm>" "$f"; then
    sed -i '0,/#include/s/#include/#include <algorithm>\n#include/' "$f"
  fi
done
```

**已确认受影响的文件示例**：
- `math/bias_add_grad/op_host/bias_add_grad_infershape.cpp` (ops-math)
- `conversion/unsqueezev3/op_host/unsqueezev3_infershape.cpp` (ops-math)
- `matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_platform_common.h` (ops-nn)
- `conv/common/op_host/op_tiling/arch35/conv_base_numblocks_decision.cpp` (ops-nn)

---

## 报告输出规范

### 报告路径

```
reports/{date}/{repo}/ut-test-report_report_{time}.md
```

### 报告结构

```
1. 报告标题与元信息
2. 测试执行状态与问题汇总
3. 问题详情记录（按执行命令-问题描述-解决方案/修复建议格式）
4. 测试统计结果（测试套件表格 + 总体统计）
5. UT 测试类型说明
6. 报告元信息
```

---

## 输出模板

### Issue 文件格式

**必须使用 gitcode-issue-creator 的模板格式生成 Issue**。

调用 `generate_issue_md.py` 脚本：
```python
from generate_issue_md import generate_title, generate_description, generate_issue_md

# 标题格式
title = generate_title('bug-report', 'ops-math op_api_ut 9 个测试失败')
# 输出: [Bug-Report|缺陷反馈]: [AI 识别] ops-math op_api_ut 9 个测试失败

# Description 格式
description = generate_description(
    'bug-report',
    description='ops-math 仓库全量 UT 测试中，op_api_ut 共执行 3470 个测试用例，其中 9 个失败。',
    environment='**软件环境**:\n- CANN 版本: 9.0.0\n- 操作系统: ...\n\n**硬件环境**:\n- NPU 型号: 910B3\n\n**问题环境**:\n- 仓库: ops-math\n- 问题类型: op_api_ut\n- 失败数: 9',
    steps='1. 执行构建命令 `bash build.sh -u --noexec --soc=ascend910b`\n2. 执行 `./math_op_api_ut`',
    expected='所有测试用例通过，或芯片兼容性问题添加跳过判断',
    logs='失败测试列表表格...',
    notes='修复建议...',
)

# 生成 Issue 文件
generate_issue_md(
    repo='ops-math',
    template_type='bug-report',
    summary='ops-math op_api_ut 9 个测试失败',
    description=description,
    labels='bug-report',
    output_dir='./reports/{date}/{repo}/issues/',
    issue_suffix='ut_failure',
)
```

**标准 Issue 格式示例**：
```markdown
# [Bug-Report|缺陷反馈]: [AI 识别] ops-math op_api_ut 9 个测试失败

**标签**: `bug-report`

---

Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述

ops-math 仓库全量 UT 测试中，op_api_ut 共执行 3470 个测试用例，其中 9 个失败（失败率 0.26%）。

### Environment / 环境信息

**软件环境**:
- CANN 版本: 9.0.0
- 操作系统: Linux aarch64

**硬件环境**:
- NPU 型号: 910B3

**问题环境**:
- 仓库: ops-math
- 问题类型: op_api_ut
- 失败数: 9

### Steps to reproduce the issue / 重现步骤

1. 执行构建命令 `bash build.sh -u --noexec --soc=ascend910b`
2. 执行 `cd build/tests/ut/op_api && ./math_op_api_ut`

### Describe the expected behavior / 预期结果

所有测试用例通过，或芯片兼容性问题添加跳过判断。

### Related log / screenshot / 日志 / 截图

| # | 失败测试 | 失败原因 |
|:--:|---------|---------|
| 1 | l2_inplace_ne_scalar_test... | 910B3 芯片不支持 BF16 |
...

### Special notes for this issue/备注 (Optional / 选填)

修复建议：...

---

**提交地址**: https://gitcode.com/cann/ops-math/issues/new
```

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（在 `{Skill特殊字段区域}` 增加）：

| 字段名称 | 内容说明 |
|---------|---------|
| 测试统计结果 | 测试套件表格（套件名/用例数/通过数/状态） |
| 被跳过测试表格 | 类型/测试内容/原因/测试数量 |
| UT测试类型说明 | 4种UT类型说明表格（类型/内容/位置/依赖） |

### 详细模板

使用统一报告格式，报告内容必须完整嵌入 Issue，不引用外部文件。

---

## 快速使用

```opencode
# 执行 ops-cv 仓库全量 UT 测试（默认）
/ut-test-report ops-cv

# 执行 ops-nn 仓库全量 UT 测试
/ut-test-report ops-nn

# 抽样执行 UT 测试（快速验证环境）
/ut-test-report ops-cv --scope sample

# 仅执行 CPU UT（无需 NPU）
/ut-test-report ops-math --ut_type cpu_only

# 仅执行特定 UT 类型
/ut-test-report ops-math --ut_type op_host
/ut-test-report ops-math --ut_type op_api
/ut-test-report ops-math --ut_type op_kernel

# Agent 自动触发（跳过询问）
/ut-test-report ops-math --skip_prompt --ut_type full
/ut-test-report ops-math --skip_prompt --ut_type cpu_only

# 指定 SOC 版本（用于 op_kernel_ut）
/ut-test-report ops-math --ut_type op_kernel --soc=ascend910b
```

### 执行范围参数

| 参数 | 说明 |
|------|------|
| 无参数（默认） | **全量执行** - 执行全部算子的 UT 测试，会询问执行方式 |
| `--scope sample` | 抽样执行 - 仅执行部分算子验证环境 |
| `--ut_type {type}` | 指定 UT 类型（full, op_host, op_api, op_kernel, aicpu_kernel, cpu_only） |
| `--skip_prompt` | 跳过用户询问，直接执行（agent 触发时使用） |
| `--soc={version}` | 指定芯片版本（用于 op_kernel_ut） |

> **重要**：默认行为是询问用户执行方式。使用 `--skip_prompt` 可跳过询问直接执行。

---

## 参数组合示例

### 用户手动执行

| 场景 | 命令 |
|------|------|
| 交互式全量测试 | `/ut-test-report ops-math` |
| 快速验证环境 | `/ut-test-report ops-math --scope sample` |
| 仅测 CPU UT | `/ut-test-report ops-math --ut_type cpu_only` |
| 仅测特定类型 | `/ut-test-report ops-math --ut_type op_host` |

### Agent 自动触发

| 场景 | 命令 |
|------|------|
| unified-scanner 全量 UT | `/ut-test-report ops-math --skip_prompt --ut_type full --scope full` |
| 无 NPU 环境 | `/ut-test-report ops-math --skip_prompt --ut_type cpu_only --scope full` |
| 仅 op_host 测试 | `/ut-test-report ops-math --skip_prompt --ut_type op_host --scope full` |

---

## 报告示例

```markdown
# ops-cv 仓库全量 UT 测试报告

**测试日期**: 2026-04-27
**仓库**: ops-cv
**测试命令**: `bash build.sh -u`

## 测试执行状态：部分成功（存在阻塞问题）

### 问题汇总

| 序号 | 问题 | 严重程度 | 状态 |
|:---:|-----|:-------:|:----:|
| 1 | BUILD_PATH 环境变量缺失 | 🔴 阻塞 | ✅ 已解决 |
| 2 | ResizeNearestNeighborV2 段错误 | 🔴 阻塞 | ❌ 未解决 |

## 问题详情记录

### 问题 1：BUILD_PATH 环境变量缺失（已解决）
...
```

---

## 技能文件结构

```
.opencode/skills/ut-test-report/
├── SKILL.md                           # 技能描述与流程
└── ../../templates/                          # 模板统一到 .opencode/templates/ 目录
    └── examples_test_report_template.md      # 报告模板（共享）
```
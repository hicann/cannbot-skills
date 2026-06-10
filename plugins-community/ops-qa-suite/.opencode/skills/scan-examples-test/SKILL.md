---
name: scan-examples-test
description: Ascend C 算子仓库全量 examples 测试执行与报告生成技能。用于执行 ops-math/ops-nn/ops-transformer/ops-cv 仓库的全量算子 examples 测试（test_aclnn_*.cpp/test_geir_*.cpp），记录测试结果，识别失败问题，按标准格式生成测试报告，自动为失败问题生成 Issue 文件。核心原则：1) 所有问题都创建 Issue；2) 报告后询问提交；3) 同类问题合并选项。当用户需要执行全量 examples 测试、分析测试失败原因、生成 examples 测试报告时使用。
---

# 仓库全量 Examples 测试报告

## 概述

本技能用于执行 Ascend C 算子仓库的全量 examples 测试，并按标准格式生成测试报告：

- 下载并安装最新 CANN toolkit 和 ops 包（从官方 master 版本）
- 检测环境安装日期，避免同一天重复下载安装
- 扫描仓库下所有算子的 examples 目录，获取算子列表
- 对比最新 ops 包中的算子，识别未包含算子
- 执行算子样例测试，收集测试结果
- 生成测试报告，包含统计信息、问题列表
- **生成 Issue 文件并询问提交**（针对失败问题和未包含算子）

---

## 输出目录结构

**所有输出基于当前工作目录（`$(pwd)`）生成**：

```
$(pwd)/reports/                                   # 当前工作目录下的 reports
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── examples-test-report_report_{time}.md   # 测试报告（带时间戳，每次新建）
        ├── operator_list.txt                      # 算子列表（无时间戳，直接覆盖）
        ├── latest_ops_list.txt                    # 最新 ops 包算子列表（无时间戳，直接覆盖）
        ├── examples_list.csv                      # examples 文件列表（无时间戳，直接覆盖）
        ├── test_results_{time}.csv                # 测试结果（带时间戳，每次新建）
        ├── issues/
        │   └── examples_failure_issue_{time}.md   # Issue 文件（带时间戳）
        └── his/                                   # 历史报告归档

~/Ascend/cann/
└── install_info.txt                               # CANN 安装信息（无时间戳，直接覆盖）
```

### 文件覆盖规则

| 文件类型 | 时间戳 | 覆盖规则 |
|---------|:-----:|---------|
| 测试报告 `*_report_*.md` | ✅ 有 | 每次新建，不覆盖 |
| Issue 文件 `*_issue_*.md` | ✅ 有 | 每次新建，不覆盖 |
| 测试结果 `*_results_*.csv` | ✅ 有 | 每次新建，不覆盖 |
| 算子列表 `*_list.txt` | ❌ 无 | **直接覆盖** |
| examples 列表 `*_list.csv` | ❌ 无 | **直接覆盖** |
| 最新 ops 列表 `*_latest_ops.txt` | ❌ 无 | **直接覆盖** |
| 安装信息 `install_info.txt` | ❌ 无 | **直接覆盖** |

---

## 官方 CANN 包下载

### 下载链接

最新 master 版本 CANN 包下载地址：

```
https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/
```

### 需要下载的包

| 包名 | 说明 | 大小（约） |
|------|------|:---------:|
| `Ascend-cann-toolkit_${version}_linux-${arch}.run` | CANN toolkit 包 | 1.2 GB |
| `Ascend-cann-${soc}-ops_${version}_linux-${arch}.run` | CANN ops 包（运行态依赖） | 2.1 GB |

### 包命名规则

- `${version}`: CANN 版本号（如 9.0.0）
- `${arch}`: CPU架构（aarch64 或 x86_64）
- `${soc}`: NPU型号（910b、950、A3 等）

---

## 工作目录要求

**重要**: 本 Skill 必须在包含算子仓库的目录下执行，报告将生成在当前工作目录的 `reports/` 子目录下。

**正确的工作目录示例**:
- `/path/to/ops-qa-suite/`（包含 ops-math/ops-nn/ops-transformer/ops-cv 子目录）
- 或任何包含目标仓库的目录

**错误的工作目录**:
- Skill 开发目录（如 `.opencode/skills/examples-test-report/`）

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv"] | 自动检测：用户指定 > 当前目录判断 > 遍历父目录 > 子目录检测 |
| repo_root | 仓库根目录 | 绝对路径 | 自动检测：通过 `repo_detector.py` 扫描确定（支持任意嵌套深度） |
| reports_dir | 报告输出目录 | 绝对路径 | 默认为 `$(pwd)/reports/{date}/{repo}/` |
| cann_install_path | CANN 安装路径 | 绝对路径 | 默认 ~/Ascend/cann |
| force_install | 强制重新安装 | 布尔值 | 默认 false（检测当天是否已安装） |
| soc_version | 目标芯片 | 枚举值 | 根据当前 NPU 自动推断 |
| test_mode | 测试模式 | 枚举值["eager", "graph", "all"] | 默认 "eager"（aclnn 测试） |
| **execution_scope** | **执行范围** | 枚举值["full", "sample"] | **默认 "full"（全量执行）** |
| skip_ops | 跳过的算子列表 | 正则表达式 | 默认为空 |
| date_str | 日期字符串 | YYYYMMDD | 默认 `$(date +"%Y%m%d")` |
| time_str | 时间字符串 | HHMMSS | 默认 `$(date +"%H%M%S")` |

### 执行范围说明

| 执行范围 | 说明 | 适用场景 |
|---------|------|---------|
| **full**（默认） | 执行全部算子的 examples 测试 | 完整验证仓库 examples 状态 |
| **sample** | 抽样执行部分算子测试 | 快速验证环境是否正常 |

### 不确定时询问机制

**触发条件**：
- 用户未明确指定执行范围
- 检测到大量算子可能无法在当前 NPU 运行
- 时间或资源可能不足以完成全量测试
- 存在算子无 simulator 支持且当前 NPU 不支持

**询问流程**：
```
请确认 examples 测试执行范围：

当前仓库有 X 个算子：
- 支持 NPU: Y 个（可直接测试）
- 需要 simulator: Z 个（需指定 --simulator 参数）
- 无法测试: W 个（无 kernel 或无接口）

1. 全量执行（默认） - 执行全部支持的算子测试（约需 X 分钟）
2. 抽样执行 - 仅执行部分算子验证环境（约需 Y 分钟）

请选择执行范围（1/2）：
```

---

## 执行流程

### Step 0：前置环境检查

**检查 NPU 设备和系统架构**：

```bash
# 获取系统架构
arch=$(uname -m)
echo "[INFO] 系统架构: $arch"

# 获取当前 NPU 型号
npu_device=$(npu-smi info | grep -E "^[0-9]+" | awk '{print $2}' | head -1)
case "$npu_device" in
    "910B3"|"910B") soc_version="ascend910b" ;;
    "910_93") soc_version="ascend910_93" ;;
    "950"|"950PR"|"950DT") soc_version="ascend950" ;;
    "310P") soc_version="ascend310p" ;;
    "A3") soc_version="ascend-A3" ;;
    *) soc_version="ascend910b" ;;  # 默认
esac

echo "[INFO] NPU 型号: $npu_device → $soc_version"
```

---

### Step 1：检测 CANN 安装日期

**检查当天是否已安装 CANN 包，避免重复下载**：

```bash
cann_install_path="${HOME}/Ascend/cann"
install_info_file="${cann_install_path}/install_info.txt"
today=$(date +"%Y%m%d")

# 检查安装目录是否存在
if [ -d "${cann_install_path}" ] && [ -f "${cann_install_path}/bin/setenv.bash" ]; then
    # 读取安装日期
    if [ -f "$install_info_file" ]; then
        install_date=$(grep "install_date" "$install_info_file" | cut -d'=' -f2)
        install_version=$(grep "version" "$install_info_file" | cut -d'=' -f2)
        
        if [ "$install_date" == "$today" ]; then
            echo "[INFO] CANN 环境已于今天 ($today) 安装"
            echo "[INFO] 版本: $install_version"
            echo "[INFO] 跳过下载安装步骤，使用现有环境"
            skip_install=true
        else
            echo "[INFO] CANN 环境安装日期: $install_date (非今天)"
            echo "[INFO] 需要重新下载安装最新版本"
            skip_install=false
        fi
    else
        # 检查目录修改时间
        dir_mtime=$(stat -c %Y "${cann_install_path}" 2>/dev/null || stat -f %m "${cann_install_path}")
        dir_date=$(date -d "@$dir_mtime" +"%Y%m%d" 2>/dev/null || date -j -f "%s" "$dir_mtime" +"%Y%m%d")
        
        if [ "$dir_date" == "$today" ]; then
            echo "[INFO] CANN 目录修改日期为今天，可能已安装"
            echo "[INFO] 建议使用现有环境"
            skip_install=true
        else
            skip_install=false
        fi
    fi
else
    echo "[INFO] CANN 环境不存在，需要下载安装"
    skip_install=false
fi
```

---

### Step 2：下载最新 CANN 包（如果需要）

**仅当 skip_install=false 时执行**：

```bash
if [ "$skip_install" == "false" ]; then
    # 获取最新版本目录
    master_url="https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-mirror/software/master/"
    latest_dir=$(curl -s "$master_url" | grep -oP 'href="[^"]+/"' | tail -1 | sed 's/href="//;s/"//')
    
    # 构建下载链接
    toolkit_url="${master_url}${latest_dir}Ascend-cann-toolkit_9.0.0_linux-${arch}.run"
    ops_url="${master_url}${latest_dir}Ascend-cann-${soc_version}-ops_9.0.0_linux-${arch}.run"
    
    # 创建下载目录
    mkdir -p ~/Ascend/packages
    
    echo "[INFO] 开始下载 toolkit 包..."
    wget -q --show-progress -O ~/Ascend/packages/Ascend-cann-toolkit_9.0.0_linux-${arch}.run "$toolkit_url"
    
    echo "[INFO] 开始下载 ops 包..."
    wget -q --show-progress -O ~/Ascend/packages/Ascend-cann-${soc_version}-ops_9.0.0_linux-${arch}.run "$ops_url"
    
    echo "[INFO] 下载完成"
fi
```

---

### Step 3：安装 CANN 包（如果需要）

**仅当 skip_install=false 时执行**：

```bash
if [ "$skip_install" == "false" ]; then
    # 设置执行权限
    chmod +x ~/Ascend/packages/*.run
    
    # 创建安装目录
    mkdir -p "${cann_install_path}"
    
    # 安装 toolkit 包
    echo "[INFO] 安装 toolkit 包..."
    ~/Ascend/packages/Ascend-cann-toolkit_9.0.0_linux-${arch}.run \
        --install --install-path="${cann_install_path}"
    
    # 安装 ops 包
    echo "[INFO] 安装 ops 包..."
    ~/Ascend/packages/Ascend-cann-${soc_version}-ops_9.0.0_linux-${arch}.run \
        --install --install-path="${cann_install_path}"
    
    # 记录安装信息（无时间戳，直接覆盖）
    echo "install_date=${today}" > "${cann_install_path}/install_info.txt"  # 直接覆盖
    echo "version=9.0.0" >> "${cann_install_path}/install_info.txt"
    echo "soc=${soc_version}" >> "${cann_install_path}/install_info.txt"
    echo "arch=${arch}" >> "${cann_install_path}/install_info.txt"
    
    echo "[INFO] 安装完成"
fi
```

---

### Step 4：激活环境变量

```bash
# source CANN 环境
source "${cann_install_path}/bin/setenv.bash"

# 验证环境
echo "[INFO] ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "[INFO] ASCEND_OPP_PATH: $ASCEND_OPP_PATH"
```

---

### Step 5：获取最新 ops 包中的算子列表

**对比仓库算子与官方 ops 包中的算子**：

```bash
# 获取官方 ops 包中的算子列表
latest_ops_path="${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe/kernel/${soc_version}/ops_${repo_type}/"
latest_ops=$(ls "$latest_ops_path" 2>/dev/null)

echo "[INFO] 最新 ops 包中的 ops_${repo_type} 算子数: $(echo $latest_ops | wc -w)"

# 保存到文件（无时间戳，直接覆盖）
output_file="../reports/${date_str}/${repo}/latest_ops_list.txt"
echo "# 最新 CANN ops 包中的算子列表" > "$output_file"  # 直接覆盖
echo "# SOC: ${soc_version}" >> "$output_file"
echo "# 时间: ${timestamp}" >> "$output_file"
echo "$latest_ops" | tr ' ' '\n' >> "$output_file"
```

---

### Step 6：扫描仓库算子 examples 目录

```bash
cd {repo_root}

# 收集仓库中的算子列表（无时间戳，直接覆盖）
output_file="../reports/${date_str}/${repo}/operator_list.txt"
echo "# {repo} 算子 examples 列表" > "$output_file"  # 直接覆盖
echo "# 生成时间: ${timestamp}" >> "$output_file"
echo "# 格式: 算子名,分类" >> "$output_file"

find . -type d -name "examples" | grep -v third_party | grep -v build | grep -v experimental | while read dir; do
    op_name=$(basename $(dirname "$dir"))
    if [[ "$op_name" != "." && "$op_name" != "examples" ]]; then
        category=$(basename $(dirname $(dirname "$dir")))
        if [[ "$category" != "." ]]; then
            echo "${op_name},${category}"
        fi
    fi
done | sort | uniq >> "$output_file"

# 收集 examples 文件列表（无时间戳，直接覆盖）
output_csv="../reports/${date_str}/${repo}/examples_list.csv"
echo "test_type,op_name,file_path" > "$output_csv"  # 直接覆盖

find . -path "*examples*" -name "test_aclnn_*.cpp" | grep -v third_party | grep -v build | grep -v experimental | while read f; do
    op=$(echo "$f" | sed -E 's|.*/([^/]+)/examples/.*|\1|')
    echo "aclnn,$op,$f"
done >> "$output_csv"

find . -path "*examples*" -name "test_geir_*.cpp" | grep -v third_party | grep -v build | grep -v experimental | while read f; do
    op=$(echo "$f" | sed -E 's|.*/([^/]+)/examples/.*|\1|')
    echo "geir,$op,$f"
done >> "$output_csv"
```

---

### Step 7：执行 examples 测试

**核心改进：检测算子是否在最新 ops 包中**：

```bash
cd {repo_root}

# 创建测试结果文件（带时间戳，每次新建）
result_file="../reports/${date_str}/${repo}/test_results_${time_str}.csv"
echo "op_name,test_type,in_latest_ops,result,error_message,time_str" > "$result_file"

# 获取仓库算子列表
repo_ops=$(cat "../reports/${date_str}/${repo}/operator_list.txt" | grep -v "^#" | cut -d',' -f1)

# 测试每个算子
for op in $repo_ops; do
    # 检查是否在最新 ops 包中
    in_latest="no"
    if echo "$latest_ops" | grep -qw "$op"; then
        in_latest="yes"
    fi
    
    # 特殊处理：resize/upsample/roi 等系列算子可能在包中以不同名称存在
    # 需要根据实际情况调整匹配逻辑
    
    echo "[测试] $op (in_latest: $in_latest)"
    
    if [[ "$in_latest" == "no" ]]; then
        # 不在最新包中，标记为问题
        echo "$op,eager,no,not_in_latest,算子不在最新 CANN ops 包中,${timestamp}" >> "$result_file"
        echo "  ⚠️ 不在最新包中 → 标记为问题"
    else
        # 执行测试
        output=$(bash build.sh --run_example "$op" eager 2>&1)
        
        if echo "$output" | grep -q "execute samples success"; then
            echo "$op,eager,yes,success,,${timestamp}" >> "$result_file"
            echo "  ✅ 成功"
        elif echo "$output" | grep -qE "ERROR|fatal error|Segmentation fault"; then
            error=$(echo "$output" | grep -E "ERROR|fatal error" | head -1 | tr '\n' ' ')
            echo "$op,eager,yes,failure,\"$error\",${timestamp}" >> "$result_file"
            echo "  ❌ 失败: $error"
        else
            echo "$op,eager,yes,success,,${timestamp}" >> "$result_file"
            echo "  ✅ 成功 (implicit)"
        fi
    fi
done
```

---

### Step 8：生成测试报告

**报告结构**：

```markdown
# {repo} 仓库 Examples 测试报告

## 测试执行状态
- 执行时间: {timestamp}
- CANN 版本: 9.0.0 (master)
- 安装日期: {install_date}
- NPU 型号: {soc_version}

## 测试统计
| 分类 | 数量 |
|------|:----:|
| ✅ 成功（在最新包中） | N |
| ⚠️ 不在最新包中 | N |
| ❌ 失败（在包中） | N |

## 🔴 不在最新 ops 包中的算子
...

## ✅ 测试通过的算子
...
```

---

### Step 9：生成 Issue 文件

**必须使用 gitcode-issue-creator 的模板格式生成 Issue**。

调用 `generate_issue_md.py` 脚本：
```python
from generate_issue_md import generate_title, generate_description, generate_issue_md

# 标题格式
title = generate_title('bug-report', 'ops-math examples 测试失败')
# 输出: [Bug-Report|缺陷反馈]: [AI 识别] ops-math examples 测试失败

# Description 格式
description = generate_description(
    'bug-report',
    description='ops-math 仓库全量 examples 测试中，部分算子测试失败。',
    environment='...',
    steps='...',
    expected='...',
    logs='...',
    notes='...',
)

# 生成 Issue 文件
generate_issue_md(
    repo='ops-math',
    template_type='bug-report',
    summary='examples 测试失败',
    description=description,
    labels='bug-report',
    output_dir='./reports/{date}/{repo}/issues/',
    issue_suffix='examples_failure',
)
```

为两类问题生成 Issue：
1. **不在最新包中的算子** - 需要向官方反馈
2. **测试失败的算子** - 需要排查修复

---

## 测试命令对照表

| 仓库类型 | 测试方式 | 测试命令 |
|---------|---------|---------|
| ops-math | 使用 CANN 内置 | `bash build.sh --run_example ${op} eager` |
| ops-nn | 使用 CANN 内置 | `bash build.sh --run_example ${op} eager` |
| ops-transformer | 使用官方 ops 包 | `bash build.sh --run_example ${op} eager` |
| ops-cv | 使用官方 ops 包 | `bash build.sh --run_example ${op} eager` |

---

## 问题分类体系

### 问题严重程度

| 级别 | 标识 | 说明 | 处理优先级 |
|:---:|:---:|-----|:---------:|
| 阻塞 | 🔴 | 导致测试无法执行或中断 | 高 |
| 失败 | 🟠 | 测试执行但结果不符合预期 | 中 |
| **未包含** | ⚠️ | **算子不在官方 ops 包中** | **高**（需向官方反馈） |
| 警告 | 🟡 | 不影响测试执行，但产生警告 | 低 |

### 常见问题类型

#### 1. 算子未包含在官方 ops 包中

| 字段 | 内容 |
|-----|------|
| 问题类型 | ⚠️ 未包含 |
| 错误输出 | 算子在仓库有源码，但官方 ops 包中无 kernel |
| 原因 | 官方未发布该算子，或该算子不支持当前 SOC |
| 解决方案 | 向官方反馈，确认算子发布计划 |

#### 2. 环境变量未配置

| 字段 | 内容 |
|-----|------|
| 问题类型 | 阻塞 |
| 错误输出 | `cannot find -lopapi_xxx` |
| 原因 | 环境变量未正确设置 |
| 解决方案 | source CANN 环境 |

#### 3. 运行时错误

| 字段 | 内容 |
|-----|------|
| 问题类型 | 失败 |
| 错误输出 | `ERROR: 161001` 或 `ERROR: 361001` |
| 原因 | 算子不支持当前输入参数或架构 |
| 解决方案 | 检查算子 tiling 实现和 SOC 支持范围 |

---

## 快速使用

### 命令格式

```
/examples-test-report {repo} [--scope {execution_scope}] [--force] [--mode {test_mode}] [options]
```

### 参数说明

| 参数 | 说明 | 必填 | 示例 |
|------|------|:---:|------|
| `repo` | 仓库名称 | ✅ | ops-cv, ops-math, ops-nn, ops-transformer |
| `--scope {scope}` | **执行范围** | ❌ | full, sample（**默认 full 全量执行**） |
| `--force` | 强制重新下载安装 | ❌ | 默认检测当天是否已安装 |
| `--mode {type}` | 测试模式 | ❌ | eager, graph, all（默认 eager） |
| `--simulator` | 使用 simulator | ❌ | 需配合 `--soc` |
| `--soc={version}` | 仿真芯片型号 | ❌ | ascend950, ascend910b |

### 使用示例

```opencode
# ops-cv 全量测试（默认）
/examples-test-report ops-cv

# ops-cv 抽样测试（快速验证环境）
/examples-test-report ops-cv --scope sample

# ops-cv 强制重新安装测试
/examples-test-report ops-cv --force

# ops-math 直接测试（使用 CANN 内置）
/examples-test-report ops-math

# 使用 simulator 仿真
/examples-test-report ops-cv --simulator --soc=ascend950
```

> **重要**：默认行为是全量执行。如果遇到不确定情况（如大量算子无法在当前 NPU 运行），会询问用户确认执行范围。

---

## 执行流程图

```
┌─────────────────────────────────────────────────────────────┐
│                Step 0: 前置环境检查                          │
│  检查系统架构 → 获取 NPU 型号 → 确定 soc_version             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Step 1: 检测 CANN 安装日期                          │
│  检查 install_info.txt 或目录修改时间                        │
│  判断是否当天已安装 → skip_install?                          │
└─────────────────────────────┬───────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │                               │
        当天已安装                        需要安装
        skip_install=true               skip_install=false
              │                               │
              │                               ▼
              │               ┌─────────────────────────────┐
              │               │    Step 2: 下载最新 CANN 包  │
              │               │  toolkit(1.2GB) + ops(2.1GB)│
              │               └───────────────┬─────────────┘
              │                               │
              │                               ▼
              │               ┌─────────────────────────────┐
              │               │    Step 3: 安装 CANN 包     │
              │               │  安装到 ~/Ascend/cann       │
              │               │  记录 install_info.txt      │
              │               └───────────────┬─────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          Step 4: 激活环境变量                                │
│  source ~/Ascend/cann/bin/setenv.bash                       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    Step 5: 获取最新 ops 包算子列表                           │
│  ls ${ASCEND_OPP_PATH}/.../kernel/${soc}/ops_${repo}/       │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    Step 6: 扫描仓库算子 examples 目录                        │
│  find examples → 生成 operator_list.txt                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    Step 7: 执行 examples 测试                               │
│  对比最新包 → 在包中则测试，不在则标记"未包含"               │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│    Step 8-9: 生成测试报告和 Issue 文件                       │
└─────────────────────────────────────────────────────────────┘
```

---

## 文件清单

生成的报告文件：

```
reports/{date}/{repo}/
├── examples-test-report_report_{time}.md     # 测试报告（每次新建）
├── operator_list.txt                          # 仓库算子列表（直接覆盖）
├── latest_ops_list.txt                        # 官方 ops 包算子列表（直接覆盖）
├── examples_list.csv                          # examples 文件列表（直接覆盖）
├── test_results_{time}.csv                    # 测试结果（每次新建）
└── issues/
    └── examples_failure_issue_{time}.md      # Issue 文件

~/Ascend/cann/
└── install_info.txt                           # CANN 安装信息（直接覆盖）
```

> **注意**：无时间戳的文件会在每次测试时直接覆盖，确保始终是最新版本。

---

## 技能文件结构

```
.opencode/skills/examples-test-report/
├── SKILL.md                                # 技能描述与流程
└── ../../templates/                          # 模板统一到 .opencode/templates/ 目录
```

---

## 与其他 Skill 的关系

| Skill | 关系 | 说明 |
|------|------|------|
| examples-analysis-guide | 前置分析 | 用于分析 examples 缺失情况 |
| gitcode-issue-creator | Issue 生成 | 测试完成后调用此 Skill 生成 Issue 文件 |
| ut-test-report | 类似功能 | UT 测试执行与报告（不同测试目标） |

---

## 更新记录

- **2026-04-30**: 修复工作目录问题：明确工作目录要求，所有输出基于 `$(pwd)` 生成
- **2026-04-30**: 新增 `repo_root` 参数说明：默认为 `$(pwd)/{repo_type}/`
- **2026-04-30**: 新增 `reports_dir` 参数：明确报告输出路径基于当前工作目录
- **2026-04-30**: 明确文件覆盖规则：无时间戳文件直接覆盖，有时间戳文件每次新建
- **2026-04-30**: 重构流程，添加官方 CANN 包下载安装步骤
- **2026-04-30**: 新增安装日期检测逻辑，避免同一天重复下载
- **2026-04-30**: 新增"不在最新包中"问题分类，对比官方 ops 包算子
- **2026-04-30**: 更新流程图和测试统计维度
- **2026-04-29**: 重构执行流程，添加算子包构建/安装步骤
- **2026-04-29**: 新增多 CANN 环境管理说明
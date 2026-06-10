---
description: Execute full examples test for specified repository and generate test report
---

执行全量 examples 测试，分析目标：$ARGUMENTS

---

## 参数格式

**命令格式**：`/scan-examples-test {repo} [options]`

| 参数 | 说明 | 必填 | 示例 |
|------|------|:---:|------|
| `repo` | 仓库名称 | ✅ | ops-cv, ops-math, ops-nn, ops-transformer |
| `--mode {type}` | 测试模式 | ❌ | eager, graph（默认 all） |
| `--simulator` | 使用 simulator 仿真 | ❌ | 需配合 `--soc` |
| `--soc={version}` | 指定仿真芯片型号 | ❌ | ascend950, ascend910b, ascend310p |
| `skip-sim` | 跳过 simulator 测试 | ❌ | 只测试支持当前 NPU 的算子 |

---

## 参数解析

```bash
# 解析参数
args="$ARGUMENTS"
repo=$(echo "$args" | awk '{print $1}')

# 验证仓库参数
if [[ -z "$repo" ]]; then
    echo "[ERROR] 必须指定仓库参数"
    echo "用法: /scan-examples-test {repo} [options]"
    echo "示例: /scan-examples-test ops-cv"
    echo "      /scan-examples-test ops-cv --mode eager"
    echo "      /scan-examples-test ops-cv --simulator --soc=ascend950"
    echo "      /scan-examples-test ops-cv skip-sim"
    exit 1
fi

# 验证仓库有效性
valid_repos="ops-cv ops-math ops-nn ops-transformer"
if ! echo "$valid_repos" | grep -qw "$repo"; then
    echo "[ERROR] 无效的仓库: $repo"
    echo "支持的仓库: $valid_repos"
    exit 1
fi

# 解析选项参数
options=$(echo "$args" | cut -d' ' -f2-)

# 默认值
test_mode="all"          # 默认全部模式
use_simulator=false      # 默认不使用 simulator
soc_version=""           # 默认空
skip_sim=false           # 默认不跳过 simulator

# 解析选项
if echo "$options" | grep -q "skip-sim"; then
    skip_sim=true
    echo "[INFO] skip-sim 模式：只测试支持当前 NPU 的算子，禁用 simulator"
fi

if echo "$options" | grep -q "--simulator"; then
    use_simulator=true
    if echo "$options" | grep -qP "--soc=\w+"; then
        soc_version=$(echo "$options" | grep -oP "--soc=\w+" | sed 's/--soc=/')
        echo "[INFO] simulator 模式：使用 ${soc_version} 仿真测试"
    else
        echo "[ERROR] --simulator 需要配合 --soc 参数指定芯片型号"
        exit 1
    fi
fi

if echo "$options" | grep -qP "--mode \w+"; then
    test_mode=$(echo "$options" | grep -oP "--mode \w+" | sed 's/--mode //')
    if [[ "$test_mode" != "eager" && "$test_mode" != "graph" ]]; then
        echo "[ERROR] 无效的测试模式: $test_mode"
        echo "支持的模式: eager, graph, all（默认）"
        exit 1
    fi
    echo "[INFO] 测试模式: ${test_mode}"
fi

# 输出参数汇总
echo "========================================"
echo "参数解析结果:"
echo "  仓库: $repo"
echo "  测试模式: $test_mode"
echo "  使用 simulator: $use_simulator"
echo "  仿真芯片: ${soc_version:-auto}"
echo "  跳过 simulator: $skip_sim"
echo "========================================"
```

---

## 执行模式

| 模式 | 命令示例 | 测试范围 | 说明 |
|------|---------|---------|------|
| **全量测试** | `/scan-examples-test ops-cv` | eager + graph | 默认模式，自动判断是否用 simulator |
| **eager 模式** | `/scan-examples-test ops-cv --mode eager` | 仅 eager | ACLNN 接口测试 |
| **graph 模式** | `/scan-examples-test ops-cv --mode graph` | 仅 graph | GEIR 图引擎测试 |
| **simulator** | `/scan-examples-test ops-cv --simulator --soc=ascend950` | 仅 eager | 强制仿真，测试支持该芯片的算子 |
| **skip-sim** | `/scan-examples-test ops-cv skip-sim` | eager + graph | 禁用 simulator，只测支持当前 NPU 的 |

---

## 模式行为矩阵

| 模式 | 当前 NPU 支持 | 当前 NPU 不支持 | simulator |
|------|:---:|:---:|:---:|
| 全量测试（默认） | 直接测试 | 自动用 simulator | ✅ 自动启用 |
| --simulator --soc=X | 强制 simulator | 跳过 | ✅ 强制启用 |
| skip-sim | 直接测试 | **跳过测试** | ❌ 禁用 |

---

## 测试类型说明

| 测试类型 | 测试内容 | 测试文件 | simulator 支持 |
|---------|---------|---------|---------------|
| **eager** | ACLNN API 调用 | test_aclnn_*.cpp | ✅ 支持 |
| **graph** | GEIR 图引擎调用 | test_geir_*.cpp | ❌ **不支持** |

> **重要限制**：graph 模式不支持 simulator，使用 --simulator 时 graph 测试会跳过

---

## 执行步骤

### Step 0：环境准备（必须）

```bash
# Source CANN 环境变量（必须先执行）
source ${ASCEND_HOME_PATH}/bin/setenv.bash

# 或手动设置
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
export ASCEND_OPP_PATH=${ASCEND_HOME_PATH}/opp

# 验证环境变量
echo "ASCEND_HOME_PATH: $ASCEND_HOME_PATH"
echo "ASCEND_OPP_PATH: $ASCEND_OPP_PATH"

# 验证关键目录
ls ${ASCEND_HOME_PATH}/include/aclnn  # eager 模式需要
ls ${ASCEND_HOME_PATH}/include/graph  # graph 模式需要
```

### Step 1：扫描算子 examples 目录

```bash
cd $repo
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# 创建输出目录
mkdir -p reports/${date_str}/${repo}

# 生成算子列表（带仓库前缀）
output_file="reports/${date_str}/${repo}/operator_list.txt"
echo "# ${repo} 算子 examples 列表" > ${output_file}
echo "# 生成时间: ${date_str}_${time_str}" >> ${output_file}
echo "# 格式: 算子名,分类" >> ${output_file}

find . -type d -name "examples" | grep -v third_party | grep -v build | grep -v experimental | while read dir; do
    op_name=$(basename $(dirname $dir))
    if [[ "$op_name" != "." && "$op_name" != "examples" ]]; then
        category=$(basename $(dirname $(dirname $dir)))
        if [[ "$category" != "." ]]; then
            echo "${op_name},${category}"
        fi
    fi
done | sort | uniq >> ${output_file}

echo "算子列表已生成: ${output_file}"
```

### Step 2：检查 examples 文件

```bash
output_csv="reports/${date_str}/${repo}/examples_list.csv"
echo "# ${repo} examples 文件列表" > ${output_csv}
echo "# 生成时间: ${date_str}_${time_str}" >> ${output_csv}
echo "test_type,op_name,file_path" >> ${output_csv}

# 收集 ACLNN examples
find . -path "*examples*" -name "test_aclnn_*.cpp" | grep -v third_party | grep -v build | grep -v experimental | while read f; do
    op=$(dirname "$f" | xargs dirname | xargs basename)
    echo "aclnn,$op,$f"
done >> ${output_csv}

# 收集 GEIR examples
find . -path "*examples*" -name "test_geir_*.cpp" | grep -v third_party | grep -v build | grep -v experimental | while read f; do
    op=$(dirname "$f" | xargs dirname | xargs basename)
    echo "geir,$op,$f"
done >> ${output_csv}

echo "Examples 文件列表已生成: ${output_csv}"
```

### Step 3：获取当前 NPU 信息

```bash
# 获取当前 NPU 设备型号
if command -v npu-smi &> /dev/null; then
    npu_device=$(npu-smi info | grep -E "^[0-9]+" | awk '{print $2}' | head -1)
    case "$npu_device" in
        "910B3"|"910B") current_soc="ascend910b" ;;
        "910_93") current_soc="ascend910_93" ;;
        "950"|"950PR"|"950DT") current_soc="ascend950" ;;
        "310P") current_soc="ascend310p" ;;
        "310B") current_soc="ascend310b" ;;
        *) current_soc="unknown" ;;
    esac
    echo "当前 NPU 设备: $npu_device → $current_soc"
else
    current_soc="unknown"
    echo "未检测到 NPU 设备"
fi

# 检查算子 README 支持的芯片型号
check_chip_support() {
    local readme=$1
    local target_soc=$2
    
    # 提取产品支持情况表格
    support_section=$(grep -A 20 "## 产品支持情况" $readme 2>/dev/null | grep -E "Atlas|Ascend|Kirin|√|×")
    
    # 芯片型号匹配
    case "$target_soc" in
        "ascend910b"|"ascend910_93")
            if echo "$support_section" | grep -qE "Atlas A2.*√|Atlas A3.*√"; then return 0; fi ;;
        "ascend950")
            if echo "$support_section" | grep -qE "950.*√"; then return 0; fi ;;
        "ascend310p")
            if echo "$support_section" | grep -qE "Atlas 推理.*√"; then return 0; fi ;;
        "ascend310b")
            if echo "$support_section" | grep -qE "Atlas 200I.*√"; then return 0; fi ;;
    esac
    return 1
}
```

### Step 4：执行 examples 测试

```bash
# 创建测试结果文件
results_csv="reports/scan-examples-test/${repo}_test_results.csv"
echo "# ${repo} examples 测试结果" > ${results_csv}
echo "# 测试时间: ${timestamp}" >> ${results_csv}
echo "# 测试模式: ${test_mode}" >> ${results_csv}
echo "# simulator: ${use_simulator} (${soc_version:-auto})" >> ${results_csv}
echo "# skip-sim: ${skip_sim}" >> ${results_csv}
echo "# 当前 NPU: ${current_soc}" >> ${results_csv}
echo "op_name,test_type,result,use_simulator,soc_used,error_message" >> ${results_csv}

# 根据测试模式确定测试范围
if [[ "$test_mode" == "all" ]]; then
    test_types="eager graph"
elif [[ "$test_mode" == "eager" ]]; then
    test_types="eager"
elif [[ "$test_mode" == "graph" ]]; then
    test_types="graph"
fi

# 读取算子列表并逐个测试
tail -n +4 ${output_file} | while IFS=',' read op_name category; do
    readme_path=$(find . -path "*/${op_name}/README.md" | head -1)
    
    for test_type in $test_types; do
        # 检查是否有对应类型的 examples
        if [[ "$test_type" == "eager" ]]; then
            if ! grep -q "^aclnn,$op_name" ${output_csv}; then
                echo "${op_name},${test_type},skipped,false,$current_soc,无 eager examples" >> ${results_csv}
                continue
            fi
        elif [[ "$test_type" == "graph" ]]; then
            if ! grep -q "^geir,$op_name" ${output_csv}; then
                echo "${op_name},${test_type},skipped,false,$current_soc,无 graph examples" >> ${results_csv}
                continue
            fi
        fi
        
        # 判断测试策略
        # 1. skip-sim 模式：只测试支持当前 NPU 的算子
        if [[ "$skip_sim" == "true" ]]; then
            if ! check_chip_support "$readme_path" "$current_soc"; then
                echo "${op_name},${test_type},skipped,false,$current_soc,不支持当前 NPU (skip-sim)" >> ${results_csv}
                echo "⚠️ $op_name $test_type 不支持当前 NPU，跳过 (skip-sim)"
                continue
            fi
            use_sim=false
            test_soc="$current_soc"
            
        # 2. --simulator 模式：强制使用 simulator
        elif [[ "$use_simulator" == "true" ]]; then
            if [[ "$test_type" == "graph" ]]; then
                echo "${op_name},${test_type},skipped,true,$soc_version,graph 不支持 simulator" >> ${results_csv}
                echo "⚠️ $op_name graph 跳过（simulator 不支持 graph）"
                continue
            fi
            if ! check_chip_support "$readme_path" "$soc_version"; then
                echo "${op_name},${test_type},skipped,true,$soc_version,不支持 $soc_version 芯片" >> ${results_csv}
                echo "⚠️ $op_name 不支持 $soc_version，跳过"
                continue
            fi
            use_sim=true
            test_soc="$soc_version"
            
        # 3. 自动判断模式
        else
            if check_chip_support "$readme_path" "$current_soc"; then
                use_sim=false
                test_soc="$current_soc"
            else
                if [[ "$test_type" == "graph" ]]; then
                    echo "${op_name},${test_type},skipped,false,$current_soc,当前 NPU 不支持，graph 无法 simulator" >> ${results_csv}
                    echo "⚠️ $op_name graph 跳过（当前 NPU 不支持，graph 无法 simulator）"
                    continue
                fi
                use_sim=true
                test_soc="ascend950"  # 默认仿真芯片
                echo "⚠️ $op_name 不支持当前 NPU，使用 simulator --soc=$test_soc"
            fi
        fi
        
        # 执行测试
        echo "测试 $op_name ($test_type mode)..."
        if [[ "$use_sim" == "true" ]]; then
            bash build.sh --run_example $op_name $test_type --simulator --soc=$test_soc 2>&1 | tee /tmp/${op_name}_${test_type}.log
        else
            bash build.sh --run_example $op_name $test_type 2>&1 | tee /tmp/${op_name}_${test_type}.log
        fi
        
        # 判断测试结果
        if grep -q "execute samples success" /tmp/${op_name}_${test_type}.log; then
            echo "${op_name},${test_type},success,$use_sim,$test_soc," >> ${results_csv}
            echo "✅ $op_name $test_type 测试成功"
        elif grep -q "Segmentation fault" /tmp/${op_name}_${test_type}.log; then
            echo "${op_name},${test_type},failure,$use_sim,$test_soc,Segmentation fault" >> ${results_csv}
            echo "❌ $op_name $test_type 测试失败（段错误）"
        else
            error_msg=$(grep -E "ERROR|error" /tmp/${op_name}_${test_type}.log | head -1 | cut -c1-100)
            echo "${op_name},${test_type},failure,$use_sim,$test_soc,$error_msg" >> ${results_csv}
            echo "❌ $op_name $test_type 测试失败"
        fi
    done
done
```

### Step 5：生成测试报告

```bash
report_file="reports/scan-examples-test/${repo}_examples_test_report_${timestamp}.md"

# 统计测试结果
total=$(tail -n +8 ${results_csv} | wc -l)
success=$(tail -n +8 ${results_csv} | grep "success" | wc -l)
failure=$(tail -n +8 ${results_csv} | grep "failure" | wc -l)
skipped=$(tail -n +8 ${results_csv} | grep "skipped" | wc -l)
pass_rate=$(echo "scale=1; $success * 100 / ($total - $skipped)" | bc 2>/dev/null || echo "N/A")

echo "========================================"
echo "测试统计:"
echo "  总测试数: $total"
echo "  成功: $success"
echo "  失败: $failure"
echo "  跳过: $skipped"
echo "  通过率: ${pass_rate}%"
echo "========================================"

# 生成报告
cat > ${report_file} << EOF
# ${repo} 仓库全量 Examples 测试报告

**测试日期**: $(date +"%Y-%m-%d %H:%M:%S")
**仓库**: ${repo}
**测试模式**: ${test_mode}
**simulator**: ${use_simulator} (${soc_version:-auto})
**skip-sim**: ${skip_sim}
**当前 NPU**: ${current_soc}

---

## 测试统计

| 指标 | 数值 |
|-----|:----:|
| 总测试数 | $total |
| 成功 | $success |
| 失败 | $failure |
| 跳过 | $skipped |
| 通过率 | ${pass_rate}% |

### 按测试类型统计

| 测试类型 | 总数 | 成功 | 失败 | 跳过 |
|---------|:---:|:---:|:---:|:---:|
$(tail -n +8 ${results_csv} | awk -F',' '{print $2}' | sort | uniq -c | while read count type; do
    s=$(tail -n +8 ${results_csv} | grep "^.*,$type,success" | wc -l)
    f=$(tail -n +8 ${results_csv} | grep "^.*,$type,failure" | wc -l)
    sk=$(tail -n +8 ${results_csv} | grep "^.*,$type,skipped" | wc -l)
    echo "| $type | $count | $s | $f | $sk |"
done)

---

## 失败问题列表

$(tail -n +8 ${results_csv} | grep "failure" | awk -F',' '{print "- **" $1 "** (" $2 "): " $6}')

---

## 跳过测试列表

$(tail -n +8 ${results_csv} | grep "skipped" | awk -F',' '{print "- **" $1 "** (" $2 "): " $6}')

---

## 报告元信息

| 字段 | 值 |
|-----|---|
| 报告路径 | ${report_file} |
| 测试时间 | ${timestamp} |
| 测试环境 | CANN $(cat ${ASCEND_HOME_PATH}/version.info 2>/dev/null | head -1 || echo "unknown") |
EOF

echo "测试报告已生成: ${report_file}"
```

### Step 6：生成测试报告与 Issue 文件

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

```bash
# 报告路径
report_file="reports/${date_str}/${repo}/scan-examples-test_report_${time_str}.md"

# 使用统一模板生成报告（模板路径：templates/examples_test_report_template.md）
# ...

# Issue 自动生成规则
# 触发时机：测试完成后发现失败问题
# Issue 类型：Bug-Report
# Issue 文件命名：{op}_examples_failure_issue_{time_str}.md

# 创建 Issues 目录
mkdir -p reports/${date_str}/${repo}/issues

# 为失败的测试生成 Issue
tail -n +8 ${results_csv} | grep "failure" | while IFS=',' read op_name test_type result use_sim soc_used error_msg; do
    issue_file="reports/${date_str}/${repo}/issues/${op_name}_${test_type}_examples_failure_issue_${time_str}.md"
    
    # 调用 tool-gitcode-issue-creator Skill 生成 Issue
    # Issue 标题格式：[Bug-Report|缺陷反馈]: [AI 识别] {repo} {op} examples 测试失败
    
    cat > ${issue_file} << EOF
[Bug-Report|缺陷反馈]: [AI 识别] ${repo} ${op_name} examples 测试失败

### Describe the current behavior / 问题描述

算子 **${op_name}** 的 examples 测试失败。

- **仓库**: ${repo}
- **算子**: ${op_name}
- **测试类型**: ${test_type}
- **错误信息**: ${error_msg}
- **使用 simulator**: ${use_sim}
- **仿真芯片**: ${soc_used}

### Environment / 环境信息

- **测试模式**: ${test_mode}
- **CANN 版本**: $(cat ${ASCEND_HOME_PATH}/version.info 2>/dev/null | head -1 || echo "unknown")
- **当前 NPU**: ${current_soc}

### Steps to reproduce the issue / 重现步骤

\`\`\`bash
cd ${repo}
bash build.sh --run_example ${op_name} ${test_type}
\`\`\`

### Related log / screenshot / 日志 / 截图

\`\`\`
$(tail -30 /tmp/${op_name}_${test_type}.log)
\`\`\`

---

**提交地址**: https://gitcode.com/cann/${repo}/issues/new
**Issue 文件**: ${issue_file}
EOF
    
    echo "Issue 文件已生成: ${issue_file}"
done

# 输出 Issue 创建提示
echo ""
echo "========================================"
echo "Issue 创建提示"
echo ""
echo "测试发现 ${failure} 个失败问题，已生成 Issue 文件："
echo ""
tail -n +8 ${results_csv} | grep "failure" | awk -F',' '{print "| " NR " | reports/scan-examples-test/issues/{repo}_" $1 "_" $2 "_examples_failure_{timestamp}.md | [Bug-Report]: " $1 " " $2 " 测试失败 |"}'
echo ""
echo "是否需要提交 Issue？"
echo "1. 全部提交 - API 直接提交所有 Issue"
echo "2. 选择提交 - 选择部分 Issue 提交"
echo "3. 手动提交 - 我会自行复制内容提交"
echo "4. 暂不提交 - 仅保留 Issue 文件"
echo "========================================"
```

---

**提交地址**: https://gitcode.com/cann/${repo}/issues/new
EOF
    
    echo "Issue 文件已生成: ${issue_file}"
done
```

---

## 完成后确认

- [ ] 已 source CANN 环境变量
- [ ] 已扫描算子 examples 目录
- [ ] 已检查芯片型号兼容性
- [ ] 已按测试模式执行测试（eager/graph/all）
- [ ] 已收集测试结果
- [ ] 已生成测试报告
- [ ] 已生成 Issue 文件（失败问题自动生成）

---

## 示例用法

| 命令 | 说明 |
|------|------|
| `/scan-examples-test ops-cv` | 全量测试（eager + graph），自动判断 simulator |
| `/scan-examples-test ops-cv --mode eager` | 仅 eager 测试 |
| `/scan-examples-test ops-cv --mode graph` | 仅 graph 测试 |
| `/scan-examples-test ops-cv --simulator --soc=ascend950` | 强制 simulator，仿真 ascend950 |
| `/scan-examples-test ops-cv --mode eager --simulator --soc=ascend910b` | 仅 eager + simulator |
| `/scan-examples-test ops-cv skip-sim` | 禁用 simulator，只测支持当前 NPU 的算子 |

---

## 输出文件

| 文件 | 路径 |
|------|------|
| 算子列表 | `reports/{date}/{repo}/operator_list.txt` |
| examples 文件列表 | `reports/{date}/{repo}/examples_list.csv` |
| 测试结果 | `reports/{date}/{repo}/test_results.csv` |
| 测试报告 | `reports/{date}/{repo}/scan-examples-test_report_{time}.md` |
| Issue 文件 | `reports/{date}/{repo}/issues/{op}_{type}_examples_failure_issue_{time}.md` |

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

---

## 注意事项

1. **必须先 source CANN 环境变量**
2. **graph 模式不支持 simulator**，使用 --simulator 时 graph 测试会跳过
3. **skip-sim 禁用 simulator**，不支持当前 NPU 的算子直接跳过
4. **输出文件带仓库前缀**，避免多仓库冲突
5. **默认测试模式为 all**（eager + graph）
6. **报告生成后自动创建 Issue 文件**，失败问题自动生成 Bug-Report 类型 Issue
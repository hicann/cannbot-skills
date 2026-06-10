---
description: Scan CMake configuration issues for specified repository
---

执行 CMake 配置问题扫描，分析目标：$ARGUMENTS

如果没有指定仓库，默认扫描所有仓库。

支持的仓库类型：
- ops-math
- ops-nn
- ops-transformer
- ops-cv
- all（所有仓库）

可选参数：
- `--scan optype` - 仅扫描 OPTYPE 问题
- `--scan ut` - 仅扫描 UT 配置问题
- `--scan all` - 扫描所有问题类型（默认）

---

## 问题类型说明

扫描以下 9 种 CMake 配置问题：

| 问题类型 | 说明 | 示例 |
|---------|------|------|
| OPTYPE 与目录名不一致 | OPTYPE 参数值与算子目录名不匹配 | `OPTYPE moe_distribute_combine_v2`（应为 v3） |
| 函数不存在 | 使用未定义的函数 | ops-transformer 使用 `add_modules_llt_sources` |
| 变量不存在 | 使用不存在的变量 | `${OPTEST_NAME}` |
| 参数名错误 | 函数参数名与仓库定义不一致 | ops-math 使用 `HOSTNAME`（应为 `UT_NAME`） |
| if 语句语法错误 | 变量未用引号包裹导致空值语法错误 | `if(${target_dir} STREQUAL "")` |
| 目标名称冲突 | 多个算子共享相同的 CMake 目标名称 | v2/v3 版本目标重复 |
| 缺少源文件错误 | 测试模块未正确添加源文件 | `No SOURCES given to target` |
| 条件判断缺失 | 缺少 OP_HOST_UT 条件导致单独运行失败 | `if(TILING_UT OR OP_API_UT)` |
| 第三方依赖错误 | FetchContent 解压失败 | `include.zip` 解压失败 |

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-cmake_report_{HHMMSS}.md
        └── issues/
            └── cmake_error_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现 BUG 类型问题

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| OPTYPE 与目录名不一致 | Bug-Report | ✅ 自动生成 |
| 函数不存在 | Bug-Report | ✅ 自动生成 |
| 变量不存在 | Bug-Report | ✅ 自动生成 |
| 参数名错误 | Bug-Report | ✅ 自动生成 |
| if 语句语法错误 | Bug-Report | ✅ 自动生成 |
| 目标名称冲突 | Bug-Report | ✅ 自动生成 |
| 缺少源文件错误 | Bug-Report | ✅ 自动生成 |
| 条件判断缺失 | Bug-Report | ✅ 自动生成 |
| 第三方依赖错误 | Bug-Report | ✅ 自动生成 |

---

## 执行步骤

### Step 1: 执行扫描脚本

```bash
date_str=$(date +"%Y%m%d")
time_str=$(date +"%H%M%S")

# 创建报告目录
mkdir -p reports/${date_str}/${repo}/issues

# 执行扫描脚本生成 JSON 数据
python .opencode/skills/scan-cmake/scripts/cmake_scan.py --scan all --repo $ARGUMENTS --workspace .
```

### Step 2: 生成报告

```bash
# 报告路径
report_file="reports/${date_str}/${repo}/scan-cmake_report_${time_str}.md"

# 使用统一模板生成报告
# 模板路径: templates/unified_report_template.md
```

### Step 3: 生成 Issue 文件

```bash
# 为 BUG 类型问题自动生成 Issue
# Issue 文件命名: cmake_error_issue_{time_str}.md
# 调用 tool-gitcode-issue-creator Skill

# 输出 Issue 创建提示
echo "扫描发现 {n} 个 BUG 类型问题，已生成 Issue 文件"
```

---

## 报告格式要求

采用统一报告模板（详见 `templates/unified_report_template.md`），包含：

### 报告结构

```markdown
# {repo} CMake 配置问题扫描报告

## 报告元信息
- 扫描时间、仓库、报告类型

## 执行摘要
- 问题统计汇总
- 按问题类型统计
- 整体评分

## 问题分类与统计
- 按仓库统计
- 按问题类型统计
- 按严重程度统计

## 问题详情记录
- 每个问题的详细信息

## GitCode Issue 文件
- 已生成的 Issue 文件列表

## 修复建议
- 批量修复命令
- 手动修复清单

## 附录
- CMake 规范参考
```

### Issue 格式字段

| 字段 | 内容 |
|------|------|
| **标题** | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} {问题类型}导致 CMake 配置异常` |
| **问题类型** | OPTYPE不一致/函数不存在/变量不存在/参数名错误/if语句语法错误/目标冲突/缺少源文件/条件缺失/依赖错误 |
| **问题文件数** | 受影响的文件数量 |
| **问题描述** | 具体问题描述 |
| **修复建议** | 具体修复方案 |

---

## 完成后确认

- [ ] 已生成 `reports/{date}/{repo}/scan-cmake_report_{time}.md`
- [ ] 已生成 Issue 文件（BUG 类型问题自动生成）
- [ ] 已输出统计摘要表格（仓库 + 问题类型）
- [ ] 已输出问题文件列表
- [ ] 每个问题都包含修复建议
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

---

## 正确写法参考

### OPTYPE 规范

```cmake
# 算子目录: moe_distribute_combine_v3
add_modules_sources_with_soc(
    OPTYPE moe_distribute_combine_v3    # ✅ 正确：与目录名一致
    ACLNNTYPE aclnn_inner)
```

### UT CMakeLists 规范（ops-transformer/ops-math）

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

### if 语句规范

```cmake
# ✅ 正确：变量用引号包裹
if("${target_dir}" STREQUAL "")

# ❌ 错误：变量未用引号包裹，空值会导致语法错误
if(${target_dir} STREQUAL "")
```

---

## 示例用法

- `/scan-cmake` - 扫描所有仓库的所有 CMake 问题
- `/scan-cmake ops-nn` - 仅扫描 ops-nn 仓库
- `/scan-cmake ops-transformer --scan optype` - 仅扫描 OPTYPE 问题
- `/scan-cmake all --scan ut` - 扫描所有仓库的 UT 配置问题
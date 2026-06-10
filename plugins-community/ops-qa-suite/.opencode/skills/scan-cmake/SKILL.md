---
name: scan-cmake
description: CMake 配置问题扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 CMakeLists.txt 文件，检测 OPTYPE 参数错误、UT 构建配置问题、目标冲突、源文件缺失等问题。当用户询问 CMake 构建错误、UT 构建失败、CMakeLists 问题检测时使用。
---

# CMake 配置问题扫描

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── cmake-scan_report_{time}.md            # 扫描报告
        ├── issues/
        │   └── cmake_error_issue_{time}.md       # Issue 文件
        └── his/                                   # 历史报告归档
```

## 概述

本技能用于扫描 Ascend C 算子仓库的 CMake 配置问题，帮助检测以下问题类型：

---

## 问题类型说明

---

## ⚠️ 重要：问题性质分类

扫描结果会区分两类问题性质：

| 问题性质 | 说明 | 仓库 | 影响程度 | 处理建议 |
|----------|------|------|----------|----------|
| **BUG** | 会导致构建失败 | ops-transformer, ops-nn | ⚠️ 高 - 必须修复 | 立即修复函数/参数/变量错误 |
| **规范问题** | 遗留无效代码 | ops-math, ops-cv | 📋 低 - 不影响构建 | 建议删除或清理 |

### 问题性质判断依据

**tests/ut/op_host/CMakeLists.txt 是否会被引入构建系统：**

| 仓库 | 引入机制 | tests CMakeLists 是否被引入 | 问题性质 |
|------|----------|---------------------------|----------|
| ops-transformer | `add_subdirectory(OP/tests)` 链式引入 | ✅ **会被引入** | BUG |
| ops-nn | `add_subdirectory(OP/tests)` 链式引入 | ✅ **会被引入** | BUG |
| ops-math | `func.cmake:add_all_ut_sources` 直接 GLOB | ❌ **不会被引入** | 规范问题 |
| ops-cv | `func.cmake:add_all_ut_sources` 直接 GLOB | ❌ **不会被引入** | 规范问题 |

**ops-math/ops-cv 不需要 tests/ut/op_host/CMakeLists.txt 的原因：**
```cmake
# ops-math/cmake/func.cmake 第672-674行
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_host ...)
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${SOURCE_DIR}/tests/ut/op_host ...)
endif()

# 通过 file(GLOB) 直接查找源文件，不需要算子自己的 CMakeLists.txt
```

---

### 问题类型一：OPTYPE 与目录名不一致

**问题描述**：`op_host/CMakeLists.txt` 中的 `OPTYPE` 参数值与算子目录名不匹配。

**典型场景**：
1. 从其他算子复制 CMakeLists.txt 时，忘记修改 OPTYPE
2. 创建算子新版本（v2/v3）时，OPTYPE 未从旧版本更新
3. 模板文件中的示例 OPTYPE 未修改

**影响**：
- CMake 目标名称冲突
- 构建时可能产生链接错误或目标重复定义
- 算子功能可能被错误识别

**示例**：
```
算子目录: moe_distribute_combine_v3
op_host/CMakeLists.txt 内容:
    OPTYPE moe_distribute_combine_v2   ← 错误！应为 moe_distribute_combine_v3
```

---

### 问题类型二：函数名错误

**问题描述**：部分仓库的 `cmake/ut.cmake` 中未定义 `add_modules_llt_sources` 函数

| 仓库 | 是否定义 add_modules_llt_sources | 正确函数名 |
|------|--------------------------------|-----------|
| ops-transformer | ❌ 未定义 | `add_modules_ut_sources` |
| ops-math | ❌ 未定义 | `add_modules_ut_sources` |
| ops-nn | ✅ 定义（ut.cmake:307） | `add_modules_llt_sources` 或 `add_modules_ut_sources` |
| ops-cv | ❌ 未定义 | `add_modules_ut_sources` |

**影响**：执行 UT 构建时 CMake 报错找不到函数定义

---

### 问题类型三：变量名错误

**问题描述**：使用不存在的变量 `OPTEST_NAME`

**正确变量名**：

| 变量名 | 用途 | 定义位置 |
|--------|------|---------|
| `OP_TILING_MODULE_NAME` | tiling UT 模块名 | ut.cmake:27-30 |
| `OP_INFERSHAPE_MODULE_NAME` | infershape UT 模块名 | ut.cmake:31-34 |
| `OP_API_MODULE_NAME` | op_api UT 模块名 | ut.cmake:122-125 |
| `OP_KERNEL_MODULE_NAME` | op_kernel UT 模块名 | ut.cmake:152-155 |

**影响**：CMake 变量为空，导致 UT 源文件无法正确添加到构建目标

---

### 问题类型四：参数名错误

**问题描述**：函数参数名与仓库定义不一致

| 仓库 | 正确参数名 | 错误参数名 |
|------|-----------|-----------|
| ops-transformer | `UT_NAME` | `HOSTNAME` |
| ops-math | `UT_NAME` | `HOSTNAME` |
| ops-nn | `HOSTNAME` | `UT_NAME` |

**影响**：参数无法正确解析，UT 源文件路径错误

---

### 问题类型五：if 语句语法错误

**问题描述**：`cmake/ut.cmake` 中的 if 语句当变量为空时导致 CMake 解析失败

**典型错误**：
```cmake
# cmake/ut.cmake 第356行
if(${target_dir} STREQUAL "")   # ← 错误！当 target_dir 为空时展开为 if( STREQUAL "")
```

**正确写法**：
```cmake
if("${target_dir}" STREQUAL "")  # ← 正确！变量用引号包裹
```

**影响**：
- 当 `target_dir` 变量为空时，if 语句展开为 `if( STREQUAL "")`
- 缺少第一个参数导致 CMake 解析失败
- 配置中断，构建无法完成

---

### 问题类型六：CMake 目标名称冲突/重复定义

**问题描述**：多个算子版本（v2/v3）或不同算子使用相同的 CMake 目标名称

**典型场景**：
- `moe_distribute_combine_v2` 和 `moe_distribute_combine_v3` 共享相同目标名
- 同名目标如 `tiling_tmp`、`gen_head`、`cases_obj` 重复创建

**示例错误日志**：
```
CMake Error at cmake/ut.cmake:421:
  add_custom_target(transpose_v2_Ascend910B1_tiling_tmp)
  add_custom_target cannot create target "transpose_v2_Ascend910B1_tiling_tmp"
  because another target with the same name already exists.
```

**影响**：
- CMake 配置失败
- 构建中断，无法生成测试目标

---

### 问题类型七：缺少源文件错误

**问题描述**：测试模块的 CMakeLists.txt 未正确添加源文件路径，导致 `No SOURCES given to target` 错误

**典型场景**：
- `kv_rms_norm_rope_cache/tests/ut/` 的 CMakeLists.txt 缺少源文件配置
- 测试源文件路径未正确添加到构建目标

**影响**：
- CMake 配置阶段报错
- 无法生成测试可执行文件

---

### 问题类型八：条件判断缺少 OP_HOST_UT

**问题描述**：CMakeLists.txt 条件判断中缺少 `OP_HOST_UT`，导致单独运行 ophost UT 时源文件未被添加

**典型错误代码**：
```cmake
# 错误：缺少 OP_HOST_UT
if(TILING_UT OR PROTO_UT OR OP_API_UT OR (UT_TEST_ALL AND NOT AICPU_ONLY))
    add_modules_llt_sources(...)
endif()
```

**正确代码**：
```cmake
# 正确：包含 OP_HOST_UT
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()
```

**触发场景**：执行 `bash build.sh -u --ophost --ops=xxx` 或 `bash build.sh -u --ops=xxx` 时

---

### 问题类型九：第三方依赖解压失败

**问题描述**：CMake FetchContent/ExternalProject 下载依赖时解压失败

**典型场景**：
- 解压 `/build/third_party/pkg/include.zip` 失败
- 下载不完整、文件损坏、或 CMake 解压配置问题

**影响**：
- CMake 配置阶段失败
- 无法正确获取第三方依赖

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv", "all"] | 默认 "all" |
| scan_type | 扫描类型 | 枚举值["optype", "ut", "all"] | 默认 "all" |
| output_format | 输出格式 | 枚举值["terminal", "markdown", "issue"] | 默认 "markdown" |
| workspace | 工作空间根目录 | 绝对路径 | 默认为 **当前工作目录** |
| repo_root | 仓库根目录路径 | 绝对路径 | 自动检测（优先于 workspace） |

> **路径推断规则**：
> 1. `--repo-root` 显式指定时，直接使用该路径作为仓库根目录
> 2. 未指定 `--repo-root` 时，自动检测仓库位置（从当前目录向上遍历查找 ops-* 仓库）
> 3. 自动检测失败时，回退到 `{workspace}/{repo_type}/` 模式
>
> **检测优先级**：`--repo-root` > **自动检测** > `--workspace`

---

## 扫描流程

### Step 1: 确定扫描范围

根据 `repo_type` 参数确定扫描的仓库：

| repo_type | 扫描路径 |
|-----------|---------|
| ops-math | `{repo_root}/ops-math/`（repo_root 自动检测或通过 --repo-root 指定） |
| ops-nn | `{repo_root}/ops-nn/` |
| ops-transformer | `{repo_root}/ops-transformer/` |
| ops-cv | `{repo_root}/ops-cv/` |
| all | 以上所有仓库 |

> **仓库路径确定**：
> - `--repo-root` 指定 → 直接使用指定路径
> - 未指定 → 自动检测（从当前目录向上遍历查找 ops-* 仓库根目录）
> - 自动检测失败 → 回退到 `{workspace}/{repo_type}/` 模式（workspace 默认为当前工作目录）

### Step 2: 执行扫描脚本

根据 `scan_type` 选择扫描脚本：

| scan_type | 执行脚本 |
|-----------|---------|
| optype | `python scripts/cmake_scan.py --scan optype --repo {repo_type} --repo-root {repo_root}` |
| ut | `python scripts/cmake_scan.py --scan ut --repo {repo_type} --repo-root {repo_root}` |
| all | `python scripts/cmake_scan.py --scan all --repo {repo_type} --repo-root {repo_root}` |

> `{repo_root}` 由自动检测机制确定（优先级：`--repo-root` > 自动检测 > `--workspace`），如检测失败则回退到 `{workspace}/{repo_type}` 模式。

### Step 3: 分析扫描结果

检查并分类问题：
1. OPTYPE 不一致问题
2. UT 函数名错误
3. 变量名错误
4. 参数名错误
5. if 语句语法错误
6. 目标冲突问题
7. 源文件缺失问题
8. 条件判断缺失问题

### Step 4: 生成报告

根据 `output_format` 参数生成输出：

| output_format | 输出内容 |
|--------------|---------|
| terminal | 统计摘要 + 问题文件列表 |
| markdown | 完整 Markdown 报告（包含修复建议） |
| issue | GitCode Issue 格式报告 |

---

## 扫描脚本

本技能提供扫描脚本 `scripts/cmake_scan.py`：

### 脚本功能

- 扫描指定仓库的所有 CMakeLists.txt 文件
- 检测九种问题类型
- 输出 JSON 结构化数据和 Markdown 报告

### 脚本用法

```bash
python scripts/cmake_scan.py --scan {scan_type} --repo {repo_type} --workspace {workspace}

参数说明：
  --scan          扫描类型（optype/ut/all）
  --repo          仓库类型（ops-math/ops-nn/ops-transformer/ops-cv/all）
  --workspace     工作空间根目录（默认当前目录）
  --output        输出 JSON 文件路径（默认 reports/{date}/{repo}/cmake_issues.json）
  --report        输出 Markdown 报告路径（默认 reports/{date}/{repo}/cmake-scan_report_{time}.md）
```

### 脚本输出

```json
{
  "scan_time": "2026-04-22",
  "scan_type": "all",
  "repos": {
    "ops-transformer": {
      "total_files": 200,
      "issue_files": 5,
      "issues": [
        {
          "file": "attention/swin_attention_score_quant/tests/ut/op_host/CMakeLists.txt",
          "line": 20,
          "issue_type": "function_not_defined",
          "detail": "使用不存在的函数 add_modules_llt_sources",
          "suggestion": "替换为 add_modules_ut_sources"
        },
        {
          "file": "mc2/moe_distribute_combine_v3/op_host/CMakeLists.txt",
          "line": 19,
          "issue_type": "optype_mismatch",
          "op_name": "moe_distribute_combine_v3",
          "optype_value": "moe_distribute_combine_v2",
          "suggestion": "修改为 OPTYPE moe_distribute_combine_v3"
        }
      ]
    }
  },
  "summary": {
    "total_issue_files": 61,
    "by_repo": { ... },
    "by_issue_type": { ... }
  }
}
```

---

## 正确写法规范

### OPTYPE 参数规范

**原则**：OPTYPE 参数值必须与算子目录名完全一致（包括版本号后缀）。

**正确示例**：
```cmake
# 算子目录: ops-transformer/mc2/moe_distribute_combine_v3/op_host/
add_modules_sources_with_soc(
    OPTYPE moe_distribute_combine_v3   # ✅ 正确：与目录名一致
    ACLNNTYPE aclnn_inner)
```

**错误示例**：
```cmake
# 算子目录: ops-transformer/mc2/moe_distribute_combine_v3/op_host/
add_modules_sources_with_soc(
    OPTYPE moe_distribute_combine_v2   # ❌ 错误：与目录名不一致
    ACLNNTYPE aclnn_inner)
```

---

### ops-transformer / ops-math UT CMakeLists 模板

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

**要点**：
- 函数名：`add_modules_ut_sources`
- 参数名：`UT_NAME`
- 变量名：`OP_TILING_MODULE_NAME`、`OP_INFERSHAPE_MODULE_NAME`、`OP_API_MODULE_NAME`

---

### ops-nn UT CMakeLists 模板

```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_llt_sources(HOSTNAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_llt_sources(HOSTNAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

if(UT_TEST_ALL OR OP_API_UT)
    add_modules_llt_sources(HOSTNAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
```

**要点**：
- 函数名：`add_modules_llt_sources` 或 `add_modules_ut_sources`（ops-nn 都支持）
- 参数名：`HOSTNAME`
- 变量名：`OP_TILING_MODULE_NAME`、`OP_INFERSHAPE_MODULE_NAME`、`OP_API_MODULE_NAME`

---

### if 语句正确写法

**原则**：变量必须用引号包裹，避免空值导致语法错误

**正确写法**：
```cmake
if("${target_dir}" STREQUAL "")
    # 处理空值情况
endif()
```

**错误写法**：
```cmake
if(${target_dir} STREQUAL "")
    # 当 target_dir 为空时展开为 if( STREQUAL "")，导致语法错误
endif()
```

---

## 输出格式

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（在 `{Skill特殊字段区域}` 增加）：

| 字段名称 | 内容说明 |
|---------|---------|
| GitCode Issue 文件 | BUG类型问题列表（Issue标题/文件路径/提交地址） |
| 规范问题列表 | 遗留无效代码问题文件列表 |

### Issue 格式模板

每个问题的报告应遵循 GitCode Bug-Report 模板格式：

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述

{算子名} 的 CMake 配置存在 {问题类型} 问题：{详细问题描述}

问题文件：`{file}` 第 {line} 行
- 当前值：{current}
- 问题类型：{issue_type}

### Environment / 环境信息

**软件环境**:
- CANN 版本: {CANN版本，如 8.0.RC1、8.5.0 等}
- 操作系统: {OS版本，如 Ubuntu 22.04、CentOS 7.9 等}

**硬件环境**:
- NPU 型号: {芯片型号，如 Ascend910B1、Ascend910B2、Ascend310P 等}
- 服务器型号: {可选，如 A2、A3 服务器}

**问题环境**:
- 仓库: {repo_type}
- 算子: {算子名}
- 问题文件: {file}
- CMake 问题类型: {issue_type}
- 问题性质: {BUG/规范问题}

### Steps to reproduce the issue / 重现步骤

1. 进入仓库目录：`cd {repo_type}`
2. 构建项目：`bash build.sh -u`
3. 观察错误：CMake 配置阶段报错

### Describe the expected behavior / 预期结果

{suggestion}

### Related log / screenshot / 日志 / 截图

```
CMake Error at {file}:{line}:
  {错误日志}
```

### Special notes for this issue/备注

{影响说明}
```

---

## 输出检查项

完成扫描后，确保输出以下内容：

### 终端输出检查

| 检查项 | 内容 |
|-------|------|
| 统计摘要 | 各仓库问题文件数、按问题类型统计 |
| 问题文件列表 | 每个仓库的问题文件路径列表 |
| Issue 创建提示 | 是否生成 Issue 文件（针对 BUG 类型问题） |

### 报告文件检查

| 文件 | 位置 | 内容 |
|------|------|------|
| Markdown 报告 | `reports/{date}/{repo}/cmake-scan_report_{time}.md` | 完整报告（带时间戳） |
| JSON 数据文件 | `reports/{date}/{repo}/cmake_issues.json` | 结构化数据 |
| Issue 文件 | `reports/{date}/{repo}/issues/cmake_error_issue_{time}.md` | BUG 问题 Issue（自动生成） |

### Issue 创建流程

**核心原则**：
| 原则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别，所有发现问题都生成 Issue |
| **报告后询问提交** | 每次生成报告后，询问用户是否提交 Issue |
| **同类问题合并选项** | 同类问题涉及多个算子时，询问是否合并（按问题类型+仓库） |

**流程概览**：
```
扫描完成 → 分类问题 → 询问合并 → 生成 Issue → 询问提交 → 执行提交
```

**Issue 文件命名**：
- **合并模式**：`reports/{date}/{repo}/issues/cmake_error_merged_issue_{time}.md`
- **单算子模式**：`reports/{date}/{repo}/issues/{op_name}_cmake_error_issue_{time}.md`

**合并场景示例**：
- 发现 5 个算子 OPTYPE 错误 → 合并标题：`[Bug-Report]: [AI 识别] {repo} OPTYPE参数错误（5个算子）`
- 发现 3 个函数不存在 → 合并标题：`[Bug-Report]: [AI 识别] {repo} CMake函数不存在（3个算子）`

**询问提交流程**：
```
已生成 Issue 文件：
| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/{date}/{repo}/issues/cmake_error_merged_issue_{time}.md | [Bug-Report]: [AI 识别] {repo} CMake配置错误（5个算子） | 5 | cann/{repo} |

是否提交 Issue 到对应仓库？
1. 是，全部提交
2. 是，选择提交
3. 否，暂不提交
4. 否，手动提交
```

### 完整检查清单

- [ ] 已输出统计摘要表格
- [ ] 已输出各仓库问题文件列表
- [ ] 已生成 Markdown 报告（Issue格式）
- [ ] 已生成 JSON 数据文件
- [ ] 每个问题都包含修复建议

---

## 参考文档

- [references/cmake-spec.md](./references/cmake-spec.md) - CMake 正确写法规范（合并版）
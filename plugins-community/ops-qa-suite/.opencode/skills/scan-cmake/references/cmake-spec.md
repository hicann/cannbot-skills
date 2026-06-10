# CMake 正确写法规范（合并版）

## 概述

本文档描述 Ascend C 算子仓库 CMake 配置的正确写法规范，用于修复和规范 CMakeLists.txt 文件。

---

## 一、OPTYPE 参数规范

### 1.1 OPTYPE 参数位置

OPTYPE 参数通常出现在以下位置：

| 文件位置 | 使用场景 |
|---------|---------|
| `op_host/CMakeLists.txt` | 定义算子源文件构建目标 |
| `op_kernel/CMakeLists.txt` | 定义 kernel 构建目标 |
| `op_api/CMakeLists.txt` | 定义 API 构建目标 |

### 1.2 基本规则

**规则一：OPTYPE 必须与目录名一致**

```
算子目录结构：
ops-transformer/mc2/moe_distribute_combine_v3/op_host/

正确的 OPTYPE：
OPTYPE moe_distribute_combine_v3
```

**规则二：版本号必须完整**

```
算子目录：moe_distribute_combine_v3

正确的 OPTYPE：
OPTYPE moe_distribute_combine_v3   ✅

错误的 OPTYPE：
OPTYPE moe_distribute_combine      ❌ 缺少版本号
OPTYPE moe_distribute_combine_v2   ❌ 版本号错误
```

**规则三：所有 OPTYPE 位置必须一致**

同一算子在不同 CMakeLists.txt 中的 OPTYPE 应保持一致：

```
op_host/CMakeLists.txt:    OPTYPE moe_distribute_combine_v3
op_kernel/CMakeLists.txt:  OPTYPE moe_distribute_combine_v3
op_api/CMakeLists.txt:     OPTYPE moe_distribute_combine_v3
```

### 1.3 各仓库的 OPTYPE 使用方式

**ops-transformer op_host/CMakeLists.txt 示例**：

```cmake
if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_distribute_combine_v3_def.cpp
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OP_MC2_ENABLE ON
        OPTYPE moe_distribute_combine_v3    # ← 必须与目录名一致
        ACLNNTYPE aclnn_inner)
else()
    add_mc2_modules_sources(
        OPTYPE moe_distribute_combine_v3    # ← 必须与目录名一致
        ACLNNTYPE aclnn_inner)
endif()
```

**ops-math / ops-nn / ops-cv op_host/CMakeLists.txt 示例**：

```cmake
add_ops_host_sources(
    OPTYPE add                          # ← 必须与目录名一致
    ACLNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
)
```

---

## 二、UT CMake 配置规范

### 2.1 各仓库配置差异

**函数定义差异**：

| 仓库 | 定义的函数 | 位置 |
|------|----------|------|
| ops-transformer | `add_modules_ut_sources` | cmake/ut.cmake:200 |
| ops-math | `add_modules_ut_sources` | cmake/ut.cmake:267 |
| ops-nn | `add_modules_llt_sources` + `add_modules_ut_sources` | cmake/ut.cmake:307, 311 |
| ops-cv | `add_modules_ut_sources` | cmake/ut.cmake |

**参数名差异**：

| 仓库 | 正确参数名 | 函数签名 |
|------|-----------|---------|
| ops-transformer | `UT_NAME` | `add_modules_ut_sources(UT_NAME ...)` |
| ops-math | `UT_NAME` | `add_modules_ut_sources(UT_NAME ...)` |
| ops-nn | `HOSTNAME` | `add_modules_llt_sources(HOSTNAME ...)` |

### 2.2 正确变量名

| 变量名 | 用途 | 值示例 |
|--------|------|-------|
| `OP_TILING_MODULE_NAME` | tiling UT 模块名 | `transformer_op_tiling_ut` |
| `OP_INFERSHAPE_MODULE_NAME` | infershape UT 模块名 | `transformer_op_infershape_ut` |
| `OP_API_MODULE_NAME` | op_api UT 模块名 | `transformer_op_api_ut` |
| `OP_KERNEL_MODULE_NAME` | op_kernel UT 模块名 | `transformer_op_kernel_ut` |

**错误变量名**：`OPTEST_NAME`（不存在）

### 2.3 标准模板

**ops-transformer / ops-math 标准模板**：

```cmake
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

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

**ops-nn 标准模板**：

```cmake
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# ============================================================================

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

---

## 三、if 语句规范

### 3.1 变量必须用引号包裹

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

### 3.2 CMake 变量含义

| 变量 | 何时为 TRUE |
|------|------------|
| `UT_TEST_ALL` | 运行 `bash build.sh -u`（运行所有 UT） |
| `OP_HOST_UT` | 运行 `bash build.sh -u --ophost` 或 `bash build.sh -u --ophost_test` |
| `OP_API_UT` | 运行 `bash build.sh -u --opapi` 或 `bash build.sh -u --opapi_test` |
| `OP_KERNEL_UT` | 运行 `bash build.sh -u --opkernel` 或 `bash build.sh -u --opkernel_test` |
| `OP_GRAPH_UT` | 运行 `bash build.sh -u --opgraph` 或 `bash build.sh -u --opgraph_test` |

### 3.3 常见命令场景

| 命令 | 变量状态 |
|------|---------|
| `bash build.sh -u` | `UT_TEST_ALL=TRUE`, 其他为 FALSE |
| `bash build.sh -u --ophost --ops=xxx` | `OP_HOST_UT=TRUE`, `UT_TEST_ALL=FALSE` |
| `bash build.sh -u --ops=xxx` | `UT_TEST_ALL=TRUE`（默认运行所有 UT） |
| `bash build.sh -u --ophost_test` | `OP_HOST_UT=TRUE`, `UT_TEST_ALL=FALSE` |

### 3.4 条件判断原则

**原则**：使用 `OR` 组合条件，确保单独运行某类 UT 时也能正确添加源文件

**正确写法**：
```cmake
if(UT_TEST_ALL OR OP_HOST_UT)
    # 添加 op_host 相关源文件
endif()
```

**错误写法**：
```cmake
if(OP_API_UT OR (UT_TEST_ALL AND NOT AICPU_ONLY))
    # 缺少 OP_HOST_UT，单独运行 ophost UT 时不会添加源文件
endif()
```

---

## 四、常见问题与修复

### 4.1 OPTYPE 问题

**问题一：版本升级时未更新 OPTYPE**

```cmake
# 问题文件：moe_distribute_combine_v3/op_host/CMakeLists.txt
# 错误代码（第19行）
OPTYPE moe_distribute_combine_v2

# 修复方案
OPTYPE moe_distribute_combine_v3
```

**问题二：复制模板后未修改**

```cmake
# 问题文件：add/op_host/CMakeLists.txt
# 错误代码（模板示例值）
OPTYPE add_example

# 修复方案
OPTYPE add
```

### 4.2 UT 函数问题

**问题一：函数不存在**

症状：CMake 报错 `Unknown CMake command "add_modules_llt_sources"`

解决：
- ops-transformer/ops-math/ops-cv：替换为 `add_modules_ut_sources`
- ops-nn：可使用 `add_modules_llt_sources` 或 `add_modules_ut_sources`

**问题二：变量为空**

症状：UT 源文件未添加到构建目标

解决：替换 `${OPTEST_NAME}` 为 `${OP_TILING_MODULE_NAME}` 等正确变量

**问题三：参数解析错误**

症状：源文件路径错误或未正确添加

解决：
- ops-transformer/ops-math：使用 `UT_NAME`
- ops-nn：使用 `HOSTNAME`

**问题四：单独运行 UT 失败**

症状：运行 `bash build.sh -u --ophost --ops=xxx` 报错

解决：添加 `OP_HOST_UT` 到条件判断

### 4.3 if 语句问题

**问题：变量为空导致语法错误**

症状：CMake 报错 `if statement had incorrect arguments`

解决：变量必须用引号包裹

```cmake
# 错误
if(${target_dir} STREQUAL "")

# 正确
if("${target_dir}" STREQUAL "")
```

---

## 五、源文件添加逻辑

### 5.1 参数说明

| 参数 | 含义 | 示例 |
|------|------|------|
| `UT_NAME` / `HOSTNAME` | UT 模块名变量 | `${OP_TILING_MODULE_NAME}` |
| `MODE` | 添加模式 | `PRIVATE` |
| `DIR` | 源文件目录 | `${CMAKE_CURRENT_SOURCE_DIR}` |

### 5.2 根据模块名自动匹配源文件

函数内部会根据模块名自动搜索对应的源文件：

| 模块名包含 | 搜索的源文件模式 |
|-----------|-----------------|
| `tiling` | `test_*_tiling.cpp` |
| `infershape` | `test_*_infershape.cpp` |
| `op_api` | `test_aclnn_*.cpp` |
| `op_kernel` | `test_*.cpp` |

---

## 六、相关函数

### 6.1 add_modules_sources_with_soc

```cmake
add_modules_sources_with_soc(
    OP_API_INDEPENDENT ON
    OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
    OP_MC2_ENABLE ON
    OPTYPE {算子名}          # ← 必须与目录名一致
    ACLNNTYPE aclnn_inner
)
```

### 6.2 add_mc2_modules_sources

```cmake
add_mc2_modules_sources(
    OPTYPE {算子名}          # ← 必须与目录名一致
    ACLNNTYPE aclnn_inner
)
```

### 6.3 add_ops_host_sources

```cmake
add_ops_host_sources(
    OPTYPE {算子名}          # ← 必须与目录名一致
    ACLNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
)
```

---

## 七、扫描检测方法

### 7.1 OPTYPE 检测逻辑

```python
# Step 1: 从文件路径提取算子目录名
file_path = "mc2/moe_distribute_combine_v3/op_host/CMakeLists.txt"
op_name = "moe_distribute_combine_v3"  # op_host 前一个目录

# Step 2: 从文件内容提取 OPTYPE 值
pattern = r'OPTYPE\s+(\S+)'
optype_value = "moe_distribute_combine_v2"  # 提取到的值

# Step 3: 对比是否一致
if optype_value != op_name:
    # 发现问题
    issue = {
        "file": file_path,
        "op_name": op_name,
        "optype_value": optype_value,
        "suggestion": f"修改为 OPTYPE {op_name}"
    }
```

### 7.2 if 语句检测逻辑

```python
# 检测 if(${var} STREQUAL "") 模式
pattern = r'if\(\s*\$\{(\w+)\}\s*STREQUAL\s*"?\s*"?'
match = re.search(pattern, line)
if match:
    var_name = match.group(1)
    # 发现问题：变量未用引号包裹
    suggestion = f'修改为 if("${var_name}" STREQUAL "")'
```

---

## 八、最佳实践

### 8.1 创建新算子时

1. 使用正确的模板或参考正确的算子示例
2. 复制后立即修改 OPTYPE 参数
3. 检查所有 CMakeLists.txt 文件的 OPTYPE 是否一致
4. 确保使用正确的函数名和参数名

### 8.2 版本升级时

1. 复制旧版本目录
2. 重命名目录为新版本名
3. 更新所有 CMakeLists.txt 中的 OPTYPE 参数
4. 检查目标名称是否会冲突

### 8.3 定期检查

使用扫描脚本定期检查仓库：

```bash
python scripts/cmake_scan.py --workspace .
```

---

## 九、参考资料

### 9.1 ut.cmake 关键代码位置

**ops-transformer/cmake/ut.cmake**：
- `add_modules_ut_sources` 函数定义：第 200 行
- `OP_TILING_MODULE_NAME` 定义：第 27-30 行
- 参数名 `UT_NAME`：第 202-203 行

**ops-nn/cmake/ut.cmake**：
- `add_modules_llt_sources` 函数定义：第 307 行
- `add_modules_ut_sources` 函数定义：第 311 行
- 参数名 `HOSTNAME`：第 313 行
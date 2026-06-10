---
name: scan-ut-analysis
description: Ascend C 算子 UT 类型分析与缺失检测技能。用于分析 ops-math/ops-nn/ops-transformer/ops-cv 算子的 UT 需求，判断何时需要 _infershape/_tiling/op_kernel/op_api UT，识别 UT 缺失情况。当用户询问 UT 类型判断、UT 缺失分析、测试覆盖策略时使用。
---

# UT 类型分析与缺失检测

## 概述

本技能用于分析 Ascend C 算子的 UT 需求，帮助判断：
- 哪些算子需要 `_infershape` UT
- 哪些算子需要 `_tiling` UT
- 哪些算子需要 `op_kernel` UT
- 哪些算子需要 `op_api` UT
- 如何识别 UT 缺失情况

---

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── ut-analysis-guide_report_{time}.md     # 扫描报告
        ├── issues/
        │   └── ut_missing_issue_{time}.md        # Issue 文件
        └── his/                                   # 历史报告归档
```

---

## 分析模式

本技能支持两种分析模式：

### 模式一：基础分析（默认）

**适用场景**：快速扫描仓库所有算子的 UT 缺失情况

**分析方法**：仅检查目录结构，对比源文件与 UT 文件是否存在

**优点**：速度快，适合批量扫描

**缺点**：可能误判（部分纯模板调用的源文件不需要独立 UT）

### 模式二：详细分析

**适用场景**：精确分析特定算子或特定类型 UT 是否真的需要

**分析方法**：深入分析源文件实现逻辑，判断是否为纯模板调用

**触发条件**：
- 用户明确要求"详细分析"、"分析实现逻辑"、"判断是否真的需要 UT"
- 用户询问"这个算子是否需要 infershape/tiling UT"
- 用户对基础分析结果有疑问，要求精确判断

**注意**：如果用户表述模糊（如"帮我扫描一下 UT 缺失"），应先询问是否需要详细分析。

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| op_name | 算子名 | 采用下划线命名法 | 用户提供或当前分析的算子 |
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv", "others"] | 自动检测（通过 repo_detector.py 从工作目录向上遍历查找 ops-* 仓库根目录） |
| repo_root | 仓库根目录路径 | 有效的目录路径 | 自动检测；可通过 `--repo-root` 参数手动指定 |
| analysis_mode | 分析模式 | 枚举值["basic", "detailed"] | 默认 "basic"，用户明确要求时为 "detailed" |

---

## 基础分析流程

### 快速判断规则

通过检查算子目录结构，判断所需 UT 类型：

```
算子目录结构检查：
├── op_graph/*_proto.h                   → 无 IR 原型则不需要 infershape UT（重要）
├── op_host/*_infershape.cpp           → 有 IR 原型则需要公共 _infershape UT
├── op_host/arch*/*_infershape.cpp     → 有 IR 原型则需要对应架构的 _infershape UT
├── op_host/*_tiling.cpp               → 有则需要公共 _tiling UT
├── op_host/*_tiling_arch*.cpp         → 有则需要对应架构的 _tiling UT
├── op_host/arch*/*_tiling*.cpp        → 有则需要对应架构的 _tiling UT
├── op_kernel/*.cpp                    → 有则需要公共 op_kernel UT
├── op_kernel/arch*/*.cpp              → 有则需要对应架构的 op_kernel UT
├── op_api/aclnn_*.cpp                 → 有则需要 op_api UT（标准写法）
├── op_host/op_api/aclnn_*.cpp         → 有则需要 op_api UT（老写法）
└── op_api/{op}.cpp (无 aclnn 接口)    → 无 aclnn 接口，不需要 UT
```

> **infershape IR 原型说明**：infershape 用于 graph 模式的 shape 推导，只有存在 IR 原型文件（`op_graph/*_proto.h`）时才需要 infershape UT。没有 IR 原型的算子即使有 infershape 实现也暂时用不到，不需要 UT。

> **opapi 特殊说明**：只有 `aclnn_*.cpp` 接口文件才需要 UT，其他文件（如 `{op}.cpp`）不需要。opapi 源文件有两种位置：标准写法 `op_api/aclnn_*.cpp`，老写法 `op_host/op_api/aclnn_*.cpp`。

---

### tiling UT 详细判断规则（重要）

tiling 源文件有**三种位置**，对应不同的 UT 要求：

#### 源文件位置分类

| 源文件位置 | 类型 | 识别方式 |
|-----------|------|---------|
| `op_host/*_tiling.cpp` | **公共 tiling** | 文件名不含 `_arch*` 后缀 |
| `op_host/*_tiling_arch35.cpp` | arch35 特定 | 公共目录，但文件名含 `_arch35` |
| `op_host/arch35/*_tiling*.cpp` | arch35 特定 | 在 arch35 子目录下 |

#### UT 可接受位置

| 源文件类型 | 可接受的 UT 位置（三种均可） |
|-----------|-----------------------------|
| 公共 tiling (`*_tiling.cpp`，无 `_arch*`) | `tests/ut/op_host/test_*_tiling.cpp` (公共目录) |
| arch35 特定 tiling | `tests/ut/op_host/arch35/test_*_tiling_arch35.cpp` |
| | `tests/ut/op_host/arch35/test_*_tiling.cpp` |
| | `tests/ut/op_host/test_*_tiling_arch35.cpp` |

#### 同时有公共和 arch tiling 的场景

当算子**同时有公共 tiling 和 arch 特定 tiling** 时：

| 源文件 | 需要的 UT |
|--------|----------|
| `op_host/{op}_tiling.cpp` (公共) | `tests/ut/op_host/test_{op}_tiling.cpp` |
| `op_host/{op}_tiling_arch35.cpp` 或 `op_host/arch35/*` | 任一 arch35 UT 位置 |

**示例**：

```
算子：masked_select_v3
源文件：
  - op_host/masked_select_v3_tiling.cpp          ← 公共 tiling，需要公共 UT
  - op_host/masked_select_v3_tiling_arch35.cpp   ← arch35 tiling
UT：
  - tests/ut/op_host/test_masked_select_v3_tiling.cpp  ← 公共 UT（缺失！）
  - tests/ut/op_host/arch35/test_masked_select_v3_tiling.cpp ← arch UT ✓
```

#### 扫描判断逻辑

```python
# Step 1: 区分公共和 arch 源文件
公共源 = find op_host -maxdepth 1 -name '*_tiling.cpp' | grep -v '_arch'
arch源 = find op_host -name '*_tiling*arch*.cpp'   # 含 _arch 后缀或在 arch 目录

# Step 2: 检查 UT
公共UT = find tests/ut/op_host -maxdepth 1 -name 'test_*_tiling.cpp' | grep -v '_arch'
archUT位置1 = find tests/ut/op_host/arch35 -name 'test_*_tiling_arch35.cpp'
archUT位置2 = find tests/ut/op_host/arch35 -name 'test_*_tiling.cpp'
archUT位置3 = find tests/ut/op_host -name 'test_*_tiling_arch35.cpp'

# Step 3: 判断缺失
if 公共源 and not 公共UT:
    缺失公共 tiling UT
if arch源 and not (archUT位置1 or archUT位置2 or archUT位置3):
    缺失 arch tiling UT
```

---

### 统一判断规则（infershape/tiling/opapi/kernel 四种类型）

四种 UT 类型遵循相同的判断规则，核心原则是：**有源文件则必须有对应 UT**。

#### 源文件位置分类（四种类型通用）

| 源文件位置 | 类型 | 识别方式 |
|-----------|------|---------|
| `op_host/{op}_xxx.cpp` (不含 `_arch*` 后缀) | **公共源文件** | 文件名不含 `_arch*`，在公共目录 |
| `op_host/{op}_xxx_arch35.cpp` | arch35 特定 | 公共目录，但文件名含 `_arch35` |
| `op_host/arch35/{op}_xxx.cpp` | arch35 特定 | 在 arch35 子目录下 |
| `op_api/aclnn_*.cpp` | **opapi 标准位置** | 在 op_api 目录下 |
| `op_host/op_api/aclnn_*.cpp` | **opapi 老写法** | 在 op_host/op_api 子目录下 |

> **说明**：opapi 有两种源文件位置写法，标准写法在 `op_api/` 目录，老写法在 `op_host/op_api/` 目录。

#### UT 可接受位置（四种类型通用）

| 源文件类型 | 可接受的 UT 位置 |
|-----------|------------------|
| 公共源文件（不含 `_arch*`） | `tests/ut/op_host/test_{op}_xxx.cpp` (**必须公共目录**) |
| arch35 特定源文件（三种均可） | `tests/ut/op_host/arch35/test_{op}_xxx_arch35.cpp` |
| | `tests/ut/op_host/arch35/test_{op}_xxx.cpp` |
| | `tests/ut/op_host/test_{op}_xxx_arch35.cpp` |
| opapi 标准源文件 | `tests/ut/op_api/test_aclnn_*.cpp` (**必须 op_api 目录**) |
| opapi 老写法源文件（两种均可） | `tests/ut/op_api/test_aclnn_*.cpp` |
| | `tests/ut/op_host/op_api/test_aclnn_*.cpp` |

#### 同时有公共和 arch 源文件时的要求

当算子**同时有公共源文件和 arch 特定源文件**时：

| 源文件 | 需要的 UT |
|--------|----------|
| 公共源文件 | **必须**在公共 UT 目录有对应 UT |
| arch 特定源文件 | 任一 arch UT 位置均可 |

**示例**：

```
算子：masked_select_v3
源文件：
  - op_host/masked_select_v3_tiling.cpp          ← 公共源，必须公共 UT
  - op_host/masked_select_v3_tiling_arch35.cpp   ← arch35 源
UT：
  - tests/ut/op_host/test_masked_select_v3_tiling.cpp  ← 公共 UT（缺失！）
  - tests/ut/op_host/arch35/test_masked_select_v3_tiling.cpp ← arch UT ✓
结论: 缺失公共 tiling UT
```

#### 扫描判断逻辑（四种类型通用）

```python
# Step 1: 区分公共和 arch 源文件
公共源 = find op_host -maxdepth 1 -name '*_xxx.cpp' | grep -v '_arch'
arch源 = find op_host -name '*_xxx*arch*.cpp'   # 含 _arch 后缀或在 arch 目录

# Step 2: 检查 UT
公共UT = find tests/ut/op_host -maxdepth 1 -name 'test_*_xxx.cpp' | grep -v '_arch'
archUT位置1 = find tests/ut/op_host/arch35 -name 'test_*_xxx_arch35.cpp'
archUT位置2 = find tests/ut/op_host/arch35 -name 'test_*_xxx.cpp'
archUT位置3 = find tests/ut/op_host -name 'test_*_xxx_arch35.cpp'

# Step 3: 判断缺失
if 公共源 and not 公共UT:
    缺失公共 UT
if arch源 and not (archUT位置1 or archUT位置2 or archUT位置3):
    缺失 arch UT
```

#### 各类型具体对应

| UT类型 | IR原型条件 | 源文件位置 | UT 位置 |
|--------|-----------|-----------|--------|
| infershape | **必须有 op_graph/*_proto.h** | `op_host/*_infershape.cpp` | `tests/ut/op_host/test_*_infershape.cpp` |
| infershape (arch) | **必须有 op_graph/*_proto.h** | `op_host/*_infershape_arch35.cpp` 或 `op_host/arch35/*` | `tests/ut/op_host/arch35/test_*_infershape*.cpp` |
| tiling | - | `op_host/*_tiling.cpp` | `tests/ut/op_host/test_*_tiling.cpp` |
| tiling (arch) | - | `op_host/*_tiling_arch35.cpp` 或 `op_host/arch35/*` | `tests/ut/op_host/arch35/test_*_tiling*.cpp` |
| kernel | - | `op_kernel/*.cpp` (排除 `_def.cpp`) | `tests/ut/op_kernel/test_*.cpp` |
| kernel (arch) | - | `op_kernel/arch35/*.cpp` | `tests/ut/op_kernel/arch35/test_*.cpp` |
| opapi (标准) | - | `op_api/aclnn_*.cpp` | `tests/ut/op_api/test_aclnn_*.cpp` |
| opapi (老写法) | - | `op_host/op_api/aclnn_*.cpp` | `tests/ut/op_api/test_aclnn_*.cpp` 或 `tests/ut/op_host/op_api/test_aclnn_*.cpp` |

#### opapi 特殊说明

opapi 有两种源文件位置写法：

| 写法类型 | 源文件位置 | UT 位置 |
|---------|-----------|--------|
| **标准写法** | `op_api/aclnn_*.cpp` | `tests/ut/op_api/test_aclnn_*.cpp` (必须 op_api 目录) |
| **老写法** | `op_host/op_api/aclnn_*.cpp` | `tests/ut/op_api/test_aclnn_*.cpp` 或 `tests/ut/op_host/op_api/test_aclnn_*.cpp` |

> **判断逻辑**：
> - 标准写法的源文件，UT **必须**在 `tests/ut/op_api/` 目录
> - 老写法的源文件，UT 可在 `tests/ut/op_api/` 或 `tests/ut/op_host/op_api/` 任一位置

---

### op_api UT 判断规则（详见统一判断规则）

opapi 遵循统一判断规则，关键点：

| 源文件位置 | 是否需要 UT | UT 位置 |
|-----------|------------|--------|
| `op_api/aclnn_*.cpp` (标准写法) | ✅ 需要 | `tests/ut/op_api/test_aclnn_*.cpp` (必须) |
| `op_host/op_api/aclnn_*.cpp` (老写法) | ✅ 需要 | `tests/ut/op_api/` 或 `tests/ut/op_host/op_api/` |
| `op_api/{op}.cpp` (无 aclnn 接口) | ❌ 不需要 | - |

> **关键**：只有存在 `aclnn_*.cpp` 接口文件的算子才需要 op_api UT。其他如 `{op}.cpp` 不需要。

**示例**：
- `math/abs`: 有 `op_api/aclnn_abs.cpp` → 需要 `tests/ut/op_api/test_aclnn_abs.cpp` ✓
- `conversion/pad_v3_grad`: 只有 `op_api/padv3grad.cpp`，无 `aclnn_*` → 不需要 UT
- 某算子老写法：有 `op_host/op_api/aclnn_xxx.cpp` → UT 可在 `tests/ut/op_api/` 或 `tests/ut/op_host/op_api/`

---

### op_kernel UT 判断规则（详见统一判断规则）

op_kernel 遵循统一判断规则，关键点：

| 源文件位置 | 是否需要 UT | UT 位置 |
|-----------|------------|--------|
| `op_kernel/*.cpp` (排除 `_def.cpp`, `tilingdata.h`) | ✅ 需要 | `tests/ut/op_kernel/test_*.cpp` |
| `op_kernel/arch35/*.cpp` | ✅ 需要 | `tests/ut/op_kernel/arch35/test_*.cpp` |

> **注意**：排除 `_def.cpp`（定义文件）和 `tilingdata.h`（数据结构头文件），这些不是实际 kernel 实现。

---

### 架构特定目录说明

某些算子会针对不同芯片架构（arch20/arch32/arch35）提供特定实现：

| 源文件位置 | 对应 UT 位置 | 说明 |
|-----------|-------------|------|
| `op_host/arch35/*_tiling_arch35.cpp` | `tests/ut/op_host/arch35/test_*_tiling_arch35.cpp` | arch35 特定 tiling |
| `op_host/arch35/*_tiling.cpp` | `tests/ut/op_host/arch35/test_*_tiling.cpp` | arch35 tiling（无后缀） |
| `op_kernel/arch35/*.cpp` | `tests/ut/op_kernel/arch35/test_*.cpp` | arch35 kernel |

**注意**：
- `arch*` 子目录下的源文件**必须**有对应的 UT
- 源文件名可能带 `_arch*` 后缀，UT 文件名也应对应

### 基础分析步骤

**Step 1: 扫描算子目录结构**

```bash
# 检查 op_host 目录及 arch* 子目录
ls {算子目录}/op_host/
ls {算子目录}/op_host/arch20/ 2>/dev/null
ls {算子目录}/op_host/arch32/ 2>/dev/null
ls {算子目录}/op_host/arch35/ 2>/dev/null

# 检查 op_kernel 目录及 arch* 子目录
ls {算子目录}/op_kernel/
ls {算子目录}/op_kernel/arch35/ 2>/dev/null

# 检查 op_api 目录（标准写法）
ls {算子目录}/op_api/

# 检查 op_host/op_api 目录（老写法）
ls {算子目录}/op_host/op_api/ 2>/dev/null
```

**Step 2: 检查已有 UT**

```bash
# 检查 tests/ut 目录
ls {算子目录}/tests/ut/op_host/ 2>/dev/null
ls {算子目录}/tests/ut/op_host/arch*/ 2>/dev/null
ls {算子目录}/tests/ut/op_kernel/ 2>/dev/null
ls {算子目录}/tests/ut/op_kernel/arch*/ 2>/dev/null
ls {算子目录}/tests/ut/op_api/ 2>/dev/null
ls {算子目录}/tests/ut/op_host/op_api/ 2>/dev/null  # opapi UT 老写法位置
```

**Step 3: 对比生成报告**

将源文件与 UT 文件对比，输出缺失列表。

---

## 详细分析流程

### 不需要独立 UT 的场景

详细分析时，需要识别以下不需要独立 UT 的情况：

#### 1. infershape 纯模板调用

**特征**：
- 源文件行数 ≤ 25 行
- 无自定义 `InferDataType` 函数
- 只调用公共模板函数，无其他自定义逻辑

**公共模板函数列表**：

| 公共函数 | 功能 | 覆盖算子示例 |
|---------|------|------------|
| `InferShape4Elewise` | 逐元素操作 shape 推导 | sqrt, sin, cos, exp, erf, abs |
| `InferShape4Broadcast` | 广播操作 shape 推导 | add, div, mul, floor_div, gcd |
| `InferShape4Reduce` | 彠减操作 shape 推导 | reduce_sum, reduce_mean, reduce_prod |
| `InferShapeForArgOps` | Arg 操作 shape 推导 | arg_max_v2, arg_min, arg_max_with_value |
| `InferShape4Unary` | 单目操作 shape 推导 | neg, reciprocal |
| `InferShape4Binary` | 双目操作 shape 推导 | add, sub |

**纯模板调用示例**：

```cpp
// sqrt_infershape.cpp (仅 15 行)
#include "register/op_impl_registry.h"
#include "infershape_elewise_util.h"

using namespace ge;
namespace ops {
IMPL_OP_INFERSHAPE(Sqrt).InferShape(Ops::Base::InferShape4Elewise);
}
```

**判断标准**：
```python
# 检查条件
lines <= 25  # 行数少
无自定义 InferDataType  # 无特殊类型转换
仅调用公共模板函数  # IMPL_OP_INFERSHAPE().InferShape(公共函数)
```

**结论**：此类 infershape 不需要独立 UT，应在公共模块测试覆盖。

#### 2. tiling 极简实现

**特征**：
- 源文件行数 < 60 行
- 无实际 tiling 计算逻辑（无 TilingData、tilingKey、workspace 计算等）
- 仅做简单参数传递

**需要 UT 的 tiling 特征**：
- 有 `GetTilingKey()` 函数
- 有 `ComputeTiling()` 或类似计算函数
- 有 tilingData 参数计算逻辑
- 有 workspace 大小计算
- 行数 > 100 行通常有复杂逻辑

#### 3. kernel 全部需要 UT

`op_kernel` 目录下的所有源文件都需要 UT，因为涉及实际计算逻辑。

#### 4. api 全部需要 UT

`op_api` 目录下的所有源文件都需要 UT，因为涉及参数校验和接口测试。

### 详细分析步骤

**Step 1: 基础分析（获取初步结果）**

先执行基础分析流程，获取源文件和 UT 文件对比结果。

**Step 2: 深入分析 infershape**

```bash
# 检查 infershape 源文件行数
wc -l {算子目录}/op_host/*_infershape.cpp

# 检查是否有自定义 InferDataType
grep -c "InferDataType4" {算子目录}/op_host/*_infershape.cpp

# 检查是否调用公共模板函数
grep -E "InferShape4Elewise|InferShape4Broadcast|InferShape4Reduce" {算子目录}/op_host/*_infershape.cpp
```

**Step 3: 深入分析 tiling**

```bash
# 检查 tiling 源文件行数
wc -l {算子目录}/op_host/*_tiling*.cpp

# 检查是否有实际计算逻辑
grep -E "TilingData|tilingKey|GetTilingKey|ComputeTiling|workspace" {算子目录}/op_host/*_tiling*.cpp
```

**Step 4: 分类统计**

| 分类 | 判断标准 | 是否需要 UT |
|------|---------|------------|
| 纯模板 infershape | 行数≤25，无自定义逻辑，调用公共函数 | ❌ 不需要独立 UT |
| 有自定义逻辑 infershape | 有 InferDataType 或行数>50 或有其他自定义逻辑 | ✅ 需要 UT |
| 极简 tiling | 行数<60，无计算逻辑 | ⚠️ 可能不需要 |
| 复杂 tiling | 有 tilingKey/workspace 计算等 | ✅ 需要 UT |
| 所有 kernel | 有源文件 | ✅ 必须 UT |
| 所有 api | 有源文件 | ✅ 必须 UT |

---

## 四种 UT 类型详解

### 1. `_infershape` UT

**触发条件**：
- 存在 `op_host/*_infershape.cpp` 文件
- **同时存在 IR 原型文件 `op_graph/*_proto.h`**

> **重要说明**：infershape 用于 graph 模式的 shape 推导，只有在算子有 IR 原型定义时才有实际用途。没有 `op_graph/*_proto.h` 文件的算子，即使有 infershape 实现也暂时不需要 UT。

**需要测试的场景**：
- 有自定义 InferDataType 逻辑（如 complex → float 转换）
- 有复杂的 shape 推导逻辑
- 有动态 shape 处理
- 有 broadcast shape 特殊处理

**不需要测试的场景**：
- **没有 IR 原型文件（op_graph/*_proto.h）** - 暂时不需要
- 纯调用公共模板函数（InferShape4Elewise 等）
- 源文件仅 10-20 行无自定义逻辑

**文件位置**：`tests/ut/op_host/test_{op}_infershape.cpp`

---

### 2. `_tiling` UT

**触发条件**：
- 存在 `op_host/*_tiling.cpp` 文件
- 存在 `op_host/arch*/*_tiling*.cpp` 文件

**测试内容**：
- tilingKey 选择逻辑
- tilingData 参数计算
- workspace 大小分配
- 多 dtype/SOC 版本的 tiling 差异

**文件位置**：
- 通用实现：`tests/ut/op_host/test_{op}_tiling.cpp`
- arch35 实现：`tests/ut/op_host/arch35/test_{op}_tiling_arch35.cpp`

---

### 3. `op_kernel` UT

**触发条件**：存在 `op_kernel/*.cpp` 文件

**测试内容**：
- 实际计算正确性
- 端到端测试
- 多 dtype 计算精度验证

**文件位置**：`tests/ut/op_kernel/test_{op}.cpp`

---

### 4. `op_api` UT

**触发条件**：存在 `aclnn_*.cpp` 接口文件（两种位置）

| 源文件位置 | 说明 |
|-----------|------|
| `op_api/aclnn_*.cpp` | 标准写法 |
| `op_host/op_api/aclnn_*.cpp` | 老写法 |

**测试内容**：
- 参数校验
- dtype 支持列表验证
- SOC 版本兼容性

**UT 文件位置**：

| 源文件类型 | UT 位置 |
|-----------|--------|
| 标准写法 | `tests/ut/op_api/test_aclnn_*.cpp` (必须) |
| 老写法 | `tests/ut/op_api/test_aclnn_*.cpp` 或 `tests/ut/op_host/op_api/test_aclnn_*.cpp` |

> **注意**：只有 `aclnn_*.cpp` 文件才需要 UT，其他如 `{op}.cpp` 不需要。

---

## SOC 版本与 dtype 对应

| SOC 版本 | 架构 | dtype 差异 |
|---------|------|-----------|
| ascend310p | arch20 | 不支持 BF16 |
| ascend910b | arch32 | 支持 BF16 |
| ascend910_93 | arch32 | 支持 BF16 |
| ascend950 | arch35 | 支持 BF16 |

---

## 输出格式

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（在 `{Skill特殊字段区域}` 增加）：

| 字段名称 | 内容说明 |
|---------|---------|
| UT覆盖情况详情 | 完整覆盖算子列表 + 部分覆盖算子列表（含缺失类型） |
| 缺失UT文件算子 | 高优先级缺失 + 中优先级缺失表格 |
| UT类型分布统计 | infershape/tiling/kernel/api 四种类型统计 |

### 报告模板（Issue友好格式）

```markdown
# {repo_type} UT 缺失分析报告

**扫描时间**: {date}
**分析模式**: {basic/detailed}
**仓库路径**: `{repo_root}`

---

## 一、扫描概览

### 1.1 统计摘要

| UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |
|---------|----------|---------|----------|
| infershape | X | Y | Z% |
| tiling | X | Y | Z% |
| kernel | X | Y | Z% |
| api | X | Y | Z% |
| **总计** | X | Y | Z% |

### 1.2 按分类统计

| 分类 | infershape缺失 | tiling缺失 | kernel缺失 | api缺失 |
|------|---------------|-----------|-----------|--------|
| activation | X | Y | Z | W |
| conv | X | Y | Z | W |
| ... | ... | ... | ... | ... |

---

## 二、缺失详情汇总表

| 序号 | 算子路径 | 缺失类型 | 有源文件 | UT路径建议 |
|-----|---------|---------|---------|-----------|
| 1 | math/abs | kernel | op_kernel/abs.cpp | tests/ut/op_kernel/test_abs.cpp |
| 2 | math/add | infershape,tiling | ... | ... |

---

## 三、批量补充建议

### 3.1 按优先级排序

| 优先级 | UT类型 | 缺失数量 | 建议优先补充 |
|-------|--------|---------|-------------|
| P0 | kernel | X | 计算逻辑最关键 |
| P1 | api | Y | 接口测试重要 |
| P2 | tiling | Z | 参数计算验证 |
| P3 | infershape | W | shape推导验证 |

### 3.2 不需要独立 UT 的算子（详细分析模式）

| 算子 | 类型 | 原因 | 行数 |
|------|------|------|------|
| sqrt | infershape | 纯模板调用 InferShape4Elewise | 15 |
| ... | ... | ... | ... |

---

**报告生成时间**: {timestamp}
```

---

## 输出检查项

示例：
- `[Requirement|需求建议]: [AI 识别] 补充 abs 的 kernel UT`
- `[Requirement|需求建议]: [AI 识别] 补充 add 的 infershape UT`

#### 2. Description 字段映射

---

## 输出检查项

完成分析后，确保输出以下内容：

### 终端输出检查

| 检查项 | 内容 |
|-------|------|
| 统计摘要 | 四种 UT 类型的有源文件数、缺失数、缺失比例 |
| 分类统计 | 按算子分类统计缺失情况 |
| 缺失详情 | 缺失算子列表（路径 + 缺失类型），至少显示前 100 个 |

### 报告文件检查

| 文件 | 位置 | 内容 |
|------|------|------|
| Markdown 报告 | `reports/{date}/{repo}/ut-analysis-guide_report_{time}.md` | 完整报告（Issue格式） |
| JSON 数据文件 | `reports/{date}/{repo}/ut-analysis-guide_report_{time}.json` | 结构化数据 |

### 完整检查清单

- [ ] 已输出统计摘要表格
- [ ] 已输出分类统计表格
- [ ] 已输出缺失详情表格（至少前 100 个）
- [ ] 已生成 Markdown 报告（Issue格式）
- [ ] 已生成 JSON 数据文件
- [ ] 问题详情已按 Issue 格式组织

---

## 用户交互规则

### 模式判断

当用户请求 UT 分析时，根据表述判断分析模式：

| 用户表述 | 分析模式 | 说明 |
|---------|---------|------|
| "扫描 UT 缺失"、"检查 UT 情况" | basic | 默认基础分析 |
| "详细分析 UT 缺失"、"分析实现逻辑" | detailed | 明确要求详细 |
| "这个算子是否需要 UT"、"精确判断" | detailed | 需要精确判断 |
| "帮我检查 XXX 算子的 UT" | 询问 | 表述模糊，应询问 |

### 模糊表述处理

当用户表述模糊时，应询问：

```
您是否需要详细分析源文件实现逻辑，以判断 infershape/tiling 是否真的需要独立 UT？

选项：
1. 基础分析（快速扫描，只检查目录结构）
2. 详细分析（深入分析源文件，识别纯模板调用等不需要 UT 的场景）
```

### Issue 生成方式询问（扫描结束后）

**触发时机**: 扫描完成后，发现 UT 缺失问题

**必须询问**: 扫描发现 N 个算子存在 UT 缺失时，需要询问用户 Issue 生成方式：

```
扫描完成！发现 {N} 个算子存在 UT 缺失问题。

请选择 Issue 生成方式：

1. 汇总 Issue（推荐） - 将所有 UT 缺失问题汇总到一个 Issue 文件中
2. 分拆 Issue - 每个缺失算子生成一个独立的 Issue 文件（{N} 个 Issue）

请输入选项编号 (1/2):
```

### Issue 生成方式说明

| 方式 | 输出文件 | 适用场景 |
|------|---------|---------|
| 汇总 Issue | `{repo}_ut_missing_issue_{timestamp}.md`（一个文件） | 问题数量多、同类问题、便于统一跟踪 |
| 分拆 Issue | `{repo}_{算子名}_ut_missing_issue_{timestamp}.md`（N 个文件） | 问题数量少、需单独跟踪每个算子 |

---

## Issue 输出格式

### 方式一：汇总格式（推荐）

将所有 UT 缺失问题汇总到一个 Issue 文件中，便于统一跟踪和批量处理。

**Issue 文件**: `reports/{date}/{repo}/issues/ut_missing_issue_{time}.md`

**Issue 标题**: `[Requirement|需求建议]: [AI 识别] {repo} 仓库 UT 测试覆盖缺失问题（共{N}个算子）`

**Issue 正文格式**:

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）

对 **{repo}** 仓库进行 UT 缺失扫描，发现 **{N} 个算子** 存在 UT 测试文件缺失问题。

### Origin（信息来源）

- **扫描时间**: {date}
- **仓库类型**: {repo}
- **扫描方法**: {basic/detailed}
- **报告文件**: reports/{date}/{repo}/ut-analysis-guide_report_{time}.md

### Benefit / Necessity （价值/作用）

补充缺失 UT 的必要性：

1. **kernel UT（缺失 {count} 个）** - 最关键
   - 涉及实际计算逻辑的正确性验证
   - 无法保证算子功能正确性

2. **api UT（缺失 {count} 个）** - 高优先级
   - 涉及 aclnn 接口的参数校验
   - 无法验证接口调用正确性

3. **infershape UT（缺失 {count} 个）**
   - shape 推导逻辑验证

4. **tiling UT（缺失 {count} 个）**
   - tiling 参数计算验证

### Design（设计方案）

#### 统计摘要

| UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |
|---------|----------|---------|----------|
| infershape | X | Y | Z% |
| tiling | X | Y | Z% |
| kernel | X | Y | Z% |
| api | X | Y | Z% |

#### 缺失算子列表

| 序号 | 算子路径 | 缺失 UT 类型 |
|:---:|----------|-------------|
| 1 | {算子路径} | {类型} |
| 2 | {算子路径} | {类型} |
| ... | ... | ... |

#### 优先级建议

| 优先级 | UT 类型 | 缺失数量 | 优先原因 |
|:------:|---------|:--------:|----------|
| P0 | kernel | X | 计算逻辑最关键 |
| P1 | api | Y | 接口测试重要 |
| P2 | infershape | Z | shape 推导验证 |
| P3 | tiling | W | 参数计算验证 |
```

---

### 方式二：分拆格式（非汇总）

每个缺失算子生成一个独立的 Issue 文件。

**Issue 文件**: `reports/{date}/{repo}/issues/{算子名}_ut_missing_issue_{time}.md`

**Issue 标题**: `[Requirement|需求建议]: [AI 识别] 补充 {算子名} 的 {UT类型} UT`

**Issue 正文格式**:

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）

该算子存在 {UT类型} 源文件 `{源文件路径}`，但缺少对应的 UT 测试文件，无法验证功能的正确性。

### Origin（信息来源）

- 扫描时间: {date}
- 仓库: {repo}
- 算子路径: `{算子路径}`
- 源文件: `{源文件路径}`
- UT 文件: 缺失

### Benefit / Necessity （价值/作用）

补充 UT 的必要性：
1. 验证 {UT类型} 功能的正确性
2. 提供回归测试保障
3. 提升代码质量

### Design（设计方案）

建议创建 UT 文件：`{建议UT路径}`

UT 内容应包含：
- 基础功能测试
- 边界条件测试
- dtype 支持验证（如适用）
```

---

## 输出检查项

完成分析后，确保输出以下内容：

### 终端输出检查

| 检查项 | 内容 |
|-------|------|
| 统计摘要 | 四种 UT 类型的有源文件数、缺失数、缺失比例 |
| 分类统计 | 按算子分类（activation/conv/index 等）统计缺失情况 |
| 缺失详情 | 缺失算子列表（路径 + 缺失类型），至少显示前 100 个 |

### 文件输出检查

| 文件 | 位置 | 内容 |
|------|------|------|
| Markdown 报告 | `reports/{date}/{repo}/ut-analysis-guide_report_{time}.md` | 完整报告，包含所有缺失算子详情 |
| JSON 数据文件 | `reports/{date}/{repo}/ut-analysis-guide_report_{time}.json` | 结构化数据，用于后续处理 |

### 完整检查清单

执行分析后，确认以下事项：

- [ ] 终端已输出统计摘要表格
- [ ] 终端已输出按分类统计表格
- [ ] 终端已输出缺失详情表格（至少前 100 个）
- [ ] 已生成 `reports/{date}/{repo}/ut-analysis-guide_report_{time}.md` 文件
- [ ] 已生成 `reports/{date}/{repo}/ut-analysis-guide_report_{time}.json` 文件
- [ ] Markdown 报告包含完整的缺失算子列表（不仅是前 100 个）

### 报告文件命名规则

| 仓库类型 | 报告文件名 |
|---------|-----------|
| ops-math | `reports/{date}/ops-math/ut-analysis-guide_report_{time}.md` / `.json` |
| ops-nn | `reports/{date}/ops-nn/ut-analysis-guide_report_{time}.md` / `.json` |
| ops-transformer | `reports/{date}/ops-transformer/ut-analysis-guide_report_{time}.md` / `.json` |
| ops-cv | `reports/{date}/ops-cv/ut-analysis-guide_report_{time}.md` / `.json` |
| others | `reports/{date}/ut-analysis-guide_report_{time}.md` / `.json` |

### 分类排序规则

报告中的分类统计应按以下顺序排列：

1. **主分类**：按仓库目录结构顺序排列
   - ops-math: `array_ops`, `comparison`, `math`, `unary`
   - ops-nn: `activation`, `control`, `conv`, `foreach`, `hash`, `index`, `loss`, `matmul`, `norm`, `optim`, `pooling`, `quant`, `rnn`, `vfusion`
   - ops-transformer: `attention`, `ffn`, `mc2`, `moe`, `posembedding`
   - ops-cv: `detection`, `image`, `objdetect`, `video`

2. **experimental 子分类**：按主分类后排列
   - `experimental/activation`, `experimental/attention`, `experimental/matmul` 等

3. **缺失详情列表**：按分类顺序分组显示，每组内按算子名排序

---

## 扫描脚本

本技能提供固定脚本，位于 `scripts/` 目录：

| 脚本 | 功能 |
|------|------|
| `scripts/ut_missing_scan.py` | 扫描算子目录，生成 JSON 数据 |
| `scripts/gen_report.py` | 从 JSON 数据生成 Markdown 报告 |

### 脚本用法

#### 1. 扫描脚本

```bash
python scripts/ut_missing_scan.py --repo ops-nn

可选参数：
  --repo          仓库类型（ops-math/ops-nn/ops-transformer/ops-cv）
  --repo-root     仓库根目录（默认根据 repo 推断）
  --op-list       算子列表文件路径（默认 reports/op_list/{repo}_operator_list.md）
  --output        输出 JSON 文件路径（默认 reports/{date}/{repo}/ut-analysis-guide_report_{time}.json）
```

#### 2. 报告生成脚本

```bash
python scripts/gen_report.py --input reports/{date}/{repo}/ut-analysis-guide_report_{time}.json

可选参数：
  --input         输入 JSON 文件路径（必需）
  --output        输出 Markdown 文件路径（默认根据 input 推断）
```

### 执行流程

**方式一：使用脚本执行**

```bash
# Step 1: 扫描生成 JSON 数据
python scripts/ut_missing_scan.py --repo ops-nn

# Step 2: 生成 Markdown 报告
python scripts/gen_report.py --input reports/{date}/{repo}/ut-analysis-guide_report_{time}.json
```

**方式二：在对话中执行**

当用户请求 UT 缺失分析时，Agent 应：

1. 调用脚本执行扫描：`python scripts/ut_missing_scan.py --repo {repo}`
2. 调用脚本生成报告：`python scripts/gen_report.py --input reports/{date}/{repo}/ut-analysis-guide_report_{time}.json`
3. 输出统计摘要到终端
4. 确认检查清单完成

### 脚本输出

扫描脚本执行后会输出：

```
仓库类型: ops-nn
仓库根目录: /path/to/ops-nn
算子列表: reports/op_list/ops-nn_operator_list.md
输出文件: reports/{date}/ops-nn/ut-analysis-guide_report_{time}.json

解析到 407 个算子
进度: 100/407
进度: 200/407
...

扫描完成，结果已保存: reports/{date}/ops-nn/ut-analysis-guide_report_{time}.json

统计摘要:
| UT 类型 | 有源文件 | 缺失 UT | 缺失比例 |
|---------|----------|---------|----------|
| infershape | 252 | 43 | 17.1% |
| tiling | 261 | 32 | 12.3% |
| kernel | 350 | 146 | 41.7% |
| api | 202 | 113 | 55.9% |
```

---

## Issue 创建流程

**核心原则**：
| 原则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别，所有 UT 缺失都生成 Issue |
| **报告后询问提交** | 每次生成报告后，询问用户是否提交 Issue |
| **同类问题合并选项** | 同类缺失涉及多个算子时，询问是否合并（按问题类型+仓库） |
| **自动化执行默认合并** | unified-scanner 调用时，默认按问题类型+仓库合并创建 Issue |

### 执行模式

| 模式 | 说明 | Issue 创建方式 |
|------|------|---------------|
| **交互模式** | 用户直接调用 `/ut-analysis-guide` | 询问用户选择合并方式 |
| **自动化模式** | unified-scanner 调用 | **默认按问题类型合并**，无需询问 |

**流程概览**：

```
交互模式：
扫描完成 → 分类问题 → 询问合并 → 生成 Issue → 询问提交 → 执行提交

自动化模式（unified-scanner 调用）：
扫描完成 → 分类问题 → 【自动按类型合并】 → 生成 Issue → 汇报结果
```

**Issue 文件命名**：
- **合并模式**：`reports/{date}/{repo}/issues/{ut_type}_missing_merged_issue_{time}.md`
- **单算子模式**：`reports/{date}/{repo}/issues/{op_name}_ut_missing_issue_{time}.md`

**合并场景示例**：
| 问题类型 | 涉及算子数 | 合并标题 |
|---------|:---:|---------|
| infershape UT缺失 | 43 | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} infershape UT缺失（43个算子）` |
| tiling UT缺失 | 32 | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} tiling UT缺失（32个算子）` |
| kernel UT缺失 | 146 | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} kernel UT缺失（146个算子）` |
| api UT缺失 | 113 | `[Bug-Report|缺陷反馈]: [AI 识别] {repo} api UT缺失（113个算子）` |

### 自动化模式 Issue 创建规则

**触发条件**：unified-scanner 调用时，无需用户交互

**默认行为**：
1. 按问题类型分类（infershape/tiling/kernel/api）
2. 每种类型生成一个合并 Issue
3. Issue 文件命名：`{repo}_{ut_type}_missing_issue_{timestamp}.md`
4. 自动汇报 Issue 创建结果

**自动化执行示例**：
```python
# 自动按问题类型创建 Issue
for ut_type in ['infershape', 'tiling', 'kernel', 'api']:
    if missing_count[ut_type] > 0:
        issue_file = f"reports/{date}/{repo}/issues/{ut_type}_missing_issue_{time}.md"
        # 调用 gitcode-issue-creator Skill 生成 Issue
```

**询问合并流程**（交互模式）：
```
发现以下 UT 缺失：
- infershape UT缺失：43个算子
- tiling UT缺失：32个算子
- kernel UT缺失：146个算子
- api UT缺失：113个算子

合并选项：
1. 合并同类问题（推荐）- 按问题类型+仓库合并
2. 不合并 - 每个算子单独一个 Issue
3. 全部合并到一个 Issue

请选择处理方式：
```

**询问提交流程**（交互模式）：
```
已生成 Issue 文件：
| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/{date}/{repo}/issues/infershape_ut_missing_merged_issue_{time}.md | [Bug-Report]: [AI 识别] {repo} infershape UT缺失（43个算子） | 43 | cann/{repo} |

是否提交 Issue 到对应仓库？
1. 是，全部提交
2. 是，选择提交
3. 否，暂不提交
4. 否，手动提交
```
---
name: scan-examples-analysis
description: Ascend C 算子 examples 缺失分析技能。用于分析 ops-math/ops-nn/ops-transformer/ops-cv 算子的 examples 需求，判断何时需要 examples 测试用例，识别 examples 缺失情况。当用户询问 examples 缺失、调用示例、测试用例时使用。
---

# Examples 缺失分析

## 概述

本技能用于分析 Ascend C 算子的 examples 需求，帮助判断：
- 哪些算子需要 examples 测试用例
- 哪些算子不需要 examples
- 如何识别 examples 缺失情况
- examples 的编写规范

---

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── examples-analysis-guide_report_{time}.md  # 扫描报告
        ├── issues/
        │   └── examples_missing_issue_{time}.md  # Issue 文件
        └── his/                                   # 历史报告归档
```

---

## 分析模式

本技能支持两种分析模式：

### 模式一：基础分析（默认）

**适用场景**：快速扫描仓库所有算子的 examples 目录是否存在

**分析方法**：仅检查目录结构，对比 examples 目录是否存在及是否有 .cpp 文件

### 模式二：智能分析

**适用场景**：精确分析算子是否真正需要 examples

**分析方法**：深入分析算子实现类型（仅 aclnn 接口 vs 有 kernel 实现）

**触发条件**：
- 用户明确要求"智能分析"、"分析是否真的需要 examples"
- 用户询问"这个算子是否需要 examples"
- 用户对基础分析结果有疑问

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| op_name | 算子名 | 采用下划线命名法 | 用户提供或当前分析的算子 |
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv", "others"] | 根据工作目录推断 |
| repo_root | 仓库根目录路径 | 字符串，指向仓库根目录 | 通过 `repo_detector.py` 自动检测；支持任意嵌套深度向上遍历查找 ops-* 仓库根目录；用户可通过 `--repo-root` 手动指定覆盖自动检测 |
| analysis_mode | 分析模式 | 枚举值["basic", "smart"] | 默认 "basic"，用户明确要求时为 "smart" |

---

## Examples 的作用

### 什么是 examples

examples 目录用于存放算子的**调用示例代码**，展示如何在实际场景中使用该算子：

- `test_aclnn_{op}.cpp`: ACL NN 接口调用示例
- `test_geir_{op}.cpp`: GEIR（图引擎）调用示例
- `test_aclnn_inplace_{op}.cpp`: inplace 操作示例（如适用）

### 标准目录结构

```
算子目录/
├── examples/                      # examples 目录
│   ├── test_aclnn_{op}.cpp       # ACL NN 调用示例
│   ├── test_geir_{op}.cpp        # GEIR 调用示例（可选）
│   └── arch35/                   # 架构特定示例（可选）
│       └── test_aclnn_{op}.cpp
├── op_kernel/                     # Ascend C kernel 实现
├── op_api/                        # aclnn 接口实现（自定义）
├── op_host/                       # Host 端实现
└── tests/                         # UT 测试
```

---

## 不需要 Examples 的场景

根据官方指导，以下场景**不需要 examples**：

### 1. 无调用接口（核心规则）

**关键判断标准**：

```
无调用接口 = 无 op_api/aclnn_*.cpp AND 无 op_graph/*_proto.{h/cpp}
```

**说明**：
- `examples` 用于展示算子调用方式
- 如果没有 `aclnn` 接口文件，也没有 `op_graph` proto 文件
- 就没有调用方式，无法编写 examples

**判断代码**：
```bash
# 检查是否有 aclnn 接口
ls {算子目录}/op_api/aclnn_*.cpp 2>/dev/null | wc -l

# 检查是否有 op_graph proto 文件
find {算子目录}/op_graph -name '*_proto.*' 2>/dev/null | wc -l

# 两者都为 0 则不需要 examples
```

**示例**：
| 算子 | aclnn 接口 | op_graph proto | examples | 判断 |
|------|-----------|---------------|----------|------|
| pad_v3_grad | 无 | 无 | 无 | **不需要** ✓ |
| assign | 无 | assign_proto.h | test_geir_assign.cpp | 需要 ✓ |
| abs | aclnn_abs.cpp | 无 | test_aclnn_abs.cpp | 需要 ✓ |

**两种 examples 调用方式**：
| 有什么文件 | 可编写 examples 类型 |
|-----------|---------------------|
| `op_api/aclnn_*.cpp` | `test_aclnn_*.cpp` |
| `op_graph/*_proto.*` | `test_geir_*.cpp` |
| 两者都有 | 可编写两种 |
| 两者都无 | **不需要 examples** |

### 2. 仅 aclnn 接口实现（无 kernel）

**正确判断标准**：

```
仅 aclnn 接口 = README 说"仅 aclnn 接口" AND op_kernel 目录无 .cpp 文件
```

**重要**：必须同时满足两个条件！如果 README 说"仅 aclnn 接口"但 op_kernel 有 .cpp 文件，说明 README 有误，实际需要 examples。

**特征**：
- README 明确说明："本目录仅包含XXX算子对应的aclnn接口"
- op_kernel 目录不存在或目录下没有 .cpp 文件（递归检查）

**判断代码**：
```bash
# 检查 README 内容
grep "仅包含.*aclnn接口" {算子目录}/README.md

# 递归检查 op_kernel 目录是否有 cpp 文件
find {算子目录}/op_kernel -name "*.cpp" | wc -l
```

**正确示例**（README 和 kernel 状态一致）：
- `math/silent_check`: README 说仅 aclnn，op_kernel 无 cpp ✓
- `math/silent_check_v2`: README 说仅 aclnn，op_kernel 无 cpp ✓
- `conversion/reshape`: README 说仅 aclnn，op_kernel 无 cpp ✓
- `conversion/contiguous`: README 说仅 aclnn，op_kernel 无 cpp ✓

**错误示例**（README 有误，实际需要 examples）：
| 算子 | README 说 | op_kernel 实际 | 正确判断 |
|------|----------|----------------|---------|
| math/cdist | 仅 aclnn | 有 1 个 cpp | **需要 examples** |
| conversion/pad | 仅 aclnn | 有 1 个 cpp | **需要 examples** |
| conversion/roll | 仅 aclnn | 有 1 个 cpp | **需要 examples** |
| math/ger | 仅 aclnn | 有 1 个 cpp | 已有 examples ✓ |

**原因**：examples 主要用于展示 **Ascend C kernel** 的调用方式，仅 aclnn 接口的算子没有自定义 kernel，无需提供调用示例。

### 3. 无 Kernel 实现

**特征**：
- 没有 op_kernel 目录
- 或 op_kernel 目录下没有 .cpp 文件（递归检查所有子目录）

**判断代码**：
```bash
# 递归检查 op_kernel 目录
find {算子目录}/op_kernel -name "*.cpp" | wc -l
# 结果为 0 表示无 kernel
```

**注意**：op_kernel/arch35 等子目录下的 .cpp 文件也算 kernel 实现！

**原因**：没有 Ascend C kernel 实现，examples 无法展示 kernel 调用。

---

## 需要 Examples 的场景

### 有 Kernel 实现的算子

**特征**：
- 有 op_kernel 目录且包含 .cpp 文件
- 有实际的 Ascend C kernel 实现

**必须提供 examples 的原因**：
1. 展示如何正确调用算子的 kernel
2. 提供端到端的使用示例
3. 帮助用户理解算子的输入输出格式
4. 作为算子功能的验证示例

---

## Examples 的两种编写方式

### 方式一：test_aclnn_*.cpp（推荐）

**适用条件**：
- CANN 公共库已提供该算子的 aclnn 接口（`aclnnop/aclnn_{op}.h`）
- 或算子仓库有自定义 op_api 实现

**特点**：
- 直接调用 aclnn 接口
- 代码简洁，易于理解
- 适合大多数场景

**代码模板**：
```cpp
#include "acl/acl.h"
#include "aclnnop/aclnn_{op}.h"  // CANN 公共库或自定义接口

int main() {
    // 1. 初始化 ACL
    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtCreateStream(&stream);
    
    // 2. 创建输入 tensor
    aclTensor* input = CreateAclTensor(...);
    
    // 3. 调用算子
    aclnn{Op}GetWorkspaceSize(...);
    aclnn{Op}(...);
    
    // 4. 获取结果
    aclrtSynchronizeStream(stream);
    
    // 5. 销毁资源
    aclDestroyTensor(input);
    aclrtDestroyStream(stream);
    aclFinalize();
}
```

### 方式二：test_geir_*.cpp

**适用条件**：
- 没有 aclnn 接口（公共库和自定义都没有）
- 需要通过图引擎构建计算图调用

**特点**：
- 通过 GE IR 构建图
- 调用 op_graph 中定义的算子原型
- 适合无 aclnn 接口的算子

**代码模板**：
```cpp
#include "graph.h"
#include "ge_api.h"
#include "array_ops.h"
#include "{op}_proto.h"

int main() {
    // 1. 构建计算图
    ge::Graph graph;
    
    // 2. 创建输入 placeholder
    auto input1 = op::Data("input1");
    
    // 3. 添加算子节点
    auto opNode = op::{Op}("op_node");
    opNode.set_input_x(input1);
    
    // 4. 构建 graph
    graph.AddOp(input1);
    graph.AddOp(opNode);
    
    // 5. 执行 graph
    ge::Session session;
    session.RunGraph(graph);
}
```

---

## 没有 op_api 也能写 Examples

### 关键认知

**op_api 和 examples 的关系**：
- `op_api` 用于自定义算子的 aclnn 接口实现
- CANN 公共库已提供大量算子的 aclnn 接口
- 没有 op_api 也可以调用 CANN 公共库的 aclnn 接口

### 实际案例

有 42 个算子**没有 op_api**但有 test_aclnn_*.cpp 文件：

| 算子 | 无 op_api | 调用的接口 |
|------|----------|-----------|
| math/affine_grid | ✓ | `aclnnop/aclnn_affine_grid.h` |
| conversion/circular_pad | ✓ | `aclnnop/aclnn_circular_pad2d.h` |
| math/cholesky | ✓ | `aclnnop/aclnn_linalg_cholesky.h` |
| math/cumprod | ✓ | `aclnnop/aclnn_cumprod.h` |
| math/digamma | ✓ | `aclnnop/aclnn_digamma.h` |

### 建议

对于有 kernel 但没有 op_api 的算子：

1. **首选**：检查 CANN 公共库是否有接口，写 test_aclnn_*.cpp
2. **备选**：无公共库接口时，写 test_geir_*.cpp
3. **两者都写**：提供更完整的调用示例

---

## 智能分析流程

### Step 1: 检查调用接口（关键步骤）

```bash
# 检查是否有 aclnn 接口
ls {算子目录}/op_api/aclnn_*.cpp 2>/dev/null | wc -l

# 检查是否有 op_graph proto 文件
find {算子目录}/op_graph -name '*_proto.*' 2>/dev/null | wc -l
```

**判断**：
- 两者都为 0 → **不需要 examples**（无调用方式）
- 有任一文件 → 继续后续检查

### Step 2: 检查 examples 目录

```bash
# 检查 examples 目录
ls {算子目录}/examples/

# 递归检查 examples 子目录（包括 arch35 等）
find {算子目录}/examples -name "*.cpp"
```

### Step 3: 检查 README 内容

```bash
# 检查 README 是否说"仅 aclnn 接口"
grep "仅包含.*aclnn接口" {算子目录}/README.md
```

**注意**：README 说"仅 aclnn 接口"不一定准确，还需验证 Step 4。

### Step 4: 验证是否有 kernel（关键步骤）

```bash
# 递归检查 op_kernel 目录是否有 cpp 文件
find {算子目录}/op_kernel -name "*.cpp" | wc -l
```

**重要**：必须检查 arch35 等子目录！

### Step 5: 综合判断

| 条件组合 | 是否需要 examples | 说明 |
|---------|-----------------|------|
| 无 aclnn + 无 proto | ❌ 不需要 | 无调用方式 |
| 有 aclnn/proto + 有 examples | ❌ 已有，无需补充 | |
| 有 aclnn/proto + 无 examples + README 仅 aclnn + 无 kernel | ❌ 不需要 | README 和 kernel 状态一致 |
| 有 aclnn/proto + 无 examples + README 仅 aclnn + **有 kernel** | ✅ **需要补充** | README 有误，实际有 kernel |
| 有 aclnn/proto + 无 examples + README 无说明 + 无 kernel | ❌ 不需要 | |
| 有 aclnn/proto + 无 examples + README 无说明 + 有 kernel | ✅ 需要补充 | |

### 发现 README 错误的案例

实际扫描发现 **13 个算子** README 说"仅 aclnn 接口"但实际有 kernel：

| 算子 | README | kernel | examples | 正确判断 |
|------|--------|--------|----------|---------|
| math/cdist | 仅 aclnn | 有 1 cpp | 无 | **需补充** |
| conversion/pad | 仅 aclnn | 有 1 cpp | 无 | **需补充** |
| conversion/roll | 仅 aclnn | 有 1 cpp | 无 | **需补充** |
| math/ger | 仅 aclnn | 有 1 cpp | 有 | README 有误 |
| math/gcd | 仅 aclnn | 有 1 cpp | 有 | README 有误 |
| math/reduce_max | 仅 aclnn | 有 1 cpp | 有 | README 有误 |

---

## 输出格式

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（在 `{Skill特殊字段区域}` 增加）：

| 字段名称 | 内容说明 |
|---------|---------|
| 分类统计 | 按目录分类的统计表格（分类/算子总数/缺失数/缺失比例） |
| 有examples算子列表 | 有完整examples测试用例的算子示例表格 |
| 缺失详情汇总表 | 算子路径/调用接口/kernel文件数/建议examples文件 |

### 报告模板（Issue友好格式）

```markdown
# {repo_type} Examples 缺失分析报告

**扫描时间**: {date}
**分析模式**: {basic/smart}
**仓库路径**: `{repo_root}`

---

## 一、扫描概览

### 1.1 统计摘要

| 指标 | 数量 |
|------|------|
| 扫描算子总数 | X |
| 有 examples 目录 | Y |
| 有测试用例文件 | Z |
| 缺少 examples 目录 | N |
| 无调用接口（不需要） | N1 |
| 仅 aclnn 接口（不需要） | N2 |
| **实际需要补充 examples** | N3 |

### 1.2 按分类统计

| 分类 | 有examples | 缺失 | 不需要 |
|------|-----------|------|--------|
| activation | X | Y | Z |
| conv | X | Y | Z |
| ... | ... | ... | ... |

---

## 二、缺失详情汇总表

| 序号 | 算子路径 | 调用接口 | kernel文件数 | 建议 examples 文件 |
|-----|---------|---------|-------------|-------------------|
| 1 | {算子路径} | aclnn | N | test_aclnn_{op}.cpp |
| 2 | {算子路径} | proto | N | test_geir_{op}.cpp |

---

## 三、不需要 examples 的算子说明

### 3.1 无调用接口（不需要 examples）

| 算子 | 说明 |
|------|------|
| {算子路径} | 无 aclnn_*.cpp 且无 op_graph proto |

### 3.2 仅 aclnn 接口实现（不需要 examples）

| 算子 | README 说明 | kernel 状态 |
|------|------------|------------|
| {算子路径} | 本目录仅包含XXX算子对应的aclnn接口 | 无 cpp 文件 |

---

## 四、判断规则说明

### 需要 examples 的条件

| 条件 | 说明 |
|------|------|
| 有调用接口 | 存在 `aclnn_*.cpp` 或 `op_graph/*_proto.*` |
| 有 kernel 实现 | `op_kernel/` 目录下有 .cpp 文件 |

### 不需要 examples 的场景

| 场景 | 判断标准 |
|------|---------|
| 无调用接口 | 无 aclnn_*.cpp 且无 op_graph proto |
| 仅 aclnn 接口 | README 说明 + 无 kernel cpp 文件 |
| 无 kernel | op_kernel 目录无 cpp 文件 |

---

**报告生成时间**: {timestamp}
```

---

### Issue 格式规范

每个 examples 缺失问题需遵循 GitCode Issue 模板格式：

---

## 用户交互规则

### 模式判断

| 用户表述 | 分析模式 | 说明 |
|---------|---------|------|
| "扫描 examples 缺失"、"检查 examples" | basic | 默认基础分析 |
| "智能分析 examples"、"分析是否真的需要" | smart | 明确要求智能分析 |
| "这个算子是否需要 examples" | smart | 需要精确判断 |

### 模糊表述处理

当用户表述模糊时，应询问：

```
您是否需要智能分析算子实现类型，以精确判断哪些算子真正需要 examples？

选项：
1. 基础分析（快速扫描，只检查 examples 目录是否存在）
2. 智能分析（深入分析，识别仅 aclnn 接口等不需要 examples 的场景）
```

### Issue 生成方式询问（扫描结束后）

**触发时机**: 扫描完成后，发现 examples 缺失问题

**必须询问**: 扫描发现 N 个算子缺少 examples 时，需要询问用户 Issue 生成方式：

```
扫描完成！发现 {N} 个算子缺少 examples 调用示例。

请选择 Issue 生成方式：

1. 汇总 Issue（推荐） - 将所有 examples 缺失问题汇总到一个 Issue 文件中
2. 分拆 Issue - 每个缺失算子生成一个独立的 Issue 文件（{N} 个 Issue）

请输入选项编号 (1/2):
```

### Issue 生成方式说明

| 方式 | 输出文件 | 适用场景 |
|------|---------|---------|
| 汇总 Issue | `{repo}_examples_missing_issue_{timestamp}.md`（一个文件） | 问题数量多、同类问题、便于统一跟踪 |
| 分拆 Issue | `{repo}_{算子名}_examples_missing_issue_{timestamp}.md`（N 个文件） | 问题数量少、需单独跟踪每个算子 |

---

## Issue 输出格式

### 方式一：汇总格式（推荐）

将所有 examples 缺失问题汇总到一个 Issue 文件中，便于统一跟踪和批量处理。

**Issue 文件**: `reports/{date}/{repo}/issues/examples_missing_issue_{time}.md`

**Issue 标题**: `[Requirement|需求建议]: [AI 识别] 补充 {repo} 仓库缺失的 examples 调用示例（共{N}个算子）`

**Issue 正文格式**:

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）

{repo} 仓库中有 {N} 个算子缺少 examples 目录，无法为用户提供调用示例。

### Origin（信息来源）

- **扫描时间**: {date}
- **仓库类型**: {repo}
- **扫描方法**: {basic/smart}
- **报告文件**: reports/{date}/{repo}/examples-analysis-guide_report_{time}.md

### Benefit / Necessity （价值/作用）

补充 examples 的必要性：

1. **展示调用方式**: examples 文件展示如何正确调用算子，为用户提供参考
2. **端到端示例**: 提供完整的调用流程示例
3. **理解输入输出**: 帮助用户理解算子的输入输出格式
4. **功能验证**: 作为算子功能的验证示例

### Design（设计方案）

#### 缺失算子列表

| 序号 | 算子路径 | 分类 | 建议 examples 文件 |
|:---:|----------|------|-------------------|
| 1 | {算子路径} | {分类} | test_aclnn_{op}.cpp |
| 2 | {算子路径} | {分类} | test_aclnn_{op}.cpp |
| ... | ... | ... | ... |

#### Examples 文件模板

每个算子建议创建以下结构：

```
{算子路径}/examples/
├── test_aclnn_{算子名}.cpp    # 主要示例文件
└── arch35/                    # 架构特定示例（可选）
    └── test_aclnn_{算子名}.cpp
```

#### 参考已有 examples

仓库中已有 examples 文件可作为参考。
```

---

### 方式二：分拆格式（非汇总）

每个缺失算子生成一个独立的 Issue 文件。

**Issue 文件**: `reports/{date}/{repo}/issues/{算子名}_examples_missing_issue_{time}.md`

**Issue 标题**: `[Requirement|需求建议]: [AI 识别] 补充 {算子名} 的 examples 调用示例`

**Issue 正文格式**:

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Backgroud（背景信息）

该算子存在 kernel 实现和调用接口，但缺少 examples 调用示例文件。

### Origin（信息来源）

- 扫描时间: {date}
- 仓库: {repo}
- 算子路径: `{算子路径}`
- kernel 文件: `op_kernel/{op}.cpp`（{数量}个）
- 调用接口: {aclnn/proto}
- examples 目录: 缺失

### Benefit / Necessity （价值/作用）

补充 examples 的必要性：
1. 展示如何正确调用算子
2. 提供端到端使用示例
3. 帮助用户理解输入输出格式
4. 作为算子功能验证示例

### Design（设计方案）

建议创建 examples 文件：`{算子路径}/examples/test_{调用方式}_{op}.cpp`

examples 内容应包含：
- 完整的调用流程（ACL初始化、tensor创建、算子调用、结果获取）
- 多 dtype 测试示例（如适用）
- 输入输出说明
```

---

## Issue 创建流程

**核心原则**：
| 原则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别，所有 examples 缺失都生成 Issue |
| **报告后询问提交** | 每次生成报告后，询问用户是否提交 Issue |
| **同类问题合并选项** | 同类缺失涉及多个算子时，询问是否合并（按问题类型+仓库） |
| **自动化执行默认合并** | unified-scanner 调用时，默认合并创建 Issue |

### 执行模式

| 模式 | 说明 | Issue 创建方式 |
|------|------|---------------|
| **交互模式** | 用户直接调用 `/examples-analysis-guide` | 询问用户选择合并方式 |
| **自动化模式** | unified-scanner 调用 | **默认合并创建 Issue**，无需询问 |

**流程概览**：

```
交互模式：
扫描完成 → 分类问题 → 询问合并 → 生成 Issue → 询问提交 → 执行提交

自动化模式（unified-scanner 调用）：
扫描完成 → 分类问题 → 【自动合并】 → 生成 Issue → 汇报结果
```

**Issue 文件命名**：
- **合并模式**：`reports/{date}/{repo}/issues/examples_missing_merged_issue_{time}.md`
- **单算子模式**：`reports/{date}/{repo}/issues/{op_name}_examples_missing_issue_{time}.md`

**合并场景示例**：
| 问题类型 | 涉及算子数 | 合并标题 |
|---------|:---:|---------|
| examples缺失 | 10 | `[Requirement|需求建议]: [AI 识别] {repo} examples测试用例缺失（10个算子）` |

### 自动化模式 Issue 创建规则

**触发条件**：unified-scanner 调用时，无需用户交互

**默认行为**：
1. 所有 examples 缺失算子合并为一个 Issue
2. Issue 文件命名：`{repo}_examples_missing_merged_issue_{timestamp}.md`
3. 自动汇报 Issue 创建结果

**自动化执行示例**：
```python
# 自动创建合并 Issue
if missing_count > 0:
    issue_file = f"reports/{date}/{repo}/issues/examples_missing_merged_issue_{time}.md"
    # 调用 gitcode-issue-creator Skill 生成 Issue
```

**询问合并流程**（交互模式）：
```
发现以下问题：
- examples缺失：10个算子（有kernel和调用接口但无examples）

合并选项：
1. 合并同类问题（推荐）- 按问题类型+仓库合并
2. 不合并 - 每个算子单独一个 Issue

请选择处理方式：
```

**询问提交流程**（交互模式）：
```
已生成 Issue 文件：
| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/{date}/{repo}/issues/examples_missing_merged_issue_{time}.md | [Requirement]: [AI 识别] {repo} examples缺失（10个算子） | 10 | cann/{repo} |

是否提交 Issue 到对应仓库？
1. 是，全部提交
2. 是，选择提交
3. 否，暂不提交
4. 否，手动提交
```
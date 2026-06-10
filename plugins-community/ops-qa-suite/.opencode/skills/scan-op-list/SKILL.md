---
name: scan-op-list
description: 算子列表一致性扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 docs/zh/op_list.md 文档，验证算子目录、分类、实现状态标记(√×)、硬件单元说明与实际代码实现的一致性。当用户需要验证算子列表文档准确性、检查op_list表格与实际实现匹配时使用。
---

# 算子列表一致性扫描

## 概述

本技能用于验证仓库级算子列表文档（`docs/zh/op_list.md`）与实际算子实现的一致性，帮助检测：
- 算子目录是否存在且路径正确
- README.md 是否缺失或为占位文档
- 算子分类是否与实际目录结构一致
- 实现状态标记（√×）是否与实际文件存在一致
- 硬件单元说明是否与实际实现一致
- 文档链接是否可正常跳转

**重要说明**：
- 只检查**列入 op_list.md 的算子**，未列入的不检查
- **跳过 experimental 目录**下的算子（生态开发者提供，不检查）

---

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── op-list-validation_report_{time}.md    # 扫描报告
        ├── issues/
        │   └── op_list_issue_{time}.md           # Issue 文件
        └── his/                                   # 历史报告归档
```

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv"] | 根据工作目录推断 |
| repo_root | 仓库根目录 | 绝对路径 | 自动检测（从任意嵌套目录向上遍历查找 ops-* 仓库），详见 `repo_detector.py` |
| output_path | 报告输出路径 | 绝对路径 | 默认为 `reports/{date}/{repo}/op-list-validation_report_{time}.md` |

---

## op_list.md 文档格式

### 表格结构

op_list.md 使用 HTML 表格格式（嵌入 Markdown），包含以下列：

| 列名 | 说明 | 数据来源 |
|------|------|---------|
| 算子分类 | 算子所属类别 | 算子目录的父目录名 |
| 算子目录 | 算子名（snake_case）+ 链接 | 算子目录名 |
| op_kernel | Kernel实现状态 | op_kernel/ 或 op_kernel_aicpu/ 目录是否存在 .asc/.cpp 文件 |
| op_host | Host实现状态（必须√）| **必须打勾，否则算子无法运行**，不验证具体文件 |
| op_api | aclnn接口状态 | op_api/ 或 op_host/op_api/ 是否存在 aclnn_*.cpp |
| op_graph | 图模式状态 | op_graph/ 是否存在 *_proto.* 文件 |
| 算子执行硬件单元 | AI Core/AI CPU | **默认 AI Core（tbe），仅检查 op_kernel_aicpu/ 是否存在来判断 AI CPU** |
| 说明 | 算子功能描述 | README.md 功能说明章节 |

### 实现状态标记格式（三种）

不同仓库使用不同的标记符号：

| 仓库 | 已实现标记 | 未实现标记 |
|------|-----------|-----------|
| ops-math | `√` | `×` |
| ops-nn, ops-transformer | `✓` | `✗` |
| ops-cv | `&check;` | `&cross;` |

**扫描时需统一识别这三种格式**。

---

## 检查项详细说明

### 检查项1：算子目录存在性与 README.md 完备性

**检查内容**：
1. op_list表格中列出的每个算子目录是否实际存在
2. 算子目录路径是否正确（是否在正确的分类目录下）
3. README.md 是否存在（必选交付件）
4. README.md 是否为占位文档
5. 文档链接是否能正常跳转到 README.md

**experimental 目录排除**：
- 路径以 `experimental/` 开头的算子自动跳过（生态开发者提供，不检查）
- 在报告中标注为"跳过的experimental算子"

**验证方法**：
```bash
# 从 op_list.md 提取算子目录名
grep -oP '<a href="[^"]+/([^/]+)/README\.md">' docs/zh/op_list.md

# 检查目录是否存在
ls {category}/{op_name}/

# 检查 README.md 是否存在
ls {category}/{op_name}/README.md

# 检查是否为占位文档（关键词匹配）
grep "该算子暂无Ascend C代码实现" {category}/{op_name}/README.md
```

**判定规则**：

| 情况 | 状态 | 问题类型 |
|------|------|---------|
| 目录存在 + README.md完备 + 链接有效 | ✅ 通过 | - |
| 目录不存在 | ❌ 失败 | 目录缺失 |
| README.md不存在 | ❌ 失败 | README缺失 |
| README.md为占位文档 | ✅ 非问题 | README占位文档（标注） |
| 链接路径错误 | ❌ 失败 | 链接错误 |
| experimental目录下的算子 | ⏭️ 跳过 | 不计入统计 |

**占位文档识别规则**：
包含以下关键词视为占位文档：
- "该算子暂无Ascend C代码实现"
- "欢迎开发者补充贡献"
- "暂无实现"、"待实现"、"待开发"

---

### 检查项2：算子分类正确性

**检查内容**：
op_list表格中的"算子分类"列是否与算子实际所在的父目录名一致。

**验证方法**：
```bash
# 从 op_list.md 提取分类和目录
# 检查算子实际父目录
dirname $(dirname {算子路径})
```

**判定规则**：

| op_list分类 | 实际父目录 | 状态 |
|-------------|-----------|------|
| 一致 | 一致 | ✅ 通过 |
| 不一致 | 不一致 | ❌ 分类错误 |

**示例**：
- op_list显示 `math` 分类，算子 `add` → 实际路径应为 `math/add/`
- op_list显示 `image` 分类，算子 `grid_sample` → 实际路径应为 `image/grid_sample/`

---

### 检查项3：实现状态标记一致性

**检查内容**：
验证 op_list表格中的实现状态标记（op_kernel、op_host、op_api、op_graph列的√×）是否与实际文件存在一致。

#### op_kernel 列验证

**实际文件检查**：
```bash
# 检查 op_kernel 目录（AI Core 实现）
find {算子目录}/op_kernel -type f \( -name "*.asc" -o -name "*.cpp" \) 2>/dev/null | grep -v "_def.cpp" | grep -v "tilingdata.h"

# 检查 op_kernel_aicpu 目录（AI CPU 实现）
find {算子目录}/op_kernel_aicpu -type f -name "*.cpp" 2>/dev/null | grep -v "_def.cpp"

# 注意：
# - op_kernel/ 存在实现文件 → AI Core 实现，op_kernel标记为√
# - op_kernel_aicpu/ 存在实现文件 → AI CPU 实现，op_kernel标记为√
# - 两者都存在 → AI Core/AI CPU 双实现
# - 都不存在 → op_kernel标记为×
```

#### op_host 列验证

**实际文件检查**（必须验证文件存在）：
```bash
# 检查 op_host 目录下的 tiling/infershape 文件
find {算子目录}/op_host -type f -name "*_tiling.cpp" -o -name "*_infershape.cpp" 2>/dev/null

# 注意：
# - 有 *_tiling.cpp 或 *_infershape.cpp → op_host标记为√
# - 无这些文件 → op_host标记为×
# - 也要检查公共模块中的实现（如 foreach_utils_host）
```

**判定规则**：

| op_list标记 | 实际文件状态 | 状态 |
|------------|------------|------|
| √/✓/&check; | op_host/*_tiling.cpp 或 *_infershape.cpp 存在 | ✅ 通过 |
| √/✓/&check; | 无实现文件 | ❌ 标记错误（应为×） |
| ×/✗/&cross; | 有实现文件 | ❌ 标记错误（应为√） |
| ×/✗/&cross; | 无实现文件 | ✅ 通过 |

#### op_api 列验证

**实际文件检查**：
```bash
# 检查 op_api 目录（标准写法，递归检查）
find {算子目录}/op_api -type f \( -name "aclnn_*.cpp" -o -name "aclnn_*.h" \) 2>/dev/null

# 检查 op_host/op_api 目录（嵌套写法，需标注整改）
find {算子目录}/op_host/op_api -type f \( -name "aclnn_*.cpp" -o -name "aclnn_*.h" \) 2>/dev/null

# 检查 CMakeLists.txt 的 ACLNNTYPE
grep "ACLNNTYPE" {算子目录}/CMakeLists.txt 2>/dev/null

# 注意：op_host/op_api/ 嵌套结构中的文件命名可能不以 aclnn_ 开头
# 例如：op_host/op_api/grid_sampler2d.cpp 也应计入 op_api 实现
```

**判定规则**：

| op_list标记 | ACLNNTYPE/文件状态 | 状态 |
|------------|-------------------|------|
| √/✓/&check; | ACLNNTYPE=aclnn 或有 aclnn_*.cpp | ✅ 通过 |
| √/✓/&check; | op_host/op_api/ 中有实现文件 | ✅ 通过（需标注整改） |
| √/✓/&check; | ACLNNTYPE=aclnn_exclude + 有aclnn文件 | ✅ 通过 |
| ×/✗/&cross; | ACLNNTYPE=aclnn_exclude + 无aclnn文件 | ✅ 通过 |
| ×/✗/&cross; | ACLNNTYPE=aclnn 或有 aclnn文件 | ❌ 标记错误 |

**重要提示**：
- ACLNNTYPE=aclnn_inner **不算 ✔**（不暴露接口）
- op_host/op_api/ 嵌套结构算 ✔ 但需标注为"目录结构需整改"
- op_api 目录应与 op_host 同级

#### op_graph 列验证

**实际文件检查**：
```bash
# 检查 op_graph 目录是否存在 proto 文件（递归检查）
find {算子目录}/op_graph -type f \( -name "*_proto.h" -o -name "*_proto.cpp" -o -name "*_proto*" \) 2>/dev/null
```

**判定规则**：

| op_list标记 | 实际文件状态 | 状态 |
|------------|------------|------|
| √/✓/&check; | 有 *_proto.* 文件 | ✅ 通过 |
| ×/✗/&cross; | 无文件或无 op_graph 目录 | ✅ 通过 |
| √/✓/&check; | 无 proto 文件 | ❌ 标记错误 |
| ×/✗/&cross; | 有 proto 文件 | ❌ 标记错误 |

**重要提示**：
- proto 文件通常位于 op_graph/ 目录顶层
- 文件名格式通常为 `{算子名}_proto.h`

---

### 检查项4：硬件单元说明一致性

**检查内容**：
验证 op_list表格中的"算子执行硬件单元"列是否与实际实现一致。

**硬件单元判断规则**（完整版）：

| 硬件单元 | 判断依据 | 优先级 | 说明 |
|---------|---------|:---:|------|
| AI Core | op_kernel/*.cpp 存在 | 1 | AscendC AI Core 实现 |
| AI CPU | op_kernel_aicpu/*.cpp 存在 | 1 | AscendC AI CPU 实现 |
| AI Core | ADD_TO_LAUNCHER_LIST_AICORE({本算子名}) 存在 | 2 | TBE AI Core 实现 |
| AI CPU | ADD_TO_LAUNCHER_LIST_AICPU({本算子名}) 存在 | 2 | TBE AI CPU 实现 |
| 仅API | 无以上任何实现 | 3 | 仅 API 接口，无底层实现 |

**检查顺序**（按优先级）：

1. **第一步：检查 AscendC 实现**
   ```bash
   # 检查 op_kernel 目录（AI Core）
   find {算子目录}/op_kernel -type f \( -name "*.asc" -o -name "*.cpp" \) 2>/dev/null | grep -v "_def.cpp"
   
   # 检查 op_kernel_aicpu 目录（AI CPU）
   find {算子目录}/op_kernel_aicpu -type f -name "*.cpp" 2>/dev/null | grep -v "_def.cpp"
   ```

2. **第二步：检查 TBE 实现（LAUNCHER 宏）**
   ```bash
   # 精确匹配算子名（必须匹配本算子名）
   op_class_name = PascalCase(算子名)
   
   # 检查 op_api/*.cpp 或 op_host/op_api/*.cpp
   grep -rn "ADD_TO_LAUNCHER_LIST_AICORE(${op_class_name}" {算子目录}/op_api/*.cpp
   grep -rn "ADD_TO_LAUNCHER_LIST_AICPU(${op_class_name}" {算子目录}/op_api/*.cpp
   ```

3. **第三步：判断硬件单元**
   - 有 op_kernel/ → **AI Core**
   - 有 op_kernel_aicpu/ → **AI CPU**
   - 有 `ADD_TO_LAUNCHER_LIST_AICORE({本算子名})` → **AI Core**（TBE）
   - 有 `ADD_TO_LAUNCHER_LIST_AICPU({本算子名})` → **AI CPU**（TBE）
   - 以上都无 → **仅API**

**判定规则**：

| op_list硬件单元 | AscendC实现 | LAUNCHER宏 | 状态 |
|----------------|-----------|-----------|------|
| AI Core | op_kernel/ 有实现 | - | ✅ 通过 |
| AI CPU | op_kernel_aicpu/ 有实现 | - | ✅ 通过 |
| AI Core | 无 AscendC 实现 | 有 AICORE 宏（本算子名） | ✅ 通过（TBE） |
| AI CPU | 无 AscendC 实现 | 有 AICPU 宏（本算子名） | ✅ 通过（TBE） |
| 仅API | 无任何实现 | 无宏 | ✅ 通过 |
| AI Core | 无任何实现 | 无宏 | ❌ 硬件单元错误（应为仅API） |
| 仅API | 有实现（任一） | 有宏（任一） | ❌ 硬件单元错误 |

**重要提示**：
- **LAUNCHER 宏必须精确匹配算子名**：`ADD_TO_LAUNCHER_LIST_AICORE({本算子PascalCase名})`
- 调用其他算子的 LAUNCHER 宏不算本算子的实现
- 例如：算子名 `grid_sample` → 宏应匹配 `ADD_TO_LAUNCHER_LIST_AICORE(GridSample)`
- **避免误判**：无 op_kernel/ 不等于无实现，TBE 实现通过 LAUNCHER 宏判断

### 检查项5：算子遗漏排查

**检查内容**：
排查除了 experimental 外其他目录是否有算子遗漏不在 op_list 中。

**判断规则**：
- 有 op_kernel/*.cpp 或 op_kernel_aicpu/*.cpp → 真正的算子目录
- 有 op_api/aclnn_*.cpp 或 ACLNNTYPE=aclnn/aclnn_inner → 真正的算子目录
- 无以上任何实现 → 非算子目录（公共模块等），不需要在 op_list 中呈现

**验证方法**：
```bash
# 遍历所有分类目录下的子目录
for category in $(ls {repo_root}); do
    for op_name in $(ls {repo_root}/${category}); do
        # 跳过 experimental 目录
        if [[ "${category}/${op_name}" == experimental/* ]]; then
            continue
        fi
        
        # 检查是否有实现
        has_impl=false
        if [[ -d "{repo_root}/${category}/${op_name}/op_kernel" ]]; then
            has_impl=true
        fi
        if [[ -d "{repo_root}/${category}/${op_name}/op_api" ]]; then
            has_impl=true
        fi
        
        # 有实现但不在 op_list 中
        if [[ "${has_impl}" == true ]] && [[ "${category}/${op_name}" not in op_list ]]; then
            echo "遗漏: ${category}/${op_name}"
        fi
    done
done
```

---

## 扫描流程

### ⚠️ 关键注意事项（必读）

**执行扫描时必须遵守以下规则**：

1. **递归检查子目录**：
   - op_kernel 文件可能在 `arch35/`、`arch50/` 等子目录中
   - 必须使用 `find -type f` 递归检查，不能仅检查顶层目录

2. **正确判断文件类型**：
   - `_def.cpp` 是定义文件，不属于实现文件
   - op_kernel 判断时排除 `_def.cpp`

3. **检查两种 op_api 结构**：
   - 标准：`op_api/aclnn_xxx.cpp`
   - 嵌套：`op_host/op_api/xxx.cpp`（文件名可能不以 aclnn 开头）

4. **硬件单元判断逻辑（简化版）**：
   - **默认 AI Core**（tbe 实现），无需检查 op_kernel/
   - **只需检查 AI CPU 实现**：op_kernel_aicpu/ 目录存在 → AI CPU 或 AI Core/AI CPU
   - **不需要检查 AscendC 实现**（op_kernel/），即使没有 op_kernel/，算子仍有 tbe 的 AI Core 实现

5. **op_host 列检查规则（简化版）**：
   - **只检查是否打勾**，不需要验证具体实现文件
   - **op_host 必须打勾**（√/✓/&check;），否则算子无法运行
   - 如果标记为 ×（叉），则为错误，需要修复

6. **避免误判**：
   - 目录存在 ≠ 实现存在
   - 必须检查实际文件内容，不能仅凭目录名判断
   - 硬件单元默认 AI Core，只有 AI CPU 实现才需特殊标记

### Step 1：解析 op_list.md 文档

```bash
# 定位文档
op_list_file="{repo_root}/docs/zh/op_list.md"

# 解析表格，提取每个算子行
# 提取字段：分类、目录名、op_kernel标记、op_host标记、op_api标记、op_graph标记、硬件单元、说明
```

### Step 2：遍历算子列表逐一验证

对于每个算子行：
1. 检查算子目录存在性
2. 检查分类正确性
3. 检查实现状态标记一致性
4. 检查硬件单元一致性
5. 检查链接跳转有效性

### Step 3：汇总扫描结果

统计：
- 通过项数
- 失败项数（按问题类型分类）
- 跳过的experimental算子数

### Step 4：生成报告与 Issue

生成 Markdown 报告 + Issue 文件（针对失败项）

---

## 输出格式

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（在 `{Skill特殊字段区域}` 增加）：

| 字段名称 | 内容说明 |
|---------|---------|
| op_list扫描统计 | 总算子数/目录缺失数/标记错误数/分类错误数/硬件单元错误数 |
| 算子详情表格 | 每个算子的验证结果（算子名/分类/目录状态/标记一致性/硬件一致性） |
| 问题详情列表 | 按问题类型分类的详细错误信息 |

### 报告模板（Issue友好格式）

```markdown
# {repo_type} 算子列表一致性扫描报告

**扫描时间**: {date}
**仓库路径**: `{repo_root}`
**扫描范围**: docs/zh/op_list.md

---

## 一、扫描概览

### 1.1 统计摘要

| 检查项 | 扫描数 | 通过数 | 失败数 |
|-------|-------|-------|-------|
| 算子目录存在性 | X | Y | Z |
| 算子分类正确性 | X | Y | Z |
| 实现状态标记一致性 | X | Y | Z |
| 硬件单元一致性 | X | Y | Z |
| **总计** | X | Y | Z |

### 1.2 问题分类统计

| 问题类型 | 数量 | 占比 |
|---------|------|------|
| 目录缺失 | X | X% |
| README缺失 | Y | Y% |
| 标记错误(op_kernel) | Z | Z% |
| 标记错误(op_api) | W | W% |
| 分类错误 | N | N% |
| 硬件单元错误 | M | M% |

---

## 二、算子详情验证结果

| 序号 | 算子目录 | 分类 | 目录状态 | op_kernel | op_host | op_api | op_graph | 硬件单元 | 整体状态 |
|:---:|---------|------|:-------:|:--------:|:-------:|:------:|:--------:|:--------:|:--------:|
| 1 | grid_sample | image | ✅存在 | ✅一致 | ✅一致 | ✅一致 | ✅一致 | ✅一致 | ✅通过 |
| 2 | xxx | xxx | ❌缺失 | - | - | - | - | - | ❌失败 |

---

## 三、问题详情（Issue格式）

### Issue #1: [Documentation|文档反馈]: [AI 识别] {算子名} 实现状态标记与实际不一致

Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）

docs/zh/op_list.md 第 {行号} 行

### Issues Section（问题文档片段）

op_list表格中 {算子名} 的 {列名} 标记为 {标记值}，但实际实现状态为 {实际状态}。

### Existing Issues（存在的问题）

- op_list显示 op_kernel = √，但 op_kernel/ 目录无 .asc/.cpp 文件
- 导致用户误以为该算子有 kernel 实现
- 影响文档准确性

---

**报告生成时间**: {timestamp}
```

---

## 输出检查项

完成扫描后，确保输出以下内容：

### 终端输出检查

| 检查项 | 内容 |
|-------|------|
| 统计摘要 | 各检查项的扫描数/通过数/失败数 |
| 问题分类统计 | 按问题类型（目录缺失、README缺失、标记错误等）统计 |
| 问题详情 | 每个失败算子的具体问题描述 |
| experimental跳过数 | 跳过的experimental目录下的算子数 |

### 报告文件检查

| 文件 | 位置 | 内容 |
|------|------|------|
| Markdown 报告 | `reports/{date}/{repo}/op-list-validation_report_{time}.md` | 完整扫描报告 |
| Issue 文件 | `reports/{date}/{repo}/issues/op_list_issue_{time}.md` | 问题详情（Issue格式） |

### Issue 创建流程

**核心原则**：
| 原则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别，所有发现问题都生成 Issue |
| **报告后询问提交** | 每次生成报告后，询问用户是否提交 Issue |
| **同类问题合并选项** | 同类问题涉及多个算子时，询问是否合并（按问题类型+仓库） |
| **自动化执行默认合并** | unified-scanner 调用时，默认合并创建 Issue |

### 执行模式

| 模式 | 说明 | Issue 创建方式 |
|------|------|---------------|
| **交互模式** | 用户直接调用 `/op-list-validation` | 询问用户选择合并方式 |
| **自动化模式** | unified-scanner 调用 | **默认合并创建 Issue**，无需询问 |

**流程概览**：

```
交互模式：
扫描完成 → 分类问题 → 询问合并 → 生成 Issue → 询问提交 → 执行提交

自动化模式（unified-scanner 调用）：
扫描完成 → 分类问题 → 【自动合并】 → 生成 Issue → 汇报结果
```

**合并场景示例**：
- 发现 10 个算子标记错误 → 合并标题：`[Documentation]: [AI 识别] {repo} op_list标记错误（10个算子）`
- 发现 5 个分类错误 → 合并标题：`[Documentation]: [AI 识别] {repo} 算子分类错误（5个算子）`

### 自动化模式 Issue 创建规则

**触发条件**：unified-scanner 调用时，无需用户交互

**默认行为**：
1. 按问题类型分类（标记错误/分类错误/目录缺失）
2. 每种类型生成一个合并 Issue
3. Issue 文件命名：`{repo}_op_list_{issue_type}_issue_{timestamp}.md`
4. 自动汇报 Issue 创建结果

**自动化执行示例**：
```python
# 自动按问题类型创建 Issue
for issue_type in ['marker_error', 'category_error', 'directory_missing']:
    if issue_count[issue_type] > 0:
        issue_file = f"reports/{date}/{repo}/issues/op_list_{issue_type}_issue_{time}.md"
        # 调用 gitcode-issue-creator Skill 生成 Issue
```

---

## 脚本直接调用

本 Skill 提供 Python 脚本，可直接执行全量验证：

```bash
# 扫描 ops-cv 仓库（全量验证）
python .opencode/skills/op-list-validation/scripts/op_list_scan.py --repo ops-cv

# 扫描 ops-nn 仓库（全量验证）
python .opencode/skills/op-list-validation/scripts/op_list_scan.py --repo ops-nn

# 指定输出路径
python .opencode/skills/op-list-validation/scripts/op_list_scan.py \
    --repo ops-math \
    --output reports/{date}/ops-math/op-list-validation_report_{time}.md

# 同时保存 JSON 数据
python .opencode/skills/op-list-validation/scripts/op_list_scan.py \
    --repo ops-transformer \
    --json reports/{date}/ops-transformer/op-list-validation_data.json
```

### 参数说明

| 参数 | 说明 | 必填 |
|------|------|:---:|
| `--repo` | 仓库类型（ops-math/ops-nn/ops-transformer/ops-cv） | ✅ |
| `--repo-root` | 仓库根目录（默认为当前目录下的 `{repo}/`） | ❌ |
| `--output` | Markdown 报告输出路径 | ❌ |
| `--json` | JSON 数据输出路径 | ❌ |

---

## 快速使用

```opencode
# 扫描 ops-cv 仓库的 op_list 一致性
/op-list-validation ops-cv

# 扫描 ops-nn 仓库
/op-list-validation ops-nn

# 扫描 ops-transformer 仓库
/op-list-validation ops-transformer

# 扫描 ops-math 仓库
/op-list-validation ops-math
```

---

## 技能文件结构

```
.opencode/skills/op-list-validation/
├── SKILL.md                           # 技能描述与流程
└── scripts/
    └── op_list_scan.py               # 全量验证脚本（Python）
```

---

## 与其他 Skill 的关系

| Skill | 关系 | 说明 |
|------|------|------|
| repo-docs-scan | 补充 | repo-docs-scan 检查 README 内容正确性，op-list-validation 检查 op_list 表格一致性 |
| op-doc-completeness | 前置 | op-doc-completeness 检查 README 存在性，op-list-validation 检查 README 链接有效性 |
| link-checker | 关联 | 可调用 link-checker 检查 op_list.md 中的断链 |

---

## 参考文档

- `references/op_list_format_spec.md`: op_list.md 格式规范详解
- `references/implementation_marker_spec.md`: 实现状态标记规范
---
name: scan-op-api-list
description: 算子接口列表一致性扫描技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 仓库的 docs/zh/op_api_list.md 文档，验证 aclnn 接口名、接口说明、确定性说明与实际代码实现的一致性。当用户需要验证aclnn接口文档准确性、检查op_api_list表格与实际实现匹配时使用。
---

# 算子接口列表一致性扫描

## 概述

本技能用于验证仓库级算子接口列表文档（`docs/zh/op_api_list.md`）与实际 aclnn 接口实现的一致性，帮助检测：
- 接口名是否与实际实现一致
- aclnn API 文档是否缺失（基于 ACLNNTYPE 判断）
- 接口链接是否能正常跳转到 aclnn API 文档
- 接口说明是否与功能实现一致
- 确定性说明是否与实际实现一致

**重要说明**：
- 只检查**列入 op_api_list.md 的接口**，未列入的不检查
- **跳过 experimental 目录**下的算子接口（生态开发者提供，不检查）
- aclnn 文档需求判断基于 **ACLNNTYPE + op_api/aclnn_xxx 文件**：
  - `ACLNNTYPE=aclnn` 或 `aclnn_inner` → 必须有文档
  - `ACLNNTYPE=aclnn_exclude` + 有 `aclnn_xxx` 文件 → 必须有文档
  - 其他情况 → 无需文档

---

## 输出目录结构

```
reports/
└── {date}/                                        # 日期目录（YYYYMMDD）
    └── {repo}/                                    # 仓库目录
        ├── op-api-list-validation_report_{time}.md  # 扫描报告
        ├── issues/
        │   └── op_api_list_issue_{time}.md        # Issue 文件
        └── his/                                   # 历史报告归档
```

---

## 入口参数

| 参数名 | 含义 | 取值约束 | 初值推断 |
|-------|------|---------|---------|
| repo_type | 仓库类型 | 枚举值["ops-math", "ops-nn", "ops-transformer", "ops-cv"] | 根据工作目录推断 |
| repo_root | 仓库根目录 | 绝对路径 | 支持自动检测，通过 `.opencode/scripts/repo_detector.py` 从任意嵌套目录向上遍历查找 ops-* 仓库根目录；也可手动指定 `--repo-root` |
| output_path | 报告输出路径 | 绝对路径 | 默认为 `reports/{date}/{repo}/op-api-list-validation_report_{time}.md` |

---

## op_api_list.md 文档格式

### 表格结构

op_api_list.md 使用 Markdown 表格格式，包含以下列：

| 列名 | 说明 | 数据来源 |
|------|------|---------|
| 接口名 | aclnn 接口名 + 链接 | op_api/aclnn_*.cpp 或 CMake 自动生成 |
| 说明 | 接口功能描述 | aclnn API 文档的功能说明章节 |
| 确定性说明（A2/A3） | Atlas A2/A3 产品确定性状态 | aclnn API 文档的约束说明章节 |
| 确定性说明（A5） | Atlas A5 产品确定性状态（可选） | aclnn API 文档的约束说明章节 |

### 确定性说明格式（固定话术三选一）

**aclnn文档中的固定话术**：

| 类型 | aclnn文档话术 |
|-----|-------------|
| 确定性实现 | aclnnXxx默认确定性实现 |
| 支持开启 | aclnnXxx默认非确定性实现,支持通过xxx开启确定性 |
| 不支持开启 | aclnnXxx默认非确定性实现,不支持通过xxx开启确定性 |

**op_api_list表格中的固定话术**：

| 类型 | op_api_list话术 |
|-----|---------------|
| 确定性实现 | 默认确定性实现 |
| 支持开启 | 默认非确定性实现,支持配置开启 |
| 不支持开启 | 默认非确定性实现,不支持配置开启 |

**特殊情况**：
```
-  （表示该产品不支持此接口）
```

**验证规则**：
- aclnn文档必须使用固定话术（三选一）
- op_api_list表格必须使用固定话术（三选一）
- 两边话术必须对应一致

---

## 检查项详细说明

### 检查项1：接口名一致性与 aclnn 文档完备性

**检查内容**：
1. op_api_list表格中的接口名是否与实际 aclnn 接口文件对应
2. 接口链接是否能正常跳转到 aclnn API 文档
3. 接口名格式是否正确（aclnn + PascalCase算子名）
4. aclnn API 文档是否缺失（基于 ACLNNTYPE 判断）

**experimental 目录排除**：
- 路径以 `experimental/` 开头的算子接口自动跳过（生态开发者提供，不检查）
- 在报告中标注为"跳过的experimental算子接口"

**接口名来源判断**：

| ACLNNTYPE | 接口来源 | 接口名格式 |
|-----------|---------|-----------|
| aclnn | CMake 自动生成 | aclnn{PascalCaseOpName} |
| aclnn_inner | CMake 自动生成 | aclnn{PascalCaseOpName}Inner |
| aclnn_exclude + op_api/aclnn_*.cpp | 手动实现 | 从文件名提取 |

**aclnn 文档需求判断**（基于 ACLNNTYPE）：

| ACLNNTYPE | op_api/aclnn_xxx | 是否需要aclnn文档 |
|-----------|-----------------|------------------|
| aclnn 或 aclnn_inner | - | ✅ 必须有 |
| aclnn_exclude | 有 aclnn_xxx.cpp/h | ✅ 必须有 |
| aclnn_exclude | 无 aclnn_xxx 文件 | ❌ 无需 |

**验证方法**：
```bash
# 从 op_api_list.md 提取接口名
grep -oP '\[aclnn[A-Za-z0-9]+\]' docs/zh/op_api_list.md

# 检查 ACLNNTYPE 参数
grep "ACLNNTYPE" {算子目录}/CMakeLists.txt

# 检查 op_api 目录是否有 aclnn_xxx 文件
find {算子目录}/op_api -name "aclnn_*.cpp" -o -name "aclnn_*.h"

# 检查 aclnn API 文档是否存在
ls {算子目录}/docs/aclnn*.md
```

**判定规则**：

| 情况 | 状态 | 问题类型 |
|------|------|---------|
| 接口名正确 + 文档存在 + 链接有效 | ✅ 通过 | - |
| 接口名错误 | ❌ 失败 | 接口名不一致 |
| 需要文档但aclnn文档不存在 | ❌ 失败 | aclnn文档缺失 |
| 无需文档（ACLNNTYPE判断） | ✅ 非问题 | 无需aclnn文档（标注） |
| 链接路径错误 | ❌ 失败 | 链接断链 |
| 接口名格式错误（非 PascalCase） | ❌ 失败 | 接口名格式错误 |
| experimental目录下的算子接口 | ⏭️ 跳过 | 不计入统计 |

**接口名转换规则**：
- 算子目录名 `grid_sample` → 接口名 `aclnnGridSample`（注意：部分接口可能有变体）
- 算子目录名 `upsample_bilinear2d_aa` → 接口名 `aclnnUpsampleBilinear2dAA`

**特殊情况处理**：
- 一个算子可能有多个 aclnn 接口（如 GridSample 有 GridSampler2D、GridSampler3D）
- 接口可能有 Backward 版本（如 GridSampler2DBackward）

---

### 检查项2：接口说明一致性

**检查内容**：
验证 op_api_list表格中的"说明"列是否与 aclnn API 文档的"功能说明"章节一致。

**验证方法**：
```bash
# 从 op_api_list.md 提取接口说明
# 从 aclnn API 文档提取功能说明
grep -A5 "## 功能说明" {算子目录}/docs/aclnn*.md
```

**一致性判断规则**：

| op_api_list说明 | aclnn文档功能说明 | 状态 |
|----------------|-----------------|------|
| 内容一致（语义相同） | 内容一致 | ✅ 通过 |
| 内容不一致 | 内容不一致 | ❌ 说明不一致 |
| 说明缺失 | 有功能说明 | ❌ 说明缺失 |
| 有说明 | 功能说明缺失 | ⚠️ 需补充文档 |

**说明提取规则**：
- op_api_list.md：提取表格中的说明文本
- aclnn文档：提取"功能说明"章节的第一段"接口功能"描述

---

### 检查项3：确定性说明一致性

**检查内容**：
验证 op_api_list表格中的"确定性说明"列是否与 aclnn API 文档一致。

**确定性判断逻辑（仅基于文档）**：

1. **首先看产品支持情况表格**：
   - 从 aclnn 文档的"产品支持情况"章节中解析产品支持状态
   - 如果某产品标记为 ×，则 op_api_list 对应列应为 `-`

2. **然后看约束说明中的确定性说明**：
   - 从 aclnn 文档的"约束说明"章节中查找"确定性计算"段落
   - 解析确定性说明的类型：
     - "默认确定性实现"
     - "默认非确定性实现，支持配置开启"
     - "默认非确定性实现，不支持配置开启"

3. **是否区分产品型号**：
   - 如果确定性说明未区分产品型号（如 `aclnnUpsampleBicubic2d默认确定性实现。`），则该说明对所有支持的产品生效
   - 如果确定性说明区分了产品型号（如 `<term>Atlas A2训练系列产品</term>：aclnnXxx默认确定性实现。`），则需分别对比各产品

4. **缺少确定性说明的处理**：
   - 如果 aclnn 文档缺少确定性说明，标记为问题（需要补充）

**验证方法**：
```bash
# 从 aclnn API 文档提取产品支持情况和确定性说明
grep -A20 "产品支持情况" {算子目录}/docs/aclnn*.md
grep -A5 "确定性计算" {算子目录}/docs/aclnn*.md
```

**判定规则**：

| op_api_list确定性说明 | aclnn文档确定性说明 | 状态 |
|----------------------|-------------------|------|
| 默认确定性实现 | "默认确定性实现" | ✅ 通过 |
| 默认非确定性，支持开启 | "支持通过aclrtCtxSetSysParamOpt开启确定性" | ✅ 通过 |
| 默认非确定性，不支持开启 | "不支持配置开启" | ✅ 通过 |
| - （产品不支持） | 产品表格标记× | ✅ 通过（跳过） |
| 默认确定性实现 | 实际为非确定性 | ❌ 确定性说明不一致 |
| 默认非确定性，支持开启 | 实际为确定性 | ❌ 确定性说明不一致 |
| 任意说明 | 缺少确定性说明 | ❌ 需补充确定性说明 |

**产品支持判断**：
- 如果 aclnn文档"产品支持情况"表格中某产品标记为 ×，则该产品确定性说明应为 `-`
- 如果产品标记为 √，则必须有确定性说明

### 检查项5：A2/A3与A5确定性说明一致性

**检查内容**：
如果 op_api_list表格中 A2/A3 和 A5 的确定性说明不一致，aclnn 文档必须特殊说明两者的不同。

**判定规则**：

| A2/A3说明 | A5说明 | aclnn文档区分 | 状态 |
|----------|--------|-------------|------|
| 一致 | 一致 | 不区分（使用'all'） | ✅ 通过 |
| 不一致 | 不一致 | 明确区分产品型号 | ✅ 通过 |
| 不一致 | 不一致 | 未区分产品型号 | ❌ 文档需补充产品型号说明 |

**验证方法**：
```bash
# 从 aclnn文档解析确定性说明（按产品型号）
# 如果 A2/A3 != A5，但文档使用 'all' → 报错
# 如果 A2/A3 != A5，但文档未完整区分 → 报错
```

---

### 检查项6：接口遗漏排查

**检查内容**：
排查是否有 aclnn 接口未列入 op_api_list。

**判断规则**：
- 有 aclnn 源码实现（aclnn_*.cpp） → 必须添加到表格
- ACLNNTYPE=aclnn/aclnn_inner 但不在列表中 → 可选添加（仅提醒）

**验证方法**：
```bash
# 遍历所有算子目录
for op_dir in $(find {repo_root} -type d -name "op_api"); do
    # 检查是否有 aclnn_*.cpp
    aclnn_files=$(find ${op_dir} -name "aclnn_*.cpp")
    
    if [[ -n "${aclnn_files}" ]]; then
        # 有实现但不在列表中 → 遗漏
        if [[ "${interface_name}" not in op_api_list ]]; then
            echo "遗漏: ${interface_name}"
        fi
    fi
done

# ACLNNTYPE=aclnn/aclnn_inner 可选登记
for op_dir in $(find {repo_root} -type d); do
    aclnn_type=$(grep ACLNNTYPE ${op_dir}/CMakeLists.txt)
    
    if [[ "${aclnn_type}" == "aclnn" ]] || [[ "${aclnn_type}" == "aclnn_inner" ]]; then
        if [[ "${interface_name}" not in op_api_list ]]; then
            echo "可选登记: ${interface_name} (${aclnn_type})"
        fi
    fi
done
```

**输出分类**：
- **必须添加**：有 aclnn 手动实现的接口遗漏
- **可选登记**：ACLNNTYPE=aclnn/aclnn_inner 不在列表中（仅提醒）

---

## 扫描流程

### Step 1：解析 op_api_list.md 文档

```bash
# 定位文档
op_api_list_file="{repo_root}/docs/zh/op_api_list.md"

# 解析表格，提取每个接口行
# 提取字段：接口名、链接路径、说明、确定性说明(A2/A3)、确定性说明(A5)
```

### Step 2：遍历接口列表逐一验证

对于每个接口行：
1. 提取接口名，转换为算子目录名
2. 检查 aclnn API 文档是否存在
3. 检查链接跳转有效性
4. 检查说明一致性
5. 检查确定性说明一致性

### Step 3：汇总扫描结果

统计：
- 通过项数
- 失败项数（按问题类型分类）
- 跳过的experimental算子接口数

### Step 4：生成报告与 Issue

生成 Markdown 报告 + Issue 文件（针对失败项）

---

## 输出格式

### 统一报告模板

本 Skill 使用统一报告模板，报告内容必须完整嵌入 Issue，不引用外部文件。

**本 Skill 特殊字段**（直接嵌入 Issue 内容）：

| 字段名称 | 内容说明 |
|---------|---------|
| op_api_list扫描统计 | 总接口数/接口名错误数/aclnn文档缺失数/链接断链数/说明不一致数/确定性错误数 |
| 接口详情表格 | 每个接口的验证结果（接口名/ACLNNTYPE/文档状态/链接状态/说明一致性/确定性一致性） |
| experimental跳过列表 | 跳过的experimental目录下的算子接口 |

### 报告模板（Issue友好格式）

```markdown
# {repo_type} 算子接口列表一致性扫描报告

**扫描时间**: {date}
**仓库路径**: `{repo_root}`
**扫描范围**: docs/zh/op_api_list.md

---

## 一、扫描概览

### 1.1 统计摘要

| 检查项 | 扫描数 | 通过数 | 失败数 |
|-------|-------|-------|-------|
| 接口名一致性 | X | Y | Z |
| 接口链接跳转 | X | Y | Z |
| 接口说明一致性 | X | Y | Z |
| 确定性说明一致性（A2/A3） | X | Y | Z |
| 确定性说明一致性（A5） | X | Y | Z |
| **总计** | X | Y | Z |

### 1.2 问题分类统计

| 问题类型 | 数量 | 占比 |
|---------|------|------|
| 接口名不一致 | X | X% |
| aclnn文档缺失 | Y | Y% |
| 链接断链 | Z | Z% |
| 说明不一致 | W | W% |
| 确定性说明错误 | N | N% |

---

## 二、接口详情验证结果

| 序号 | 接口名 | 算子目录 | 文档状态 | 链接状态 | 说明一致性 | 确定性一致性(A2/A3) | 确定性一致性(A5) | 整体状态 |
|:---:|-------|---------|:-------:|:-------:|:---------:|:------------------:|:---------------:|:--------:|
| 1 | aclnnGridSampler2D | grid_sample | ✅存在 | ✅有效 | ✅一致 | ✅一致 | ✅一致 | ✅通过 |
| 2 | aclnnXxx | xxx | ❌缺失 | ❌断链 | - | - | - | ❌失败 |

---

## 三、问题详情（Issue格式）

### Issue #1: [Documentation|文档反馈]: [AI 识别] aclnn{接口名} 确定性说明与实际不一致

Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）

docs/zh/op_api_list.md 第 {行号} 行

### Issues Section（问题文档片段）

op_api_list表格中 aclnn{接口名} 的确定性说明为 "{表格值}"，但 aclnn API 文档中声明为 "{实际值}"。

### Existing Issues（存在的问题）

- op_api_list显示 "默认确定性实现"，但 aclnn文档声明为非确定性
- 用户可能误解接口的确定性行为
- 影响用户对接口性能和一致性的判断

---

## 四、确定性说明规范说明

### 确定性说明格式规范

| 类型 | op_api_list格式 | aclnn文档格式 | 说明 |
|------|----------------|--------------|------|
| 确定性实现 | 默认确定性实现 | aclnn{Op}默认确定性实现 | 所有支持产品均确定性 |
| 非确定性支持开启 | 默认非确定性实现，支持配置开启 | 支持通过aclrtCtxSetSysParamOpt开启确定性 | 可配置开启确定性 |
| 非确定性不支持开启 | 默认非确定性实现，不支持配置开启 | 不支持配置开启 | 无法开启确定性 |
| 产品不支持 | - | 产品表格标记× | 该产品不支持此接口 |

### 接口名转换规则

| 算子目录名 | 接口名（PascalCase） | 示例 |
|-----------|---------------------|------|
| grid_sample | GridSample | aclnnGridSampler2D, aclnnGridSampler3D |
| upsample_bilinear2d_aa | UpsampleBilinear2dAA | aclnnUpsampleBilinear2dAA |
| three_interpolate_backward | ThreeInterpolateBackward | aclnnThreeInterpolateBackward |

---

**提交地址**: https://gitcode.com/cann/{repo}/issues/new
```

---

## 输出检查项

完成扫描后，确保输出以下内容：

### 终端输出检查

| 检查项 | 内容 |
|-------|------|
| 统计摘要 | 五类检查项的扫描数/通过数/失败数 |
| 问题分类统计 | 按问题类型统计数量和占比 |
| 问题详情 | 每个失败接口的具体问题描述 |
| experimental跳过数 | 跳过的experimental目录下的算子接口数 |

### 报告文件检查

| 文件 | 位置 | 内容 |
|------|------|------|
| Markdown 报告 | `reports/{date}/{repo}/op-api-list-validation_report_{time}.md` | 完整扫描报告 |
| Issue 文件 | `reports/{date}/{repo}/issues/op_api_list_issue_{time}.md` | 问题详情（Issue格式） |

---

## 脚本直接调用

本 Skill 提供 Python 脚本，可直接执行全量验证：

```bash
# 扫描 ops-cv 仓库（全量验证）
python .opencode/skills/op-api-list-validation/scripts/op_api_list_scan.py --repo ops-cv

# 扫描 ops-nn 仓库（全量验证）
python .opencode/skills/op-api-list-validation/scripts/op_api_list_scan.py --repo ops-nn

# 指定输出路径
python .opencode/skills/op-api-list-validation/scripts/op_api_list_scan.py \
    --repo ops-math \
    --output reports/{date}/ops-math/op-api-list-validation_report_{time}.md

# 同时保存 JSON 数据
python .opencode/skills/op-api-list-validation/scripts/op_api_list_scan.py \
    --repo ops-transformer \
    --json reports/{date}/ops-transformer/op-api-list-validation_data.json
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
# 扫描 ops-cv 仓库的 op_api_list 一致性
/op-api-list-validation ops-cv

# 扫描 ops-nn 仓库
/op-api-list-validation ops-nn

# 扫描 ops-transformer 仓库
/op-api-list-validation ops-transformer

# 扫描 ops-math 仓库
/op-api-list-validation ops-math
```

---

## 技能文件结构

```
.opencode/skills/op-api-list-validation/
├── SKILL.md                           # 技能描述与流程
└── scripts/
    └── op_api_list_scan.py           # 全量验证脚本（Python）
```

---

## 与其他 Skill 的关系

| Skill | 关系 | 说明 |
|------|------|------|
| op-doc-completeness | 补充 | op-doc-completeness 检查 aclnn 文档存在性，op-api-list-validation 检查文档内容一致性 |
| op-list-validation | 关联 | op-list-validation 检查 op_api 列标记，op-api-list-validation 检查具体接口详情 |
| link-checker | 关联 | 可调用 link-checker 检查 op_api_list.md 中的断链 |

---

## 参考文档

以下规范直接嵌入 Issue 内容，无需引用外部文件：

- op_api_list.md 格式规范：表格包含接口名、说明、确定性说明列
- 确定性说明规范：三种格式（确定性实现、非确定性支持开启、非确定性不支持开启）
- 接口名转换规则：算子目录名 snake_case → 接口名 PascalCase
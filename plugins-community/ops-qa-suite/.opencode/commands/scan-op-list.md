---
description: Scan operator list consistency (op_list.md vs actual implementation)
---

执行算子列表一致性扫描，分析目标：$ARGUMENTS

如果没有指定仓库，默认扫描 ops-cv。

支持的仓库类型：
- ops-math
- ops-nn
- ops-transformer
- ops-cv

---

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── {YYYYMMDD}/                           # 日期目录
    └── {repo}/                           # 仓库目录
        ├── scan-op-list_report_{HHMMSS}.md
        └── issues/
            └── op_list_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现标记错误、分类错误、硬件单元错误

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 目录缺失 | Documentation | ✅ **自动生成** |
| 实现状态标记错误 | Documentation | ✅ **自动生成** |
| 分类错误 | Documentation | ✅ **自动生成** |
| 硬件单元说明错误 | Documentation | ✅ **自动生成** |

---

## 检查项说明

### 检查项1：算子目录存在性

| 检查内容 | 验证方法 |
|---------|---------|
| op_list表格中的算子目录是否实际存在 | `ls {category}/{op_name}/` |
| README.md 是否存在 | `ls {category}/{op_name}/README.md` |
| 文档链接是否能跳转 | 链接路径检查 |

### 检查项2：算子分类正确性

| op_list分类 | 实际父目录 | 状态 |
|-------------|-----------|------|
| 一致 | 一致 | ✅ 通过 |
| 不一致 | 不一致 | ❌ 分类错误 |

### 检查项3：实现状态标记一致性

**标记格式识别（三种）**：
- ops-math: `√` / `×`
- ops-nn, ops-transformer: `✓` / `✗`
- ops-cv: `&check;` / `&cross;`

| 列名 | 实际文件检查 | 判断依据 |
|------|------------|---------|
| op_kernel | op_kernel/*.asc, *.cpp | 有文件 = √ |
| op_host | op_host/*_tiling.cpp, *_infershape.cpp | 有文件 = √ |
| op_api | ACLNNTYPE + op_api/aclnn_*.cpp | ACLNNTYPE=aclnn 或有aclnn文件 = √ |
| op_graph | op_graph/*_proto.* | 有proto文件 = √ |

### 检查项4：硬件单元一致性

| op_list硬件单元 | 实际实现判断 |
|----------------|------------|
| AI Core | op_kernel/ 存在 |
| AI CPU | op_kernel_aicpu/ 存在 |
| AI Core/AI CPU | 两者都存在 |

---

## 报告格式要求

采用 Issue 友好格式，便于直接转换为 GitCode Issue 提交：

### 报告结构

```
一、扫描概览
- 统计摘要表格
- 问题分类统计

二、算子详情验证结果
- 每个算子的验证结果表格

三、问题详情（Issue格式）
- 每个失败项按 Issue 模板组织

四、缺失算子列表
- 实际存在但未列入op_list的算子
```

### Issue 格式字段

| 字段 | 内容 |
|------|------|
| **标题** | `[Documentation|文档反馈]: {算子名} 实现状态标记与实际不一致` |
| **Document Link** | docs/zh/op_list.md 行号 |
| **Issues Section** | op_list显示值 vs 实际状态 |
| **Existing Issues** | 文档错误的影响 |
| **Expected Behavior** | 应修改的正确标记值 |
| **Suggested Fix** | 具体修复操作 |

---

## 执行流程

Step 1: 解析 docs/zh/op_list.md 表格
Step 2: 提取算子目录、分类、实现标记、硬件单元
Step 3: 验证算子目录存在性
Step 4: 验证分类正确性
Step 5: 验证实现状态标记一致性
Step 6: 验证硬件单元一致性
Step 7: 检查遗漏算子
Step 8: 生成报告（Issue格式）

---

## 完成后确认

- [ ] 已输出统计摘要表格
- [ ] 已输出问题分类统计
- [ ] 已输出算子详情验证结果表格
- [ ] 已输出问题详情（Issue格式）
- [ ] 已输出缺失算子列表（如有）
- [ ] 已生成报告 `reports/{date}/{repo}/scan-op-list_report_{time}.md`
- [ ] 已生成 Issue 文件（针对失败项）
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

示例用法：
- `/scan-op-list ops-cv` - 扫描 ops-cv 算子列表一致性
- `/scan-op-list ops-nn` - 扫描 ops-nn 算子列表一致性
- `/scan-op-list` - 默认扫描 ops-cv
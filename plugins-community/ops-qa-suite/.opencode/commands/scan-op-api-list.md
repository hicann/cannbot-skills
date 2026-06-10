---
description: Scan operator API list consistency (op_api_list.md vs actual aclnn implementation)
---

执行算子接口列表一致性扫描，分析目标：$ARGUMENTS

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
        ├── scan-op-api-list_report_{HHMMSS}.md
        └── issues/
            └── op_api_list_issue_{HHMMSS}.md
```

### Issue 自动生成规则

**触发时机**: 扫描完成后，发现接口名错误、链接断链、确定性说明错误

| 问题类型 | Issue 类型 | 自动生成 |
|---------|-----------|---------|
| 接口名不一致 | Documentation | ✅ **自动生成** |
| aclnn文档缺失 | Documentation | ✅ **自动生成** |
| 链接断链 | Documentation | ✅ **自动生成** |
| 确定性说明错误 | Documentation | ✅ **自动生成** |

---

## 检查项说明

### 检查项1：接口名一致性与链接跳转

| 检查内容 | 验证方法 |
|---------|---------|
| 接口名格式正确（aclnn + PascalCase） | 接口名格式检查 |
| aclnn API 文档存在 | `ls {算子目录}/docs/aclnn*.md` |
| 链接跳转有效 | 链接路径检查 |

**接口名来源判断**：

| ACLNNTYPE | 接口来源 | 接口名格式 |
|-----------|---------|-----------|
| aclnn | CMake 自动生成 | aclnn{PascalCaseOpName} |
| aclnn_inner | CMake 自动生成 | aclnn{PascalCaseOpName}Inner |
| aclnn_exclude + op_api/aclnn_*.cpp | 手动实现 | 从文件名提取 |

### 检查项2：接口说明一致性

| op_api_list说明 | aclnn文档功能说明 | 状态 |
|----------------|-----------------|------|
| 内容一致 | 内容一致 | ✅ 通过 |
| 内容不一致 | 内容不一致 | ❌ 说明不一致 |

### 检查项3：确定性说明一致性

**确定性说明格式（三种）**：

| 类型 | op_api_list格式 | aclnn文档格式 |
|------|----------------|--------------|
| 确定性实现 | 默认确定性实现 | aclnn{Op}默认确定性实现 |
| 非确定性支持开启 | 默认非确定性实现，支持配置开启 | 支持通过aclrtCtxSetSysParamOpt开启确定性 |
| 非确定性不支持开启 | 默认非确定性实现，不支持配置开启 | 不支持配置开启 |
| 产品不支持 | - | 产品表格标记× |

**一致性判断**：

| op_api_list说明 | aclnn文档声明 | _def.cpp属性 | 状态 |
|----------------|--------------|------------|------|
| 一致 | 一致 | 一致 | ✅ 通过 |
| 不一致 | 不一致 | 不一致 | ❌ 确定性说明错误 |

---

## 报告格式要求

采用 Issue 友好格式，便于直接转换为 GitCode Issue 提交：

### 报告结构

```
一、扫描概览
- 统计摘要表格
- 问题分类统计

二、接口详情验证结果
- 每个接口的验证结果表格

三、问题详情（Issue格式）
- 每个失败项按 Issue 模板组织

四、遗漏接口列表
- 实际存在但未列入op_api_list的接口

五、确定性说明规范说明
- 确定性格式规范表
- 接口名转换规则
```

### Issue 格式字段

| 字段 | 内容 |
|------|------|
| **标题** | `[Documentation|文档反馈]: aclnn{接口名} 确定性说明与实际不一致` |
| **Document Link** | docs/zh/op_api_list.md 行号 |
| **Issues Section** | op_api_list显示值 vs 实际值 |
| **Existing Issues** | 文档错误的影响 |
| **Expected Behavior** | 应修改的正确说明 |
| **Suggested Fix** | 具体修复操作 |

---

## 执行流程

Step 1: 解析 docs/zh/op_api_list.md 表格
Step 2: 提取接口名、链接、说明、确定性说明
Step 3: 验证接口名一致性
Step 4: 验证链接跳转有效性
Step 5: 验证接口说明一致性
Step 6: 验证确定性说明一致性
Step 7: 检查遗漏接口
Step 8: 生成报告（Issue格式）

---

## 完成后确认

- [ ] 已输出统计摘要表格
- [ ] 已输出问题分类统计
- [ ] 已输出接口详情验证结果表格
- [ ] 已输出问题详情（Issue格式）
- [ ] 已输出遗漏接口列表（如有）
- [ ] 已输出确定性说明规范说明
- [ ] 已生成报告 `reports/{date}/{repo}/scan-op-api-list_report_{time}.md`
- [ ] 已生成 Issue 文件（针对失败项）
- [ ] 已遵循统一规范 `templates/issue_workflow_spec.md`

示例用法：
- `/scan-op-api-list ops-cv` - 扫描 ops-cv 算子接口列表一致性
- `/scan-op-api-list ops-nn` - 扫描 ops-nn 算子接口列表一致性
- `/scan-op-api-list` - 默认扫描 ops-cv
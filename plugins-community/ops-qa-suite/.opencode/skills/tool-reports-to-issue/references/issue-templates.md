# GitCode Issue 预设模板

当仓库无 `.gitcode/ISSUE_TEMPLATE/` 或 `.github/ISSUE_TEMPLATE/` 目录时，使用本文件的预设模板。

---

## 模板列表

| 模板类型 | 标题格式 | 标签 | 适用场景 |
|---------|---------|------|---------|
| Bug-Report | `[Bug-Report|缺陷反馈]: <简述>` | bug-report | 发现缺陷需要反馈 |
| Documentation | `[Documentation|文档反馈]: <简述>` | documentation | 文档问题反馈 |
| Requirement | `[Requirement|需求建议]: <简述>` | requirement | 新需求建议 |
| Question | `[Question|问题咨询]: <简述>` | question | 咨询或讨论问题 |
| Blank | 无固定格式 | 无 | 上述都不满足时使用 |

---

## 1. Bug-Report|缺陷反馈

### 标题格式

**单算子**: `[Bug-Report|缺陷反馈]: [AI 识别] {repo} {op_name} {问题简述}`

**合并**: `[Bug-Report|缺陷反馈]: [AI 识别] {repo} {问题简述}（{n}个算子）`

### 必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `问题描述` | 描述产生了什么问题 | CMake 配置中 OPTYPE 参数值与算子目录名不匹配 |
| `环境信息` | 昇腾硬件型号与软件环境 | CANN 8.5.0, Ubuntu 22.04, Ascend910B1 |
| `重现步骤` | 如何重现该缺陷 | 1. 克隆仓库 2. 执行 cmake ... |
| `预期结果` | 期望的行为 | cmake 应正常通过，无 OPTYPE 错误 |

### Description 模板

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Describe the current behavior / 问题描述

{问题描述内容}

### Environment / 环境信息

**软件环境**:
- CANN 版本: {CANN版本，如 8.0.RC1、8.5.0 等}
- 操作系统: {OS版本，如 Ubuntu 22.04、CentOS 7.9 等}

**硬件环境**:
- NPU 型号: {芯片型号，如 Ascend910B1、Ascend910B2、Ascend310P 等}
- 服务器型号: {可选，如 A2、A3 服务器}

**问题环境**:
- 仓库: {repo}
- 问题类型: {issue_type}
- 问题文件数: {count}
- 问题性质: {BUG/规范问题}

### Steps to reproduce the issue / 重现步骤

{重现步骤内容}

### Describe the expected behavior / 预期结果

{预期结果内容}

### Related log / screenshot / 日志 / 截图

{日志/截图内容}

### Special notes for this issue / 备注 (Optional / 选填)

{备注内容}
```

---

## 2. Documentation|文档反馈

### 标题格式

**单算子**: `[Documentation|文档反馈]: [AI 识别] {repo} {op_name} {简述}`

**合并**: `[Documentation|文档反馈]: [AI 识别] {repo} {简述}（{n}个算子）`

### 必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `文档链接` | 有问题的文档链接 | https://gitcode.com/cann/ops-math/blob/main/docs/zh/README.md |
| `问题文档片段` | 问题文档片段或截图 | 第 10 行链接指向不存在的文件 |

### Description 模板

```markdown
Thanks for sending an issue! Please fill in the following template to help quickly solve your problem.

### Document Link（文档链接）

{文档链接}

### Issues Section（问题文档片段）

{问题文档片段/截图}

### Existing Issues（存在的问题）

{问题描述}

### Expected Behavior（预期结果）

{预期结果内容}
```

---

## 3. Requirement|需求建议

### 标题格式

**单算子**: `[Requirement|需求建议]: [AI 识别] {repo} {op_name} {简述}`

**合并**: `[Requirement|需求建议]: [AI 识别] {repo} {简述}（{n}个算子）`

### 必填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `背景信息` | 功能是什么，解决什么问题 | 该算子缺少 examples 测试用例，用户无法快速验证功能 |
| `信息来源` | 哪个部门/团队提出的需求 | 扫描报告分析结果 |

### 选填字段

| 字段 | 说明 | 示例 |
|------|------|------|
| `价值/作用` | 需求的价值，应用场景 | 提升用户上手体验，降低学习成本 |
| `设计方案` | 设计的总体思路 | 参考 Add 算子的 examples 结构 |

### Description 模板

```markdown
Thanks for sending an requirement! Please fill in the following template to help quickly solve your problem.

### Background（背景信息）

{背景信息内容}

### Origin（信息来源）

{信息来源}

### Benefit / Necessity（价值/作用）

{价值/作用内容}

### Design（设计方案）

{设计方案内容}
```

---

## 4. Question|问题咨询

### 标题格式

**单算子**: `[Question|问题咨询]: [AI 识别] {repo} {op_name} {简述}`

**合并**: `[Question|问题咨询]: [AI 识别] {repo} {简述}（{n}个算子）`

### 必填字段

| 字段 | 说明 |
|------|------|
| `问题描述` | 用户的问题描述 |

### Description 模板

```markdown
Welcome to ask questions and discuss with other members.

### 问题描述

{问题描述内容}
```

---

## 5. Blank|空白模板

### 标题格式

`[AI 识别] {用户自定义标题}`

### Description

用户自定义内容，无固定格式。

---

## YAML 表单模板格式（参考）

当仓库有 `.gitcode/ISSUE_TEMPLATE/*.yml` 时，格式如下：

```yaml
name: Bug-Report|缺陷反馈
description: 当您发现了一个缺陷，需要向社区反馈时，请使用此模板。
title: "[Bug-Report|缺陷反馈]: "
labels: ["bug-report"]
body:
  - type: markdown
    attributes:
      value: |
        感谢您提交 issues！请填写如下模板...
  - type: textarea
    attributes:
      label: 问题描述
      description: 请描述您的缺陷详细信息
      value: |
        ####  一、问题描述 （必填）
        ####  二、环境信息 （可选）
        ####  三、重现步骤 （可选）
        ####  四、预期结果 （可选）
    validations:
      required: true
```

### YAML 关键字段

| 字段 | 必填 | 说明 |
|------|:---:|------|
| `name` | ✅ | 模板名称（选择器显示） |
| `description` | ✅ | 模板描述（选择器显示） |
| `title` | ❌ | 预设标题前缀（如 `[Bug]:`） |
| `labels` | ❌ | 自动应用的标签列表 |
| `assignees` | ❌ | 自动分配的用户列表 |
| `body` | ✅ | 表单字段定义 |

### body 字段类型

| type | 说明 | 必填字段 |
|------|------|---------|
| `markdown` | 静态文本（说明信息） | `attributes.value` |
| `textarea` | 多行文本输入 | `attributes.label` |
| `input` | 单行文本输入 | `attributes.label` |
| `dropdown` | 下拉选择 | `attributes.label`, `attributes.options` |
| `checkboxes` | 多选框 | `attributes.label`, `attributes.options` |

---

## Markdown front-matter 格式（参考）

当仓库有 `.github/ISSUE_TEMPLATE/*.md` 时，格式如下：

```markdown
---
name: "Bug 报告"
about: "报告一个问题帮助我们改进"
title: "【BUG】:"
labels: ["BUG"]
assignees: 'username'
---

### BUG 类型

<!-- 请在这里描述你所遇到的 BUG 类型 -->

### 复现步骤

<!-- 请在这里描述你遇到该 BUG 时的页面及步骤 -->
```

### front-matter 字段

| 字段 | 必填 | 说明 |
|------|:---:|------|
| `name` | ✅ | 模板名称（含中文使用双引号） |
| `about` | ❌ | 模板解释说明 |
| `title` | ❌ | Issue 预设标题 |
| `labels` | ❌ | Issue 的 labels（多个需中括号） |
| `assignees` | ❌ | Issue 默认指派人 |
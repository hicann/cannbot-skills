---
name: tool-reports-to-issue
description: 扫描报告转 Issue 工具。根据扫描报告或问题列表批量生成 GitCode Issue，支持模板查询、智能合并、自动填充。核心能力：问题类型→模板匹配、同类问题合并策略、Issue 内容生成。通用能力（模板查询、API 提交）引用 gitcode-toolkit（软链接 infra）。当用户提供扫描报告、问题列表、要求"创建 Issue / 根据报告创建 Issue / 批量创建 Issue"时触发此 skill。
license: MIT
---

# 扫描报告转 Issue 工具

根据扫描报告或问题列表批量生成 GitCode Issue，支持模板查询、智能合并、自动填充。

---

## 核心原则

| 厩则 | 说明 |
|------|------|
| **所有问题都创建 Issue** | 不考虑问题级别（严重/中等/轻微），所有发现问题都生成 Issue 文件 |
| **报告后询问提交** | 每次生成报告后，必须询问用户是否提交到 GitCode |
| **同类问题合并选项** | 同类问题涉及多个算子时，询问用户是否合并成一个 Issue（按问题类型+仓库合并） |
| **模板优先查询仓库** | 创建 Issue 前先查询仓库模板，无模板时使用预设模板 |

---

## 引用 gitcode-toolkit（软链接 infra）

本 skill 的通用能力引用 gitcode-toolkit（软链接指向 infra/gitcode-toolkit）：

| 能力 | 来源 | 说明 |
|------|------|------|
| **环境预检** | [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md) | Step 0: token / git / curl / tmp |
| **模板查询** | [gitcode-toolkit/SKILL.md#Issue 创建工作流](../gitcode-toolkit/SKILL.md#issue-创建工作流) Step 2 | API 查询、优先级、预设模板 |
| **API 创建 Issue** | [gitcode-toolkit/SKILL.md#Issue 创建工作流](../gitcode-toolkit/SKILL.md#issue-创建工作流) Step 6 | curl 命令、权限、错误处理 |
| **日志记录** | [gitcode-toolkit/references/logging-conventions.md](../gitcode-toolkit/references/logging-conventions.md) | 日志格式规范 |

---

## 工作流程

```
Step 0    环境预检（引用 gitcode-toolkit）
Step 1    解析输入（扫描报告 / 问题列表）
Step 2    模板查询（引用 gitcode-toolkit：API 查询 → 预设模板 fallback）
Step 3    模板匹配（业务层：问题类型 → 模板选择）
Step 4    内容生成（业务层：智能合并 → 填充模板 → 生成 Issue 文件）
Step 5    用户确认
Step 6    执行提交（引用 gitcode-toolkit：API 创建或手动提交链接）
Step 7    记录日志（引用 gitcode-toolkit）

> **Step 3、4 为业务层**，本 skill 实现。其余步骤引用 gitcode-toolkit（软链接 infra）。

---

## Step 0：环境预检

> 引用 [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md)

执行时机：拿到输入数据后、解析任何业务字段前，立即执行。

必检项：
1. **GitCode Token**：用户消息 → 环境变量 `GITCODE_TOKEN` → 都没有则询问
2. **git / curl**：缺失则询问是否继续
3. **/tmp 可写**

---

## Step 1：解析输入（业务层）

### 输入类型

| 输入类型 | 解析方式 | 说明 |
|---------|---------|------|
| **扫描报告** | 从报告提取问题列表、问题分类、算子路径 | ops-qa-suite 扫描结果 |
| **问题列表** | 直接使用用户提供的问题列表 | 用户手动提供 |
| **问题描述** | 单个问题描述 | 用户手动创建单个 Issue |

### 扫描报告解析字段

| 字段 | 说明 | 用途 |
|------|------|------|
| `repo` | 仓库名称 | Issue 目标仓库 |
| `issue_type` | 问题类型（README缺失、UT缺失等） | 模板选择 |
| `op_name` | 算子名称 | Issue 标题、合并判断 |
| `op_class` | 算子分类 | Issue 内容生成 |
| `problem_count` | 问题数量 | 合并策略判断 |
| `description` | 问题描述 | Issue body 内容 |

---

## Step 2：模板查询

> 引用 [gitcode-toolkit/SKILL.md#Issue 创建工作流 Step 2](../gitcode-toolkit/SKILL.md#step-2-获取-issue-模板)

**模板优先级**（gitcode-toolkit 定义）：

| 优先级 | 路径 | 说明 |
|:---:|------|------|
| 1 | `.gitcode/ISSUE_TEMPLATE/*.zh-CN.yml` | GitCode 中文模板 |
| 2 | `.gitcode/ISSUE_TEMPLATE/*.yml` | GitCode YAML 表单 |
| 3 | `.github/ISSUE_TEMPLATE/*.yml` | GitHub YAML（兼容） |
| 4 | `.github/ISSUE_TEMPLATE/*.md` | GitHub Markdown（兼容） |
| 5 | **预设模板** | 仓库无模板时使用 |

**API 查询命令**（见 gitcode-toolkit）：

```bash
# 查询模板目录
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE?access_token=${token}"

# 获取模板内容并解码
curl -s "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/contents/.gitcode/ISSUE_TEMPLATE/bug_report.yml?access_token=${token}" | jq -r '.content' | base64 -d
```

---

## Step 3：模板匹配（业务层）

> 详见 [references/template-matching.md](references/template-matching.md)

### 问题类型 → 模板选择

| 问题类型 | 推荐模板 | 标签 | 标题前缀 |
|---------|---------|------|---------|
| **README缺失** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **aclnn文档缺失** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **CMake配置错误** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **UT缺失** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **UT测试失败** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **Examples缺失** | Requirement | requirement | `[Requirement|需求建议]:` |
| **Examples失败** | Bug-Report | bug-report | `[Bug-Report|缺陷反馈]:` |
| **断链问题** | Documentation | documentation | `[Documentation|文档反馈]:` |
| **功能需求** | Requirement | requirement | `[Requirement|需求建议]:` |
| **咨询问题** | Question | question | `[Question|问题咨询]:` |

### 模板不存在时的降级

按 `requirement` → `bug-report` → `documentation` 顺序降级。

---

## Step 4：内容生成（业务层）

> 详见 [references/batch-creation-guide.md](references/batch-creation-guide.md)

### 智能合并策略

```python
MERGE_THRESHOLD = {
    "auto_merge": 10,       # ≥10 个算子 → 自动合并（仅提示用户）
    "ask_merge": 3,         # 3-10 个算子 → 询问用户
    "no_merge": 1,          # 1-2 个算子 → 不合并
}
```

| 问题数量 | 策略 | 用户交互 |
|:-------:|------|---------|
| ≥ 10 | **自动合并** | 仅提示，不询问 |
| 3-9 | **询问合并** | 展示选项 |
| 1-2 | **不合并** | 直接创建 |

### Issue 文件命名

| 模式 | 命名格式 |
|------|---------|
| **合并** | `reports/{date}/{repo}/issues/{issue_type}_merged_issue_{time}.md` |
| **单算子** | `reports/{date}/{repo}/issues/{op_name}_{issue_type}_issue_{time}.md` |

### 标题格式

| 模板类型 | 单问题标题 | 合并标题 |
|---------|-----------|---------|
| Bug-Report | `[Bug-Report]: [AI 识别] {repo} {op_name} {简述}` | `[Bug-Report]: [AI 识别] {repo} {简述}（{n}个算子）` |
| Documentation | `[Documentation]: [AI 识别] {repo} {op_name} {简述}` | `[Documentation]: [AI 识别] {repo} {简述}（{n}个算子）` |
| Requirement | `[Requirement]: [AI 识别] {repo} {op_name} {简述}` | `[Requirement]: [AI 识别] {repo} {简述}（{n}个算子）` |

### 合并 Issue 内容格式

```markdown
# [模板类型]: [AI 识别] {repo} {问题类型}（{n}个算子）

**标签**: `{标签}`
**生成时间**: {timestamp}
**涉及算子数**: {n}

---

## Missing Operators List（问题算子列表）

| 序号 | 算子名称 | 算子路径 | 状态 |
|:---:|---------|---------|:---:|
| 1 | {op_name_1} | {repo}/{op_class}/{op_name_1}/ | ❌ {问题描述} |

## Existing Issues（存在的问题）

{问题描述详情}

## Suggested Fix（修复建议）

{修复建议}

---

**提交地址**: https://gitcode.com/cann/{repo}/issues/new
```

---

## Step 5：用户确认

生成 Issue 文件后，询问用户是否提交：

```
已生成 {n} 个 Issue 文件：

| 序号 | Issue 文件 | Issue 标题 | 涉及算子数 | 目标仓库 |
|:---:|-----------|-----------|:---:|---------|
| 1 | reports/{date}/{repo}/issues/readme_missing_merged_issue_{time}.md | [Documentation]: ... | 15 | cann/{repo} |

是否提交 Issue 到对应仓库？
1. 是，全部提交
2. 是，选择提交
3. 否，暂不提交
4. 否，手动提交
```

---

## Step 6：执行提交

> 引用 [gitcode-toolkit/SKILL.md#Issue 创建工作流 Step 6](../gitcode-toolkit/SKILL.md#step-6-创建-issue)

**curl 命令**（见 gitcode-toolkit）：

```bash
# 创建 Issue
curl -X POST "https://api.gitcode.com/api/v5/repos/${owner}/${repo}/issues?access_token=${token}" \
  -H "Content-Type: application/json" \
  -d '{"title": "${title}", "body": "${body}", "labels": "${labels}"}'
```

**权限**：Reporter+ 可 API 创建，Guest 需手动提交。

**失败处理**：
- 403：降级为手动提交（提供链接）
- 401：提示提供新 token
- 404：确认仓库路径

---

## Step 7：记录日志

> 引用 [gitcode-toolkit/references/logging-conventions.md](../gitcode-toolkit/references/logging-conventions.md)

日志文件：`logs/issue-create_{YYYYMMDD}_{HHMMSS}.log`

---

## 脚本工具

| 脚本 | 功能 | 用法 |
|------|------|------|
| `scripts/generate_issue_md.py` | 生成 Issue MD 文件 | `python generate_issue_md.py --repo ops-math --template bug-report --summary "..."` |

---

## 参考文档

**本 skill 业务文档**：
- [references/issue-templates.md](references/issue-templates.md) — 预设模板（Bug-Report、Documentation、Requirement、Question、Blank）
- [references/batch-creation-guide.md](references/batch-creation-guide.md) — 批量创建指南（合并策略、文件命名、自动填充）
- [references/template-matching.md](references/template-matching.md) — 模板匹配规则（问题类型 → 模板选择）

**引用 gitcode-toolkit 文档**：
- [gitcode-toolkit/SKILL.md#Issue 创建工作流](../gitcode-toolkit/SKILL.md#issue-创建工作流) — 完整 Issue 创建流程（Step 1-7）
- [gitcode-toolkit/references/gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md) — GitCode Issue API
- [gitcode-toolkit/references/env-check.md](../gitcode-toolkit/references/env-check.md) — 环境预检
- [gitcode-toolkit/references/token-config.md](../gitcode-toolkit/references/token-config.md) — Token 配置
- [gitcode-toolkit/references/logging-conventions.md](../gitcode-toolkit/references/logging-conventions.md) — 日志规范
---
name: fixer-broken-link
description: Ascend C 算子仓库 Markdown 断链修复技能。用于扫描 ops-math/ops-nn/ops-transformer/ops-cv 算子仓库的 Markdown 文件断链，自动修复并创建 PR。当用户需要修复文档断链、检查链接有效性时使用。
---

# Markdown 断链修复技能

## API 与环境配置

本 skill 的 PR 创建功能引用 gitcode-toolkit（软链接 infra）的 PR 创建工作流：

- [gitcode-toolkit/SKILL.md](../gitcode-toolkit/SKILL.md) — PR 创建工作流（Step 1-7）
- [gitcode-api.md](../gitcode-toolkit/references/gitcode-api.md) — GitCode PR API 详细文档
- [env-check.md](../gitcode-toolkit/references/env-check.md) — 环境预检（token / git / curl / /tmp）
- [token-config.md](../gitcode-toolkit/references/token-config.md) — Token 配置优先级

## 概述

扫描仓库 Markdown 文件中的断链，自动修复并创建 PR，支持：
- **断链检测**：扫描 `.md` 文件中的链接（相对路径、绝对路径）
- **自动修复**：修正路径错误、删除不存在的链接
- **批量处理**：支持批量修复多个断链
- **PR 创建**：修复后自动创建 PR（可选）

## 入口参数

| 参数 | 必填 | 说明 |
|------|:---:|------|
| `repo` | ✅ | 仓库名（ops-math, ops-nn, ops-transformer, ops-cv） |
| `--scope` | ❌ | 扫描范围：`all`（全部）、`readme`（仅 README） |
| `--fix` | ❌ | 自动修复断链（默认仅报告） |
| `--create-pr` | ❌ | 修复后创建 PR |
| `--dry-run` | ❌ | 仅模拟修复，不实际修改文件 |

## 执行流程

```
Step 1: 仓库检测与准备
    │
    ├── 检测仓库路径（当前目录/父目录/自动 clone）
    ├── 更新到 master 分支最新代码
    │
    ▼
Step 2: 断链扫描
    │
    ├── 扫描所有 Markdown 文件
    ├── 解析链接（相对路径、绝对路径、外部链接）
    ├── 验证链接目标是否存在
    ├── 分类断链类型
    │
    ▼
Step 3: 断链分析与分类
    │
    ├── 路径错误：修正路径
    ├── 文件不存在：删除链接或标注待补充
    ├── 外部链接：跳过（无法修复）
    ├── 链接换行：合并单行
    │
    ▼
Step 4: 自动修复（可选）
    │
    ├── 修正路径错误
    ├── 删除不存在链接
    ├── 合并换行链接
    │
    ▼
Step 5: 创建 PR（可选）
    │
    ├── 创建修复分支
    ├── 提交修改
    ├── 推送到 fork
    ├── 创建 PR 到 upstream
```

## 断链类型与修复策略

| 断链类型 | 示例 | 修复策略 |
|---------|------|---------|
| **路径层级错误** | `./examples/test.cpp` → `./examples/arch35/test.cpp` | 修正路径 |
| **文件不存在** | `./docs/api.md`（目录不存在） | 删除链接，标注待补充 |
| **链接换行** | `[text](path/` + `file.cpp)` | 合并单行 |
| **跨算子引用错误** | `../other_op/examples/` | 修正路径或删除 |
| **外部链接失效** | `https://example.com/doc.md` | 跳过，记录报告 |

## 报告与 Issue 输出规范

> **遵循统一规范**: 详见 `../../templates/issue_workflow_spec.md`

### 目录结构

```
reports/
└── broken-link-fixer/
    ├── {repo}_broken_link_report_{timestamp}.md     # 断链扫描报告
    ├── {repo}_broken_link_fix_report_{timestamp}.md # 修复报告
    ├── issues/
    │   ├── {repo}_broken_link_high_{timestamp}.md   # 高优先级断链 Issue
    │   ├── {repo}_broken_link_medium_{timestamp}.md # 中优先级断链 Issue
    │   └── {repo}_broken_link_low_{timestamp}.md    # 低优先级断链 Issue
    └── his/
```

### Issue 自动生成规则

| 问题类型 | Issue 类型 | 优先级 | 自动生成 |
|---------|:---:|:---:|:---:|
| 路径错误（可修复） | Documentation | 中 | ✅ |
| 文件不存在（需人工） | Documentation | 低 | ✅ |
| 外部链接失效 | Documentation | 低 | ✅（标注无法自动修复） |

## Issue 模板格式

```markdown
# [Documentation]: [AI 识别] {repo} README 断链问题

## 文档链接

{readme_path}

## 问题摘要

{断链数量} 处断链分布在 {文件数} 个 README 中：

| 断链类型 | 数量 | 可自动修复 |
|---------|:---:|:---:|
| 路径错误 | {n} | ✅ |
| 文件不存在 | {n} | ❌ |
| 外部链接失效 | {n} | ❌ |

## 详细问题列表

### 可自动修复

| README | 链接 | 问题 | 修复方案 |
|--------|------|------|---------|

### 需人工处理

| README | 链接 | 问题 | 建议 |
|--------|------|------|---------|

## 建议修复

1. 使用 `fixer-broken-link` Skill 自动修复可修复的断链
2. 手动处理需要补充文档的断链
```

## 快速使用

```bash
# 扫描断链（仅报告）
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math

# 扫描并自动修复
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math --fix

# 修复并创建 PR
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math --fix --create-pr

# 仅扫描 README
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math --scope readme

# dry-run 模式
python .opencode/skills/fixer-broken-link/scripts/scan_links.py --repo ops-math --fix --dry-run
```

## PR 创建流程

修复完成后，如需创建 PR，请引用 gitcode-toolkit 的 PR 创建工作流：

```
1. 扫描断链 → fixer-broken-link（仅报告）
2. 分析结果 → 确认可修复的断链
3. 自动修复 → fixer-broken-link --fix
4. 创建 PR  → 参考 gitcode-toolkit/SKILL.md 的 PR 创建工作流
```

**PR 创建步骤**（详见 gitcode-toolkit/SKILL.md）：

| 步骤 | 说明 |
|-----|------|
| Step 1 | 获取信息（分支名、commit历史、目标仓库） |
| Step 2 | 获取 PR 模板（`.gitcode/PULL_REQUEST_TEMPLATE.zh-CN.md`） |
| Step 3 | 分析并填充模板（从 commit messages 汇总生成） |
| Step 4 | 用户确认 |
| Step 5 | 推送分支 |
| Step 6 | 创建 PR（调用 GitCode API） |
| Step 7 | 记录日志 |

## 技能文件结构

```
.opencode/skills/fixer-broken-link/
├── SKILL.md              # 本文件
└── scripts/
    ├── scan_links.py     # 断链扫描脚本
    ├── fix_links.py      # 断链修复脚本
    └── create_fix_pr.py  # 创建修复 PR 脚本
```
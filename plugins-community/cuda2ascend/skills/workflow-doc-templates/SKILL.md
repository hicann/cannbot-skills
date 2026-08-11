---
name: workflow-doc-templates
description: 交付件模板，提供设计文档、验收报告等中间交付件的格式模板。触发：产出需求/方案/验收报告/算子文档/开发日志/Issue 等交付件时，先加载对应模板作为格式基准。
---

# 交付件模板索引

本仓交付件的默认格式模板集中在 `references/` 下，文件名按工作流统一流程表编号 + 中文标题命名（共享件 `环境信息.md` 与算子无关，无编号前缀）。产出对应交付件时，先读取模板作为格式基准，按占位符 `{...}` 填充实际内容。

## 模板索引（编号对应流程表）

| 交付件 | 流程编号 | 模板 |
|--------|----------|------|
| 环境信息文档 | 0 | [references/环境信息.md](references/环境信息.md) |
| 需求文档 | 1.1 | [references/1.1-需求分析.md](references/1.1-需求分析.md) |
| 测试方案文档 | 2.1 | [references/2.1-测试方案设计.md](references/2.1-测试方案设计.md) |
| 开发方案文档 | 2.2 | [references/2.2-开发方案设计.md](references/2.2-开发方案设计.md) |
| 联调报告 | 3.4 | [references/3.4-联调报告.md](references/3.4-联调报告.md) |
| 功能验收报告 | CP3 | [references/CP3-功能验收报告.md](references/CP3-功能验收报告.md) |
| 性能验收报告 | CP4 | [references/CP4-性能验收报告.md](references/CP4-性能验收报告.md) |
| 代码检视报告 | CP5 | [references/CP5-代码检视报告.md](references/CP5-代码检视报告.md) |
| 算子文档 | 6.1 | [references/6.1-算子文档.md](references/6.1-算子文档.md) |
| 开发报告 | 7.1 | [references/7.1-开发报告.md](references/7.1-开发报告.md) |
| 经验总结 | 7.2 | [references/7.2-经验总结.md](references/7.2-经验总结.md) |

## 跨阶段模板

| 交付件 | 用途 | 模板 |
|--------|------|------|
| 开发日志 | 全程状态与进度跟踪 | [references/LOG-开发日志.md](references/LOG-开发日志.md) |
| 问题记录 | 开发中问题的记录与闭环 | [references/Issue-问题记录.md](references/Issue-问题记录.md) |

## 说明

- 交付件模板与流程表编号一一对应。

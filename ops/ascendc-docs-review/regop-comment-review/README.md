# 算子原型注释审查 Skill

## 介绍

算子原型注释审查 Skill 是一款交互式注释审查工具，专为 Committer 和 Reviewer 设计，帮助高效审查 GitCode 上 Pull Request 中算子原型头文件（通常位于 `op_graph/inc` 的 `.h` 文件）的 `/** ... */` 注释变更。仅关注原型注释的规格导出合规性，不审查代码文件。也支持本地文件、目录、注释片段的评审。

## 核心功能

- **原型与注释一致性**：算子名、参数名、个数、类型与 `REG_OP` 定义完全一致；PR 中原型与注释同步修改。
- **通用注释格式**：注释块形式、`\n` 换行、`@li`、表格、公式乘号空格、无多余符号、英文写作。
- **字段写作规范**：Brief、Inputs、Attributes、Outputs、Constraints、Data Types、可选字段（Graphs/see/Quantization/Third-party/Restrictions）逐字段检视。
- **芯片约束规范**：约束按产品英文名分条目，严禁出现 910B、910C、V350 等芯片代号。
- **仅输出报告**：审查产物为结构化审查报告，不向 GitCode 提交任何评论。
- **结构化报告**：生成清晰的审查报告，按严重程度（CRITICAL/HIGH/MEDIUM/LOW）分类。

## 支持的审查维度

| 维度 | 最高严重级 | 说明 |
|------|-----------|------|
| 原型与注释一致性 | CRITICAL | 算子名、参数名/个数/类型与 REG_OP 定义完全一致；PR 中原型与注释同步修改 |
| 通用注释格式 | HIGH | 注释块形式、\n 换行、@li、表格、公式、多余符号、英文写作 |
| 功能简介 Brief | HIGH | @brief 无冒号、功能描述详尽、代码段标记 |
| 输入 Inputs | CRITICAL | 字段名精确、逐参数解释完整 |
| 属性 Attributes | CRITICAL | 字段名精确、默认值、Reserved 标注 |
| 输出 Outputs | CRITICAL | 字段名精确、逐参数解释完整 |
| 约束 Constraints | CRITICAL | 芯片约束用产品名、严禁芯片代号、废弃话术 |
| 数据类型 Data Types | MEDIUM | 由脚本自动生成，不得手写 |
| 可选字段 | HIGH | Graphs/see/Quantization/Third-party/Restrictions |

## 使用方式

支持两种评审场景，通过参数区分：

| 场景 | 调用方式 | 分支要求 |
|------|---------|---------|
| 用户本地评审 | `/regop-comment-review "文档路径"` | 无需切换分支，直接在本地仓库当前状态评审 |
| 评审其他用户 PR | `/regop-comment-review PR：<pr_number>` | 必须先切换到 PR 对应分支 |

审查产物为结构化审查报告，不向 GitCode 提交任何评论。

## 适用人群

- **Committer**：在合并代码前进行算子原型注释质量把控
- **Reviewer**：高效完成原型头文件注释审查任务
- **团队**：提升算子规格导出资料的质量，避免注释问题传导到发布件

## 技术栈

- OpenCode LLM 集成
- Git（PR 模式下用于同步远程分支与获取 diff）

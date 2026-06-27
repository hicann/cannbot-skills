---
team_name: ops-code-reviewer
eval_mode: text
---

# Case 1: 代码审查的核心流程和维度

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: ascendc-code-review;ascendc-code-summarizer;ascendc-precision-debug

## Prompt

我写了一个 Ascend C 算子，想做个代码审查。请介绍一下 ops-code-reviewer 团队的审查流程，审查主要关注哪些维度？

## Expected Output

回复应覆盖以下要点：
1. 代码审查流程包含加载 ascendc-code-review skill 并按完整工作流执行
2. 审查关注安全规范检查、代码质量评估等多个维度
3. 审查过程包含行号校对、问题分类、最终判定等步骤
4. 报告统一撰写，子 Agent 专注执行审查任务

## Expectations

- [contains] ascendc-code-review

---

# Case 2: 不适用代码审查的场景

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我有一段 Python 数据处理脚本，不是 Ascend C 算子代码，可以用 ops-code-reviewer 做审查吗？它适合审查什么样的代码？

## Expected Output

回复应说明 ops-code-reviewer 专注于 Ascend C 算子代码的审查，非算子代码（如 Python 数据处理脚本）不是它的审查范围。审查主要面向 Ascend C Kernel 实现、Tiling 策略、API 使用正确性等算子开发相关的代码质量

## Expectations

---

# Case 3: 审查前需要确认的信息

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想让你帮我审查一下代码，先告诉我你需要哪些信息，不要直接开始审查。

## Expected Output

回复应在审查前主动确认必要信息：待审查的代码文件路径、算子功能描述、审查的重点关注领域（如安全性、性能、规范等），而不是在缺少这些信息的情况下直接开始审查

## Expectations


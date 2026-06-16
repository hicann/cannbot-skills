---
skill_name: ascendc-performance-best-practices
eval_mode: text
---
# Case 1: 按算子族组织的优化知识

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-performance-best-practices 技能如何组织性能优化知识？有哪些算子族分类？它们分别面向什么目标架构？不需要执行任何工具调用。

## Expected Output

回复应说明该技能按算子族（operator family）组织性能优化知识，每个算子族下汇集对应的优化经验与参考代码总结。具体的算子族分类和目标架构信息在 SKILL.md 中定义，如需详细分类清单建议加载技能后查看完整 SKILL.md。

## Expectations

---

# Case 2: MatMul 族在 DAV_3510 上的优化

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-performance-best-practices 中 MatMul 算子族有哪些优化知识？在 DAV_3510 架构上适用的优化指南是什么？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应引用 MatMul 族的优化策略内容，说明 DAV_3510 架构上 MatMul 算子的优化策略和最佳实践。应列出 MatMul 族包含的变体和可用的优化策略类型。

## Expectations

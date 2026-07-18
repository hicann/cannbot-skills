---
skill_name: ascendc-performance-best-practices
eval_mode: text
---
# Case 2: MatMul 族在 DAV_3510 上的优化

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

ascendc-performance-best-practices 中 MatMul 算子族有哪些优化知识？在 DAV_3510 架构上适用的优化指南是什么？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应引用 MatMul 族的优化策略内容，说明 DAV_3510 架构上 MatMul 算子的优化策略和最佳实践。应列出 MatMul 族包含的变体和可用的优化策略类型。

---

# Case 3: Vector 算子性能优化咨询（正向看护）

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 170000
- Ascend Platform: A2
- Distractor skills: ascendc-perf-optimize;ascendc-tiling-design

## Prompt

我想了解 ascendc-performance-best-practices 中 Vector 算子族在 DAV_3510 上的优化策略和最佳实践，请加载技能帮我查看具体内容。

## Expected Output

回复应介绍 Vector 算子族在 DAV_3510 上的优化思路和主要方向。

---

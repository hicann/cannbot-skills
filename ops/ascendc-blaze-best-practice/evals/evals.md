---
skill_name: ascendc-blaze-best-practice
eval_mode: text
---
# Case 1: Matmul Blaze 三模板选型

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C Matmul 单算子开发中，Blaze 路径提供了三种模板选项：纯AIC、StreamK、FixpOpti。请介绍这三种模板各自的特点和适用场景。在实际项目中选择时应该考虑哪些因素？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应对比纯AIC、StreamK 和 FixpOpti 三种 Blaze 模板的适用场景和特点，帮助开发者根据算子特征选择合适的模板。


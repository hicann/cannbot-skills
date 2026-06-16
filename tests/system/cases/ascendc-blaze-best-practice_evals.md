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

## Expectations

---

# Case 2: RegBase 与 MemBase Epilogue 路径选择

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C Blaze 路径的 Matmul 算子开发中，epilogue 阶段有 RegBase 和 MemBase 两种路径。请介绍这两种路径的区别，哪一种路径是推荐的？各自适用于什么场景？RegBase 路径使用哪些 API？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 Blaze epilogue 路径选择的推荐策略：
- 推荐使用 RegBase 路径（epilogue_fusion_regbase.h），使用 __VEC_SCOPE__ + AscendC::Reg::* API
- MemBase 路径（epilogue_fusion_membase.h）仅适用于单个 vector 操作且有明确可用 AscendC::  API 的场景，如 AscendC::Mul/Add/Div
- RegBase 路径提供了更好的性能和灵活性，是默认推荐方案
- RegBase API 约束和陷阱可参考 ascendc-regbase-best-practice 技能

## Expectations

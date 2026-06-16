---
skill_name: ops-precision-standard
eval_mode: text
---
# Case 1: 精度标准选择的决策流程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C 算子开发中，如何选择合适的精度标准？请说明决策树的判断流程。不需要执行任何工具调用。

## Expected Output

回复应说明如何根据算子类型和数据类型选择合适的精度标准。

## Expectations

---

# Case 2: FP16 浮点运算的精度验收标准

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

Ascend C FP16 浮点计算的精度验收标准是什么？atol 和 rtol 的参考值是多少？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 Ascend C FP16 浮点计算的精度验收标准，包括 atol 和 rtol 的参考值范围，以及验收判断公式。

## Expectations

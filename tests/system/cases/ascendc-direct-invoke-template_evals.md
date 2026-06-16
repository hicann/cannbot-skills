---
skill_name: ascendc-direct-invoke-template
eval_mode: text
---
# Case 1: 创建 Vector 算子直调工程的方法

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍如何使用 ascendc-direct-invoke-template 技能创建一个 Ascend C Kernel 直调工程。我想开发一个 Vector 算子（比如 Add），应该使用什么模板？需要遵循哪些步骤？不需要执行任何工具调用。

## Expected Output

回复应说明创建 Vector 算子直调工程的方法：使用 add_custom 样例工程作为模板，修改 kernel.cpp/kernel.h 中的核函数实现和 host.cpp 中的调用参数，调整 CMakeLists.txt。该模板包含完整的 Vector 直调流程（<<<>>> 内核调用、数据搬入搬出、结果比对）。

## Expectations

---

# Case 2: 直调模板不适用的场景

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

ascendc-direct-invoke-template 技能有哪些不适用的场景？哪些算子类型不能使用这个模板？请简要说明。不需要执行任何工具调用。

## Expected Output

回复应说明 ascendc-direct-invoke-template 的使用边界：纯 Matmul/Cube 算子（非融合场景）不在本模板覆盖范围内，非 mxfp8 的 Cube+Vector 混合算子也不支持。模板主要覆盖单核 Vector 算子和 mxfp8 matmul + eltwise 融合两类场景。

## Expectations

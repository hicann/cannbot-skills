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

---

# Case 3: 创建 Kernel 直调工程（正向看护）

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 360000
- Max Tokens (glm-5): 340000
- Ascend Platform: A2
- Distractor skills: ascendc-registry-invoke-template;ascendc-api-best-practices

## Prompt

我要创建一个 Ascend C Kernel 直调工程来快速验证一个 Vector Add 算子，请加载 ascendc-direct-invoke-template 技能指导我完成工程搭建。

## Expected Output

回复应指导使用 add_custom 样例工程作为模板创建 Kernel 直调工程，说明需要修改的核心文件（kernel.cpp、host.cpp、CMakeLists.txt）和关键步骤。

---

# Case 4: Python 数据处理咨询（负向看护）

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我有一个包含缺失值的 CSV 数据集，想用 Python 做数据清洗和预处理，请问 pandas 有哪些好用的函数可以用？

## Expected Output

回复应介绍 Python 数据清洗的常见思路和方法，不涉及 Ascend C 算子开发相关内容。


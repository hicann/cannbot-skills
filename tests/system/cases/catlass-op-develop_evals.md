---
skill_name: catlass-op-develop
eval_mode: text
---
# Case 1: CATLASS Kernel 代码生成过程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

CATLASS 算子从设计选型到生成最终 kernel 代码的流程是什么？最终产出哪些代码？不需要执行任何工具调用。

## Expected Output

回复应说明 CATLASS kernel 代码生成流程：通过 using 链依次组装 BlockMmad、BlockEpilogue、BlockScheduler、Kernel 组件；然后构造 Kernel::Params 对象传入设备相关参数（如形状、指针、workspace 等）；最后在 Device 侧通过 Kernel{}(params) 完成调用。最终产出包括 kernel using 链、Params 构造代码和 Device 端调用代码。

## Expectations

---

# Case 2: CATLASS 实现前预备知识阅读要求

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在编写 CATLASS kernel 实现代码之前，必须先阅读哪些资料？请说明必须完成的预备知识阅读步骤。不需要执行任何工具调用。

## Expected Output

回复应说明必须先阅读工作区 `./catlass/` 目录下的三部分内容：README.md（了解库定位和目录结构）、docs/ 目录（理解算子组装知识与实现约束）、以及设计文档指定的参考 examples/ 样例目录（确认组件组合与 main() 到 op_kernel 的拆分模式）。必须强调未完成上述阅读禁止进入实现。

## Expectations

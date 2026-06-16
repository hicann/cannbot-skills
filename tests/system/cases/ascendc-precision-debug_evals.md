---
skill_name: ascendc-precision-debug
eval_mode: text
---
# Case 1: 精度调试的核心流程与哲学

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C 算子开发中，遇到精度问题时应该如何调试？精度调试的核心方法论是什么？请介绍精度调试的四个主要步骤，以及为什么 FP16 在累加时容易出现精度损失（Catastrophic Cancellation）。不需要执行任何工具调用。

## Expected Output

回复应说明 Ascend C 精度调试的核心方法论是"逐层缩小、分阶段验证、定位首个异常点"。四个主要步骤为数据比对导出、分阶段验证、流水线同步检查和精度损失修复。还应解释 FP16 Catastrophic Cancellation 的原因。

## Expectations

---

# Case 2: 精度调试的前置条件

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在进行 Ascend C 算子精度调试之前，需要满足哪些前置条件？为什么必须准备可复现的最小用例和黄金数据？需要检查哪些步骤才能确保可以开始调试？不需要执行任何工具调用。

## Expected Output

回复应说明精度调试的前置条件：环境就绪、算子工程可编译运行、精度比对工具可用、准备标杆数据（Golden Data）和最小用例。还应说明最小用例和黄金数据的重要性。

## Expectations

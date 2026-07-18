---
skill_name: ascendc-registry-invoke-to-direct-invoke
eval_mode: text
---
# Case 1: Kernel 模板转换的核心原则

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我需要将一个自定义算子项目从注册调用模式转换为 `<<<>>>` 直接调用模式。Kernel 模板的转换应遵循什么核心原则？算子实现代码需要修改吗？不需要执行任何工具调用。

## Expected Output

回复应说明转换的核心原则是保持 kernel 实现代码不变、只改造调用方式。算子计算逻辑完全不需要修改，只需移除注册框架代码并将调用方式改为直调语法。

---

# Case 2: 允许的修改内容

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在将自定义算子从注册调用模式转换为直接调用模式时，哪些修改是允许的？哪些是绝对不允许的？具体来说，include 包含、命名空间和注册框架代码分别应该如何处理？不需要执行任何工具调用。

## Expected Output

回复应说明转换过程中允许 include 替换、命名空间包裹和移除注册框架代码等修改。绝对禁止修改 kernel 计算逻辑。

---

# Case 4: 算子性能分析咨询（负向看护）

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

我的 Ascend C 算子性能不足，如何分析 profiling 数据找到瓶颈？

## Expected Output

回复应聚焦于 profiling 数据分析的通用方法和性能瓶颈定位的一般思路（如分析算子耗时分布、定位热点函数等），不涉及算子调用模式转换。


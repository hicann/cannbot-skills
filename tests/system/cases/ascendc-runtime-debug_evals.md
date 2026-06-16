---
skill_name: ascendc-runtime-debug
eval_mode: text
---
# Case 1: 运行时错误码范围与分类

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在 Ascend C 算子开发中遇到运行时错误时，如何根据错误码进行定位？常见的错误码有哪些范围分类，分别代表什么类型的错误？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 Ascend C 运行时错误码的主要范围分类：161xxx 表示设备侧 Kernel 执行错误，361xxx 表示 Host 侧应用层错误，561xxx 表示运行时环境错误。调试时应先根据百位数字判断错误层面再定位原因。

## Expectations

---

# Case 2: 507035 错误码的诊断

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 150000
- Max Tokens (glm-5): 140000
- Ascend Platform: A2

## Prompt

在算子运行时出现 507035 错误码是什么意思？可能的原因有哪些？我该如何诊断和修复这个问题？可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明 507035 错误码的含义、可能的原因以及诊断和修复方法。

## Expectations

---
skill_name: tilelang-review
---

# Case 1: 代码格式检查与修复流程

## Config
- Max Tokens: 100000
- Ascend Platform: A2

## Prompt

请介绍 tilelang-review 技能的核心工作流：代码格式检查和修复的顺序是什么？检查和修复为什么要分离？需要用到哪些工具？不需要执行任何工具调用。

## Expected Output

回复应说明代码格式检查和修复的工作流程：先运行检查工具生成报告，再询问用户是否执行修复，只有用户确认后才进行修复。

## Expectations
- [contains] ruff
- [contains] clang-format
- [contains] check-python.sh
- [contains] AskUserQuestion

---

# Case 2: 检查和修复分离原则

## Config
- Max Tokens: 80000
- Ascend Platform: A2

## Prompt

代码格式检查的核心原则是什么？为什么检查阶段不能自动修复？

## Expected Output

回复应说明核心原则是"检查和修复分离"：检查阶段只运行检查脚本生成报告，不修改任何文件。修复前必须用醒目方式询问用户，经用户明确同意后才执行修复。这是为了防止未授权的代码修改，保证一致性，让用户了解所有问题后再决定是否修复。AskUserQuestion 工具的使用必须严格遵循固定格式，不能私自添加额外选项。

## Expectations
- [contains] 检查和修复分离
- [contains] 必须等待用户确认
- [contains] 修复阶段
- [contains] 检查阶段

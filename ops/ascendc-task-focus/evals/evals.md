---
skill_name: ascendc-task-focus
---
# Case 1: 创建Add算子ST测试任务计划

## Config
- Eval Mode: file_based
- Max Tokens: 150000
- Max Tokens (deepseek-v4-flash): 180000
- Max Tokens (glm-5): 160000
- Ascend Platform: A2

## Prompt

我需要开发一个Add算子的ST测试，大概需要5个步骤：需求分析、API调研、方案设计、代码实现、编译测试。请帮我创建一个todo.md来管理这个任务。

## Expected Output

创建的todo.md文件应包含：# 任务标题（体现Add算子ST测试）、## 目标（1-2句话）、## 待办事项（5个步骤用- [ ]勾选框列出）、## 进度（0/5），内容结构完整

## Expectations

- [file_exists] todo.md
- [file_list] *.md

---

# Case 2: 不应创建任务计划的场景

## Config
- Max Tokens: 80000
- Max Tokens (deepseek-v4-flash): 100000
- Max Tokens (glm-5): 90000
- Ascend Platform: A2

## Prompt

我只是想快速查一下npu-smi info命令怎么用，不需要创建任务计划。

## Expected Output

不创建todo.md，直接简洁回答npu-smi info的用法。因为这是快速信息查询（<3步），不符合ascendc-task-focus的使用条件

## Expectations

---

# Case 3: 创建精度调试任务计划

## Config
- Max Tokens: 200000
- Max Tokens (deepseek-v4-flash): 240000
- Max Tokens (glm-5): 220000
- Ascend Platform: A2

## Prompt

Softmax算子FP16精度验证失败了，误差2.3e-2，需要定位并解决精度问题。请帮我创建一个精度调试的todo.md，包含误差记录表格和调试计数，并在回复中展示完整内容。

## Expected Output

在回复中展示完整的精度调试模式todo.md内容，包含：## 调试计数（0/7）、## 待办事项（误差分析→Printf调试→常见陷阱排查→二分调试→实施修复→验证修复的勾选框列表）、## 误差记录表格（含初始误差2.3e-2）、## 进度（0/N），用户无需打开文件即可看到完整结构

## Expectations

---

# Case 4: 创建分阶段开发任务计划

## Config
- Max Tokens: 250000
- Max Tokens (deepseek-v4-flash): 300000
- Max Tokens (glm-5): 275000
- Eval Mode: file_based
- Ascend Platform: A2

## Prompt

我要从零开发SoftmaxV5算子，涉及6个阶段：需求分析、API调研、方案设计、代码实现、测试验证、文档完善，每个阶段有多个子任务。请帮我创建一个分阶段的todo.md并在回复中展示完整内容。

## Expected Output

在回复中展示完整的分阶段todo.md内容：按阶段1~6分别列出子任务，每个阶段有独立的进度统计（如阶段1: 0/3），## 进度区域显示总体进度，步骤用- [ ]勾选框格式，用户可直接看到完整的阶段划分和任务安排

## Expectations


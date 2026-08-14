---
description: Ascend C 算子代码检视团队
mode: primary
skills:
  - ascendc-code-review
permission:
  external_directory: allow
---

# CANNBot Code Review Team

## 执行方式

1. 加载 `ascendc-code-review` skill
2. 按 skill 的 SKILL.md 路由到对应 workflow
3. 严格按 workflow 定义的完整工作流执行
4. skill 内的 steps 会自动探测运行环境：
   - 子 Agent 可用 → 并行加速
   - 子 Agent 不可用 → 串行执行

## 注意事项

- 流程定义以 skill（workflows/ + steps/ + references/）为准
- 检视子 agent 分两类：通用检视子 agent（走假设检验，输出 PASS/FAIL/SUSPICIOUS）和专项检视子 agent（不走假设检验，如 design-check、代码风格 STYLE）；另有预研子 agent（code-summarizer 等，检视前预研，不产出检视意见）。详细逻辑在对应 step 文件中
- 禁止跳过行号校对
- 报告统一撰写，子 Agent 禁止生成报告文件

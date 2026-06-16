---
skill_name: catlass-op-design
eval_mode: text
---
# Case 1: CATLASS 算子设计组件选型说明

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在设计中设计一个 CATLASS 算子时，需要选择哪些组件？请逐一列出并简要说明每个组件的作用。不需要执行任何工具调用。

## Expected Output

回复应说明 CATLASS 算子设计需要选择的七大组件：ArchTag（芯片架构标识）、DispatchPolicy（流水调度方式）、TileShape（包括 L1/L0 的 tile 形状）、BlockMmad（矩阵乘计算块）、BlockEpilogue（后处理块）、BlockScheduler（任务块调度策略）和 Kernel 类型（算子内核模板）。应逐一解释各组件在算子组装链中的角色。

## Expectations

---

# Case 2: CATLASS 设计前预备知识阅读要求

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

在开始 CATLASS 算子组件选型之前，必须先阅读哪些资料？请说明必须完成的预备知识阅读步骤。可以加载技能，不需要执行其他外部工具调用。

## Expected Output

回复应说明必须先阅读工作区中 `./catlass/` 目录下的三部分内容：README.md（了解库定位和整体架构）、docs/ 目录（理解分层设计与选型依据）、以及 examples/ 下与目标算子最接近的样例目录（提炼已验证的组件组合与实现模式）。必须强调未完成上述阅读禁止进入选型。

## Expectations

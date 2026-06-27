---
team_name: pypto-op-orchestrator
eval_mode: text
---

# Case 1: PyPTO 算子开发编排流程

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: pypto-op-design;pypto-op-develop;pypto-intent-understand;pypto-api-explore

## Prompt

我想用 PyPTO 开发一个算子，pypto-op-orchestrator 团队是怎么工作的？完整的开发流程是什么样的？

## Expected Output

回复应覆盖以下要点：
1. PyPTO 算子端到端开发编排 Agent，作为唯一流程 owner，负责 7 阶段状态机管理
2. 核心流程包含需求理解 → API 探索 → 设计 → 开发 → 测试等多个阶段
3. 调度三个 Subagent（分析、开发、性能调优）分阶段执行
4. 支持状态持久化、中断恢复和重试限制

## Expectations

- [contains] pypto

---

# Case 2: PyPTO 适用的算子类型

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

什么类型的算子适合用 PyPTO 来开发？什么样的算子不适合？我想了解一下适用范围。

## Expected Output

回复应说明 PyPTO 适用于使用 PyPTO DSL 在昇腾 NPU 上开发自定义算子，不适用于非 PyPTO 生态的开发方式。在选择开发方式时应考虑算子的复杂度、性能需求和 PyPTO 框架的支持范围

## Expectations

---

# Case 3: 启动编排前需要的信息

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想用 PyPTO 开发一个算子，先帮我看看需要准备哪些信息，先别开始开发。

## Expected Output

回复应在开始开发前主动确认必要信息：算子功能描述和数学定义、输入输出规格、目标架构、性能需求等，而不是在缺少这些信息的情况下直接开始编排

## Expectations


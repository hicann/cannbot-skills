---
team_name: tilelang-op-orchestrator
eval_mode: text
---

# Case 1: TileLang-Ascend 算子开发流程

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2
- Distractor skills: tilelang-op-design;tilelang-op-developer;tilelang-programming-model-guide;tilelang-api-best-practices

## Prompt

我想用 TileLang 在昇腾 NPU 上开发算子，tilelang-op-orchestrator 团队是怎么组织的？开发流程是怎样的？

## Expected Output

回复应覆盖以下要点：
1. TileLang-Ascend 算子端到端开发编排 Agent，负责 3 阶段状态机管理
2. 流程包含：设计 → 开发 → 审查，支持设计回退和失败恢复
3. 调度三个 Subagent 分阶段执行，支持状态持久化和检查点恢复
4. 支持 Developer 模式（自动化）和 Expert 模式（手动控制）两种编程范式

## Expectations

- [contains] tilelang

---

# Case 2: Developer 模式与 Expert 模式的选择

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

TileLang 的 Developer 模式和 Expert 模式有什么区别？我该在什么场景下用哪种模式？

## Expected Output

回复应说明两种模式的区别和选择依据：Developer 模式自动化程度高，适合标准场景；Expert 模式提供手动控制能力，适合需要精细调优的场景。应根据算子复杂度和性能需求选择合适的编程模式

## Expectations

---

# Case 3: 开发前需要的信息

## Config
- Max Tokens: 300000
- Timeout: 900
- Ascend Platform: A2

## Prompt

我想用 TileLang 开发一个算子，先告诉我需要提供哪些信息，不需要现在就开始开发。

## Expected Output

回复应在开发前主动确认必要信息：算子功能描述和数学定义、输入输出规格、目标芯片型号、性能需求等，而不是在缺少这些信息的情况下直接开始开发流程

## Expectations


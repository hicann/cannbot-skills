---
skill_name: ascendc-perf-optimize
eval_mode: text
---
# Case 1: 性能优化四步流程

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

请介绍 ascendc-perf-optimize 技能中 Ascend C 算子的性能优化工作流程是什么？需要经过哪些步骤？每一步分别关注什么？不需要执行任何工具调用。

## Expected Output

回复应详细描述性能优化的 4 步流程：步骤 1 Tiling 建模（确定最优 Tiling 参数），步骤 2 流水分析（结合仿真图和 profiling 数据做三阶段诊断），步骤 3 性能优化策略制定（根据 bound 类型选择优化手段），步骤 4 Tiling 参数回修。应使用中文编号。

## Expectations

---

# Case 2: 性能优化所需输入

## Config
- Max Tokens: 100000
- Max Tokens (deepseek-v4-flash): 120000
- Max Tokens (glm-5): 110000
- Ascend Platform: A2

## Prompt

进行 Ascend C 算子性能优化需要提供哪些输入信息？需要准备哪些数据和材料？不需要执行任何工具调用。

## Expected Output

回复应说明性能优化所需的输入，包括算子代码、Tiling 参数与建模数据、流水仿真图、Profiling 数据、硬件配置信息等。这些输入是进行性能瓶颈分析和优化方案设计的基础。

## Expectations

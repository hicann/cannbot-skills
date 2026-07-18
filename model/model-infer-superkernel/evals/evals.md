---
skill_name: model-infer-superkernel
eval_mode: text
---

# Case 1: SuperKernel 适用约束

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

SuperKernel 在昇腾 NPU 上启用有哪些前提约束？硬件、执行模式、推理阶段分别有什么要求？

## Expected Output

回复应说明 SuperKernel 仅支持 Atlas A3 硬件、仅 ge_graph 执行模式、仅在 decode 阶段生效

## Expectations

- [contains] ge_graph
- [contains] A3
- [skill_activated] model-infer-superkernel

# Case 2: superkernel_scope 标记方式

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

启用 SuperKernel 时 superkernel_scope 该加在哪里、怎么用？

## Expected Output

回复应说明导入上下文管理器，在 decode 方法中用 superkernel_scope 标记融合范围

## Expectations

- [skill_activated] model-infer-superkernel

# Case 3: 二进制融合收益机制

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

SuperKernel 通过算子二进制融合减少调度开销，为什么只在 decode 阶段有收益？说明原因和机制。

## Expected Output

回复应说明 decode 单步算子多、调度开销占比高，二进制融合减少任务下发开销；Prefill 计算量大、调度占比低收益有限

## Expectations

- [skill_activated] model-infer-superkernel

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我给模型启用 SuperKernel。

## Expected Output

回复应先确认启用前提条件再动手，而不是缺信息直接改造

## Expectations

- [skill_activated] model-infer-superkernel

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-graph-mode;model-infer-fusion;model-infer-prefetch

## Prompt

A3 上 decode 调度开销太大，想用算子二进制融合减少任务下发，往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-superkernel，给出 SuperKernel 二进制融合方向，即使存在图模式、融合算子、预取等相似 skill 也应选 SuperKernel 专项

## Expectations

- [skill_activated] model-infer-superkernel

# Case 6: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我在 Atlas A2 上用 eager 模式跑推理，想启用 SuperKernel，可以吗？

## Expected Output

回复应说明 SuperKernel 仅支持 A3 + ge_graph + decode，A2 和 eager 模式不满足条件，应建议用户先满足硬件和执行模式前提

## Expectations

- [skill_activated] model-infer-superkernel


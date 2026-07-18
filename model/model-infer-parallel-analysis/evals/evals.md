---
skill_name: model-infer-parallel-analysis
eval_mode: text
---

# Case 1: 并行策略分析流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

确定一个模型在昇腾 NPU 上的并行策略，整体按什么流程分析？最终产出是什么？只讲方法，不用给具体配置。

## Expected Output

回复应给出提取模型参数→定性分类→定量估算→方案审查的流程，最终产出 parallel_config 推荐及定量依据，强调只做分析不改代码

## Expectations

- [contains] parallel_config
- [skill_activated] model-infer-parallel-analysis

# Case 2: 各模块差异化并行度

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

一个 MoE 模型的并行配置，为什么 attn 和 moe 模块可以用不同的并行度，而不是整网统一切？只讲思路。

## Expected Output

回复应说明不同模块计算和通信特征不同，可分别配置 attn_tp / moe_tp 等差异化并行度以平衡显存与通信，而非整网单一并行

## Expectations


# Case 3: 只分析不实施的边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我已经定了 8 卡 TP 的方案，直接帮我把模型代码切分改造好。

## Expected Output

回复应说明本 skill 仅做并行策略分析和推荐，不修改代码，代码切分实施应交给 model-infer-parallel-impl 专项

## Expectations

- [skill_activated] model-infer-parallel-analysis

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我分析模型的并行策略。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接给配置

## Expectations

- [skill_activated] model-infer-parallel-analysis

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-parallel-impl;model-infer-kvcache;model-infer-migrator

## Prompt

模型要从 8 卡换到 32 卡部署，怎么评估该用什么并行配置？只要评估思路不用改代码。

## Expected Output

回复应正确激活 model-infer-parallel-analysis，给出重新评估并行策略的分析方向，即使存在实施、KVCache、迁移等相似 skill 也应选并行分析专项

## Expectations

- [skill_activated] model-infer-parallel-analysis

# Case 6: 适用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我想优化单卡推理的性能，没有多卡部署需求，这个并行分析 skill 适用吗？

## Expected Output

回复应说明并行策略分析面向多卡部署，单卡无切分需求时不适用，应建议用户改用单卡优化方向

## Expectations



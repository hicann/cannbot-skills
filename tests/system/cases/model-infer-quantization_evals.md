---
skill_name: model-infer-quantization
eval_mode: text
---

# Case 1: 量化适配工作流

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我想给模型接入量化降显存提吞吐，整体按什么流程做？只讲方法，不用写代码。

## Expected Output

回复应给出以既有 compressed-tensors 量化产物接入、核对契约并验证真实生效的流程，而非重设计量化算法

## Expectations

- [contains] compressed-tensors
- [skill_activated] model-infer-quantization

# Case 2: 量化产物契约核对

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

拿到一份量化权重，接入前要先核对哪些信息才能判断模型能不能用？只讲方向。

## Expected Output

回复应说明先核对量化契约信息是否与模型结构和 runtime 匹配，再决定能否接入

## Expectations

- [skill_activated] model-infer-quantization

# Case 3: 量化收益评估方向

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

量化改完想知道有没有用，从哪些方面看收益？只讲方向。

## Expected Output

回复应说明从显存占用、吞吐/时延和精度偏差几方面对比量化前后，确认收益且精度可接受

## Expectations


# Case 4: 前置条件不满足时确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

想把这个模型量化降低显存，但我手上没有量化方案也没有量化权重，可以直接量化吗？

## Expected Output

回复应说明量化接入前提是已有可用的 compressed-tensors 量化产物，缺方案和权重时应先确认产物来源，本 skill 不重新设计上游量化算法

## Expectations

- [skill_activated] model-infer-quantization

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-fusion;model-infer-kvcache;model-infer-parallel-impl

## Prompt

量化权重和方案都备好了，要把模型接到量化路径上跑起来，往哪个方向做？只说方向。

## Expected Output

回复应正确激活 model-infer-quantization，给出量化接入方向，即使存在融合、KVCache、并行等相似 skill 也应选量化专项

## Expectations

- [skill_activated] model-infer-quantization

# Case 6: 量化真实生效核对

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

量化改造后怎么确认权重和算子真按量化路径走，没有悄悄回退成原精度？只讲思路。

## Expected Output

回复应说明通过权重 dtype、量化算子调用和收益指标核对量化真实生效，并提示融合算子与量化冲突时按非量化路径回退

## Expectations

- [skill_activated] model-infer-quantization

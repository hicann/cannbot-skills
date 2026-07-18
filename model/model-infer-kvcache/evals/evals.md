---
skill_name: model-infer-kvcache
eval_mode: text
---

# Case 1: KVCache 三种模式机制概述

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

昇腾 NPU 推理里 KVCache 有连续缓存、分页注意力（Paged Attention）、MLA 压缩三种模式。请分别说明每种模式怎么存 KV、机制上的区别，不用判断哪种最好。

## Expected Output

回复应分别讲清三种模式的存储机制：连续缓存按序列连续存储；分页注意力用物理块存储、靠 block_table 映射；MLA 缓存压缩维度。三者机制区别说明即可，不要求给出选型结论

## Expectations

- [skill_activated] model-infer-kvcache

# Case 2: Paged Attention 的 block_table 与 slot_mapping 构造

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

分页注意力里 block_table 和 slot_mapping 分别怎么构造？slot_mapping 为什么每个 decode 步都要重算？

## Expected Output

回复应说明 block_table 是逻辑块到物理块的映射、推理全程不变；slot_mapping 决定新 token 写入的物理位置，随 kv_len 变化每步重算

## Expectations

- [contains] kv_len
- [contains] block_table
- [skill_activated] model-infer-kvcache

# Case 3: MLA 压缩缓存省显存原理

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

MLA 压缩缓存为什么比完整 KV 省显存？absorb 路径会怎么影响 FA 算子的入参？

## Expected Output

回复应说明 MLA 只缓存压缩维度，远小于完整 KV；absorb 把 V 投影吸收进 O 投影，key 和 value 传同一份压缩缓存

## Expectations

- [contains] absorb
- [skill_activated] model-infer-kvcache

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我给模型接入 KVCache。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接给改造代码

## Expectations

- [skill_activated] model-infer-kvcache

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-fusion;model-infer-parallel-analysis;model-infer-graph-mode

## Prompt

我的模型 KV 缓存占显存太大，想做分页管理降低占用，该往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-kvcache，给出分页注意力等缓存优化方向，即使存在融合算子、并行、图模式等相似 skill 也应选 KVCache 专项

## Expectations

- [skill_activated] model-infer-kvcache

# Case 6: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我想优化训练阶段的显存占用，KVCache 这个 skill 能帮我吗？

## Expected Output

回复应说明本 skill 仅覆盖推理阶段的 KVCache 优化，不适用于训练显存优化，应建议用户改用训练侧方案

## Expectations



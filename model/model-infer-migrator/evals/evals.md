---
skill_name: model-infer-migrator
eval_mode: text
---

# Case 1: 模型框架适配流程与产出

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

把一个 HF 模型适配到昇腾 NPU 推理框架，整体按什么流程做？最终产出哪些标准文件？只讲流程不用写代码。

## Expected Output

回复应给出代码准备→权重管理→跑通验证→基线采集的流程，产出符合仓库规范的 Runner、infer.py、infer.sh、YAML 配置等标准目录文件

## Expectations

- [skill_activated] model-infer-migrator

# Case 2: 基线采集产物

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

适配跑通后采集性能基线，需要落地什么基线产物供后续优化对比？只讲方向。

## Expected Output

回复应说明采集标准化性能基线并落地 baseline_metadata.json，记录 Prefill/Decode 耗时等指标供后续优化对比

## Expectations

- [skill_activated] model-infer-migrator

# Case 3: 不同迁移场景的处理差异

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

新模型从 HF 适配、已有模型跑不通修复、仅采集基线，这三种情况入手点有什么不同？只讲思路。

## Expected Output

回复应区分三类场景：从 HF/本地代码做完整框架适配、修复已有适配跑不通问题、对已可运行模型直接采集基线，按起点不同选择对应流程

## Expectations

- [skill_activated] model-infer-migrator

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我适配一个模型到昇腾 NPU。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接开始适配

## Expectations

- [skill_activated] model-infer-migrator

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-parallel-analysis;model-infer-kvcache;model-infer-fusion

## Prompt

一个新模型还没在 NPU 上跑通，我想先把它适配进框架并跑出基线，往哪个方向做？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-migrator，给出框架适配 + 基线采集方向，即使存在并行、KVCache、融合等相似 skill 也应选迁移基线专项

## Expectations

- [skill_activated] model-infer-migrator

# Case 6: 适用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

模型已经跑通也有基线了，我想提升 Decode 性能，这个迁移 skill 该做吗？

## Expected Output

回复应说明本 skill 负责适配跑通和基线建立，性能提升应交给融合算子、KVCache、图模式等优化专项，不在本 skill 范围

## Expectations



---
skill_name: model-infer-parallel-impl
eval_mode: text
---

# Case 1: 并行切分实施流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

已经定好并行方案，要把模型代码改成多卡并行，整体按什么步骤实施？只讲流程不用写代码。

## Expected Output

回复应给出按 parallel_config 创建通信组→逐模块并行层替换→Embedding/LMHead 并行→YAML 配置→权重处理→验证的实施流程，强调前置依赖已确认的 parallel_config

## Expectations

- [contains] parallel_config
- [skill_activated] model-infer-parallel-impl

# Case 2: 并行线性层替换方式

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

TP 切分时模型里的线性层要怎么替换成并行版本？列切和行切分别用什么？只讲思路。

## Expected Output

回复应说明按列切用 ColumnParallelLinear、按行切用 RowParallelLinear 替换，按模块通信特征选择切分方向

## Expectations

- [contains] ColumnParallelLinear
- [skill_activated] model-infer-parallel-impl

# Case 3: 权重处理

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

并行切分改完代码后，权重还需要做什么处理才能加载？只讲方向。

## Expected Output

回复应说明权重需按并行配置做切分转换，或启用在线切分，使各 rank 加载对应分片

## Expectations


# Case 4: 前置条件不满足时确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我把模型直接改成 8 卡并行。

## Expected Output

回复应说明实施前提是 parallel_config 已由并行策略分析确定并确认，若未确定应先用 model-infer-parallel-analysis 定方案，而不是直接开切

## Expectations

- [skill_activated] model-infer-parallel-impl

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-parallel-analysis;model-infer-kvcache;model-infer-migrator

## Prompt

并行配置已经确认好了，现在要把代码切分、建通信组、改 YAML，往哪个方向做？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-parallel-impl，给出并行化代码改造方向，即使存在分析、KVCache、迁移等相似 skill 也应选并行实施专项

## Expectations

- [skill_activated] model-infer-parallel-impl

# Case 6: 通信组创建与层替换次序

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

多卡并行实施时，通信组创建和并行层替换的先后关系是什么？为什么不能先替换层再建通信组？只讲思路。

## Expected Output

回复应说明先按 parallel_config 创建各模块通信组，并行层替换依赖通信组做集合通信，故通信组必须先建，否则层替换无组可用

## Expectations

- [contains] parallel_config


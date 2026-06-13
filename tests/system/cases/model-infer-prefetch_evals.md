---
skill_name: model-infer-prefetch
eval_mode: text
---

# Case 1: 权重预取优化流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

想给模型加权重预取提速，整体按什么步骤做？只讲方法不用写代码。

## Expected Output

回复应给出确定预取位置→计算预取大小→代码实现→性能验证的流程，强调先靠 profiling 找到 memory-bound 热点算子再下手

## Expectations

- [skill_activated] model-infer-prefetch

# Case 2: 适用前提判断

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

什么样的算子瓶颈才适合用权重预取来优化？怎么从 profiling 判断？

## Expected Output

回复应说明预取适合 memory-bound 的 MatMul/QBMM/GMM 等权重搬运热点，compute-bound 算子加预取无收益，需先看 profiling 确认搬运占比

## Expectations

- [contains] memory-bound
- [skill_activated] model-infer-prefetch

# Case 3: 预取位置与依赖窗口

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

预取调用插在哪、依赖节点怎么选才能真正重叠搬运？只讲方向。

## Expected Output

回复应说明把预取插在目标算子前、用其直接前驱作依赖节点，留足搬运窗口让权重预取与前驱计算重叠，依赖不在同路径需另处理

## Expectations

- [skill_activated] model-infer-prefetch

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我给模型加预取。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接插预取代码

## Expectations

- [skill_activated] model-infer-prefetch

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-multi-stream;model-infer-fusion;model-infer-graph-mode

## Prompt

profiling 显示 MatMul 是 memory-bound 热点，想把权重提前搬到片上和计算重叠，往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-prefetch，给出权重预取方向，即使存在多流、融合、图模式等相似 skill 也应选预取专项

## Expectations

- [skill_activated] model-infer-prefetch

# Case 6: 预取大小确定

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

预取大小该怎么定，给太大或太小分别有什么问题？只讲思路。

## Expected Output

回复应说明预取大小按目标权重张量大小估算、受片上空间约束，太大挤占搬运/超容反伤、太小盖不住搬运不够重叠，需结合权重维度和数据类型字节数计算

## Expectations

- [skill_activated] model-infer-prefetch


---
skill_name: model-infer-multi-stream
eval_mode: text
---

# Case 1: 多流整网优化流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

想给模型做多流整网优化，整体按什么步骤推进？只讲方法不用写代码。

## Expected Output

回复应给出先做整网模块 DAG 与模块间并行性分析、再做模块算子级分析、最后开发调试与验收的流程，先识别可并行模块再做 stream overlap

## Expectations

- [contains] DAG
- [skill_activated] model-infer-multi-stream

# Case 2: eager 与 TorchAir 多流路径定界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

模型是 eager 模式和 TorchAir 图模式时，多流改造的方式不同吗？怎么定哪条路径？只讲思路。

## Expected Output

回复应说明 eager 与 TorchAir 多流改造接口和做法不同，需按模型当前执行模式选择对应多流路径，不能混用

## Expectations

- [skill_activated] model-infer-multi-stream

# Case 3: 控核分配

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

计算和通信放双流后互相抢 NPU 核导致没收益，怎么按比例分核避免抢占？只讲方向。

## Expected Output

回复应说明通过控核用 limit_core_num 给计算和通信流分配核数，避免两条流抢同一批核，平衡后才有 overlap 收益

## Expectations

- [contains] limit_core_num
- [skill_activated] model-infer-multi-stream

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我给模型加多流优化。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接改造

## Expectations

- [skill_activated] model-infer-multi-stream

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-prefetch;model-infer-graph-mode;model-infer-fusion

## Prompt

模型 decode 时计算和通信串行执行很慢，想让两者重叠起来提速，往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-multi-stream，给出多流 overlap 重叠计算与通信的方向，即使存在预取、图模式、融合等相似 skill 也应选多流专项

## Expectations

- [skill_activated] model-infer-multi-stream

# Case 6: 适用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我只想把一个手写小算子换成更快的融合算子，不涉及整网，这个多流 skill 适用吗？

## Expected Output

回复应说明本 skill 面向整网模块级多流并行，单个算子替换不在范围，应建议用户改用融合算子等专项

## Expectations



---
skill_name: model-infer-graph-mode
eval_mode: text
---

# Case 1: 两种图模式后端机制

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

昇腾 NPU 上有 npugraph_ex 和 GE 两种图模式后端。请说明两者在适配方式上的区别，不用判断哪种更好。

## Expected Output

回复应分别说明 npugraph_ex 和 GE 两种图模式后端的适配差异，机制区别讲清即可，不要求给选型结论

## Expectations

- [contains] npugraph_ex
- [skill_activated] model-infer-graph-mode

# Case 2: 图模式仅 Decode 约束

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

LLM 模型适配图模式时，为什么只对 Decode 阶段启用，Prefill 阶段保持 eager？只回答原因即可。

## Expected Output

回复应说明 Decode 阶段 shape 固定可图编译加速，Prefill 阶段序列动态、图模式收益低甚至不可用，故 Prefill 保持 eager

## Expectations

- [skill_activated] model-infer-graph-mode

# Case 3: 动态 shape 与 mark_static

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

Decode 阶段 actual_seq_lengths 每步变化时，dynamic 和 mark_static 该怎么配置？

## Expected Output

回复应说明配置 dynamic=True、用 mark_static 标记除 actual_seq_lengths 外的静态输入，actual_seq_lengths 保持动态

## Expectations

- [contains] mark_static
- [skill_activated] model-infer-graph-mode

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

帮我把模型适配成图模式。

## Expected Output

回复应先确认必要信息再动手，而不是缺信息直接改造

## Expectations

- [skill_activated] model-infer-graph-mode

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-fusion;model-infer-kvcache;model-infer-prefetch

## Prompt

我的模型 Decode 单步太慢，想用 torch.compile 编译整图减少调度开销，往哪个方向改？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-graph-mode，给出图模式适配方向，即使存在融合算子、KVCache、预取等相似 skill 也应选图模式专项

## Expectations

- [skill_activated] model-infer-graph-mode

# Case 6: 图中断定位与修复

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

模型 compile 后图编译反复 graph break 性能上不去，怎么定位是哪里中断、一般往哪些方向修？只讲思路。

## Expected Output

回复应说明定位图中断点（dynamic shape、数据依赖控制流、不支持算子等）的方法，并给出消除中断或缩小动态范围的修复方向

## Expectations

- [contains] graph break
- [skill_activated] model-infer-graph-mode


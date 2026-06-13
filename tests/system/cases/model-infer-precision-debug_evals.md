---
skill_name: model-infer-precision-debug
eval_mode: text
---

# Case 1: 精度问题排查流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

KVCache/FA 改造后精度对不上，整体按什么流程定位？只讲方法不用写代码。

## Expected Output

回复应给出快速诊断→分模块定位→逐层精细对比→匹配常见陷阱的流程，先缩小到出问题的模块再逐层二分对比

## Expectations

- [skill_activated] model-infer-precision-debug

# Case 2: Prefill 与 Decode 精度不一致定位

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

模型 Prefill 输出正常但 Decode 越来越偏，这种 Prefill 和 Decode 不一致一般往哪个方向查？只讲思路。

## Expected Output

回复应指向 Decode 阶段缓存更新/读取、slot 写入位置、FA 入参与 Prefill 分支差异，分阶段隔离定位

## Expectations

- [contains] kv_len
- [skill_activated] model-infer-precision-debug

# Case 3: 缓存写入位置错误排查

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

怀疑 KV 写到了错误的缓存位置导致精度异常，怎么排查写入位置是否正确？只讲方向。

## Expected Output

回复应说明检查 slot_mapping 计算和 block_table 映射是否正确，确认每步新 token 写入的物理位置与预期一致

## Expectations

- [contains] slot_mapping
- [skill_activated] model-infer-precision-debug

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我的模型精度不对，帮我调。

## Expected Output

回复应先确认必要信息再动手，而不是缺现象直接给修复

## Expectations

- [skill_activated] model-infer-precision-debug

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-kvcache;model-infer-fusion;model-infer-runtime-debug

## Prompt

KVCache 改成分页注意力后，输出和基线对不上但不报错，往哪个方向排查？只说方向不用写代码。

## Expected Output

回复应正确激活 model-infer-precision-debug，给出精度对齐排查方向，即使存在 KVCache、融合、运行时调试等相似 skill 也应选精度诊断专项

## Expectations

- [skill_activated] model-infer-precision-debug

# Case 6: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我的模型推理直接 crash 报 aicore timeout，帮我用精度调试 skill 修一下。

## Expected Output

回复应说明本 skill 只诊断精度偏差，crash/timeout/OOM 等运行时错误不在范围，应建议改用 model-infer-runtime-debug

## Expectations

- [skill_activated] model-infer-precision-debug


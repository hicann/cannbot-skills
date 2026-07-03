---
skill_name: model-train-accuracy-debug
eval_mode: text
---

# Case 1: 精度异常整体定界流程

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

换了个 NPU 自定义算子后训练 loss 和基线对不上，但有可对照的基线环境，整体按什么流程定位？只讲方法不用写代码。

## Expected Output

回复应给出 先开确定性计算复现 → 代码审查高风险改动 → 确认异常特征 → msprobe dump/compare 定界 的流程，并强调先固定随机性排除非确定性因素

## Expectations

- [skill_activated] model-train-accuracy-debug

# Case 2: 确定性优先

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我切了分支后 loss 偏了，想直接用 msprobe dump 对比，应该先做什么？只讲思路。

## Expected Output

回复应指出在任何 dump/compare 前必须先开启确定性计算（固定 seed + deterministic）并在基线与异常环境复现一次，若开启后问题消失则判为随机性问题不再深挖

## Expectations

- [contains] msprobe
- [skill_activated] model-train-accuracy-debug

# Case 3: NaN 定位方向

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

训练中途出现 NaN，应该往哪个方向查？只讲方向不用写代码。

## Expected Output

回复应先判定 NaN 是否出现在前向 logits，前向已 NaN 走 msprobe overflow_check，前向正常而反向出 NaN 再用 detect_anomaly 定位首个异常算子

## Expectations

- [contains] detect_anomaly
- [skill_activated] model-train-accuracy-debug

# Case 4: 信息不足时主动确认

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我的模型训练精度不对，帮我调。

## Expected Output

回复应先确认基线环境、异常环境、变更列表、确定性设置等必要信息再动手，而不是缺基线对照就直接给修复

## Expectations

- [skill_activated] model-train-accuracy-debug

# Case 5: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-train-oom-analysis;model-infer-precision-debug;model-train-log-visualization

## Prompt

切换并行策略后训练 loss 曲线偏离基线，有基线可对照，往哪个方向排查？只说方向不用写代码。

## Expected Output

回复应正确激活 model-train-accuracy-debug，给出有标杆的训练精度定界方向，即使存在 OOM 分析、推理精度调试等相似 skill 也应选训练精度诊断专项

## Expectations

- [skill_activated] model-train-accuracy-debug

# Case 6: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我没有任何基线，就是觉得训练效果差，帮我用精度调试 skill 定位。

## Expected Output

回复应说明本 skill 依赖可对照的基线（标杆数据/环境），无基线的精度问题不在适用范围，建议先建立基线或改用其他手段

## Expectations

- [skill_activated] model-train-accuracy-debug

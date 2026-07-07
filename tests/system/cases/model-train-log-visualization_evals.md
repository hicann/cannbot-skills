---
skill_name: model-train-log-visualization
eval_mode: text
---

# Case 1: 单日志画曲线

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我有一份 NPU 训练的 stdout 日志，想画 loss 和 grad_norm 曲线，怎么做？

## Expected Output

回复应说明用本技能的绘图脚本按 step 解析日志并绘制 loss/grad_norm 曲线，并先确认主日志路径等输入

## Expectations

- [skill_activated] model-train-log-visualization

# Case 2: 双日志对比

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我有正常和异常两份训练日志，想画在同一张图上对比 loss，还想看误差曲线。

## Expected Output

回复应说明双日志对比会绘制 loss/grad_norm 及 loss 的绝对/相对误差曲线，并提示无共同 step 时脚本会报错退出避免误导

## Expectations

- [contains] grad_norm
- [skill_activated] model-train-log-visualization

# Case 3: 追加可选指标

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

画训练日志曲线时，我还想顺便看 memory、tps、mfu 这些指标，能加吗？

## Expected Output

回复应说明可通过 --metrics 追加 memory/tps/mfu 等可选指标，并提示指标不存在时会跳过并告警

## Expectations

- [skill_activated] model-train-log-visualization

# Case 4: 正向看护-多 skill 环境下正确触发

## Config
- Max Tokens: 150000
- Timeout: 900
- Distractor skills: model-infer-runtime-debug;model-infer-precision-debug;model-infer-superkernel

## Prompt

帮我把两份训练日志的 loss 曲线画出来对比一下。

## Expected Output

回复应正确激活 model-train-log-visualization 进行日志解析与绘图，即使存在 OOM 分析、精度调试等相似 skill 也应选日志可视化专项

## Expectations

- [skill_activated] model-train-log-visualization

# Case 5: 使用边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我的训练 loss 偏了，帮我用日志可视化 skill 找出精度问题的根因。

## Expected Output

回复应说明本技能只负责从日志解析并绘制曲线，不做精度根因定位，定位根因应改用对应的精度调试 skill（可先用本技能画曲线辅助观察）

## Expectations

- [skill_activated] model-train-log-visualization

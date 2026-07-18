---
team_name: model-infer-sota-approach
eval_mode: text
---

# Case 1: baseline 之上的探索式优化编排

## Config
- Max Tokens: 200000
- Timeout: 900
- Distractor skills: model-infer-multi-stream;model-infer-fusion;model-infer-profiling;model-infer-perf-breakdown

## Prompt

我有一个已经能跑通、精度可复现的 PyTorch NPU 推理 baseline，想在它之上继续做多方向的性能优化。整体流程怎么编排？先给方案不用动手。

## Expected Output

回复应正确激活 model-infer-sota-approach 团队，给出 baseline 之上、profiling 驱动的编排：确认场景与目标 → 跑通精度基线 → 采集 baseline profiling → 分析 → 多来源并行发现候选 → 初始化 Plan Dashboard → Plan 实施/review/派生循环 → 最终验收；强调 Plan/round 自循环、性能以 perf-breakdown 报告的 Δ% 为准、只编排不替工、具体优化下沉调用单点 skill。

## Expectations

- [contains] profiling

# Case 2: 与 model-infer-optimize 的分工边界

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

我的模型还没适配进推理框架、也没有 baseline，能直接用 model-infer-sota-approach 做优化吗？只回答适不适用并说明原因。

## Expected Output

回复应说明 model-infer-sota-approach 的前置是"已有可运行、可复现精度的 baseline"，没有 baseline 时不适用；应先用 model-infer-optimize（从零到 baseline 的基础工作流）或 model-infer-migrator 建立 baseline，再回到本流程。两者以 baseline 为界、前后衔接、互补不替代。

## Expectations

- [contains] baseline

# Case 3: 性能收益的判据

## Config
- Max Tokens: 150000
- Timeout: 900

## Prompt

在这个优化编排里，怎么判断某一轮改动到底有没有带来性能收益？只讲判据。

## Expected Output

回复应说明性能收益一律以 profile-analyzer / model-infer-perf-breakdown 的分析报告（时间分布、与 baseline 的 Δ%）为判据，不以裸 wall-clock 计时下结论；必要时按同一场景、同一口径重采 profiling 再与 baseline 对照。

## Expectations

- [contains] Δ

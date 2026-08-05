---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "INF/-INF/NAN 输出通过标准:按 INF_NAN_MODE 与三方比对判定"
description: "inf/nan 输出无法算误差,须比对第三方/CPU 并结合公式;是否比较取决于芯片与 INF_NAN_MODE_ENABLE;Golden 为 inf/nan 时 NPU 须与之完全一致。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#5.3.3 inf/-inf/nan 通过标准
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, inf-nan, mode-enable]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
输出值为 inf/-inf/nan 时无法计算相对/绝对误差,须比对第三方芯片/CPU 的输出并结合计算公式判断:
- Ascend910A 及之前的芯片,或 910B 及之后配置 INF_NAN_MODE_ENABLE=0(未开 inf 模式):inf 的计算值不参与精度比较。
- Ascend910B 及之后配置 INF_NAN_MODE_ENABLE=1(开启 inf 模式):要求 inf 输出情况一致,或与更高精度 Golden 相比 NPU 更接近 Golden。
- 正确条件(满足任一):不论 Golden 为何值,NPU 值与标杆值一致(nan、inf/-inf);若 Golden 是 inf/-inf/nan,NPU 值与 Golden 完全一致(无论标杆值是否与 Golden 相同)。
- 计算错误条件(仅一种):Golden 是 inf/-inf/nan 时,NPU 值与 Golden 不一致但标杆值与 Golden 一致。
- 需异常排查:Golden、NPU 结果、标杆值三者都不一致时,须进一步验证基准与标杆的有效性。

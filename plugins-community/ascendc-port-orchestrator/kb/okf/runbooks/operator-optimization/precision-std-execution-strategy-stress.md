---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "执行策略:压测次数与硬件平台覆盖、Device 初始化为 Nan"
description: "L0/L1 单用例执行 50 次、L2 执行 1000 次以捕获偶现精度问题;覆盖指定训练/推理平台;测试前把 Device 地址初始化为 Nan。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#3 执行策略
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, stress-test, platform-coverage]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
按精度等级重复执行(压测)以捕获偶现精度问题,并覆盖指定硬件平台:
- 压测次数:L0 单用例执行 50 次;L1 单用例执行 50 次;L2 单用例执行 1000 次。
- 硬件平台覆盖:训练 = 910A、910B2、910B3、910C 及后续平台;推理 = 910B2、910B3、310P、310B 及后续平台。
- 支持确定性计算的算子:单用例多次执行结果须一致。
- 开始测试前须将 Device 地址初始化为 Nan,以捕获越界计算问题。

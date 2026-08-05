---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "基准测试方法论（Profiling-First）"
description: "arch22→arch35 移植与正向→反向生成的同条件性能采集、CPU 真值精度验收和报告规则。"
confidence: single_run
original_id: doc/shared/BENCHMARK_METHODOLOGY.md
timestamp_inferred: true
tags: [benchmark, methodology, msprof, precision, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
# 基准测试方法论

## Profiling-First

每次优化前先用 `msprof` 采集目标 NPU 数据，定位计算、搬运、同步或调度瓶颈。

## 同条件 A/B

- 基线与候选使用相同 SoC、CANN、编译选项、输入、warmup、迭代次数和设备状态。
- 使用 `aclrtEvent` 记录设备侧时间，输出清零和数据搬运不得进入内核计时区间。
- 至少预热 3 次、测量 10 次，并报告均值和离散情况。

## 精度验收

- 优先使用 CPU FP64 真值；无法构造时使用经过审计的 CPU PyTorch 规格实现。
- arch22→arch35 移植可记录源代与目标代 CANN 结果，但不能替代真值。
- 正向→反向生成校验所有梯度的 shape、dtype、有限值和误差指标。
- 优化前后复用同一输入集，不能放宽既定阈值。

## 报告

记录环境指纹、输入规模、测量配置、设备侧时延、精度指标、确定性、失败用例和 profiling 结论。只有精度通过后才报告性能提升。

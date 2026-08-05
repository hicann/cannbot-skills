---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof vec_ratio ≠ atomicAdd share (SIMT scalar GM reads also go through the VEC pipe)"
description: "In SIMT mode, `aiv_vec_ratio=0.99` was misread as \"99% atomicAdd\". Measurement showed that removing atomicAdd saves only 4.4% (158us/3605us). 95%+ of VEC time is scalar GM random reads (indirect addre"
phenomenon: perf_regression
signal:
  - "msprof shows high vec_ratio in SIMT backward kernel"
confidence: single_run
original_id: OL-20
timestamp_inferred: true
tags: [ascendc, profiling_interpretation, ol-20]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
msprof shows high vec_ratio in SIMT backward kernel

## 教训 / 根因
In SIMT mode, `aiv_vec_ratio=0.99` was misread as "99% atomicAdd". Measurement showed that removing atomicAdd saves only 4.4% (158us/3605us). 95%+ of VEC time is scalar GM random reads (indirect addressing such as `input[expert*hdim+tid]`). NPU has no L2 cache; every scalar GM read goes to HBM through the VEC pipe. Must compare "with atomicAdd" vs "without atomicAdd" kernels to determine the actual atomicAdd cost. → P-P24 Sort-to-Reuse optimizing GM read amplification is the key.

## 证据
E11 msprof: BwdFull=3605us, grad_weight_only(no atomicAdd)=3447us, diff=158us(4.4%)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-20（category=profiling_interpretation，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->

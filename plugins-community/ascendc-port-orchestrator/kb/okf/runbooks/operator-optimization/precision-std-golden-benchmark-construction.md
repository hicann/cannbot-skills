---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "精度标杆构造:竞品对标优先,逐级降级备选"
description: "Golden 选择遵循竞品对标优先、逐级降级:①第三方芯片同功能算子 ②小算子拼接等效片段(融合/量化) ③自构 CPU 实现(非标准 dtype)。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#5.3.1 精度标杆构造 / 4.1 比对方法
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, golden-reference, benchmark]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
精度真值优先采用 CPU FP64 实现；无法直接使用 FP64 时，采用经过审计的 CPU PyTorch 规格实现，并记录内部 dtype 与 cast 时机。

目标实现直接与 CPU 真值比较。arch22 与 arch35 的 CANN 输出只用于定位迁移差异，不能替代真值。

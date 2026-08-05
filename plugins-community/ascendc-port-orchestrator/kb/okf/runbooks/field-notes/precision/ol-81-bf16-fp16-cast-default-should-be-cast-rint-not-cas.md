---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bf16/fp16 Cast default should be CAST_RINT, not CAST_ROUND (IEEE RNE)"
description: "AscendC's `CAST_ROUND` and `CAST_RINT` are **different** rounding modes: - `CAST_ROUND`: **round half UP** — if the first bit of the fraction being rounded is 1, round up (i.e. 0.5 always rounds up) -"
phenomenon: precision_issue
signal:
  - "kernel uses `Cast(dst, src, RoundMode::CAST_ROUND, count)` for fp32→bf16 or fp32→fp16 and needs to bit-exactly match the torch_npu/PyTorch reference"
confidence: single_run
original_id: OL-81
timestamp_inferred: true
tags: [cast_round, cast_rint, ascendc, precision, ol-81]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
kernel uses `Cast(dst, src, RoundMode::CAST_ROUND, count)` for fp32→bf16 or fp32→fp16 and needs to bit-exactly match the torch_npu/PyTorch reference

## 教训 / 根因
AscendC's `CAST_ROUND` and `CAST_RINT` are **different** rounding modes:
  - `CAST_ROUND`: **round half UP** — if the first bit of the fraction being rounded is 1, round up (i.e. 0.5 always rounds up)
  - `CAST_RINT`: **IEEE 754 RNE** — round half to even (0.5 decided by parity of the last kept bit)
  - torch_npu / PyTorch / almost all IEEE 754 hardware defaults to **RNE**
  - **Therefore, to align with PyTorch bf16/fp16 behavior you must use CAST_RINT, not CAST_ROUND**

## 证据
E2 level (empirical: a single flag flip raised bf16 from 1/9 PASS to 8/9 PASS)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-81（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->

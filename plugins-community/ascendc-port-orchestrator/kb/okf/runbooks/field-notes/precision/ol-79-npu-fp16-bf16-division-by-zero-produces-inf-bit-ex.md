---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "NPU fp16/bf16 division by zero produces inf — bit-exact match with PyTorch CPU"
description: "Ascend950PR fp16/bf16 division (including CANN torch_npu Div and AscendC VEC Div) produces `inf` when the divisor is near zero, with behavior **exactly matching** PyTorch CPU (bit-exact match). `pow(s"
phenomenon: precision_issue
signal:
  - "kernel has `A / std` or `pow(std, -3)` style ops that may produce inf, and the reference in native dtype also produces inf"
confidence: single_run
original_id: OL-79
timestamp_inferred: true
tags: [inf, ascendc, precision, ol-79]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
kernel has `A / std` or `pow(std, -3)` style ops that may produce inf, and the reference in native dtype also produces inf

## 教训 / 根因
Ascend950PR fp16/bf16 division (including CANN torch_npu Div and AscendC VEC Div) produces `inf` when the divisor is near zero, with behavior **exactly matching** PyTorch CPU (bit-exact match). `pow(std, -3)` overflow also produces `inf`, matching CPU. This means **no special handling of inf/NaN is needed**: do the division in native dtype directly and the resulting inf/NaN will match the reference exactly. Do NOT clamp early or cast to fp32 to avoid inf — that will instead cause a precision mismatch (reference expects inf but you returned a finite value).

## 证据
2026-04-16 A5 experiment `tests/repro/fp16_inf_div_test.py`: 6 fp16 division groups (including 1/0=inf, 100/6.1e-5=inf, -1/-0=inf) CPU vs NPU exact match; pow(std,-3) 4 groups exact match; bf16 3 groups exact match. E1 level (hardware measured).
  - 7_MoeGatingTopKSoftmax Phase D iter 1 (2026-04-17): the `is_finished` row used a -3.4e38 sentinel → max-subtract collapses to zero → softmax = 1/N uniform, which is wrong. PyTorch semantics: a fully-masked row softmax → NaN. Fix: short-circuit `is_finished` directly to NaN output. Same spirit as OL-79 (native semantics naturally match).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-79（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->

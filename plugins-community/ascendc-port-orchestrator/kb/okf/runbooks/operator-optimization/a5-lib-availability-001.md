---
schema_version: okf.v1
kind: guide
type: optimization_runbook
source_family: curated
title: "A3→A5 same-tier library availability table: matmul→MatmulImpl (bit-exact), conv→no library path (AIV is a downgrade), fused attention→none"
description: "Measured (not statically declared) availability of same-tier library paths on A5/CANN 9.2.0: MatmulImpl(reserved==0) is bit-exact for fp32/fp16/bf16; conv has no MatmulImpl to fall back to (AIV path = correctness-preserving but 30-60x slower on large K); fused attention has no library path (fix in place). Static declarations cannot prove library availability."
confidence: single_run
original_id: a5-lib-availability-001
timestamp_inferred: false
tags: [ascendc, crossgen, matmul, conv, attention, library-availability]
created_at: 2026-08-26T12:48:00Z
updated_at: 2026-08-26T12:48:00Z
---
# 可用性表（实测，非静态声明推导）

| 算子类 | A5 同档库路径 | 实证 | 退路 |
|---|---|---|---|
| matmul | **MatmulImpl（reserved==0，MDL scheduler）** | fp32/fp16/bf16 全部 bit 级正确（diff=0.0），15/15 + 24 组大/小 shape + 15 组 tiny 全 PASS；MatmulBothTrans 复验 55/55 | 手工裸 mmad（reserved==2/3）blockM≥2 静默算错 → 路由守卫禁用 |
| conv | **无**（没有 MatmulImpl 可退） | 全部 AIC 裸 cube 变体静默 NaN；AIV 前置后置 kernel 逐值正确 | **AIV Vector 直接卷积是真降级**：59/59 PASS（1.2e-7~5.5e-6），大 K 慢 30~60× |
| 融合 attention | **GE 内嵌融合库：无**；独立算子评测：**可用 MatmulImpl / regbase MatmulBase 实现 QK^T 与 PV**（45 证据：MatmulImpl(reserved==0) 对 fp32/fp16/bf16 bit 级正确 15/15+24+15；MatmulBothTrans 复验 55/55；FA 走 wholeport 模板 MIX_AIC_1_2 路线） | 45/55/57 原位/纯VEC 是当时线路选择，**不是库不可用的证据** | **优先 regbase MIX（MatmulBase QK^T/PV + UB online softmax，模式4 同步）**；纯 VEC 才是真降级（FA oc 线实测 187.9ms @ B2,S2048,N128,D128 vs cube 期望毫秒级，10-100× 代差） |

## 动作规则

1. 阶梯2（同档库替换）只在表里确认有库路径时才可用——目前仅 matmul 类。conv 的 AIV 是降级不是同档替换，必须走阶梯3 死亡证据流程。
2. 表里"无"指 GE 内嵌融合类库路径；独立算子评测的矩阵乘基元（MatmulImpl/MatmulBase）不是"无关库"——attention/conv 的矩阵乘部分仍可走该基元（阶梯2）。只有整算子无任何基元可用才视为无库。
3. 降级必须带证据（最小原语复现/归档豁免）+ 声明 perf debt 量级（conv AIV 兜底 = 大 K 30-60×）+ 恢复路线（A5 重写 cube）。

## 证据

- matmul：`docs/mmta_fp16_rootcause_report.md`；conv：`docs/conv2d_a5_rootcause_report.md`；复验：`docs/mbt_a5_rootcause_report.md`。

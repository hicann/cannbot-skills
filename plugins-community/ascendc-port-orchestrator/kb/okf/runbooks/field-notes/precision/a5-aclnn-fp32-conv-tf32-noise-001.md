---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A5 上 aclnn fp32 Conv2d golden 自身带 ~4e-4 相对噪声（疑似内部 tf32）——fp32 精度评测的'参照地板'不是 0"
description: "A5 上 aclnn fp32 Conv2d golden 自身带 ~4e-4 相对噪声（疑似内部 tf32）——fp32 精度评测的'参照地板'不是 0. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0 + torch_npu(driver 25.1.rc1); mode=port_a3_to_a5; op_class=所有以 aclnn fp32 conv/线性层为 golden 的精度评测. Provenance: 42_CoTAttention fp32 残留归因（2026-08-27，kimi 主线会话同权重 NPU-vs-CPU golden 对照，卡 3）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-a5-aclnn-fp32-conv-tf32-noise-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, golden, tf32, aclnn, fp32, gate-unsatisfiable]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# A5 aclnn fp32 conv golden 不是真 fp32：同权重 NPU vs CPU golden MERE ~5.7e-3（max ~2.5e-3），疑似 tf32 内部路径

## 事实（42_CoTAttention case1，卡 3，同进程同权重对照）
- 实验：`a5-data/cot-dsh-diag/golden_npu_vs_cpu.py`——golden Model 在 NPU 跑出权重后
  deepcopy 同一份权重到 CPU 重算，消除 seed/权重差异（注意 hash(key) 含 device，
  跨设备直接比会踩 `v351-reference-weight-seed-nondeterminism-001`）。
- 结果：NPU golden vs CPU golden MERE **5.69e-3**、max_abs 2.49e-3、mean_abs 1.96e-4；
  逐 stage：3x3 gconv max 2.49e-3、1x1 conv max 2.48e-3、1x1+relu+1x1 max 1.60e-3。
  相对量级 ~4e-4 ≈ 2^-11 —— tf32（10 位尾数）签名，不是 fp32 累加序噪声（那会是 ~1e-6）。
- 对照：候选 hi/lo 3-term 算法的 CPU 地板 vs CPU golden 只有 MERE 2.74e-6
  （`floor_test.py`；4-term 无增益 2.737e-6——**不要再提加 Al@Bl 第 4 项**）。
- 推论：候选对 NPU golden 的残留 MERE（本例 4.46e-3）主要是 **golden 参照自身噪声**。
  候选越接近真 fp32，反而离 NPU golden 越远（噪声是 golden 的）。

## 动作规则
1. fp32 case O5 FAIL 时先看量级：候选 MERE 与"同权重 NPU-vs-CPU golden MERE"同数量级 →
   参照噪声主导，候选可能已无缺陷；先跑同权重 golden 对照再决定改不改 kernel。
2. 冻结门（如 fp32 rtol 1.22e-4）严于参照噪声时，逐元素 matched_ratio 会被 golden 噪声
   系统性压低——按 near-miss 验收规则（AGENTS.md §3）处理并附 golden 噪声证据，不要让
   worker 为追平 tf32 噪声反复改候选。
3. 评测基建若能把 golden 挪到 CPU 计算（或 fp64 参照）再比，可绕开此地板；但这改变
   vendor 口径，需用户裁决。

## 证据
- a5-data/cot-dsh-diag/golden_npu_vs_cpu.log（卡 3，2026-08-27 18:26）。
- floor_test.py 输出：3-term 2.7436e-6 / 4-term 2.7370e-6。
- **判据不可满足的直接证据**（2026-08-27 21:50，卡 3，`case1_strict_gate.py`，
  harness 严格口径复刻：small 分支 |gold|<6.1e-5 要求 |d|≤2^-30、normal 分支纯
  rel≤2^-13 无 atol）：**干净 CPU fp32 golden 自己打 NPU golden，strict matched_ratio
  只有 0.2384**（门要求 0.9）——冻结门对 NPU aclnn fp32 golden 数学上不可满足；
  同设置下候选 vs 干净 CPU golden = 0.958（>0.9）、allclose 口径 1.0000。
  复现度锚点：候选 vs NPU golden = 0.2291 ≈ O5 正式报告的 0.228。
  即 fp32 严门 matched_ratio ~0.23 的 FAIL 全部是参照噪声伪影，与候选质量无关。
- **race 修复后复测确认 + 翻案驳回**（2026-08-27 23:55，卡 3，mean9 V→MTE2 race 修复后）：
  cand vs goldNPU = 0.2284、goldCPU vs goldNPU = 0.238、cand vs goldCPU = 0.9556——
  天花板不变。kw-32 翻案（"全精度 CPU sim vs CPU 参照 = 0.9874-0.9988 → 判据可满足"）
  比较对象错误：harness 打的是 NPU golden 而非 CPU 参照，sim 忠实度早已承认，
  改变不了对噪声 golden 的 0.238 天花板。判据不可满足终判维持。

## 关联 KB
- `v351-reference-weight-seed-nondeterminism-001`（跨设备/跨进程权重 seed 陷阱——做参照对照实验前必读）。
- `v351-datacopypad-extparams-stride-byte-unit-001`（同算子候选侧真根因）。

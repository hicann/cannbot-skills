---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "位真模拟器先过跨 case 保真校验再断案——模拟器自身 bug 会伪装成 kernel 精度缺陷"
description: "probe/worker 自写 bit-true 模拟器(kernel_sim)诊断 kernel 前必须先做跨 case 保真校验(对已知正确 case 位真复现设备输出),保真不过则模拟器结论一律作废. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=any(精度调查). Provenance: 29_FlashAttentionBwd 复盘(2026-08-29,dsh 复审发现 probe-3 kernel_sim.py dq 累加索引 bug,probe Iter-4 \"dq 发散 10-1000×\"结论作废)"
phenomenon: precision_issue
confidence: multi_run
original_id: user-kernel-sim-fidelity-check-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, bit-true-simulator, probe, diagnosis-discipline, fidelity-check]
created_at: "2026-08-30T02:30:00Z"
updated_at: "2026-08-30T02:30:00Z"
---

# 位真模拟器是诊断工具也是新代码：先证它对，再用它断案

## 事实（29_FlashAttentionBwd，Ascend950DT + CANN 9.2.0）
- 29 线 probe-3 用 kernel_sim.py（CPU 位真模拟 kernel 流水线）得出"dq 累加发散
  10-1000×，kernel 算法缺陷"；dsh 复审发现**模拟器自身** dq 累加索引 bug，修复后
  发散消失——**结论作废**。同期设备实测（fp64 真值矩阵）证明 kernel 无缺陷。
- 形态与 42 线教训同构：dump 跨 run 比较前必先证确定性；模拟器下结论前必先证保真。

## 动作规则（精度调查顺序补丁）
1. 跨 run 确定性 → 2. 位真模拟 → 3. 语义分歧 的铁律不变，但第 2 步增加前置：
   **模拟器对已知正确 case 必须位真复现设备输出**（跨 case 保真校验，至少覆盖
   PASS case + FAIL case 各若干），保真不过则模拟器结论一律作废、先修模拟器。
2. 模拟器结论与设备实测冲突时，默认信设备实测（模拟器是近似模型，设备是本体）。

## 证据
- `docs/review-29-dsh-20260829.md`（kernel_sim bug 定位与修复）；workspace
  `29_FlashAttentionBwd/probes/kernel_sim.py`（修复后版本）。

## 关联 KB
- OL-111（on-device pilot：probe 结论落到 kernel 改动前先在设备上小规模验证——若执行此条，模拟器假结论不会直接驱动改 kernel）。
- OL-207（建立可复现性先于构建代理调查：模拟器即代理，未证保真的代理调查全是浪费）。
- OL-208（确定性测试纪律只对被测目标有效——代理本身可被证伪；保真校验就是对模拟器这个 target 的证伪门）。

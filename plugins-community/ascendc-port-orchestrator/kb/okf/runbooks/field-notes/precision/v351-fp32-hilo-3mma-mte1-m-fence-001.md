---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "【阴性结果】V351 fp32 hi/lo 3-MMA GEMM '缺 MTE1_M/M_MTE1 硬事件围栏'假设——管线级实测证伪，勿再作为首选假设"
description: "【阴性结果】V351 fp32 hi/lo 3-MMA GEMM '缺 MTE1_M/M_MTE1 硬事件围栏'假设——管线级实测证伪，勿再作为首选假设. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=cube/attention fp32 hi/lo 分解 GEMM. Provenance: 42_CoTAttention（2026-08-27，dsh 提出 + 主线 kimi 会话决定性实验证伪）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-fp32-hilo-3mma-mte1-m-fence-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, negative-result, fp32, hi-lo, 3mma, fence]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# 【阴性结果】fp32 hi/lo 3-MMA 的 MTE1↔M 围栏假设：dump 反演支持，管线级实测证伪——fence 加了没效果

## 事实（42_CoTAttention，Ascend950DT + CANN 9.2.0）
- 假设（dsh 诊断报告 §3.3）：fp32 独有的"同一 L0C 上 3 个 Mmad 连加 + depth-2 L0A/L0B slot 复用"在 LoadData(MTE1)→Mmad(cube) 之间只有 TQue 隐式事件 + PipeBarrier<PIPE_ALL>，V351 不排序跨管 → MMA 读未落定分形。静态面自洽，且对 kimi 会话一份 fenced 变体 dump 的 SVD 反演给出 matched 0.9994。
- **决定性反证（主线 kimi 会话，卡 3，ASCEND_LAUNCH_BLOCKING=1 背靠背）**：
  - unfenced kw-23 原样：4 个 GEMM 输出全部 bit 级正确（k1AccT per-block max ~4e-6，vAccT max 4.3e-6）——**好条件下无围栏的 GEMM 数值本来就是对的**。
  - dsh 的 fenced+hostfix 构建（work3_fencefix）同条件同 case：stage 表与 unfenced 逐项几乎一致（最终 MERE 0.49 vs 0.40，差异在噪声内）——**fence 对管线结果无可测影响**。
  - dsh 的 0.9994 来自对单一 dump 快照的反演，未能被"实跑它自己的 fenced 构建"复现（dsh 未抢到北京窗口，E4 未执行）。
- 同期 kimi 会话另报告"自定义 AIC launch 静默丢弃（dbg 标记全零、返回码全 0、丢弃率时变 0%↔100%、BLOCKING 免疫）"——但主线实测：**同一时刻 torch aclnn 与完整 pipeline（13 个 kernel 全部执行、GEMM bit 正确）都正常，只有 kimi dbg 电池的 nop/gemm 探针 100% 丢弃** → 该现象疑似 dbg 构建/探针自身问题，不能外推为"环境性丢 launch"，**不能用来解释 kw-23 评测失败**（评测失败 MERE 0.42 与 unfenced 阻塞复跑 MERE 0.40 一致 = 确定性计算缺陷级联，非时变窗口）。

## 动作规则
1. fp32 hi/lo 3-MMA 算错时，**不要先加 MTE1_M/M_MTE1 围栏**——先跑"好条件对照"（BLOCKING + 逐 launch 同步 + 分级 dump）：GEMM 各 stage 对 CPU 真值是否 bit 级正确。本案真凶是 im2col 尾 tile（见 `v351-datacopypad-32b-roundup-oob-001`）+ 残留系统性偏差。
2. "对 dump 的反演分析"（SVD 重建等）不能替代"实跑修改后构建"——dsh 的教训：反演结论（fence 0.9994）与实跑（fence 无效）矛盾时，以实跑为准。
3. 围栏不是有害（加了不崩），但把它当根修会浪费迭代并掩盖真凶。
4. launch 丢弃类现象要区分"全链路受影响"还是"仅某探针/某构建"：同时刻跑 torch matmul 对照 + 完整 pipeline 对照，三者结论不同则嫌疑在探针自身。

## 证据
- 决定性实验日志：cot-dsh-diag/run_case1.log（unfenced 阻塞，GEMM bit 正确）、expB_work3_case1.log（fenced 等效）、两轮 battery_nop_vs_gemm 100% 丢弃 vs 同时刻 pipeline 正常。
- dsh 报告 docs/cot-fp32-gemm-dsh-diag-20260827.md（假设与反演）；kimi 报告 docs/cot-fp32-gemm-kimi-diag-20260827.md（dbg 电池现象）。

## 关联 KB
- `v351-pipe-all-tbuf-stale-001`（PIPE_ALL 不排序跨 pipe——对 VEC/TBuf 成立；本案提示不要把它机械外推到 cube 的 L0 分形通路）。
- `v351-datacopypad-32b-roundup-oob-001`（本案真凶之一）。

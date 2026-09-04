---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 纯 PipeBarrier<PIPE_ALL> + TBuf 不排序跨 pipe"
description: "V351 纯 PipeBarrier<PIPE_ALL> + TBuf 不排序跨 pipe. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=vector/attention 纯 TBuf<VECCALC> 多 kernel. Provenance: 端口移植战役 55_OutlookAttention failures_ledger iter kw-5（2026-08-25 沉淀）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-pipe-all-tbuf-stale-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, pipebarrier, tbuf, event-pairing, pb21]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# 纯 PipeBarrier<PIPE_ALL>(57 个全错)在 V351 上不排序 TBuf 的跨 pipe 依赖；须逐 kernel 事件 id 精确摆位（非机械修复）

## 事实（55_OutlookAttention，Ascend950DT + CANN 9.2.0 实测）
- 55 的 8 个 kernel 全部是纯 `TBuf<VECCALC>` + `PipeBarrier<PIPE_ALL>()`（共 57 个 barrier、**零** SetFlag/WaitFlag），是 PB-21 的教科书形态（55 ledger 行 60-61）。
- V351 严格 pipe 分离（EC-81）：`PipeBarrier<PIPE_ALL>` 在 TBuf 上**不排序** MTE2→V / V→MTE3 / V→MTE2；VEC 读到 stale/partial UB，每个 kernel 算出"有限垃圾"（finite garbage），级联放大——全部 case MERE 133-338、matched_ratio ~1e-5、candidate_nan_count=0（55 ledger 行 55-58）。
- 同一代码在 A3(V220) 上能跑对，仅靠 V220 的隐式跨 pipe 转发（55 ledger 行 63-64）。
- 修复（55 ledger 行 64-68）：**逐 kernel** 在 Init 里取一次事件 id（MTE2_V / V_MTE3 / V_MTE2），在每个跨 pipe 边界插显式 SetFlag/WaitFlag——V_MTE2 在 GM→UB 覆盖一个 V 读过 buffer 之前、MTE2_V 在 GM→UB 之后 VEC 读之前、V_MTE3 在 UB→GM 写回之前。**57 个 PipeBarrier<PIPE_ALL> 全部保留**（V→V drain 仍需要，EC-77），只加事件配对、不改任何算术。
- 坑：修完 kw-5 后 55 仍 all-cases-wrong（行 71），说明 PIPE_ALL 问题真实存在但不是该算子唯一根因——事件配对修复是必要非充分。

## 动作规则
1. 迁移源码特征：纯 `PipeBarrier<PIPE_ALL>` 且零 SetFlag/WaitFlag → 判 PB-21 类 stale-UB，直接按上方事件配对模板修，不要先跑一轮全量评测确认。
2. **这不是机械修复**：每个 kernel 的 buffer 读写顺序不同，事件 id 摆位必须按该 kernel 的实际跨 pipe 依赖逐个分析；57-D4（57 ledger 行 12）和 55-kw5 各烧了一整轮才走完这一步。
3. PIPE_ALL 不要删：V→V 同 pipe 排序仍需 barrier（EC-77）；只补跨 pipe 的 SetFlag/WaitFlag。
4. 诊断技巧：判别"kernel 未执行 vs 执行了但算错"用 NaN 哨兵——host 侧把输出张量预填 NaN，全 NaN=未执行、有限垃圾=执行了但错（55-kw4 固化，55 ledger 行 42-54）。

## 证据
- 55_OutlookAttention `failures_ledger.md` 行 55-69（iter kw-5）：57 个 PIPE_ALL 全错判定、修复模板、kw-4 NaN 哨兵证伪"未执行"。
- 57_ParallelPolarizedSelfAttention_evo `failures_ledger.md` 行 10（iter D2）：同族前驱问题——`PipeBarrier<PIPE_V>` 在 `Cast→FreeTensor→MTE2` 槽位复用序列上不足以排序 V→MTE2（PB-47/EC-77/EC-81），7 个槽位复用点换 PIPE_ALL；行 12（iter D4）TQue FreeTensor 前再补 PIPE_ALL fence 6 处。

## 关联 KB
- PB-21 / EC-81 / P-P75（PIPE_ALL 不排序跨 pipe）；PB-47 / EC-77（PIPE_V 不足、V→V drain）；EC-13（arch35 SetFlag/WaitFlag 签名）。
- trap_scan 规则 2+3（合并的有序诊断，方案文档 §W1）。
- `v351-aiv-tque-depth2-001`（depth 修复后常紧跟着爆这颗雷）。

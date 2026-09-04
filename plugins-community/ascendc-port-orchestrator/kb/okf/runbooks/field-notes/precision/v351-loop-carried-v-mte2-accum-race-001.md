---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 累加循环'Add(ubAcc,ubAcc,ubTmp) 后无屏障、下轮迭代 DataCopyPad 覆写 ubTmp'的 V→MTE2 跨迭代竞争 → 输出 run 间非确定（max_abs 可达 O(1)）；修复 = 循环末尾 PipeBarrier<PIPE_ALL>；诊断法 = 跨 run 确定性二分"
description: "V351 累加循环'Add(ubAcc,ubAcc,ubTmp) 后无屏障、下轮迭代 DataCopyPad 覆写 ubTmp'的 V→MTE2 跨迭代竞争 → 输出 run 间非确定（max_abs 可达 O(1)）；修复 = 循环末尾 PipeBarrier<PIPE_ALL>；诊断法 = 跨 run 确定性二分. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=所有'循环内 GM 分块读入 UB 临时缓冲 + 向量累加'模式的 vector kernel（mean/sum fold、分块 reduce、多 tap 卷积累加）. Provenance: 42_CoTAttention case0 fp16 根因（2026-08-27 主线 kimi 会话，卡 3：fuse 位真模拟 → 跨 run 确定性二分 → 源码定位 → 修复实测全链 bit-exact）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-loop-carried-v-mte2-accum-race-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, race, pipebarrier, nondeterminism, accumulation-loop]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# 循环承载累加的 V→MTE2 竞争：输出非确定、通道半区指纹；修复 = 迭代末尾 PipeBarrier<PIPE_ALL>

## 事实（42_CoTAttention MeanFoldCore，Ascend950DT + CANN 9.2.0，卡 3 全实测）
- 病态模式（mean over 9 taps 的 gg 循环）：
  ```cpp
  for (gg = 0; gg < 9; ++gg) {
      DataCopyPad(ubTmp[...], attAccGM_[...], rcp, pp);   // MTE2 写 ubTmp
      PipeBarrier<PIPE_ALL>();                             // 只保 MTE2→V 方向
      SetFlag/WaitFlag(MTE2_V);
      ... Cast ...
      Add(ubSum, ubSum, ubTmp, TILE);                      // V 读 ubTmp
  }   // ← 循环回绕，下一轮 MTE2 写 ubTmp 与本轮 V Add 读 ubTmp 并发 = 竞争
  ```
  循环体内的 barrier/flag 只保护"MTE2 拷贝完成 → V 读"方向；**"V 读完 → 下轮 MTE2 覆写"方向无任何屏障**。
- 实测后果：attMean run 间 max_abs **0.817**（同进程同输入跑两遍），经 softmax 放大成 out matched 0.877；同 kernel 的 vT 路径（独立缓冲 ubV）完全确定 → 逐 run 对比能把竞争钉到单 kernel 单路径。
- **"前半通道指纹"假象**：TILE_W=512 → mBlocks=2 → tiles 0..23（通道 0..191）全落 subblock 0、tiles 24..47 落 subblock 1，竞争窗口在两 subblock 上不同 → 失配呈通道半区聚集（前半 27.7% vs 后半 4.7%）。**通道聚集 ≠ 语义/索引 bug，先查 tile→core 映射再归因。**
- 修复：gg 循环末尾 `Add` 之后加 `PipeBarrier<PIPE_ALL>();`（V 排空后再放下一轮 MTE2）。实测：全链 11 级 run 间 bit-exact；vs golden attP matched 0.9285→1.0、out 0.8771→0.9982、半区指纹消除（0.44/0.56 均匀）。

## 诊断法（可复用，比读源码快）
1. **位真模拟先行**：dump 相邻两级 + 按 kernel 舍入链在 CPU 重建（含中间 fp16 舍入），失配 → 该级可疑；但 **stage dump 来自不同 run 时，必须先证跨 run 确定性**，否则把非确定假象当 kernel 缺陷（本案 fuse 一度被冤枉）。
2. **跨 run 确定性二分**：对每级 N 跑 `COT_STAGE=N` 两遍逐元素对比；第一个非 0 的级 = 竞争进入点；其余级 bit-exact 可同时洗清所有 GEMM/前后级。
3. 非确定幅度分级：O(1) 级（0.8）= 读到完全不同的数据（race/越界）；1e-3 级 = 约 1 ulp 累积顺序差异。

## 动作规则
1. 审查任何"循环内 DataCopyPad 入临时 UB + 向量累加到另一 UB"的 kernel：检查循环回绕边界有没有 V→MTE2 屏障（PipeBarrier<PIPE_ALL> 或 SetFlag/WaitFlag(V_MTE2) 对），没有即判缺陷——**即使单次跑结果"看起来对"**，竞争是时序概率事件，插桩/换卡/换负载都会变脸。
2. 精度调查顺序：跨 run 确定性 → 位真模拟 → 语义分歧（CPU sim），不要跳步。
3. 通道/空间聚集的失配指纹先做 tile→core/subblock 映射演算，再谈数据语义。
4. 与 `v351-pipe-all-tbuf-stale-001` 的关系：那条是"TBuf+PIPE_ALL 不可靠要显式事件对"；本条是**事件对只覆盖了循环体内一个方向、跨迭代方向裸奔**——事件对齐全不等于方向齐全，循环回绕边界要单独查。

## 证据
- a5-data/cot-dsh-diag/isolate42/ic_determinism_bisect.py（修复前 stage8=0.817/其余=0；修复后全 0）、ic_fuse_bitexact.py（位真模拟模板）、修复后 stage dump：attP 1.0 / out 0.9982。
- 修复落点：workspace 42_CoTAttention/kernel/cot_attention_kernels.h MeanFoldCore gg 循环（Add 后 PipeBarrier<PIPE_ALL>），r26 已 resume 全量 O5。

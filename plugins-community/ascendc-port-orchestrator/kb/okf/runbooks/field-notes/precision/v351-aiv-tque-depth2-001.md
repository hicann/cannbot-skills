---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 AIV TQue depth-2 静默损坏"
description: "V351 AIV TQue depth-2 静默损坏. Applies: soc=Ascend950DT/950PR(V351/arch35); cann=9.2.0; mode=port_a3_to_a5; op_class=vector/attention 含 TQue 队列. Provenance: 端口移植战役 57_ParallelPolarizedSelfAttention_evo failures_ledger iter D1（2026-08-25 沉淀）"
phenomenon: precision_issue
confidence: multi_run
original_id: user-v351-aiv-tque-depth2-001
classified_by: llm-assisted
timestamp_inferred: false
tags: [ascendc, v351, arch35, precision, tque, depth2, silent-corruption, aiv]
created_at: "2026-08-28T03:00:00Z"
updated_at: "2026-08-28T03:00:00Z"
---

# A3 验证过的 TQue depth-2 在 A5(V351) 上静默损坏；升 depth 4 修复（联动重算 UB 预算）

## 事实（57_ParallelPolarizedSelfAttention_evo，Ascend950DT + CANN 9.2.0 实测）
- A3 上跑通的 `TQue<...,2>` + InitBuffer depth-2 队列在 A5 上输出全零：O5 评测 45/50 FAIL，`matched_count==small_count` ⇒ 候选输出全零（57 ledger 行 9）。
- KB PB-2 记载 depth-2 TQue 在 Ascend950PR 上 99.5% 损坏率；57 是其在 V351 的活实例。
- 修复：3 个 TQue（xQue_/xOutQue_/wQue_）depth 2→4 + InitBuffer depth 参数 2→4（57 ledger 行 9）。
- **联动代价**：depth 翻倍直接放大 UB 占用；57 在 depth-4 就位后撞上 UB 越 248KB（行 11 的 A5 UB overflow 分支），升 depth 后必须立即重算全部 InitBuffer 字节和。

## 动作规则
1. 迁移源码出现 `TQue<..., 2>` 或 InitBuffer depth 参数为 2 时，一律升 depth 4 起步，不要先跑评测再烧一轮确认。
2. 升完 depth 立刻重算 UB 预算（InitBuffer 字节求和按 tiling 最大值 ≤248KB 可用上限），越界就减容/转 TQue<VECIN,1>/重排 buffer 顺序。
3. 判别信号：全 case mismatch 且输出全零（以输出张量统计为准；注意 small_count=0 时 `matched_count==small_count` 是伪迹，见 57-D3）。

## 证据
- 57_ParallelPolarizedSelfAttention_evo `failures_ledger.md` 行 9（iter D1）：depth 2→4 后从"45/50 全零"变为部分 PASS（进入下一轮根因）。
- UB 联动：同 ledger 行 11（iter D3）depth-4 在位时 11 个冻结 shape 的 InitBuffer 总和超 A5 248KB 可用。

## 关联 KB
- PB-2（depth-2 TQue 在 950PR 99.5% 损坏）。
- `v351-pipe-all-tbuf-stale-001`（队列 FreeTensor 后的跨 pipe 排序问题，常是修完 depth 后的下一颗雷）。
- trap_scan 规则 1（方案文档 §W1）。

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "生成新 kernel 时优先参考功能最相似的已有实现"
description: "写新 SIMT kernel 前，从已完成参考 kernel（clear/find_and_update/init_table/insert_and_evict/dump/rehash）中挑功能最相似者作为代码风格模板。"
confidence: single_run
original_id: SIMT_PATTERNS.md#7-已完成参考-kernel-一览
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, reference-kernel, code-reuse]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
生成新 kernel 时，优先参考功能最相似的已有实现作为代码风格模板，而不是从零编写。已完成参考 kernel 及其核心操作（路径均在 `hkv_hashtable/<name>/v35/<name>.h`）：

| Kernel | 核心操作 |
|--------|----------|
| `clear_kernel_vf` | 遍历所有槽位，写入 EMPTY_KEY / EMPTY_SCORE，重置桶大小 |
| `find_and_update_kernel_vf` | 哈希定位 → 线性探测 → 返回 value 指针和 found 标志 |
| `init_table_kernel`（多函数） | 初始化桶指针、填充 EMPTY_KEY / EMPTY_SCORE |
| `insert_and_evict_kernel_vf` | 插入 key，满时按 score 淘汰最旧 key |
| `dump_kernel_vf` | 导出 key-value-score，使用 GROUP_SIZE 协作组 |
| `rehash_kernel_vf` | 表容量翻倍后重新哈希，碎片整理 |

选型指引：按目标算子的核心操作类别匹配——纯遍历写入参考 clear/init；哈希查找参考 find_and_update；带原子插入+淘汰参考 insert_and_evict；需要协作组 shuffle 的参考 dump/rehash（它们额外带 GROUP_SIZE）。

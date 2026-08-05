---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "DISPATCH_GROUP_SIZE 协作组分发宏的嵌套顺序"
description: "协作组算子（insert_and_evict/dump/rehash）用 DISPATCH_GROUP_SIZE 把运行时 group_size 映射为编译时 GROUP_SIZE，嵌套自外向内为 GROUP_SIZE → VALUE_SIZE → EVICT_STRATEGY → VF_CALL。"
confidence: single_run
original_id: SIMT_PATTERNS.md#8.3-DISPATCH_GROUP_SIZE-使用说明
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, simt, optimization, dispatch-macro, cooperative-groups]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
含协作组 shuffle 的算子需在 `.cpp` dispatcher 中加 `DISPATCH_GROUP_SIZE`，把运行时 `group_size`（2/4/8/16/32）映射为编译时 `GROUP_SIZE` 模板参数。

- 适用算子：`insert_and_evict_kernel`、`dump_kernel`、`rehash_kernel`。
- 嵌套顺序（从内到外）：最内层 `Simt::VF_CALL<...>` 调 kernel → `DISPATCH_EVICT_STRATEGY`（如需）→ `DISPATCH_VALUE_SIZE` → 最外层 `DISPATCH_GROUP_SIZE`（如需）。
- 使用 `DISPATCH_GROUP_SIZE` 的算子，其 `_vf` 函数模板需额外添加 `GROUP_SIZE` 模板参数。

示例（insert_and_evict）：
```cpp
DISPATCH_GROUP_SIZE(group_size,
  DISPATCH_VALUE_SIZE(value_size,
    DISPATCH_EVICT_STRATEGY(evict_strategy,
      (Simt::VF_CALL<
          insert_and_evict_kernel_vf<uint64_t, DTYPE, uint64_t, STRATEGY, GROUP_SIZE>
      >(...)))));
```

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Double-Buffer Accumulation with Flush Threshold"
description: "Problem: When accumulating many grad rows for the same index after sorting, if the accumulation count is large (high fan-in), fp16/bf16 precision is gradually lost. Pattern: Maintain two UB accumulati"
confidence: single_run
original_id: P-P40
timestamp_inferred: true
tags: [scatter_add, optimization, p-p40, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: When accumulating many grad rows for the same index after sorting, if the accumulation count is large (high fan-in), fp16/bf16 precision is gradually lost.

**Pattern**: Maintain two UB accumulation buffers (addRes[0] and addRes[1]) and switch them when the index changes:
```
switchId = false
for each (index, grad) in sorted_data:
    if index == lastIndex:
        Add(addRes[switchId], addRes[switchId], grad)  // UB vector add
        count++
        if count >= FLUSH_LIMIT:  // fp16: 10, fp32: can be larger
            CopyOut(addRes[switchId], GM_output)  // uses SetAtomicAdd
            Duplicate(addRes[switchId], 0)  // clear
            count = 0
    else:
        CopyOut(addRes[switchId], GM_output)  // flush old buffer
        switchId = !switchId
        Duplicate(addRes[switchId], 0)  // clear new buffer
        Copy(addRes[switchId], grad)  // begin new accumulation
        lastIndex = index
        count = 1
```

**Flush threshold**: **LIMIT_COUNT_NUM = 10** (fp16/bf16). After accumulating > 10 fp16 values, precision loss starts to appear.

**Reason for dual buffers**: while buffer A is flushing to GM (MTE3 pipe), buffer B can start accumulating the next index (VEC pipe), achieving MTE3/VEC overlap.

**Evidence**: CANN embedding_dense_grad_v2.h:255-332, LIMIT_COUNT_NUM=10 (line 26). E1 level.

**Stop condition**: When the index is random (unsorted), it degrades to flushing after every single accumulation. Requires `embDim * sizeof(CT) * 2 < UB available space`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P40，convert_patterns_to_okf.py）。confidence 未升格。 -->

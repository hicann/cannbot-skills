---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Persistent-core dispatch (Persistent Kernel)"
description: "Anti-pattern: numBlocks = token_num (with many tokens, most blocks queue up time-slicing the 56 cores) cpp numBlocks = token_num; // e.g. 4096 blocks -> 56 cores queue 73 rounds for (int tid = threadI"
severity: high
confidence: single_run
original_id: P-P22
timestamp_inferred: true
tags: [thread_utilization, optimization, p-p22, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern**: `numBlocks = token_num` (with many tokens, most blocks queue up time-slicing the 56 cores)
```cpp
numBlocks = token_num;  // e.g. 4096 blocks -> 56 cores queue 73 rounds
for (int tid = threadIdx.x; tid < hidden_dim; tid += blockDim.x) {
    // handle 1 token
}
```

**Correct pattern**: 56 persistent blocks, each iterating over multiple tokens
```cpp
numBlocks = 56;  // MAX_AIV_CORES
for (uint32_t token = block_index; token < token_num; token += total_block_num) {
    // handle 1 token (same as original logic)
}
```

**Effect**: Eliminates block-scheduling overhead. medium (512 tokens) 3.2x speedup; xlarge (4096 tokens) 1.86x speedup.

**Applicability**:
- token_num >> MAX_AIV_CORES (56), with modest per-token work (scheduling overhead dominates)
- Forward-type kernels (no atomicAdd write conflicts)
- msprof scalar_ratio > 0.2 (significant indirect-addressing/scheduling-related code -> reducing scheduling is effective)
- **Does not apply** to backward (msprof vec_ratio ~= 1.0, compute-bound; scheduling overhead is not the bottleneck)

**Trigger condition (generator MUST check)**:
- `numBlocks = token_num` seen with token_num possibly >> 56 -> suggest generating a persistent variant
- msprof scalar_ratio > 0.2 -> persistent may be effective

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P22，convert_patterns_to_okf.py）。confidence 未升格。 -->

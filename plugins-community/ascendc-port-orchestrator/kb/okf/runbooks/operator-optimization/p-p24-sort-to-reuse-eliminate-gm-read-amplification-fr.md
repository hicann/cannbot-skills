---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Sort-to-Reuse — eliminate GM read amplification from indirect addressing"
description: "Problem: multiple work items read the same GM data via indirect indexing; each access is an independent HBM read. // Anti-pattern: per-token iteration, input[expert] is re-read N times by N tokens for"
confidence: single_run
original_id: P-P24
timestamp_inferred: true
tags: [memory_access, optimization, vec_ratio, p-p24, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: multiple work items read the same GM data via indirect indexing; each access is an independent HBM read.

```
// Anti-pattern: per-token iteration, input[expert] is re-read N times by N tokens
for token in all_tokens:
    expert = index[token]
    for d in hidden_dim:
        val = input[expert * hdim + d]  // 128 tokens share the same expert, read 128 times!
```

**Root cause**: this Ascend NPU SIMT path does not provide an effective cache for the repeated scalar reads. Every scalar GM read reaches HBM; when N work items read the same row via `arr[index[i]]`, actual HBM read volume is N × row_size rather than 1 × row_size.

**msprof verification** (E11, SG backward xlarge):
- `vec_ratio = 0.99` **does not equal** 99% atomicAdd
- Removing atomicAdd only saves 4.4% (158us / 3605us)
- **95%+ of VEC time is scalar GM random reads** (SIMT scalar reads go through the VEC pipe)
- In SIMT mode, `msprof vec_ratio` = sum of GM reads + compute + atomicAdd; it cannot be separated

**Correct pattern**: Sort-to-Reuse — sort by the indirect index so shared data is loaded once.

```
// Step 1: counting sort edges by expert_index → sorted_edges[], expert_offsets[]
// Step 2: per-expert processing
for expert in all_experts:
    DataCopy(local_buf, input[expert * hdim], hdim)  // read 1 time, not 128
    for edge in expert_run:
        token = sorted_edges[edge]
        // compute with local_buf, no need to re-read input[expert]
```

**Effect (measured)**:
| | GM reads | Time |
|---|:-:|:-:|
| per-token SIMT | 32K × 4096 = **134M** reads | 3447us |
| per-expert SIMD sorted | 256 × 4096 = **1M** reads | 265us |
| **Read reduction** | **128x** | **13x speedup** |

**Trigger condition**: you see `arr[index[i]]`-form indirect GM reads + multiple work items sharing the same index value → consider sort-to-reuse.

**Applicability**:
- MoE scatter-gather: `input[expert_index]` shared by N tokens
- GNN message passing: `features[neighbor_id]` shared by multiple edges
- Embedding lookup: `embedding[token_id]` shared across multiple positions
- Any indirect addressing pattern with fan-out

**Not applicable**:
- Index is fully unique (no sharing) → sorting yields no reuse benefit
- Data size < UB capacity → load once; no sorting needed

**Relation to P-P21**: P-P21 (sorted-edge accumulation) focuses on eliminating atomicAdd write conflicts. P-P24 focuses on eliminating GM read amplification. They typically appear together (the same sort fixes both read and write), but P-P24's benefit is much larger than P-P21's (95% vs 4.4% for SG backward).

**Relation to msprof interpretation**: a high SIMT-mode `vec_ratio` is not necessarily an atomicAdd bottleneck. You must compare timings of "with atomicAdd" vs "without atomicAdd" kernels to confirm. If the difference is <10%, the real bottleneck is GM reads.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P24，convert_patterns_to_okf.py）。confidence 未升格。 -->

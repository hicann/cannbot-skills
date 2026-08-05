---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "CANN op precision contract — bf16/fp16 reduction must use fp32 promotion"
description: "Trigger: kernel needs to reduce a bf16/fp16 tensor (sum, mean, max, min, prod, etc.). Pattern: every CANN reduction operator's DAG follows a fixed 5-step structure: CopyIn<T> → Cast<fp32, T> → ReduceX"
confidence: single_run
original_id: P-P52
timestamp_inferred: true
tags: [precision, optimization, cast_rint, cast_round, p-p52, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: kernel needs to reduce a bf16/fp16 tensor (sum, mean, max, min, prod, etc.).

**Pattern**: every CANN reduction operator's DAG follows a **fixed 5-step structure**:
```
CopyIn<T>  →  Cast<fp32, T>  →  ReduceXxxOp<fp32>  →  Cast<T, fp32>  →  CopyOut<T>
```
i.e., **bf16/fp16 input must first be cast to fp32, reduction is performed in fp32, and the result is cast back to the original dtype**.

**Source evidence** (`reduce_sum_dag.h:28-38`):
```cpp
template <typename T, typename PromteT>  // T = input dtype, PromteT = fp32
struct ReduceSumDag {
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;    // T → fp32
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, Cast0>;    // reduce in fp32
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;     // fp32 → T
};
```

**In our kernel writing style**:
```cpp
// bf16/fp16 input
Cast(fp32Buf, srcBf16, RoundMode::CAST_NONE, count);  // T → fp32 (lossless)
PipeBarrier<PIPE_V>();
ReduceSumP47(fp32Buf, count);  // or AscendC::ReduceSum<float, Pattern::Reduce::AR, true>
// result in fp32Buf[0]
// if you need to put the scalar result back into T, then Cast(..., CAST_RINT)
```

**Key**: Cast must use `CAST_RINT` (IEEE RNE), not `CAST_ROUND` (round half up) — see OL-81.

**Applicability**: all ops with bf16/fp16 input + internal reduction (LayerNorm, GroupNorm, AdaIN, Softmax, Sum, Mean, Var, etc., including backward)

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P52，convert_patterns_to_okf.py）。confidence 未升格。 -->

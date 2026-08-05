---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT gather achieves ~1.0x vs CANN for random-access patterns"
description: "For torch.gather, SIMT per-element (512 threads x 56 blocks) hits mean 0.86x vs CANN by using dcache for random reads instead of per-element DMA (SIMD V1 was 0.006x). Slow on large fp16/bf16 dim=last (0.16-0.4x)."
confidence: single_run
original_id: OL-40
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-40, gather, index-select, simt, random-access, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: `torch.gather`, `index_select`, or any per-element indirect-addressing op.

**选型**: For `torch.gather`, SIMT per-element with **512 threads × 56 blocks** achieves mean **0.86x**, median **0.89x** vs CANN.
- SIMT beats CANN on small tensors (~1.2x) and large fp32 (~1.1x).
- SIMT is slower on large fp16/bf16 with `dim=last` (0.16–0.4x).

**Key insight**: SIMT uses **dcache for random reads** instead of per-element DMA (the V1 SIMD approach ran at 0.006x). The remaining gap (esp. the fp16 slow cases) is an optimization knowledge gap, not a hardware limitation — CANN uses the same AscendC API (see OL-42).

**Evidence**: Gather V2 (2026-04-10): SIMD 0.006x → SIMT 0.86x mean (158x improvement).

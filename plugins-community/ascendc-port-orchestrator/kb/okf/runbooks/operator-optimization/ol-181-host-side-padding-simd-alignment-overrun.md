---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Host-side padding resolves multi-block SIMD elementwise alignment overrun"
description: "For multi-block SIMD elementwise kernels where DataCopy writes 32B-aligned element counts, pad the host allocation by 16 elems (fp16/bf16) or 8 (fp32) so the last tile can't overflow."
confidence: single_run
original_id: OL-181
classified_by: llm-assisted
timestamp_inferred: true
tags: [memory-access, optimization, ol-181, simd, alignment, datacopy]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
When a multi-block SIMD elementwise kernel uses `DataCopy` to write **32B-aligned element counts**, the aligned count on the last tile can exceed the logical element count and overrun the buffer. The fix is to pad the host-side allocation by the alignment margin:
- **16 elements** for fp16 / bf16
- **8 elements** for fp32

Applied as `INPUT_PAD` and `OUT_PAD` on the host allocation, this absorbs the aligned tail write on the final tile without changing the logical semantics — the extra padded slots simply catch the over-aligned `DataCopy` write instead of corrupting adjacent memory.

Verified on soc=Ascend910_9382 (V220), cann=9.0.0. Evidence: 2_SwiGLU a3-ds ko-4 (2026-05-21) — `INPUT_PAD` and `OUT_PAD` of 16 elements prevented the `DataCopy` overflow on the last tile.

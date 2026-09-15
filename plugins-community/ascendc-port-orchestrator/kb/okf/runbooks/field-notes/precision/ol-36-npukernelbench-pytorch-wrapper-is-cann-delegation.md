---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "NPUKernelBench — PyTorch wrapper is CANN delegation (PROHIBITED)"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "when considering \"simple\" implementation via PyTorch ops"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-36
timestamp_inferred: true
tags: [ascendc, ol-36]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (20_FusedRopeWithQkNormAndKvCacheUpdate, 22_HybridAttentionMaskPreparation). Do not downgrade.
- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when considering "simple" implementation via PyTorch ops
- **Lesson**: Implementing model_new_ascendc.py by calling PyTorch ops (torch.permute, torch.sort, F.layer_norm, etc.) delegates to CANN via torch_npu. This is the SAME as calling aclnn* APIs inside a kernel — it's wrapper hacking. ALL computation must use AscendC primitives (DataCopy, VEC ops, TQue/TBuf, scalar GetValue/SetValue).
- **Evidence**: Permute (#12) was caught as CANN delegation and reverted (2026-04-09)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-36（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

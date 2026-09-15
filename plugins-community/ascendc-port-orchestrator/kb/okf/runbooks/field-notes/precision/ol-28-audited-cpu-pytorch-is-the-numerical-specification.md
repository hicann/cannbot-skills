---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Audited CPU PyTorch is the numerical specification"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "any arch22→arch35 migration or forward→backward generation task"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-28
timestamp_inferred: true
tags: [ascendc, ol-28]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ **VALIDATED-CPU — this entry is the anchor for the CPU-truth approach**. PyTorch CPU is the specification, not any single device implementation. Aligns with OL-97.
- **Category**: process, trust_calibration
- **Loaded by**: Analyzer, Builder, QA, Lead
- **Trigger**: any arch22→arch35 migration or forward→backward generation task
- **Lesson**: device implementations may contain rounding or edge-case behavior that differs from the mathematical contract. The target implementation must be checked against an audited CPU specification.
  **The verification chain should be**:
  ```
  PyTorch definition (spec, mathematical contract)
  AscendC implementation
      ↓ verify
  bit-exact or within-tolerance against the PyTorch spec
  ```
  **Hard rules**:
  1. Build or audit the CPU forward specification before implementation.
  2. Derive backward truth from the same forward specification.
  3. Record any arch22/CANN versus CPU differences as diagnostics, never as a replacement truth.
  4. Verify arch35 outputs and every requested gradient against CPU truth.
- **Evidence**: project precision-audit practice summarized by OL-97.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-28（category=process, trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

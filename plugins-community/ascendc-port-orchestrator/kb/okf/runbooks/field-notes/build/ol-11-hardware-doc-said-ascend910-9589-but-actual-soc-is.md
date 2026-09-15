---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Hardware doc said Ascend910_9589 but actual SOC is Ascend950PR_9589"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "when configuring SOC_VERSION for CMake builds or kernel compilation"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-11
timestamp_inferred: true
tags: [ascend910_9589, ascend950pr_9589, ascendc, ol-11]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: trust_calibration
- **Loaded by**: Builder, QA
- **Trigger**: when configuring SOC_VERSION for CMake builds or kernel compilation
- **Lesson**: The hardware documentation and initial setup guides listed the SOC as `Ascend910_9589`. The actual chip is `Ascend950PR_9589` -- the only Ascend variant that supports both SIMT and SIMD modes. Using the wrong SOC version causes silent compilation of kernels for the wrong architecture, which may run but produce incorrect results or suboptimal code. Always verify SOC version with `npu-smi info` on the actual hardware before trusting any documentation.
- **Evidence**: MEMORY.md ("Correct SOC version: Ascend950PR_9589"), docs/design/PLUGIN_PARADIGM_NOTES.md#ascend-chip-comparison

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-11（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

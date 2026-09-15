---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Performance benchmarking must use the target framework's standard tools"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "any performance report or benchmark-result publication"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-31
timestamp_inferred: true
tags: [ascendc, ol-31]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process, measurement
- **Loaded by**: QA, Builder, Lead
- **Trigger**: any performance report or benchmark-result publication
- **Lesson**: In GELU evaluation, comparing C++ aclrtlaunch raw invocation against Python torch_npu invocation produced the wrong conclusion "CANN is 2-4x faster". After switching to NPUKernelBench's standard tool (utils/performance.py), results became 0.83-1.11x (all ≥0.8x PASS).
  **Root cause**: raw invocation has no Python dispatch overhead (~10us); torch_npu invocation does. The two timing baselines are not comparable.
  **Hard rules**:
  1. Use the target benchmark framework's standard evaluation tool; do not write your own timing code
  2. Ensure reference and candidate go through the same call path (same Python overhead)
  3. If you do write your own timing tool, document its differences from the standard tool
  4. Integration shape: kernel(.cpp/.h) + pybind11.cpp + model_new_ascendc.py(ModelNew)
  5. Build: `utils/build_ascendc.py`, Verify: `utils/verification_ascendc.py`, Perf: `utils/performance.py`
- **Evidence**: GELU evaluation (2026-04-08), NPUKernelBench framework analysis

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-31（category=process, measurement，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

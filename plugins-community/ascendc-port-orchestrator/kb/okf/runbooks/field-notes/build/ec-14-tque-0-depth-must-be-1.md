---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`TQue<..., 0>` — depth must be >= 1"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-14
timestamp_inferred: true
tags: [tque, ascendc, ec-14]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: static assertion failed: must use AllocTensor<LocalTensor&> api while tque's depth is zero
  ```
- **Root cause**: `TQue` template's second parameter is the depth (number of buffer slots). Depth 0 means "use pass-by-reference AllocTensor API" which has a completely different usage pattern. Standard AllocTensor/EnQue/DeQue/FreeTensor requires depth >= 1.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::TQue<AscendC::TPosition::VECIN, 0> xQueue_;

  // AFTER (works):
  AscendC::TQue<AscendC::TPosition::VECIN, 1> xQueue_;
  ```
- **Evidence**: Cumsum V1 build failure (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-14，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

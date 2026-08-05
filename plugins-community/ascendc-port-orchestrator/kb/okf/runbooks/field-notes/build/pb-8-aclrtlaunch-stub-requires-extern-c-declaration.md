---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "aclrtlaunch Stub Requires extern \"C\" Declaration"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Linker error undefined reference to aclrtlaunch_xxx(...) when calling kernel from test code"
confidence: single_run
original_id: PB-8
timestamp_inferred: true
tags: [ascendc, pb-8]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Linker error `undefined reference to aclrtlaunch_xxx(...)` when calling kernel from test code
- **Root cause**: Auto-generated `host_stub.cpp` exports functions as C symbols (no name mangling). Test code declaring them as C++ gets mangled names → linker mismatch.
- **Workaround**: Always use `extern "C" { uint32_t aclrtlaunch_xxx(...); }` in test code
- **Status**: By design (not a bug, but easy to forget)
- **Evidence**: MXFP4 test (2026-04-07)

## Operational Issues

<!-- 迁移自 porter kb/target/ascendc/（PB-8，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

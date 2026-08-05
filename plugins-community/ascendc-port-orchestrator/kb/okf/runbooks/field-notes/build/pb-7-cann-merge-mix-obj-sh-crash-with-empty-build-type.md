---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CANN merge_mix_obj.sh Crash with Empty --build-type"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "make fails at 95% with Error 1 in merge_mix_obj.sh — shift 2 fails"
confidence: single_run
original_id: PB-7
timestamp_inferred: true
tags: [make, cmake, cmake_build_type, ascendc, pb-7]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh` — `shift 2` fails
- **Root cause**: `cmake` invokes `merge_mix_obj.sh --build-type` without a value when `CMAKE_BUILD_TYPE` is unset. The bash `shift 2` fails because only 1 arg remains.
- **Affected**: CANN 9.0.0, AIV-only kernels (AIC dir empty, merge step still runs)
- **Workaround**: Always set `-DCMAKE_BUILD_TYPE=Release` in cmake invocation
- **Status**: OPEN (CANN build system bug)
- **Evidence**: MXFP4 project (2026-04-07), `merge_mix_obj.sh` line `shift 2` on `--build-type`

## Build Integration Issues

<!-- 迁移自 porter kb/target/ascendc/（PB-7，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

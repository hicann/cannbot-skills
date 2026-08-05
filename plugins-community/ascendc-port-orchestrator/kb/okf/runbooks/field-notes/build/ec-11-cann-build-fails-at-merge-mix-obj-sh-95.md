---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CANN Build Fails at merge_mix_obj.sh (95%)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "make fails at 95% with Error 1 in merge_mix_obj.sh"
confidence: single_run
original_id: EC-11
timestamp_inferred: true
tags: [make, cmake_build_type, ascendc, ec-11]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `make` fails at 95% with `Error 1` in `merge_mix_obj.sh`
- **Root cause**: `CMAKE_BUILD_TYPE` not set → cmake passes empty `--build-type` to `merge_mix_obj.sh` → `shift 2` fails
- **Fix**: Always pass `-DCMAKE_BUILD_TYPE=Release` to cmake
- **Related**: PB-7 in PLATFORM_BUGS.md

<!-- 迁移自 porter kb/target/ascendc/（EC-11，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

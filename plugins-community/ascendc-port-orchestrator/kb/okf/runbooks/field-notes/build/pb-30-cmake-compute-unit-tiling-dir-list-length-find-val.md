---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CMake `COMPUTE_UNIT` ≠ `TILING_DIR` list length → `find_value_by_key` FATAL_ERROR [V351, build-system]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2"
phenomenon: build_failure
signal:
  - "At CMake configure time, find_value_by_key() (internal CANN cmake helper) calls message(FATAL_ERROR ...) complaining about COMPUTE_UNIT and TILING_DIR list-leng"
confidence: single_run
original_id: PB-30
timestamp_inferred: true
tags: [compute_unit, tiling_dir, find_value_by_key, ascendc, pb-30]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §354-367`

**Symptom**: At CMake configure time, `find_value_by_key()` (internal CANN cmake helper) calls `message(FATAL_ERROR ...)` complaining about COMPUTE_UNIT and TILING_DIR list-length mismatch.

**Root cause**: CMake silently drops empty strings (`""`) from list arguments. When the author intends `TILING_DIR = ["", "", "arch35"]` to mean "A3/910b/910_93 use root, A5 uses arch35/", CMake collapses it to `["arch35"]` (1 element) while `COMPUTE_UNIT` stays at 3 elements → length mismatch.

**Anti-pattern** (BAD — empty string eaten):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   ""           ""              "arch35"
# Actual after parse: COMPUTE_UNIT=3 items, TILING_DIR=1 item → FATAL_ERROR
```

**Fix** (use named subdirs even when not strictly needed):
```cmake
COMPUTE_UNIT "ascend910b" "ascend910_93" "ascend950"
TILING_DIR   "default"    "default"      "arch35"
# Then create op_host/default/ alongside op_host/arch35/
```

**Detection signature**:
```bash
# Count list items in each — must match
awk '/COMPUTE_UNIT/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "cu",n}
     /TILING_DIR/{n=0; for(i=2;i<=NF;i++) if($i!~/^$/) n++; print "td",n}' \
     op_host/CMakeLists.txt
```

**Evidence**:
- PR 103 codifies as Trap #2; ties directly to CMake list-arg semantics
- Reproducible: `cmake -DLIST_VAR="" ""` produces 0-length list, not 2-length

**Mitigation gate**: `aog-self-critic` post-worker — when adding an `ascend950` element to COMPUTE_UNIT, REQUIRE that TILING_DIR has a corresponding non-empty entry; if mismatched, suggest the named-subdir fix.

<!-- 迁移自 porter kb/target/ascendc/（PB-30，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

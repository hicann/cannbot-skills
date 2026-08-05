---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "BF16 `__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113` guards MUST be removed in `arch35/` ports [V351, port_a3_to_a5]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5"
phenomenon: build_failure
signal:
  - "Compile failure inside arch35/ kernel header — a BF16-typed path that should compile is dead-code-eliminated, leading to \"undefined function\" errors at the call"
confidence: single_run
original_id: EC-49
timestamp_inferred: true
tags: [ascendc, ec-49]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=all_port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §289-303 + OL-142`

**Symptom**: Compile failure inside `arch35/` kernel header — a BF16-typed path that should compile is dead-code-eliminated, leading to "undefined function" errors at the call site OR (worse) silent fallback to wrong-dtype path.

**Root cause**: A3 source liberally wraps BF16 / new-dtype paths in negative guards `#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))`. On A5 (`__NPU_ARCH__ == 3510`) these conditions evaluate false → guard becomes `#if !false` = include. ACCIDENTALLY correct, but fragile — if a future A5 SoC variant gets a different numeric ID, the guard would silently exclude code. More importantly, A3 BF16 paths inside the guards are STILL written for V220 codegen; on A5 they may use deprecated APIs.

**Fix (mechanical, every port_a3_to_a5 L1 step)**:

```bash
# Identify in A3 source
grep -nE "__NPU_ARCH__\s*==\s*(3003|3113)" op_kernel/*.h op_kernel/*.cpp

# For each match: remove the guard, keep the body unconditional
```

```cpp
// BEFORE (A3 source)
#if !(defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3003 || __NPU_ARCH__ == 3113))
    BF16_PATH;
#endif

// AFTER (arch35/ port) — guard removed
BF16_PATH;
```

**Anti-pattern**: copy A3 source verbatim into `arch35/` without removing these guards. The build "works" by accident (3510 ≠ 3003 ≠ 3113) but the port is unstable.

**Detection signature**: after `arch35/` files are written, `grep -E "__NPU_ARCH__\s*==\s*(3003|3113)" arch35/*.h arch35/*.cpp` MUST return zero hits.

**Evidence**: PR 103 §289-303 codifies as canonical L1 step

**Mitigation gate**: `aog-self-critic` post-worker — auto-grep `arch35/` for residual 3003/3113 references; reject finalize if found.

**Cross-reference**: OL-142 (NPU_ARCH macros), EC-47 (ToFloat fix often needed after this removal).

<!-- 迁移自 porter kb/target/ascendc/（EC-49，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

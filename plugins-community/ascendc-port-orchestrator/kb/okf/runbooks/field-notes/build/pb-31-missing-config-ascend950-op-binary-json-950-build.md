---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Missing `config/ascend950/<op>_binary.json` → 950 build SILENTLY SKIPPED [V351, build-system]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2"
phenomenon: build_failure
signal:
  - "Build appears to succeed (no error / warning), but the resulting op package does NOT include the A5 (ascend950) kernel binary. At runtime, aclnn or torch_npu re"
confidence: single_run
original_id: PB-31
timestamp_inferred: true
tags: [ascendc, pb-31]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all_port_a3_to_a5; phase=O2`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 SKILL.md §369-372`

**Symptom**: Build appears to succeed (no error / warning), but the resulting op package does NOT include the A5 (ascend950) kernel binary. At runtime, aclnn or torch_npu reports "operator not supported on this device" or similar generic missing-op error.

**Root cause**: The CANN build system gates per-SoC kernel compilation on the presence of `op_host/config/<compute_unit>/<op_name>_binary.json`. The check is "if file exists, compile for this SoC; else skip without warning". For port_a3_to_a5, if the dev added `OpAICoreConfig` for ascend950 in `_def.cpp` AND modified CMakeLists.txt AND created `arch35/` source — but FORGOT to copy `config/ascend950/<op>_binary.json` — the 950 path is silently skipped and the bug surfaces only at runtime on the target device.

**Anti-pattern**:
```bash
ls op_host/config/
# ascend910b/  ascend910_93/    ← ✘ no ascend950/
```

**Fix**:
```bash
mkdir -p op_host/config/ascend950
cp op_host/config/ascend910b/<op>_binary.json       op_host/config/ascend950/
cp op_host/config/ascend910b/<op>_simplified_key.ini op_host/config/ascend950/
# Adjust _binary.json contents if A5 supports different dtypes (FP8/HiFloat8/etc.)
```

**Detection signature**:
```bash
# After kw declares done, verify:
test -f op_host/config/ascend950/${op}_binary.json &&
test -f op_host/config/ascend950/${op}_simplified_key.ini ||
  echo "BUG: config/ascend950/ incomplete — 950 build will be silently skipped"
```

**Evidence**:
- PR 103 codifies as Trap #3; explicitly described as "缺失则 950 编译被静默跳过（无报错），极易遗漏"
- Direct hardware reproduction: when build/ pipeline doesn't even reach compile-arch35 step

**Mitigation gate**: `aog-self-critic` post-worker pass MUST verify both files exist in `config/ascend950/`; reject finalize if missing. Additionally, `aog-prior-art-verify` Phase 3 (build candidate) should fail-fast if these files don't exist.

**Other instances (predicted)**: any future A3→A5 port. The 14 no-upstream ops in our scan list (cohort 1+2) will all need this check at finalize time.

<!-- 迁移自 porter kb/target/ascendc/（PB-31，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

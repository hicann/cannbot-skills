---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Build-System Pattern(BSP)子库说明"
description: "target/ascendc/build_system — Build-System Pattern (BSP) KB subdir Patterns about HOW CANN op kernels are BUILT + LAUNCHED, not WHAT they compute. Distinct from kernel-structural patterns (patterns/PA"
confidence: single_run
original_id: doc/target/ascendc/build_system/README.md
timestamp_inferred: true
tags: [build-system, bsp, launch-glue, cann-learner-mode6, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
# target/ascendc/build_system — Build-System Pattern (BSP) KB subdir

Patterns about HOW CANN op kernels are BUILT + LAUNCHED, not WHAT they compute.
Distinct from kernel-structural patterns (`patterns/PATTERN_INDEX.md`, prefix `P-P*`):

| Layer | KB subtree | Prefix | Mode (cann_learner) |
|---|---|---|---|
| Kernel algorithm/structural | `patterns/` | `P-P*` | Mode 5 (`kernel_structural`, default) |
| Build-system / launch glue | `target/ascendc/build_system/` | `BSP-N` | Mode 6 (`build_system`, NEW 2026-05-21) |

## Why a separate subtree

FA Pattern A iter 1-5 (~$53 spend, all 5 worker-scope hypotheses falsified) demonstrated that V220 MIX_AIC_1_2 cube+vec mixed-mode kernels need correct **build/launch glue** in addition to correct kernel code. The required recipes live in `CMakeLists.txt` + `register_*.cpp` + `op_proto*.cpp` + `*_apt.cpp` — NOT in `kernel.h`. cann_learner Mode 5's brief restricted scope to "2-5 kernel files (header + impl + tiling)", which structurally excluded the build glue. Mode 6 extends scope; BSP-N is the corresponding KB destination.

## File layout (proposed)

- `candidates.md` — unverified BSP-N entries (cann_learner Mode 6 output, append-only)
- `PRINCIPLES.md` — canonical BSP-N entries promoted from candidates (Mode 2 review-merged)
- `README.md` — this file (overview + indexing convention)

## BSP-N entry shape

Same layered shape as P-P entries but topics differ:
- **Title**: build-system principle (e.g. "Per-source-file -DASCENDC_MATMUL_AICORE isolation for V220 mixed-mode builds")
- **applies_to**: `soc=... ; cann=... ; op_class=...` (same scope-tag convention)
- **verified_on**: which CANN module's CMakeLists.txt + ops with which dispatch macros
- **Concrete anchor**: CMake snippet (`target_compile_definitions(...)` block) or `register_*.cpp` snippet, using public CMake / aclrt API only
- **Predicted other instances**: which op-classes share the build-system pattern
- **Risks / pre-promotion**: CANN-version-specific concerns (e.g. recipe verified on CANN 9.0.0; may differ on 9.1)

## Scanner gate adjustments for build-system content (cann_learner Mode 6)

- **C34a identifier-denylist**: CMake content has different vocabulary. `target_compile_definitions`, `add_dependencies`, `add_modules_sources`, `find_package` are public CMake API → safe. Internal Bazel target names (e.g. `_internal_*`, `:ops_internal_*`) → denylist.
- **C34c copy-shape threshold**: build-system files have higher legitimate boilerplate N-gram overlap (`cmake_minimum_required(VERSION X)`, `project(...)`). Threshold should be relaxed (10% vs Mode 5's 5%) OR boilerplate prefix excluded.
- **C34b compile-gate**: CMake content can be syntax-validated via `cmake --debug-trycompile --parse-only` (not strictly "compile" but valid-syntax check).
- **C35 KB-overlap**: cross-ref vs existing OL/PB/EC entries on build-system topics + any prior BSP-N.

## First-targeted run

Mode 6 will be first invoked on `flash_attention_score`:
```bash
PYTHONPATH=src/scripts python3 -m cann_learn.mode5_runner \
  --extraction-mode build_system \
  --op 3_FusionAttention \
  --workspace workspace/3_FusionAttention \
  --module-path /home/npu_user/workspace/cann/ops-transformer/attention/flash_attention_score \
  --kb-root src/skills/references \
  --api-catalog src/skills/references/target/ascendc/API_CATALOG.md \
  --allow-finalized-without-researcher-iter
```

Expected output: BSP candidates documenting per-source-file flag isolation + launch macro routing + register glue patterns that the next fo spawn can apply to unblock FA's silent-hang root cause.

## Cross-references

- Design doc: `docs/design/KB_DESIGN_NOTES.md#cann-learn-mode6-build-system-extraction-2026-05-21`
- Origin context: `workspace/3_FusionAttention/FO_BRIEF_PATTERN_A_5_ITER_FALSIFICATION_CHAIN.md`
- Code changes: `src/scripts/cann_learn/{mode5_runner,agent_spawn,summary_schema}.py` + tests
- Agent doc: `src/agents/aog-cann-learner.md` §Phase B (mode-dependent scope)

<!-- 迁移自 porter kb/target/ascendc/build_system/README.md(整档忠实搬运,convert_docs_to_okf.py)。跨 op 参考/方法论知识,非机械家族。 -->

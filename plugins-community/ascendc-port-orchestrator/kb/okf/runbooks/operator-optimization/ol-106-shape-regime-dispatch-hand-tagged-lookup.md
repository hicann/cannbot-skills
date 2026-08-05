---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Specialized AscendC variants per shape regime — workbench dispatcher is a hand-tagged lookup, not a generic detector"
description: "The per-shape-regime dispatch principle is real, but the workbench avg_pool3_d dispatcher is a 6-entry hand-tagged scenario lookup that ValueErrors on any shape outside its curated set (HYBRID: algorithmic REDUCE_D, tag-driven SPLIT_C/W/MULTI_W) — a generic algorithmic dispatcher must be authored separately."
confidence: single_run
original_id: OL-106
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-106, dispatch, shape-regime, pooling]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** implementing N-D pooling, N-D reduce, conv, or any op whose optimal memory-access / parallelism layout depends on shape regime (small-channel vs large-channel, row-parallel vs col-parallel).

### The principle vs the workbench reality

The idea "ship specialized AscendC variants per shape regime + a dispatcher" is sound. But do NOT assume the workbench `avg_pool3_d` archive gives you a reusable **generic shape-regime detector** — empirical inspection shows it is a hand-tagged scenario lookup table.

Inspection of `vendor/workbench_master/archive_tasks/avg_pool3_d/` (br_430 HEAD `7c37033`):
1. `model.py:6-72` defines `SCENARIOS` — a Python list of **exactly 6 dicts**, each with a hard-coded `shape` tuple plus an optional `implementation_mode` tag. Only 3 of 6 (multi_w, split_c, split_w) carry that tag; the other 3 (big_kernel, normal, reduce_d) fall through to default `"auto"`.
2. `model.py:73` builds `SCENARIO_BY_SHAPE = {s["shape"]: s for s in SCENARIOS}` — a 6-key lookup.
3. `model_new_ascendc.py:31-40` `_resolve_scenario` does `SCENARIO_BY_SHAPE.get(shape)`; on a miss it **raises `ValueError("Unsupported avg_pool3_d input shape …")`**.
4. `model_new_ascendc.py:80-103` `_resolve_impl_mode` is **HYBRID**:
   - The `IMPL_REDUCE_D` branch IS algorithmic (checks `k_h==1 && k_w==1 && s_h==1 && s_w==1 && p_h==0 && p_w==0` from the scenario dict).
   - The `IMPL_SPLIT_C` / `IMPL_SPLIT_W` / `IMPL_MULTI_W` branches read `scenario.get("implementation_mode", "auto")` — driven by the lookup table's hand-curated tag, NOT by shape-regime detection logic.
5. Cross-check vs op#26: our op#26 benchmark has **43 unique 5D shapes across 72 cases**, with **ZERO** overlap with the archive's 6 `SCENARIO_BY_SHAPE` keys. Porting the archive verbatim would `ValueError` on every benchmark case.

### Why this matters

The naive framing "5 specialized kernels + a dispatcher that dispatches by shape regime" oversells the dispatcher's robustness: it is brittle by shape, not a generic detector. If you need cross-shape coverage you must author the **algorithmic** dispatch logic yourself; only the `REDUCE_D` branch is genuinely shape-derived, the others are hand-tagged per known shape.

The same archive-specific shape gate is mirrored by the alternate wrapper, so this is an archive-wide adapter pattern, not an AscendC-only issue.

### Validation status

EMPIRICAL-FINDING (downgraded from PENDING-VALIDATION), a3 agent 2026-04-29. Reviewed via `opencode` (Minimax M2.5 local) AND `codex` (gpt-5.5) — both confirmed the amendment accurate and recommended EMPIRICAL-FINDING severity (not retraction). The alternate-wrapper mirror was included in that audit.

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class finalize-gate matrix — 7 finalize-time gates wired in benchmark plugin (PR #125)"
description: "applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class via is_fa_class predicate)"
phenomenon: precision_issue
signal:
  - "applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class via is_fa_class predicate)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-183
timestamp_inferred: true
tags: [ascendc, ol-183]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class via is_fa_class predicate)`
`verified_on: workspace/3_FusionAttention (2026-05-23 e2e smoke)`

**Principle**: kw worker spawning on an FA-class op MUST plan kernel + verification.json output to satisfy 7 finalize-time gates registered by `benchmark.extra_finalize_checks()`. Gates execute through `finalize_pipeline.py` line 1867 plugin-extras dispatch (Phase 2 PR #125, commits f02f1fe8 → 9641b1f3). See `docs/design/FA_CLASS_PROBLEM_SOLUTION_DESIGN.md` §4 for design rationale.

**The 7 gates (each op_class-aware via _is_fa_workspace predicate; non-FA workspaces skip)**:

1. **fa_class_acceptance_fresh_pass_a** (gate 3) — `verification.json.precision.pass_a` mtime must exceed `kernel/*.{h,cpp}` mtime. **Means**: re-run Pass A after every kernel edit. Stale verdicts rejected.
2. **fa_class_regression_baseline** (gate 4) — `precision.pass_a.tier1_pass >= archived_baseline` (default baseline=1 for FA, derived from PARTIAL_PERSIST PR #125-era state). **Means**: never ship fewer passing cases than the archive shipped.
3. **fa_class_scope_out_of_scope_sentinel** (gate 5, DEBT-116) — every `EVAL_ERR` case in `canonical_p2t_summary.json` MUST have an error string containing `_OutOfScope`. **Means**: declare unsupported regimes EXPLICITLY (see OL-184); generic RuntimeError = structural failure.
4. **fa_class_source_contamination** (gate 6, DEBT-117) — if `cann_learn_summary.json` exists, its scanners must show `leak_score=0`, `copy_shape_score<0.05`, `compile_pass_rate=1.0`, `self_review_verdict="PASS"`. **Means**: cann_learner C34a/b/c scanner gates also act as finalize gate.
5. **fa_class_case_variant_map_present** (gate 8) — `case_variant_map.json` artifact present + non-empty cases list. `msprof_unavailable` rows are acceptable.
6. **fa_class_v2_atomicity** (+v2, DEBT-118) — if kernel/*.h contains `IterateAll<false>` (v2 multi-partition marker), archive's `canonical_p2t_summary.results[verdict==PASS_T1]` set MUST be a subset of current's set. **Means**: v2 kernel rewrite cannot regress any v1-passing case.
7. **fa_class_case6_structural_validity** (+case-6, DEBT-119) — `canonical_p2t_summary.results[case=6]` verdict ∈ {`PASS_T1`, `EVAL_ERR with _OutOfScope`}. **Means**: case 6 either passes or fails-gracefully-with-sentinel; unexpected crash = structural failure.

**Coverage gap (acknowledge per Phase C e2e validation 2026-05-23)**: these gates only fire on PARTIAL_PASS or PASS verification.json `status`. For PARTIAL_PERSIST workspaces (current FA state), finalize_pipeline takes a shortcut return BEFORE the extras loop — gates are bypassed. Tracked as DEBT-124 in ROADMAP §6 for main to resolve.

**Application**: When `worker_signal == structural_rewrite_needed` for FA-class kw, plan kernel + outputs against all 7 above. Do NOT discover them at finalize time — pre-plan saves iterations.

**Cross-ref**: OL-159 (FA-class AscendC template-assembly requirement, V220-specific), OL-160 (safety-net name coupling), OL-184 (`_OutOfScope` sentinel convention companion).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-183（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

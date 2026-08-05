---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "[REGRESSION-RISK BLOCKED] Always-depth=4 TQue<VECIN> proposal — would override OL-63 thin-compute carve-out"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec verified_on: n/a (BLOCKED — single-test counter-evidence on uninitialized data) unverified_on: all regression_risk_classifi"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec"
confidence: inferred
status: stub
original_id: CAND-PP86
timestamp_inferred: true
tags: [candidate, inferred, cand-pp86]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=tile-loop-vec`
`verified_on: n/a (BLOCKED — single-test counter-evidence on uninitialized data)`
`unverified_on: all`
`regression_risk_classification: pattern #3 (contradicting evidence-anchored guidance)`
`status: NEEDS_USER_INPUT — do NOT promote without multi-op + msprof + initialized-data evidence`

**Source**: `workspace/regression_risk_test_p0aax/knowledge_update.md` Findings §1
(adversarial fixture for the P0aax regression-risk gate). Demoted by the gate
on 2026-05-07 (run_ts 20260507T081851Z) from a HARD-severity proposal to amend
canonical OL-63.

**Proposed claim** (as submitted): always use `TQue<VECIN, depth=4>` regardless
of per-tile VEC compute weight; remove OL-63's depth=2 thin-compute branch +
the `VEC < 2× MTE2` litmus test.

**Why blocked**: OL-63 currently has 2 multi-op multi-date Evidence rows
(GELU 2026-04-14: depth=4 wins on compute-heavy; DynamicQuant ko-1 iter3
2026-05-02: depth 2→4 **regressed honest mean perf by 7 %** on thin-compute
tile loop) AND a top-of-entry decision rule + measurable litmus. The fixture
offers single-test counter-evidence on uninitialized data (op "frobnicate",
TILE=512, +3 % depth=4 win) — well within typical run-to-run variance for
elementwise tile loops, no msprof, no per-pipe-stage breakdown, no application
of the litmus the OL prescribes. See
`workspace/regression_risk_test_p0aax/kb_scan/regression_risk_20260507T081851Z.md`
for full reasoning.

**Promote when**: ≥3 independent ops (different op classes, different per-tile
compute profiles spanning thin AND heavy) show depth=4 wins, msprof confirms
the wins are pipeline-overlap-driven (not compute-noise), and the
DynamicQuant ko-1 –7 % result is reproduced under depth=4 to demonstrate the
prior measurement was either flaky or environment-specific. Until then, OL-63
remains the authoritative rule.

**Anti-pattern this candidate would re-introduce if accepted prematurely**:
"single-op weak counter-evidence overrides multi-op-evidenced decision rule" —
the kind of premature generalization the regression-risk gate exists to catch.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP86，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

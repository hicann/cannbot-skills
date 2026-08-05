---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "[REGRESSION-RISK BLOCKED] Pure-TBuf-for-dequant-output proposal — would override P-P77 + OL-94 + 6_QuantMatmul Finding #3"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant verified_on: n/a (BLOCKED — single-test \"no corruption observed\" on adversarial fixture) unverified_on: all regression_risk"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant"
confidence: inferred
status: stub
original_id: CAND-PP87
timestamp_inferred: true
tags: [candidate, inferred, cand-pp87]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quant-dequant`
`verified_on: n/a (BLOCKED — single-test "no corruption observed" on adversarial fixture)`
`unverified_on: all`
`regression_risk_classification: pattern #2 (re-introducing anti-pattern) + pattern #3 (contradicting evidence-anchored guidance)`
`status: NEEDS_USER_INPUT — do NOT promote; absence-of-evidence ≠ evidence-of-absence for pipe-ordering bugs`

**Source**: `workspace/regression_risk_test_p0aax/knowledge_update.md` Findings §2
(adversarial fixture). Demoted by the gate on 2026-05-07
(run_ts 20260507T081851Z) from a HARD-severity proposal to archive the
6_QuantMatmul TQue-VECOUT recommendation + add a new P-P stating pure TBuf is
fine for dequant output writes.

**Proposed claim** (as submitted): pure-TBuf-based dequant output write
pipeline is fine; the `TQue<VECOUT>` requirement (P-P77, OL-94, 6_QuantMatmul
Finding #3) is over-cautious; revert to pure TBuf for all dequant pipelines.

**Why blocked**:
- **P-P77** (`patterns/PATTERN_INDEX.md` line 88) explicitly catalogues
  bare-TBuf in this slot as an anti-pattern — op#27 27_MultiMaskAttentionAggregation
  Phase D iter-5 saw 9/10 wrong-output runs with PipeBarrier<PIPE_ALL>; 6/10 wrong
  with extra PipeBarrier<PIPE_V>.
- **OL-94** decision table cites the official AscendC doc:
  `TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作` and
  `EnQue调用会发射同步指令set` — TBuf has no sync mechanism by design.
- **6_QuantMatmul Finding #3** has a concrete per-row probe: even rows = 0,
  odd rows = correct values; rebuilding with TQue<VECOUT> + EnQue/DeQue → all
  64 rows correct. Already merged as evidence (batch 20260507T081021Z).
- The fixture offers single-op "no corruption observed" — but pipe-ordering
  bugs are schedule-sensitive; absence-of-corruption in one configuration
  does NOT generalize. The fixture made no attempt to reproduce the exact
  per-row probe pattern.

See `workspace/regression_risk_test_p0aax/kb_scan/regression_risk_20260507T081851Z.md`
for full reasoning.

**Promote when**: the proposed pure-TBuf pipeline is run against (a) the exact
per-row probe pattern from 6_QuantMatmul Finding #3 with bit-identical PASS,
AND (b) the op#27 27_MultiMaskAttentionAggregation regression case (≥10
independent runs PASS), AND (c) at least one additional dequant-class op
(e.g., 11_DequantSwigluQuant). Until those reproductions exist, P-P77 + OL-94
+ 6_QuantMatmul Finding #3 remain authoritative.

**Anti-pattern this candidate would re-introduce if accepted prematurely**:
the "TBuf as sync-capable queue substitute" misuse that A-P61, OL-94,
P-P77 collectively catalogue. Schedule-sensitive determinism bugs require
multi-run, multi-op, deliberately-adversarial reproduction — not single-pass
"didn't see it" reports.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP87，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

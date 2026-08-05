---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Direction-4 anti-pattern — cast-point fix without reduction-shape pilot"
description: "Anti-pattern: when probe observes \"CANN gets bit-exact 0.0 MERE vs ours has finite drift\" on a REDUCTION output (e.g. grad_weight = (...).sum()), the immediate hypothesis is \"CANN must keep fp32 inter"
phenomenon: build_failure
signal:
  - "Anti-pattern: when probe observes \"CANN gets bit-exact 0.0 MERE vs ours has finite drift\" on a REDUCTION output (e.g. grad_weight = (...).sum()), the immediate"
confidence: inferred
status: stub
original_id: CAND-PP82
timestamp_inferred: true
tags: [candidate, inferred, cand-pp82]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Anti-pattern**: when probe observes "CANN gets bit-exact 0.0 MERE vs ours has finite drift" on a REDUCTION output (e.g. `grad_weight = (...).sum()`), the immediate hypothesis is "CANN must keep fp32 internally and cast at the very end; we're casting too early — emit fp32 + pybind `.to(T)` will close the gap." This is the Direction-4 hypothesis.

**Why it's anti-pattern**: this hypothesis ignores the more likely root cause — CANN's bit-exactness with CPU truth often means CANN reproduces **CPU's reduction SHAPE** (lane-by-lane accumulation order), not that CANN has a different cast point. Our kernel may already be doing fp32-internal-with-cast-at-emit; the gap is in the cross-tile/cross-N reduction shape itself, not the cast point.

**Cost of applying without diagnostic**:
- 2-3 iters of build-verify-revert (per OL-111 measurement risk) consumed
- Cases shift identity, not count (OL-110 fail-floor invariant)
- Opportunity cost: real fix path (reproduce reduction-shape via per-lane accumulator) gets buried

**Recommended diagnostic BEFORE Direction 4**:
Write a probe that bit-compares ours, CANN, and CPU **on the same reduction algorithm** (literal `(go * x_normalized).sum()` element-by-element on identical inputs). Three outcomes:
1. CANN matches CPU bit-exactly when both fp16 → CANN reproduces CPU lane-by-lane order. Direction 4 won't help; need reduction-shape fix.
2. CANN deviates from CPU but matches ours when both fp16 → ours is OL-110 sub-family residual; document and ship at fail-floor.
3. CANN matches CPU only at fp32 emit, deviates at fp16 emit → Direction 4 IS the right fix. Apply with OL-111 pilot.

**Without this diagnostic, Direction 4 is OL-85 reward-hacking** (case-specific overfitting that doesn't address root cause).

**Evidence**:
- 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03): 11 cases match Direction-4-temptation pattern; deferred per OL-111 risk + this anti-pattern recognition.

**Promote when**: 2nd op surfaces this exact temptation pattern AND probe runs the recommended diagnostic AND outcome is documented (any of the 3 paths above). This codifies the diagnostic ritual into a verified pattern.

**Source**: op#14 14_AdaptiveInstanceNormalization2DBackward pp-1 (2026-05-03), workspace/14_adaptive_instance_norm_bwd/knowledge_update.md Candidate 3.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP82，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

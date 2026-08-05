---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "fp32-only geometric CV op — software fp32 sin/cos (not hw Sin/Cos), and audit the L4 \"Subnormal Config\" escalation against operand-reachability before escalating"
description: "applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-geometric/port_a3 (rotated-ROI pooling, affine/rotate warp, any op needing trig of an angle attr) verified_on: roi_align_rotated kw-1"
phenomenon: build_failure
signal:
  - "A fp32-only geometric/CV op needs a transcendental of a geometric attribute — most commonly sin(theta)/cos(theta) of an ROI/box angle, but also rotation matrice"
confidence: inferred
status: stub
original_id: CAND-PORTA3-CV-SOFTWARE-TRIG-L1
timestamp_inferred: true
tags: [candidate, inferred, sin, cos, __simt_callee__, cand-porta3-cv-software-trig-l1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.0; bisheng=n/a; op_class=cv-geometric/port_a3 (rotated-ROI pooling, affine/rotate warp, any op needing trig of an angle attr)`
`verified_on: roi_align_rotated kw-1 2026-06-21 A5/Ascend950PR_9579 (36/36 T1)`
`unverified_on: a3 (Ascend910_V220) — A5 evidence does not auto-transfer`

**Trigger**: A fp32-only geometric/CV op needs a transcendental of a geometric attribute — most commonly `sin(theta)`/`cos(theta)` of an ROI/box angle, but also rotation matrices, polar conversions. There is NO lower-precision dtype to absorb the hw-transcendental fp16-grade error (OL-103: hw `Sin`/`Cos` ≈ fp16 mantissa), and the reference is CPU/fp64 truth at the fp32 T1 floor.

**Recommendation (two coupled rules)**:
1. **Use a software fp32 trig** — a Cephes-style range-reduction + minimax polynomial for `sin`/`cos` in SIMT scalar (`__simt_callee__`), ~1e-7 accuracy. This is the OL-103 software-transcendental play extended from sigmoid/exp to trig. Scalar fp32 `/` in SIMT stays full-precision (no software-reciprocal needed for the divisions threaded through coordinate math — see OL-103 roi_align_rotated evidence). Result: max ours-MERE 1.10e-5 ≪ fp32 T1 floor 1.22e-4.
2. **Audit the L4 "Subnormal Config" escalation against operand-reachability before escalating.** The l1-implementation-guide decision-tree flags Div/Sin/Cos as an L4 (Subnormal Config) signal. But that escalation is only needed if subnormal *operands* are actually reachable. **Geometric/coordinate math (ROI box coords, angles, bilinear weights) never produces subnormal operands** — coordinates are O(1)..O(image-size), angles bounded, weights ∈ [0,1]. So the L4 signal can be **audit-overridden to L1** for this op class. Escalating to L4 on the syntactic Div/Sin/Cos signal alone — without checking reachability — wastes the researcher-route cost (OL-156) on an op that's a clean single-shot L1 SIMT VF kernel.

**Why it's a candidate not yet an OL**: single-op evidence (roi_align_rotated only). Promote to OL once a second fp32-only geometric CV op (e.g. a rotate/affine-warp port) confirms both the software-trig precision win AND the L4-reachability-override.

**Cross-ref**: OL-103 (hw transcendental fp16 floor + software-fp32 mitigation; roi evidence adds Sin/Cos + scalar-`/`), OL-105 (software-fp32 SIMD lowering caveats), OL-150 (SIMT VF one-thread-per-output paradigm), OL-156 (L4 STRUCTURAL escalation signature — what this audit avoids over-firing).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PORTA3-CV-SOFTWARE-TRIG-L1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

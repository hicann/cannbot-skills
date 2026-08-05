---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Two-kernel split (AIC_ONLY + AIV_ONLY via separate ACLRT_LAUNCH_KERNEL on same stream) as PB-34 mitigation Pattern C — empirically broken on V351"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused verified_on: soc=Ascend950PR; cann=9.0.0 Anti-pattern (verified broken): splitting a fused cube+vec op into two separate kernels (A"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused"
confidence: inferred
status: stub
original_id: CAND-PP102
timestamp_inferred: true
tags: [candidate, inferred, cand-pp102]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=all_cube_vec_fused`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern (verified broken): splitting a fused cube+vec op into two separate kernels (AIC_ONLY + AIV_ONLY) launched as two ACLRT_LAUNCH_KERNEL calls on the same stream — proposed as PB-34 manual-CrossCoreSetFlag deadlock mitigation. Empirically: AIV kernel returns ret=0 but writes nothing (matches `13_Cat` silent-no-exec signature). Probe required (constant-write probe to AIV-only entry) before declaring this infrastructure-unsupported vs algorithm-bug.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP102，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "An alternate-oracle / CPU-reference precision probe must gate on the dtype rounding floor with AND, never max_abs ≤ T1 OR mean ≤ T2"
description: "A probe scoring a kernel vs a higher-precision oracle must accept on max_abs ≤ dtype_floor (AND), never a disjunction with a loose T1 or OR-mean escape that green-lights real divergence."
phenomenon: precision_issue
signal:
  - "a CPU/alt-oracle precision probe uses an accept gate like (max_abs ≤ T1) OR (mean ≤ T2) with T1 well above the dtype rounding floor"
confidence: single_run
original_id: OL-203
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, verification-gate, ol-203, dtype-floor, loosened-allclose, anti-cheat]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
A verification probe scores a kernel against a higher-precision alternate oracle (CPU fp32, a second reference impl) and accepts with a disjunction like `(max_abs ≤ T1) OR (mean ≤ T2)`, where `T1` sits well above the dtype rounding floor.

## 根因 / 教训
The accept gate must be `max_abs ≤ dtype_floor` (**AND, single criterion**) — NOT a disjunction with `T1` above the floor. A loose `T1` (or an OR-mean escape hatch) is a **loosened-allclose anti-pattern**: it green-lights real divergence because a large outlier passes under the inflated `max_abs` bound, or the mean washes it out. Dtype rounding floors: **fp16 ≈ 4.9e-4, bf16 ≈ 3.9e-3** (one ULP at the relevant magnitude). The verdict being loose is **independent** of whether the raw numbers happen to land within floor — a verdict produced by a loose gate is not a floor-gated result and must be re-scored.

**Concrete anchor**: FA-A5 hybrid-oracle CPU probe shipped `t1 = (max_abs ≤ 2e-2) OR (mean ≤ 1e-3)` — `2e-2` is 5× the bf16 floor and 40× the fp16 floor. The raw measurements (bf16 2.37e-3, fp16 2.61e-4) DID fall within floor, but the verdict was not floor-gated; fix = `t1 = (max_abs ≤ dtype_floor)`, drop the OR-mean fallback, re-score.

When the oracle is cross-hardware (NPU-output vs CPU-truth), the CPU truth must additionally match the NPU's numerical semantics (fp32-accumulate, same reduction order), not a naive `torch.allclose`.

**Cross-ref**: OL-102/OL-104 (CPU-reference precision regimes); `feedback_hw_floor_label_is_lazy_excuse` (don't excuse divergence as hw-floor without an apples-to-apples probe). Other predicted instances: any op verified against a CPU/alt reference (device-op-can't-run ops, pass-B CPU-truth paths); any probe author tempted to widen tolerance to make a case "pass".

Verified on soc=Ascend950PR, cann=9.0.0 (FA-A5 `3_FusionAttention` D>768 hybrid-oracle probe, 2026-06-02).

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Disambiguate a mode-defaulting reference capture empirically against the captured data, not from kernel source or a cross-project default"
description: "Pick the ported mode of a mode-defaulting capture API empirically — smallest max_abs_diff vs the captured data — not from the device-kernel source or a cross-project archive default."
phenomenon: precision_issue
signal:
  - "The reference/capture API has a mode-selecting default argument (approximate=, rounding mode, upcast toggle) and the op's own device kernel may implement a DIFFERENT mode than the capture actually exercised."
confidence: single_run
original_id: OL-269
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, reference-truth, ol-269, mode-disambiguation, gelu, erfc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Many ops expose more than one numerically-distinct mode selected by a **default-valued argument** on the capture/reference API (an `approximate=` mode, a rounding mode, an upcast/reduction-order toggle). The op's own device kernel may implement a **different** mode than the one the capture invocation actually exercised. So the port target's semantics **cannot** be inferred from (a) reading the device-kernel source, nor (b) trusting a cross-project "PASS-of-record" archive's default.

## 根因 / 教训

Establish the target mode **empirically from the captured truth**: compute each candidate mode on CPU and pick the one whose `max_abs_diff` vs the captured outputs is orders of magnitude smaller. A ~3-order-of-magnitude separation is a decisive disambiguation; a marginal gap means keep probing.

### Concrete anchor (gelu port_a3→a5)

- A3 truth was captured via `torch_npu.npu_gelu(x)` — default `approximate='none'` = exact **erf** GELU `0.5·x·(1+erf(x/√2))`.
- The CANN `Gelu` arch35 device kernel implements the **tanh** approximation — the *other* mode of the same op (not a contradiction).
- Disambiguation on 29 fp32 cases: `worst|a3−erf| = 5.14e-7` vs `worst|a3−tanh| = 4.74e-4` → 3 orders of separation → truth is **erf**.
- Ported as the `Erfc`-DIRECT decomposition `y = 0.5·x·erfc(-x/√2)` (via `1+erf(z) ≡ erfc(-z)`): `Muls(-x/√2) → Erfc → Mul(x) → Muls(0.5)` — 29/29 T1 PASS (worst 5.44e-7 vs cpu64, 7.15e-7 vs a3 hw-erf).
- **Not** the naïve `0.5·x·(1+erf(x/√2))` form: `1+erf` catastrophically cancels in the fp32 negative tail (x<−4, `erf→−1`), ~2.6% rel at x≈−4.75 even with an accurate fp32 `Erf`. See [[OL-271]] for the erfc-vs-1+erf numerical-stability rule.

### Cross-project corollary

The a5_ops PASS-of-record archive for this exact op DEFAULTED its `model_new_ascendc.py` to `approximate='tanh'` and claimed A3 dispatches to tanh — the **opposite** of this engine's empirically-erf edge_dataset. The archive's kernel was both-mode so the *shape* was reusable, but its *default mode* did not match. Lesson: a cross-project PASS-of-record is a useful **shape** reference, but its truth-capture may differ from yours — re-measure your own edge_dataset; do not adopt its default.

### Evidence

- gelu-kw-2 (2026-07-01, Ascend950PR, CANN 9.0.0): erf-vs-tanh disambiguation via 3-order max_abs_diff separation on 29 fp32 cases; ported erf as erfc-direct, 29/29 T1 PASS.

### Other instances (predicted)

Any op whose reference/capture API has a mode-selecting default argument: `approximate=` (GELU and variants), rounding/cast modes, reduction-order / upcast toggles, or fused-vs-decomposed flags. Compute both candidates on CPU against your own captured truth before choosing.

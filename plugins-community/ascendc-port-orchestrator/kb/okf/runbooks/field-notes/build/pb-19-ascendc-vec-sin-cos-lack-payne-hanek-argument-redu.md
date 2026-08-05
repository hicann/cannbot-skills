---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC VEC `Sin()` / `Cos()` lack Payne-Hanek argument reduction — ±inf / huge-error on |x| ≥ 1e10"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "For |x| ≥ 1e10, the primitive produces numerically huge (~1e21 at |x|=1e10) or ±inf values. Reference torch.sin(x) on NPU (dispatches through CANN aclnnSin) pro"
confidence: single_run
original_id: PB-19
timestamp_inferred: true
tags: [false, ascendc, pb-19]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Status**: CONFIRMED (2026-04-22, op#16)
- **Affected**: AscendC SIMD `Sin<T, false>(dst, src, tmp, count)` / `Cos<T, false>(...)` on Ascend950PR / CANN 9.0.0 / bisheng 2026-03-21. Both `false` and `true` (HIGH_PRECISION) modes exhibit the same limit.
- **Symptom**: For `|x| ≥ 1e10`, the primitive produces numerically huge (~1e21 at |x|=1e10) or ±inf values. Reference `torch.sin(x)` on NPU (dispatches through CANN aclnnSin) produces finite, correct values at the same inputs.
- **Root cause**: AscendC VEC trig primitives use a fixed-precision polynomial evaluation after a simple modulo-2π argument reduction that loses precision at large magnitudes. aclnnSin implements proper Payne-Hanek reduction (multi-limb multiply by 2/π with catastrophic-cancellation-resistant carry chain), which our VEC primitive does not.
- **Affected usage**: any kernel using AscendC VEC `Sin()`/`Cos()`/`Tan()` on inputs not pre-constrained to `[-π, π]` (or similar small range).
- **Workaround**:
  - **Domain-scoped ops** (RoPE, sinusoidal PE, etc.): theta values are naturally in `[-π, π]` — primitive works correctly, no action needed. Mark adversarial out-of-domain edge_dataset cases as PARTIAL honestly rather than waiving.
  - **Domain-unconstrained ops**: implement Payne-Hanek reduction in kernel (~50-100 LoC, +compute), or call torch.sin via pybind pre-kernel (not CANN-delegation since it's a different compute phase).
- **Detection**: edge_dataset cases with `dist_large_mag` distribution (|x|~1e29) will expose this. If kernel uses trig primitives AND operational domain is bounded, document as "out-of-domain limit" rather than attempting to fix.
- **Evidence**: op#16 Batched2DRopePositionEncodingBackward edge_dataset 29/31 PASS; failing cases 21/22 (`dist_large_mag_seed{0,1}`, |t|~1e29): kernel output ±inf, torch.sin output finite. 50/50 benchmark cases (realistic RoPE domain) all PASS — confirms primitive is correct in operational range.
- **Related**: EC-35 (AIV libm split) is orthogonal — that's about scalar `cosf`/`sinf` unavailability in SIMT; PB-19 is about SIMD `Cos()`/`Sin()` range limit. Both constrain how trig is implemented on A5.

<!-- 迁移自 porter kb/target/ascendc/（PB-19，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

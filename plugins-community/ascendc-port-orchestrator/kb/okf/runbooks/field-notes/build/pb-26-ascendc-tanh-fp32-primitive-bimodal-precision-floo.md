---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC `Tanh<fp32>` primitive — bimodal precision floor; catastrophic small-x identity loss"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "AscendC::Tanh<float> exhibits a bimodal floor, not the uniform ~1-ULP ceiling previously inferred from end-to-end op#1 GELU measurement."
confidence: single_run
original_id: PB-26
timestamp_inferred: true
tags: [107000, ascendc, pb-26]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **applies_to**: `soc=Ascend950PR_9579; cann=9.0.0 (V100R001C25B046); bisheng=15.0.5+2026-04-13`
- **applies_to**: `soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=15.0.5+2026-01-28`
- **last_verified**: 2026-05-07 (A3 cross-arch confirmation)
- **status**: CONFIRMED on A5 AND A3 — bimodal Tanh floor is **chip-family-wide**, not arch-specific. Bisheng Tanh polynomial doesn't preserve `tanh(x)≈x` near zero on either arch.
- **Severity**: HIGH for kernels that call `Tanh` on inputs near zero (residual norms, KV cache normalize). LOW for transcendental kernels that pre-amplify away from zero (GELU's `0.7978·(x + 0.0447·x³)` pre-amp masks the failure mode).
- **Symptom (measured, both A5 and A3)**: `AscendC::Tanh<float>` exhibits a **bimodal floor**, not the uniform ~1-ULP ceiling previously inferred from end-to-end op#1 GELU measurement.
  - **A5 `|x| ≥ 0.1`**: clean 2-ULP uniform floor across mid / transition / plateau. max_abs_err 1.37e-7. **Saturation band is bit-clean.**
  - **A5 `|x| < 0.1`**: catastrophic. Worst case 1599 ULP at x≈1.7e-4, blowing up to 2.7M ULP at x=1e-7. CPU `numpy.tanh(1e-7)` returns exactly `1e-7` to fp32; NPU returns `1.192e-7`.
  - **A3 `|x| ≥ 0.1`**: max 4 ULP (slightly looser than A5 but same class). Transition band (3..5): max 2 ULP, mean 0.29.
  - **A3 `|x| < 0.1`**: same failure-mode-class — up to 906 ULP in band [1e-4, 0.1].
  - **Joint conclusion**: bimodal floor is bisheng-Tanh-polynomial wide; saturation band is consistently bit-clean (≤4 ULP); near-zero band consistently fails (small-x identity loss). The original OL-103 "saturation is the worst case" framing was wrong on both arches.
- **Root cause hypothesis (unconfirmed)**: bisheng's `Tanh` polynomial implementation lacks a small-x bypass / Taylor-series fallback that established public math libraries (Cephes, fdlibm, libm) use. Abs-err is uniformly tiny (≤1e-7) but expected-output magnitude is ALSO near zero, so ULP measurement at near-0 outputs explodes.
- **Bisheng-version sensitivity (2026-05-07 cross-arch finding)**: A5 bisheng `2026-04-13` is **strictly worse** at small-x (2.7M ULP) than A3 bisheng `2026-01-28` (906 ULP), even though A5 is the newer chip. The newer bisheng build appears to have regressed the Tanh polynomial near zero. Worth re-running A5 probe on next bisheng release to see if this is monotonic. Bisheng build stamp is the load-bearing version field — driver / CANN minor are not.
- **A3-side build note (V220-specific, separate from precision claim)**: V220 (Ascend910_9382 arch) does NOT honor `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` — using it on V220 causes `RegisterAscendBinary 107000`. Probe must use single-block ctypes host launch instead. This is a host-side caller pattern, not a kernel-side precision issue.
- **Workaround**:
  - For GELU and Tanh-class kernels — use sigmoid-form rewrite: `0.5·x·(1+tanh(y)) = x / (1 + exp(-2y))` via `Mul + Axpy + Muls + Exp + Adds + Div`. Confirmed by CANN's own arch35 GELU source (`ops-nn/activation/gelu/op_kernel/arch35/gelu_dag.h` — does NOT use the `Tanh` primitive). See P-P88.
  - For other Tanh-using kernels (residual / KV-cache near-zero paths) — audit input domain. If inputs span `|x| < 0.1`, consider direct `Exp+Add+Div` Cephes-form `tanh(y) = 1 - 2/(exp(2y)+1)` or precondition to amplify away from zero.
- **Detection**: edge_dataset cases with `dist_small_mag` distribution (|x|~1e-30 to 1e-20) or `dist_denormal` will expose this when input crosses through `|x| < 0.1`. If kernel uses `Tanh(...)` AND operational domain includes near-zero, document explicitly.
- **Evidence**:
  - A5 isolated probe `workspace/_probes/tanh_sigmoid_precision_a5_cann9.0.0/PROBE_REPORT.md` (P0aae, agent a57355497e0a0e575, 2026-05-06). Sweep 14,349 fp32 points across `[-10, 10]`. Histogram: `0`→7240, `1`→4180, `2`→795, `4`→792, `8`→599, `16`→377, `≥32`→366. Worst 5 inputs all in `|x|<1e-4` band.
  - A3 isolated probe (DS-side, 2026-05-07) `workspace/_probes/tanh_sigmoid_precision_a3/PROBE_REPORT.md`. 66% bit-exact, 96% within 1 ULP overall. `|x| ∈ [1e-4, 0.1]` band peaks at 906 ULP. Build note: V220 (arch35-only) does not honor `KERNEL_TASK_TYPE_DEFAULT` — use single-block ctypes launch instead (host-side caller pattern, not relevant to the precision claim itself).
- **Cross-reference**:
  - OL-103 §Refined-statement (still mentions inferred Tanh ceiling — to be softened to point at PB-24 once DS A3 data lands; held per user direction "wait for both arches before shipping KB edits").
  - P-P88 (Cephes-form rewrite recommendation; CANN-source-confirmed for GELU; small-x failure mode is the real algorithmic reason, not the previously-imagined saturation cancellation).
  - P0aac (ar_brief Phase R-B step 5 — researcher mandated to consult public math-library literature before declaring "no vendor strategy known"). Closes the harness blind spot that caused op#1 1_GELU iterations to miss this finding.

<!-- 迁移自 porter kb/target/ascendc/（PB-26，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

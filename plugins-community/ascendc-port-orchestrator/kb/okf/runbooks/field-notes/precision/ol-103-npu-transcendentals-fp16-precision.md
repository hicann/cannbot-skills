---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "NPU hardware transcendentals are ~fp16 piecewise-polynomial — kernels using them cannot pass fp32 thresholds vs CPU libm"
description: "NPU hw Exp/Sigmoid/Tanh/Reciprocal/Sqrt/Rsqrt/Log/GeluV2/Div are ~fp16-mantissa piecewise-poly approximations; an fp32 path using them locks MARE at ~2^-9..2^-10, physically unable to pass an fp32 threshold against CPU libm. Use a Tier-1 software fp32 path when output is fp32 vs a CPU ref."
phenomenon: precision_issue
signal:
  - "kernel calls a hw transcendental (Exp/Sigmoid/Tanh/Reciprocal/Sqrt/Rsqrt/Log/GeluV2/Div) in an fp32 path AND the verifier reference is CPU PyTorch (libm)"
  - "kernel MARE stalls at ~2^-9..2^-10 (fp16 magnitude) and will not drop to the fp32 threshold no matter the tuning"
confidence: single_run
original_id: OL-103
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-103, transcendentals, fp32, hardware]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Kernel uses an NPU hw transcendental in an fp32 path AND the output contains fp32 (or is verified at an fp32 threshold), with the reference being CPU PyTorch (libm-based). The kernel's MARE locks at ~fp16 magnitude and never reaches the fp32 threshold.

## 根因 / 教训

NPU 910B Atlas A2 AI Vector Core's hw transcendentals (`Exp`, `Sigmoid`, `Tanh`, `Reciprocal`, `Sqrt`, `Rsqrt`, `Log`, `GeluV2`, `Div`) are implemented as **piecewise polynomial approximations with ~fp16 mantissa precision**. They are NOT fp32-grade. PyTorch CPU's libm equivalents (`expf`, `sigmoidf`, …) are ~0.5 ULP fp32-grade.

So a kernel that calls hw `Exp` in an fp32 path produces fp32 storage with **~10 bits real mantissa, 13+ bits noise**. Comparing against CPU libm at the fp32 MERE threshold (`2^-13`) is **physically impossible to pass** — kernel MARE locks at ~`2^-9` to `2^-10` (fp16 magnitude).

### Three-tier classification

| Tier | Chain feature | NPU-vs-CPU MARE | fp32-threshold result |
|------|---------------|-----------------|-----------------------|
| Tier 1 | Only bit-exact ops (Mul/Add/Sub/Cast/Shift/ReduceMax/Mins/Maxs/Duplicate/ReinterpretCast/Compare-Select/Abs/DataCopy) | ~`2^-22` (near fp32 ULP) | **passes** |
| Tier 2 | Contains a hw transcendental | locked ~`2^-9`..`2^-10` | fp32 UNREACHABLE; fp16 threshold passes |
| Tier 3 | Output is int8/int16 quantized integer | max_abs_diff = 1 LSB (quant step) | strict-equal fails; ±1 LSB tolerance passes |

### Decision tree (reference is CPU fp32)

```
Output dtype int8/int16?
  └── yes → Tier 3 (use hw transcendentals; ±1 LSB absorbs fp16 noise)
  └── no  → Reference type?
            ├── NPU vendor kernel → both ends use NPU activation → cancellable → Tier 2 OK
            └── CPU fp32 / libm   → Output contains fp32?
                                     ├── yes → MUST use a Tier 1 software fp32 path
                                     │         (write sigmoid/exp/1/x via bit-exact ops only)
                                     └── no  → Output absorbed by quant; Tier 2 OK with fp16 thresholds
```

### Tier 1 software fp32 sigmoid (canonical, sketch)

`sigmoid(x) = 1 / (1 + exp(-x))`, all via bit-exact ops:
1. Clamp x to `[-50, 50]`.
2. Software `exp(-x)`: `k = round(-x · …)` (source text truncated in the batch excerpt at the exp reduction step).

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — re-discovered from external workbench KB (`Just-it/AscendOpGenAgent` `skills/ascendc/ascendc-translator/references/dequant_kernel_patterns.md`, br_430). `applies_to_backend: all`.

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 manual-cube operand-load forms COMPILE + run clean on A5 but compute garbage — build-success is NOT A5-validation [V351/A5, port_a3, cube, silent-wrong-result]"
description: "applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0; bisheng=n/a; op_class=all (any cube op that hand-builds L0A/L0B operand loads with V220 fractal addressing)"
phenomenon: build_failure
signal:
  - "a port_a3 cube kernel whose operand load into L0A/L0B is written with a V220 manual fractal-addressing form — either (a) a per-M-fractal LoadData2DParams (V1) i"
confidence: single_run
original_id: PB-43
timestamp_inferred: true
tags: [470000, ascendc, pb-43]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0; bisheng=n/a; op_class=all (any cube op that hand-builds L0A/L0B operand loads with V220 fractal addressing)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (where these forms are CORRECT — this bug is the A5-side non-equivalence, not a V220 bug)`

- **Severity**: HIGH — silent wrong result, NOT a crash or compile error. The kernel builds, the `.so` links, the kernel launches and returns success; the output is simply wrong. Nothing in the build/run signal warns you.
- **Symptom**: a port_a3 cube kernel whose operand load into L0A/L0B is written with a **V220 manual fractal-addressing form** — either (a) a **per-M-fractal `LoadData2DParams` (V1) `i`-loop** (`startIndex=i`, `dst=aL0[C0*i*kAligned]` advancing one 16-row fractal per iteration), or (b) a **3D im2col helper `LoadNzL1ToZzL0A`** with a manual `colC0Stride` — produces RUNTIME GARBAGE on A5. Observed on FlashAttention: attn `max_rel ~470000×`, `sm_max`/`sm_sum` all FAIL. The magnitude is roughly right (K-contraction accumulation IS happening) but the values are wrong (operand fragment rows/stride are arranged wrong in L0).
- **Mechanism**: the V220 cube fractal-addressing semantics (the per-fractal `startIndex` advance + the Zz destination placement) are **not equivalent on the A5 cube**. The same source that is correct on Ascend910/V220 reads operand fragments from the wrong rows/stride on A5 — a layout non-equivalence between arch22 and arch35 cube load paths, not a math/accumulation error.
- **Detection trap (the load-bearing lesson)**: **build-success ≠ A5-validation.** Because there is no compile error and no runtime error, a worker who only checks "did it build / did it run" will wrongly conclude success. The bug is ONLY caught by an actual numerical `pass_a` on A5 hardware. Worse, **controlled-input probes partially mask it**: identity/one-hot operand probes (e.g. K=identity, single-`d` one-hot) often pass or mostly-pass (the wrong-row arrangement happens to coincide for sparse inputs), while **dense random-signed inputs go full-garbage**. Do NOT trust a clean controlled-probe as validation — the arbiter is dense-input pass_a on hardware.
- **Fix**: see **OL-197** (the resolution half) — replace the manual per-fractal / 3D-helper form with the **arch35-native single 2D `LoadData2DParamsV2`** mStep-encoded load (`mStartPosition=0`, `mStep=ceil(M/16)`, `kStep=GetBlockNum<T>(K)`, `srcStride=dstStride=mStep`; no `colC0Stride`); **B-operand `ifTranspose=!isRightTranspose`** (the negate is load-bearing). Convert the `kRemain>0` / `D%BASE_K≠0` tail path too (it carries the same V1 i-loop and stays dormant for D=128 / mult-16 shapes).
- **Evidence**: flash_attention_score port_a3 on Ascend950PR_9579 (2026-05-29) — pre-fix attn 0/8 (garbage `max_rel ~470000×`), sm_max/sm_sum all FAIL; after the OL-197 2D-`LoadData2DParamsV2` rewrite at BOTH cube sites (MM1 QK^T + MM2 PV), attn 8/8 (`max_abs 2.4e-4`), sm_max 8/8, sm_sum 8/8 (origin/main `45fdc7c0`). Root cause was reached after 5+ empirically-refuted hypotheses arbitrated on A5 hardware (zero wrong fixes shipped). Cross-ref `CAND-V220-V351-FA-DIFF-1` (the V220-monolithic vs V351-per-engine structural-port companion).
- **Other instances (predicted)**: any port_a3 cube op that hand-writes a V220 manual `LoadData2D` operand load into L0A/L0B and is "ported" to A5 by only stripping the `__CCE_AICORE__==220` gate without converting the load form — non-FA GEMM-family ports, fused cube+vec ops, backward cube kernels. The general guard: a V220→V351 cube port that compiles is unvalidated until dense-input pass_a runs on A5.

<!-- 迁移自 porter kb/target/ascendc/（PB-43，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

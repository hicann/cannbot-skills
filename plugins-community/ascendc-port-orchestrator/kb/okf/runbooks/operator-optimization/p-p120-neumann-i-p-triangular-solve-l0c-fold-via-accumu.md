---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Neumann / (I+P) triangular-solve L0C-fold via accumulate-mode Mmad (`cmatrixInitVal`): compute `R@(I+P)=R+R@P` in ONE L0C pass with NO separate +I vector-add"
description: "For an a3 (Ascend910_9382, arch22) cube computing an iterative (I+P)-style matmul chain — canonically the Neumann power-product triangular solve A=(I+x)^-1=Π_i(I+x^{2^i}) (each step R_next=R@(I+P), P="
severity: high
confidence: single_run
original_id: P-P120
timestamp_inferred: true
tags: [fa_class, optimization, cmatrixinitval, add, gated_delta_rule, p-p120, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For an a3 (Ascend910_9382, arch22) cube computing an iterative **(I+P)-style matmul chain** — canonically the Neumann power-product triangular solve `A=(I+x)^-1=Π_i(I+x^{2^i})` (each step `R_next=R@(I+P)`, `P=x^{2^i}`) used to invert `(I+strict-lower)` in chunked gated-delta-rule / gated-linear-attention. Naive lowering emits a cube `R@P` PLUS a vector `Add` of the identity/additive term PLUS the MIX barriers between them — ~9 AIV ops for a 5-step chain, pure overhead since the cube can carry the additive term. **Fold on the cube via `MmadParams.cmatrixInitVal`**: (1) first Mmad with `cmatrixInitVal=1` seeds the L0C accumulator with the identity/additive term (`R`, i.e. `R@I`) — NO vector op; (2) second Mmad with `cmatrixInitVal=0` accumulates `R@P` onto the SAME L0C → `R+R@P=R@(I+P)`; (3) hold ONE L0C accumulator across the product chain, Fixpipe only at the end. Same `cmatrixInitVal=(ki==0)` accumulate idiom as the FA K-tile loop (`fa_class/cv_reference_concrete_params.md::matmul_primitive`), repurposed to inject an ALGEBRAIC additive term instead of accumulating K-tiles. **Measured**: cuts AIV ops + barriers to ~6 vs ~9 for a 5-step Neumann chain; precision unchanged (additive term exact in the fp32 L0C accumulator — more faithful than an fp16 round-tripped vector add). Reference STRUCTURE: cv-ref `gated_delta_rule` NeumannSolve. Watch the solve SIGN (CAND-GDR-1) — the fold is sign-agnostic, so the `(I+strict)^-1` vs `(I-strict)^-1` sign bug is orthogonal and handled at the operand level. Full body: `patterns/domains/a3_mix_small_matmul_cube.md`. Cross-ref P-P117 (the `T=(I+strict)^-1` solve site), P-P119 (sibling cube-emit decision), `fa_class/cv_reference_concrete_params.md` (cmatrixInitVal params), CAND-GDR-1. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX/neumann-triangular-solve; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); cross-witness=gated_delta_rule CV-fusion (customer Kimi-K3, 910B2C/220x/CANN8.5.1, 8.85× geomean, PR#200, 2026-07-20); unverified_on: Ascend950PR`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P120，convert_patterns_to_okf.py）。confidence 未升格。 -->

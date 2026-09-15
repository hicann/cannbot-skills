---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Many-small-matmul a3 MIX: emit a HAND-ROLLED single-block cube primitive (`Nd2Nz→LoadData3D→Mmad→Fixpipe[F322F16]`) per matmul, NOT `MatmulImpl::IterateAll` — the library's per-call tiling scalar setup DOMINATES (aic_scalar-bound, not MAC-bound)"
description: "For an a3 (Ascend910_9382, arch22) MIX chunked-recurrence / gated-linear-attention op (P-P117) where EACH chunk performs MANY SMALL matmuls (M,N,K ≤ 128, one L0C tile, no inner K-loop; a gated_delta_r"
severity: high
confidence: single_run
original_id: P-P119
timestamp_inferred: true
tags: [fa_class, optimization, nd2nz, mmad, matmul_primitive, p-p119, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For an a3 (Ascend910_9382, arch22) MIX **chunked-recurrence / gated-linear-attention** op (P-P117) where EACH chunk performs MANY SMALL matmuls (`M,N,K ≤ 128`, one L0C tile, no inner K-loop; a gated_delta_rule chunk runs ~8+ such matmuls × Nc chunks × heads → hundreds-to-thousands of tiny matmuls per op). `MatmulImpl::IterateAll<sync=true>` (P-P68 NON-KFC cube) is correct for a FEW LARGE GEMMs, but for MANY SMALL matmuls its per-call tiling/address scalar setup is fixed-cost and DOMINATES the tiny `≤128³` MAC work → the cube becomes **aic_scalar-bound**. **msprof evidence** (a3/Ascend910_9382, GDR fwd, IterateAll version): `aic_scalar ≈32%` / `aic_mac ≈3.7%` / `cube_util ≈17%` (~9× more time in scalar tiling setup than actual MACs — the per-call-overhead signature). **Fix**: emit a hand-rolled single-block cube per matmul — `Nd2Nz` (GM/UB→L1, ND→NZ) → `LoadData/LoadData3D` (L1→L0A/L0B) → `Mmad` (single pass, ≤128³) → `Fixpipe[F322F16]` (L0C fp32→fp16). Block dims compile-time-known → NO tiling scalar setup per call. Reuse the exact `matmul_primitive` param rules from `fa_class/cv_reference_concrete_params.md` (Mmad 4-arg accumulate + per-context `Fixpipe srcStride`: `/C0` for L0C→workspace, ELEMENTS for L0C→GM; wrong unit → 507015 ECC read) — do NOT re-derive. **Measured** (device, DS 2026-07-20): this change ALONE closed a ~1.3× compute-bound deficit vs the cv-reference to PARITY (T4096 −35%, T1024 −30% device-time), precision unchanged (still 16/16 @ fp64). **Decision rule**: few-large → IterateAll (P-P68); many-small (≤128, hundreds+, chunked-recurrence) → hand-rolled primitive, confirm via msprof `aic_scalar ≫ aic_mac`. Precision-invariant across both (same Mmad math) — a PERF lever, never correctness. Full body: `patterns/domains/a3_mix_small_matmul_cube.md`. Cross-ref P-P117 (the many-small call-site), P-P116 (a3 MIX sync template), P-P68 (the IterateAll it replaces for many-small / keeps for few-large), `fa_class/cv_reference_concrete_params.md` (matmul_primitive params), CAND-GDR-3. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX/many-small-matmul; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P119，convert_patterns_to_okf.py）。confidence 未升格。 -->

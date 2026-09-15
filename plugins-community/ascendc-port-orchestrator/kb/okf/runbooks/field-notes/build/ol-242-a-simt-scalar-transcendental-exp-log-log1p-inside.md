---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A SIMT-scalar transcendental (exp/log/log1p inside a `__simt_callee__` / VF context) has a DIRECT scalar intrinsic shipped in the CANN binary — call it, do NOT hand-fit a range-reduce + Horner polynomial"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV-SIMT; op_class=all (any SIMT-scalar op needing exp/log/softplus/sigmoid)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV-SIMT; op_class=all (any SIMT-scalar op needing exp/log/softplus/sigmoid)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-242
timestamp_inferred: true
tags: [__simt_callee__, ascendc, ol-242]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV-SIMT; op_class=all (any SIMT-scalar op needing exp/log/softplus/sigmoid)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 family — A5 evidence does not transfer automatically; the simt_api header set is arch/CANN-version specific, re-check before relying on A3)`

**Principle**: For per-element scalar transcendentals (`exp` / `log` / `log1p`, and by extension `softplus` / `sigmoid` built on them) consumed INSIDE a `__simt_callee__` / SIMT-VF (`__simt_vf__`) context, the CANN binary ships DIRECT fp32-grade scalar intrinsics. Use those library calls; do NOT hand-fit a Cody-Waite range-reduce + Horner polynomial. The fitted-polynomial route is only justified for the VECTOR transcendentals — hardware vector `Exp`/`Ln` on `LocalTensor<T>` are fp16-grade (see OL-103 §per-primitive precision table) and may genuinely need a Cephes-form rewrite (P-P88) — but that vector-precision reasoning does NOT carry to the scalar path, which already has an accurate library.

**Concrete anchor**: `expf` / `logf` / `log1pf` are declared `__simt_callee__` in `cann-9.1.T500/.../asc/include/simt_api/math_functions.h` (pulled via `device_types.h:27`). On-device A/B vs a hand-fitted `SsfExp` (Cody-Waite + Horner deg-8) / `SsfLog` (atanh series): the library calls agree to ~6e-7 (fp32-grade) AND are not slower (perf −1.8% fp32 / −1.1% bf16). A 60-line fitted polynomial was replaced by 3 library calls (kept behind `-DSSF_USE_FITTED_TRANSCENDENTALS` as fallback). The replace-decision met all 3 gates: (exists in binary) + (accurate vs fp64) + (not slower).

**Distinct from EC-35** (do not conflate the two paths): EC-35 documents `expf`/`logf` resolving to libm names (`cosf`, `expf`, …) that are UNDEFINED at the AIV-vector device-object LINK step (`ld.lld -m aicorelinux`) for `KERNEL_TYPE_AIV_ONLY` binaries — that is the AIV-VECTOR-kernel link path, whose fix is the SIMT→GM→SIMD split. OL-242 is the SEPARATE `simt_api` `__simt_callee__` SCALAR path: there the intrinsics ARE declared and DO link inside a VF callee. So "transcendentals don't link in AIV" (EC-35, vector path) and "transcendentals DO exist as SIMT scalar intrinsics" (OL-242, scalar path) are both true on different paths — check which path your callee is on before choosing.

## Evidence
- selective_scan_source_a5 fwd_simt ② (2026-06-22, A5/Ascend950PR_957b, CANN 9.1.T500): on-device A/B, fitted `SsfExp`/`SsfLog` vs `expf`/`logf`/`log1pf` library calls — agree ~6e-7, perf −1.8% (fp32) / −1.1% (bf16); whitebox-derived (on-device probe, not guessed). KB-gap that caused the original hand-fit: the LANGUAGE_REFERENCE had no SIMT-scalar exp/log entry, so devs fitted a polynomial from scratch.

## Other instances (predicted)
Any SIMT-scalar op needing exp/log/softplus/sigmoid in a `__simt_callee__`/VF callee — scalar-path softmax, GELU/SiLU scalar variants, layernorm scalar reductions, any RNN/SSM gate. Before fitting a polynomial for ANY scalar transcendental, grep `simt_api/math_functions.h` for the `f`-suffixed name.

Cross-ref: [[EC-35]] (the DISTINCT AIV-vector libm-link path — read the distinction above), [[EC-74]] (`__simt_callee__` marker requirement for VF callees on CANN ≥9.1.T500), OL-103 (vector-primitive precision floor — why the VECTOR path may still need a rewrite), P-P88 (Cephes-form rewrite — applies to the vector path, NOT this scalar path).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-242（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

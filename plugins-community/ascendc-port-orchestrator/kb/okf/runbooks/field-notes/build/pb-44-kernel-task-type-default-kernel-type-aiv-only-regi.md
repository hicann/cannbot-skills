---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` → `RegisterAscendBinary aiv 107000` register-FAIL on A5/950PR under the CANN 9.0.0 toolkit; CANN 9.1.T500 registers clean — A5-safety is toolkit-version-gated (refines PB-28) [V351/A5, cann-version, kernel-registration, refines-PB-28]"
description: "applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0 (107000 register-FAIL) / 9.1.T500 (clean); macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)"
phenomenon: build_failure
signal:
  - "an A5/950PR kernel using KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) builds clean but launch fails RegisterAscendBinary aiv ret 107000 (program register fail"
confidence: single_run
original_id: PB-44
timestamp_inferred: true
tags: [107000, 507034, 507015, ascendc, pb-44]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (V351 / A5); cann=9.0.0 (107000 register-FAIL) / 9.1.T500 (clean); macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0 (107000) AND cann=9.1.T500 (matched) — clean single-variable (CANN-only) A/B`

- **Severity**: HIGH — build/link/compile PASS (bisheng OK), but launch-time `RegisterAscendBinary aiv ret 107000` → program register failed → Status FAIL. Kernel never registers/runs; end-to-end blocked. No degraded-mode workaround under 9.0.0.
- **Status**: CONFIRMED 2026-06-16 by a clean single-variable A/B on npu_dev3 (.171); both arm logs read independently, same-kernel md5 independently verified.
- **Refines PB-28**: PB-28 states this macro is "arch35-only ... does NOT apply on Ascend950PR (where the macro is the canonical entry-form)." That A5-safety is **toolkit-version-gated**: under the **CANN 9.0.0** toolkit on 950PR the macro STILL fails registration with `107000`; under **CANN 9.1.T500** it registers cleanly (the canonical-on-A5 behaviour PB-28 describes). So "legal/canonical on arch35" is NOT the same as "the 9.0.0 toolkit's 950PR registration loader can register it." No conflict with PB-28 — its A5-clean claim holds for 9.1.T500.
- **Symptom**: an A5/950PR kernel using `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` builds clean but launch fails `RegisterAscendBinary aiv ret 107000` (program register failed) when built against the CANN 9.0.0 toolkit. A downstream `507034` vector-core-timeout may follow — it is a **consequence** of the failed registration, NOT an independent cause. Rebuilding the SAME kernel against CANN 9.1.T500 → registers, runs, all cases matched.
- **Evidence (clean single-variable A/B, main-verified 2026-06-16)**: same `4_Abs` AIV_ONLY kernel (`abs_kernels.cpp` md5 `1ff22805c8b6e52bff3f2c288cc47bfc`, md5 independently verified), fixed SOC `Ascend950PR_957b`, identical nsenter/env, ONLY the CANN toolkit flipped:
  - CANN 9.0.0 → `RegisterAscendBinary aiv ret 107000` → register failed → Status FAIL (+ downstream `507034`).
  - CANN 9.1.T500 → all cases matched, MERE=0 / MARE=0 (50/50).
  - Arm logs: `/tmp/ab2_v_cann-*.log` on npu_dev3 (.171).
  - Surfacing: primary repro by the independent reviewer (agent-open); the version-direction lead came from the back agent noting its FAG `flash_attention_grad` AIV_ONLY kernels built/ran clean under 9.1.T500 (which pointed at the toolkit version). Note: that FAG observation only evidences "AIV_ONLY registers under 9.1.T500"; it is NOT part of the causal A/B (which is the single-kernel CANN flip above).
- **Fix / workaround**: build AIV_ONLY-macro kernels on A5/950PR against **CANN 9.1.T500** (present in the npu_dev3 container). If pinned to the 9.0.0 toolkit, fall back to the bare `__global__ __aicore__` entry-form (per PB-28's V220 fix) and re-verify.
- **Detection**: build PASS + launch-time `RegisterAscendBinary aiv ret 107000` on A5 → check the CANN toolkit version; if 9.0.0, rebuild against 9.1.T500.
- **Cross-ref**: PB-28 (same macro + `107000` signature on V220; THIS entry refines its "A5-safe" claim to toolkit-version-gated), PB-40 (`RegisterAscendBinary mix ret 107000` for MIX multi-entry on V220 — different trigger, same error code). DISTINCT from the GDN regbase `507015` aicore-trap (MIX cube/vector CrossCore sync on 9.1.T500) — that is a different failure mode, not this; see **PB-45**.

<!-- 迁移自 porter kb/target/ascendc/（PB-44，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

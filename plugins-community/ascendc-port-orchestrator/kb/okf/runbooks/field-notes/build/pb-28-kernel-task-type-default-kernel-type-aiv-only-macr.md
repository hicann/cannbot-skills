---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` macro is arch35-only — `RegisterAscendBinary 107000` on V220 [V220]"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel source uses KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY) as the entry-form macro (the canonical A5 pattern in many op#X archives). Build succeeds, but"
confidence: single_run
original_id: PB-28
timestamp_inferred: true
tags: [107000, ascendc, pb-28]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Severity**: HIGH (build succeeds but launch fails; kernel emits AIC `.o` without runnable binary; no degraded-mode workaround)
- **Status**: CONFIRMED 2026-05-07/2026-05-08 DS A3 cold-starts — multiple ops (4_Abs, 22_Nonzero, 5_Cumsum) hit this on every kernel that started from an A5 archive copy
- **applies_to**: soc=Ascend910_9382 (V220 single-die — A2/A3); does NOT apply on Ascend950PR (A5/V351/arch35 where the macro is the canonical entry-form)
- **Symptom**: Kernel source uses `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` as the entry-form macro (the canonical A5 pattern in many op#X archives). Build succeeds, but launch fails with `RegisterAscendBinary 107000` on V220. The macro expands to arch35-only entry-attribute metadata that V220's kernel registration loader rejects.
- **Fix**: On V220 use the bare `__global__ __aicore__ void <kernel>(args)` entry-form, no macro wrapper:
  ```cpp
  // V220 (A2/A3) — bare entry, no KERNEL_TASK_TYPE_DEFAULT
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, GM_ADDR y, ...) {
      // body
  }

  // A5 (Ascend950PR/V351/arch35) — KERNEL_TASK_TYPE_DEFAULT canonical
  extern "C" __global__ __aicore__ void my_kernel(GM_ADDR x, GM_ADDR y, ...) {
      KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
      // body
  }
  ```
- **Detection**: build PASS but launch-time `RegisterAscendBinary 107000` is the smoking gun. Pre-build grep: `grep -E "KERNEL_TASK_TYPE_DEFAULT\(KERNEL_TYPE_AIV_ONLY\)" workspace/<op>/kernel/*.{cpp,h}` — if any hit AND TARGET ∈ {a3, a2, a3-ds, a2-ds}, rewrite to bare form before build. (Note: grep MUST include the `KERNEL_TYPE_AIV_ONLY` argument — see scope-clarification below.)
- **Scope is `KERNEL_TYPE_AIV_ONLY` ONLY — do NOT generalize**: this entry covers the `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` macro specifically. Other `KERNEL_TYPE_*` variants are NOT subject to the same arch35 restriction:
  - `KERNEL_TYPE_MIX_AIC_1_2` is **valid on V220** (verified by CANN's own `flash_attention_score` arch22 source — see `patterns/unverified/candidates.md` CAND-FA1). Wrapping a mixed cube+vec entry in `#if __NPU_ARCH__ >= 3510` because "PB-28 says it's arch35-only" is the wrong inference — the historical kw-3 hard-hang behind that defensive guard was NOT a `MIX_AIC_1_2` register-binary failure; it was the `MatmulImpl<> + manual CrossCoreSet/Wait` deadlock now codified as [PB-34](#pb-34-matmulimpl-with-manual-crosscoresetwaitflag-mix_aic_1_2-deadlock-on-v220).
  - `KERNEL_TYPE_AIC_ONLY` and `KERNEL_TYPE_MIX_AIC_1_1` are out of scope of PB-28 — no V220 register-binary evidence either way; if you encounter `107000` with those, file a separate PB entry.
  - **Anti-pattern caught 2026-05-21** (3_FusionAttention kw-1): kernel comment "PB-28: KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) is arch35-only" — false attribution. PB-28 never claimed that.
- **Evidence**: Promoted to PB-28 2026-05-09 from DS-local `src/scripts/env_quirks_a3-ds.json` quirk #4 (DS-flagged repeat hit across 4_Abs / 22_Nonzero / 5_Cumsum cold-starts). Pattern was the cross-arch issue of porting an A5 kernel to A3 without rewriting the entry-form. **All confirmed instances used `KERNEL_TYPE_AIV_ONLY`** — no `MIX_AIC_*` instance has ever produced `RegisterAscendBinary 107000`.
- **Positive-side confirmation (arch35 canonical form works)**: recurrent_gated_delta_rule kw-1 (2026-06-18, A5 Ascend950PR_957b / arch35, CANN 9.1.T500): a genuinely vec-only recurrent-decode kernel using `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` built + launched + ran 30/30 clean on arch35 with no RegisterAscendBinary rejection — confirms the macro is the canonical entry-form on A5 (the 107000 rejection is V220-only, exactly as scoped above). The port_a3 brief §2b V220-reject note does NOT apply on arch35.
- **Cross-ref**: `hardware/target/ascend910c.md` § Kernel-launch (V220 A3 entry form), `hardware/target/ascend910b.md` § Kernel-launch (V220 A2 entry form, same family), `env_quirks_a3-ds.json` quirk #4 (DS env preflight catalog), `patterns/unverified/candidates.md` CAND-FA1 (`MIX_AIC_1_2` V220 source-derived evidence), PB-34 (the actual mixed-mode failure mode kw-3 hit, mis-blamed on PB-28).

<!-- 迁移自 porter kb/target/ascendc/（PB-28，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

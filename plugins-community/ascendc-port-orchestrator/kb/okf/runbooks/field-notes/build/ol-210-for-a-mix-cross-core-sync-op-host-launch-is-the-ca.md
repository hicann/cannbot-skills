---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "For a MIX cross-core-sync op, host/launch is the cascade differentiator — vendor FA = single-op-aclnn + crossCoreSync:1 + global flags + no-cascade, so isolation is necessary-not-sufficient"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/cube-MIX (MIX_AIC_1_2 cross-core-sync)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/cube-MIX (MIX_AIC_1_2 cross-core-sync)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-210
timestamp_inferred: true
tags: [161002, ascendc, ol-210]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=attention-fwd/cube-MIX (MIX_AIC_1_2 cross-core-sync)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (FA-A5 graybox 2026-06-04/05)`

FA-A5-specific characterizations that transfer to the FA-class designer/triage + future framework-launch orchestration (these are NOT a cascade root — the cascade is a documented unreproducible residual `DEBT-FA-CASCADE-UNREPRO`, ROADMAP §6, commit `3f201374`):
- **vendor production FA = single-op `aclnnFlashAttentionScore`** (torch_npu `npu_fusion_attention` dispatches OpApi/aclnn, NOT GE-graph) + kernel.json `crossCoreSync:1` + `kernelType MIX_AIC` + `taskRation 1:2` — identical metadata to our op-build. Vendor has **GLOBAL cross-group flags** (same as ours) and does **NOT** cascade → `crossCoreSync:1`/flag-isolation is necessary-not-sufficient and is NOT the cascade mechanism.
- **byte-identical kernel + framework-launch ≠ custom-`<<<>>>`-launch**: our device kernel is a faithful byte-copy of vendor's (`regbase_buffer.h` md5 `f792bb24`); the one cascade fire was under our custom `<<<>>>` launch. With the kernel identical, the differentiator is **HOST/LAUNCH**. fix-DIRECTION = adopt the framework launch ("copy the host" — see `feedback_faithful_port_copy_host_not_just_kernel`).
- **`CalcTschBlockDim`** = the TSCH-encoded blockDim host-tiling value (exposed `PlatformAscendC` method, `platform_ascendc.h:125` — call it, do NOT copy-inline vendor arithmetic). Vendor encodes the AIC/AIV ratio into blockDim; raw `SetBlockDim` does not. (Characterization only — not the fix.)
- **A5 GE production host = closed built-in TBE op-store**; a custom op cannot be loaded into it locally (read-only mount) → "our custom-op build under CANN's real GE production host" is NOT locally reproducible (E1). The accessible route is single-op aclnn (= framework launch), gated on `rc=161002` (ACLNN-param-invalid at execute; static-diff LEAN = a fixable marshalling slip, path-likely-alive).

**Evidence**: FA-A5 graybox 2026-06-04/05 (DS measurements + independent prototype static-diff + main grounding), committed in `DEBT-FA-CASCADE-UNREPRO`. Cross-ref `fa_class/` designer KB (`patterns/domains/fa_class_template.md`, `fa_class/cross_core_sync.md`), OL-206 (the cross-core RESULT-handshake hand-roll ladder), OL-207/208/209 (the methodology that bounded this investigation), `feedback_faithful_port_copy_host_not_just_kernel`.

**Hardware-grounding of the "global cross-group flags" fact (whitepaper 2026-06-07)**: the Ascend950 whitepaper §4.4 (STARS2.0) states the hardware sync-flag space is **up to 128K single-bit OR up to 4096 32-bit multi-bit flags** for inter-task-stream synchronization. This means any per-op "flagId range 0–10" (or a narrower "0–5") observed at the `CrossCoreSetFlag`/`CrossCoreWaitFlag` API surface is a **CANN-API / per-Group allocation convention, NOT a hardware ceiling** — the silicon has orders of magnitude more flags. Operationally this CONFIRMS the OL-210 observation that vendor FA and our op both run on the same GLOBAL cross-group flag pool (the 128K/4096 space is chip-wide, partitioned by STARS Group, not per-AIC-private), so flag-id collision/isolation is a software-allocation question, not a hardware-scarcity one. See OL-211 (STARS2.0) for the scheduler-level mechanism.

**Other instances (predicted)**: any MIX_AIC_1_2 cross-core-sync op (attention fwd/bwd, fused-norm+matmul, MoE finalize); the FA-class template-route optimizer (audits build-pipeline / host-faithfulness / block-connection per DEBT-143, not kernel-rewrite).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-210（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

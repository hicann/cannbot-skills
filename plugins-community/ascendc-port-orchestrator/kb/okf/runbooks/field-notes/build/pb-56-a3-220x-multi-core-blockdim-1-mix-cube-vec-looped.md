---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "a3/220x multi-core (blockDim>1) MIX cube↔vec LOOPED bidirectional handshake deadlocks (507014)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "a hand-rolled per-core cube↔vec handshake in the LOOPED bidirectional form hangs: torch.npu.synchronize() blocks >90s, no aicore exception, prints kernel-launch"
confidence: single_run
original_id: PB-56
timestamp_inferred: true
tags: [999998, 507015, ascendc, pb-56]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
  soc: Ascend910_9382
  cann: 9.0.0
```
- **Status**: RESOLVED (2026-07-20, device-measured — multi-core FA ACHIEVED, see Resolution). This PB documents that a **HAND-ROLLED** per-core cube↔vec cross-core LOOPED ring deadlocks (mode3/mode5) AND — the key point — that FA multi-core does NOT need it. NOT a hardware wall; NOT "perf N/A" (a3 vendor baseline = `npu_fusion_attention`, measured running ~102µs).
- **Affected**: only **hand-rolled** multi-core (blockDim>1) MIX 1:1 cube↔vec handshakes on a3/220x (`CrossCoreSetFlag`/GM-slot per-core). Single-core (blockDim=1) hand-rolled handshake is fine (works + verified 17/17). The library/KFC multi-core path is NOT affected — it does not hand-roll.
- **Symptom**: a **hand-rolled** per-core cube↔vec handshake in the **LOOPED bidirectional** form hangs: `torch.npu.synchronize()` blocks >90s, no aicore exception, prints kernel-launch line then no result. Device latches AICore=100% (recovers on process-kill / >120s idle). **1-shot (non-looped)** concurrent multi-group MIX runs clean.
- **Root cause (device-measured, by elimination)**: NOT distinct-vs-shared flag-id orphan-credit — BOTH distinct-id (mode3) AND the shipped c220 `pto::TSYNC_CVID` shared-flag+GM-slot recipe (mode5, **per-round** sync) deadlock. §4's arch35 mode-4 resolution does NOT port (`INTRA_MODE=4` collapses to mode-0 on c220 via `mode & 0x3` mask). Deeper than flag-id disambiguation. See `fa_class/cross_core_sync.md` §5 for the 4-mode device table. **The deeper lesson: do not hand-roll multi-core cube↔vec sync — use the library.**
- **Resolution (device-measured 2026-07-20 — multi-core FA ACHIEVED)**: extend the VERIFIED single-core base (library `MatmulImpl IterateAll<sync=true>` + per-pair PB-55 handshake) to blockDim>1, ONE head-slice per core, keeping ONLY the per-pair `MODE2`/`CV_CORE_SYNC` handshake — NO cross-core ring. **Runs DEADLOCK-FREE, precision 20/20** (5 shapes × cores{1,2,4,20}, cos 0.999998+, deterministic, bit-identical to 1-core), perf **0.186× vendor** (550µs vs `npu_fusion_attention` ~102µs @ 20 cores; recovered 14.3× over single-core 7866µs). **Key insight: FA multi-core is per-head-INDEPENDENT** — each core is a self-contained single-core FA, `MODE2` scopes the handshake to that core's OWN AIC↔AIV group (zero cross-core interaction), so the LOOPED cross-core ring (mode3/mode5 deadlock) is UNNECESSARY for head-parallel FA. Perf gap is STRUCTURAL (non-flash: materializes full [S,S] score+prob to GM per head + row-serial softmax vs vendor's fused online-softmax L1/UB-resident); next lever = flash/online-softmax rework (cf independent generated-kernel witness 0.51×), NOT a sync fix. NOTE: the *async*-KFC path (`matmul::Matmul`+KfcServer) remains a3-standalone-BLOCKED (CAND-KFC-standalone-bootstrap-teardown / PB-53 / PB-54) — the `IterateAll<sync=true>` route sidesteps it. Artifacts (in-repo, disk-verifiable): `kb/okf/reference/porter/patterns/` — `fa_a3_multicore_result.md` + `fa_a3_multicore_mc_verify.json` (20/20) + `fa_a3_multicore_mc_perf.json`. See `cross_core_sync.md` §5.
- **DETECTION methodology (stomp-probe)**: to distinguish a real multi-core deadlock from a kernel bug, build a minimal blockDim=2 handshake probe with 4 modes (1-shot distinct / 1-shot shared / looped distinct / looped shared+GM-slot). Run the CANDIDATE mode FIRST on a **verified-clean** device (a hang LATCHES the device — budget for it), and the known-deadlock mode LAST as control. Verify device-clean (`npu-smi` AICore≈0%, no proc, no zombie) before EACH run; a hang is a valid negative verdict; do NOT hammer a latched device (>120s idle to recover). Write the run-log incrementally to disk (the probe can die on API error — disk is the salvage).
- **large-D note (507015 refuted for the device-side head-loop)**: the 507015 fault class (single-core, D>128, N>1) is for hand-rolled **HOST** multi-head tiling; a **device-side head-loop** kernel (`MatmulImpl IterateAll<sync=true>` per head, internal K-tiling) does NOT hit it — D=256/384/512/768/1024/1280 all computed + matched golden (cos≥0.99998) at single-core on a3/220x.
- **Evidence**: DS device build+probes 2026-07-19/20 → `kb/okf/reference/porter/patterns/fa_a3_multicore_gmslot_run_log.md` (hand-rolled mode 0/1/3/5 characterization) + the multi-core fa_a3_multicore_{result.md,mc_verify.json,mc_perf.json} in that dir. Grounded in c220 `pto/npu/a2a3` TSyncCVID.hpp / TSync_Custom.hpp / TPush.hpp. Cross-ref: `fa_class/cross_core_sync.md` §5 (sibling to the §4 V351/A5 RUNNABLE handshake), PB-55 (single-core both-AIV-subblocks-must-Set).

<!-- 迁移自 porter kb/target/ascendc/（PB-56，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

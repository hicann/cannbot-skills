---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`MIX_AIC_1_2` AIC↔AIV cross-core handshake is DIRECTION-ASYMMETRIC — the reverse (AIV→AIC) flag is per-subblock-COUNTED, not broadcast → a single-setter reverse handshake DEADLOCKS [220x/Ascend910_9382, mixed-mode-sync]"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TYPE_MIX_AIC_1_2"
phenomenon: build_failure
signal:
  - "a multi-stage KERNEL_TYPE_MIX_AIC_1_2 kernel (e.g. a 2-cube + softmax FlashAttention core) HANGS forever — host torch.npu.synchronize() times out / process exit"
confidence: single_run
original_id: PB-55
timestamp_inferred: true
tags: [999999, 507014, 507015, 507035, mix_aic_1_2, ascendc, pb-55]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: soc=Ascend910_9382 (Atlas A3 / 220x single-die); cann=9.0.0; 2026-07-18 — DS hand-authored FA core (S=Q@Kᵀ cube#1 → softmax vector fp32-accum → O=P@V cube#2; both matmuls MatmulImpl::IterateAll<sync=true>) on a3 device. Cosine 0.999999 vs pure-torch attention golden; allclose(rtol=1e-2,atol=1e-3) 100% PASS; deterministic across 3 fresh processes (bit-identical). Distinct flag ids per handshake; never flag id 0 (PB-35).`
`unverified_on: soc=Ascend950PR_9579 (V351 / A5) — NOT tested; the asymmetry may differ on arch35, do NOT claim it there.`
`status: DEVICE-CONFIRMED + FIX DEVICE-VALIDATED 2026-07-18 (a3 / Ascend910_9382 / CANN 9.0.0). Bisection-pinned single-setter deadlock; the both-subblocks-signal fix passes precision + determinism.` **[cann corrected 9.1.0 → 9.0.0 on 2026-07-18: the container is CANN 9.0.0 (`version.info` Version=9.0.0; no 9.1.0 exists); device re-confirmed the MIX builds+runs+precision-passes on 9.0.0. The earlier `9.1.0` was an unverified (propagated) label — it made an op-gen worker wrongly decline the MIX on a false version-mismatch. `verified_on ⊆ applies_to` both now 9.0.0.]**

- **Severity**: HIGH (a PURE wait-deadlock — host-side timeout / process exit 124 at `torch.npu.synchronize()`, with **NO fault code in a fresh plog**; trivially misdiagnosed as an algorithm/precision bug or blamed on the 507014/507015/507035 FAULT family of PB-34 which it is NOT).
- **Symptom**: a multi-stage `KERNEL_TYPE_MIX_AIC_1_2` kernel (e.g. a 2-cube + softmax FlashAttention core) HANGS forever — host `torch.npu.synchronize()` times out / process exits 124 — with a CLEAN plog (no `LaunchAscendKernel` error code, no aicore exception). Occurs specifically when the REVERSE `AIV→AIC` handshake flag is raised from only ONE AIV subblock of the 1:2 pair.
- **Mechanism (bisection-pinned)**: in `MIX_AIC_1_2` (1 AIC : 2 AIV) the `CrossCore` flag handshake is DIRECTION-asymmetric:
  - **Forward `AIC→AIV`** (e.g. `FLAG_S`): **BROADCAST** — one AIC `CrossCoreSetFlag<MODE2, PIPE_FIX>(flag)` releases BOTH AIV subblocks' `CrossCoreWaitFlag`. (Verified by the bisect variant: cube#1 + forward handshake + softmax, reverse handshake removed → SYNC_OK.)
  - **Reverse `AIV→AIC`** (e.g. `FLAG_P`): **per-subblock-COUNTED, NOT broadcast** — the single AIC `CrossCoreWaitFlag(flag)` requires a `CrossCoreSetFlag` from EVERY AIV subblock of the 1:2 pair. If only subblock 0 sets it, the AIC blocks FOREVER. (Verified by the bisect variant: reverse handshake kept but single-setter, cube#2 removed → DEADLOCK, exit 124, no fault code.)
- **Fix (device-validated)**: BOTH AIV subblocks must `CrossCoreSetFlag(reverse_flag)`. The single logical WRITER of the shared buffer (e.g. subblock 0 writes P) still writes ALONE; both subblocks merely SIGNAL the reverse flag. **Mode-2 multi-setter suffices — §4 / mode-4 (1:1 per-subblock-disjoint-ids) is NOT required for this** particular reverse-direction count.
  ```cpp
  // AIV side — reverse AIV→AIC handshake (FLAG_P). BOTH subblocks must SIGNAL:
  if (subBlockIdx == 0) {
      // subblock 0 is the sole WRITER of the shared P buffer …
      CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);   // signal
  } else {
      CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);   // subblock 1 signals too (no write)
  }
  // AIC side: a SINGLE CrossCoreWaitFlag(FLAG_P) now unblocks (count = 2 reached).
  ```
- **Detection** (pre-build static guard):
  ```bash
  # Warn: a MIX_AIC_1_2 kernel with an AIV→AIC reverse CrossCoreWaitFlag on AIC but
  # only ONE CrossCoreSetFlag(<reverse_flag>) reachable on the AIV side → count never reached.
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      grep -q "KERNEL_TYPE_MIX_AIC_1_2" "$f" || continue
      revflags=$(grep -oE "CrossCoreWaitFlag(<[^>]*>)?\(([A-Za-z0-9_]+)\)" "$f")
      # for each reverse flag the AIC waits on, count how many CrossCoreSetFlag(flag) the AIV emits;
      # a single-setter under MODE2 for a reverse flag is the PB-55 smell.
      echo "$f: inspect reverse-flag setter multiplicity"
  done
  ```
- **Companion minimal-MIX witness (same device, same date)**: a 1-cube + 1-vec MIX runs on a3 via `MatmulImpl` `sync=true` + a SINGLE canonical `CrossCore` handshake (the `add_lora` pattern) + the standard `aclrtlaunch` runtime — which ALREADY supplies FFTS via `rtGetC2cCtrlAddr` (`rtGetC2cCtrlAddr` → `main_kernel<<<blockDim, nullptr, stream>>>(..., fftsAddr)`). So the "host FFTS gap" is NOT a gap for the standard framework build; the multi-stage cross-sync half is what PB-55 resolves.
- **Cross-reference**: **PB-34** (`MatmulImpl<>` + manual `CrossCore` FFTS sync-slot COLLISION → 507014/507015/507035 FAULTS — a DIFFERENT mechanism; PB-55 is a PURE wait-deadlock from handshake mis-COUNTING, and it uses `IterateAll<sync=true>` which AVOIDS PB-34's async-KFC slot-collision path). **PB-35** (`event_t(0)` / flag-id-0 collision — PB-55 uses distinct flag ids per handshake and never id 0). `CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP` + **DEBT-222** (PB-55 is the DEVICE-MEASURED resolution of the multi-stage-cross-sync half of that gap). `fa_class/cross_core_sync.md` §4 (the mode-4 / disjoint-id cross-core recipe — related but NOT required for this reverse-direction count).

<!-- 迁移自 porter kb/target/ascendc/（PB-55，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

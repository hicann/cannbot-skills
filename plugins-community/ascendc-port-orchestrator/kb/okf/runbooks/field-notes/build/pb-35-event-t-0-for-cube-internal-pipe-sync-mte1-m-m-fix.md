---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`event_t(0)` for cube-internal pipe sync (`MTE1_M` / `M_FIX` / `MTE2_MTE1`) collides with AIC↔AIV CrossCoreSetFlag `FLAG_CANON_DONE` chain in `MIX_AIC_1_2` mode → silent hang [V220, mixed-mode-sync]"
description: "applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)"
phenomenon: build_failure
signal:
  - "Mixed cube+vec kernel using Pattern A (tile-MMAD primitives + manual CrossCoreSetFlag<0x2>(FLAG_X) chain at flag IDs 0..7). Cube tile body adds pipe-sync events"
confidence: single_run
original_id: PB-35
timestamp_inferred: true
tags: [mte1_m, m_fix, mte2_mte1, flag_canon_done, mix_aic_1_2, ascendc, pb-35]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
`verified_on: a5_ops:3_FusionAttention kw-4 cycle 3 (run buksn5pky 2026-05-21T09:00Z) — Pattern A tile-MMAD primitives + SetFlag/WaitFlag MTE1_M with event_t(0) → kernel enqueues, torch.npu.synchronize() hangs past 90s timeout, no fault thrown`
`unverified_on: Ascend950PR_9579 (V351 / A5) — probe_a5_v300_fa_sync 2026-05-23 Pattern C probe was malformed (cross-core HardEvent semantics instead of intra-AIC cube-internal pipe sync); produced a hang for unrelated reason. However, Pattern A on the same V351/CANN combo demonstrated that MIX_AIC_1_2 + MatmulImpl<> + cross-core flag chain works cleanly — weakening but not definitively closing PB-35 for V351. Follow-up probe needed (intra-AIC SetFlag<HardEvent::M_FIX>(event_t(4..7)) cube primitive-decomp on V351). Cross-ref: workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md`
`confirmed_on: Ascend950PR_9579 (V351 / A5) — kw-gb2 hermetic graybox 2026-06-03 — CONFIRMED for the USER-OWNED-cube + HAND-ROLLED-cross-core-flags case. A cube-MIX FA built from canonical KB (user-owned Mmad tile primitives + hand-rolled CrossCoreSetFlag<0x2>/WaitFlag chain, NOT library matmul) DEADLOCKED at runtime: torch.npu.synchronize() hangs, no aicore exception. Scope clarification of the "weakens PB-35" note above: the prior negative-evidence (probe_a5_v300_fa_sync Pattern A runs clean) applies ONLY to the LIBRARY-matmul path (MatmulImpl<> + the matmul API's own internal cross-core sync). It does NOT cover the user-owned-Mmad + user-hand-rolled-flag case, which DOES deadlock on V351. Root cause of the hand-roll deadlock is now identified (see cross_core_sync.md §4 RUNNABLE): the hand-roll used SYNC MODE 2 (1:2 ratio) + a SHARED flag id for both AIV sub-blocks; the working wholeport uses MODE 4 (1:1, AIV0/AIV1 individually triggerable) + DISJOINT per-sub-block flag ids (id and id+16). The fix is public-API runnable.`

- **Severity**: HIGH (silent hang; no error code; only symptom is sync timeout — easy to misdiagnose as algorithm bug rather than sync collision)
- **Symptom**: Mixed cube+vec kernel using Pattern A (tile-MMAD primitives + manual `CrossCoreSetFlag<0x2>(FLAG_X)` chain at flag IDs 0..7). Cube tile body adds pipe-sync events via `SetFlag<HardEvent::MTE1_M>(event_t(0))` / `WaitFlag<HardEvent::MTE1_M>(event_t(0))` between LoadData and Mmad. Build clean. Kernel enqueue succeeds (`kernel(...)` returns). `torch.npu.synchronize()` then hangs past timeout. No `LaunchAscendKernel` error code, no aicore exception.
- **Root cause**: In `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` mode, low event IDs (0, 1) are shared between the AIC↔AIV cross-core `CrossCoreSetFlag<0x2>(flagId)` chain AND the cube-internal hardware-pipe `SetFlag<HardEvent::X>(event_t(N))` events. Using `event_t(0)` for cube-internal pipe sync collides with the cross-core flag ID 0 (typically `FLAG_CANON_DONE` in FA-class kernels): the cube's MTE1→M wait blocks on a counter that the AIV's CrossCoreSetFlag<0x2>(0) is also feeding, but with incompatible producer/consumer semantics. Result: deadlock with no observable error state.
- **Fix**: Use distinct event IDs ≥ 4 for cube-internal pipe sync. Reserve IDs 0..3 for cross-core flags (the canonical FA-class chain `FLAG_CANON_DONE=0` / `FLAG_MM1_DONE=1` / `FLAG_SOFTMAX_DONE=2` / `FLAG_MM2_DONE=3`). Practical scheme for Pattern A FA tile:
  ```cpp
  // Cross-core flags (reserved IDs 0..3 — used in fused_kernels.cpp top-level):
  constexpr int32_t FLAG_CANON_DONE   = 0;
  constexpr int32_t FLAG_MM1_DONE     = 1;
  constexpr int32_t FLAG_SOFTMAX_DONE = 2;
  constexpr int32_t FLAG_MM2_DONE     = 3;

  // Cube-internal pipe sync (use IDs ≥ 4):
  AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(event_t(4));
  AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(event_t(4));
  AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(event_t(5));
  AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(event_t(5));
  AscendC::SetFlag<AscendC::HardEvent::M_FIX>(event_t(6));
  AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(event_t(6));
  ```
  Per `ascend950pr.md` § Cross-core sync: user-owned flag ID range is `0..7` (FFTS_MAX_FLAG=7); reserved barrier IDs at `8..10`. Cube-internal pipe events and cross-core flags share that range, so they MUST be allocated disjointly.
- **Detection** (pre-build static guard):
  ```bash
  # Hard-warn: a kernel file using BOTH CrossCoreSetFlag<0x2>(0..3) AND SetFlag<HardEvent::*>(event_t(0..3))
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      cross_core_ids=$(grep -oE "CrossCore(Set|Wait)Flag<0x2[^>]*>\([0-9]+\)" "$f" | grep -oE "\([0-9]+\)" | tr -d "()" | sort -u)
      pipe_ids=$(grep -oE "(Set|Wait)Flag<.*HardEvent::[A-Z_]+>\(event_t\([0-9]+\)" "$f" | grep -oE "event_t\([0-9]+" | grep -oE "[0-9]+" | sort -u)
      overlap=$(comm -12 <(echo "$cross_core_ids") <(echo "$pipe_ids"))
      [ -n "$overlap" ] && echo "PB-35 violation $f: shared IDs $overlap"
  done
  ```
- **Anti-pattern (DO NOT EMIT)**:
  ```cpp
  // BAD — event_t(0) collides with FLAG_CANON_DONE (= 0) in cross-core chain
  // ... AIV side issues: CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_CANON_DONE);  // ID 0
  // ... AIC side does:
  DataCopy(l1A, qGm, Nd2NzParams{1, S, D, 0, D, D/16, 16, 0});
  SetFlag<HardEvent::MTE2_MTE1>(event_t(0));  // ← COLLIDES with cross-core ID 0
  WaitFlag<HardEvent::MTE2_MTE1>(event_t(0)); // ← hangs forever
  LoadData(l0A, l1A, LoadData2DParams{...});
  ```
- **Evidence**:
  - 3_FusionAttention kw-4 cycle 3 (`buksn5pky` 2026-05-21T09:00Z): Pattern A tile-MMAD primitives with corrected `Nd2NzParams` shape. Phase 1 (no pipe sync) → fault `0x8000004000` (L0B read/write conflict, sync genuinely missing). Phase 2 (added pipe sync at `event_t(0)`) → silent hang at `torch.npu.synchronize()` past 90s. No fault thrown. Cost: $10.47 / 30min — Phase 2 hang was the directly-observed evidence for this PB.
  - 3_FusionAttention `fusion_attention_fused_kernels.cpp` (existing kw-1 baseline): cross-core flag IDs `FLAG_CANON_DONE=0`, `FLAG_MM1_DONE=1`, `FLAG_SOFTMAX_DONE=2`, `FLAG_MM2_DONE=3` are all in the 0..3 range that the cube-internal pipe sync at `event_t(0)` would collide with.
  - **3_FusionAttention kw-5 cycle (iter 4, 2026-05-21T~14:00Z) — "Use IDs ≥ 4" Fix EMPIRICALLY FALSIFIED**: tested 3 distinct sync schemes — (a) raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 (distinct IDs ≥ 2, dodging cross-core 0/1); (b) `GetTPipePtr()->FetchEventID(HardEvent::X)` runtime-allocated (canonical pattern). BOTH produce the same silent-hang signature as `event_t(0)` — kernel enqueues cleanly, `torch.npu.synchronize()` hangs past 45s with no aicore exception. The "low IDs collide with cross-core" hypothesis (and thus the Fix proposal of "use IDs ≥ 4") has been EXPERIMENTALLY DISPROVEN at the case_3 [4,64,512] BSH fp16 head=8 shape. The deadlock is at a deeper layer than event-ID allocation — root cause hypotheses now open for fo: (1) `CrossCoreSetFlag<0x2>` chain imposes barriers that collide with `HardEvent::M_FIX` regardless of event ID (cross-core semantics are uniform per HardEvent class); (2) `MIX_AIC_1_2` requires uniform Cross* sync across ALL pipe events on the cube side, not local SetFlag/WaitFlag mixed with CrossCoreSetFlag for cross-core; (3) FFTSCNT mailbox semantics may prevent any cube tile-MMAD with internal sync regardless of event ID scheme. **Implication for Fix section above**: the "Use distinct event IDs ≥ 4" Fix is a HYPOTHESIS that closes the visible `event_t(0)` collision but does NOT close the actual deadlock; the deeper sync-discipline question is fo-scope. Pattern A on V220 MIX_AIC_1_2 with user-owned cube-internal pipe sync remains UNSOLVED in canonical KB. Mitigation: stay on AIV-only VEC fallback for fp16 FA on V220 until canonical V220 cube workflow lands (likely via `aog-cann-learner` Mode 5 extraction of CANN ops_transformer arch22 `flash_attention_score` kernel structure).
  - **RESOLVED for V351 cross-core direction (2026-06-03, cann-learn Mode 5)**: the V351/A5 cube↔vec cross-core deadlock (the `MIX 1:1` AIC↔AIV handshake, distinct from the intra-AIC cube-internal pipe sync this PB is named for) is now closed as a RUNNABLE public-API pattern in `fa_class/cross_core_sync.md` §4. The deadlock-avoiding handshake is: SYNC MODE 4 (1:1, AIV0/AIV1 individually triggerable) + per-sub-block disjoint flag ids (`id` and `id+16`) + direction-pinned literal pipe + Set-after-own-write. Verdict: PUBLIC-API-runnable — the FA whole-port's working sync is customer-reproducible by hand (no privileged vendor class required). NOTE this resolves the **cross-core** edge; the separate **intra-AIC cube-internal** pipe-sync deadlock (the `event_t` collision this PB is primarily about) is still governed by the falsification above when user-owned tile-MMAD is used.
- **Cross-reference**: PB-34 (the other end of the cube+vec sync minefield — Matmul library vs user-owned flags); CAND-FA1 (Pattern A recommendation; pre-PB-35 anchor in CAND-FA1 used `event_t(0)` in pseudocode — needs amendment to `event_t(4..7)` after PR lands); `ascend950pr.md` § Cross-core sync (user-owned ID range `0..7`, reserved barrier IDs at `8..10`); 3_FusionAttention workspace knowledge_update.md Finding 14 + Finding 16; `patterns/unverified/candidates.md` CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP (the open-hypothesis follow-up candidate).

<!-- 迁移自 porter kb/target/ascendc/（PB-35，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

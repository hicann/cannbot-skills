---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V→S sync after `Adds(dst_ub, scratch_ub, 0.0f, count)` when next iter's S pipe writes the same `scratch_ub` via `SetValue`"
description: "Source: op27 27_MultiMaskAttentionAggregation a3 V220 (2026-04-28) — aog-kernel-worker iter-5 secondary fix Validation status: 1 op observed; the fix WAS applied and KEPT, but its independent contribu"
phenomenon: build_failure
signal:
  - "(op#27 narrative):"
confidence: inferred
status: stub
original_id: CAND-PP76
timestamp_inferred: true
tags: [candidate, inferred, scratch_ub, setvalue, local_task, finalf, cand-pp76]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: op#27 27_MultiMaskAttentionAggregation a3 V220 (2026-04-28) — aog-kernel-worker iter-5 secondary fix

**Validation status**: 1 op observed; the fix WAS applied and KEPT, but its independent contribution is **inconclusive** because the dominant cause was found later to be stale `LOCAL_TASK` staging (OL-93). The pattern is plausible from first principles but its empirical signature was contaminated by the staging gap.

**Symptom** (op#27 narrative):
- Inner cls loop populates a small UB scratch buffer per class via `finalF.SetValue(cls, scalar)` (S pipe write)
- Then `S→V sync` + `Adds(finalT, finalF, 0.0f, C_pad)` (V pipe read of `finalF`)
- Next iter starts with `finalF.SetValue(cls, scalar)` again (S pipe write of `finalF` while iter-N's V read MAY still be in flight)
- Without an explicit `SetFlag<HardEvent::V_S> + WaitFlag<HardEvent::V_S>` AFTER the Adds, iter N+1's S write can race iter N's still-in-flight V read

**Proposed fix**:
```cpp
Adds<float>(finalT, finalF, 0.0f, C_pad);
event_t evvs = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(evvs);
WaitFlag<HardEvent::V_S>(evvs);
// (iter boundary; next iter's SetValue(finalF, ...) is now safe)
```

**Why "candidate" not promoted**: op#27's full-PASS state was achieved by `cp workspace → LOCAL_TASK + rebuild` (per OL-93). Pre-rebuild, with this V→S fix already in the on-disk kernel, the kernel still showed ~10 % intermittent precision mismatch — i.e. this fix did NOT resolve the symptom in isolation. We cannot tell whether (a) the V→S race is real but the build was running stale code that lacked it, OR (b) the V→S race is theoretical-only and never fires in practice on V220. Distinguishing requires a controlled probe: build kernel W/ vs W/O the V→S sync from a freshly-staged `LOCAL_TASK`, observe det count over N runs.

**Promotion criteria**:
1. Reproduce the race on a separate op (≥2 ops total) where staging is provably clean (`diff workspace/{op}/kernel LOCAL_TASK/kernel` empty before each verify)
2. Show that omitting the V→S flag produces non-determinism that adding it eliminates (binary contrast on identical staging)
3. Document the specific S/V/MTE pattern that triggers it (probably: scalar fan-in into a UB scratch reused across iter boundaries)

**Until promotion**: workers MAY apply this fix as defensive practice when the access pattern matches (per-iter S-pipe scalar fan-in into a UB region read by a later V op then re-written next iter), but should NOT claim it resolves an intermittent non-determinism — run OL-93 staging diff first.

**Related**: P-P74 (TBuf→TQue auto-sync — the more general/effective pattern; V→S sync is a targeted complement when the racing buffer is filled by S pipe rather than rotated by a queue), OL-93 (the staging-gap red herring this candidate is conjugate to), A-P61 (determinism anti-patterns).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP76，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

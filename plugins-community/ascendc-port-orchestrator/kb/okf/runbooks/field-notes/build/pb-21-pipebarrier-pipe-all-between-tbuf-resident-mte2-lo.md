---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`PipeBarrier<PIPE_ALL>()` between TBuf-resident MTE2 load and V compute → silent crash 507015 (V220-confirmed)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "kernel using a manual TBuf pipeline pattern (TBuf<VECCALC> bufA_, bufB_; + DataCopyPad → PipeBarrier<PIPE_ALL>() → VEC compute) crashes at runtime with aclrtLau"
confidence: single_run
original_id: PB-21
timestamp_inferred: true
tags: [507015, ascendc, pb-21]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL
- **Status**: OPEN (CANN 9.0.0)
- **Affected hardware**: Atlas A3 (V220 / Ascend910_93) confirmed. A5 / Ascend950PR — UNVERIFIED (likely affected given V220→A5 inheritance of most VEC pipeline behavior; re-run a minimal repro on A5 before broadening this entry's scope).
- **Symptom**: kernel using a manual TBuf pipeline pattern (`TBuf<VECCALC> bufA_, bufB_;` + `DataCopyPad → PipeBarrier<PIPE_ALL>() → VEC compute`) crashes at runtime with `aclrtLaunchKernel` returning error code **507015**. No exception, no Python traceback — kernel silently terminates after launch and the host blocks at the next sync. aicpu / aiv logs show MTE2→V handoff aborted mid-pipeline.
- **Trigger pattern**: TBuf (NOT TQue) pipeline mixed with `PipeBarrier<PIPE_ALL>()` as the synchronization primitive between MTE2 load and V compute. The bug fires reliably when ALL of:
  - All loads/computes run on a single `TBuf<VECCALC>` (no `TQue<VECIN>` queue rotation)
  - Sync between MTE2 (DataCopy) and V (Cast/Mul/Add) is via `PipeBarrier<PIPE_ALL>()` rather than explicit `SetFlag<HardEvent::MTE2_V>(eventId) + WaitFlag<HardEvent::MTE2_V>(eventId)`
  - Loop body has ≥ 2 iterations of MTE2→V on the same TBuf
- **Fix**: replace `PipeBarrier<PIPE_ALL>()` with explicit `SetFlag<HardEvent::MTE2_V>(eventId) + WaitFlag<HardEvent::MTE2_V>(eventId)` (and analogous `V_MTE3` between V compute and DataCopy back to GM). The event ID is fetched via `uint16_t ev = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);` once per kernel and reused across loop iterations. Reference template: `patterns/domains/platform_compat.md` §"Manual TBuf pipeline with explicit event sync" (P-P70).
- **Why it happens (hypothesis, unconfirmed)**: V220 `PipeBarrier<PIPE_ALL>()` semantics on TBuf (vs TQue) appear to skip MTE2→V completion guarantees. TQue `<VECIN, depth=2>` has hardware-managed queue rotation that includes the implicit barrier; TBuf does not. CANN docs do not call this out explicitly; CANN's own kernels using TBuf consistently use explicit event sync.
- **Decision rule** (when to use TBuf+manual sync vs TQue auto-rotation): see OL-94.
- **Evidence**: op#27 `27_MultiMaskAttentionAggregation` a3 V220 cold-start (2026-04-28) — worker initial impl used `TBuf + PipeBarrier<PIPE_ALL>()` per natural CANN-style port → silent crash 507015 across all cases. Five compile/precision iters wasted before probe identified the sync primitive as the culprit. Switched to explicit `SetFlag<HardEvent::MTE2_V>/WaitFlag<HardEvent::MTE2_V>` → 50/50 PASS, det 100/100. Probe report: `output/npukernelbench-a3/src/kernels/27_MultiMaskAttentionAggregation/probe_report.md` (a3 PR #2 v2 archive).
- **Cross-reference**: F-P4 (PipeBarrier alignment) covers a different PipeBarrier failure mode (alignment); PB-21 is specifically the TBuf+PIPE_ALL combo. PB-9 (UB→UB DataCopy on V220) is another V220-only sync nuance. OL-94 has the broader "when to pick which sync mechanism" decision rule.

<!-- 迁移自 porter kb/target/ascendc/（PB-21，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

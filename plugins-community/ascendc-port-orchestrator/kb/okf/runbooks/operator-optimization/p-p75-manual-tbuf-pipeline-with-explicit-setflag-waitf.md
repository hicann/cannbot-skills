---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Manual TBuf pipeline with explicit `SetFlag/WaitFlag<MTE2_V>` event sync (V220-confirmed; A5 likely-applicable)"
description: "### Trigger Pure-VEC pipeline kernel (DataCopy in → VEC compute → DataCopy out, repeated per row/tile) where the chosen UB-resident primitive is TBuf<VECCALC> rather than TQue<VECIN/VECOUT>. See OL-94"
severity: high
confidence: single_run
original_id: P-P75
timestamp_inferred: true
tags: [platform_compat, optimization, fetcheventid, setflag, waitflag, datacopypad, s_v, p-p75, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

### Trigger
Pure-VEC pipeline kernel (DataCopy in → VEC compute → DataCopy out, repeated per row/tile) where the chosen UB-resident primitive is `TBuf<VECCALC>` rather than `TQue<VECIN/VECOUT>`. See OL-94 for the decision rule on which primitive to pick; this pattern is for the TBuf branch.

### Why this template exists
The natural CANN-style port using `TBuf + PipeBarrier<PIPE_ALL>()` triggers PB-21 (silent crash 507015) on V220 — `PipeBarrier<PIPE_ALL>()` does NOT carry MTE2→V completion guarantees on TBuf-resident pipelines. Explicit `SetFlag/WaitFlag` is the only safe pattern. Worker sessions across two model classes (the A5 backend on op#27 and DS V4 on a similar fused op) hit this trap; this template + PB-21 + OL-94 close the loop.

### Pattern (concrete piece — copy and adapt, not a full kernel)

```cpp
#include "kernel_operator.h"
using namespace AscendC;

class MyTBufKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, /* tiling args */) {
        gmX_.SetGlobalBuffer((__gm__ T*)x, /*size*/);
        gmY_.SetGlobalBuffer((__gm__ T*)y, /*size*/);
        // Allocate UB buffers (TBuf, NOT TQue):
        pipe_.InitBuffer(bufA_, /*per-row bytes*/);
        pipe_.InitBuffer(bufB_, /*per-row bytes*/);
        // Fetch event IDs ONCE per kernel (do NOT re-fetch per iter):
        evMte2V_ = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        evVMte3_ = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    }

    __aicore__ inline void Process() {
        for (int r = 0; r < numRows_; ++r) {
            LocalTensor<T> a = bufA_.Get<T>();
            LocalTensor<T> b = bufB_.Get<T>();

            // Stage 1: MTE2 — load row r
            DataCopyPad(a, gmX_[r * H_], copyParams_, padParams_);
            SetFlag<HardEvent::MTE2_V>(evMte2V_);     // mark MTE2 done
            WaitFlag<HardEvent::MTE2_V>(evMte2V_);    // V waits for MTE2

            // Stage 2: V — compute (Cast / Mul / Add / etc.)
            Cast(b, a, RoundMode::CAST_NONE, H_);
            // ... more VEC ops on a / b ...
            SetFlag<HardEvent::V_MTE3>(evVMte3_);     // mark V done
            WaitFlag<HardEvent::V_MTE3>(evVMte3_);    // MTE3 waits for V

            // Stage 3: MTE3 — store row r
            DataCopy(gmY_[r * H_], b, H_);
        }
    }

private:
    GlobalTensor<T> gmX_, gmY_;
    TPipe pipe_;
    TBuf<TPosition::VECCALC> bufA_, bufB_;
    uint16_t evMte2V_, evVMte3_;
    int numRows_, H_;
    DataCopyExtParams copyParams_;
    DataCopyPadExtParams<T> padParams_;
};
```

### Critical rules
1. **Fetch event IDs ONCE** outside the loop (`FetchEventID` allocates from a finite pool — re-fetching per iter exhausts it).
2. **Pair every `SetFlag` with a `WaitFlag`** at the next stage boundary. Missing pair → silent stall or race.
3. **Do NOT use `PipeBarrier<PIPE_ALL>()`** between MTE2 and V on TBuf — PB-21 (silent crash 507015 on V220).
4. If two TBufs participate in the same MTE2→V handoff (e.g. loading both `a` and `b` per row before compute), use ONE `SetFlag<MTE2_V>` after both `DataCopyPad` calls + ONE `WaitFlag<MTE2_V>`. The barrier is per-stage, not per-buffer.
5. For multi-stage VEC compute that includes scalar dependencies, intermediate `S_V` / `V_S` events may be needed (see PB-9 / PB-20 for the scalar-pipe-on-V220 nuances).

### Anti-pattern (compiles, runs, silent crash 507015 on V220)
```cpp
DataCopyPad(a, gmX_[r * H_], cp, padParams_);
PipeBarrier<PIPE_ALL>();    // ❌ does NOT guarantee MTE2_V on V220 TBuf
Cast(b, a, RoundMode::CAST_NONE, H_);  // V op fires before MTE2 complete → garbage / crash
```
Replace with the `SetFlag/WaitFlag<MTE2_V>` pair shown above.

### When NOT to apply (use TQue instead)
- Standard pointwise / cast / strided copy chains where dataflow fits a 3-stage pipeline cleanly — TQue's auto-rotation is cheaper to author and equivalent in perf.
- See OL-94 decision table for the full pick-list.

### Evidence
- **op#27 `27_MultiMaskAttentionAggregation` a3 V220** (2026-04-28): worker initial impl with `TBuf + PipeBarrier<PIPE_ALL>()` → silent crash 507015 across all cases, 5 iters wasted. Switched to this `SetFlag/WaitFlag<MTE2_V>` pattern → 50/50 PASS, det 100/100. Probe report at `output/npukernelbench-a3/src/kernels/27_MultiMaskAttentionAggregation/probe_report.md` (a3 PR #2 v2).
- **DS V4 worker session** (2026-04-28): weaker model defaulted to TQue on an op needing TBuf (multi-buffer aliasing across phases). Crashed at runtime; recovered by switching to TBuf + this pattern after 5 iters. Surfaced the gap that prompted P-P75 + OL-94 + PB-21 codification.

### Cross-reference
- **OL-94**: when to pick TQue vs TBuf (decision rule + table).
- **PB-21**: the specific silent-crash-507015 trap this pattern avoids.
- **PB-9**: V220 UB→UB DataCopy nuance (different sync issue).
- **P-P28** (TQue<4> auto pipeline): when TQue is the right pick — TQue auto-rotation replaces the manual sync pattern when dataflow fits.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P75，convert_patterns_to_okf.py）。confidence 未升格。 -->

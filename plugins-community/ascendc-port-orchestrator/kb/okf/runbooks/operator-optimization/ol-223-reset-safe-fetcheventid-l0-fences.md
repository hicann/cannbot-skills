---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "In a Reset-bearing multi-stage MIX cube kernel, drive cube-internal L0 pipe sync with Reset-safe dynamic back-to-back FetchEventID fences — NOT persistent Buffer<> credits, NOT hardcoded event ids"
description: "Inside a Reset-bearing multi-stage MIX cube, drive cube-internal L0 fences with self-contained back-to-back FetchEventID Set+Wait — never persistent Buffer<> credits or hardcoded event ids."
confidence: single_run
original_id: OL-223
classified_by: llm-assisted
timestamp_inferred: true
tags: [sync-correctness, optimization, ol-223, fetcheventid, reset, user-cube, mix]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR (V351/A5, arch35) / CANN 9.1.T500 / multi-stage MIX AIC+AIV user-cube. Verified on Ascend950PR_957b. On V220 the intra-AIC user-cube MIX case remains open (PB-35 shows even FetchEventID-allocated intra-AIC fences still hung there); this OL is the V351/arch35 resolution.

**Principle**: when a user-owned (hand-rolled, non-library) cube matmul lives inside a multi-stage MIX kernel that calls `pipe_->Reset()` between stages (which frees the GLOBAL event pool — see PB-45), every cube-internal hardware-pipe fence (MTE2→MTE1, MTE1→M, M→FIX, FIX→MTE2) must be a **self-contained, back-to-back `FetchEventID` Set+Wait** that allocates and consumes its id within the single `Run` call. Carry NO sync state across the Reset boundary.

**Decision rule (which sync model for a cube inside a Reset loop)**:
- Each `Run` fully self-contained on the AIC (GM→L1→L0A/B→Mmad→Fixpipe→GM in one call) → use dynamic back-to-back fences. Nothing to overlap survives the Reset, so dropping `Buffer<>` double-buffering costs only intra-call pipeline overlap (perf), not correctness.
- The persistent `Buffer<>` credit model (priming `SetFlag<EventC2P>` at Init) is correct ONLY when Init is called ONCE for the kernel lifetime (e.g. a library `MatmulImpl` before the stage loop — the GDN light-port path, OL-220), NOT per-stage after a Reset.
- NEVER hardcode a literal `EVENT_ID0..N` for a cube fence — it aliases the surrounding code's managed `FetchEventID`/`AllocEventID` ids (→ the `507015` trap, PB-45 failure mode B).
- Keep the AIC↔AIV `CrossCoreSetFlag` workspace-GM handshake SEPARATE from these intra-AIC fences; never reuse a cross-core flag id for an intra-AIC fence. The working FA whole-port maps the same way: WorkspaceQueue ring = cross-core, `Buffer` L0 events = intra-AIC.

**Concrete anchor** (Reset-safe fence — dynamic id, no persistence):
```cpp
template <HardEvent EVT>
__aicore__ inline void RbSetWaitFlag() {
    event_t e = GetTPipePtr()->FetchEventID(EVT);   // dynamic id from the (post-Reset) global pool
    SetFlag<EVT>(e);
    WaitFlag<EVT>(e);                                // back-to-back: no credit persists past this call
}
// self-contained single-tile cube Run (raw L0 TBufs re-Init per stage; no Buffer<> credits):
//   GM->L1 (nd2nz)                          ; RbSetWaitFlag<MTE2_MTE1>()
//   LoadDataToL0A / LoadDataToL0B (fractal) ; RbSetWaitFlag<MTE1_M>()
//   Mmad(L0C, L0A, L0B, k=REAL_K)           ; RbSetWaitFlag<M_FIX>()
//   Fixpipe(GM, L0C, F322BF16)              ; RbSetWaitFlag<FIX_MTE2>()
```

**Evidence**: GDN `chunk_gated_delta_rule` regbase (A5/V351 arch35, CANN 9.1.T500, 2026-06-16). Cross-ref: PB-45 (Reset frees the global event pool; hardcoded-id `507015` trap), OL-220 (the library `MatmulImpl` once-Init path where the `Buffer<>` credit model IS correct).

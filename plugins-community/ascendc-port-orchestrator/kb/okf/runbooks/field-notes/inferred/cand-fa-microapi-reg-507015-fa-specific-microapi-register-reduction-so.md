---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-specific MicroAPI register-reduction softmax hits 507015 aicore exception (un-root-caused — PREMATURE, NOT a verified finding)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=attention (FA register-based softmax reduction) Status: UN-ROOT-CAUSED open issue, NOT a verified KB finding. Kept as candidate/roadmap only. Do NOT p"
phenomenon: build_failure
signal:
  - "porting the FA-A5 softmax to the MicroAPI register-compute path (__VEC_SCOPE__ + RegTensor + register-based reduction, vs the mem-based LocalTensor path that DI"
confidence: inferred
status: stub
original_id: CAND-FA-MICROAPI-REG-507015
timestamp_inferred: true
tags: [candidate, inferred, __vec_scope__, regtensor, localtensor, nd2nzparams, cand-fa-microapi-reg-507015]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=attention (FA register-based softmax reduction)`

**Status: UN-ROOT-CAUSED open issue, NOT a verified KB finding.** Kept as candidate/roadmap only. Do NOT promote to canonical KB until root-caused with a minimal reproducer.

**Symptom**: porting the FA-A5 softmax to the MicroAPI register-compute path (`__VEC_SCOPE__` + `RegTensor` + register-based reduction, vs the mem-based `LocalTensor` path that DID work — P-P101) hits a runtime `507015 aicore exception`. The TRIVIAL register elementwise path runs clean at runtime (OL-54 runtime-clean evidence, `tests/repro/regbase_minimal.cpp`), so the MicroAPI register infrastructure works for simple elementwise — the crash is specific to the FA register-REDUCTION usage in the cube-coresident FA context.

**Why premature**: not bisected to a mechanism; no minimized reproducer; candidate causes (register-reduction in the FA cube-coresident context, register pressure, or a `__VEC_SCOPE__`/sync-scope issue) not isolated. The de-scalarize WIN was achieved by routing AROUND this via the mem-based VEC softmax (P-P101); the register path remains an open question, not a result.

**Cross-ref**: P-P101 (the mem-based route-around that works + is verified — the de-scalarize win lives there, NOT here); OL-54 (trivial register elementwise runtime-clean — the contrast that scopes this crash to register-reduction specifically); the 507015 Mmad/Nd2Nz CANDs above (DIFFERENT 507015 flavors — those are cube layout / `Nd2NzParams` faults; this is MicroAPI register-reduction in the vec path).

## CAND-FA-A5-KFC-WORKSPACE (custom-launch large-D GM-staging needs `SetSysWorkspaceForce` on dav-c310/3510) [PROVISIONAL — pending DS corroborate]

`applies_to: soc=Ascend950PR (V351/A5, dav-c310/__NPU_ARCH__==3510); cann=9.0.0; bisheng=n/a; op_class=all (custom-<<<>>>-launch cube/matmul op)`
`status: PROVISIONAL — mechanism source-grounded + 2×2-reconciled + d256 disk-verified; aggregate result (independent prototype FA-A5 31→35) pending DS build-from-SHA corroborate (SHA 4b2f79b8)`

**Mechanism (source-grounded)**: on a hand-written `<<<>>>` launch (no aclnn/GE framework), `GetUserWorkspace(workspace)` ignores its argument and returns the GLOBAL `g_sysWorkspaceReserved + RESERVED_WORKSPACE` (16 MiB); on dav-c310/3510 the base is `GetSysWorkSpacePtr() = __get_kfc_workspace_addr()`. The deprecated `SetSysWorkspace` only sets the global `if (g_sysWorkspaceReserved == nullptr)` → silent no-op if already set / optimized → `GetUserWorkspace` returns `nullptr+16MB=0x1000000` garbage → D>192 GM-staging OOB `507015` while D≤128 (UB-resident) passes. Fix: call `AscendC::SetSysWorkspaceForce(workspace)` (unconditional) before `GetUserWorkspace`, workspace sized `data + 16MB`.

**2×2 reconciliation (delta-proof, not flaky re-run)**: layout-alone (no Force) = no output; with `SetSysWorkspaceForce` = clean. The earlier `SetSysWorkspace` measured-negative STANDS as a fact (it was a silent no-op), reconciled — not "newer-wins".

**Promotion gate**: DS build-from-SHA 4b2f79b8 corroborate (device.o-recompile, +4 large-D, original 31 no-regress) + the `g_sysWorkspaceReserved` dump showing plain=nullptr/garbage vs Force=alloc-base. **Cross-ref PB-41** (the verified V220/multi-core instance of the same workspace-registration contract — this CAND is the A5/3510 single-core large-D extension).

## CAND-FA-A5-WORKSPACE-BIFURCATION (hand-rolled-launch workspace-binding root bifurcates — do NOT cross-lane-generalize) [PROVISIONAL]

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=FA/cube custom-launch`
`status: PROVISIONAL — independent prototype multi-core side DS-confirmed; independent prototype large-D side pending corroborate`

**Principle**: a "hand-rolled launch missing framework workspace-binding" symptom can bifurcate into RELATED but DISTINCT roots — do not transfer one lane's fix to the other without measuring. FA-A5 instance: (a) multi-core lane FFTS cross-core sync-scratch (fixed by `SetSysWorkspace`+16MB) = **DS-confirmed**; (b) single-core lane large-D GM-staging (needs `SetSysWorkspaceForce`/kfc base, plain `SetSysWorkspace` is a silent no-op on 3510) = **provisional**. A cross-lane transfer ("apply an independent run's SetSysWorkspace to single-core large-D path") was MEASURED-refuted mid-session — same symptom class, different mechanism. Lesson: measure each lane's root; shared toolkit API (`SetSysWorkspace*`) ≠ shared root.

**Cross-ref**: PB-41, CAND-FA-A5-KFC-WORKSPACE, `feedback_passcount_variance_first_hypothesis_is_nondeterminism` (sibling "measure don't cross-lane-generalize" discipline).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-MICROAPI-REG-507015，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

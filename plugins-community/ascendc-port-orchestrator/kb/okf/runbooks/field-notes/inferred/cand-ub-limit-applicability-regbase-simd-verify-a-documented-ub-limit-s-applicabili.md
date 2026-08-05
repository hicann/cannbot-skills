---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "verify a documented UB limit's APPLICABILITY (loud/silent? SIMT-scoped? shipped kernel already exceeds it?) before invoking it as a ship-blocker"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=AIV; op_class=all (UB-budget, regbase-SIMD) verified_on: soc=Ascend950PR; cann=9.0.0 (bracket-probe + per-grad bit-identical, selective_scan bwd DB, PR"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=AIV; op_class=all (UB-budget, regbase-SIMD)"
confidence: inferred
status: stub
original_id: CAND-UB-LIMIT-APPLICABILITY-REGBASE-SIMD
timestamp_inferred: true
tags: [candidate, inferred, getcorememsize, __simt_callee__, __vec_scope__, xall, waitflag, cand-ub-limit-applicability-regbase-simd]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=AIV; op_class=all (UB-budget, regbase-SIMD)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (bracket-probe + per-grad bit-identical, selective_scan bwd DB, PR #82, 2026-07-01→02)`

**Principle**: Before invoking a documented UB-size limit as a ship-blocker, verify it applies to YOUR kernel class along three axes: (1) **LOUD or SILENT failure?** Bracket-probe the ceiling — on Ascend950PR, InitBuffer accepts ~255.5KB and rejects 256.0KB with a LOUD `507035` OOB → the true enforced ceiling is ~255.5KB, and `GetCoreMemSize`'s 248KB is a conservative advisory, not the enforced cap. A loud-failing ceiling is not the silent-OOB class. (2) **Is the limit SCOPED to a class you're not in?** PB-32's 40KB **silent** OOB is the SIMT DCache reserve, scoped to SIMT (L3 / `__simt_callee__`) kernels; it does NOT bind a regbase-SIMD kernel (MicroAPI `__VEC_SCOPE__`, no SIMT threads / DCache). (3) **Does the ALREADY-SHIPPED kernel already exceed the claimed limit?** A merged kernel running correct at 213.75KB > 208KB "SIMT-effective" proves the 40KB reserve doesn't bind that kernel family.
**Reserve-semantics safety check** (for a SIMD kernel allocating above the conservative advisory but below the loud ceiling): per-grad bit-for-bit md5 match to a known-good base at full customer scale + determinism across runs — a silent-reserve-corruption would perturb the grads or break determinism; bit-identical + deterministic ⇒ the reserved band is not clobbered under the real workload.
**Concrete anchor**: selective_scan bwd input-DB at 251.25KB (N=16 CH256) crosses the 248KB advisory but < the ~255.5KB loud ceiling; regbase-SIMD so PB-32 N/A; DB grads bit-identical to base at full customer scale (B=8 D=192 L=5000) → safe, shipped PR #82.
**Evidence**: selective_scan_bwd_simd DB UB-safety (2026-07-01→02): bracket-probe (255.5KB runs / 256KB→507035) + per-grad md5 bit-identical to base + PB-32 scope read. Cross-ref PB-32 (the SIMT 40KB reserve this scopes against).
**Other instances (predicted)**: any regbase-SIMD kernel near the UB ceiling; any ship-block decision citing a documented limit — check loud/silent + class-scope + shipped-precedent before taking the number at face value.
**Promote when**: a 2nd op's ship-decision is corrected by checking a limit's applicability (loud/silent, class-scope, or shipped-precedent) rather than taking the documented number at face value. backend=ascendc. Cross-ref: OL-245 (regbase amortization — the orthogonal UB-round-trip lever), OL-231 (issue-bound — the sibling roofline blind spot), MFU tool assessment (PR#76 latency-hiding modeling). Source: expert wiki (SelectiveScan反向自动生成算子优化, 0627/0630) + owner-directed measurement 2026-07-01. backend=ascendc.

### CAND-DB-COARSE-FENCE-CATCHES-PREFETCH: a coarse pipe-fence issued after a prefetch defeats double-buffering
`applies_to: soc=Ascend950PR; cann=9.x; bisheng=n/a; op_class=all (any double-buffered chunk loop)`

**Principle**: When adding input double-buffering (prefetch chunk N±1 while chunk N computes), any COARSE
pipe-fence that waits "all preceding same-pipe ops" — e.g. `SyncMTE2toV()` = SetFlag+WaitFlag
`<HardEvent::MTE2_V>` on a SHARED event id — issued in program order AFTER the prefetch will make the
compute pipe WAIT FOR THE PREFETCH to complete (the fence fires only after ALL preceding MTE2, including
the just-issued prefetch loads). This DESTROYS the overlap the DB was meant to create AND adds the fence's
own cost → net REGRESSION, and the aggregate vec_ratio DROPS (more compute-idle).

**Fix (either)**: (a) issue the coarse fence BEFORE the prefetch so it waits only the current chunk's own
loads; the prefetch (issued after) then overlaps compute. (b) use a DEDICATED event that waits only the
target buffer — do NOT reuse the coarse event that catches the prefetch's in-flight loads (the prefetch has
its own load-done event).

**Split-flag corollary (overlap a single-slot buffer with NO 2nd UB slot)**: if a buffer is read LATE in the
chunk (e.g. an `xall` read only at the post-pass), issue `SetFlag<MTE2_V>(dedicated_id)` right after its load
(before the prefetch) and DEFER the matching `WaitFlag` to just before its first read. Its load then overlaps
all intervening compute, without a 2nd slot. (ss-bwd: this lifted the gain from +0.85% to +1.17%.)

**Determinism note**: input DB reorders LOADS (MTE2) vs COMPUTE (VEC) only; it does NOT change the vector
accumulation order, so grads stay BIT-IDENTICAL to the pre-DB kernel — PROVIDED the DB keeps CH / chunk
partitioning identical (only adds a prefetch). Verify with non-aligned / tail-block shapes (odd L, small tail):
aligned-S pass + non-aligned-S flip = a DB-changed-partitioning bug. (agent-back confirmed the same
coarse-fence placement trap exists in the backward-plugin grad-reduce loops.)

**Evidence**: ss-bwd selective_scan_full_grad PASS B input DB, 2026-07-02, .171 Ascend950PR_957b (PR #87).
v1 (coarse SyncMTE2toV AFTER prefetch) = -1% (vec_ratio 0.894→0.889, vec-idle UP); v1.1 (fence BEFORE
prefetch) = +0.85% (vec_ratio 0.896→0.907); v2 (+split-flag xall late-load) = +1.17%. Instruction-timeline
(msprof --instr-profiling, `/aog-msprof-timeline`): VEC-idle-&-MTE2-busy stall 707→492us (-30%), MTE2
overlapped 18.6%→29.5%. All byte-identical + deterministic incl. tail cases.

**Other instances (predicted)**: any AscendC chunk-loop DB (attention grad, scan, conv im2col prefetch,
backward-plugin grad-reduce). The trap is universal to prefetch + shared-event coarse fences.

**PROCEDURE companion**: to MEASURE the stall this principle addresses, use the `/aog-msprof-timeline` skill
(instruction-level per-pipe gap), NOT the aggregate vec_ratio (which hides it).

**Status**: NEEDS_REVISION — mechanical scanners pending; kb-manager review before promotion.

### CAND-O5-REPRO-VIA-RUNNER-NOT-MANUAL-SSH: reproduce a finalize-gate (Phase-O5) failure through the orchestrator's OWN runner, not a hand-rolled SSH — the two can diverge on which path they read
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5; scope=lane-isolated`

**Principle**: When a finalize/verify gate reports a failure you cannot explain, reproduce it through the SAME code path the orchestrator uses, not through a manual shell reproduction. A hand-rolled `ssh <container> ...` reproduction resolves paths, env, and defaults differently from the orchestrator runner — most sharply under lane isolation (DEBT-159), where manual SSH lands in the shared `/root/AscendOpGenAgent/current_task` while the runner reads the lane-aware `AscendOpGenAgent_lane{N}/current_task`. Diagnosing against the wrong path produces a "fix" that verifies by hand but never clears the gate. This is the general failure mode behind PB-52 (two prior spawns "fixed" the wrong path and the signature repeated).

**Concrete anchor (the runner-faithful repro)**: drive the gate through its own entry point and un-suppress the child stderr the canonical command hides:
```
# Reproduce O5 exactly as the FSM does:
phase_o5.post_verify_for_finalize(ws, op, lane, runner=phase_o5_runner.ssh_runner)
# then spy on the canonical/fallback remote_cmd with a subprocess.run wrapper that
# strips the `>/dev/null 2>&1` so the real import/exec failure is visible.
```
The canonical `docker_cmd` suppresses stderr and returns rc=0 from a trailing `rm -f`, so a failing grader import surfaces only as "no JSON in stdout". Un-suppressing stderr + going through `phase_o5_runner` (not a shell) isolates the exact failing path + import in one pass.

**Skill-shape note** (Mode-1 step 2.6): this reads partly as a PROCEDURE (a repro command sequence). The transferable payload is the PRINCIPLE — *runner-faithful repro over manual-shell repro when paths can diverge* — which belongs in KB; the exact command recipe could migrate into a debug skill if it recurs. Kept here as one artifact because the principle and its one concrete recipe are tightly coupled and single-instance.

**Evidence**: gelu port_a3_to_a5 kw spawn 3 (2026-07-05, A5 Ascend950PR_957b, CANN 9.0.0, lane 0) — manual SSH hit `/root/...` (87/87 by hand) while the runner read the lane-0 path; the divergence hid the bug from two prior spawns until the O5 reproduction was driven through `phase_o5_runner`. Single instance → unverified.

**Promote when**: a 2nd finalize-gate failure (any op / any gate with an SSH-backed runner) is resolved by switching from a manual-shell repro to a runner-faithful repro that exposes a path/env divergence.

**Cross-ref**: PB-52 (the concrete lane-aware grader-import bug this methodology surfaced), DEBT-159 (lane isolation — the divergence source), P0kk / phase_o5 (the re-measure gate). backend=ascendc.

**Status**: NEEDS_REVISION — single-instance process candidate; kb-manager review before promotion.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-UB-LIMIT-APPLICABILITY-REGBASE-SIMD，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

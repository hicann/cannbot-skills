---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Prefer the managed cross-core abstraction for a MIX cube↔vec result handshake — hand-rolling exposes uncovered sub-cases one at a time"
description: "A hand-rolled cube↔vec result handshake must get sync-mode, per-AIV flags, asymmetric pipe, managed id, and GM visibility all right at once — prefer the managed cross-core abstraction."
confidence: single_run
original_id: OL-206
classified_by: llm-assisted
timestamp_inferred: true
tags: [cross-core-sync, optimization, ol-206, flash-attention, cube-vec-mix, race]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.0.0 / MIX 1:1 cube↔vec software-pipelined result handshake (FlashAttention / fused norm+matmul / MoE finalize). Verified on FA-A5 graybox kw-gb4→kw-gb5, 2026-06-03.

**Principle**: a hand-rolled cube↔vec *result* handshake on arch35-AIV must get MANY independent dimensions right **simultaneously**:
- sync mode 4, not 2;
- one flag per consumer AIV (`id` + `id+16` for 1-AIC→2-AIV; a single flag races the 2nd AIV);
- an ASYMMETRIC pipe (producer `Set` on `PIPE_FIX`, consumer `Wait` on `PIPE_V` — not the producer's pipe);
- a managed non-colliding event-id (not a hand-picked literal);
- and for a GM-routed result, an explicit data-visibility ordering for the consumer's MTE2 GM-read (flag-pipe ordering alone is insufficient).

Fixing any one dimension exposes the next still-uncovered sub-case, so a recipe-driven hand-roll converges only after iterating through *all* of them. The managed BaseApi cross-core abstraction (`Buffer<SyncType=CROSS_CORE_SYNC_FORWARD>` with `.SetCrossCore()`/`.WaitCrossCore()`) encapsulates every dimension (picks the pipe, manages the id, chooses routing + fence) and is bit-exact + deadlock-free.

**Decision**: for a MIX cube↔vec result handshake, PREFER the abstraction; hand-roll only with a specific reason, and if you do, expect to walk the entire `cross_core_sync.md §4(C)` sub-case ladder.

**Concrete anchor (the ladder, each rung a separate iteration)**: FA-A5 graybox —
- (gb2) `<0x2>` symmetric → DEADLOCK → mode-4 + `+16` dual-flag;
- (gb3) result `nan` → softmax-stat broadcast (CAND-FA-SOFTMAX-STAT-1);
- (gb4/gb5) consumer `Wait` on `PIPE_FIX`→`PIPE_V` compiles but a GM-routed result (consumer first-touch = MTE2 GM-read) STILL races (`softmax_sum` 0↔64 across fresh-process runs, `507015` aivec OOB) → needs route-result-through-UB or an explicit GM visibility fence.

Five iterations, each closing one rung. The whole-port reference never hand-rolls it — it routes through the abstraction and is bit-exact 64/64 on the first build.

**Cross-ref**: `cross_core_sync.md §4` (full hand-roll recipe + GM-vs-UB sub-case ladder), CAND-FA-SOFTMAX-STAT-1, `feedback_fsm_orchestration_crutch_whitebox_first` (reading the working reference's mechanism beats N hand-roll iterations), `feedback_passcount_variance_first_hypothesis_is_nondeterminism`, P-P103 (FA-class template).

**Evidence**: FA-A5 `flash_attention_score` graybox kw-gb4 (case7 minimal repro: 1-AIC+2-AIV `softmax_sum` drift 55) + kw-gb5 (PIPE_V compiles, GM-routed still races), 2026-06-03; whole-port reference bit-exact 64/64 via the abstraction (`1f00bbdc`). Predicted to recur on any MIX 1:1 cube↔vec software-pipelined op with a producer→consumer result path.

## 证据
- FA-A5 `3_FusionAttention` GQA (2026-06-02, VERIFIED 2-way): host-wiring fix 43→48/64; +5 GQA cases (idx37-40,48) bit-exact max_abs=0.0 — confirmed independently by independent prototype (kw-27 ×2, NPU4) AND DS (bb1d7644 --clean build .so aef8e6df, NPU1, exact-ulp re-score), zero regression both. commit bb1d7644 (host-only +23/-5, device byte-identical). First validated instance. NOTE: the kw harness initially reported 50/64 because its bf16 acceptance tolerance was set to 2e-2 (~5× the bf16 floor 3.9e-3, ~40× the fp16 floor) — the OL-203 loosened-allclose anti-pattern; exact-ulp re-score (fp16 4.9e-4 / bf16 3.9e-3) = honest 48/64 (the loose tol falsely passed 2 pse cases idx47/60). Harness fixed to exact-ulp floor (`bench64_case.py _T1_TOL`). Cross-validates OL-203.
- FA-A5 pse (2026-06-02, VERIFIED 2-way, DS exact-ulp = +8 → 48→56/64): `wp_pse.h`(298L) PseInfo/PseType/DataCopyIn<hasPse> machinery complete + Nd-path vf pse-apply (ADD-then-scale) + pse launcher instantiated; host pybind had no pse/sink param, dispatched only the dense launcher, never filled PseInfo. Fix (host-only, commit b2f4703e, +90 LOC, NO kernel change) = pse/sink GM through entry+launcher+pybind + fill PseInfo (pseType + layout PSE_S1S2/PSE_1S2 by shape) + dispatch pse launcher, PLUS a NUMERICAL-SEMANTICS host adaptation: sink·scale pre-multiply (kernel uses raw sink in already-scaled space, oracle uses sink·scale → pybind pre-multiplies). 9/9 pse cases verified at exact-ulp on the CORRECT build (8 bit-exact max_abs=0 + idx46 fp16 at floor; idx60 = 0.0 bit-exact deterministic). **idx60 CORRECTION (2026-06-02, SUPERSEDES the earlier "real FAIL / NON-DETERMINISTIC / route-aog-determinism-analyzer" verdict — that verdict was a STALE-BUILD PROVENANCE artifact, NOT a real non-determinism bug)**: idx60 (fp16 pse+sink, scale 0.088) was transiently reported as non-deterministic (max_abs 0.00134-0.0034, varying) — but every such measurement ran a STALE deployed .so compiled from the OLD kw-28 pybind. The build compiles `kernel/pybind11.cpp`; a docker-cp that updated only the wholeport copy did NOT update the file the build actually compiled → the un-fixed kw-28 kernel was measured throughout. True root = a DETERMINISTIC sink·scale OVER-correction: kw-28 applied a sink logit the oracle (npu_fa) does NOT apply (npu_fa's sink is a per-head constant that cancels in the softmax row-max shift = no-op); spurious at small scale 0.088 → the ~0.003 residual. The fix is `sink=null` (do not apply the absent sink) — which SUPERSEDES the earlier "sink·scale pre-multiply" host adaptation noted above for the sink cases (that pre-multiply was itself the over-correction). On the fresh-clone verified build (DS static-md5 confirmed the compiled source = the fixed pybind), idx60 = max_abs 0.0, deterministic (8/8 runs). NOT non-determinism, NOT uninit-workspace, NOT a separate class — a host-semantics over-correction (same OL-205 host-feature-dispatch family) masked by stale-build provenance. **META-LESSON (durable): before deep-diving an apparent non-determinism or "kernel" bug, statically verify the md5 of the source the build ACTUALLY compiled matches the cited commit — a stale deployed .so spawns an entire chain of conclusions (non-det / uninit-workspace / N×-CANN / "value not fixed") built on void data; DS's md5 check of the compiled source was the single unblock.** Final state on the correct build (DS authoritative, fresh-clone): 64/64 = 59 exact-ulp-T1 + 5 T2 (long-seq S512 / inner_precise=1 / sink, ~4× floor, within-CANN-band per prior CPU-golden). zero regression (5 GQA + 3 V1 high-D + 4 mask + 3 fp32 stay bit-exact). NOTE: independent prototype first reported +9/57 by mislabeling idx60 as bf16 (passes under bf16 floor 3.9e-3); DS read the V2 json directly → idx60 is fp16 → exact-ulp 56/64. This was the harness's 3rd loose-tol/wrong-floor count inflation (GQA 50→48, pse 57→56).
- FA-A5 mask-sm1 (2026-06-02, VERIFIED 2-way DS exact-ulp = +4 → 56→60/64): `wp_attenmask.h` hasAtten machinery present (BoolCopyInRegbase + Select) but no `hasAtten=true` launcher instantiated + host didn't pass atten_mask GM. Fix (host-wiring, commit 8ef7301b) = instantiate hasAtten launcher + dispatch + pass mask GM PER-BATCH `[B,S1,S2]` (single `[S1,S2]` face corrupts batch>0) + tiling NO_COMPRESS/BS1S2. Polarity EMPIRICALLY probed (not assumed): bool True=mask-out/False=keep (oracle diff 8.7e-5). 4 cases bit-exact max_abs=0 incl B=2 per-batch, zero regression. DS-confirmed (.so 9daac258 --clean build).
- FA-A5 fp32 (2026-06-02, host-wiring CONFIRMED — the earlier "structural exception" assumption fully REFUTED): kernel cube/vec HAS a complete fp32 path (`IsSameType<INPUT_T,float>` branches in wp_block_cube.h:148/157/361/387/547/710 + wp_block_vec_base.h vec1ScmBlockFp32). The runtime "supports fp16/bf16 only" ERROR was HOST/LAUNCHER-level (TORCH_CHECK + no fp32 launcher instantiated). Fix (host-wiring, commit 8c1cbcbb) = instantiate fp32 launcher + relax TORCH_CHECK + dispatch; verify-by-build confirmed a REAL MIX_AIC_1_2 fp32 device kernel (binary check, not a downcast). idx15/16/17 PASS_T1 max_abs=0.0 bit-exact = +3 (independent prototype self-verified exact-ulp + DS-VERIFIED total 63/64 at the fp32-fix point (→ 64/64 final after the idx60 stale-build correction in the pse evidence above), .so 5574f3d9 — real fp32 kernel `wp_fa_fp32_bnsd_d64_mix_aic/aiv` confirmed non-downcast, parent 0 → 10 fp32 symbols). **5th confirmed host-wiring instance — the "structural" verdict required an actual build failure, which never came; labeling it structural from an assumption was the error. Do NOT label "structural" before building.**

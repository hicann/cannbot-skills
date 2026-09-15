---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Chunked gated-delta-rule / gated-linear-attention BACKWARD — a3 MIX 3-pass reverse-recurrence (parallel PASS A using the passed-in state h, one REVERSE-recurrence PASS B for dstate, per-chunk PASS C for the step5/6 grads; ON-DEVICE decay-fold + reverse-cumsum(dg) + GQA head-sum + matmul-with-ones reductions)"
description: "For the BACKWARD of a chunked gated-delta-rule / gated-linear-attention on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch) — six gradients dq,dk,dv,dg,db,dh0 from (q,k,v,g,beta,A,h,do,dht,i"
severity: high
confidence: single_run
original_id: P-P118
timestamp_inferred: true
tags: [fa_class, optimization, dstate, dg5, dq5, op_host, setorgshape, p-p118, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For the BACKWARD of a chunked gated-delta-rule / gated-linear-attention on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch) — six gradients `dq,dk,dv,dg,db,dh0` from `(q,k,v,g,beta,A,h,do,dht,initial_state)`. **Two math simplifications**: (1) use the passed-in recurrent state `h` DIRECTLY (the reference recomputes it identically) → NO in-kernel forward sweep, `vn=u-w@h[c]`, PASS A is parallel per-chunk; (2) only step4 (`dstate`) is a cross-chunk recurrence, and it runs in REVERSE chunk order (PASS B); steps 1/3/5/6 are per-chunk. **3 passes** (per head, blockdim=nHead, MIX_AIC_1_1): PASS A (fwd c) `w=A@kbg; u=A@vb; vn=u-w@h[c]; AmT=mUpInc⊙(kn@qsT); dv=AmT@do; dsi=qs^T@do`; PASS B (REVERSE c) `dh[c]=dstate; dv[c]+=kd@dstate; dstate=dstate*sc+dsi-w^T@dv` (fp32 GM accum), `dh0=dstate`; PASS C (fwd c) step5+step6 per-chunk matmul assembly (verified vs model.py — signs/masks/transposes + the order `dg5` from `dq5` PRE-`ds@k`). **ON-DEVICE (all compute in AscendC, NO host torch)**: per-chunk operand fold via `Exp()` (`g` arrives already per-chunk cumsum'd so kernel does exp only); `dg`/`db` reductions as matmul-with-ones → `[C,16]` fp32 accumulators (host col0); `reverse-cumsum(dg)` in-chunk scalar prefix-sum on UB; GQA head-sum(dq,dk) a device compaction stage. **Do NOT host-fold with torch exp/cumsum/sum** — an earlier host-fold bwd variant scored delegation violations pinpointed in `op_host`. Transposes via runtime `SetTensorA/B(bool)` (P-P69). Partial chunk (T=200): no fill needed (that case has `dht=0`; causal mask + zero-pad vanish the tail). MIX substrate same as fwd: NON-KFC cube (P-P68, `SetOrgShape` per reused-mm call = CAND-GDR-3), intra-AIV UB fence (CAND-GDR-2), whole-device `SyncAll<false>()` (PB-57: needs `MIX_AIC_1_1`). **Device-verified 11/11 @ fp64 customer gate (rtol=0.02 per-gradient, ~14-18× headroom, deterministic, NaN-free, input-mutation-safe), 2.22–11.25× vs the PyTorch-NPU reference** (a3/Ascend910_9382, DS 2026-07-20). Backward-specific gotchas: CAND-GDR-BWD-1 (dsMask2 swapaxes = head relocation not [C,C] transpose), CAND-GDR-BWD-2 (`.clone()` dstate init or it aliases + mutates input dht), CAND-GDR-BWD-3 (thread `initial_state` when building the passed-in h in the test harness). Full body: `patterns/domains/gated_delta_rule_bwd_a3_recurrence.md`. Cross-ref P-P117 (forward), P-P116 (a3 MIX sync template), P-P68, P-P69, PB-57, CAND-GDR-2/3, CAND-GDR-BWD-1/2/3. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention-backward/CUBE_MIX; verified_on=gated_delta_rule bwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P118，convert_patterns_to_okf.py）。confidence 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Chunked gated-delta-rule / gated-linear-attention forward — a3 MIX multi-chunk recurrence (per-head sequential chunk loop + persistent GM state `S[D,D]` + ON-DEVICE per-chunk decay-fold so every kernel matmul is a plain `A@B`)"
description: "For a chunked gated-delta-rule / gated-linear-attention forward on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch). Per head, loop chunks c=0..ceil(T/64)-1 sequentially with persistent stat"
severity: high
confidence: single_run
original_id: P-P117
timestamp_inferred: true
tags: [fa_class, optimization, mul, cast_rint, gc_last, matmulimpl, setorgshape, p-p117, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For a chunked gated-delta-rule / gated-linear-attention forward on a3 (Ascend910_9382, arch22; arbitrary T, GQA, multi-batch). Per head, loop chunks `c=0..ceil(T/64)-1` **sequentially** with persistent state `S[128,128]` in GM (init 0). The kernel folds per-chunk operands **ON-DEVICE** (AIV Stage 0 — NO host torch compute; the no-delegation rule requires all compute in AscendC) from `gc=cumsum(g)` (in-chunk scalar prefix-sum), `eg=Exp(gc)`: `kb=k*beta*eg`, `knT=(k*exp(-gc))^T`, `qs=scale*q*eg`, `vb=v*beta`, `kdT=(k*exp(gc_last-gc))^T`, `sc=eg[C-1]=exp(gc_last)` — into scratch slots `S_KB/S_QS/S_VB/S_KNT/S_KDT`, so EVERY kernel matmul is a plain `A@B`. Fold uses `Broadcast<T,2,1>` (row-scale kb/qs/vb) + repeat-`Mul` broadcast (col-scale transposed knT/kdT, the sibling `recurrent_gated_delta_rule::MatVecMul` idiom); fp32 UB, `CAST_RINT` to fp16 GM. Host does layout marshaling only (GQA head-expand, zero-pad, chunk-reshape, cast, head-major permute incl. raw `kT`). **Do NOT fold on the host with torch cumsum/exp** — an earlier host-fold variant scored 4 delegation violations. Zero-pad `T→Nc*64` + per-chunk cumsum makes the padded tail carry `gc_last` automatically and padded rows/cols vanish under causal masking (no explicit last-chunk fill). Within-chunk: `Acc=kb@knT`; `L=-strict(Acc)`; `T=(I+strict(Acc))^-1` via Neumann power product `prod(I+L^{2^k})` k=0..5 (see CAND-GDR-1 for the solve-sign gotcha); `Am=incl(qs@knT)`; `U=T@vb`. Cross-chunk: `W=T@kb`; `WS=W@S`; `o_inter=qs@S`; `vn=U-WS`; `o=Am@vn+o_inter`; `S=S*sc+kdT@vn`. `Nc==1` (S=0) collapses to single-chunk. MIX: matmuls on AIC (`MatmulImpl`+`IterateAll<sync=true>`, arch22-safe NON-KFC cube per P-P68 — reuse one mm object needs `SetOrgShape` per call, CAND-GDR-3), elementwise on AIV (incl. the Stage-0 on-device decay fold), whole-device `SyncAll<false>()` sync (see **PB-57: needs `KERNEL_TYPE_MIX_AIC_1_1`**; intra-AIV UB reuse needs a `PipeBarrier<PIPE_ALL>` fence, CAND-GDR-2). GQA via host `repeat_interleave` head-expand (layout). **Device-verified 16/16 @ fp64 customer gate (rtol=0.02, ≥17× headroom, deterministic, NaN-free), 9–28× device-time vs torch_npu** (a3/Ascend910_9382, DS 2026-07-20). Full body: `patterns/domains/gated_delta_rule_a3_recurrence.md`. Cross-ref P-P68 (NON-KFC cube), P-P116 (a3 MIX sync template), PB-57 (1:1 macro), CAND-GDR-1/2/3/4. `applies_to: soc=Ascend910_9382 (a3/arch22); cann=9.1.0; op_class=gated-linear-attention/CUBE_MIX; verified_on=gated_delta_rule fwd a3 (DS 2026-07-20); unverified_on: Ascend950PR`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P117，convert_patterns_to_okf.py）。confidence 未升格。 -->

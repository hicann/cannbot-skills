---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Fused multi-output elementwise gradient — one kernel pass loads shared inputs once and emits all N outputs, NOT one kernel per output"
description: "applies_to: soc=Ascend910_9382 (V220/arch22); cann=9.0.0; bisheng=n/a; op_class=elementwise_backward verified_on: soc=Ascend910_9382 (V220/arch22); cann=9.0.0 unverified_on: soc=Ascend950PR (V351/arch"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220/arch22); cann=9.0.0; bisheng=n/a; op_class=elementwise_backward"
confidence: inferred
status: stub
original_id: CAND-PP109
timestamp_inferred: true
tags: [candidate, inferred, mul_grad, add_grad, sub_grad, div_grad, unverified_on, cand-pp109]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220/arch22); cann=9.0.0; bisheng=n/a; op_class=elementwise_backward`
`verified_on: soc=Ascend910_9382 (V220/arch22); cann=9.0.0`
`unverified_on: soc=Ascend950PR (V351/arch35); dtype=fp16/bf16; large-N tiling-loop path (single-tile only so far)`

Pattern: an elementwise backward op that produces N input-gradients sharing the same upstream loads (e.g. `mul_grad`: `grad_x=gy*w`, `grad_w=gy*x`; generalizes to `add_grad`/`sub_grad`/`div_grad` and any elementwise map whose VJP reuses the forward inputs/grad-output) is cleanly expressed as a **single kernel pass**: load the shared inputs (x, w, gy) once into UB, emit each output with one VEC op, store all outputs. This halves/N-folds MTE2 DataCopy traffic and launch overhead vs N separate single-output launches. Deterministic by construction (one core owns each align-8 chunk, each output element written exactly once, no SetAtomicAdd) — satisfies `DET_POLICY=required` with no determinism-specific code.

Measured (mul_grad kw-1, 2026-05-30, a3/Ascend910_9382): 2/2 PASS fp32 vs fp64 oracle, det 2/2 bit-identical, **2.99x median** vs the torch autograd backward path (~49us ours vs ~140us autograd: two backward-mul launches + engine traversal) at N=8/16 (both launch-overhead-bound). 0 compile-fix, 0 precision-fix iters.

**Promote when**: a SECOND elementwise-backward op (add_grad / div_grad / a 3+-output elementwise VJP) ships single-pass with measured perf >= 1.0x vs the per-output-launch baseline — confirms the fusion generalizes beyond the 2-output mul case. Also lift the tiling-loop path (current evidence is single-tile, small-N only) and an A5/V351 + fp16/bf16 instance to retire the `unverified_on` line.

**Cross-ref**: CAND-PP106 (the IDENTICAL-grad-duplication sibling, `d_a==d_b==grad_x` emit-same-value-to-2-buffers; PP109 here is the shared-load / DIFFERENT-grad fusion — kept distinct until a 2nd op lets a maintainer promote them to a unified canonical multi-output-backward pattern), OL-103 Tier-1 (pure-Mul chain passes fp32 threshold; the precision side of this op), OL-181 (output buffer padded to Align8(N) for DataCopy overflow, narrowed in pybind), PB-22 (DataCopy 32B-aligned `cnt=Align8(len)`), OL-160 (canonical `model_new_ascendc.py` entry-point reused), CAND-FAG-4 (multi-output FUSED-bwd dispatch — the cube+atomic-add cousin for attention-grad; this CAND is the cheap elementwise analogue with by-construction det and no atomics).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP109，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

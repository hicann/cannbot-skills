---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "scan/recurrence ops must accumulate in fp32 internally, independent of I/O dtype"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like) status: CANDIDATE (selective_scan_fwd: fp16/bf16 T2-达标 2026-06-17,"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like)"
confidence: inferred
status: stub
original_id: CAND-SCAN-FP32-ACCUM
timestamp_inferred: true
tags: [candidate, inferred, cand-scan-fp32-accum]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=scan/recurrence/linear-attention (Mamba SSM, cumulative, RNN-like)`
`status: CANDIDATE (selective_scan_fwd: fp16/bf16 verified against fp64 truth; graybox-reproduce proof-gate pending)`

**Principle**: keep the L-scan recurrence state (`x = deltaA*x + deltaB_u`) AND the output reduction in **fp32 internally throughout**, casting to the I/O dtype (fp16/bf16) only at the final store. The low-precision output then lands within the dtype's quantization range against fp64 truth. The recurrence accumulator precision, not the I/O dtype, governs correctness.

**Generalizable rule** (scan/recurrence class — Mamba SSM, cumulative-sum, RNN-like, linear-attention): **NEVER accumulate a recurrence in the low-precision I/O dtype; ALWAYS use an fp32 accumulator + cast-at-end.** This is the load-bearing pattern for low-precision (fp16/bf16) 达标 on these ops. (Conversely, fp32 *output* on the same op can hit an irreducible near-zero-cancellation MARE floor — that is a separate, dtype-intrinsic limit, NOT fixed by accumulator precision; see the fp32-T1-floor finding.)

**Scope boundary (C35 reconcile, main 2026-06-18) — FORWARD-accumulator; NOT superseded by the backward entry**: this rule governs the *forward* recurrence accumulator precision (fp32 internal, cast at store). The sibling `CAND-SSM-BWD-WEIGHTGRAD-FP32` governs the *backward* weight-grad **output dtype** and explicitly notes the backward accumulator was ALREADY fp32 (consistent with this rule, not a correction of it). Complementary, not contradictory: forward → accumulator-precision; backward → output-dtype. This entry is **NOT superseded** — the in-channel "supersede" framing was corrected after reading both entries.

**Concrete anchor**: selective_scan SIMT kernel keeps state + reduction in fp32 and casts only at store.

**Evidence**: selective_scan_fwd 2026-06-17 (independent prototype whitebox against fp64 truth): C2 fp16 MERE 1.1e-6 / MARE 4.5e-3 (gate 9.77e-3) 达标; C3 bf16 MERE 7e-7 / MARE 7.5e-3 (gate 7.81e-2) 达标.

**Provenance**: derived from independent prototype selective_scan_fwd T2 whitebox 2026-06-17 (owner-directed precision-alignment). Promotion gated on graybox-kw reproduction (#94 proof-gate).

**Cross-reference**: CAND-GDN-CHUNK-RECURRENT-COMPOSE (sibling scan/recurrent op).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SCAN-FP32-ACCUM，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

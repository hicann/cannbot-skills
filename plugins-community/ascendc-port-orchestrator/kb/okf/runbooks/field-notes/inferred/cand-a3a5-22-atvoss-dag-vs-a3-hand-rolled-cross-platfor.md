---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "atvoss-DAG-vs-A3-hand-rolled cross-platform 1-ULP drift concentrates on the highest-arithmetic-intensity output"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5; arch_pair=A3(V220)↔A5(V351) verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: apply_adam_w_v2) unverified_on: other"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5; arch_pair=A3(V220)↔A5(V351)"
confidence: inferred
status: stub
original_id: CAND-A3A5-22
timestamp_inferred: true
tags: [candidate, inferred, apply_adam_w_v2_b16.h, var, max_grad_norm, cand-a3a5-22]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5; arch_pair=A3(V220)↔A5(V351)`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: apply_adam_w_v2)`
`unverified_on: other Adam-family ports; other normalize/fuse/quant atvoss-DAG ports`

**Predicted rule** (forward-looking, 1-op evidence):
When the upstream A3 kernel uses bespoke per-dtype `.h` files (e.g.
`<op>_b16.h` / `<op>_fp.h` / `<op>_mix_dtype.h`) and the A5 port uses an
`atvoss/elewise/elewise_sch.h` + `ElementwiseSch<schMode, OpDag>` DAG, expect
1-ULP cross-platform drift concentrated on the **highest-arithmetic-intensity
output** (the tensor whose computation graph has the most Mul/Add chain
ordering choices). Other outputs typically remain bit-exact. T1
bit-exactness across all outputs across A3↔A5 is NOT achievable on this
class of port — T2 within per-dtype ULP-floor is the realistic target.

**Why**: atvoss reorders Mul/Add chains for compute-graph fusion. The
reordering is mathematically equivalent (same operations, different
parenthesization) but **bit-different** under IEEE-754 — error
amplifies with arithmetic intensity, so the worst-drift output is the one
with the longest chained reduce/EMA/normalize sequence.

**Concrete anchor (apply_adam_w_v2, 2026-05-14)**:
| Output | A3 path (hand-rolled `.h`) | A5 path (atvoss DAG) | Observed drift |
|---|---|---|---|
| `m` (1st-moment EMA) | `apply_adam_w_v2_b16.h` → tight Cast/Mul/Sub/Add chain | DAG node `OpGradCast_ × OpBeta1Sub1` → accumulator | max |Δm| ≈ 1.5e-8 fp32 (1-ULP), 6.1e-5 fp16 (1-ULP @ scale 0.5), 4.9e-4 bf16 (sub-ULP @ scale 1.0) |
| `v` (2nd-moment) | bespoke `.h` | DAG | **bit-exact** |
| `var` (param) | bespoke `.h` | DAG | **bit-exact** |
| `max_grad_norm` | bespoke `.h` | DAG | **bit-exact** |

`m` has 2 Muls + 1 Sub + 1 Cast in the EMA update — highest AI; the
other outputs have fewer chain links and survive bit-exact. Signature
matches OL-83 (two valid fp32 paths → 1-ULP boundary drift) but
**operates at port level, not single-op precision-probe level**.

**Relation to OL-83**:
- OL-83 = a single op + two FMA-grouping orderings on the same SoC give 1-ULP drift
- This candidate = cross-SoC A3↔A5 port + structural difference (bespoke per-dtype `.h` vs DAG) gives 1-ULP drift, **localized to the highest-AI output**
- Not contradictory — this candidate is a *port-specific predictor* for which output OL-83-class residual will land on, BEFORE the precision probe runs.

**How to use during port verification**:
1. Inventory port-source: if A3 has `<op>_b16.h` / `<op>_fp.h` family AND A5 uses `ElementwiseSch<OpDag>`, EXPECT cross-platform 1-ULP residual on the heaviest-chain output.
2. Don't waste a precision-probe spawn trying to chase bit-exactness on that output across SoCs. Accept T2 within per-dtype ULP-floor.
3. DO still verify the *other* outputs are bit-exact — if more than the heaviest-chain output drifts, the port has a real bug, not a structural residual.

**Promotion gate**: needs ≥ 1 additional Adam-family port confirmation
(e.g. apply_adam_w / apply_came / apply_lamb), or any non-Adam atvoss-DAG
port (norm-fuse-quant, layernorm-quant) showing the same "drift concentrates
on highest-AI output, others bit-exact" signature. If a second case shows
drift on a low-AI output instead, the candidate's "highest-AI" claim is
wrong and the rule needs revision before promotion.

**Cross-ref**:
- OL-83 (1-ULP single-position drift between two valid fp32 paths — same-SoC root)
- OL-141 (target arch35 advisory-inventory rule — context where this drift class shows up)
- OL-81 (CAST_RINT for bf16/fp16 output cast — the DAG's output cast convention)
- OL-118 (when fp32 output should NOT be cast — different concern: this candidate is about *how* the cast surfaces drift on intermediate accumulators)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-22，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

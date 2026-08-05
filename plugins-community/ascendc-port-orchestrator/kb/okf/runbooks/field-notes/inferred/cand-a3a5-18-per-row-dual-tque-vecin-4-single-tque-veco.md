---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Per-row dual `TQue<VECIN,4>` + single `TQue<VECOUT,4>` pipeline for split-input pointwise ops"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-split-input,gate-and-multiply verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: fatrelu_mul) unverified_on: other SwiG"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-split-input,gate-and-multiply"
confidence: inferred
status: stub
original_id: CAND-A3A5-18
timestamp_inferred: true
tags: [candidate, inferred, pingpongflag, cand-a3a5-18]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=elementwise-split-input,gate-and-multiply`
`verified_on: soc=Ascend950PR; cann=9.0.0 (1-op evidence: fatrelu_mul)`
`unverified_on: other SwiGLU/GeGLU/ReGLU/gating-style ports`

**Predicted rule** (forward-looking, 1-op evidence):
For ops where each row has two correlated input slices (split-input pointwise: x1 = first half of input row, x2 = second half), use **two depth-4 input queues simultaneously** + one depth-4 output queue. The driver loop becomes:

```cpp
for row in my_rows:
    AllocTensor(x1_q); AllocTensor(x2_q);
    DataCopyPad(x1, ...); DataCopyPad(x2, ...);
    EnQue(x1_q); EnQue(x2_q);
    // -- Compute step --
    DeQue(x1_q); DeQue(x2_q); AllocTensor(out_q);
    Cast / CompareScalar / Select / Mul / Cast;
    FreeTensor(x1); FreeTensor(x2); EnQue(out_q);
    // -- CopyOut step --
    DeQue(out_q); DataCopyPad(out, ...); FreeTensor(out);
```

**Why this beats upstream V220's manual sync choreography**: V220 uses explicit `pingPongFlag` + `SetFlag<HardEvent::MTE3_MTE2>` + `WaitFlag<HardEvent::MTE2_V>` to overlap pipelines. TQue depth=4 on A5 gives equivalent MTE2/VEC/MTE3 overlap with much less surface for sync-bug regressions; no manual flag-management code path means no per-dtype flag-mismatch failure mode.

**Concrete anchor (fatrelu_mul, 2026-05-17)**:
Two parallel `TQue<QuePosition::VECIN, 4>` for x1/x2 + one `TQue<QuePosition::VECOUT, 4>` for output. 8/8 T1 PASS bit-exact vs A3 ground truth; median ratio 1.054× over A3 baseline on small-shape inputs (max 8K elements). Per-tile compute (Cast + CompareScalar + Select + Mul + Cast) is heavy enough to amortize depth=4 allocator overhead — falls in OL-63's "heavy compute → depth=4" regime, not the "thin compute → depth=2" regime.

**Applicability** (predicted): any split-input pointwise op family —
- SwiGLU / clipped_swiglu (x1 = SiLU-gate, x2 = up-projection)
- GeGLU / ReGLU / FATReLU / clipped_silu_mul
- Mul-of-two-halves activation variants
- Two-input element-wise gating (output = f(x1) * x2 for some f)

**Promotion gate**: needs validation on 2+ additional ports in this family (e.g. swiglu, geglu) confirming depth=4 dual-VECIN holds the perf ratio above OL-143's 0.6× floor without per-tile thinning regression. If a second case is thin-compute (single VEC op per tile) and depth=4 regresses vs depth=2 (per OL-63's thin-compute branch), the rule should be scoped to "heavy compute split-input" rather than "all split-input".

**Cross-ref**:
- OL-63 (TQue<VECIN,4> depth decision — this candidate is a specific application to the dual-VECIN case)
- OL-143 (L1 mechanical port — this pattern is the canonical L1 layout for SwiGLU/FATReLU-family)
- P-P28 (per-tile depth=4 baseline pattern — extends to dual queue here)
- OL-115 (depth=2 + explicit prefetch — the alternative for thin per-tile compute; not preferred for split-input gate-and-multiply where compute is heavy)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-A3A5-18，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

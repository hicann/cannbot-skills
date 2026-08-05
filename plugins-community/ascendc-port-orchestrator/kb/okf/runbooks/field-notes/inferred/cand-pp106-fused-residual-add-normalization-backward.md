---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Fused residual-add + normalization backward — the two add-branch input grads are bit-identical; emit grad_x to both output buffers from one compute pass"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad) verified_on: soc=Ascend910_V220; cann=9.0.0 (fused_add_rmsnorm_grad 4/4 PASS) statu"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad)"
confidence: inferred
status: stub
original_id: CAND-PP106
timestamp_inferred: true
tags: [candidate, inferred, grad_x, rms_norm_grad, cand-pp106]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; op_class=norm_family_backward (fused residual-add + rms/layer norm grad)`
`verified_on: soc=Ascend910_V220; cann=9.0.0 (fused_add_rmsnorm_grad 4/4 PASS)`
`status: UNCONFIRMED — single op; needs a 2nd fused-add-norm op (e.g. add+layernorm grad) to confirm`

**Pattern**: for a "fused residual-add + `<normalization>`" backward where the forward is `x = a + b; y = norm(x) * w`, the two add-branch input grads are BIT-IDENTICAL: `d_a == d_b == grad_x`. This is the identity Jacobian of `x = a + b` (∂x/∂a = ∂x/∂b = I), so there is NO separate backward to compute for the second branch. Emit `grad_x` to BOTH output GM buffers from ONE compute pass. The marginal cost over the non-fused norm-grad is exactly **+1 input load + 1 extra grad store** — NOT a second backward.

```
forward:  x = a + b;  y = (x * rsqrt(mean(x^2)+eps)) * w   // add + rmsnorm
backward: grad_x = r*(g - (x*r*r)*sum_H(g*x)/H)            // the rms_norm_grad grad_x
          d_a = grad_x;  d_b = grad_x                       // identity branch — store twice, no recompute
```

**Evidence**: fused_add_rmsnorm_grad (2026-06-03, port_a3_to_a5 V220, KB-only / analytic backward derived from scratch, bit-exact vs fp64 autograd oracle, max_diff ~1e-16) — 4/4 PASS first verify, 0 precision-fix iters. Structurally the op is "`rms_norm_grad`'s grad_x emitted to two output buffers + the same grad_w cross-row reduction"; the leading `x = a + b` add is the only forward delta and it vanishes to an identity on the backward side.

**Promote when**: a 2nd fused-add-norm backward (e.g. add + layernorm grad) confirms the d_a==d_b==grad_x identity-store recipe transfers — i.e. it's a norm-family rule, not a one-op coincidence.

**Cross-ref**: rms_norm_grad / OL-103 (the norm-family transcendental NR-Rsqrt backward technique this fused op consumes — the add-branch dup is the only structural delta on top), OL-75 (the partial+reduce template the grad_w cross-row reduction uses), CAND-PP104 (sibling norm-family backward launch-overhead candidate), CAND-PP109 (the shared-load/different-grad multi-output sibling — PP106 here is identical-grad duplication).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP106，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

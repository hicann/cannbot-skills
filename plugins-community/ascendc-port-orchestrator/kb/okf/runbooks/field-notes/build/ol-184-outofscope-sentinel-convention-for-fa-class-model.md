---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`_OutOfScope` sentinel convention for FA-class `model_new_ascendc.py`"
description: "applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class); model_new_ascendc.py"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class); model_new_ascendc.py"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-184
timestamp_inferred: true
tags: [_outofscope, ascendc, ol-184]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; op_class=FUSED_SOFTMAX (FA-class); model_new_ascendc.py`
`verified_on: workspace/3_FusionAttention canonical_p2t_summary 2026-05-23 (5 GQA cases flagged by gate 5)`

**Principle**: An FA-class kernel intentionally ships scope-bounded at v1.0 (e.g. `S=128, D=64, fp16, BNSD` only — see FA_CLASS_PROBLEM_SOLUTION_DESIGN.md §3.1 v-table). Cases outside the support window MUST declare out-of-scope EXPLICITLY by raising `_OutOfScope` exception BEFORE the kernel runs, NOT by letting the kernel/reference crash with `RuntimeError`.

**Canonical implementation pattern** (paste into `model_new_ascendc.py:ModelNew.forward` head):

```python
class _OutOfScope(Exception):
    """Sentinel: harness records EVAL_ERR with error string containing
    '_OutOfScope' → gate 5 (DEBT-116) accepts as v1.0 scope boundary."""
    pass

class ModelNew(torch.nn.Module):
    def forward(self, query, key, value, *args, **kwargs):
        # Explicit scope gate — declare unsupported regimes BEFORE kernel call
        S = query.shape[-2]
        D = query.shape[-1]
        if query.dtype != torch.float16:
            raise _OutOfScope(f"dtype {query.dtype} not supported (fp16 only at v1.0)")
        if S * D > 8192:
            raise _OutOfScope(f"S*D = {S}*{D} = {S*D} > UB budget 8192")
        if key.shape[1] != query.shape[1]:  # GQA case
            raise _OutOfScope(f"GQA reference not implemented (query heads={query.shape[1]}, key heads={key.shape[1]})")
        # ... kernel call below this line
```

**Why this matters**:
- Gate 5 (fa_class_scope_out_of_scope_sentinel, OL-183) catches every `EVAL_ERR` case whose error string lacks `_OutOfScope`. A generic `RuntimeError` from the reference (e.g. `aten::matmul` size mismatch) makes the gate fire = the agent didn't think about scope boundaries.
- Empirically verified 2026-05-23 in `workspace/3_FusionAttention`: 5 GQA cases (37, 38, 39, 40, 41) currently crash with `RuntimeError: Expected size for first two dimensions of batch2 tensor to be...` instead of declaring `_OutOfScope` for missing `repeat_interleave`. Gate 5 flags all 5.
- Fix is at the `model_new_ascendc.py` boundary, NOT the kernel boundary. The kernel doesn't need to know about GQA scope — model_new's forward() raises `_OutOfScope` first.

**Anti-pattern**:
- Catching `RuntimeError` and re-raising as `_OutOfScope` — that LAUNDERS unexpected crashes as deliberate scope skips. Gate 5 can't distinguish; reviewers might. Don't do it. Only raise `_OutOfScope` for KNOWN unsupported regimes the agent explicitly chose to skip at v1.0.
- Soft-failing with a Python warning and continuing — kernel still runs on unsupported input, produces garbage, hides the scope problem in noisy output. Always raise.

**Application**: For ANY FA-class workspace where v1.0 scope is narrower than the fixture, the kw worker MUST emit explicit `_OutOfScope` raises for every unsupported axis (dtype, layout, S*D, S*Skv, GQA, pse, sparse_mode, etc.) BEFORE the kernel call. Without this, gate 5 will fire on archive promote.

**Cross-ref**: OL-183 (the gate matrix companion), `docs/design/FA_CLASS_PROBLEM_SOLUTION_DESIGN.md` §3.1 (v-table scope boundaries by phase).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-184（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

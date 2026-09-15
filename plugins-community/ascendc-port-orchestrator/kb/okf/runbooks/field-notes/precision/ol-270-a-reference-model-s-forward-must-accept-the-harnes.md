---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A reference model's `forward()` must accept the harness case-dict kwargs shape, not only the flat positional signature — else native/cpu-truth provisioning fail-closes silently and grading degrades to a stricter fallback"
description: "<!-- applies_to_backend: all -->"
phenomenon: precision_issue
signal:
  - "<!-- applies_to_backend: all -->"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-270
timestamp_inferred: true
tags: [provision_native_capture, case, ascendc, ol-270]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all; phase=provisioning`
`verified_on: soc=Ascend950PR; cann=9.0.0 (rms_norm port_a3_to_a5)`

**Principle**: the harness's native / cpu-truth provisioners (`provision_native_capture` and the CPU-truth path) call the reference model as `model(**case)`, where `case` is the full case_gen record `{idx, name, shape, inputs:{...}, meta}` — NOT the op's flat positional signature `(x, gamma, epsilon)`. A `forward()` written to accept ONLY the flat positional args raises `unexpected keyword 'idx'`, the provisioner **fail-closes** (native_capture absent), and grading silently falls back to a stricter baseline truth source. For a near-zero / tight-tolerance op this can quietly regress the carve-out; the failure is invisible because it looks like "native capture just wasn't produced."

**Contract to satisfy** — the model's `forward()` must accept BOTH call shapes:
```python
def forward(self, x=None, gamma=None, epsilon=None, *, inputs=None, **kwargs):
    if inputs is not None:            # harness case-dict call: model(**case)
        x = inputs["x"]; gamma = inputs["gamma"]; epsilon = inputs.get("epsilon", eps_default)
    # ... flat positional call (x, gamma, epsilon) also works
```
Absorb the record's extra top-level keys (`idx`, `name`, `shape`, `meta`) via `**kwargs`, and unpack the nested `inputs` dict.

**Detection**: `provision_native_capture` logs `unexpected keyword 'idx'` (or another record key) and produces no native_capture; downstream grading is running against the fallback truth source rather than the native capture you intended.

## Evidence
- rms_norm kw-1 (2026-07-02, A5 Ascend950PR_957b, port_a3_to_a5, CANN 9.0.0): model.py initially accepted only `(x, gamma, epsilon)`; `provision_native_capture` fail-closed on `unexpected keyword 'idx'`. Fixed by absorbing `**kwargs` + unpacking the nested `inputs` key. All 33 cases still PASSed T1 strict here (so no observed regression), but for a near-zero op the silent fallback would have degraded the fp32 carve-out.

## Other instances (predicted)
Any op-gen mode that authors a fresh reference `model.py` / `model_new_ascendc.py` and relies on the native-capture or CPU-truth provisioner: port_a3_to_a5 CPU-truth-deferred, backward-gen, cross-gen-port. Also any harness that dispatches the reference via `model(**case_record)` rather than positional args.

Cross-ref: OL-97 (cpu-truth side), OL-262 (verify the reference/truth-source before blaming the kernel — same "a silently-wrong truth source masquerades as a kernel result" family).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-270（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

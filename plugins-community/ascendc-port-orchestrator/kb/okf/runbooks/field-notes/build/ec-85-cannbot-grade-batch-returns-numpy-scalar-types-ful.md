---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "cannbot `grade_batch` returns numpy scalar types + full per-case arrays → `TypeError: not JSON serializable` and a 100s-of-MB verdict — coerce recursively to native + trim to a scalar summary before persisting"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (grading-harness plumbing, SoC-independent)"
phenomenon: build_failure
signal:
  - "persisting verification.json after grading a batch through cannbot grade_batch raises TypeError: Object of type bool_ / float64 is not JSON serializable; separa"
confidence: single_run
original_id: EC-85
timestamp_inferred: true
tags: [grade_batch, ascendc, ec-85]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=all (grading-harness plumbing, SoC-independent)`
`verified_on: soc=Ascend950PR (selective_scan_full_grad kw-1 2026-06-18 / A5)`

- **Symptom**: persisting `verification.json` after grading a batch through cannbot `grade_batch` raises `TypeError: Object of type bool_ / float64 is not JSON serializable`; separately, the raw grade result embeds full per-case arrays that bloat the verdict file (observed ~329 MB) with data that does not belong in a verdict artifact.
- **Root cause**: `grade_batch` returns numpy scalar types (`np.bool_`, `np.float64`) and full per-case numpy arrays inside its result dict; the stdlib `json` encoder cannot serialize numpy scalars, and the arrays are unbounded in size.
- **Fix**: (1) recursively coerce the result with a `_native()` helper before `json.dump` — `np.generic → .item()`, `np.ndarray → summary stats or drop`, recurse into `dict`/`list`; (2) trim per-case arrays to a scalar summary (pass/total counts + representative stats), not the raw arrays.
- **Detection**: a `json.dump(grade_result)` on a cannbot output raising the numpy-type `TypeError`, or a `verification.json` in the 100s-of-MB range.
- **Cross-ref**: OL-272 (backward-mode pybind build/deploy — same class of backward-mode harness plumbing carve-outs the worker must handle itself). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/（EC-85，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

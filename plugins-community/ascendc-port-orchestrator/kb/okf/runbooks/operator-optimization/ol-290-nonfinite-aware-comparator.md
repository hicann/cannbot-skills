---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Short-circuit exact non-finite equality before tolerance arithmetic"
description: "A diff-based comparator rejects matching infinities because inf-inf is NaN; match signed infinities and policy-allowed NaNs first, then compare finite pairs."
paradigm: ascendc
confidence: single_run
original_id: OL-290
timestamp_inferred: false
tags: [ascendc, verifier, precision, infinity, nan, allclose, ol-290]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=precision verifier/backward truth; backend=ascendc`

## Principle

`abs(actual - expected) <= atol + rtol*abs(expected)` falsely rejects matching infinities because `inf-inf` is NaN. Build an exact-match mask first; include matching NaNs only when the declared policy allows it. Apply tolerance only to remaining finite pairs. Finite/non-finite mismatch and opposite-sign infinity always fail.

```python
a = actual.float()
e = expected.float()
both_nan = bool(equal_nan) & torch.isnan(a) & torch.isnan(e)
exact = (a == e) | both_nan
finite_pair = torch.isfinite(a) & torch.isfinite(e)
within = finite_pair & ((a - e).abs() <= atol + rtol * e.abs())
ok = exact | within
```

Report finite error metrics separately from NaN/+inf/-inf counts. This implements the canonical INF/NAN precision policy; it is not a waiver for unexpected non-finite output.

**Evidence / provenance**: derived from historical card TR-OL-27. On 5_Cumsum (2026-05-20), four fp16 large-magnitude cases had bit-equal non-finite outputs but a diff-only comparator failed; exact-first comparison changed Pass B 104/108→108/108 while Pass A stayed 51/51. The arithmetic mechanism is IEEE-derived; the correction is measured.

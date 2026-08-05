---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Form a safe GM address before any potentially out-of-bounds load"
description: "Branch before the read or clamp coordinates before address formation and select the fill value afterward; a false output predicate does not sanitize an invalid address."
paradigm: ascendc
confidence: single_run
original_id: OL-287
timestamp_inferred: false
tags: [ascendc, memory-safety, bounds-check, gather, padding, ol-287]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=padding/gather/interpolation/windowed op; backend=ascendc`

## Principle

A false output predicate is not proof that an already-formed invalid GM address is safe. Either guard the GM read with control flow, or compute `in_bounds`, clamp every coordinate before forming the address, read from the safe address, and select the fill value afterward.

```text
in_bounds = all(0 <= coord[d] < shape[d])
safe[d]   = clamp(coord[d], 0, shape[d] - 1)
loaded    = GM[ravel(safe)]
result    = in_bounds ? loaded : fill_value
```

Handle empty dimensions separately. Test negative, exact-upper-bound, far-OOB and valid-boundary coordinates. Applies to pad, gather/scatter validation, convolution/pooling windows, interpolation and sparse-gradient paths.

**Evidence / provenance**: derived from historical card TR-OL-16. The safe-address implementation passed 40/40 15_Pad cases on 2026-05-17, including a logical coordinate of −4. That validates the safe formulation; the unsafe alternative was not executed, so no backend-specific masked-load lowering claim is retained.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "AIV-utilization activation gate — parallelism-redistribution only helps when B < TOTAL_AIV"
description: "Any optimization that redistributes work within the AIV pool (multi-AIV-per-row, K_ROWS_PER_AIV outer fusion) is inert unless AIVs are under-utilized at baseline; check ceil(B/TOTAL_AIV) host-side before writing the rewrite."
confidence: single_run
original_id: OL-124
classified_by: llm-assisted
timestamp_inferred: true
tags: [aiv-utilization, optimization, ol-124, parallelism, activation-gate]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**Activation gate (the principle).** Any kernel optimization that redistributes work *within* the existing AIV pool — making each row use multiple AIVs (multi-AIV-per-row), having each AIV process multiple rows per launch (K_ROWS_PER_AIV outer fusion), or any parallelism-shape change — only delivers measurable speedup when AIVs are actually under-utilized at baseline. Pre-derivation check:
```
rows_per_aiv_baseline = ceil(B / TOTAL_AIV)
if rows_per_aiv_baseline >= 1 across ALL test cases:
    parallelism-redistribution is INERT — every AIV is already busy
elif min(rows_per_aiv) < 1:
    redistribution may help — proceed with Mechanism A or B
```

**Pre-implementation check (host-side, BEFORE writing any kernel rewrite):**
```python
import json
cases = [json.loads(l) for l in open("vendor/.../<op>.json")]
ratios = [num_rows(c) / TOTAL_AIV for c in cases]   # 56 on a5 V220
print(f"min={min(ratios):.2f} median={sorted(ratios)[len(ratios)//2]:.2f}")
# min >= 1.0 across all cases → every AIV has ≥1 row → redistribution inert
```
If the gate fails (every benchmark case has rows ≥ TOTAL_AIV), do NOT spend iter budget on parallelism-redistribution rewrites — the mechanism can't fire. Document the inertness as a structural ceiling and pursue a different axis (per-row scalar overhead, MTE2/VEC overlap, fused-operator extraction).

**Mechanism A — Multi-AIV-per-row partition** (P AIVs cooperate on 1 wide row):
```
P_effective = min(P_target, max(1, TOTAL_AIV / B))
```
If `B ≥ TOTAL_AIV` (e.g. B=640, TOTAL_AIV=56) every AIV is already busy on different rows; no AIV is free to partner and the partition path degenerates to the single-AIV-per-row fallback. Adding the partition machinery (Phase-1 partial-store + `SyncAll<true>()` Phase 1.5 + Phase-2 tournament merge) without the parallelism gain is a net regression: tournament-merge tax with zero benefit.

**Mechanism B — K_ROWS_PER_AIV outer-loop fusion** (1 AIV processes K rows per launch to amortize aclrtLaunchKernel overhead): same gate. If every AIV already has `my_rows_ ≥ 1` from the existing block-stride scheduler, wrapping the inner per-row loop with an outer K-stride is algebraically a no-op — the inner loop already iterates `my_rows_` times per launch. Compile-time K-fusion only adds extra K iter steps to the same code path. It only pays off when the baseline scheduler leaves AIVs idle (rows < AIVs).

**Decision rule (host pybind side, NOT kernel runtime):**
- `B < TOTAL_AIV / P_target` → `P = P_target` (full partition gain)
- `TOTAL_AIV / P_target ≤ B < TOTAL_AIV` → `P = floor(TOTAL_AIV / B)` (partial, degenerate gain)
- `B ≥ TOTAL_AIV` → `P = 1` (single-AIV path; partition won't fire)

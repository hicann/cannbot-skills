---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Always msprof the reference op BEFORE writing structural-impossibility framing — meta-process for perf claims"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "about to write a claim of the form \"perf X is unreachable because of structural property Y\" / \"no Kind-1/Kind-2 path closes the gap\" / \"ko-N plateau'd at Z, acc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-117
timestamp_inferred: true
tags: [ascendc, ol-117]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / orchestrator-discipline
- **Loaded by**: orchestrator (BEFORE writing `.workflow_exception_O5*` / `.orchestrator_skip_*` / any waiver document with bar-lowering language about perf)
- **Trigger**: about to write a claim of the form "perf X is unreachable because of structural property Y" / "no Kind-1/Kind-2 path closes the gap" / "ko-N plateau'd at Z, accept honest limitation" — applied to any op with a CANN reference

### Principle

Structural-impossibility claims about perf MUST be grounded in per-pipe attribution data from msprof on BOTH our kernel AND the reference. Without that evidence, the claim is assertion-only — a reward-hacking shortcut that lets the orchestrator stop iterating without proving the iteration would not pay off.

The 5-minute msprof on the CANN reference op (`aclnnXxx`) is the load-bearing artifact:
1. **BlockDim** — same architecture as ours, or different?
2. **AIV/AIC mix** — pure VEC, pure cube, or mixed? (Tells you which pipe is the bottleneck.)
3. **aiv_vec_ratio** — how much of cycles spent in VEC compute? Compare to ours.
4. **aiv_mte2_ratio** / **aiv_mte3_ratio** — how aggressively does the reference stream memory? (HIGH MTE2 ratio + HIGH VEC ratio simultaneously = good pipeline overlap, NOT excess bandwidth.)
5. **aiv_scalar_ratio** — scalar overhead vs ours.
6. **Duration** — absolute wall-clock, sanity check.

The gap between our and reference's per-pipe ratios IS the directive for the next ko iter. "We have 0.43 MTE2 vs CANN's 0.75 → pipeline is exposed → manual prefetch (OL-115) is the lever" is a concrete, falsifiable directive. "Two-pass tile loop has 2× MTE2 tax" without msprof is a guess.

### Mandatory protocol when about to write a "perf unreachable" claim

1. Run msprof on the CANN reference op for at least one representative case from the failing-perf set.
2. Run msprof on our current kernel for the same case.
3. Tabulate per-pipe ratios + duration side-by-side. Save as `workspace/{op}/probes/probe_outputs/cann_reference_msprof_summary.md` (or similar).
4. Identify the SPECIFIC pipe where we under-utilize relative to CANN — that's the directive for ko-2/ko-3.
5. If after attribution you still cannot identify a specific lever, escalate to `/aog-hardware-probe` for an empirical capability test; do NOT label "structural" without per-pipe evidence.

### Concrete anchor (29_DynamicQuant ko-1 → ko-2 transition)

```
Before msprof on CANN reference (orchestrator's WRONG framing):
  "CANN is single fused; two-pass = 2× MTE2 tax; structural impossibility"
  → about to write .workflow_exception_O5

After 5-min msprof (reality):
  CANN BlockDim=56 AIV (same as ours), Cube=0% (same)
  CANN aiv_vec_ratio=0.82, aiv_mte2_ratio=0.753 (HIGHER than ours, not lower)
  → gap is pipeline-OVERLAP, not memory-tax
  → directive: manual prefetch (OL-115) + scalar lift (OL-116)
  → ko-2 single iter: 0.339× → 0.609× (+79%, crosses threshold)

Cost: 5 min to gather data; saved a wrong customer-archive artifact and unlocked +20% on the kernel.
```

### Evidence

- 29_DynamicQuant ko-1 → ko-2 (2026-05-02): orchestrator was about to ship `.workflow_exception_O5` claiming "structural 2× MTE2 tax"; aog-self-critic invoked at user prompt → 4 BLOCKS (C5 platform-blame, C13 claim without verification, C14 single-agent customer-archive write, C20 available tool not used). msprof on CANN reference disproved every claim. ko-2 with prefetch directive crossed 0.6× threshold in single iter.

### Other instances (predicted)

Any op-gen session where ko plateaus and orchestrator considers documenting "honest limitation" framing. Specifically high-risk:
- Ops with CANN-fused references where "CANN is fused, we're not" temptation is strong (almost all our op-gen ops have CANN-fused references)
- Ops where ko has 3+ iters of marginal gains (orchestrator naturally drifts toward "diminishing returns" framing)
- Ops where honest A/B vs profiler-artifact perf differ significantly (orchestrator drift toward profiler-artifact value to claim PASS — caught 3 times this session as bar-lowering pattern)

### Related

- aog-self-critic catalog C5 (premature platform-blame) / C13 (claim without verification) / C14 (single-agent customer-archive write) / C20 (available tool not used) / C25 (premature stop after root cause)
- CLAUDE.md "Independent Performance Verification" — performance.py profiler-artifact values are NOT canonical; honest A/B (perf_counter) is
- OL-115 (manual prefetch — typically the directive that emerges from msprof-on-reference)
- OL-116 (loop-invariant lifting — typically a co-directive)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-117（category=process / orchestrator-discipline，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

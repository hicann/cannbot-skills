---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "On-device pilot before applying any probe directive — CPU-fp32 simulation does not bit-equal NPU-fp32 unit ordering"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "precision-probe writes a probe_report.md with a projected outcome (case-by-case MARE delta or PASS prediction) and a kernel-side directive"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-111
timestamp_inferred: true
tags: [ascendc, ol-111]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Category**: process / probe-worker handoff
- **Loaded by**: aog-kernel-worker (when receiving Kind-1 fix directive from probe), aog-precision-probe (when projecting fix outcomes)
- **Trigger**: precision-probe writes a probe_report.md with a projected outcome (case-by-case MARE delta or PASS prediction) and a kernel-side directive

### Principle

A precision-probe's CPU-fp32 simulation of an alternative reduction algorithm (e.g. pairwise tree vs linear) is **necessary but not sufficient** evidence that the directive will close the projected cases on the actual NPU. When the worker implements the directive and measures the result, the per-case MARE outcome may differ from the projection by ~10× (or change sign) for two reasons:

1. **NPU fp32 vector unit ordering** can differ from CPU fp32 (lane-pair grouping, FMA contraction, intermediate-rounding rules at instruction boundaries).
2. **AscendC API patterns** (e.g. binary-counter pairwise via TBuf array vs halving pairwise via in-place `Adds` over shrinking range) lower to different machine sequences than the simulated form.

### Mandatory protocol when receiving a probe directive

1. **Implement the directive faithfully** (read probe_report.md "Bit-level measurements" section first).
2. **Build + verify on actual NPU** — Pass A inline + Pass B edge_dataset.
3. **Compare measured vs projected outcomes case-by-case**:
   - If measured ≈ projected → directive validated, may declare DONE.
   - If measured ≠ projected on ≥1 case (especially regression on a case currently passing baseline): **anti-overfit gate fires** — REVERT to baseline, classify all failing cases as an fp32 reduction-order residual, write knowledge_update.md noting the projection-vs-measurement gap.
4. **Never declare a probe-directive fix successful by code reading alone.** The build-verify cycle is non-negotiable.

### Concrete anchor (named-op)

19_FusedResidualRmsNormBackward kw-3 (2026-05-01): probe `frrmsnb-pp-1` projected `tree_seq` would close cases 23+33 (case 33 projected MARE 1.19e-3 PASS). Worker implemented K1 binary-counter pairwise tree across rows, measured: case 33 STILL FAIL at MARE 9.05e-3 (~7.6× higher than projection); case 14 worsened 1.52e-3 → 1.10e-2; case 43 NEW REGRESSION (MARE 4.28e-3, was passing baseline). Anti-overfit REVERT to linear-linear baseline 47/50.

### Evidence

- 19_FusedResidualRmsNormBackward (2026-05-01, kw-3): projection vs measurement diverged on 3 of 3 affected cases. CPU-fp32 simulation in probe iter2 used halving pairwise; worker's NPU implementation used binary-counter pairwise (different lowering, different rounding sequence). REVERT applied per OL-85.

### Other instances (predicted)

Any precision-probe directive that projects MARE/MAD outcomes from a Python/numpy/torch CPU-fp32 simulation. Particularly affected:
- Reduction-tree restructures (this OL's anchor case)
- FMA grouping changes (`Mul → Adds` order swaps)
- Intermediate dtype changes (fp16 → fp32 promotion in mid-pipeline)
- Iterate vs IterateAll cube-vec coordination changes

### Related

- OL-85 (anti-overfit gate — provides the REVERT verdict mechanism)
- OL-110 (reduction-tree fail-floor invariant — the "why measured diverges" root cause)
- C14 (self-critic: KB/DEBT writes from single-agent data require independent re-verification — this OL is the worker-probe analogue)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-111（category=process / probe-worker handoff，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

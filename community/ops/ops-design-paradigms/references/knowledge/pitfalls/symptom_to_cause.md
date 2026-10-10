---
name: regbase_symptom_to_cause
description: Symptom-driven lookup for mapping regbase runtime and review failures to the most likely root causes and first checks.
title: Symptom to Cause
purpose: Start from the visible failure shape and branch quickly into the most likely root causes and next reads.
read_when:
  - The symptom is clearer than the mechanism.
  - You need a first triage path before choosing a narrower pitfall note.
not_for:
  - Greenfield design
  - Exhaustive algorithm analysis
keywords:
  - symptom
  - root cause
  - triage
  - failure signature
next_reads:
  - common_traps.md
  - precision_failures.md
  - api_misuse.md
depth: foundation
topic_type: pitfall
type: knowledge_card
platform: ascend950-regbase
tags: [symptoms, root_cause, triage, debugging, regbase]
---

# Symptom to Cause

Use this lookup when the symptom is clearer than the mechanism. Start with the most visible failure shape, run the first checks, then branch into the detailed pitfall card.

| Symptom | Likely Root Cause | First Checks | Next Read |
|---|---|---|---|
| Output all zero or nearly all zero | wrong branch path, copy path not doing what the design assumes, invalid denominator guard collapsing too much, stale or missing compute stage | confirm the task is still on regbase semantics, inspect the first non-trivial intermediate, check whether zeroing happens before or after compute | [[api_misuse]], [[regbase_vs_membase_confusions]] |
| Random garbage values or huge spikes | wrong API family, wrong dtype interpretation, unguarded reciprocal or divide, sync issue that looks numeric | check whether failures move between runs, check cast and branch assumptions before touching tolerances | [[api_misuse]], [[../api/regbase_api_sync]] |
| Small shapes pass, large shapes fail | accumulation drift, tile-merge bug, hidden tail bug, range growth triggering overflow | compare aligned short-axis and long-axis cases, locate the first bad merge or tail | [[precision_failures]], [[common_traps]] |
| Only tail elements are wrong | mask or tail handling issue, alignment assumptions, incorrect valid-length propagation | find the first bad index and compare it to tile boundaries | [[api_misuse]], [[precision_failures]] |
| `fp32` passes but `fp16` or `bf16` fails | promotion placed too late, long reduction in low precision, unstable nonlinear chain | compare the first divergent intermediate between dtypes | [[precision_guide]], [[precision_failures]] |
| Softmax or probability row does not sum near 1 | missing max-shift, unstable `exp` chain, wrong reduction width, cast-down too early | inspect row max, shifted values, exponential sum, and final normalization | [[precision_failures]], [[precision_guide]] |
| Norm or variance path gives `nan` or very large values | negative variance from drift, epsilon added too late, reciprocal or `rsqrt` on tiny value | inspect pre-`rsqrt` values, check epsilon placement | [[precision_failures]], [[precision_guide]] |
| Diff sign is mostly consistent across the tensor | scale, offset, cast, or constant-factor mistake rather than random precision noise | inspect constant parameters and cast mode assumptions | [[api_misuse]], [[precision_failures]] |
| Mismatch location changes between runs | synchronization or branch-selection issue, not a stable numerical formula issue | rerun the same input, compare failing indices, inspect sync assumptions | [[../api/regbase_api_sync]], [[common_traps]] |
| Review says "fell back to default template thinking" | regbase task was designed from membase template habits instead of regbase knowledge | identify where queue/pipeline assumptions replaced regbase semantics | [[regbase_vs_membase_confusions]], [[api_misuse]] |

## How To Use This Table

1. Choose the row that best matches the first visible failure.
2. Run the first checks before editing code.
3. If multiple rows fit, prefer the one that explains the earliest divergence, not the loudest final symptom.
4. If the failure shape changes between reruns, treat synchronization or branch confusion as higher priority than pure precision tuning.

## Related Documents

- [[common_traps]]
- [[precision_guide]]
- [[api_misuse]]
- [[precision_failures]]
- [[regbase_vs_membase_confusions]]

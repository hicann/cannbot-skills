---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Using TBuf for loop-persistent state is safe (does not contradict PB-11)"
description: "Recurrent state kept in a TBuf<VECCALC> and updated in-place via VEC ops across a loop is safe — PB-11's stale-data warning only covers the DataCopy(GM→TBuf) repeated-reload case, not VEC-produced state."
confidence: single_run
original_id: OL-78
classified_by: llm-assisted
timestamp_inferred: true
tags: [pipeline-design, optimization, ol-78, tbuf, recurrent, pb-11]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Context:** a recurrent kernel where a state buffer lives in UB and is updated in-place via VEC ops across the full loop, rather than being reloaded from GM each iteration.

PB-11 ("TBuf data may be stale across multi-iteration loops") refers specifically to the **DataCopy(GM→TBuf) repeated-reload** scenario. It does NOT apply to recurrent state that is produced by the current iteration's own VEC write.

**Rule:**
- Keeping recurrent state in `TBuf<VECCALC>` and updating it each iteration via VEC ops (Exp/Mul/Add/Max/…) writing back into the same TBuf is **safe** — the data is always produced by the current iteration, so there is no stale-data problem.
- The condition that triggers PB-11 is: do NOT `DataCopy(GM→TBuf)` into the same TBuf inside the loop. That reload path is what goes stale.

So: compute-and-write-back into a TBuf is safe; reload-from-GM into a TBuf inside the loop is the PB-11 footgun.

### Evidence — the two cases mutually confirm PB-11's exact boundary

- **Safe (VEC-produced state):** 30_TimeDecayExponentialStabilization keeps `max_state` / `num_state` / `den_state` as `TBuf<VECCALC>`, updated each step in the seq_len loop via Exp+Mul+Add+Max → 50/50 PASS.
- **Unsafe (GM reload):** 19_FusedResidualRmsNormBackward reloaded the weight TBuf from GM inside the loop and triggered PB-11 (7 cases FAILed).

E3 level.

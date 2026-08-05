---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Vector Counter mode replaces Normal mode to simplify tail handling"
description: "Vector Counter mode lets you specify total element count directly; hardware auto-handles main/tail block mask and iteration, avoiding manual repeatTimes/mask/tail handling. May simplify unaligned scenarios vs Normal mode."
confidence: single_run
original_id: OL-51
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-51, counter-mode, tail-handling, vec, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: VEC ops need to handle unaligned tails (currently using Align8/Align16 + explicit mask management).

**选型**: **Counter mode** lets you specify the total element count directly; the hardware automatically handles the main and tail blocks' mask / iteration. This avoids manually computing `repeatTimes`, setting masks, or handling tails — the code is cleaner and less error-prone. Current kernels all use Normal mode; Counter mode may simplify unaligned scenarios.

**Source**: hiascend.com best practices (2026-04).

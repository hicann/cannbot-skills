---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Early-stopping criteria after an optimization pass"
description: "Stop or escalate when the target kernel exceeds 60% of its theoretical peak or further gains require an algorithmic redesign."
confidence: single_run
original_id: ROOFLINE_MODEL.md#using-the-model-early-stopping
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, early-stopping, escalation]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

After an optimization iteration, stop (or escalate to a human) when any of these hold:

- **Kernel is > 60% of theoretical peak** → consider stopping; remaining headroom is small.
- **Further improvement requires > 2x reduction in data movement** → likely needs an algorithm redesign; escalate to a human rather than micro-optimizing.

These gate the optimization loop so effort isn't spent grinding against a hardware or algorithmic wall once the roofline says the kernel is already near its ceiling.

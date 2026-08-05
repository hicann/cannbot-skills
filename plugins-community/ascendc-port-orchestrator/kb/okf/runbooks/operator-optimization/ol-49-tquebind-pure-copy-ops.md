---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Pure-copy ops — TQueBind replaces the VECIN→VECOUT bridge"
description: "Pure-copy ops (Cat/Split/Pad/Repeat) can bind VECIN with VECOUT via TQueBind and skip the Adds(0.0f) bridge (EC-21 workaround) for higher efficiency. UNVERIFIED on A5 — needs experimental confirmation on CANN 9.0.0."
confidence: single_run
original_id: OL-49
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-49, tquebind, pure-copy, unverified, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: data-movement ops (Cat / Split / Pad / Repeat) with no real VEC compute.

**选型**: Pure-copy ops can bind VECIN with VECOUT using the **TQueBind** interface and skip the `Adds(0.0f)` bridge. Current Cat/Split/Pad all use the Adds bridge (EC-21 workaround; see OL-37); TQueBind is a more efficient alternative.

**Status**: **UNVERIFIED on A5** — need to experimentally confirm the TQueBind API works on CANN 9.0.0 before adopting.

**Source**: hiascend.com best practices (2026-04).

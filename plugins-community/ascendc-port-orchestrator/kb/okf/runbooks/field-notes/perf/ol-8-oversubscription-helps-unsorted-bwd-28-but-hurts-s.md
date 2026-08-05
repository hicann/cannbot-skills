---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Oversubscription helps unsorted bwd (-28%) but hurts sorted bwd (+17%)"
description: "Block oversubscription (nblk=448 vs 56) reduced unsorted backward time by 28% by dispersing atomicAdd contention across more time-sliced blocks. But after implementing sorted-edge register accumulatio"
phenomenon: perf_regression
signal:
  - "when tuning launch parameters (nblk) for scatter-add backward kernels"
confidence: inferred
classified_by: llm-assisted
original_id: OL-8
timestamp_inferred: true
tags: [ascendc, ol-8, llm-classified]
created_at: 2026-07-10T16:00:00Z
updated_at: 2026-07-10T16:00:00Z
---
## 现象 / 触发
when tuning launch parameters (nblk) for scatter-add backward kernels

## 教训 / 根因
Block oversubscription (nblk=448 vs 56) reduced unsorted backward time by 28% by dispersing atomicAdd contention across more time-sliced blocks. But after implementing sorted-edge register accumulation (which eliminates atomicAdd contention entirely), the same oversubscription increased backward time by ~17% due to pure launch overhead with no contention left to disperse. The root cause is that oversubscription's benefit comes solely from dispersing atomicAdd contention -- once sort+register-accum removes that contention, only the overhead remains. Launch parameter sweeps must be re-run on the final kernel variant, not carried over from intermediate versions.

## 证据
EXPERT_FEEDBACK.md E7-8 (nblk sweep tables for unsorted vs sorted), MSPROF_AGENT_GUIDE.md (nblk=56 vs 448 msprof data)

<!-- LLM-辅助分类迁移(convert 二次 pass, migrate_ol_dispositions.py)。confidence=inferred,待人工确认。原 OL-8。 -->

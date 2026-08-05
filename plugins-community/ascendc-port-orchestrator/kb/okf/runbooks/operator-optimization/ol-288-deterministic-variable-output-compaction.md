---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Deterministic variable-size output uses count, exclusive offsets, then ordered emit"
description: "For data-dependent output length, count per fixed partition, prefix the counts, then emit each partition in local input order to a disjoint output range."
paradigm: ascendc
confidence: single_run
original_id: OL-288
timestamp_inferred: false
tags: [ascendc, stream-compaction, prefix-sum, determinism, variable-output, ol-288]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=stream compaction/data-dependent output; backend=ascendc`

## Principle

When output length depends on input values: (1) each fixed input partition writes its count; (2) compute an exclusive prefix over counts and allocate the exact output; (3) each partition recomputes its predicate and emits in local input order into its disjoint range. A single global atomic counter loses stable order and makes allocation schedule-dependent.

The prefix may be a small host metadata operation or a dedicated device scan, but it must not delegate the input-to-output data path. During arch22→arch35 migration, preserve partition order and the count/emit predicate exactly. Validate empty output, all-selected, partition boundaries, ordering and rerun determinism. Cross-ref P-P80/P-P83 and P-P61.

**Evidence / provenance**: derived from historical card TR-OL-25. An A3 22_Nonzero prototype measured on 2026-05-17 achieved Pass A 50/50, Pass B 10/10 bit-exact, determinism 50/50 and 1.094× using this decomposition. Historical device API details are not transferred; AscendC mechanics are grounded by P-P80/P-P83.

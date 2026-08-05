---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Precision evidence: CPU truth and source-arch behavior capture"
description: "CPU fp64 anchors the declared math while an authorized source-arch run records observable migration behavior; target prior-art may inform generation but is not final truth."
phenomenon: precision_issue
signal:
  - "the generated result agrees with CPU truth but differs from the source-arch capture, or vice versa"
confidence: single_run
original_id: OL-97
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-97, cpu-truth, source-capture, verification]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 现象 / 触发

A generated operator agrees with CPU fp64 truth but differs from an authorized arch22 behavior
capture, or agrees with the capture but violates the declared equation.

## 根因 / 教训

Treat these as complementary evidence channels. CPU fp64 anchors the mathematical contract. A run
of the user-selected operation on the source-arch NPU records observable migration behavior,
including attributes, optional inputs, output structure, and numerical conventions. Record the
reference source in the manifest and classify disagreements by region, reduction order, and dtype.

An existing arch35 implementation, installed target operator, target binary, profile, or archive
may be consulted as prior-art for generation and research. It is not migration or backward truth;
thresholds must come from the declared operator contract rather than an unrelated verifier.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Probe option-sensitive CPU and NPU reference semantics before deriving the formula"
description: "Measure CPU-vs-NPU and NPU option-A-vs-option-B behavior first; framework bindings may ignore, gate or collapse an option before device dispatch."
paradigm: ascendc
confidence: single_run
original_id: OL-292
timestamp_inferred: false
tags: [ascendc, reference, truth, option-flag, preflight, backward, ol-292]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---

`applies_to: soc=all; cann=all; op_class=reference/truth preflight; backend=ascendc`

## Principle

A framework binding may ignore, gate or collapse a documented option before reaching the device primitive. Before generating a forward or backward candidate, evaluate at least two option values on identical inputs and compare CPU-vs-NPU plus NPU-option-A-vs-option-B. If options collapse, record the measured behavior and identify the required authority: API mathematical truth, declared test reference behavior, or shipped-device parity. Never substitute one silently.

Save inputs, option values, outputs and framework/CANN versions. For optional-tensor-gated features also vary tensor presence (OL-202). Differentiate backward truth from the exact measured forward semantics. Cross-ref OL-89/OL-109.

**Evidence / provenance**: derived from historical card TR-OL-9. On A3 (2026-05-04, torch_npu 2.7.1/CANN 8.3), `gelu(approximate=none)` differed from CPU/textbook erf by 4.74e-4 but matched tanh approximation within 4.77e-7; two option values were bit-identical on NPU while CPU distinguished them by 4.73e-4. The measurements are empirical; declaring truth authority is the process conclusion.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Whole-port precision gaps concentrate in the host feature-dispatch layer, not the kernel"
description: "In generated multi-feature ops, feature-variant gaps often trace to under-wired host dispatch; inspect current generated code and selected-source contract first."
confidence: single_run
original_id: OL-205
classified_by: llm-assisted
timestamp_inferred: true
tags: [feature-dispatch, optimization, ol-205, whole-port, host-wiring, precision-gap]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.0.0 / generated cube+vec-fused multi-feature ops. Verified on FA-A5 GQA, 2026-06-02.

**Principle**: the current generated device code may implement a feature while the task-owned host
launcher fails to pass or dispatch it (GQA, pse, mask, dtype). Symptom: common cases pass and feature
variants fail. Inspect the current generated code and selected-source contract before blaming device
math. Target prior art is advisory only, not a body-copy source or truth oracle.

**Diagnostic rule (any whole-port multi-feature op)**: when a feature-variant case fails, FIRST `grep` the kernel headers (`wp_*.h` / `op_kernel/`) for that feature's machinery — the grouping-index decompose, the `hasPse`/`hasAtten` template branch, the info-struct type, the dtype path.
- (a) Machinery EXISTS → the gap is host-wiring: add the host param + fill the info struct + dispatch to the feature launcher (cheap, host-only, no kernel change).
- (b) Machinery ABSENT → kernel rewrite (structural, expensive — e.g. a missing fp32 vec path).

Do NOT conclude "kernel bug / hard floor" before this grep — the cheap host-only fix is the high-probability path.

**Two host-fix caveats**:
1. The host fix is not always pure wiring — it can include a NUMERICAL-SEMANTICS adaptation (e.g. pre-scaling a bias/sink term when kernel and oracle disagree on whether it lives in raw or scaled space).
2. The feature's SEMANTICS (enum type, layout, scale-space) AND the case's DTYPE must be **empirically probed against the oracle / read from the benchmark data** — NEVER inferred from enum-name convention or trusted from a worker's self-reported tag. Evidence: the FA pse fix mis-inferred `pse_type` from convention (MUL_ADD) until an empirical probe refuted it (oracle is ADD_MUL, diff 4e-4 vs 1.04); and a worker mislabeled an fp16 case as bf16, inflating the pass count by passing it under the looser bf16 floor. Read dtype from the `.json`, probe semantics against the oracle.

**Concrete anchor**: FA-A5 GQA — kernel had full n2G head-grouping (`wp_kernel_base.h` `constInfo.n2G = n2Size*gSize` + n2oIdx/goIdx decompose; `wp_block_cube.h:440` K/V offset uses `n2Size` head-stride) but `pybind11_wp.cpp:309-310` hardcoded `n2Size=N; gSize=1` (MHA-only) → n2oIdx ≥ N_kv OOB read of K → NaN. The kernel machinery existed; the host launcher under-wired it.

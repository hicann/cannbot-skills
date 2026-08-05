---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A pybind-wrapped custom-op's e2e wall is not apples-to-apples vs a native vendor op — compare kernel-device time and decompose the wall before blaming the kernel"
description: "A pybind-wrapped kernel's e2e wall carries per-call pybind/sync overhead a native torch_npu op never pays, so the wall ratio under-counts a real kernel win. Report a kernel-device msprof A/B."
phenomenon: perf_regression
signal:
  - "a real kernel-level speedup (msprof task-duration) shows NEUTRAL at the e2e wall when A/B-comparing a pybind-wrapped custom op vs a native vendor op"
confidence: single_run
original_id: OL-201
classified_by: llm-assisted
timestamp_inferred: true
tags: [perf, benchmarking, ol-201, pybind-overhead, kernel-vs-wall, msprof]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
A benchmark times "ours" as a pybind-bound custom kernel (`ModelNew.forward` → pybind marshalling → ACL launch → sync → result copy-back) and "vendor" as a NATIVE torch_npu op (`npu_fusion_attention`, etc. — a C++ dispatch with no pybind layer). A real kernel improvement (msprof task-duration) can be REAL yet **invisible at the wall**, and the ours/vendor wall ratio **under-counts** the kernel's true value.

## 根因 / 教训
The two wall measurements are NOT apples-to-apples: ours carries a per-call pybind/wrapper overhead the native vendor op never pays. **The fair comparison is a kernel-device A/B (msprof task-duration)**, or a native-integration A/B (bind ours as a torch_npu custom op, not the benchmark pybind wrapper).

**Concrete anchor**: FA-A5 path-B (2026-05-31). A mem-VEC softmax rewrite (P-P101) cut kernel-msprof task-duration **−24% (349→264us)** but the same-card e2e wall A/B was **NEUTRAL**. Wall decompose of the BNSD-native case (no host layout-fold): wall 105us = kernel-msprof 36.6us + ~68us non-kernel; that ~68us is ~2× the kernel for a 36us kernel and is dominated by pybind marshalling + per-call sync — overhead `torch_npu.npu_fusion_attention` (native dispatch) does not traverse. The harness confirms it: `perf_ab_main.py` times ours = `model(...)` (ModelNew.forward, pybind+host) vs vendor = `npu_fusion_attention` (native). The directional caveat is verified; the exact pybind-vs-ACL-launch split of the ~68us is **needs-measure**.

**Action rule**: before reporting an "ours is N× vendor" e2e-wall verdict for a pybind-wrapped op, (1) ALSO report a kernel-device msprof A/B; (2) if the wall gap >> the kernel gap, decompose the wall (host pre/post + launch + pybind + sync) and label which components are harness-artifact vs real customer cost; (3) do NOT attribute a wall gap to the kernel without that decomposition. A host layout-transform (e.g. a BNSD-fold `permute().contiguous()` the vendor avoids via a native-layout kernel) IS a real customer cost; the pybind/per-call wrapper overhead generally is NOT (a real customer integration binds the kernel as a native op).

**Cross-ref**: P-P101 (the kernel-side softmax rewrite whose −24% this scopes — kernel≠wall); OL-200 (kernel-side cube/vec pipeline — a different perf axis, also kernel-time not wall).

Verified on soc=Ascend950PR, cann=9.0.0 (FA-A5 path-B same-card A/B, 2026-05-31).

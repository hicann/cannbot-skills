---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "msprof / torch_npu.profiler ANALYZE stage fails `Operation not permitted` (EPERM) in the `.171` npu_dev3 container despite Privileged=true → use NPU `aclrtEvent` elapsed_time device-time as the fallback"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
confidence: single_run
original_id: PB-51
timestamp_inferred: true
tags: [aclrtevent, npu_dev3, msprof, ascendc, pb-51]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`

In the `.171` (203.0.113.171) `npu_dev3` container, both `torch_npu.profiler` and the `msprof` CLI collect raw profiling data but the **analyze/export stage fails with `Operation not permitted` (EPERM)** — so per-launch isolated `kernel_details.csv` device-time is UNAVAILABLE there, even though the container reports `Privileged=true` (the EPERM is from a profiling-syscall/securityfs restriction the privileged flag doesn't cover).

- **Workaround**: measure device-time via NPU events — `aclrtEvent` / `torch.npu.Event(enable_timing=True)` `start/end` + `elapsed_time` around the kernel launch (this is device-side timing, NOT host wall-clock). Validate it once against any host where msprof analyze DOES work (e.g. a committed msprof number) — observed agreement ~7%. The Event device-time is a valid same-host A/B basis (the EPERM is an export-tooling limit, not a timing-accuracy limit).
- **Distinct from the .141 policy**: `.141` FORBIDS msprof by team policy (different reason); `.171` ALLOWS msprof but its analyze stage EPERMs in npu_dev3. So for `.171` perf A/B, use NPU-Event device-time.
- **Status**: OPEN (container/securityfs profiling-syscall restriction).
- **Evidence**: selective_scan bwd perf (2026-06-30, .171 npu_dev3, CANN 9.0.0/9.1.0, PR #71): msprof analyze EPERM on both tools; NPU-Event device-time used + cross-validated ~7% vs the committed msprof figure.
- **Cross-reference**: [[feedback_report_perf_only_from_probe_device_time]] (the device-time-not-wall discipline — NPU-Event IS device-time, satisfies it), OL-245 (the perf A/B this surfaced under).

<!-- 迁移自 porter kb/target/ascendc/（PB-51，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

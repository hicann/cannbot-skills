---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Before escalating a 507035 on a freshly-built kernel as infra, run a read-only vendor-op probe on the SAME device"
description: "507035 on a kernel you just built looks the same whether the device is sick or your kernel faults; run a known-good vendor op on the same device — if it PASSES, the fault is kernel-side, not infra."
phenomenon: build_failure
signal:
  - "device error type 3, error code 507035 (or 507014/507015 siblings) on a freshly-built custom kernel's first cases; host error code does not distinguish device-sick from kernel-bug."
confidence: single_run
original_id: OL-274
classified_by: llm-assisted
timestamp_inferred: true
tags: [507035, build, ol-274, discrimination-probe, infra-triage, p9]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

`device error type 3, error code 507035` on a kernel you just built is the **same host-visible symptom** whether the cause is a sick/contended device (infra) or a UB/GM fault in your own kernel. The P9 anti-pressure reflex ("don't cry flaky infra") is satisfied **not** by asserting the device is fine, but by a cheap read-only discrimination probe.

## 根因 / 教训

Run a known-good vendor op on the **same device** to discriminate. If the vendor op PASSES, the device's full launch→execute→sync compute path is healthy → the 507035 is kernel-side. Only if the vendor op ALSO faults do you escalate to device/contention (then check AICore via `npu-smi` per OL-189). This is a stronger probe than `npu-smi` health alone: `npu-smi` reports board/AICore status, but a passing vendor **compute** op proves the whole compute path is good on that device.

```python
# read-only probe on the same device before blaming infra
import torch, torch_npu
x = torch.randn(4096, device="npu:0")
y = torch.nn.functional.gelu(x).sum()   # vendor op, no custom kernel
torch.npu.synchronize()                  # PASS ⇒ device compute path healthy ⇒ fault is your kernel
```

Once the device is cleared, triage the kernel-side 507035 sub-classes:
- UB overflow ([[OL-273]])
- MTE DDR out-of-range (EC-53)
- DataCopyPad UB→GM on V220 (EC-23)
- unallocated TBuf / missing InitBuffer (EC-62)
- SOC-arg / env mismatch (EC-27)
- small-tensor H2D (EC-56)

### Evidence

- gelu tanh-approx port (2026-07-05, port_a3_to_a5 V220→arch35, A5 Ascend950PR / CANN 9.0.0): first fp32 case faulted `npuSynchronizeDevice ... device error type 3, error code is 507035`. A read-only probe (`F.gelu` + `sum` on device 0) PASSED → device healthy → root-caused to fp32 UB overflow ([[OL-273]]), fixed without any P9 infra escalation or retry-loop. 87/87 PASS after the buffer-budget fix.

### Other instances (predicted)

Any freshly-built or freshly-deployed custom kernel that faults 507035 (or the 507014/507015 aicore siblings) on its first cases: perf-remeasure builds, hardware probes, cold-start regens. The probe generalizes to any accelerator where a vendor op and a custom kernel share a device and the host error code does not distinguish device-sick from kernel-bug.

### Cross-references

- P9 (ANTI_PRESSURE — "don't cry infra"; this entry is the cheap probe that satisfies P9 without asserting device health blindly)
- OL-189 (507015 under NPU contention)
- OL-273 (the fp32 UB overflow this probe localized)

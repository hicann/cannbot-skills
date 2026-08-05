---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Small NPU tensors (<32 bytes) trigger torch_npu Slice kernel crash 507035 — pad tiling buffer to ≥256 bytes"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all"
phenomenon: build_failure
signal:
  - "creating a small NPU tensor (e.g., 12-byte tiling struct via torch.empty(12, dtype=uint8).to(device)) triggers a torch_npu internal Slice kernel crash: aclrtLau"
confidence: single_run
original_id: EC-56
timestamp_inferred: true
tags: [507035, ascendc, ec-56]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 — torch_npu Slice kernel may not have this minimum-size constraint; verify on A5 before assuming pattern applies)`

(ds agent extraction 2026-05-19 from earlier ds-branch commit e13d49db; originally numbered EC-48 in ds branch but main now has different EC-48 — renumbered to EC-56.)

- **Error pattern**: creating a small NPU tensor (e.g., 12-byte tiling struct via `torch.empty(12, dtype=uint8).to(device)`) triggers a torch_npu internal Slice kernel crash: `aclrtLaunchKernelWithHostArgs failed, return: 507035`. The crash is in torch_npu's H2D transfer path, not in the user kernel.
- **Root cause**: torch_npu's internal `.to(device)` transfer path invokes a Slice kernel for sub-32-byte tensors, and the Slice kernel on V220 does not handle very small payloads correctly.
- **Fix**: pad the host tiling buffer to ≥256 bytes before `.to(device)`:
  ```python
  PAD_BYTES = 256
  tiling_bytes = struct.pack(...)  # e.g. 12 bytes
  padded = tiling_bytes + b'\x00' * (PAD_BYTES - len(tiling_bytes))
  tiling_tensor = torch.frombuffer(bytearray(padded), dtype=torch.uint8).to(device)
  ```
- **Detection**: if `torch::empty({N}, ...).to(npu_device)` crashes with 507035 and N < 32, suspect this. Pad to 256 bytes and retry.
- **Evidence**: op#3 Add ds kw-1 (2026-05-11, Ascend910_9382 V220, CANN 9.0.0): 12-byte AddTiling struct → `.to(device)` crashed with Slice kernel 507035. Padded to 256 bytes → transfer succeeded.
- **Cross-ref**: OL-77 (GM tiling struct byte-by-byte read — tiling design pattern that generates the small host POD this crash affects).

<!-- 迁移自 porter kb/target/ascendc/（EC-56，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

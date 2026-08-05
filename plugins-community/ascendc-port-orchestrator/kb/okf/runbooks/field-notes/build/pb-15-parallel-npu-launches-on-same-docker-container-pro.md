---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Parallel NPU launches on same docker container produce cross-kernel state pollution"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Reverify outputs show spurious max_abs_diff ≈ 3e+38 (near fp32 inf) for kernels that return bf16 tensors in normal value ranges. Sequential re-runs of same (ker"
confidence: single_run
original_id: PB-15
timestamp_inferred: true
tags: [npu_dev3, ascendc, pb-15]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Status**: CONFIRMED (2026-04-21)
- **Affected**: A5 container `npu_dev3` with multiple `_<op>_ext.so` pybind modules launching on NPU 0, 1, 2, 3 concurrently via separate Python processes via `docker exec ... &` bash-backgrounded.
- **Symptom**: Reverify outputs show spurious `max_abs_diff ≈ 3e+38` (near fp32 inf) for kernels that return bf16 tensors in normal value ranges. Sequential re-runs of same (kernel, input-seed) pair show 0 drift. **False positive magnitude drift only under concurrent NPU launches.**
- **Root cause (hypothesis)**: aclrt/CANN stream state is container-global, not per-NPU. Concurrent launches across NPU 0..3 via independent Python processes of the same container share some runtime state — possibly allocator, HCCL, or operator proto registration — and one kernel's output buffer gets transiently corrupted while another kernel is mid-launch.
- **Workaround**: **Serialize all NPU reverify/test invocations within a single container.** If parallel test speedup is required, use 1 docker container per NPU (not just distinct `--device=npu:X` on same container). A second independent container per NPU isolates allocator state.
- **Detection**: Any time reverify scaffolding shows `ko_max > 1e10` for kernels producing bf16/fp16 outputs in normal ranges, suspect crosstalk. Re-run sequentially to verify. The 2026-04-21 drift-triage session was initially misdiagnosed as "real kernel bug" until sequential re-run showed the crosstalk signature.
- **Evidence**: 2026-04-21 batch reverify across 13 L2 PASS ops. Parallel-4-NPU re-run of op#9/19/21/26 showed `kernel_drift` for all 4 with `ko_max=3e+38` for op#19 case 5. Standalone re-run of each showed aggregate: op#9 spec_ambiguous, op#19 torch_npu_drift, op#21 **all_bit_exact 10/10**, op#26 spec_ambiguous. All "drift" signals from parallel run disappeared.
- **Related**: `src/scripts/batch_oracle_reverify.sh` header note (added 2026-04-21).

<!-- 迁移自 porter kb/target/ascendc/（PB-15，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

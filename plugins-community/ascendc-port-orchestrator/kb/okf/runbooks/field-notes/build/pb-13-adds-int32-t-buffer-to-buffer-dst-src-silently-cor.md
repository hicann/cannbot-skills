---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`Adds<int32_t>` buffer-to-buffer (dst ≠ src) silently corrupts output at N=1088"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Adds<int32_t>(dst, src, 0, N) where dst != src (buffer-to-buffer \"copy via zero-add\" pattern) silently produces wrong output data. Kernel compiles and runs with"
confidence: single_run
original_id: PB-13
timestamp_inferred: true
tags: [npu_dev3, ascendc, pb-13]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Status**: OPEN (observed 2026-04-19, needs next CANN version re-verify)
- **Affected**: CANN 9.0.0 (`/usr/local/Ascend/cann-9.0.0`), SOC Ascend950PR_9589, bisheng compiler as shipped with CANN 9.0.0 on A5 container `npu_dev3`. NOT yet retested on CANN 9.0.T501 or later.
- **Symptom**: `Adds<int32_t>(dst, src, 0, N)` where `dst != src` (buffer-to-buffer "copy via zero-add" pattern) silently produces wrong output data. Kernel compiles and runs without crash; downstream verification detects the corruption.
  - Specific observation: at count N=1088 (TOPK_CAP used by 9_TopKTopP V3.3 kind-2 rewrite), after `Adds<int32_t>` only the first entry appeared correct and the rest were zeroed/garbled (manifest: 49/50 cases fail precision with pattern "only gmax retained per row, rest become -inf").
  - **In-place** `Adds<int32_t>(buf, buf, 0, N)` works fine in the same build.
  - `Adds<float>` (same code shape, different dtype) works fine at same N — suggests bisheng codegen bug is `int32_t`-specific for this pattern.
- **Root cause**: Unconfirmed. Likely bisheng codegen quirk for the non-in-place pattern on int32. Minimal repro not yet created.
- **Workaround**: For copy-back of int32 buffers, use scalar loop (`for (int i=0; i<N; i++) dst.SetValue(i, src.GetValue(i));`) or `Cast int32→fp32, Adds<float>, Cast fp32→int32` roundtrip (only safe if all int32 values fit exactly in fp32 range — 24-bit signed range, int32 values > 2^24 would round).
- **Detection**: If you use `Adds<int32_t>` as a copy-back and see downstream results with all but the first element wrong, suspect this bug. Confirm by comparing to scalar loop baseline.
- **Perf impact**: On 9_TopKTopP V3.3 kind-2 rewrite, the scalar-loop workaround for int32 copy-back is a key reason V3.3 hit 0.191x sum-ratio (vs R3b 0.222x). R3b was measured on different NPU state (possibly different bisheng patch level) where this bug may not have manifested — hence R3b achieved full VEC copy-back. Re-verify on next CANN version to confirm whether R3b's VEC Adds<int32_t> approach now works.
- **Re-validation checklist when CANN updates** (per user directive 2026-04-19):
  1. Write minimal repro: `Adds<int32_t>(dst_buf, src_buf, 0, 1088)` with dst != src, check output bit-exact vs scalar loop
  2. Test at N ∈ {256, 512, 1024, 1088, 2048, 4096} to see if N-dependent
  3. If fixed: remove scalar-loop workaround from V3.3 kernel (or new kernels) and re-benchmark
- **Evidence**: 9_TopKTopP V3.3 kind-2 rewrite, Phase D iter 2 (2026-04-19). Kernel: `output/npukernelbench/src/kernels/9_TopKTopP/kernel/topktopp_kernel.h` (after V3.3 archival — TBD) workaround uses hybrid `Adds<float>` for val + scalar loop for idx. verification.json records the failed VEC Adds<int32_t> attempt. V3.2 t2 optimizer Opt4b earlier saw a related failure with same symptom — cross-confirmed on two independent attempts in same session.

<!-- 迁移自 porter kb/target/ascendc/（PB-13，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

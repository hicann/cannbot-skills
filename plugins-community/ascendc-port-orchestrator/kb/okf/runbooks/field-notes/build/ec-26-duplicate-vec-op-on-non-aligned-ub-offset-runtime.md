---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Duplicate / VEC op on non-aligned UB offset → runtime error 340"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel compiles successfully but at runtime reports \"UB address not aligned\" / error code 340 / aclrtSynchronizeStream failure."
confidence: single_run
original_id: EC-26
timestamp_inferred: true
tags: [aclrtsynchronizestream, ascendc, ec-26]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (22_HybridAttentionMaskPreparation). Do not downgrade.
- **Error pattern**: Kernel compiles successfully but at runtime reports "UB address not aligned" / error code 340 / `aclrtSynchronizeStream` failure.
- **Trigger conditions**: `Duplicate(ubBuf[offset], 0.0f, count)` or similar VEC op, where `offset` does not meet dtype alignment:
  - fp32: `offset` must be a multiple of 8 (32B alignment)
  - fp16/bf16: `offset` must be a multiple of 16 (32B alignment)
- **Root cause**: AscendC VEC instructions require the LocalTensor's start address to be 32B aligned. If the code does a partial tile fill (`Duplicate(buf[orig_n], 0, pad_count)` to zero the tail), `orig_n` may not be aligned and triggers a hardware exception.
- **Fix options**:
  1. **Preferred (used by 29_TanhGatedResidualAddBackward)**: in the pybind layer, use `torch::zeros + .copy_()` to pre-fill the input tensor to the aligned size — the kernel no longer needs to manually zero padding (padded elements naturally produce 0 products, which do not affect reduce).
  2. Use `Duplicate(buf, 0, total_aligned_count)` to zero the entire buffer instead of just the tail.
  3. Switch to DataCopyPad for non-aligned GM↔UB transfers — but note EC-23 warns that DataCopyPad UB→GM crashes on A5.
- **Detection**: if perf tests report error code 340 and the kernel uses partial-tile / tail handling, first grep `Duplicate.*\[.*\]` to locate non-aligned start addresses.
- **Evidence**: 29_TanhGatedResidualAddBackward V1 used `Duplicate(wA[orig_in_tile], 0.0f, pad)` → error code 340. After switching to pybind pre-padding, 50/50 PASS. Worker tool count saved ~5 iterations (avoided manual debug).
  - 7_MoeGatingTopKSoftmax (2026-04-17): `Duplicate<float>(expFp32Local[N_], -INF, padCount)` where N_=5120/7168 is non-aligned → err 340. Fix: delete the padding Duplicate — since `x = -inf` input naturally produces `exp = 0`, no padding is needed. Same EC-26 reset pattern.

<!-- 迁移自 porter kb/target/ascendc/（EC-26，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cap per_core_do_block_num by block_num when porting UB-budget tiling arch22->arch35"
description: "Porting UB-derived per_core_do_block_num from arch22 to arch35 unchanged can oversize singleSize on small shapes; the kernel reads past GM and silently returns wrong results. Cap it by block_num."
confidence: single_run
original_id: OL-174
classified_by: llm-assisted
timestamp_inferred: true
tags: [host-tiling, optimization, ol-174, port-a3-to-a5, ub-budget, silent-wrong-result]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=quant_optim, reduction_with_per_block_scale`.
Source: `workspace/apply_adam_w_quant` 13-spawn debug session 2026-05-14.

**Principle**: When porting an op's `op_host/<op>_tiling.cpp` from A2/A3 (V220, UB=192KB) to A5
(V351, UB=256KB) **without modifying the tiling math**, the UB-derived
`per_core_do_block_num = max_ub_size / one_block_size` is sized for the larger UB. On A5 this value
can be **significantly larger than the op's actual `block_num`** (total work). The kernel then uses
it as the per-row block count and computes `singleSize = blockSize * oneCoreDoBlockNumPerRow` —
sizing DataCopyIn/Out beyond the actual GM extents. The kernel **reads garbage (or zeros) past the
end of the input tensors, silently producing wrong results for small-shape cases**.

**Concrete failure** (apply_adam_w_quant on A5):
- `max_ub_size=256KB`, `one_block_size≈4608B` → `per_core_do_block_num≈57`.
- Case 4 (4 rows × 256 = 4 blocks total): `block_num=4` but `oneCoreDoBlockNumPerRow=57`.
- Kernel `singleSize = 256 × 57 = 14592` bytes, but GM has only 1024.
- `DataCopyIn(absMaxM, absMaxMGm, oneCoreDoBlockNumPerRow)` reads 53 garbage absmax values past the
  4 real ones. Garbage absmax (often 0 due to UB initial state) → dequantized `m_old=0` →
  `m_new = β1·0 + (1−β1)·grad·gn` → A5 `absmax_m ≈ (1−β1)·gn·max|grad| ≈ 0.0099` vs A3
  `absmax_m ≈ 1.82` (which correctly includes the `m_old` momentum term).
- 10 mechanically-distinct kernel-source codegens (kw-1..kw-5, iter-A, iter-B, H7, H11 + revert) all
  gave bit-identical residual — proving the bug is structurally **outside the kernel body, in host
  tiling**.

**Diagnostic signature** (when to suspect this OL):
1. Same kernel binary works for some shapes, fails for others.
2. Failing shapes are SMALL relative to `per_core_do_block_num`.
3. Output magnitude on failing cases matches "input scaled by (1 − coefficient)" — one term in a
   weighted sum is dropping.
4. Kernel-source codegen variants (V-pipe edits, buffer relocations, primitive swaps) ALL produce
   bit-identical residual on failing cases.
5. CPU truth confirms the A3 reference is algorithmically correct (rules out an upstream-CANN bug).

**Fix pattern** (host tiling — cap by `block_num`):
```cpp
uint64_t ub_budget_blocks =
    (max_ub_size - QMAP_SIZE * SIZE_OF_FLOAT * NUM_OF_QMAP) / one_block_size;
uint64_t per_core_do_block_num = std::min(ub_budget_blocks, block_num);
```
This ensures `singleSize = blockSize * per_core_do_block_num ≤ blockSize * block_num`, matching the
actual workload so the kernel never iterates / loads past the real GM extents.

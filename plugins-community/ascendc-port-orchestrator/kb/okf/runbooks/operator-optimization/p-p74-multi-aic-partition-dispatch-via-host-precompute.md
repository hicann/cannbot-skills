---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-AIC partition-dispatch via host-precomputed segment offsets"
description: "> ID coordination note: a3 PR #2 reserves P-P70-P-P73 for fused-quant patterns. P-P74 is the next free slot at the time of writing. If a3 PR #2 lands after this entry but with overlapping numbering, r"
confidence: single_run
original_id: P-P74
timestamp_inferred: true
tags: [platform_compat, optimization, grouped_mm, matmulapistatictiling, tcubetiling, setatomicadd, p-p74, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

> ID coordination note: a3 PR #2 reserves P-P70-P-P73 for fused-quant patterns. P-P74 is the next free slot at the time of writing. If a3 PR #2 lands after this entry but with overlapping numbering, rebase by shifting whichever side is later.

### When to use
Any kernel whose output is the row-concatenation of N independent sub-computations where the per-segment row counts are known on the host. Direct multi-AIC extension of P-P68 (which covers the single-AIC case). Concrete instances:
- Grouped / segmented matmul (`torch._grouped_mm`, `F.grouped_mm`, MoE expert dispatch)
- Variable-length / jagged batched GEMM where each "batch" has a distinct row count
- Segmented attention / segmented sparse-gather / GroupedConv where each group's compute is independent and outputs occupy a contiguous output row range

Trigger: source has a Python-level loop over groups (`for g: out_g = compute(A_g, ...)`) followed by a row-axis concat, OR a single fused op (`grouped_mm`) whose semantics are equivalent.

### Pattern
- **Dispatch**: `blockDim = G` (number of segments). Each AIC owns exactly one segment's compute. Layer P-P68's per-AIC machinery (constexpr `MatmulApiStaticTiling`, on-stack `TCubeTiling`) inside.
- **Host pre-compute** (pybind11.cpp): build `cum_out[G+1]` cumulative output row offsets in fp32 int32 vector, push as int32 NPU tensor. **Sentinel**: set `cum_out[G] = total_rows` so the kernel reads `end_row = cum_out[g+1]` uniformly with no special-case for the last segment.
- **Per-AIC kernel decode** — uniform-vs-variable input slicing is a host flag, not a per-op detail:
  ```cpp
  const int32_t bid = GetBlockIdx();
  // For segments with variable input rows (e.g. 2D-A grouped_mm):
  //   input_row_off = offsets[bid]; M_g = offsets[bid+1] - input_row_off;
  // For segments with uniform input rows (e.g. 3D-A stacked grouped_mm):
  //   input_row_off = bid * m_uniform; M_g = m_uniform;
  // Output side always uses cum_out[bid] / cum_out[bid+1] regardless.
  ```
- **Determinism by-construction**: each output row owned by exactly one AIC; no `SetAtomicAdd`; per-AIC mmad order fixed by constexpr tiling; `IterateAll<sync=true>` + `End()` per AIC. `DET_POLICY=required` satisfied without atomicCAS / lock-bit games.
- **Reference availability fallback** (when target torch+CANN lacks `grouped_mm`): use OL-89 prose-spec extension to write a workspace-local `model.py` that loops `torch.matmul(A_g, B[g])` per segment and concatenates. Decomposition is mechanical (no rounding ambiguity, no contraction-order ambiguity), satisfies `verification_ascendc.py` Pass A, matches per-AIC kernel output bit-exact in fp32.

### Limitation (when to reach for the next pattern)
`blockDim = G` caps parallelism at the number of segments. For G ≤ AIV_count with large per-segment GEMMs (M·N·K > ~1M elements), a single AIC cannot match the reference's 56-AIC `aclnnMatmul`. Worst-case ratio drops while median stays > 1× because small-G cases dominate. The next architectural move is a 2D dispatch (`blockDim = G × per_segment_tile_count`) with K-split or L0C-stage merge — aog-kernel-optimizer territory, not kw correctness.

### Evidence
- 2_GroupedMatmul (op#2, 2026-04-28): 50/50 + 16/16 PASS bit-exact, det 50/50 identical, median 1.05×, gmean 0.83× (PASS by V3.3.6 median methodology). 5th level-3 op closing the matmul family. **0 build iters + 0 precision iters first try** — cube playbook (P-P68 + P-P69 + P-P74) fully amortized.
- See OL-93 for the op#2-specific evidence record and 2D-A vs 3D-A flag handling.

### Combine with P-P68 / P-P69
- P-P74 specifies the multi-AIC dispatch architecture (blockDim=G, host-side row partitioning, sentinel-extended offsets, bid-as-segment decode).
- P-P68 supplies each AIC's GEMM machinery (constexpr static tiling, on-stack TCubeTiling, non-`__gm__` `Init`).
- P-P69 supplies any per-segment transpose (runtime `SetTensor*` bool, never template ISTRANS).
- Use all three together for grouped/segmented cube ops with logical transpose.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P74，convert_patterns_to_okf.py）。confidence 未升格。 -->

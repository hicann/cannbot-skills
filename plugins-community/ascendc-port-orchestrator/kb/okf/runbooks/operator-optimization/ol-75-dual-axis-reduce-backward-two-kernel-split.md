---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Dual-axis reduce backward: two-kernel per-row + cross-row separation"
description: "A backward op needing both a per-row reduction and a cross-row reduction is more reliable/efficient split into two kernels — but only when per-core vector work is large enough to amortize the extra launch; tiny tensors lose to a single fused kernel."
confidence: single_run
original_id: OL-75
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-75, backward, reduce, multi-kernel]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

When a backward op needs **both** a per-row reduction (e.g. mean/sum over the H dim) and a cross-row reduction (e.g. sum over the B*S dims), splitting the two reduction directions into two kernels is more reliable and efficient than a single kernel that tries to do both at once.

**Kernel 1** — each core owns its row partition:
- computes the per-row outputs (e.g. `grad_x`);
- accumulates a partial cross-row sum `[H]` in UB;
- writes the partial `[H]` to `workspace[core * H_padded]`.

**Kernel 2** — partitions the H dim across cores:
- each core reads the corresponding column range across all partial arrays;
- sums them;
- writes final `grad_weight[H]`.

Because Kernel 2 partitions along H (not rows), each core only reduces a small number of partial values (e.g. 56, one per Kernel-1 core), and outputs are naturally column-partitioned — no need for the P-P49 8-aligned slot.

### Scope condition (size-sensitivity) — applies to ANY partial+reduce multi-launch backward template

The "more efficient" claim holds ONLY when per-core vector work is large enough to amortize the extra launch + cross-core sync overhead of the multi-kernel split. The template issues ≥2 launches (1 main + 1 reduce per cross-row output). On tiny tensors the fixed launch overhead dominates and the split LOSES to a single fused kernel.

**Decision rule:**
- Large per-row width (H ≳ 1024, vector work ≫ launch overhead) → multi-kernel split wins.
- Tiny tensors (total ≤ a few KB elements, launch-overhead-bound) → prefer a single fused kernel.
- Re-measure rather than assume the template transfers across tensor regimes.

See candidate CAND-PP104 for the small-tensor fused-rewrite mitigation (pending optimizer confirmation).

### Evidence

- **Large-tensor regime — split WINS:** 19_FusedResidualRmsNormBackward, where `grad_weight = sum_{B,S}(go*norm)` and `mean_grad_norm = mean_H(grad_norm*norm)`. Kernel 1 handles rows + partial `[H]`; Kernel 2 sums partials. 50/50 PASS, 1.31x mean. workspace = `56 * H_padded * 4B ≈ 56*8192*4 = 1.75MB` for the largest case. E3 level.
- **Small-tensor regime — split LOSES (counter-instance):** group_norm_grad (2026-06-03, port_a3_to_a5 V220, authored from scratch). Same partial+reduce template (main per-(n,g)-group kernel + reduce kernel per output dweight/dbias → 3 launches). On GroupNorm's tiny tensors (≤1024 elems): ours ~150-200µs flat (launch-overhead-bound) vs vendor single fused CANN kernel ~28-49µs → **ratio 0.25×, stable across 3 runs / 2 devices**. Precision + determinism 4/4 PASS first try; the regression is purely launch-overhead, not correctness — confirms the size-sensitivity scope condition.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`ArithProgression` for vectorized arange / index-sequence generation"
description: "Trigger: kernel needs a vector of [start, start+step, start+2step, ...] (row offsets for gather, position indices for RoPE, identity permutation, stride vectors). Pattern: cpp // Generates count eleme"
confidence: single_run
original_id: P-P63
timestamp_inferred: true
tags: [reduction_quant, optimization, arithprogression, add, adds, p-p63, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: kernel needs a vector of `[start, start+step, start+2*step, ...]` (row offsets for gather, position indices for RoPE, identity permutation, stride vectors).

**Pattern**:
```cpp
// Generates count elements: dst[i] = start + i * step
ArithProgression<int32_t>(dst, /*start=*/0, /*step=*/blockTablesStride, /*count=*/seqsNum);
```

This is one VEC instruction. Replaces:
```cpp
for (int i = 0; i < count; i++) dst.SetValue(i, start + i * step);   // O(N) scalar ops at ~50ns each
```

**Use cases observed**: row offsets for indirect gather, identity permutation for sort initial state, stride vectors for non-contiguous access, position indices for embeddings.

**Type variants**: `ArithProgression<int32_t>`, `<float>`, `<int64_t>` all available.

**Combine with `Add`/`Adds`**: ArithProgression for the base sequence + vector `Add(dst, base_seq, offset_vector, count)` composes complex index patterns in O(1) instructions.

### Performance-critical cross-references (2026-06-24, add_rms_norm_quant V1->V2)

When implementing a fused reduction+normalize+quantize kernel on A5/arch35,
the following OL entries MUST be consulted BEFORE writing Phase B code:

- **OL-256** (Divs->Muls for fp32 normalize): Use `Muls(x, 1.0f/rms)`, not `Divs(x, rms)`.
  This is precision-safe in fp32 per ALWAYS_LOADED_RULES §5 fp32 carve-out. Contributed ~45%
  of the 1.94x speedup in add_rms_norm_quant V1->V2.
- **OL-257** (VEC cost model): Divs is Tier 3 (1-2 elem/cyc), Muls is Tier 0 (8-16 elem/cyc).
  Count Tier 3+4 ops and PipeBarriers per row before committing the design.
- **OL-258** (TQue double-buffering): Use TQue QBUF_DEPTH=2 for >=2 GM reads/row.
  Contributed ~15% of the 1.94x speedup.
- **OL-245** (regbase default): A5 SIMD compute chains default to regbase, not Membase.
  Eliminates per-op PipeBarriers in multi-op chains.

The add_rms_norm_quant V1->V2 A/B comparison (2026-06-24, NPU 0, 196 cases) showed that
applying ALL FOUR optimizations together improved geo_mean from 0.47x to 0.91x vs PyTorch NPU,
while improving precision (196/196 vs 194/196 PASS).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/reduction_quant.md（P-P63，convert_patterns_to_okf.py）。confidence 未升格。 -->

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Two-pass tile loop for D > UB-budget per-row reductions"
description: "When a single row D exceeds the per-AIV UB budget, split ProcessRow into Pass-1 streaming reduction to a row scalar, then Pass-2 tile-reload emit — doubles input MTE2 traffic but bounds UB."
confidence: single_run
original_id: OL-114
classified_by: llm-assisted
timestamp_inferred: true
tags: [tiling, optimization, ol-114, ub-management, two-pass]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** a per-row reduction op (max-abs, sum, var, softmax, quant) where the largest expected `D × per-row-buffers` exceeds the per-AIV UB budget (256 KB on a5). Loaded by aog-kernel-worker at Phase A architecture.

**Pattern.** When a single row's D can't keep all per-row intermediate buffers resident in UB simultaneously, decompose `ProcessRow(int64_t r)` into a two-pass tile loop:
- **Pass 1 — streaming reduction across tiles:** per tile, copy GM→UB, compute a partial reduction, accumulate into a row-wide scalar (e.g. `row_max`). Tile-level intermediates are recycled across iterations.
- **Pass 2 — emit using the row scalar:** per tile, **reload the same tile from GM**, apply the per-element transform using the row scalar (e.g. quantize with `inv_scale = 1/(row_max/127)`), copy UB→GM.

**Tradeoff.** Doubles MTE2 traffic for the input (each tile is read twice). Fast path for `D ≤ TILE`: skip the Pass-2 reload and keep the tile resident.

### Concrete anchor (TILE=4096, fp16/bf16 input → fp32 work → int8 output)
UB budget at TILE elements:
```
xQue (depth=2 × TILE × 2B)            =  4 × TILE bytes
workBuf + tmpBuf + i32Buf (3 × fp32)  = 12 × TILE bytes
fp16Buf                               =  2 × TILE bytes
outQue (depth=2 × TILE × 1B)          =  2 × TILE bytes
──────────────────────────────────────────────────────
Total                                 ≈ 20 × TILE bytes
```
TILE=4096 → 80 KB per AIV. The 256 KB UB limit supports TILE up to ~13K. Choose TILE from the smallest D in the case mix — too small wastes MTE2, too large blows UB on small rows.

### Evidence
- 29_DynamicQuant kw-2 (2026-05-01): kw-1 attempted single-pass-fits-in-UB and crashed at D ≥ 11008 with err 263/81 (UB overflow). kw-2 implemented the two-pass tile loop with TILE=4096 → 42/42 benchmark + 11/11 edge_dataset PASS. Perf 0.50x (Pass-2 reload cost) — handed to the optimizer for a Pass-2-cache-when-D-fits fast path.

### Other instances (predicted)
- Per-row softmax / log-softmax with large vocab (any LM logits)
- Per-row layer norm / RMSNorm with H > UB budget (common in weight-grad reductions)
- Per-row L2-norm + scale (clip-by-norm)
- Any `(reduce → broadcast-back → elementwise-emit)` where the emit step needs both the reduce result and the original input

### Related
- P-P45 (single-pass UB-resident — prefer it when D fits; OL-114 is the fallback)
- P-P46 (Cast chain fp32→int32→fp16→int8 — the per-tile emit step in DynamicQuant uses this)
- EC-23 (DataCopyPad UB→GM crash — sidestepped by pre-padding input row stride to TILE multiples in pybind)
- OL-63 (TQue depth — tile loop benefits from depth=2 in_que for MTE2/VEC overlap)

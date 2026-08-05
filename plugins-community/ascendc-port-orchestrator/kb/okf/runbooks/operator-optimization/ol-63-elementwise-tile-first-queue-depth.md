---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Elementwise ops — tile-first UB allocation, queue-depth-second"
description: "For elementwise / per-tile ops decide tile size first (max safe tile at depth=2), then upgrade to queue depth=4 only if UB allows WITHOUT shrinking the tile — larger tile beats deeper queues."
confidence: single_run
original_id: OL-63
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-63, elementwise, tile-size, queue-depth]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: `soc=Ascend950PR; cann=9.0.0; op_class=elementwise-vec`. Verified on Ascend950PR.
**Unverified on** Ascend910_V220 (A3 chip family — pipe-stage scheduling differs; A3 should
re-validate the depth=4 vs depth=2 trade-off independently).

**Trigger**: generating elementwise ops (GELU, SiLU, Abs, Add, Muls, …) OR per-tile reduction
kernels (DynamicQuant, RmsNorm, etc.). Loaded by Generator.

**UPDATED 2026-06-24**: the decision order was reversed. It used to be "default depth=4"; it is
now "determine the max safe tile with depth=2 first, only upgrade to depth=4 if UB allows
WITHOUT shrinking the tile." Larger tile is the dominant lever.

### Decision order (tile-first)

**Step 1 — Determine max safe tile with depth=2:**
- fp32 path (no TBuf): `MAX_TILE = UB_SAFE / (3 queues × 2 slots × sizeof(T))`.
  Example A5 (248KB UB): `248×1024 / (3×2×4) ≈ 10,582` → cap at 8192 (HW practical limit).
- fp16/bf16 path (2 TBuf for fp32 promotion):
  `MAX_TILE = (UB_SAFE − 2×TBuf_bytes) / (3×2×sizeof(T) + 2×sizeof(float))`,
  with `TBuf_bytes = MAX_TILE × sizeof(float)` — solve for MAX_TILE.
- Cap MAX_TILE at `block_size` (no point tiling larger than one outer block).
- Prefer powers of 2 (8192, 4096, 2048) for alignment.

**Step 2 — Estimate VEC/MTE2 ratio at this tile:**
- `VEC_cycles ≈ per_element_ops × tile × cycles_per_vec_op`
- `MTE2_cycles ≈ tile × sizeof(T) / mte2_bw_per_cycle`
- `VEC ≥ 3× MTE2` → compute-heavy (SwiGlu 5-op chain, GELU, Erf, Tanh).
- `VEC < 3× MTE2` → thin compute (DynamicQuant, RmsNorm pass-1, single Cast+op).

**Step 3 — Try depth=4 ONLY if UB allows WITHOUT shrinking the tile:**
- Extra UB for depth=4: `3 queues × 2 extra slots × tile_bytes`.
- If it fits in remaining UB AND VEC ≥ 3× MTE2 → use depth=4.
- If it does NOT fit → keep depth=2 + max tile (the tile benefit dominates).
- **NEVER shrink the tile to make room for deeper queues** — larger tile wins.

### Litmus (swi_glu)

swi_glu is a 5-VEC-op chain (Muls + Exp + Adds + Div + Mul), so VEC ≈ 5× MTE2 → compute-heavy.
Yet depth=4 with tile=2048 (V1) wastes 32KB UB on extra queue slots. depth=2 with tile=8192
(V5) processes 4× more elements per iteration → 2× fewer iterations, and that iteration-count
reduction dominates the modest pipeline-overlap gain from depth=4. **Tile size first, queue
depth second.** (V5 tile=8192+depth=2 reached 2.11× geo_mean vs V1 tile=2048+depth=4; multi-core
is the larger factor but tile alone accounts for ~2× of the gap.)

### Evidence

- GELU regression 2026-04-14: old kernel `TQue<VECIN,4>` = 1.07×, new kernel depth=2 = 0.65×.
  The full gap was insufficient pipeline overlap (a compute-heavy kernel benefits from depth=4).
- DynamicQuant ko-1 iter3 2026-05-02: per-tile compute is Cast+Abs+ReduceMax over ~4K cycles
  (thin). Bumping `IN_QUE_DEPTH` 2 → 4 **regressed honest mean perf by 7%**.

Note: the source OL text was truncated mid-sentence in the DynamicQuant evidence line
("Reas…"); reproduced only as far as the source provided.

---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Boundary-clamped interpolation — a single OOB→0 gather reproduces zeros/border/reflection padding because the clamped coordinate's overflow neighbor always carries zero interpolation weight"
description: "applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=sampling-interpolation (grid_sample, interpolate, affine-warp) verified_on: soc=Ascend950PR_957b; cann=9.1.T500 unverified_on: so"
severity: medium
confidence: single_run
original_id: P-P105
timestamp_inferred: true
tags: [memory_access, optimization, clamp, reflect, size, coord, fetch, p-p105, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=sampling-interpolation (grid_sample, interpolate, affine-warp)`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 family — this is dtype-agnostic coordinate arithmetic and should transfer, but no cross-arch witness yet)`

**Principle**: A linear-interpolation sampler (bilinear / trilinear) supporting multiple padding modes does NOT need per-mode neighbor-index clamping if (1) the coordinate is mapped into the valid range `[0, size-1]` by a per-mode clamp BEFORE neighbor selection (border → `clamp`, reflection → `reflect`, zeros → identity), and (2) the per-pixel fetch helper returns `0` for any out-of-bounds index. After the clamp, the floor neighbor `x0 = floor(coord)` is always in-bounds; the overflow neighbor `x0 + 1` only lands OOB (at exactly `size`) when `coord` sits on the integer upper boundary `size - 1`, where the fractional weight `dx = coord - x0 == 0`. Since that neighbor's contribution is `dx * fetch(x0+1) = 0` regardless of what `fetch` returns, a unified `fetch` that yields `0` for OOB is correct for all three modes. The zeros mode then "falls out" for free: for zeros, the coord is NOT clamped, so any OOB neighbor genuinely contributes `0` — exactly what the unified fetch produces.

**Concrete anchor (grid_sample, kw-1 2026-06-20)**:

```cpp
// per-mode coordinate clamp into [0, size-1] (zeros = identity, border = clamp, reflection = reflect)
float cx = gs_clip(coord_x, W, paddingMode, alignCorners);
int64_t x0 = gs_floor_i(cx);
float dx = cx - (float)x0;                 // dx == 0 exactly when cx is on the integer boundary
// unified fetch returns 0 for OOB (mirrors upstream arch35 GetInputPointValue):
float v00 = gs_fetch(img, base_nc, y0, x0,     H, W);   // x0 always in-bounds after clip
float v01 = gs_fetch(img, base_nc, y0, x0 + 1, H, W);   // OOB only when dx==0 → weight kills it
float top = v00 * (1.0f - dx) + v01 * dx;               // border/reflection/zeros all correct
```

**Why it's safe (one fetch, no per-mode branch)**: the only way `x0 + 1 == size` (OOB) after `gs_clip` is `cx == size - 1` (integer boundary), which forces `dx == 0`; `weight * fetch == 0` whatever `fetch` returns. This matches upstream arch35 `GetInputPointValue` semantics line-for-line — no per-mode neighbor clamp is authored.

**When this applies**:
- Linear/bilinear/trilinear sampling where the coordinate is clamped/reflected into range before neighbor pick
- Padding modes are {zeros, border, reflection} (the standard `F.grid_sample` / `align_corners` family)
- The interpolation weight on a neighbor goes to 0 exactly at the boundary where that neighbor would overflow

**Anti-pattern (don't apply when)**:
- Nearest-neighbor sampling (no fractional weight to zero out the OOB neighbor) — needs explicit clamp
- A padding mode whose OOB contribution is nonzero (e.g. a constant-fill ≠ 0, or wrap/circular padding) — the "weight kills it" argument fails
- Cubic interpolation (4-tap): the far taps can be OOB with nonzero weight — this 2-tap argument does not extend without per-tap analysis

**Evidence**: grid_sample port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500): one `gs_fetch` (OOB→0) + one `gs_clip` covers zeros/border/reflection × align_corners {false,true}; precision 29/29 T1 PASS (fp16 23/23, fp32 3/3, bf16 3/3), 29/29 deterministic. Verified against `F.grid_sample` fp32 CPU truth cross-checked line-by-line vs arch35 `GetInputPointValue`.

**Other instances (predicted)**: `interpolate`/`upsample` bilinear with boundary handling, affine-grid warp samplers, any 2-tap-per-axis sampler with a pre-clamp + zero-fill OOB fetch.

**Cross-reference**: OL-150 (SIMT programming model — the per-thread gather this sampler runs as), EC-74 (`__simt_callee__` for the VF-called fetch/clip helpers), A.2.6 dual-input faithful-reference rule (model.py reproduces the arch35 unnormalize/clip sequence).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P105，convert_patterns_to_okf.py）。confidence 未升格。 -->

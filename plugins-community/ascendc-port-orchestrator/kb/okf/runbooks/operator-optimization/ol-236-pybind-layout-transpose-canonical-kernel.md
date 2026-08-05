---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "When the reference host wrapper transposes layout around the kernel call, reproduce those transposes in the pybind and author a canonical-layout kernel"
description: "Pybind layout marshaling (.t().contiguous()) that mirrors an l0op::Transpose the reference host wrapper itself performs is data movement, not compute delegation — it lets you drop in-kernel de-interleave machinery and author a clean canonical-layout kernel."
original_id: OL-236
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [port-a3-to-a5, layout, transpose, ol-236, pybind-purity, optimization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** a `port_a3_to_a5` op whose vendor host wrapper (`op_api/aclnn_*.cpp`) performs layout transposes around its kernel call, and whose V220 kernel carries in-kernel de-interleave machinery to consume the caller's native layout. `applies_to: soc=Ascend950PR; cann=9.0.0; mode: port_a3_to_a5; op_class=all`.

### Principle

When a vendor host wrapper does **layout transposes** (`l0op::Transpose` / `l0op::Contiguous` with an explicit `perm`) immediately around its kernel call, the A5 harness pybind may reproduce those *exact same* transposes with `.t().contiguous()` and author a clean **canonical-layout** kernel — instead of porting the V220 kernel's in-kernel de-interleave machinery (`GatherMask` predicates + `Brcb` + boundary copy-out) that exists only to consume the caller's native layout. This is "understand -> generate": the layout shuffle was always host-side data movement in the reference; relocating it from `l0op::Transpose` to a pybind `.t().contiguous()` mirrors the reference, it does not delegate the op's compute. The math stays 100% AscendC primitives.

### Why this is NOT CANN/torch delegation (the line that matters)

The transpose only re-strides bytes; it computes nothing op-specific. It is acceptable BECAUSE it provably mirrors a transpose the reference host wrapper itself performs (cite the `l0op::Transpose perm{...}` site as provenance). The forbidden pattern is calling torch/aclnn to do the *op's arithmetic* — that is unaffected by this rule. Stay within the pybind-purity envelope: layout / `contiguous` / metadata only.

### Concrete anchor (iou_v2)

`op_api/aclnn_iou.cpp` transposes aligned inputs `(m,4)->(4,m)` and transposes the non-aligned output `(n,m)->(m,n)` via `l0op::Transpose` with `perm={1,0}`. The V220 kernel instead de-interleaved box-major `(m,4)` inputs in-kernel via `GatherMask` (src1Pattern 3/4/5/6) + `Brcb`. The A5 port dropped that, authored a coord-major kernel, and moved the `(m,4)<->(4,m)` marshaling into the pybind as `.t().contiguous()` (2 inputs + 1 output) — kernel shrank to ~330 lines vs the V220 ~700-line GatherMask/Brcb/boundary path.

### Perf caveat (optimizer lever)

Each pybind transpose is a fixed-cost GM->GM strided copy (~0.1 ms for the 3 transposes here). On small shapes this fixed cost can sink an individual case below the perf floor (iou_v2 min 0.38x at m<256) even when the aggregate clears it (geomean 1.21x). When small-shape parity matters, the lever is to read the caller's native layout directly in-kernel (eliminate the input transposes) — i.e. re-introduce a *minimal* in-kernel de-interleave only where the transpose overhead dominates. Trade kernel simplicity for transpose elimination, not the other way around.

## 证据
- iou_v2 kw-1 (2026-06-21, port_a3_to_a5 V220->arch35, A5 Ascend950PR): first instance — bounding-box IoU/IoF (ops-cv/objdetect), vec-only. 33/33 PASS_T1 (max ours_MARE 6.85e-7), geomean perf 1.21x.

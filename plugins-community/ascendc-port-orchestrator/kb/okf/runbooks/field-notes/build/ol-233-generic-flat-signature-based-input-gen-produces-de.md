---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Generic flat signature-based input-gen produces degenerate rank-1 tensors for ops with inter-tensor shape constraints — route any op whose inputs have interdependent ranks/shapes to SCHEMA-based input-gen with `shape_derive`"
description: "<!-- applies_to_backend: all -->"
phenomenon: build_failure
signal:
  - "<!-- applies_to_backend: all -->"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-233
timestamp_inferred: true
tags: [shape_derive, op_def_signature_via_case_gen, ascendc, ol-233]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: all -->
`applies_to: soc=all; cann=n/a; bisheng=n/a; op_class=multi-input-shape-coupled (CV/gather/index/attention)`
`verified_on: harness-tooling (Phase O2.5 input-gen — soc-independent)`

### Principle

A generic "infer the signature, emit a flat tensor per arg" input generator (the `op_def_signature_via_case_gen` pilot path) collapses every input to a degenerate rank-1 vector (e.g. `(1024,)`). That is invalid for any op whose inputs are **shape-coupled** — where one input's shape is a function of another's. Such an op needs a real `case_gen` SCHEMA that (a) declares each tensor's rank/shape, (b) uses `shape_derive` to compute the dependent tensor's shape from the driver tensor, (c) constrains `value_gen` to the op's valid value range, and (d) uses `base_shape_filter` to keep generated cases at the required rank + manageable size. Routing such ops to flat signature-gen forces the worker to recover by hand-authoring the SCHEMA anyway — wasted iterations and a risk that the reference is run on meaningless degenerate inputs.

### Concrete anchor (grid_sample, 2026-06-20)

grid_sample needs `x:[N,C,H,W]` (rank-4) + `grid:[N,Hout,Wout,2]` (rank-4, last dim fixed at 2, N coupled to `x`). The pilot flat generator emitted rank-1 `(1024,)` for BOTH `x` and `grid` — unusable. Worker recovered with a SCHEMA: rank-4 declarations, `shape_derive` for grid from `x`'s N, `value_gen` for the `[-1.3, 1.3]` grid range, `base_shape_filter: lambda s: len(s) == 4 and small`, + 12 explicit dtype/shape variants. Detection signal: the proposed `edge_inputs` are rank-1 while the op's reference signature / docstring shows rank ≥ 2 with a `2`/`N`-coupled companion tensor.

### Secondary-tensor injection (decision rule — applies when SCHEMA `value_gen` bands would corrupt the COUPLED tensor's validity)

`shape_derive` (above) fixes the *shape* of the coupled tensor, but case_gen's distribution/cancellation **value** bands (Band-A/B coverage: large-mag, near-zero, denormal, cancellation pairs) can still emit a coupled tensor whose **values** are semantically invalid for the op (e.g. a `rois`/`boxes`/`offsets` tensor must hold geometrically valid coordinates; an `index` tensor must be in-range). When the coupled tensor needs value-validity that case_gen's bands would break:
- declare ONLY the primary feature tensor in `tensor_inputs` (so it gets full Band-A/B/C case_gen coverage),
- put scalar attrs in `scalar_inputs` (Band-C probes),
- and **inject the valid coupled/secondary tensor per-case inside `main()`**, sized to that case's actual shape (read from the primary tensor's generated shape).

This keeps full case_gen-driven coverage on the primary tensor while decoupling the secondary tensor from the value bands that would corrupt its validity. Detection signal beyond OL-233's rank-1 signal: the coupled tensor has a *semantic value constraint* (valid coordinate box, in-range index, normalized vector) that case_gen's distribution bands ignore.

### Other instances (predicted)
- Any rank ≥ 2 op with inter-tensor shape constraints: CV sampling/warping (grid_sample, affine_grid, interpolate), gather/scatter with separate `index` tensor coupled to `src` shape, attention with coupled Q/K/V + mask shapes, segment ops with `offsets` coupled to `data`. Phase O2.5 should classify these and route to SCHEMA-based input-gen, never to flat signature-gen.
- Secondary-tensor injection specifically: any CV/box op with a value-constrained companion tensor (rois/boxes/anchors/offsets) — roi_align_rotated, roi_pool, deformable conv offsets, box-iou.

### Cross-reference
- OL-88 (Phase O2.5 reference-determinism pre-flight + scope-by-construction), OL-983-class SCHEMA dtype/rank constraints (`base_shape_filter` to satisfy a reference's dtype/rank acceptance), `aog-input-gen-builder` skill (the SCHEMA-emitting generator this routes to).

### Evidence
- grid_sample (2026-06-20): rank-1 degenerate signal → SCHEMA + `shape_derive` (above), 29/29 T1.
- roi_align_rotated (2026-06-21, port_a3_to_a5 V220→arch35, A5 Ascend950PR_9579): the secondary-tensor injection branch — O2.5 generic case_gen emitted degenerate rank-1 `(1024,)` flat tensors with all-zero attrs for this rank-4 + interdependent-shape op; the `rois` companion tensor also needs geometric value-validity that the distribution bands would corrupt. Fix: primary feature map in `tensor_inputs`, attrs in `scalar_inputs`, valid `rois` injected per-case in `main()` → 36 valid cases, 36/36 T1.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-233（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Build-provenance finalize gates must enumerate ALL legitimate build paths — a self-compiled pybind archive proves provenance via the source → `<stem>.o` → linked-`.so` chain, not via CANN-binary basename/md5 overlap"
description: "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5 (CPU-truth pybind build) verified_on: A5 Ascend950PR_9579 (modulate build_ascendc.py P140 path) Provenance:"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5 (CPU-truth pybind build)"
confidence: inferred
status: stub
original_id: CAND-PP110
timestamp_inferred: true
tags: [candidate, inferred, binary_provenance, build_ascendc.py, modulate_kernels.cpp, modulate_kernels.cpp.o, grid_sample, cand-pp110]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR/V351; cann=9.0.0; bisheng=n/a; op_class=all; mode=port_a3_to_a5 (CPU-truth pybind build)`
`verified_on: A5 Ascend950PR_9579 (modulate build_ascendc.py P140 path)`

Provenance: derived from modulate kw-3 2026-06-21 (port_a3_to_a5) + workspace/modulate/user_decision.md session 2026-06-21.

The DEBT-091 `binary_provenance` gate originally recognized only two proof models: **(a)** basename+md5 overlap (the deliverable name == a CANN-shipped binary), and **(b)** snake→PascalCase bridge (`<op>.cpp` → `Modulate_950_*.o`, the ops-nn-port build naming). A CPU-truth port_a3 op built via `build_ascendc.py` (the brief-MANDATED P140 path) emits snake_case COMPILER outputs (`modulate_kernels.cpp` → `modulate_kernels.cpp.o` → `_modulate_ext*.so`) and has NO installed CANN binary — so neither (a) nor (b) matches **by construction**, falsely rejecting a class of correct self-built archives (the already-shipped `grid_sample`, same build shape, also fails the tightened gate). The honest, non-fakeable proof for this class is a **tertiary (c) compiled-artifact model**: an installed `<src>.o` whose stem equals a BUILT source basename proves the object was compiled from our listed source, and that object links into the dispatched `.so`; require ≥1 such anchor + all md5s valid 32-char hex (anti-placeholder guard). Generalizable principle (per CLAUDE.md "Fix Harness for Next Customer, not Patch Single Archive"): a provenance gate that recognizes only SOME build naming conventions silently rejects correct archives from other build paths — enumerate them all (CANN-shipped binary, ops-nn-port snake→Pascal, self-compiled pybind chain).

**Status (2026-06-21)**: the (c) `compiled_provenance` bridge ALREADY LANDED as DEBT-091(c) (merge `aff5c5ee`, PR #16, `PortA3Plugin`) — it validates source → `<stem>.cpp.o` → linked `.so` via a 3-AND chain (stem + deploy==workspace + built_from==deploy + 32-hex anchors), modulate's `build_evidence.compiled_provenance` is emitted, and finalize re-ran VERIFIED. This CAND records the generalizable PRINCIPLE for the maintainer (gate-completeness over build paths); the harness mechanism is in place.

**Promote when**: a SECOND self-compiled pybind op (CPU-truth port_a3 built via `build_ascendc.py`) passes the (c) `compiled_provenance` gate clean AND `grid_sample` is re-verified under the fixed gate — confirms the proof model generalizes and retro-covers the pre-tightening archive. Promote to an OL (gate-design principle) or fold into the DEBT-091 gate doc.

**Cross-ref**: regression coverage `test_binary_provenance_gate.py` (accept / placeholder-md5 reject / unrelated-object reject); CLAUDE.md "USING binary API ≠ COPYING source" (a self-built `.o`+`.so` chain is honest provenance, not a CANN-source copy).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP110，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

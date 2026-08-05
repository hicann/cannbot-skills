---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A `__VEC_SCOPE__` regbase VF that hardcodes its element/tile count to a fixed inner-dim constant produces garbage above that constant — derive the count from the RUNTIME inner dim"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all"
confidence: single_run
original_id: EC-76
timestamp_inferred: true
tags: [__vec_scope__, ascendc, ec-76]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all`

A regbase MicroAPI VF that processes a `[outer, inner]` chunk must compute its loop/tile count from the **runtime** inner dim, not a value that happened to equal the inner dim on the shape it was first written for. A VF written for a fixed `inner=16` and coded as `ntFull = ceil(outer*16 / VL)` only covers the first 16 inner-lanes; for `inner>16` the upper `outer*(inner-16)` elements are NEVER written → the consumer reads uninitialized UB → garbage output. The bug is **masked whenever runtime inner == the hardcoded constant** (e.g. the original `dstate=16` customer), so it ships silently and only surfaces when a later caller uses a larger inner dim.

**Concrete anchor** (selective_scan fwd regbase build/prodC VFs): `uint16_t ntFull = (uint16_t)((cl * N + VLf - 1) / VLf);` — `N` is the runtime `dstate`, NOT a literal `16`. Tail over-process (when `cl*N` is not a multiple of VL=64) only touches the +64 buffer padding → identical semantics.

**Evidence**: selective_scan fwd-SIMD (2026-06-24, PR #52). `cl*16` → N=32 upper half garbage; `cl*N` → N∈{8,16,24,32,48,64} all 0-wrong at dtype floor. N=16 (customer) was correct either way (16==N).

**Other instances (predicted)**: any regbase/MicroAPI VF parameterized over a runtime inner/state/head dim — attention head-dim VFs, normalization feature-dim VFs, any `[L, D]` chunk VF — where an early single-shape author bakes the first D as a literal.

<!-- 迁移自 porter kb/target/ascendc/（EC-76，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

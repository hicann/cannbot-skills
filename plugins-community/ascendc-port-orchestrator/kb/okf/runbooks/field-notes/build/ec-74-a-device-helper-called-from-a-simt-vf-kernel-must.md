---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A device helper called from a `__simt_vf__` kernel must be `__simt_callee__` on CANN ≥9.1.T500 — plain `__aicore__ inline` (which EC-1 prescribes) no longer suffices"
description: "<!-- applies_to_backend: ascendc -->"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-74
timestamp_inferred: true
tags: [__simt_vf__, __simt_callee__, ascendc, ec-74]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR_957b; cann=9.1.T500; bisheng=n/a; op_class=simt-l3`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500`
`unverified_on: cann<9.1.T500 (older CANN compiled the same code with plain `__aicore__ inline` helpers — see 28_Interpolate)`

- **Error pattern**:
  ```
  note: candidate function not viable: simt_vf function can only call simt_callee function
  ```
  (the call site inside the `__simt_vf__` kernel fails to resolve the helper; the helper itself is already `__aicore__ inline`, so EC-1's fix is already in place yet the build still fails)
- **Root cause**: CANN ≥9.1.T500 tightened the SIMT calling convention — a function invoked from a `__simt_vf__` kernel must carry the `__simt_callee__` marker, not merely `__aicore__`. EC-1 (`__aicore__` on helpers) is necessary but no longer sufficient on this CANN. The upstream arch35 SIMT source already marks every helper `__simt_callee__ __aicore__ __attribute__((always_inline))`; older CANN versions (e.g. the one that compiled the finalized 28_Interpolate) accepted plain `__aicore__ inline` helpers, masking the requirement.
- **Fix**: add `__simt_callee__` (keep `__aicore__`; `__attribute__((always_inline))` recommended to match upstream) to EVERY device helper reachable from a `__simt_vf__` kernel:
  ```cpp
  // BEFORE (fails on CANN 9.1.T500: "simt_vf function can only call simt_callee function"):
  template <typename T>
  __aicore__ inline float gs_to_float(T v) { return static_cast<float>(v); }

  // AFTER (compiles):
  template <typename T>
  __simt_callee__ __aicore__ __attribute__((always_inline)) inline float gs_to_float(T v) { return static_cast<float>(v); }
  ```
- **Evidence**: grid_sample port_a3_to_a5 kw-1 (2026-06-20, A5/Ascend950PR_957b/CANN 9.1.T500): all VF-called helpers (`gs_to_float`/`gs_from_float`/`gs_fetch`/`gs_clip`/...) initially `__aicore__ inline` → compile failed with the `simt_callee` note; one compile iter to add `__simt_callee__ __attribute__((always_inline))` to each → build PASS, 29/29 T1 precision.
- **Other instances (predicted)**: any greenfield or ported SIMT (L3) kernel on CANN ≥9.1.T500 whose `__simt_vf__` body calls per-thread scalar helpers (gather/scatter index math, coordinate clamp, dtype cast helpers). When porting an older SIMT scaffold (e.g. 28_Interpolate) forward to 9.1.T500, expect this even though the original compiled clean.
- **Related**: EC-1 (the prior, weaker `__aicore__`-on-helper requirement this supersedes for SIMT VF callees), OL-150 (SIMT programming model — `__simt_vf__`/`LAUNCH_BOUND`/`Simt::VF_CALL`), OL-151 (SIMT helper APIs).

<!-- 迁移自 porter kb/target/ascendc/（EC-74，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Thin-TU wrapping of staged V220 algorithm headers for port_a3 V220-pure entry (default-OFF arch35)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure verified_on: soc=Ascend950PR; cann=9.0.0 Pattern: minimum-viable port_a3 V220-pure entry-point structure when default-OFF arch"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure"
confidence: inferred
status: stub
original_id: CAND-PP100
timestamp_inferred: true
tags: [candidate, inferred, pybind11.cpp, cand-pp100]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: minimum-viable port_a3 V220-pure entry-point structure when default-OFF arch35 (no upstream V351 arch35/ available):
- `<op>_kernels.cpp` — thin worker TU wrapping V220 op_kernel/*.h staged headers
- `pybind11.cpp` — host marshalling, runtime ascendcPlatform queries
- minimum helpers under `kernel/` — only what the worker TU references
- **forbidden**: `#include "arch35/<op>.h"` (default-OFF), `<op>_apt.cpp` (V351 amped TU)

Source: flat_quant kw-1 2026-05-23 (8/8 T1 BIT_EXACT + 2.24× perf in single spawn, ~250 LOC delta).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP100，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

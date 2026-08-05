---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 `op_kernel/*_base.h` `#define bfloat16_t int16_t` fallback silently disables bf16 on A5 — author local A5-safe copy under `workspace/kernel/`"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=norm-quant-port_a3-V220-pure"
phenomenon: build_failure
signal:
  - "A5 port of V220-pure port_a3 norm/quant op compiles + builds clean but bf16 inputs silently produce wrong results (or fp16-truncated results). Surface error mes"
confidence: single_run
original_id: EC-65
timestamp_inferred: true
tags: [ascendc, ec-65]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=norm-quant-port_a3-V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

- **Symptom**: A5 port of V220-pure port_a3 norm/quant op compiles + builds clean but bf16 inputs silently produce wrong results (or fp16-truncated results). Surface error message often absent — output values just don't match reference. Probe shows downstream casts producing int16-like values instead of bfloat16.
- **Root cause**: Upstream V220 `op_kernel/*_base.h` (norm and quant family base headers) contains a fallback `#define bfloat16_t int16_t` near the top, used when V220 toolchain bfloat16 support is absent. On A5 (Ascend950PR) with CANN 9.0.0 + bisheng that natively supports bfloat16, this fallback is wrong — bfloat16 should be the native type, not int16_t aliased. The fallback `#define` shadows the real type at compile time; downstream `Cast<bfloat16_t, ...>` becomes `Cast<int16_t, ...>`, silently producing wrong bit-patterns.
- **Affected**: port_a3 V220-pure norm/quant family ports (RMSNorm, LayerNorm, quant variants) that include `op_kernel/<op>_base.h` from upstream verbatim.
- **Workaround**: Author a local A5-safe copy of the base header under `workspace/<op>/kernel/` (e.g., `<op>_base_a5.h`) with the fallback `#define` removed, and `#include` the local copy instead of the upstream V220 header. Do NOT modify the upstream submodule.
- **Status**: OPEN (upstream-V220-side compatibility fallback that is hostile to A5)
- **Cross-ref**: KB_INDEX EC-1..EC-65 row; OL-185 V220→V351 calibration anchor.

<!-- 迁移自 porter kb/target/ascendc/（EC-65，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

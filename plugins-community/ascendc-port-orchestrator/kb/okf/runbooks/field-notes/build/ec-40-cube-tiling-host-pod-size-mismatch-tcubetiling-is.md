---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Cube tiling host POD size mismatch — `TCubeTiling` is 50 int32 (200 B), not 51"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-40
timestamp_inferred: true
tags: [tcubetiling, ascendc, ec-40]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom (compile error)**:
  ```
  batchmatmul_tiling.h:80:37: error: static assertion failed:
      <name>TilingHost size must match TCubeTiling
  ```
- **Trigger**: a host-side POD that mirrors `TCubeTiling` (so pybind can `torch::empty + .copy_()` the tiling into NPU memory) is declared with the wrong field count — typically estimated as 51 from incomplete docs, but the actual count for CANN 9.0.0 is **50 int32 fields = 200 bytes**.
- **Fix**: Inspect `kernel_tiling.h` `struct TCubeTiling` field-by-field and count exactly. For CANN 9.0.0: 50 int32 fields. Mirror with `static_assert(sizeof(TilingHost) == 50 * sizeof(int32_t))`. **Recount whenever CANN version changes** — this is brittle to upstream additions.
- **Better long-term fix (Opt2 path, OL-91 step 3)**: Eliminate the host POD entirely by using on-stack `TCubeTiling tiling{}` filled in the kernel from scalar args + the non-`__gm__` `Init(const TCubeTiling*, TPipe*)` overload. No host H2D, no size-match concern, ~5–10 µs faster per call.
- **Evidence**: 1_BatchMatmul (2026-04-28) Phase C iter 2. Once op#1 documented the exact count, op#4 / op#5 / op#3 all built clean on first try.

<!-- 迁移自 porter kb/target/ascendc/（EC-40，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

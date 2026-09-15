---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Host↔device tiling struct layout mismatch from mixed-width fields — use explicit packing"
description: "applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-178
timestamp_inferred: true
tags: [ascendc, ol-178]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; bisheng=all; op_class=all_data_movement`
`verified_on: soc=Ascend910_9382; cann=9.0.0; bisheng=15.0.5`

**Principle**: When a tiling struct (shared between host GCC and device bisheng) contains mixed-width integer fields (int32_t alongside int64_t), the two compilers may insert different padding, leading to field misalignment.

**Fix**: Wrap the tiling struct in `#pragma pack(push, 1)` / `#pragma pack(pop)` in the kernel header.

**Evidence**: 12_Permute a3-ds kw-3 iter-3 (2026-05-21, Ascend910_9382 V220). `#pragma pack(1)` fixed stride array garbage reads.

**Cross-ref**: OL-77 (GM tiling struct field-by-field read — kernel-side rule); EC-40 (Cube tiling host POD size mismatch).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-178（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

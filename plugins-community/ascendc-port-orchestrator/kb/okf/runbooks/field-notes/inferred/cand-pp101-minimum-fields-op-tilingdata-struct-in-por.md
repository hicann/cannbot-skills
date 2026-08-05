---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Minimum-fields `<Op>TilingData` struct in port_a3 V220-pure path"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure verified_on: soc=Ascend950PR; cann=9.0.0 Pattern: when porting V220 cube+vec fused op via thin-TU wrapping (CAND-PP100), grep"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure"
confidence: inferred
status: stub
original_id: CAND-PP101
timestamp_inferred: true
tags: [candidate, inferred, begin_tiling_data_def, tcubetiling, cand-pp101]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=port_a3_to_a5_V220-pure`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: when porting V220 cube+vec fused op via thin-TU wrapping (CAND-PP100), grep `tilingData->` first in the worker TU + helpers, mirror only the accessed fields in your local `<Op>TilingData` struct. Skip the upstream `BEGIN_TILING_DATA_DEF` + nested `TCubeTiling` reconstruction.

Source: flat_quant kw-1 2026-05-23 (4-field struct sufficed where upstream had 11-field nested struct).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP101，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Pybind hardcoded platform constants — mirror upstream runtime ascendcPlatform queries to avoid silent V220-vs-V351 mismatch"
description: "applies_to: soc=all; cann=9.0.0+; op_class=all verified_on: soc=Ascend950PR; cann=9.0.0 Anti-pattern: pybind11.cpp hardcoding platform constants (AIC_NUM, AIV_NUM, LIB_API_WS_BYTES) at host-side. On V"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=9.0.0+; op_class=all"
confidence: inferred
status: stub
original_id: CAND-PP99
timestamp_inferred: true
tags: [candidate, inferred, aic_num, aiv_num, lib_api_ws_bytes, ascendcplatform, cand-pp99]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=9.0.0+; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Anti-pattern: pybind11.cpp hardcoding platform constants (`AIC_NUM`, `AIV_NUM`, `LIB_API_WS_BYTES`) at host-side. On V220→V351 ports these values silently mismatch (V220 hard-coded to V220 numbers, deployed on V351 → wrong workspace size + wrong block dim). Use runtime `ascendcPlatform` queries (`GetCoreNumAic()` / `GetCoreNumAiv()` / `GetLibApiWorkSpaceSize()`) mirroring upstream pybind11.cpp pattern.

Source: lightning_indexer_grad kw-NEW 2026-05-23 anti-patterns §3.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP99，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

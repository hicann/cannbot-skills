---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CPU-truth construction rule for per-tensor int8 dynamic quant — multiply by precomputed `_INV_127`, NOT divide by `127.0`"
description: "applies_to: soc=all; cann=all; op_class=quant verified_on: soc=Ascend950PR; cann=9.0.0 Pattern: per-tensor int8 dynamic quant CPU-truth synthesis must mirror CANN's Muls(absmax, DYNAMIC_QUANT_FACTOR)"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; op_class=quant"
confidence: inferred
status: stub
original_id: CAND-PP103
timestamp_inferred: true
tags: [candidate, inferred, _inv_127, cand-pp103]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; op_class=quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`

Pattern: per-tensor int8 dynamic quant CPU-truth synthesis must mirror CANN's `Muls(absmax, DYNAMIC_QUANT_FACTOR)` semantics — multiply by precomputed `_INV_127 = 1.0/127.0` constant. Do NOT divide by `127.0` directly. Mathematically equivalent but LSB-different on fp32 → silent bit-exact drift when CPU truth uses divide.

Source: grouped_matmul_swiglu_quant_v2 kw-3 2026-05-24 (Path-B truth synthesis, 8/8 cases bit-exact post-fix).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP103，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

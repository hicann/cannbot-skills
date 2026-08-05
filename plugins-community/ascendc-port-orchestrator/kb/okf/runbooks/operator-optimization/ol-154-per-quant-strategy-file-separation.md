---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Per-quant-strategy file separation in multi-quant-mode ops"
description: "When an op ships 3+ quant strategies, upstream A5 ports prefer one per-strategy header in op_kernel/arch35/ dispatched by tilingkey from a thin dispatcher, not one template-parameterized file."
confidence: single_run
original_id: OL-154
classified_by: llm-assisted
timestamp_inferred: true
tags: [multi-variant-organization, optimization, ol-154, quantization, moe, tiling-key-dispatch]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=quant-multi-strategy, moe-quant, attention-quant`.
Source: `docs/analysis/UPSTREAM_A5_VALIDATION_SWEEP_2026_05_14.md` — `moe_init_routing_v3` inventory.

**Rule**: When an op supports **≥ 3 quantization strategies** (dynamic / static / per-tensor /
per-token / hifloat8 / mxfp8 / ...), upstream A5 ports prefer a **per-strategy header file** in
`op_kernel/arch35/`, dispatched by tilingkey from a thin dispatcher. Do NOT collapse the strategies
into one template-parameterized file.

**Why**: each quant strategy has distinct codegen —
- Different `CastTrait` constants (SatMode rule differs per output dtype, per OL-146).
- Different `Pack` patterns (1-step for FP8/HiFloat8; 3-step for INT8; none for FP4).
- Different scale-matrix layout (mxfp8 uses `fp8_e8m0_t` e8m0 fractal per OL-145; per-tensor static =
  single scalar; per-token dynamic = per-row scale tensor).
- Different precision-correctness path (dynamic quant needs absmax-reduce; static uses a pre-computed
  scale).

A template-parameterized merge loses the specialization opportunity — the compiler keeps unused code
paths in each instantiation — and the explicit-config-per-file pattern is more readable for
maintainers.

**Canonical example** — `moe_init_routing_v3/op_kernel/arch35/` has 6 quant strategies × one file
each: `moe_v3_gather_dynamic_quant.h` (per-row absmax dynamic INT8/FP8),
`moe_v3_gather_static_quant.h` (pre-computed scale), `moe_v3_gather_hif8_pertensor_quant.h`,
`moe_v3_gather_hif8_pertoken_quant.h`, `moe_v3_gather_hif8_quant.h` (generic HiFloat8 dispatcher),
`moe_v3_gather_mxfp8_quant.h` (MicroScaling FP8 with e8m0 block scale). The `_apt.cpp` includes all
per-strategy headers and selects the gather variant with `TILING_KEY_IS(...)` guards
(`if (TILING_KEY_IS(<dynamic-int8>)) { MoeV3GatherDynamicQuant<...> op; op.Init(...); op.Process(); }`
... 6 strategies total).

**When to apply**: op declares ≥ 3 distinct quant strategies in its `_def.cpp` DataType config, OR
the op-class must ship variants for {INT8, FP8 E4M3, FP8 E5M2, HiFloat8, mxfp8} in production.

**When NOT to apply**: op has only 1 quant strategy (single-file is fine, e.g. INT8-only
`rms_norm_quant`), OR 2 strategies sharing ≥ 90% of code (template-parameterize is fine).

**Relation**: this is a specialization of `P-P91` (multi-variant kernel files dispatched by
`TILING_KEY`).

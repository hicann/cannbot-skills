---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Non-aligned tail write to GM via `DataCopyPad` + `DataCopyExtParams` byte-level `blockLen` — replaces host-side pad+narrow cheat [V351+V220, data-movement, anti-cheat]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=elementwise+quant+strided-write applies_to_backend: ascendc verified_on: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5 (2026-05-18 task"
confidence: single_run
original_id: P-P98
timestamp_inferred: true
tags: [patterns-index, optimization, datacopypad, datacopyextparams, blocklen, p-p98, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=elementwise+quant+strided-write`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5 (2026-05-18 task #22 hardware probe — workspace/probe_datacopypad_v300_tail/PROBE_REPORT.md: blockLen ∈ {31,33,47,63} all wrote exactly N bytes, no overflow, no runtime error)`

**EC-23 scope note**: EC-23 documents V220 `DataCopyPad` UB→GM crash; this is **V220-only**. On V351, the primitive works cleanly per the hardware probe above. P-P98 applies on V351; foreach-class ops with non-aligned tails MUST use DataCopyPad in kernel rather than host-side pad+narrow workaround in pybind.

**Trigger**: kernel writes a row / tail / variable-length output whose byte length is NOT a multiple of 32 (e.g., int8 outputs with row width `H` bytes, fp16 outputs with column width not 16-multiple, variable-length scatter tails). Naive `DataCopy(gm, ub, count)` silently truncates to nearest 32B; pre-allocating padded output + post-trim in pybind is the cheat we want to retire (OL-167).

**Technique**: use `DataCopyPad(gm, ub, DataCopyExtParams)` for the non-aligned write. `DataCopyExtParams.blockLen` is BYTES (not 32B-blocks), supports `1..2097151`, no alignment constraint on the write size. UB source still needs 32B-aligned start; framework reads the aligned-up source block, writes exactly `blockLen` bytes to GM (downstream bytes untouched).

```cpp
// 3-line anchor — UB→GM non-aligned write:
DataCopyExtParams cp{
    /*blockCount=*/ static_cast<uint16_t>(numRows),
    /*blockLen=*/   static_cast<uint32_t>(rowBytes),        // bytes, can be ANY 1..2097151
    /*srcStride=*/  static_cast<uint32_t>(ubStrideBytes),   // GM-side bytes, UB-side 32B-blocks
    /*dstStride=*/  static_cast<uint32_t>(gmStrideBytes),
    /*rsv=*/        0,
};
DataCopyPad(dstGm, srcUb, cp);   // GM receives exactly blockCount × blockLen bytes
```

**Symmetric form for GM→UB non-aligned reads (when relevant)**: requires `DataCopyPadExtParams<T>` to specify padding (right-pad to 32B with `padValue` or random):

```cpp
DataCopyExtParams cp{1, /*byteLen=*/47, 0, 0, 0};
DataCopyPadExtParams<half> pad{/*isPad=*/true, /*leftPad=*/0, /*rightPad=*/8, half(0)};   // 47B + 8×2 = 63B ≥ 32B aligned
DataCopyPad(dstUb, srcGm, cp, pad);
```

**Anti-pattern (don't apply when)**:
- `count * sizeof(T) % 32 == 0`: just use plain `DataCopy(gm, ub, count)` — DataCopyPad path has slightly heavier setup, no benefit when alignment is naturally met.
- UB→UB transfer: `DataCopyPad` on UB→UB routes through GM internally (VECIN→GM→TSCM with ND→NZ conversion) — much slower than direct `DataCopy(ub, ub, count)`. Use plain `DataCopy` for UB↔UB.
- Aligned wide stride with 32B-block granularity: plain `DataCopyParams` with 32B-block `blockLen` is more compact when the math naturally lands on block boundaries (e.g., matrix row-write with row width = 32B multiple).
- "Tail is just a few extra bytes, I'll pad in pybind": NO — that's the cheat this pattern exists to replace (OL-167).

**Other instances (predicted)**:
- Quantization output writers (int8 / int4 outputs with arbitrary H).
- Variable-length scatter/gather tails (NNZ-driven write extent).
- Sparse op outputs where NNZ is data-dependent and not 32-aligned.
- Cross-row strided writes where row width is dtype-dependent (fp16 with odd column count).
- Any kernel currently relying on `align_up64(rowBytes, 32)` over-allocation in pybind followed by `narrow + contiguous` — should migrate to DataCopyPad in kernel.

**Evidence**:
- Direct API reference: `~/workspace/a5/tasks/datacopy_api.md` §2.3 (silent truncation behavior) + §2.4 (DataCopyPad usage) + §3.3 (UB→GM examples) + §6.1 (alignment trap).
- Migration target archives (currently using the cheat — to be reworked when their op-gen is re-run with this pattern in worker brief): `11_DequantSwigluQuant_v3.2_cold` (pybind11.cpp:213-259), elu (a3_to_a5_port pybind11.cpp:52,69,100), clipped_swiglu (a3_to_a5_port pybind11.cpp:108).
- P149 finalize gate logs (`finalize_pipeline.py:GateID.PYBIND_HOST_BUSINESS_LOGIC`): catches `narrow(..., 0, ...).contiguous()` post-kernel in pybind; this pattern is the kernel-side fix that prevents P149 retraction.

**Cross-reference**:
- OL-167 — the principle (DataCopy silent truncation + anti-cheat policy)
- EC-23 — DataCopyPad UB→GM V220 crash mitigation (orthogonal: this is V220 build-side gotcha)
- `vendor/ascendc-kernelgen-data/npu_benchmark/level2/8_QuantScatter.py` — typical op shape where this pattern applies (int8 output, non-32B row widths)

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P98，convert_patterns_to_okf.py）。confidence 未升格。 -->

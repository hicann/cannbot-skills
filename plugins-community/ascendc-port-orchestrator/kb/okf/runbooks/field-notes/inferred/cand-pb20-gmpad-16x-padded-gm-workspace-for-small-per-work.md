---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "16x-padded GM workspace for small per-work-unit scalar outputs in pure-AIV class kernels"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-03-21; op_class=normalization (mean/rstd-emit), dynamic-quant (per-row scale-emit), any pure-AIV class kernel with small scalar GM outputs verifie"
phenomenon: build_failure
signal:
  - "kernel needs to write small per-work-unit scalar outputs (e.g., per-(n, g) mean / rstd in normalization ops, per-row dynamic quant scale, per-token statistic in"
confidence: inferred
status: stub
original_id: CAND-PB20-GMPAD
timestamp_inferred: true
tags: [candidate, inferred, mean, rstd, datacopy, num_groups, layernorm, cand-pb20-gmpad]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=2026-03-21; op_class=normalization (mean/rstd-emit), dynamic-quant (per-row scale-emit), any pure-AIV class kernel with small scalar GM outputs`
`verified_on: soc=Ascend950PR; cann=9.0.0 (group_norm_silu_quant — single-op evidence)`
`unverified_on: soc=Ascend910_9382 (A3); soc=Ascend910B3 (A2)`

**Trigger**: kernel needs to write small per-work-unit scalar outputs (e.g., per-(n, g) `mean` / `rstd` in normalization ops, per-row dynamic quant scale, per-token statistic in fused-attn) to GM from a **pure-AIV class kernel** (the `extern "C" __global__ __aicore__ void f(...) { KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); class.Init(...).Process(); }` pattern). PB-20's existing workaround ("DataCopy UB→GM") presumes the output buffer is large enough for natural 32B-aligned DataCopy blocks. When the per-work-unit output is a SINGLE scalar (e.g., 4 bytes fp32), the 32B-alignment + inter-AIV-write-race combination silently corrupts adjacent slots.

**Pattern**: in pybind, allocate the scalar output buffer with an extra 16-element trailing dim:
```cpp
auto mean_ws = torch::empty({N, num_groups, 16}, opts);    // 16x oversize on innermost
```
In kernel, every AIV writes a 16-element T block to its `(n, g) * 16` GM offset via `DataCopy`:
```cpp
LocalTensor<T> ub_block = ubBuf_.Get<T>();
ub_block.SetValue(0, mean_t);                              // scalar value at lane 0
DataCopy(gmMean_[idx * 16], ub_block, 16);                 // 32B-aligned block write
```
After kernel returns, pybind extracts lane-0 across the trailing dim to recover the natural shape:
```cpp
auto mean = mean_ws.select(/*dim=*/2, /*index=*/0).contiguous();   // back to {N, num_groups}
```

**Why it works**: 16 elements of T (fp16: 32B, fp32: 64B) is always ≥ the 32B DataCopy MTE3 granularity, so every AIV's write lands in its own naturally-aligned 32B block — no inter-AIV write race, no alignment fault, no silent corruption. The cost is a 16× oversize on the (typically small) statistic output — negligible when `num_groups` ≤ 32 and N is bounded.

**Cost**: 16× memory oversize on the small output buffer only (the main data path is unaffected). For typical norm ops with `mean[N, G]` where `G ≤ 32`, the overhead is `N·G·16·sizeof(T)` bytes — bounded.

**Reusable across**: `LayerNorm` / `InstanceNorm` / `RMSNorm` / `BatchNorm` (mean/rstd emit), dynamic-quant ops (per-row scale emit), fused-attn statistics, any pure-AIV class kernel that emits one or a few scalars per work unit.

**Evidence**:
- group_norm_silu_quant (2026-05-13, A5 fused GroupNorm+SiLU+Quant port from A3 aclnn): iter 1→2 tried `const_cast<__gm__ T*>(gmMean_.GetPhyAddr())` + `mean_ptr[idx] = mean_t` → silent garbage on mean/rstd (max_abs_diff 2.09e+28 on case 6 bf16 — uninitialized FP_MAX sentinel). Iter 3→4 applied 16x-pad pattern → mean/rstd bit-exact (0.0 diff) on all 8 cases. Diff size: ~30 lines in `EmitMeanRstd` helper + 6 lines in pybind11.cpp.

**Promote-to-canonical criteria**:
1. Validated on ≥ 1 more op in a different op_class (e.g., a dynamic-quant op with per-row scale emit, or a different normalization op).
2. Compile-gate against public AscendC headers (C34b).
3. Confirms PB-20 workaround sub-bullet is the right host for this refinement, OR is distinct enough to promote as P-P (data-movement pattern).

**Related**:
- PB-20 (parent — pure-AIV class kernel cannot use `GlobalTensor::SetValue` or raw `__gm__ p[i]=v`; DataCopy UB→GM is the only working pattern). CAND-PB20-GMPAD is the scalar-output sub-case of PB-20's workaround.
- Iron law §5 (literal translation — keep scalar outputs as separate GM tensors, don't pack them creatively into the main output buffer).

## A3→A5 arch35 upstream-port patterns (2026-05-13, from `/aog-prior-art-verify` Phase 6)

> **Source**: offline analysis of CANN team's A5 ports for `ada_layer_norm` (norm/, 4 files / 1466 lines) and `fused_quant_mat_mul` (matmul/, 3 files / 226 lines). Phase 3-5 hardware verify pending — these are upstream_pass candidates IF verify confirms upstream actually meets our precision+perf bar; if upstream FAILS verify, these flip to upstream_fail (anti-pattern) tagging.
> **Detection skill**: `src/skills/aog-prior-art-verify/SKILL.md`
> **Mechanical scanners**: candidates must clear C34a (identifier denylist), C34b (compile-gate), C34c (n-gram copy-shape ≤ 5%), C35 (KB-overlap) before promotion.

### CAND-A35-PORT-1: arch35 port is structural reshape, NOT mechanical V220-strip

**Source**: upstream_pass (cross-op evidence: ada_layer_norm + fused_quant_mat_mul)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all; unverified_on=Ascend910_V220 (A3 — pattern is A5-specific)`

**Principle**: an A3 op WITHOUT `__CCE_AICORE__ == 220` guards or `impl/dav_c220/*` deps is NOT a candidate for mechanical V220-strip when porting to A5. It needs structural reshape — either algorithm-split (welford vs full_load, see CAND-A35-DISPATCH-1) or composition over re-impl (`FusedOpType` template arg, see CAND-A35-COMPOSITION-1). The presence of `op_kernel/arch35/` upstream IS the canonical signal that CANN team found this structural reshape; if no `arch35/` exists, the op is either trivially portable OR not yet ported.

**Decision rule (for kw briefs in port_a3_to_a5 mode)**:
1. Read `<backward>/<op>/op_kernel/arch35/`; if present, that IS the port → route through prior-art-verify
2. If absent, check A3 source for `__CCE_AICORE__ == 220` guards; if present → mechanical strip is the right port
3. If absent AND no V220 guards → structural reshape required → flag as researcher-grade, not single-shot kw

**Anti-pattern (counter-example)**: the harness-generated `ada_layer_norm` kernel postmortemed at `output/a3_to_a5_port/src/kernels/ada_layer_norm.postmortem-handrolled-partial/` (923 lines, 0.38× perf) tried mechanical-port from A3's `AdaLayerNormND` class, ignoring upstream's structural split — cost $69.20 / 244 min wasted.

### CAND-A35-DISPATCH-1: tilingkey-driven multi-algorithm class split

**Source**: upstream_pass (ada_layer_norm: `AdaLayerNormFullLoad` + `AdaLayerNormWelford` selected by tilingkey in `impl.h`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=norm`

**Principle**: when an op has two qualitatively-different code paths (full-load vs sliced; fast-path vs general; one-pass vs Welford), prefer SPLIT into two top-level template-instantiated classes selected by a thin tilingkey dispatcher header. Avoid intra-class `if/else` branching on regime — the compiler can't dead-code-eliminate across template instantiations and the inner loop pays dispatch overhead per iteration.

**Concrete (3 files):**
```
op_kernel/arch35/
├── <op>_full_load.h    # AdaLayerNormFullLoad<T,U,Y,OP_CODE> class
├── <op>_welford.h      # AdaLayerNormWelford<T,U,Y,OP_CODE> class
└── <op>_impl.h         # template-method definitions (NOT a class) — pulled in by both
```
The `cpp` entry point reads the tilingkey, `if constexpr` instantiates one of the two classes, calls `Init` + `Process`. Each class is fully specialized at compile-time → no runtime dispatch overhead inside the hot loop.

### CAND-A35-COMPOSITION-1: fused matmul/conv ops compose reusable arch35 kernel classes via `FusedOpType` template param

**Source**: upstream_pass (fused_quant_mat_mul A5 imports `QuantBatchMatmulV3::MatmulAswKernel*` with `FusedOpType::SWIGLU | RELU | NONE`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=matmul-fused; predicted_applicable_to=conv-fused, attention-fused`

**Principle**: when a fused op = base op (matmul/conv/attn) + epilogue (RELU/SWIGLU/GELU/QUANT), prefer template-parameter composition over class-level re-implementation. CANN arch35 provides templated kernel classes with epilogue template params; the fused-op kernel becomes a thin tilingkey dispatcher (~226 lines for fused_quant_mat_mul, vs ~600 lines of equivalent A3 code).

**Decision rule (for kw briefs)**:
1. Identify the base op the fused op composes (matmul, conv, ...)
2. Check `cann/ops-nn/<base_op_dir>/op_kernel/arch35/` for reusable templated kernel classes
3. If `FusedOpType` (or equivalent epilogue-template-param) is available, use it
4. Re-implementing from scratch is the wrong default for fused matmul/conv on A5

**Concrete signature**:
```cpp
#include "../../quant_batch_matmul_v3/arch35/qbmm_cube_on_the_fly_abl1_full_load.h"

MatMulASWKernel<DTYPE_X1, ..., FusedOpType::SWIGLU> op;
op.Init(x1, x2, bias, scale, yScale, y, user, &tilingData, &tPipe);
op.Process();
```

### CAND-A35-DISPATCH-2: 2-level tilingkey (`op_type` × `kernel_type`) cross-product

**Source**: upstream_pass (fused_quant_mat_mul `_tilingkey.h`)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=cube-fused`

**Principle**: for fused-matmul/cube ops with multiple L1-load strategies × multiple epilogue fusions, declare TWO independent tilingkey dimensions (`TPL_OPTYPE`, `TPL_KERNELTYPE`) instead of flattening to one keyspace. Cross-product instantiation in a nested `if constexpr` chain. Lets tiling code reason about L1-load-strategy independent of fusion, avoids combinatorial explosion in tilingkey enum + makes adding a new fusion or new L1-strategy a single-dimension change.

### CAND-A35-CAST-1: typed `CastTrait` / `LayerNormConfig` / `NormalizeConfig` constants over implicit primitives

**Source**: upstream_pass (ada_layer_norm common.h)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: declare `constexpr CastTrait castTraitX = {RegLayout, SatMode, MaskMergeMode, RoundMode}` constants once per direction pair (b16→b32, b32→b16, f32→i16, f16→i8, f32→fp8, f32→hif8) and use named traits at call sites: `Cast(dst, src, castTraitB16ToB32, len)`. Locks `SatMode::NO_SAT` + `RoundMode::CAST_RINT` explicitly — A5 defaults may differ from V220 and naming the contract prevents silent numerical drift.

```cpp
constexpr CastTrait castTraitB16ToB32 = {
    RegLayout::ZERO, SatMode::UNKNOWN, MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr CastTrait castTraitB32ToB16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
constexpr CastTrait castTraitF32ToI16 = {
    RegLayout::ZERO, SatMode::NO_SAT, MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
```

**Anti-pattern**: relying on default-argument `Cast(dst, src, len)` overload — A5's defaults are not the same as A3's.

### CAND-A35-SIMD-1: `__VEC_SCOPE__` + `RegTensor<T>` register-level SIMD micro-API

**Source**: upstream_pass (ada_layer_norm welford.h)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: arch35 exposes a register-level SIMD micro-API (`__VEC_SCOPE__` scope, `RegTensor<T>` register, `MaskReg` + `UpdateMask<T>(len)` for tail-handling) that A3 lacks. Hot inner loops should be lowered to this API when possible — vendor `LocalTensor`-based primitives are correct but pay UB round-trip latency that the register-level API avoids.

```cpp
__VEC_SCOPE__ {
    RegTensor<float> x;
    MaskReg p = UpdateMask<float>(remaining_length);
    Duplicate(x, 0.0f);
    DataCopy(dst_ub, x, p);
}
```

Functions called from `__VEC_SCOPE__` blocks need `__simd_callee__` annotation to inline correctly.

### CAND-A35-FORMAT-1: constexpr `CubeFormat` from FORMAT_* macros

**Source**: upstream_pass (fused_quant_mat_mul A5, also QuantBatchMatmulV3 arch35)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=cube-fused`

**Principle**: cube engine specializes by `CubeFormat::{ND, NZ}` template parameter, which should be a `constexpr` derived from compile-time `FORMAT_X1` / `FORMAT_X2` / `FORMAT_Y` macros, NOT a runtime branch. A3 used runtime format checks; A5 fast-path requires constexpr.

```cpp
#if defined(FORMAT_X1) && FORMAT_X1 == FORMAT_FRACTAL_NZ
constexpr CubeFormat format_x1 = CubeFormat::NZ;
#else
constexpr CubeFormat format_x1 = CubeFormat::ND;
#endif
```

### CAND-A35-TASKTYPE-1: explicit `KERNEL_TASK_TYPE_DEFAULT` per code path

**Source**: upstream_pass (ada_layer_norm.cpp head, fused_quant_mat_mul.cpp per-block)
**Scope**: `soc=Ascend950PR; cann=9.0.0; op_class=all`

**Principle**: declare `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY | KERNEL_TYPE_AIC_ONLY | KERNEL_TYPE_MIX_AIC_1_1)` at the head of each `TPL_OPTYPE` block matching the actual cube/vec ratio used by that path. The runtime uses this to allocate the right resource lanes; default (no declaration) is suboptimal for cube-heavy fused matmul or vec-heavy norm ops.

### Promotion gate notes

These 7 candidates need:
1. Cross-op evidence beyond ada_layer_norm + fused_quant_mat_mul. **CAND-A35-DISPATCH-1** + **CAND-A35-COMPOSITION-1** could merge into a single "tilingkey dispatcher + template specialization" canonical OL/P-P after a third op confirms.
2. Phase 3-5 hardware verify of upstream MUST land before promotion — if upstream FAILS verify on our edge_dataset, these flip to upstream_fail tagging (anti-pattern) and the harness regen brief needs the FAILED rows annotated. Current state: NOT_YET_VERIFIED.
3. Mechanical scanners: C34a/b/c/35 against public AscendC headers + KB overlap. The snippet code samples above are sampled (not verbatim copies) — should pass C34c n-gram ≤ 5%.

**Linked to**: `OL-141` (target `op_kernel/arch35/` is advisory prior art; never skip generation or truth).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PB20-GMPAD，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

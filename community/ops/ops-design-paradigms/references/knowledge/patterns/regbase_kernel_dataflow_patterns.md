---
title: Regbase Kernel Dataflow Patterns
purpose: Explain the three-layer regbase structure: outer kernel shell, UB-level CopyIn/Compute/CopyOut flow, and VF/reg-compute body.
read_when:
  - You are structuring a new regbase kernel or reading an existing reference implementation.
  - The task needs a clear separation between shell logic, UB staging, and VF micro-instructions.
not_for:
  - Host runtime setup only
  - Pure precision debugging
keywords:
  - dataflow
  - copyin compute copyout
  - VF
  - reg compute
next_reads:
  - regbase_buffer_partitioning.md
  - regbase_sync_patterns.md
  - ../api/regbase_api_whitelist.md
depth: foundation
topic_type: pattern
---

# Regbase Kernel Dataflow Patterns

This note explains the most important structural pattern in regbase work: the outer kernel shell, the UB-level tile flow, and the inner VF/reg-compute body are different layers and should be designed separately.

## 1. The Three Working Layers

For kernel-side implementation, read the operator in three layers:

1. **Kernel shell**
   - `__global__` entry
   - `TILING_KEY_IS(...)`
   - `TPipe`
   - `Init` / `Process`
2. **UB tile flow**
   - `CopyIn -> Compute -> CopyOut`
   - `GlobalTensor`, `LocalTensor`, `TQue`
3. **VF body**
   - `__VEC_SCOPE__`
   - `RegTensor`
   - `MaskReg`
   - `Reg::*` / `MicroAPI::*` VF-safe loads such as `DataCopy<..., LoadDist::...>`
   - arithmetic / compare / cast / select
   - `Reg::*` / `MicroAPI::*` VF-safe stores such as `DataCopy<..., StoreDist::...>`

Do not flatten these three layers into one description. The outer shell organizes work; the VF body performs the core register math.

## 2. Minimal Outer-Shell Pattern

```cpp
extern "C" __global__ __aicore__ void op_entry(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingData);
    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        KernelOp op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}
```

Reference shape:

- a small unary regbase outer-shell implementation with one clear `TILING_KEY_IS(...)` branch and one compact `Init -> Process` flow

## 3. Minimal UB-Level Flow Pattern

```cpp
__aicore__ inline void Process()
{
    for (uint32_t i = 0; i < tileNum_; i++) {
        uint32_t count = (i == tileNum_ - 1) ? tailTileElementNum_ : TILE_LENGTH;
        CopyIn(i, count);
        Compute(count);
        CopyOut(i, count);
    }
}
```

This level decides:

- how the tile loop runs
- where the tail is handled
- where UB-stage ownership changes

It does not yet explain the register math.

## 4. Minimal VF Body Pattern

```cpp
__VEC_SCOPE__
{
    RegTensor<float> vin, vtmp, vout;
    MaskReg mask;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask = UpdateMask<float>(size);
        DataCopy<float, LoadDist::DIST_NORM>(vin, inAddr + i * VL);
        Exp(vtmp, vin, mask);
        Adds(vtmp, vtmp, 1.0f, mask);
        Div(vout, vin, vtmp, mask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(outAddr + i * VL, vout, mask);
    }
}
```

Here:

- `inAddr` and `outAddr` are UB addresses
- load/store instructions move data between UB and register objects
- the actual math stays register-centric

Reference patterns:

- a small regbase body that clearly separates UB staging from VF math
- a fused register-chain implementation that is useful for studying VF composition, but not as a full outer-shell starter

## 5. Dtype-Split Pattern

A stable regbase pattern is to keep the outer shell mostly unchanged while splitting the VF body by dtype.

Typical split:

- fp32 path: direct load and compute
- fp16 / bf16 path:
  - load packed narrow type
  - unpack or cast to fp32 register state
  - compute in fp32 if needed
  - cast back and pack for store

This keeps routing visible and avoids mixing dtype-specific casts into the host packet.

## 6. When The Pattern Needs To Grow

The simple three-layer pattern is enough for:

- unary activation
- binary math
- short fused chains
- many light cast/compare/select tasks

The pattern needs explicit extension when:

- multiple source tiles are synchronized manually
- MTE/vector overlap is manually orchestrated with flags
- the operator has a strong reduction or transpose component
- cross-core coordination becomes part of the design

At that point, keep the same layers, but add the advanced sync and buffer rules from:

- [[regbase_sync_patterns]]
- [[regbase_buffer_partitioning]]

## Related Documents

- [[../regbase_development_guide]]
- [[regbase_buffer_partitioning]]
- [[regbase_sync_patterns]]
- [[regbase_operator_patterns]]
- [[../api/regbase_api_whitelist]]

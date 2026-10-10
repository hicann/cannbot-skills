# Regbase Development Guide

This is the practical mainline guide for developing a regbase operator. Use it when you need the shortest path from a task statement to a workable `ascend950 / regbase` implementation plan.

This guide is intentionally grounded in real operator code. Do not treat regbase development as a blank-page exercise.

## 1. Start From A Real Regbase Reference

Do not start a new regbase operator from pure imagination. Before drafting code, inspect at least one existing regbase operator implementation.

Use [[../reference-ops/open_source_operator_table]] to locate a candidate operator, then inspect its actual source files:

- outer entry and routing: usually `op_kernel/*_apt.cpp` or `op_kernel/*.cpp`
- regbase body: usually `op_kernel/arch35/*_regbase.h` or another arch35 header

For the first pass, the minimum useful reference review is:

1. How the kernel entry is declared and routed with `TILING_KEY_IS(...)`
2. How `Init` and `Process` divide work
3. How UB-level `CopyIn -> Compute -> CopyOut` is organized
4. How `Compute` enters `__VEC_SCOPE__` and uses `RegTensor`, `MaskReg`, `LoadDist`, and `StoreDist`
5. How dtype-specific VF paths are split

Good starting reference shapes:

- a small unary regbase implementation with a clear `_apt.cpp` entry and `arch35/*_regbase.h` body
- a fused register-chain implementation that makes the VF path explicit without hiding it behind too much framework structure
- an advanced explicit-sync implementation only when the task really needs manual overlap or ping-pong coordination

Reference code is for learning the existing regbase writing style. It is not automatic permission to copy whole implementations.

## 2. The Correct Layering Model

The easiest way to get regbase wrong is to flatten all layers into one.

Use this four-layer model instead:

1. **Host and tiling layer**
   - chooses dtype route, block split, tile shape, and launch parameters
2. **Kernel outer shell**
   - owns `TPipe`, `GlobalTensor`, `TQue`, `LocalTensor`, `Init`, and `Process`
3. **UB-level dataflow**
   - moves tiles between GM and UB through `CopyIn -> Compute -> CopyOut`
4. **VF / reg-compute inner body**
   - runs inside `__VEC_SCOPE__`
   - uses `RegTensor`, `MaskReg`, `LoadDist`, `StoreDist`, and VF-safe reg-compute calls exposed as `AscendC::Reg::*` or the common alias `AscendC::MicroAPI::*`

That means regbase is not “VF only” and it is not “queue-free”. Real regbase implementations often still have an outer shell and UB-stage movement. The regbase identity comes from the compute core being written as VF/reg-compute code instead of as a `LocalTensor`-centric compute body.

See also:

- [[patterns/regbase_kernel_dataflow_patterns]]
- [[dev-experience/regbase_programming_notes]]

## 3. Prepare The Host And Tiling Packet First

Before writing VF code, make sure the host side can explain:

- which dtype route is active
- which `TILING_KEY_IS(...)` branch should run
- how many blocks/cores are used
- how much work each block owns
- how tail elements are handled

Small routing example:

```cpp
extern "C" __global__ __aicore__ void op_entry(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(TilingData);
    GET_TILING_DATA_WITH_STRUCT(TilingData, tilingData, tiling);

    TPipe pipe;
    if (TILING_KEY_IS(101UL)) {
        KernelOp<DTYPE_X> op;
        op.Init(x, y, workspace, &tilingData, &pipe);
        op.Process();
    }
}
```

Use the reference operator to decide:

- whether one `TILING_KEY` is enough or multiple routes are needed
- whether dtype differences only affect VF internals or also affect UB planning
- whether the task needs one kernel branch or a split between simple and advanced routes

See also:

- [[patterns/kernel_design_patterns]]
- [[patterns/tiling_patterns]]
- [[dev-experience/requirements_analysis_patterns]]

## 4. Build The Kernel Outer Shell

The outer shell is where regbase code still looks like a normal kernel implementation. Typical responsibilities are:

- bind GM tensors
- initialize `TPipe`
- allocate `TQue` or local UB buffers
- decide tile loop count and tail count
- call `CopyIn`, `Compute`, and `CopyOut`

Minimal structure:

```cpp
class KernelOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const TilingData* tiling, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t count);
    __aicore__ inline void Compute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t count);

    TPipe* pipe_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    TQue<QuePosition::VECIN, 1> queIn_;
    TQue<QuePosition::VECOUT, 1> queOut_;
};
```

Do not put register-level math directly into `Process`. Keep the outer shell readable and make the VF core visible as its own stage.

## 5. Organize UB-Level `CopyIn -> Compute -> CopyOut`

At the UB level, a regbase kernel still needs a stable tile pipeline:

1. `CopyIn`: bring the current tile from GM into UB-local state
2. `Compute`: consume that UB-local tile and run the VF body
3. `CopyOut`: write the finished UB-local tile back to GM

This level is about UB ownership, not register instructions. Objects here are usually:

- `GlobalTensor<T>`
- `LocalTensor<T>`
- `TQue`
- raw `__ubuf__` pointers derived from UB buffers

That is why “CopyIn / Compute / CopyOut” is the right kernel-level language, while “Load / Compute / Store” belongs one layer deeper inside the VF body.

See also:

- [[patterns/regbase_kernel_dataflow_patterns]]
- [[patterns/regbase_buffer_partitioning]]

## 6. Put Register-Level Math Inside VF

Inside `Compute`, the regbase-specific part is the VF body. This is where you enter `__VEC_SCOPE__` and operate on register objects instead of treating UB buffers as the final compute model.

Representative fragment from a real implementation:

```cpp
__VEC_SCOPE__
{
    RegTensor<float> vregIn, vregAbs, vregZero, vregOut;
    MaskReg mask, cmpMask;

    for (uint16_t i = 0; i < vfLoopNum; i++) {
        mask = UpdateMask<float>(size);

        DataCopy<float, LoadDist::DIST_NORM>(vregIn, inAddr + i * VL);
        Abs(vregAbs, vregIn, mask);
        CompareScalar<float, CMPMODE::GT>(cmpMask, vregAbs, lambd_, mask);
        Duplicate(vregZero, 0.0f, mask);
        Select<float>(vregOut, vregIn, vregZero, cmpMask);
        DataCopy<float, StoreDist::DIST_NORM_B32>(outAddr + i * VL, vregOut, mask);
    }
}
```

This is the core regbase writing style:

- load from UB address into `RegTensor`
- compute on `RegTensor`
- use `MaskReg` to express the active lanes
- store back from `RegTensor` to UB address

For fp16/bf16 routes, the common pattern is:

1. load packed narrow type from UB
2. unpack or cast to a wider register type
3. compute in fp32 if needed
4. cast back
5. pack and store

See also:

- [[api/regbase_api_whitelist]]
- [[patterns/regbase_kernel_dataflow_patterns]]
- [[patterns/regbase_buffer_partitioning]]

## 7. Plan Buffers By Domain

A good regbase design says which state lives in which domain.

Typical domains:

- **GM domain**
  - input tensors
  - output tensors
  - tiling packet
- **UB domain**
  - staged tiles
  - ping-pong input/output buffers
  - temporary UB storage shared across tile stages
- **register domain**
  - `RegTensor`
  - `MaskReg`
  - short-lived cast, compare, select, and fused arithmetic state

Two practical rules:

- if data is reused across multiple VF steps but stays within one tile, it usually belongs in UB or registers, not GM
- if a value only exists to serve one VF chain, prefer keeping it as a register object rather than materializing another UB buffer

See also:

- [[patterns/regbase_buffer_partitioning]]
- [[dev-experience/regbase_kernel_case_notes]]

## 8. Treat Synchronization By Layer

Synchronization decisions only make sense if you first know which layer you are in.

At the **UB pipeline layer**:

- stage handoff is usually represented by the queue lifecycle or by explicit stage ordering in `Process`
- when the kernel uses queue-based UB movement, `EnQue` / `DeQue` express local stage readiness

At the **VF layer**:

- `MaskReg` is lane control, not synchronization
- `LoadDist` / `StoreDist` describe load/store distribution, not inter-stage fencing
- most simple VF bodies do not need explicit event flags

At the **advanced pipeline layer**:

- explicit `SetFlag` / `WaitFlag` show up when the design overlaps MTE and vector work manually
- use that only after confirming the reference operator really needs it

At the **cross-core layer**:

- cross-core flags are for genuine inter-core coordination
- do not use them as a substitute for local UB stage management

See also:

- [[patterns/regbase_sync_patterns]]
- [[api/regbase_api_sync]]

## 9. What To Check Before Writing Code

Before coding the first implementation:

1. pick at least one real regbase reference operator
2. identify its outer shell file and regbase body file
3. decide your `TILING_KEY` and dtype routes
4. sketch `CopyIn -> Compute -> CopyOut`
5. identify the VF loop shape and `VL`
6. decide which values stay in registers and which stay in UB
7. decide whether the task needs only default stage ordering or advanced explicit sync

If you cannot answer these seven items, the design is still too vague.

## 10. Common Mistakes

- Treating the whole kernel as one giant VF function
- Treating UB-level movement and register-level compute as the same layer
- Writing VF internals before the host and tiling packet are stable
- Choosing APIs before the dataflow and route are clear
- Assuming “regbase means no `TPipe`, no `TQue`, no `LocalTensor`”
- Copying a full reference implementation instead of borrowing its proven writing style
- Using a reference operator only for naming, without inspecting its actual kernel code

## Related Documents

- [[patterns/regbase_kernel_dataflow_patterns]]
- [[patterns/regbase_buffer_partitioning]]
- [[patterns/regbase_sync_patterns]]
- [[patterns/regbase_operator_patterns]]
- [[api/regbase_api_whitelist]]
- [[api/regbase_api_reference]]
- [[dev-experience/regbase_programming_notes]]
- [[dev-experience/regbase_kernel_case_notes]]
- [[../reference-ops/open_source_operator_table]]

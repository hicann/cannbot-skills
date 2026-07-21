# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

ASCEND_DSL_KNOWLEDGE = '''
### Ascend DSL Specification 

The Ascend DSL program consists of two parts:

1. **Host Function** — performs Core Partitioning + Tiling Strategy
2. **Kernel Function** — performs UB Allocation + Computation Logic

================================================================================
### 1. HOST FUNCTION REQUIREMENTS

--------------------------------------------------------------------------------
#### 1.1 Core Partitioning (Host)
Define how the workload is divided among hardware cores.

Requirements:
- Use `tl.num_vec_cores()` to dynamically query the number of Vector (AIV) cores on
  the current hardware. **Never hard-code a constant like `n_cores = 16`.**
- Handle the zero-work case *before* computing `n_used` or `base` (e.g., if
  `total_work_units == 0`, return early on the host and do not launch the kernel).
- For non-zero work, cap active cores to avoid empty work:
    - `n_used = min(n_cores, total_work_units)`
- Use **pivot distribution** to handle the case where workload is not divisible by core count:
    - `base = total_work_units // n_used`   — each core gets at least `base` units
    - `pivot = total_work_units % n_used`   — first `pivot` cores get one extra unit
- In the kernel, each core computes its own count and start offset:
    - `my_count = base + (1 if pid < pivot else 0)`
    - `my_start = pid * base + min(pid, pivot)`
- Should state whether inter-core synchronization is required (typically none)

Example:
    n_cores = tl.num_vec_cores()
    if total_rows == 0:
        # No work to do; host can safely return without launching the kernel.
        return
    n_used  = min(n_cores, total_rows)
    base_rows = total_rows // n_used
    pivot     = total_rows % n_used

APIs:
- `tl.num_vec_cores()` — returns the number of Vector cores on the current platform
  (maps to `ascendcPlatform.GetCoreNumAiv()` in AscendC tiling)

--------------------------------------------------------------------------------
#### 1.2 Tiling Strategy (Host)
Define how global tensors are partitioned to fit UB/L1 memory.

Requirements:
- Explicitly define tiling parameters (`tile_length`, `block_M`, `block_K`, …)
- Explain UB/L1 usage constraints that motivate tile size
- Tiling Strategy section is required even if the entire row fits in UB.

--------------------------------------------------------------------------------
#### Host Launch Rule
The host must *not* launch the kernel when there is no work:
- If `total_work_units == 0`, return early on the host and skip the launch.
- Otherwise, launch the kernel using:

    kernel_name[n_used](args...)

where `n_used = min(tl.num_vec_cores(), total_work_units)` and `total_work_units > 0`.

================================================================================
### 2. KERNEL FUNCTION REQUIREMENTS
The kernel performs all on-chip computation.

--------------------------------------------------------------------------------
#### 2.1 Allocate UB / L1 Buffers (Kernel)
All on-chip temporary buffers must be allocated here.

Allowed APIs:
- `tl.alloc_ub(size, dtype=...)`
- `tl.alloc_l1(shape, dtype=...)`

Rules:
- All UB buffers must be explicitly declared
- No alias constructs; if two roles share one buffer, simply reuse the variable name
- A UB buffer used in a copyin or copyout operation is dedicated to that data transfer.

Examples:
- `x_ub    = tl.alloc_ub(tile_length, dtype=tl.float32)`
- `out_ub = tl.alloc_ub(tile_length, dtype=tl.float32)`

--------------------------------------------------------------------------------
#### 2.2 Computation Logic (Kernel)
The kernel must explicitly load → compute → store, using **one or more** copyin / compute / copyout blocks.

Multiple blocks are allowed:
- Multiple `with tl.copyin():`
- Multiple `with tl.compute():`
- Multiple `with tl.copyout():`

##### A. Copy-In Block  
    with tl.copyin():
        tl.load(global_ptr + offsets, ub_buf)

Rules:
- All global → UB loads must be inside `tl.copyin()`

##### B. Compute Block  
    with tl.compute():
        tl.vadd(out_ub, a_ub, b_ub)
        tl.reduce_sum(dst_ub, src_ub, shared_ub)
        scalar = tl.extract_scalar(shared_ub, 0)
        ...

###### Supported Compute APIs

-------------------------
Group 1: Reduction Ops
-------------------------
API Names:
- reduce_sum
- reduce_max
- reduce_min

Parameters:
- dst_ub     : destination buffer
- src_ub     : source buffer
- shared_ub : temporary workspace (size == src_ub)

Reduction aliasing rules:
- shared_ub is a temporary workspace; its size equal src_ub.
- `dst_ub`, `src_ub`, `shared_ub` may be the same buffer
- But `dst_ub` must **not** alias `src_ub` if `src_ub` values are needed later

Example:
    tl.reduce_sum(dst_ub, src_ub, shared_ub)

-------------------------
Group 2: Unary Element-wise Ops
-------------------------
API Names:
- vexp, vsqrt, vrsqrt, vabs, vreciprocal
- vtanh, vsin, vln, verf

Parameters:
- out_ub
- src_ub
- count (optional, default = len(src_ub))

Example:
    tl.vexp(out_ub, x_ub, count=len(x_ub))

-------------------------
Group 3: Binary Element-wise Ops (Tensor–Tensor)
-------------------------
API Names:
- vadd, vsub, vmul, vdiv, vmax, vmin

Parameters:
- dst_ub
- src_ub1
- src_ub2
- count (optional, default = len(src_ub1))

Example:
    tl.vadd(out_ub, a_ub, b_ub, count=len(a_ub))

-------------------------
Group 4: Binary Element-wise Ops (Tensor–Scalar)
-------------------------
API Names:
- vadd_scalar, vsub_scalar, vmul_scalar
- vdiv_scalar, vmax_scalar
- vclamp_max, vclamp_min

Parameters:
- dst_ub
- src_ub
- scalar_value
- count (optional, default = len(src_ub))

Example:
    tl.vdiv_scalar(out_ub, x_ub, scalar, count=len(x_ub))

-------------------------
Group 5: Scalar Extraction
-------------------------
API Name:
- extract_scalar

Parameters:
- ub buffer or global pointer
- index

Rules:
- Used to convert a single element into a scalar value.
- Scalars may be used in subsequent compute ops.

Examples:
    scalar = tl.extract_scalar(ub, 0)
    scalar = tl.extract_scalar(global_ptr, index)

-------------------------
Group 6: Duplicate (Fill)
-------------------------
API Name:
- duplicate

Description:
- Fills a UB buffer with a scalar value.

Parameters:
- dst_ub
- scalar_value
- count (optional, default = len(dst_ub))

Example:
    tl.duplicate(dst_ub, 0.0, count=len(dst_ub))


##### C. Copy-Out Block  
    with tl.copyout():
        tl.store(global_ptr + offsets, ub_buf)

Rules:
- All UB → global stores must be inside a `tl.copyout()` block

--------------------------------------------------------------------------------

================================================================================
### General Rules
- DSL is descriptive, not executable; natural-language comments are required
- Host handles global planning; Kernel handles on-chip execution
- All compute ops write to their output buffer; none return values
'''
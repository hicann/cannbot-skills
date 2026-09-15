---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "MicroScaling (MX) format — mxfp8 / mxfp4 cube path with fp8_e8m0_t shared scale [V351, microscaling]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=matmul-mx,attention-mx"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=matmul-mx,attention-mx"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-145
timestamp_inferred: true
tags: [fp8_e8m0_t, ascendc, ol-145]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=matmul-mx,attention-mx`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/datacopy/asc_copy_l12l0a_mx.md + asc_copy_l12l0b_mx.md; migration/220x到351x架构变更.md "扩展LoadData搬运指令"`

**Rule**: A5 cube path supports **MicroScaling (MX)** quantization — a 32-element block shares one `fp8_e8m0_t` exponent-only scale, applied at LoadData stage from L1 to L0A/L0B. The scale matrix has a FIXED fractal layout `(16, 2, 16)` and uses dedicated copy intrinsics. This is the foundation for mxfp8 (FP8 mantissa + e8m0 shared scale) and mxfp4 (FP4 mantissa + e8m0 shared scale) matmul kernels.

**Why this matters**: mxfp8 / mxfp4 are NOT just "smaller fp8 / fp4" — they're a separate data layout family. A naive port that treats them as element-wise fp8/fp4 will produce wrong arithmetic.

**Hardware path**: MX scale matrix lives in L1 Buffer alongside the data, gets shipped to L0A/L0B via specific intrinsics, and is consumed by Mmad with `LoadDataWithMxScaling` (extended LoadData) before multiply. From 220x→351x architecture changes: "扩展LoadData搬运指令。新增支持MicroScaling（MX）场景的数据搬运".

**Scale matrix intrinsics** (cube datamove, PIPE_MTE1):

```cpp
// L1 → L0A scale copy
__aicore__ inline void asc_copy_l12l0a_mx(
    uint64_t dst,                  // L0A address / 16 (32B-aligned)
    __cbuf__ fp8_e8m0_t* src,      // L1 source — fp8_e8m0_t scale
    uint16_t x_start_pos,          // M-axis start, unit = 1 fractal (32B)
    uint16_t y_start_pos,          // K-axis start, unit = 32B
    uint8_t  x_step,               // M-axis length, fractals [0..255]
    uint8_t  y_step,               // K-axis length, 32B units [0..255]
    uint16_t src_stride,           // L1 fractal stride (32B)
    uint16_t dst_stride);          // L0A fractal stride (32B)

// L1 → L0B scale copy (mirror signature)
__aicore__ inline void asc_copy_l12l0b_mx(...);

// Sync variants (block until done)
asc_copy_l12l0a_mx_sync(...);
asc_copy_l12l0b_mx_sync(...);
```

**Critical constraints**:
- Scale matrix fractal is **always (16, 2, 16)** — corresponds to data L0A/L0B fractal (16, 32, 16). Scale occupies **1/16th of L0A address space**.
- `dst` is the L0A/L0B address divided by 16 (not the raw pointer). Convert via `static_cast<uint64_t>(reinterpret_cast<uintptr_t>(dst)) / 16`.
- Allowed `dst` types: `fp4x2_e2m1_t`, `fp4x2_e1m2_t`, `fp8_e5m2_t`, `fp8_e4m3fn_t`. The scale (`src`) is always `fp8_e8m0_t`.
- Both src and dst must be **32-byte aligned**.

**Concrete launch sequence** (for an mxfp8 matmul):

```cpp
// Per-tile in cube kernel
__cbuf__ fp8_e8m0_t scale_a[1024];      // L1 scale matrix for A
__ca__   fp8_e5m2_t  data_a[1024];      // L0A data matrix
// ... corresponding scale_b / data_b for B
uint64_t mx_dst_a = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(data_a)) / 16;

// Stage scale into L0A
asc_copy_l12l0a_mx(mx_dst_a, scale_a,
    /*x_start*/0, /*y_start*/0,
    /*x_step*/0, /*y_step*/0,
    /*src_stride*/8, /*dst_stride*/8);

// Then standard Mmad — hardware applies scale during multiply
Mmad(...);
```

**Why `dst / 16`**: the scale matrix is 1/16 the size of the data matrix (one e8m0 per 32-element block). The hardware-internal pointer arithmetic indexes by fractal count, not bytes — dividing the raw L0A address by 16 gives the scale-fractal index.

**Per-block granularity**: e8m0 scale is applied per 32-element block. Workload designer chooses block boundaries; in mxfp8, every 32 consecutive elements of the K dimension share one scale.

**Evidence**:
- PR 103 imports both `asc_copy_l12l0a_mx.md` + `asc_copy_l12l0b_mx.md` (public API docs at `asc-devkit/docs/api/context/c_api/cube_datamove/`)
- Constraint section: "scale矩阵的分形固定为(16, 2, 16)，对应L0A Buffer的分形为(16, 32, 16)，占L0A Buffer地址的1 / 16"
- Pipe type: `PIPE_MTE1` (matches cube data-movement pipe)

**Implications for our op-gen**:
1. **Detection** — when porting / authoring a matmul-class op flagged as mxfp8 / mxfp4, the kernel MUST use `asc_copy_l12l0a_mx` / `asc_copy_l12l0b_mx` (NOT raw `DataCopy`). Single Cast at LoadData time is wrong.
2. **Tiling** — host tiling code must compute scale fractal counts and emit them alongside data fractal counts.
3. **dtype matrix in `_def.cpp`** — kernel registering for mxfp8 must declare both `DT_FLOAT8_E5M2` (or `DT_FLOAT8_E4M3FN`) for data AND `DT_FLOAT8_E8M0` for scale.
4. **No A3 backport** — these intrinsics are V351-only. mxfp8 / mxfp4 kernels have **no A3 reference** in `port_a3_to_a5` mode; they are greenfield A5 author.

**Other instances (predicted)**: future mxfp8 matmul, mxfp4 matmul, mxfp8 attention (FlashAttention with MX-quantized KV cache), MoE+MX-quant (`grouped_matmul_swiglu_quant` variant), KV-cache compression. Likely applies to `fused_quant_mat_mul` if we extend it to MX.

**Cross-reference**:
- OL-144 (narrow-float type family — supplies the mantissa types)
- OL-146 (CastTrait for FP32→FP8/FP4 — used during Cast before MX-scaled store)
- `kb/okf/runbooks/hardware/target-ascend950pr.md` — UB / L0A / L0B capacity affects MX block scheduling
- `cann/asc-devkit/docs/api/context/c_api/cube_datamove/asc_copy_l12l0[ab]_mx.md` — authoritative API doc

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-145（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->

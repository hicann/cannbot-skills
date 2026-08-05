# GMM SwiGLU Quant A8W8 — Host Tiling Advisory Cookbook

> **P-P104-HOST** v2.0 (2026-06-12) — companion to P-P104 v2.0 cookbook.
> `applies_to: soc=Ascend950PR/V351; cann=9.1.T500; op=grouped_matmul_swiglu_quant A8W8 fusion path`
> **HOW TO USE**: treat each block as an advisory checklist for fields, lifecycle, and failure probes.
> Re-derive task-owned host tiling from the selected arch22 contract and current arch35 public APIs;
> do not paste this snapshot into `pybind11.cpp` and declare generation success.

---

## ⛔ PRE-FLIGHT: pybind11 anti-pattern traps

### TRAP H1: ⛔ DO NOT `#include "kernel_tiling.h"` or any `matmul/` header.

These pull in `MatmulApiTiling` which requires KFC channels that don't exist under `ACLRT_LAUNCH_KERNEL`.

**Self-check**: `grep -n 'kernel_tiling\|matmul/' kernel/pybind11.cpp` must return ZERO.

### TRAP H2: ⛔ DO NOT use SDK `TCubeTiling` struct directly.

The `TCubeTiling` struct is SDK-internal and may not be available in NPUKernelBench's build environment. Use the manual `HostTCubeTiling` struct from §1 below (50 plain int32_t fields — identical binary layout).

### TRAP H3: ⛔ DO NOT compute workspace as `M*N*4 + 20MB` (V2 formula).

Use `16MB + M * N * sizeof(int32_t)` exactly.

---

## §1 — TilingData struct: derive and verify the byte contract

```cpp
// Advisory field inventory. Re-derive the task-owned struct and assert its byte contract.
// This replaces V2's GMMSwigluQuantV2TilingFusionData + TCubeTiling dependency.
// 50 int32_t fields = exact binary layout of SDK's TCubeTiling.

#pragma pack(push, 1)
struct HostTCubeTiling {
    // Fields 0-13: basic config
    int32_t usedCoreNum;    int32_t M;    int32_t N;    int32_t Ka;    int32_t Kb;
    int32_t singleCoreM;    int32_t singleCoreN;    int32_t singleCoreK;
    int32_t baseM;          int32_t baseN;          int32_t baseK;
    int32_t depthA1;        int32_t depthB1;
    int32_t stepM;          int32_t stepN;
    // Fields 14-27: advanced config
    int32_t stepKa;         int32_t stepKb;
    int32_t isBias;         int32_t iterateOrder;
    int32_t shareMode;      int32_t shareL1Size;     int32_t shareL0CSize;     int32_t shareUbSize;
    int32_t bufferDepthA1;  int32_t bufferDepthB1;
    int32_t reserve00;      int32_t reserve01;       int32_t reserve02;       int32_t reserve03;
    // Fields 28-41: fractal/cache config
    int32_t reserve04;      int32_t reserve05;       int32_t reserve06;       int32_t reserve07;
    int32_t reserve08;      int32_t reserve09;       int32_t reserve10;       int32_t reserve11;
    int32_t reserve12;      int32_t reserve13;       int32_t reserve14;       int32_t reserve15;
    // Fields 42-49: tail reserve — ZERO-INIT all of them
    int32_t reserve16;      int32_t reserve17;       int32_t reserve18;       int32_t reserve19;
    int32_t reserve20;      int32_t reserve21;       int32_t reserve22;       int32_t reserve23;
};

struct <OpName>TilingData {     // ← fill: e.g. GMMSwigluQuantTilingData
    int64_t cubeBlockDim;       // = aicCoreNum
    int64_t vectorBlockDim;     // = aivCoreNum
    int64_t groupNum;
    int64_t K;
    int64_t N;
    int64_t M;
    int64_t ubFactorDimx;       // UB row batch factor: CalcUBFactorDimX(N) → {1,2,4}
    int64_t ubFactorDimy;       // = N / 2  (SPILI_NUM=2)
    int64_t actRight;           // = N / 2  (activation right-half offset)
    int64_t groupListType;      // 0=prefix-sum, 1=absolute
    int8_t  isSingleTensor;     // 1=single 5D NZ weight, 0=multi 4D per-weight NZ
    int8_t  rsvd[7];
    HostTCubeTiling mmTilingData;
};
#pragma pack(pop)
```

---

## §2 — Constants: derive from the selected contract and validate

```cpp
// ⛔ These are HARDCODED. Do NOT make configurable, do NOT read from env.
static constexpr size_t SYS_WORKSPACE_SIZE = 16 * 1024 * 1024;  // 16MB front reserve (Rule 0.3)
static constexpr int64_t BASE_M = 128;
static constexpr int64_t BASE_N = 256;
static constexpr int64_t BASE_K = 128;
static constexpr int64_t INT32_SIZE = 4;
```

---

## §3 — ExtractConfig: advisory lifecycle and weight-format checklist

```cpp
static void ExtractConfig(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& groupList,
    int64_t& M, int64_t& K, int64_t& N, int64_t& groupNum,
    int64_t& groupListType, int8_t& isSingleTensor,
    int64_t& ubFactorDimx,
    int64_t& aicCoreNum, int64_t& aivCoreNum)
{
    M = x.size(0);
    K = x.size(1);
    groupNum = groupList.size(0);
    groupListType = 1;  // default: absolute counts per group

    // Weight format detection — derive from the selected-source schema and validate each branch.
    int wDim = weight.dim();
    if (wDim == 5) {
        N = weight.size(1) * weight.size(4);  // 5D NZ: [E, N/32, K/16, 16, 32]
        isSingleTensor = 1;
    } else if (wDim == 4) {
        N = weight.size(0) * weight.size(3);  // 4D multi-NZ: [N/32, K/16, 16, 32]
        isSingleTensor = 0;
    } else {
        N = weight.size(1);                   // ND fallback
        isSingleTensor = 1;
    }

    ubFactorDimx = CalcUBFactorDimX(N);

    // PlatformInfo: query at runtime — do NOT hardcode
    // ⛔ Use torch_npu to query, or read from env/config
    aicCoreNum = 20;   // V351 default: 20 AIC cores
    aivCoreNum = 40;   // V351 default: 40 AIV cores (2× AIC for MIX_AIC_1_2)
}

static int32_t CalcUBFactorDimX(int64_t N) {
    // Advisory historical thresholds; select and validate task-owned thresholds.
    if (N < 4600)       return 4;
    if (N < 8192)       return 2;
    return 1;
}
```

---

## §4 — ComputeTiling: task-owned derivation checklist

```cpp
static <OpName>TilingData ComputeTiling(     // ← fill struct name
    int64_t M, int64_t K, int64_t N, int64_t groupNum,
    int64_t groupListType, int8_t isSingleTensor,
    int64_t ubFactorDimx, int64_t aicCoreNum, int64_t aivCoreNum)
{
    <OpName>TilingData td;                   // ← fill struct name
    memset(&td, 0, sizeof(td));

    // --- Section 1: fusion tiling fields ---
    td.cubeBlockDim   = aicCoreNum;
    td.vectorBlockDim = aivCoreNum;
    td.groupNum       = groupNum;
    td.K = K;  td.N = N;  td.M = M;
    td.ubFactorDimx   = ubFactorDimx;
    td.ubFactorDimy   = N / 2;    // SPILI_NUM = 2
    td.actRight       = N / 2;
    td.groupListType  = groupListType;
    td.isSingleTensor = isSingleTensor;

    // --- Section 2: HostTCubeTiling fields ---
    // ⛔ DO NOT call MatmulApiTiling. Fill fields directly.
    // ⛔ DO NOT #include kernel_tiling.h.
    auto& mm = td.mmTilingData;
    mm.usedCoreNum  = (int32_t)aicCoreNum;
    mm.M  = (int32_t)M;   mm.N  = (int32_t)N;
    mm.Ka = (int32_t)K;   mm.Kb = (int32_t)K;
    mm.singleCoreM = (int32_t)BASE_M;  mm.singleCoreN = (int32_t)BASE_N;
    mm.singleCoreK = (int32_t)BASE_K;
    mm.baseM = (int32_t)BASE_M;  mm.baseN = (int32_t)BASE_N;  mm.baseK = (int32_t)BASE_K;
    mm.depthA1 = 8;  mm.depthB1 = 8;
    mm.stepM = 1;  mm.stepN = 1;
    mm.stepKa = 4;  mm.stepKb = 4;
    mm.isBias = 0;
    mm.iterateOrder = 0;
    mm.shareMode = 0;

    return td;
}
```

---

## §5 — Launch kernel: author from the task-owned entry contract

```cpp
static void Launch<OpName>(                    // ← fill op name
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& weightScale,
    const torch::Tensor& xScale,
    const torch::Tensor& groupList,
    torch::Tensor& y,
    torch::Tensor& yScale)
{
    int64_t M, K, N, groupNum, groupListType, ubFactorDimx, aicCoreNum, aivCoreNum;
    int8_t isSingleTensor;
    ExtractConfig(x, weight, groupList,
                  M, K, N, groupNum, groupListType, isSingleTensor,
                  ubFactorDimx, aicCoreNum, aivCoreNum);

    auto td = ComputeTiling(M, K, N, groupNum, groupListType, isSingleTensor,
                            ubFactorDimx, aicCoreNum, aivCoreNum);

    // ⛔ Rule 0.3: 16MB front reserve + M*N*sizeof(int32_t)
    // ⛔ NOT V2 formula M*N*4 + 20MB
    size_t userDataSize = (size_t)M * (size_t)N * INT32_SIZE;
    size_t totalWorkspace = SYS_WORKSPACE_SIZE + userDataSize;

    auto wsTensor = torch::empty(
        {(int64_t)totalWorkspace},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kPrivateUse1));
    void* userWs = (char*)wsTensor.data_ptr() + SYS_WORKSPACE_SIZE;

    // Serialize tiling data to pass via GM
    auto tilingTensor = torch::from_blob(
        &td, {(int64_t)sizeof(td)},
        torch::TensorOptions().dtype(torch::kInt8).device(torch::kPrivateUse1));

    // ⛔ ACLRT_LAUNCH_KERNEL — the ONLY launch mechanism for standalone
    // ⛔ DO NOT use aclnn, aclop, or GE graph launch
    ACLRT_LAUNCH_KERNEL(<op_name>_kernel)(       // ← fill kernel function name
        aicCoreNum, nullptr,                     // blockDim, stream
        x.data_ptr(), weight.data_ptr(),
        weightScale.data_ptr(), xScale.data_ptr(),
        groupList.data_ptr(),
        y.data_ptr(), yScale.data_ptr(),
        userWs, tilingTensor.data_ptr());

    torch_npu::npu_synchronize();
}
```

---

## §6 — pybind11 module registration: author from the declared schema

```cpp
// ⛔ Keep this structure. Only change function names and tensor counts.
TORCH_LIBRARY(<op_name>, m) {                   // ← fill op name
    m.def("<op_name>_kernel", &Launch<OpName>); // ← fill names
}
```

---

## §7 — Host-side anti-pattern checklist (run BEFORE Phase C build)

```bash
# Check H1: NO KFC-dependent headers
grep -n 'kernel_tiling.h\|matmul_api.h\|matmul/' kernel/pybind11.cpp
# Must return ZERO.

# Check H2: NO SDK TCubeTiling usage (we use HostTCubeTiling)
grep -n 'TCubeTiling[^D]' kernel/pybind11.cpp
# Must return ZERO (except HostTCubeTiling).

# Check H3: Workspace formula correct
grep -n '20\s*\*\s*1024\s*\*\s*1024\|20MB\|20\s*\*\s*MB' kernel/pybind11.cpp
# Must return ZERO. V2's 20MB padding is WRONG for standalone.

# Check H4: 16MB front reserve present
grep -n '16\s*\*\s*1024\s*\*\s*1024' kernel/pybind11.cpp
# Must return AT LEAST 1 match.

# Check H5: NO torch:: / at:: computation in pybind
grep -n 'torch::sum\|torch::mean\|torch::matmul\|torch::exp\|at::' kernel/pybind11.cpp
# Must return ZERO. Computation is in kernel, not pybind.
```

---

## Appendix A — Host-side V220→V300 substitution table

| V220 pattern (found in op_host/) | V300 replacement (write this) | Reason |
|---|---|---|
| `#include "kernel_tiling.h"` | DELETE — use `HostTCubeTiling` struct | No KFC |
| `#include "matmul/..."` | DELETE | No Matmul library |
| `MatmulApiTiling mmTiling;` | `HostTCubeTiling mmTiling;` | SDK struct unavailable |
| `mmTiling.SetBaseM(...)` | `mmTiling.baseM = (int32_t)BASE_M;` | No setters |
| `workspace = M*N*4 + 20*1024*1024` | `workspace = SYS_WORKSPACE_SIZE + M*N*INT32_SIZE` | 16MB, not 20MB |
| `TILING_KEY_IS(2)` or `A8W4_MSD_...` | DELETE — use `A8W8_FUSION_KEY_MODE = 3` | V1 split dead weight |
| `PlatformInfo::GetCoreNum()` | Query via torch_npu or config, or use V351 defaults (20/40) | Standalone launch |
| `opApi::` or `OpApi` calls | DELETE — pure ACLRT_LAUNCH_KERNEL | No CANN framework APIs |

## Appendix B — W4 minimal host config

For Phase B initial build, use these hardcoded values:
- aicCoreNum = 20, aivCoreNum = 40 (V351 MIX_AIC_1_2)
- M = 128, K = 256, N = 128, groupNum = 1, groupListType = 1 (absolute)
- ubFactorDimx = 4, ubFactorDimy = N/2, actRight = N/2

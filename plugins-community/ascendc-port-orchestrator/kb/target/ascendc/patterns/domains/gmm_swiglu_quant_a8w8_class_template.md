# GMM SwiGLU Quant A8W8 — COOKBOOK (mechanical recipe, not reference)

> **P-P104** v2.1 (2026-06-16) — v2.0 + Appendix C (workspace 16MB reserve from v2 source + FA host tiling reference).
> `applies_to: soc=Ascend950PR/V351; cann=9.1.T500; op_class=grouped-matmul-swiglu-quant / CUBE_MIX (gmm family, A8W8 path)`
> **HOW TO USE**: Read step N → write the code shown → move to step N+1. Do NOT interpret, do NOT optimize, do NOT skip.
>
> Grey-box agent assembles from: (1) arch22 (V220) source + (2) THIS COOKBOOK. MUST NOT peek at arch35 reference.

---

## ⛔ PRE-FLIGHT: Three fatal traps — check BEFORE writing any code

### TRAP 1: Matmul<> = KFC deadlock in standalone pybind11 mode — use hand-rolled Mmad instead. ⛔ DO NOT USE `Matmul<>` / `MatmulImpl<>` / `MMImplType` / `matmul::` ANYWHERE in a pybind11 kernel.

**Context (v2 sibling clarification)**: `MatmulImpl<>` IS technically functional on A5/V351 hardware — the v2 sibling (grouped_matmul_swiglu_quant_v2) proved it, achieving 19/19 PASS_WITHIN_TOLERANCE using `MatmulImpl<MM_CFG=CFG_NORM>` with proper framework bootstrap. The restriction is NOT about A5 hardware capability — it's about the **deployment mode**. `MatmulImpl<>` internally depends on KFC (Kernel Fusion Controller) for cross-core sync and workspace management. KFC requires the CANN operator framework's workspace bootstrap (`SetSysWorkspaceForce` + auto_gen `WORKSPACE_PARAM_OFFSET` + `KfcCommServer::Init()`). In standalone pybind11 mode (`ACLRT_LAUNCH_KERNEL`, no operator framework), KFC cannot bootstrap → AIC-side KFC server never processes AIV requests → AIV hangs → `aicore timeout 507014`. **port_a3 mode produces standalone pybind11 kernels, so the MatmulImpl path is structurally unavailable.**

**What you write instead**: hand-rolled `AscendC::Mmad` calls. See Step 3 for exact code.

**Self-check after Phase B**: `grep -n 'Matmul[^T]\|matmul::\|MatmulImpl\|MMImplType' kernel/*.cpp kernel/*.h` must return ZERO lines. If it returns anything, you used the wrong API — rewrite.

### TRAP 2: Flag 8/9/10 = V351 barrier-reserved. ⛔ DO NOT USE flagId 8, 9, or 10.

**What you write instead**: flagId 7 (AIC→AIV signal), flagId 6 (AIV→AIC backpressure). See Step 4.

**Self-check**: `grep -n 'WaitFlag(0x8)\|WaitFlag(0x9)\|WaitFlag(8)\|WaitFlag(9)\|SetFlag.*(0x8)\|SetFlag.*(0x9)\|SetFlag.*(8)\|SetFlag.*(9)' kernel/*.cpp kernel/*.h` must return ZERO.

### TRAP 3: Workspace needs 16MB front reserve. ⛔ DO NOT use V2 formula `M*N*4 + 20MB`.

**What you write instead**: `WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024; total = WORK_SPACE_RESERVE_SIZE + M * N * sizeof(int32_t);`

---

## STEP 1 — File skeleton: emit exactly these 5 files

| # | File | Role |
|---|---|---|
| 1 | `kernel/<op>_kernels.cpp` | Build TU: X-macro variant list + extern-C kernel entries |
| 2 | `kernel/<op>_kernel.h` | Orchestrator: MIX_AIC_1_2 Process dispatch + Cube/Vec class includes |
| 3 | `kernel/<op>_cube.h` | Cube class: hand-rolled Mmad (Step 3), flag=7/6 (Step 4) |
| 4 | `kernel/<op>_vec.h` | Vec class: ProcessDSQ body (Step 5), flag sync (Step 4) |
| 5 | `kernel/pybind11.cpp` | ACLRT_LAUNCH_KERNEL shim + host tiling struct + ComputeTiling (Step 6) |

**Emit them in this order**: `_cube.h` → `_vec.h` → `_kernel.h` → `_kernels.cpp` → `pybind11.cpp`.

---

## STEP 2 — W4 minimal config (the ONLY config to implement)

Start with exactly ONE variant. Do NOT implement all 12 until this one compiles AND runs.

```
dequantDtype = 1   (fp16/half)
wFormat      = NZ  (fractal NZ layout)
transB       = false
```

**Hard values for Phase B**:
```
baseM = 128, baseN = 256, baseK = 128
M     = 128 (single group fits in one baseM)
N     = 256
K     = 128
cubeBlockDim  = aicCoreNum (runtime: 20 for V351)
vectorBlockDim = aivCoreNum (runtime: 40 for V351)
```

---

## STEP 3 — Cube class: hand-rolled Mmad (copy this skeleton, fill `<...>`)

```cpp
// kernel/<op>_cube.h
#pragma once
#include "kernel_operator.h"

using namespace AscendC;

template <typename DTYPE_X, typename DTYPE_WEIGHT, bool TRANS_B>
class CubeProcess {
  static constexpr int32_t BASE_M  = 128;
  static constexpr int32_t BASE_N  = 256;
  static constexpr int32_t BASE_K  = 128;
  static constexpr int32_t C0      = 16;  // C0=16 for fp16/bf16; 8 for int8
  static constexpr int32_t C0_HALF = 16;

  // --- Mmad-based matmul (NOT Matmul<>) ---
  __aicore__ void MatmulAccumulate(
      LocalTensor<int32_t>& l0c,
      const LocalTensor<DTYPE_X>& l0a,
      const LocalTensor<DTYPE_WEIGHT>& l0b,
      int32_t realM, int32_t realN, int32_t baseK, bool isFirstKTile)
  {
      // ⛔ DO NOT call Matmul<> / MatmulImpl<> / MMImplType here.
      // ⛔ DO NOT #include "matmul/matmul_api.h" or any matmul/ header.
      MmadParams mp;
      mp.M = realM;
      mp.N = realN;
      mp.K = baseK;
      mp.cmatrixInitVal = isFirstKTile;  // true for ki==0, false for ki>0
      mp.cmatrixSource  = false;          // accumulate into L0C
      Mmad(l0c, l0a, l0b, l0c, mp);
  }

  // --- K-tile loop: CeilDiv(K, BASE_K) iterations ---
  __aicore__ void ProcessCubeBlock(
      int32_t basicBlockIdx,
      /* ... per-block params from host tiling ... */)
  {
      int32_t kTiles = (K + BASE_K - 1) / BASE_K;
      for (int32_t ki = 0; ki < kTiles; ki++) {
          int32_t kOffset = ki * BASE_K;
          int32_t kSlice  = (ki == kTiles - 1) ? (K - kOffset) : BASE_K;
          // Load L1→L0A, L1→L0B (Nz fractal)
          LoadNzL1ToZnL0A(/* l0a, l1_x, kOffset, ... */);
          LoadNzL1ToZnL0B(/* l0b, l1_w, kOffset, ... */);
          SetWaitFlag<HardEvent::FIX_MTE2>();  // wait L0 loads
          MatmulAccumulate(/* l0c, l0a, l0b, realM, realN, kSlice, ki==0 */);
      }
      // Fixpipe L0C (int32) → workspace (ND)
      FixpipeParamsC310 fp(realM, realN);
      FixpipeC310(workspaceGm_[wsOffset], l0c, fp);
  }
};
```

**⛔ ANTI-PATTERN GUARD**: If your cube header contains `#include "matmul/` or `MatmulType<` or `Matmul<` or `MatmulImpl<`, delete the file and restart Step 3. You are using the deadlock path.

---

## STEP 4 — Flag sync: copy these EXACT 4 lines into your code

Where the V220 reference uses `CrossCoreSetFlag<X, PIPE>(0x8)` or `CrossCoreWaitFlag(0x8)`:

```cpp
// ⛔ DELETE all V220 flag lines first, then paste these EXACT replacements:

// In CubeProcess(), after each Round's Fixpipe:
CrossCoreSetFlag<0x2, PIPE_FIX>(7);     // signal AIV: Cube Round done

// In CubeProcess(), backpressure every 14 Rounds:
CrossCoreWaitFlag(6);                    // wait for AIV backpressure

// In VectorProcess(), after each Round's ProcessDSQ:
CrossCoreSetFlag<0x2, PIPE_MTE3>(6);    // backpressure to AIC

// In VectorProcess(), at start of ProcessDSQ:
CrossCoreWaitFlag(7);                    // wait for Cube this Round
```

**Values are HARDCODED — do NOT parameterize, do NOT make configurable:**
- MODE: always `0x2` (MIX_AIC_1_2)
- flagId: always `7` (AIC→AIV) and `6` (AIV→AIC)
- PIPE for SetFlag in Cube: `PIPE_FIX`
- PIPE for SetFlag in Vec: `PIPE_MTE3`
- Backpressure depth: `14` Rounds

---

## STEP 5 — ProcessDSQ body (Vec class)

This is the fused Dequant-SwiGLU-Quant body. The computation is unchanged from V220 — only the sync primitives change (Step 4).

```
ProcessDSQ(groupId, globalOffset, calcCount, isSyncAll):

  // --- (A) sync: wait for Cube THIS Round ---
  CrossCoreWaitFlag(7)     // ← was 0x8 in V220
  SyncAll<true>()

  // --- (B) dequant: x_deq = (x_int8 * x_scale) ---
  //   For each token in group: cast int8→fp16, Muls(x_int8, x_scale[batch]) → x_deq
  //   (V220 logic unchanged — copy the dequant loop from V220 source)

  // --- (C) gate-up projection ---
  //   Matmul: x_deq[*, 0:K/2] × weight_gate[K/2, K] → gate_pre_act
  //   (V220 logic unchanged — but uses Mmad, NOT Matmul<>)

  // --- (D) SwiGLU: SiLU(gate) ⊙ up ---
  //   up_result   = matmul(x_deq[*, K/2:K], weight_up)
  //   gate_result = SiLU(gate_pre_act)   // manual: Sigmoid → Mul
  //   activated   = Mul(gate_result, up_result)
  //   (V220 logic unchanged)

  // --- (E) quant: activated → int8 ---
  //   quant_scale = 127.0f / WholeReduceMax(|activated|)
  //   ⛔ USE: Muls(activated, 1.0f/127.0f * quant_scale)  ← multiply by reciprocal
  //   ⛔ NEVER: Div(activated, 127.0f)                     ← division path
  //   (CAND-PP103: quant uses 1/127 multiply, not divide)

  // --- (F) cross-round sync ---
  //   every 14 Rounds: SyncAll<true>() + CrossCoreSetFlag<0x2, PIPE_MTE3>(6)
```

---

## STEP 6 — Host tiling (pybind11.cpp): copy this EXACT skeleton

### 6.1 TilingData struct (copy verbatim, fill the `<...>` fields)

```cpp
// P-P104-HOST advisory field inventory — re-derive and verify; do not paste as generated code.
// ⛔ TCubeTiling is SDK-internal. Use this HostTCubeTiling struct instead.
// ⛔ DO NOT #include "kernel_tiling.h" — it pulls MatmulApiTiling which needs KFC.

#pragma pack(push, 1)
struct HostTCubeTiling {
    // 50 int32_t fields — exact binary layout of SDK's TCubeTiling.
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
    // Fields 42-49: tail reserve
    int32_t reserve16;      int32_t reserve17;       int32_t reserve18;       int32_t reserve19;
    int32_t reserve20;      int32_t reserve21;       int32_t reserve22;       int32_t reserve23;
};

struct GMMSwigluQuantTilingData {
    int64_t cubeBlockDim;
    int64_t vectorBlockDim;
    int64_t groupNum;
    int64_t K;
    int64_t N;
    int64_t M;
    int64_t ubFactorDimx;
    int64_t ubFactorDimy;
    int64_t actRight;
    int64_t groupListType;
    int8_t  isSingleTensor;
    int8_t  rsvd[7];
    HostTCubeTiling mmTilingData;
};
#pragma pack(pop)
```

### 6.2 Workspace formula (copy verbatim)

```cpp
static constexpr size_t WORK_SPACE_RESERVE_SIZE = 16 * 1024 * 1024;  // 16 MB front reserve

size_t CalcWorkspaceSize(int64_t M, int64_t N) {
    // ⛔ DO NOT use V2 formula M*N*4 + 20MB.
    // ⛔ DO NOT call into ACLNN for workspace query.
    size_t userData = (size_t)(M * N) * sizeof(int32_t);
    return WORK_SPACE_RESERVE_SIZE + userData;
}
// Layout: [16MB front reserve | user-data (M*N*int32_t)]
// At kernel entry: uint8_t* userPtr = (uint8_t*)workspace + WORK_SPACE_RESERVE_SIZE;
```

### 6.3 ComputeTiling (follow this EXACT sequence)

```cpp
static GMMSwigluQuantTilingData ComputeTiling(
    int64_t M, int64_t K, int64_t N,
    int64_t groupNum, int64_t groupListType, int8_t isSingleTensor,
    int64_t aicCoreNum, int64_t aivCoreNum)
{
    GMMSwigluQuantTilingData td;
    memset(&td, 0, sizeof(td));

    // Step 1: copy shape fields
    td.M = M;  td.K = K;  td.N = N;
    td.groupNum = groupNum;
    td.groupListType = groupListType;
    td.isSingleTensor = isSingleTensor;

    // Step 2: compute derived fields
    td.ubFactorDimx = CalcUBFactorDimX(N);
    td.ubFactorDimy = N / 2;
    td.actRight     = N / 2;

    // Step 3: block dims
    td.cubeBlockDim   = aicCoreNum;
    td.vectorBlockDim = aivCoreNum;

    // Step 4: fill TCubeTiling (Mmad hand-rolled, NOT via MatmulApiTiling)
    // ⛔ DO NOT call MatmulApiTiling or #include "kernel_tiling.h"
    td.mmTilingData.usedCoreNum  = (int32_t)aicCoreNum;
    td.mmTilingData.M            = (int32_t)M;
    td.mmTilingData.N            = (int32_t)N;
    td.mmTilingData.Ka           = (int32_t)K;
    td.mmTilingData.Kb           = (int32_t)K;
    td.mmTilingData.singleCoreM  = 128;
    td.mmTilingData.singleCoreN  = 256;
    td.mmTilingData.singleCoreK  = 128;
    td.mmTilingData.baseM        = 128;
    td.mmTilingData.baseN        = 256;
    td.mmTilingData.baseK        = 128;

    return td;
}

static int32_t CalcUBFactorDimX(int64_t N) {
    // ⛔ Copy exactly, do not modify thresholds
    if (N < 4600)       return 4;
    if (N < 8192)       return 2;
    return 1;
}
```

---

## STEP 7 — X-macro variant dispatch (_kernels.cpp)

### 7.1 W4 minimal: ONE variant only

```cpp
// kernel/<op>_kernels.cpp
// W4 minimal — ONE variant. Expand to 12 after this compiles + runs.
#include "<op>_kernel.h"

// --- single W4 variant ---
#define INSTANTIATE_GMM_SWIGLU_QUANT(dtype, wfmt, tb) \
  extern "C" __global__ __aicore__ void \
  gmm_swiglu_quant_##dtype##_##wfmt##_tb##tb( \
      GmInt8* x, GmInt8* weight, GmFloat* weight_scale, GmFloat* x_scale, \
      GmInt64* group_list, GmInt8* y, GmFloat* y_scale, \
      GmUint8* workspace, GmUint8* tilingGm) { \
    using CubeT = CubeProcess<DTYPE_##dtype, DTYPE_##dtype, tb>; \
    using VecT  = VecProcess<DTYPE_##dtype, tb>; \
    KernelEntry<CubeT, VecT, DTYPE_##dtype, tb>( \
        x, weight, weight_scale, x_scale, group_list, y, y_scale, workspace, tilingGm); \
  }

// ⛔ W4: compile exactly ONE. The dtype/format/tb values come from Step 2.
// ⛔ Expand to 12 after this variant passes Phase D precision.
```

### 7.2 Full variant space (implement AFTER W4 passes)

```
{dtype: half/bf16/fp32} × {wFormat: NZ/ND} × {transB: false/true} = 12 variants
```

The macro body is the SAME for all 12 — only dtype, wFormat, transB differ.

---

## STEP 8 — Build TU anti-pattern checklist (run BEFORE Phase C)

These are fatal — if any check fails, fix before building:

```bash
# Check 1: NO Matmul<> deadlock path
grep -rn 'Matmul[^T]\|matmul::\|MatmulImpl\|MMImplType' kernel/*.cpp kernel/*.h
# Must return ZERO. Non-zero = you used the wrong API.

# Check 2: NO forbidden flag IDs
grep -rn 'WaitFlag(0x8)\|WaitFlag(0x9)\|SetFlag.*(0x8)\|SetFlag.*(0x9)' kernel/*.cpp kernel/*.h
# Must return ZERO. Non-zero = V351 barrier collision.

# Check 3: NO arch35 include wrapping
grep -rn '#include.*arch35' kernel/*.cpp kernel/pybind11.cpp
# Must return ZERO.

# Check 4: REQUIRED patterns present
grep -rn 'CrossCoreSetFlag.*(7)\|CrossCoreWaitFlag(7)' kernel/*.cpp kernel/*.h
# Must return AT LEAST 2 matches (one in cube, one in vec).

# Check 5: NO KFC-dependent headers
grep -rn 'kernel_tiling.h\|matmul_api.h\|matmul/' kernel/*.cpp kernel/*.h kernel/pybind11.cpp
# Must return ZERO except possibly in prestaged V220 headers (which are read-only).
```

---

## STEP 9 — Minimal verification pass (Phase D)

After build succeeds with the W4 single variant:

1. Run `edge_runner.py` for the W4 config only (M=128, K=256, N=128, single group)
2. Verify PASS on fp16 + NZ + transB=false
3. Only AFTER this passes: expand to full 12-variant X-macro
4. Run full edge_runner.py for all cases

---

## APPENDIX A — V220→V300 mechanical substitution table

When reading V220 source, apply these substitutions MECHANICALLY:

| V220 pattern (found in prestaged source) | V300 replacement (write this) | Reason |
|---|---|---|
| `Matmul<...>(l0c, l0a, l0b, ...)` | `Mmad(l0c, l0a, l0b, l0c, mp)` with `MmadParams` | Rule 0.1: KFC deadlock |
| `#include "matmul/..."` | DELETE the include | No matmul headers needed |
| `#include "kernel_tiling.h"` | DELETE the include | Pulls MatmulApiTiling |
| `MatmulApiTiling(...)` | Direct field assignment: `td.baseM=128; td.baseN=256; td.baseK=128;` | No KFC bootstrap |
| `CrossCoreSetFlag<..., PIPE_FIX>(0x8)` | `CrossCoreSetFlag<0x2, PIPE_FIX>(7)` | Rule 0.2 |
| `CrossCoreSetFlag<..., PIPE_MTE2>(0x9)` | `CrossCoreSetFlag<0x2, PIPE_MTE3>(6)` | Rule 0.2 |
| `CrossCoreWaitFlag(0x8)` | `CrossCoreWaitFlag(7)` | Rule 0.2 |
| `CrossCoreWaitFlag(0x9)` | `CrossCoreWaitFlag(6)` | Rule 0.2 |
| `workspace = M*N*4 + 20*1024*1024` | `workspace = 16*1024*1024 + M*N*sizeof(int32_t)` | Rule 0.3 |
| `__CCE_AICORE__ == 220` | `__CCE_AICORE__ >= 351` or DELETE (arch35 always) | Arch guard |
| `TCubeTiling` (SDK struct) | `HostTCubeTiling` (manual 50-int32_t struct from Step 6) | SDK struct unavailable in NPUKernelBench |
| `Div(x, 127.0f)` (quant) | `Muls(x, 1.0f/127.0f)` (multiply by reciprocal) | CAND-PP103 |
| `WholeReduceMax(x, ...)` (fp32 mask) | `WholeReduceMax(x, ...)` with mask≤64 | CAND-V351-AIV-... |
| `SwiGLU(x)` (high-level API) | Manual 5-op: Sigmoid→Mul→Mul (canonical SiLU⊙Gate) | No high-level SwiGLU in standalone |

---

## APPENDIX B — FA class worked example (read for Mmad/flag/workspace reference)

These files in the FA archive show the same Rules applied correctly:
- `patterns/domains/fa_class/templates/op_kernel/flash_attention_score_cube.h` — 9 Mmad calls, no Matmul<>
- `output/a3_to_a5_port/flash_attention_score/src/kernels/flash_attention_score/kernel/` — full working kernel

Cross-ref: P-P102 (cube-MIX scaffold), P-P70 (fused dequant→activation→quant pipeline),
P-P104-HOST (host tiling companion template), cube_vector_fusion.md (general CUBE_MIX patterns).

---

## APPENDIX C — Workspace & Tiling constants (extracted from v2 & FA host tiling)

### C.1 Workspace formula (COPY VERBATIM)

From FA `wp_fa_host_tiling.h:364-371` + v2 `grouped_matmul_swiglu_quant_v2_tiling.h:132`:

```cpp
// ⛔ DO NOT use V2 formula M*N*4 + 20MB.
// SYS_WORKSPACE_SIZE = exactly 16MB front reserve (CANN host convention).
static constexpr uint32_t SYS_WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);

// Layout: [16MB front reserve | user-data]
// GetUserWorkspace() returns base + 16MB → user-data starts here.
// At kernel entry: uint8_t* userPtr = (uint8_t*)workspace + SYS_WORKSPACE_SIZE;

// User data size = workSpaceMTemp * N * elementSize * doubleBufferFactor
//   workSpaceMTemp = mLimit (per-core row count)
//   elementSize    = sizeof(half)=2 (fp16 path) or sizeof(int32_t)=4 (int32 path)
//   Refer to v2: grouped_matmul_swiglu_quant_v2_base_tiling.cpp:389-403
size_t CalcWorkspaceSize(int64_t mLimit, int64_t N, int elementBytes) {
    size_t userData = (size_t)(mLimit * N) * elementBytes * 2;  // double buffer
    return SYS_WORKSPACE_SIZE + userData;
}
```

### C.2 Base tiling constants (from v2 source)

```cpp
// From grouped_matmul_swiglu_quant_v2_tiling.h
constexpr int64_t A8W4_BASEM = 128;
constexpr int64_t A8W4_BASEK = 256;
constexpr int64_t A8W4_BASEN = 256;
constexpr int64_t SIZE_OF_HALF_2 = 2;
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t SWIGLU_REDUCE_FACTOR = 2;
constexpr int64_t INT32_DTYPE_SIZE = 4;
constexpr int64_t USER_WORKSPACE_LIMIT = static_cast<int64_t>(64 * 1024 * 1024);
```

### C.3 TCubeTiling field mapping (Step 6 struct → SDK TCubeTiling)

```
HostTCubeTiling field     SDK TCubeTiling field    Value (from v2)
─────────────────────────────────────────────────────────────────
usedCoreNum               usedCoreNum              aicCoreNum (20 for V351)
M                         M                        M (total rows per group)
N                         N                        N (total columns)
Ka                        Ka                       K (activation dim)
Kb                        Kb                       K (weight dim)
singleCoreM               singleCoreM              128 (= A8W4_BASEM)
singleCoreN               singleCoreN              256 (= A8W4_BASEN)
singleCoreK               singleCoreK              256 (= A8W4_BASEK)
baseM                     baseM                    128
baseN                     baseN                    256
baseK                     baseK                    256
```

### C.4 Edge case constants (probe-verified)

```cpp
// Small-N boundary handling (for N=32 edge case, probe 7-iter verified):
// - When N <= 64: chunk_rows = 1 (scalar per-row, avoid VEC→S gap)
// - When N > 64:  chunk_rows = 4 (vectorized)
// WholeReduceMax mask: ≤64 for fp32 (CAND-V351-AIV-WholeReduceMax)

// Buffer sizing — use runtime N, not compile-time constant:
//   chunk_rows  = (N <= 64) ? 1 : 4;
//   num_chunks  = (GmmBaseParams::MAX_CHUNK_ROWS + chunk_rows - 1) / chunk_rows;
//   buffer_elts = chunk_rows * N;  // per-chunk element count
```

---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 arch35 service-class structure (cube_block + vector_block + vf/* MicroAPI vector fission) for backward op-family port targets"
description: "applies_to: soc=Ascend950PR_9579 (V351/arch35); cann=9.0.0; bisheng=15.0.5; op_class=backward (gradient) multi-stage cube+vec non-FA-class (no online-softmax tile-scheduling) verified_on: sparse_light"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR_9579 (V351/arch35); cann=9.0.0; bisheng=15.0.5; op_class=backward (gradient) multi-stage cube+vec | non-FA-class (no online-softmax"
confidence: inferred
status: stub
original_id: CAND-V351-arch35-RegBase-service-class-skeleton
timestamp_inferred: true
tags: [candidate, inferred, ascend_is_aic, ascend_is_aiv, templates_def, processvector0, processvectorn, cand-v351-arch35-regbase-service-class-skeleton]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR_9579 (V351/arch35); cann=9.0.0; bisheng=15.0.5; op_class=backward (gradient) multi-stage cube+vec | non-FA-class (no online-softmax tile-scheduling)`
`verified_on: sparse_lightning_indexer_grad_kl_loss arch35 service-class structure (CANN reference 2026-05-24, 3758 LOC read; NOT yet shipped on ours; complementary evidence layer to CAND-V220-to-V351-PortPattern-CubeVecFusedOp which covered sync-mode delta)`
`unverified_on: kw runtime ship using these service-class skeletons; V220 (V220 LIG_grad uses different vec primitives — V351 vf/ MicroAPI is arch35-only)`

**Companion to CAND-V220-to-V351-PortPattern-CubeVecFusedOp**: that entry covers V220→V351 **sync-mode delta** (mode2→mode4 + outer-only KFC-internal); THIS entry covers V351 arch35 **service-class skeleton** (how cube_block + vector_block + vf/* fit together inside the V351 op kernel TU). Together = sync + structure = full V220→V351 port template.

### Principle: V351 arch35 op kernel splits along 4 axes
1. **Top-level kernel class**: `KernelBase<CubeBlockType, VecBlockType>` template with `ASCEND_IS_AIC`/`ASCEND_IS_AIV` branches in `Init()` and `Process*()` methods. Owns shared TPipe, BufferManager<UB/L1>, ConstInfo/RunInfo structs.
2. **Cube block** (header `*_cube_block.h`): manages L0A/L0B/L0C buffer pool + matmul invocation + AIC sync. Template params via macro-generated `TEMPLATES_DEF`.
3. **Vector block** (header `*_vector_block.h`): N-stage AIV pipeline (`ProcessVector0` ... `ProcessVectorN`); each stage is one vf-process function. Manages UB buffer pool + Vec sync.
4. **Vector fission `vf/*.h`** subdir: each vec primitive (Cast/Add/Mul/Nd2Nz format conversion/etc.) is one `__simd_vf__` function using `MicroAPI` namespace — RegTensor + LoadAlign/StoreAlign + MaskReg.

### Concrete inline skeleton (self-contained, customer-runnable):

```cpp
// File: <op>_regbase_common.h
namespace <OpNs> {
constexpr uint32_t L0_MAX_SIZE = 64 * 1024;
constexpr uint32_t L1_MAX_SIZE = 512 * 1024;
constexpr uint32_t UB_MAX_SIZE = 128 * 1024;   // per-op allocation, full V351 UB=256KB
constexpr uint32_t L0C_MAX_SIZE = 256 * 1024;
constexpr uint32_t MODE_NUM_2 = 2;
constexpr uint32_t MODE_NUM_3 = 3;

// Dual-flag arrays for ping-pong, indexed by [taskIdMod2]
constexpr uint8_t SYNC_A_TO_B_FLAG[2] = {N, N+1};
constexpr uint8_t SYNC_B_TO_A_FLAG[2] = {M, M+1};

struct ConstInfo { /* per-kernel constants + tilingData slice */ };
struct RunInfo   { /* per-task: taskId, bIdx, taskIdMod2, ... */ };
struct KRunInfo  { /* per-K-loop: kTaskId, kTaskIdMod2, ... */ };

#define CUBE_BLOCK_TRAITS_TYPE_FIELDS(X)  X(INPUT_T) X(OUT_T) X(T)
#define CUBE_BLOCK_TRAITS_CONST_FIELDS(X) X(LAYOUT, OpLayout, OpLayout::TND)
#define TEMPLATES_DEF template <CUBE_BLOCK_TRAITS_TYPE_FIELDS(GEN_TYPE_PARAM) \
                                CUBE_BLOCK_TRAITS_CONST_FIELDS(GEN_CONST_PARAM) bool end = true>
}  // namespace
```

```cpp
// File: <op>_kernels.cpp (dispatcher TU)
#include "kernel_operator.h"

// PB-28 KB ENTRY CORRECTION (2026-05-25 02:17Z): KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)
// is NOT arch35-only. V220 ACCEPTS this macro natively per:
// (1) CANN canonical FA: flash_attention_score.cpp:379 uses it unconditionally on V220 build
// (2) Independent empirical 2026-05-25 02:17Z: removed arch-guard, V220 build + .so load PASS,
//     no RegisterAscendBinary 107000 error
// Earlier KB entry (independent prototype commit 1162679d) attributed 9/61→42/61 jump to PB-28 guard +
// 7-tuple ModelNew fix combined; falsification shows the 7-tuple was the load-bearing fix.
// PB-28 arch-guard was defensive over-application. KB entry PB-28 itself needs amendment.
// DO NOT add arch-guard around this macro for V220 builds — it's V220-native.
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

// ... kernel entry points (one per dtype+layout combo)
```

```cpp
// File: <op>_cube_block.h
namespace <OpNs> {
using T = float;
TEMPLATES_DEF
class OpBlockCube {
public:
    static constexpr uint8_t SYNC_MODE = 2;       // mode=2 manual flag-based for backward op (mode=4 KFC-internal only for forward FA-family per companion CAND)
    static constexpr uint32_t M_SPLIT_SIZE = 128;
    static constexpr uint32_t N_SPLIT_SIZE = 128;
    static constexpr uint32_t K_SPLIT_SIZE = 128;

    BufferManager<BufferType::L1> *l1BufferManagerPtr;
    BufferManager<BufferType::L0A> l0aBufferManager;
    BufferManager<BufferType::L0B> l0bBufferManager;
    BufferManager<BufferType::L0C> l0cBufferManager;
    BuffersPolicyDB<BufferType::L0A> l0aBuf;                                  // ping-pong
    BuffersPolicyDB<BufferType::L0B> l0bBuf;
    BuffersPolicy3buff<BufferType::L0C> commonL0CBuf;                          // 3-buffer for deeper pipeline
    BuffersPolicySingleBuffer<BufferType::L1> sYL1Buf;

    __aicore__ inline void ComputeMm1(Buffer<UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                       BuffersPolicyDB<L1, SyncType::CROSS_CORE_SYNC_BOTH> &sYL1Buf,
                                       RunInfo &runInfo, ConstInfo &constInfo, KRunInfo &kInfo) {
        // AIC waits for AIV gather completion
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(SYNC_A_TO_B_FLAG[kInfo.kTaskIdMod2]);
        // ... matmul invocation (op-specific) ...
        // AIC signals AIV that result is ready in workspace
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
    }
};
}
```

```cpp
// File: <op>_vector_block.h
#include "vf/vf_process_vec0.h"
#include "vf/vf_process_vec1.h"
// ... per-stage vf headers

namespace <OpNs> {
TEMPLATES_DEF
class OpBlockVec {
public:
    __aicore__ inline void ProcessVector1(Buffer<UB, SyncType::CROSS_CORE_SYNC_BOTH> &bmm1ResBuf,
                                          RunInfo &runInfo, KRunInfo &kInfo) {
        // AIV waits for AIC matmul result
        CrossCoreWaitFlag<2, PIPE_V>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
        // ... call vf_process_vec1 primitives ...
        // AIV releases buffer
        CrossCoreSetFlag<2, PIPE_V>(SYNC_B_TO_A_FLAG[kInfo.kTaskIdMod2]);
    }
};
}
```

```cpp
// File: vf/vf_process_vecN.h
namespace AscendC {
using namespace MicroAPI;
template <typename INPUT_T>
__simd_vf__ inline void OpSpecificMicroOp(__ubuf__ INPUT_T *dstUb, __ubuf__ INPUT_T *srcUb,
                                          uint32_t m, uint32_t n) {
    RegTensor<INPUT_T> vreg_x;
    MaskReg preg = UpdateMask<uint16_t>(/*repeatSize=*/128);
    for (uint16_t i = 0; i < m; ++i) {
        LoadAlign(vreg_x, srcUb + i * n);
        StoreAlign<INPUT_T,
                   MicroAPI::DataCopyMode::DATA_BLOCK_COPY,
                   MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            ((__ubuf__ INPUT_T *&)dstUb), vreg_x, /*blockStride=*/m, /*repeatStride=*/1, preg);
    }
}
}  // namespace
```

```cpp
// File: <op>_kernel_base.h
namespace <OpNs> {
template <typename CubeBlockType, typename VecBlockType>
class OpKernelBase {
public:
    __aicore__ inline void Init(GM_ADDR ...inputs, GM_ADDR workspace,
                                const TilingData *tiling, TPipe *tPipe) {
        pipe = tPipe;
        tilingData = tiling;
        SetConstInfo();
        InitWorkspace(workspace);
        if ASCEND_IS_AIV {
            vecBlock.InitParams(constInfo, tilingData);
            vecBlock.InitGlobalBuffer(/* AIV-side GMs */);
            vecBlock.InitBuffers(pipe);
        } else if ASCEND_IS_AIC {
            cubeBlock.SetCubeBlockParams(tPipe, &l1BufferManager);
            cubeBlock.InitCubeBuffers();
            cubeBlock.InitGlobalBuffer(/* AIC-side GMs */);
        }
    }
    __aicore__ inline void Process() {
        // Outer loop with [taskId % MODE_NUM_2] ping-pong on kRunInfos[]
        KRunInfo kRunInfos[MODE_NUM_2];
        for (int32_t taskId = 0; taskId < runInfo.kLoopTimes + 1; ++taskId) {
            KRunInfo &cur = kRunInfos[taskId % MODE_NUM_2];
            KRunInfo &prev = kRunInfos[(taskId + 1) % MODE_NUM_2];
            // ... AIV produces cur, AIC consumes prev (deep pipeline) ...
        }
    }
    TPipe *pipe;
    const TilingData *__restrict tilingData;
    BufferManager<BufferType::UB> ubBufferManager;
    BufferManager<BufferType::L1> l1BufferManager;
    BuffersPolicyDB<BufferType::UB, SyncType::CROSS_CORE_SYNC_BOTH> bmm1Buffers;
    ConstInfo constInfo;
    CubeBlockType cubeBlock;
    VecBlockType vecBlock;
};
}
```

### Tiling data class hierarchy (V351 RegBase pattern)

```cpp
namespace optiling {
class BaseParamsRegbase {
    int32_t bSize, n2Size, s1Size, s2Size, dSize, kSize;
    float scaleValue;
    uint8_t layoutType;
    int32_t get_bSize() const { return bSize; }
    void set_bSize(int32_t p) { this->bSize = p; }
    // ... pair for each field
};
class MultiCoreParamsRegbase {
    uint32_t coreNum;
    int64_t splitFactorSize, totalSize;
    int64_t bS1Index[MAX_CORE_NUM_REGBASE];  // 36 max for V351
};
class VecApiParamsRegbase {
    SoftMaxTiling softmaxTilingData;          // for Vec API sub-tiling if applicable
};
class RegBaseTilingData {
    BaseParamsRegbase baseParams;
    MultiCoreParamsRegbase multiCoreParams;
    VecApiParamsRegbase vectorParams;
};
}
```

### Template tiling key (compile-time variant selector)

```cpp
ASCENDC_TPL_ARGS_DECL(OpName,
    ASCENDC_TPL_BOOL_DECL(VARIANT_BOOL, 0, 1),
    ASCENDC_TPL_UINT_DECL(VARIANT_RANGE, 4, ASCENDC_TPL_UI_LIST, 0, 1, 2),
    ASCENDC_TPL_UINT_DECL(LAYOUT, 4, ASCENDC_TPL_UI_LIST, 0, 1),
    ASCENDC_TPL_BOOL_DECL(DETERMINISTIC, 0, 1),
);
ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_BOOL_SEL(VARIANT_BOOL, 0, 1),
        // ... valid combinations
        ASCENDC_TPL_TILING_STRUCT_SEL(optiling::RegBaseTilingData)
    )
);
```

### Reject_cond — do NOT apply this skeleton when:
- Op is FA-forward class (online-softmax tile-scheduling) — different sync paradigm, see CAND-V220-to-V351-PortPattern body (mode=4 + outer-only KFC-internal for forward LIG-family)
- Op is pure-VEC (no cube stage) — vector fission still applies but cube_block + L0A/L0B/L0C abstractions unused
- Op doesn't fit `MIX_AIC_1_2` topology — `BuffersPolicy{DB,3buff,SingleBuffer}` abstractions assume cube+AIV producer-consumer; pure-cube or pure-AIV patterns differ

### Sync-mode nuance vs companion CAND (open audit item):

CAND-V220-to-V351-PortPattern-CubeVecFusedOp documents V220 mode=2 → V351 mode=4 + KFC-internal-implicit (no per-block manual sync) based on forward LIG arch22→arch35 evidence. THIS entry documents V351 backward sparse_LIG_grad_kl using **mode=2 + per-block manual sync with dual-flag ping-pong** (verified via direct grep of `CrossCoreSetFlag<2, PIPE_X>` 30+ occurrences in `kernel_base.h` + `cube_block.h` + `vector_block.h`).

**Reconciliation hypothesis** (needs main verify on forward LIG arch35 source):
- **V351 forward LIG-family** (lightning_indexer arch35): mode=4 + KFC-internal (cgmct-style sync wrapper at outer loop, no per-block inner sync)
- **V351 backward LIG-family** (sparse_lightning_indexer_grad_kl_loss arch35): mode=2 + manual per-block sync + dual-flag `[taskIdMod2]` ping-pong (canonical V220-style paradigm preserved on V351 hardware)

Possible reasons for divergence:
- Forward op tile-scheduling has multi-task-per-block pipeline depth → benefits from KFC-internal sync via cgmct wrapper
- Backward op task-per-block 1:1 → manual flag-based sync sufficient + no cgmct wrapper overhead
- Or main's "mode=4" claim from forward source may need verify against newer commit (forward LIG arch35 kernel.h is 30705 bytes — not full read yet)

**Promote when** (separate from companion CAND):
1. LIG_grad ships via this service-class skeleton (bg orch `bmz9tfk7b`) — verifies skeleton at runtime
2. A SECOND V351 backward op (e.g. attention_grad sans softmax, MoE-finalize backward, fused-quant-matmul backward) applies the skeleton + ships clean
3. Sync-mode nuance reconciled: confirm forward = mode=4 + outer-KFC; backward = mode=2 + per-block-manual; or unify if same paradigm proves to apply both

**Other instances (predicted)**:
- LIG_grad (current bg orch `bmz9tfk7b`) — direct beneficiary
- dense_lightning_indexer_grad_kl_loss arch35 (CANN sibling, parallel-pair with sparse — paradigm identical)
- attention_grad backward without softmax
- MoE-finalize backward
- Custom V351 backward fused-norm-matmul / fused-quant-matmul

**Customer-impact**: customer brings V220 backward multi-stage cube+vec op to harness → harness applies (companion CAND sync paradigm + this CAND service-class skeleton) → kw generates V351 kernel TU **without CANN-source access**. Fully self-contained per Zheng 2026-05-24T21:28Z directive.

**Cross-link**: companion CAND-V220-to-V351-PortPattern-CubeVecFusedOp (commit fb13a899 on origin/main — sync paradigm delta); CAND-V351-AIV-WholeReduceMax-fp32-mask-cap (commit 5260fd68 — V351 AIV reduction primitive gotcha applicable to backward ops); OL-185 (op_class L2/L3 calibration anchor); CAND-V220-V351-FA-DIFF-1 (forward FA-class differs from this skeleton — that's L4 path).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-V351-arch35-RegBase-service-class-skeleton，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->

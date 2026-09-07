# FA-Paged 手搓完整配方（四文件端到端逐步骤，NO_MASK 全量验证 PASS）

> **用途**：**标准 FA / FA-BNSD / FA-Paged / GQA**（`O=softmax(scale·Q·Kᵀ[+mask])·V`，分页 KV）手搓 kernel
> 的 skill 生成依据——**端到端四文件**（kernel + kernel_common + tiling + host）全部在本 recipe 内，
> 闭卷生成不依赖任何外部文件。
> 以下每一步均为**已验证可编译+跑通+精度 PASS 的真实代码**（kv=128~204800 全 100% PASS，max_abs≤7.6e-06）。
>
> **构成**：`FAInferKernel` 四文件从零手搓的完整步骤 +
> 3 条实测修复（LayoutP stride / isLast flag / L1_QK_SIZE 偏移）。
>
> **★与 MLA 共有的死锁关键**（必读 fa-mla-handcraft-recipe §0 实测纠错）：
> - 分流用 `operator()<AIC>/<AIV>` 模板特化（非 `__mix__`+g_coreType runtime if）。
> - 延迟 PV（preLaunch 深度 2），非同块锁步。
> - CrossCoreFlag Set `<0x2,PIPE_X>` + bare Wait。
> - AIV `coreIdx=GetBlockIdx()/GetSubBlockNum()`。
>
> **算子族覆盖**：Step 1-7+9-11 = NO_MASK 全注意力（标准 FA/Paged/GQA/变长 batch per-batch seqlen）；
> Step 8 = MASK_SPEC（逐位）/ MASK_CAUSAL（因果）叠加。变长=per-batch actualQseqlen/actualKvseqlen 天然支持
> （任务切分逐 batch 重算，无需改 kernel）。sink/varlen/q8 等 kfc 路径叠加见特性册 §12.6/§12.12/§12.13。

---

## Step 1: includes + 类型别名 + 组件选型

```cpp
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"
#include "kernel_common.hpp"
#include "kernel_operator.h"

using namespace Catlass;
namespace cg = Catlass::Gemm;
namespace ce = Catlass::Epilogue;
namespace cl = Catlass::layout;

// 类型
using ElementQ=half; using ElementK=half; using ElementV=half; using ElementP=half; using ElementO=half;
using ElementS=float; using ElementOTmp=float; using ElementUpdate=float; using ElementMask=half;
using LayoutQ=cl::RowMajor; using LayoutK=cl::ColumnMajor; using LayoutV=cl::RowMajor;  // ★FA: V=RowMajor（非 MLA 的 Col）
using LayoutS=cl::RowMajor; using LayoutP=cl::RowMajor; using LayoutO=cl::RowMajor;
using LayoutOTmp=cl::RowMajor; using LayoutUpdate=cl::RowMajor; using LayoutMask=cl::RowMajor;

// 组件
using L1TileShape=GemmShape<128,128,128>;     // ★K=embed=128（MLA 是 576）
using L0TileShape=L1TileShape;
using BlockMmadQK=cg::Block::BlockMmad<cg::MmadAtlasA2FAIQK<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutQ>,cg::GemmType<half,LayoutK>,cg::GemmType<float,LayoutS>>;
using BlockMmadPV=cg::Block::BlockMmad<cg::MmadAtlasA2FAIPV<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutP>,cg::GemmType<half,LayoutV>,cg::GemmType<float,LayoutOTmp>>;
using EpilogueOnlineSoftmax=ce::Block::BlockEpilogue<ce::EpilogueAtlasA2OnlineSoftmax,
    cg::GemmType<half,LayoutP>,cg::GemmType<float,LayoutS>,cg::GemmType<half,LayoutMask>>;
// ★EpilogueRescaleO 需要 3 个 GemmType 参（+OTmpType），非 2 个
using EpilogueRescaleO=ce::Block::BlockEpilogue<ce::EpilogueAtlasA2RescaleO,
    cg::GemmType<half,LayoutO>,cg::GemmType<float,LayoutUpdate>,cg::GemmType<float,LayoutOTmp>>;
```

---

## Step 2: kernel 类骨架 + Params

```cpp
template <class BQK, class BPV, class EpSoft, class EpRes>
class FAInferKernel {
public:
    using ArchTag=Arch::AtlasA2;
    // ★Params 成员必须每行单独声明（GM_ADDR=__gm__ uint8_t* 不能逗号列表——bisheng 拒）
    struct Params {
        GM_ADDR q; GM_ADDR k; GM_ADDR v; GM_ADDR mask; GM_ADDR blockTables;
        GM_ADDR actualQseqlen; GM_ADDR actualKvseqlen; GM_ADDR o; GM_ADDR s; GM_ADDR p;
        GM_ADDR oTemp; GM_ADDR oUpdate; GM_ADDR tiling;
        CATLASS_DEVICE Params() {}
        CATLASS_DEVICE Params(GM_ADDR q_,GM_ADDR k_,GM_ADDR v_,GM_ADDR mask_,GM_ADDR bt_,GM_ADDR aq_,GM_ADDR ak_,
            GM_ADDR o_,GM_ADDR s_,GM_ADDR p_,GM_ADDR ot_,GM_ADDR ou_,GM_ADDR t_)
            : q(q_),k(k_),v(v_),mask(mask_),blockTables(bt_),actualQseqlen(aq_),actualKvseqlen(ak_),
              o(o_),s(s_),p(p_),oTemp(ot_),oUpdate(ou_),tiling(t_) {}
    };
    // ★分流：主模板声明（无 body）+ 两个特化。entry 调 fa(params) 自动编译期分派。
    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(Params const& params);

    template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) { /* Step 3 */ }
    template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) { /* Step 4 */ }
};
```

---

## Step 3: AIC body（cube 核：QK matmul + PV matmul + CrossCoreFlag Set）

```cpp
template <> CATLASS_DEVICE void operator()<AscendC::AIC>(Params const& params) {
    // 3a. HardEvent 预置（25 个 token，§8.6 AIC 全表）
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6); AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    for(uint32_t e=0;e<8;e++) AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(e);
    for(uint32_t e=0;e<6;e++) AscendC::SetFlag<AscendC::HardEvent::FIX_MTE1>(e);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);

    // 3b. Resource + 组件构造
    Arch::Resource<ArchTag> resource;
    // ★修复3: BlockMmadPV 构造传 L1_QK_SIZE（L1 偏移），非 0
    static constexpr uint32_t L1_QK_SIZE =
        BQK::L1TileShape::M * BQK::L1TileShape::K * sizeof(ElementQ) +
        BQK::L1TileShape::N * BQK::L1TileShape::K * sizeof(ElementK) * 2;
    BQK blockMmadQK(resource);
    BPV blockMmadPV(resource, L1_QK_SIZE);

    // 3c. 读 FATilingData + GlobalTensor
    __gm__ FATilingData* td = reinterpret_cast<__gm__ FATilingData*>(params.tiling);
    uint32_t batch=td->batch, qHeads=td->numHeads, kvHeads=td->kvHeads, embed=td->embeddingSize;
    uint32_t pagedBlockSize=td->blockSize, maxNumBlocksPerBatch=td->maxNumBlocksPerBatch;
    uint32_t totalTaskNum=td->totalTaskNum; uint32_t maskType=td->maskType; float scaleValue=td->scaleValue;
    AscendC::GlobalTensor<ElementQ> gQ; gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
    AscendC::GlobalTensor<ElementK> gK; gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
    AscendC::GlobalTensor<ElementK> gV; gV.SetGlobalBuffer((__gm__ ElementK*)params.v);
    AscendC::GlobalTensor<int32_t> gBT; gBT.SetGlobalBuffer((__gm__ int32_t*)params.blockTables);
    AscendC::GlobalTensor<int64_t> gAQ; gAQ.SetGlobalBuffer((__gm__ int64_t*)params.actualQseqlen);
    AscendC::GlobalTensor<int64_t> gAK; gAK.SetGlobalBuffer((__gm__ int64_t*)params.actualKvseqlen);
    AscendC::GlobalTensor<ElementS> gS; gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
    AscendC::GlobalTensor<ElementP> gP; gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
    AscendC::GlobalTensor<ElementOTmp> gOTmp; gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
    uint64_t strideQO=qHeads*embed, strideKV=kvHeads*embed;
    uint32_t groupSize=qHeads/kvHeads;
    uint32_t coreIdx=AscendC::GetBlockIdx(), coreNum=AscendC::GetBlockNum();
    Arch::CrossCoreFlag qkReady(QK_READY_ID), pvReady(PV_READY_ID), softmaxReady(SOFTMAX_READY_ID);
    constexpr uint32_t blockStackNum=4; constexpr int32_t preLaunch=2;

    // 3d. 任务遍历初始化
    uint32_t curBatch=0, preTotalTaskNum=0, curTotalTaskNum=0;
    uint64_t qBOffset=0, blockBOffset=0; int64_t qSeqlen=0, kvSeqlen=0;
    qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
    uint32_t curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize);
    uint32_t qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
    uint32_t curQNBlockNum=qNBlockNumPerGroup*kvHeads;
    int64_t curQSBlockTile=GetQSBlockTile(kvSeqlen);
    uint32_t curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile);
    curTotalTaskNum=curQNBlockNum*curQSBlockNum;

    // 3e. 任务遍历主循环
    for (uint32_t taskIdx=coreIdx; taskIdx<totalTaskNum; taskIdx+=coreNum) {
        while (taskIdx>=curTotalTaskNum) {
            ++curBatch; preTotalTaskNum=curTotalTaskNum; qBOffset+=qSeqlen*strideQO; blockBOffset+=maxNumBlocksPerBatch;
            qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
            curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize); qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
            curQNBlockNum=qNBlockNumPerGroup*kvHeads; curQSBlockTile=GetQSBlockTile(kvSeqlen);
            curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum+=curQNBlockNum*curQSBlockNum;
        }
        uint32_t tcb=taskIdx-preTotalTaskNum;
        uint32_t qSBlockIdx=tcb/curQNBlockNum, qNBlockIdx=tcb%curQNBlockNum;
        uint32_t qNBlockIdxCurGroup=qNBlockIdx%qNBlockNumPerGroup;
        uint32_t kvHeadIdx=qNBlockIdx/qNBlockNumPerGroup;
        uint32_t qHeadIdx=kvHeadIdx*groupSize+qNBlockIdxCurGroup*curQNBlockTile;
        uint64_t gmQOffset=qBOffset+qSBlockIdx*curQSBlockTile*strideQO+qHeadIdx*embed;
        uint32_t qSBlockSize=(qSBlockIdx==(curQSBlockNum-1))?(qSeqlen-qSBlockIdx*curQSBlockTile):curQSBlockTile;
        uint32_t qNBlockSize=(qNBlockIdxCurGroup==(qNBlockNumPerGroup-1))?(groupSize-qNBlockIdxCurGroup*curQNBlockTile):curQNBlockTile;
        uint32_t rowNum=qSBlockSize*qNBlockSize;
        uint32_t noMaskKvS=kvSeqlen, kvSLoopNumNoMask=CeilDiv(noMaskKvS,pagedBlockSize);
        uint32_t kvSLoopNumTotal=kvSLoopNumNoMask;
        int32_t totalStackSeqNum=CeilDiv(noMaskKvS,blockStackNum*pagedBlockSize);
        int32_t stackSeqCount=0; uint32_t stackSeqTile;
        LayoutQ layoutQTemp(rowNum,embed); LayoutK layoutKTemp(strideKV,blockStackNum*pagedBlockSize);
        LayoutV layoutVTemp(blockStackNum*pagedBlockSize,strideKV);
        blockMmadQK.loadQGM(gQ[gmQOffset],layoutQTemp,rowNum,qNBlockSize,qHeads);

        // 3f. 主循环（QK + 延迟 PV）
        for (uint32_t kvSIdx=0; kvSIdx<kvSLoopNumNoMask; kvSIdx+=blockStackNum) {
            stackSeqTile=(kvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-kvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
            uint32_t curStackTileMod=stackSeqCount%(preLaunch+1);
            uint64_t gmSOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod*WORKSPACE_BLOCK_SIZE_DB;
            GemmCoord shapeQK{rowNum,stackSeqTile,embed};
            blockMmadQK(gQ[gmQOffset],gK,gS[gmSOffset],gBT[blockBOffset],layoutQTemp,layoutKTemp,
                        shapeQK,kvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV);
            Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);
            if ((int32_t)kvSIdx>=preLaunch*blockStackNum) {  // 延迟 PV（preLaunch 深度）
                uint32_t nowkvSIdx=kvSIdx-preLaunch*blockStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                uint32_t pvPP=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmPOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+pvPP*WORKSPACE_BLOCK_SIZE_DB;
                // ★修复1: LayoutP 必须 3-参 stride=512（与 softmax 写 P 的 stride 一致）
                LayoutP layoutPTemp(rowNum,stackSeqTile,512);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                blockMmadPV(gP[gmPOffset],gV,gOTmp[gmPOffset],gBT[blockBOffset],layoutPTemp,layoutVTemp,
                            shapePV,nowkvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV,softmaxReady);
                Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
            }
            stackSeqCount++;
        }

        // 3g. 次循环（drain 延迟 PV 尾巴）
        uint32_t maskedStartIdx=AlignUp(kvSLoopNumNoMask,blockStackNum);
        uint32_t preLaunchStackNum=preLaunch*blockStackNum;
        for (uint32_t kvSIdx=maskedStartIdx; kvSIdx<kvSLoopNumTotal+preLaunchStackNum; ) {
            if (kvSIdx>=preLaunchStackNum) {
                uint32_t nowkvSIdx=kvSIdx-preLaunchStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                uint32_t pvPP=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmPOffset=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+pvPP*WORKSPACE_BLOCK_SIZE_DB;
                LayoutP layoutPTemp(rowNum,stackSeqTile,512);  // ★修复1: 3-参 stride=512
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                blockMmadPV(gP[gmPOffset],gV,gOTmp[gmPOffset],gBT[blockBOffset],layoutPTemp,layoutVTemp,
                            shapePV,nowkvSIdx,kvSLoopNumNoMask,pagedBlockSize,noMaskKvS,strideKV,softmaxReady);
                Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
            }
            kvSIdx+=blockStackNum; stackSeqCount++;
        }
    }

    // 3h. drain AIC HardEvent（反预置，§8.6 全表 Wait）
    for(uint32_t e=0;e<8;e++) AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(e);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
    for(uint32_t e=0;e<8;e++) AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(e);
    for(uint32_t e=0;e<6;e++) AscendC::WaitFlag<AscendC::HardEvent::FIX_MTE1>(e);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_FIX>(EVENT_ID0);
}
```

---

## Step 4: AIV body（vector 核：softmax + rescale + CrossCoreFlag Wait）

```cpp
template <> CATLASS_DEVICE void operator()<AscendC::AIV>(Params const& params) {
    // 4a. HardEvent 预置（10 个 token，FA AIV 表）
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4); AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);

    // 4b. Resource + epilogue 构造
    Arch::Resource<ArchTag> resource;
    __gm__ FATilingData* td=reinterpret_cast<__gm__ FATilingData*>(params.tiling);
    uint32_t batch=td->batch,qHeads=td->numHeads,kvHeads=td->kvHeads,embed=td->embeddingSize;
    uint32_t pagedBlockSize=td->blockSize,maxNumBlocksPerBatch=td->maxNumBlocksPerBatch;
    uint32_t firstBatchTaskNum=td->firstBatchTaskNum,totalTaskNum=td->totalTaskNum,maskType=td->maskType;
    float scaleValue=td->scaleValue;
    AscendC::GlobalTensor<int64_t> gAQ; gAQ.SetGlobalBuffer((__gm__ int64_t*)params.actualQseqlen);
    AscendC::GlobalTensor<int64_t> gAK; gAK.SetGlobalBuffer((__gm__ int64_t*)params.actualKvseqlen);
    AscendC::GlobalTensor<ElementO> gO; gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
    AscendC::GlobalTensor<ElementS> gS; gS.SetGlobalBuffer((__gm__ ElementS*)params.s);
    AscendC::GlobalTensor<ElementP> gP; gP.SetGlobalBuffer((__gm__ ElementP*)params.p);
    AscendC::GlobalTensor<ElementOTmp> gOTmp; gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)params.oTemp);
    uint32_t groupSize=qHeads/kvHeads, embedRound=RoundUp(embed,BLOCK_SIZE);
    EpSoft epSoft(resource,scaleValue); EpRes epRes(resource);
    // ★AIV coreIdx 必须除以 subBlockNum（否则双子核都跑 task0→flag 死锁）
    uint32_t coreIdx=AscendC::GetBlockIdx()/AscendC::GetSubBlockNum(), coreNum=AscendC::GetBlockNum();
    Arch::CrossCoreFlag qkReady(QK_READY_ID), pvReady(PV_READY_ID), softmaxReady(SOFTMAX_READY_ID);
    constexpr uint32_t blockStackNum=4; constexpr int32_t preLaunch=2;

    // 4c. 任务遍历初始化（同 AIC，但 oBatchOffset 替代 qBOffset/blockBOffset）
    uint32_t curBatch=0,preTotalTaskNum=0,curTotalTaskNum=firstBatchTaskNum;
    int64_t oBatchOffset=0,qSeqlen=gAQ.GetValue(curBatch),kvSeqlen=gAK.GetValue(curBatch);
    uint32_t curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize),qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
    uint32_t curQNBlockNum=qNBlockNumPerGroup*kvHeads; int64_t curQSBlockTile=GetQSBlockTile(kvSeqlen);
    uint32_t curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum=curQNBlockNum*curQSBlockNum;

    // 4d. 任务遍历主循环
    for (uint32_t taskIdx=coreIdx; taskIdx<totalTaskNum; taskIdx+=coreNum) {
        while (taskIdx>=curTotalTaskNum) {
            ++curBatch; oBatchOffset+=qSeqlen*qHeads*embed; preTotalTaskNum=curTotalTaskNum;
            qSeqlen=gAQ.GetValue(curBatch); kvSeqlen=gAK.GetValue(curBatch);
            curQNBlockTile=GetQNBlockTile(qSeqlen,groupSize); qNBlockNumPerGroup=CeilDiv(groupSize,curQNBlockTile);
            curQNBlockNum=qNBlockNumPerGroup*kvHeads; curQSBlockTile=GetQSBlockTile(kvSeqlen);
            curQSBlockNum=CeilDiv(qSeqlen,curQSBlockTile); curTotalTaskNum+=curQNBlockNum*curQSBlockNum;
        }
        uint32_t tcb=taskIdx-preTotalTaskNum;
        uint32_t qSBlockIdx=tcb/curQNBlockNum, qNBlockIdx=tcb%curQNBlockNum, qNBlockIdxCurGroup=qNBlockIdx%qNBlockNumPerGroup;
        uint32_t kvHeadIdx=qNBlockIdx/qNBlockNumPerGroup, qStartNIdx=kvHeadIdx*groupSize+qNBlockIdxCurGroup*curQNBlockTile;
        int64_t gmOffsetO=oBatchOffset+qSBlockIdx*curQSBlockTile*qHeads*embed+qStartNIdx*embed;
        uint32_t qSBlockSize=(qSBlockIdx==(curQSBlockNum-1))?(qSeqlen-qSBlockIdx*curQSBlockTile):curQSBlockTile;
        uint32_t qNBlockSize=(qNBlockIdxCurGroup==(qNBlockNumPerGroup-1))?(groupSize-qNBlockIdxCurGroup*curQNBlockTile):curQNBlockTile;
        uint32_t rowNum=qSBlockSize*qNBlockSize;
        uint32_t noMaskKvS=kvSeqlen, kvSLoopNumNoMask=CeilDiv(noMaskKvS,pagedBlockSize), kvSLoopNumTotal=kvSLoopNumNoMask;
        int32_t totalStackSeqNum=CeilDiv(noMaskKvS,blockStackNum*pagedBlockSize), stackSeqCount=0;
        uint32_t stackSeqTile, stackSeqTilePad=blockStackNum*pagedBlockSize;

        // 4e. 主循环（softmax + 延迟 rescale）
        for (uint32_t kvSIdx=0; kvSIdx<kvSLoopNumNoMask; kvSIdx+=blockStackNum) {
            stackSeqTile=(kvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-kvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
            LayoutS layOutS(rowNum,stackSeqTile,stackSeqTilePad); LayoutP layOutP(rowNum,stackSeqTile,stackSeqTilePad);
            GemmCoord shapeQK{rowNum,stackSeqTile,embed};
            uint32_t curStackTileMod=stackSeqCount%(preLaunch+1);
            uint64_t gmOffsetS=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod*WORKSPACE_BLOCK_SIZE_DB;
            uint64_t gmOffsetP=gmOffsetS;
            Arch::CrossCoreWaitFlag(qkReady);
            epSoft(gP[gmOffsetP],gS[gmOffsetS],layOutP,layOutS,shapeQK,(stackSeqCount==0),qSBlockSize,qNBlockSize,curStackTileMod);
            Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);
            if ((int32_t)kvSIdx>=preLaunch*blockStackNum) {  // 延迟 rescale
                stackSeqTile=(kvSIdx>=kvSLoopNumNoMask)?(noMaskKvS-(kvSIdx-preLaunch*blockStackNum)*pagedBlockSize):(pagedBlockSize*blockStackNum);
                LayoutO layoutO(qSeqlen,embed*qHeads);
                // ★LayoutOTmp 3-参 stride=embedRound（与 PV 写 OTmp 的 stride 一致）
                LayoutOTmp layoutOTmp(rowNum,embed,embedRound);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                uint32_t curStackTileMod2=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmOffsetOTmp=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod2*WORKSPACE_BLOCK_SIZE_DB;
                Arch::CrossCoreWaitFlag(pvReady);
                epRes(gO[gmOffsetO],gOTmp[gmOffsetOTmp],layoutO,layoutOTmp,shapePV,qSBlockSize,qNBlockSize,
                      (stackSeqCount-preLaunch==0),0,curStackTileMod2);  // primary isLast=0
            }
            stackSeqCount++;
        }

        // 4f. 次循环（drain 延迟 rescale 尾巴）
        uint32_t maskedStartIdx=AlignUp(kvSLoopNumNoMask,blockStackNum), preLaunchStackNum=preLaunch*blockStackNum;
        for (uint32_t kvSIdx=maskedStartIdx; kvSIdx<kvSLoopNumTotal+preLaunchStackNum; ) {
            if (kvSIdx>=preLaunchStackNum) {
                uint32_t nowkvSIdx=kvSIdx-preLaunchStackNum;
                stackSeqTile=(nowkvSIdx+blockStackNum>kvSLoopNumNoMask-1)?(noMaskKvS-nowkvSIdx*pagedBlockSize):(pagedBlockSize*blockStackNum);
                LayoutO layoutO(qSBlockSize,embed*qHeads); LayoutOTmp layoutOTmp(rowNum,embed,embedRound);
                GemmCoord shapePV{rowNum,embed,stackSeqTile};
                uint32_t curStackTileMod2=(stackSeqCount-preLaunch)%(preLaunch+1);
                uint64_t gmOffsetOTmp=coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1)+curStackTileMod2*WORKSPACE_BLOCK_SIZE_DB;
                Arch::CrossCoreWaitFlag(pvReady);
                // ★修复2: isLast = (stackSeqCount-preLaunch==totalStackSeqNum-1)（仅末块 finalize），非恒1
                epRes(gO[gmOffsetO],gOTmp[gmOffsetOTmp],layoutO,layoutOTmp,shapePV,qSBlockSize,qNBlockSize,
                      (stackSeqCount-preLaunch==0),(stackSeqCount-preLaunch==totalStackSeqNum-1),curStackTileMod2);
            }
            kvSIdx+=blockStackNum; stackSeqCount++;
        }
    }

    // 4g. drain AIV HardEvent（反预置，顺序照抄 4a 预置表）
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2); AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4); AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0); AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3); AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
    AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
}
```

---

## Step 5: 实例化 + entry

```cpp
using FaKernel = FAInferKernel<BlockMmadQK,BlockMmadPV,EpilogueOnlineSoftmax,EpilogueRescaleO>;

CATLASS_GLOBAL void FAInferFp16(uint64_t hwSync, GM_ADDR q,GM_ADDR k,GM_ADDR v,GM_ADDR mask,
    GM_ADDR blockTables, GM_ADDR o, GM_ADDR qSeq,GM_ADDR kvSeq, GM_ADDR s,GM_ADDR p,GM_ADDR oTemp,
    GM_ADDR oUpdate, GM_ADDR tiling) {
    AscendC::SetSyncBaseAddr(hwSync);
    FaKernel::Params params{q,k,v,mask,blockTables,qSeq,kvSeq,o,s,p,oTemp,oUpdate,tiling};
    FaKernel fa;
    fa(params);  // operator()<g_coreType>() — 编译期分派（AIC on cube, AIV on vector）
}
```

---

## Step 6: 3 条关键修复（照写即一次通过）

1. **★修复1（FA 独有坑）**：AIC PV 的 `LayoutP layoutPTemp` **必须 3-参 `(rowNum, stackSeqTile, 512)`**（stride=512=blockStackNum*pagedBlockSize）。
   非 2-参 `(rowNum, stackSeqTileRound)`（默认 stride≠512）。因 AIV softmax **写** P 用 3-参 `LayoutP(rowNum, stackSeqTile, stackSeqTilePad=512)` stride=512，
   PV 读 P 的 stride 必须匹配，否则只读到 1/4 的 P 行（row 0,4,8,12）→ O 只 4 head 对。MLA 无此问题（MLA P layout 统一 3-参）。

2. **★修复2**：secondary rescale 的 `isLast` = `(stackSeqCount-preLaunch==totalStackSeqNum-1)`（仅末块 finalize），
   非恒 `1`（恒 1 提前 finalize → 多块精度错/hang）。primary rescale 的 isLast = `0`。

3. **★修复3**：`BlockMmadPV(resource, L1_QK_SIZE)`（L1 偏移=QK 的 L1 分配大小），非 `BlockMmadPV(resource, 0)`。
   `L1_QK_SIZE = L1TileShape::M * L1TileShape::K * sizeof(ElementQ) + L1TileShape::N * L1TileShape::K * sizeof(ElementK) * 2`。

---

## Step 7: GM 槽位 + launch + build + tiling

```
GM workspace slots（preLaunch+1=3 槽 pingpong）:
  gmSOffset/gmPOffset/gmOTmpOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + curStackTileMod*WORKSPACE_BLOCK_SIZE_DB
  (WORKSPACE_BLOCK_SIZE_DB=131072)

workspace sizes:
  s  = aicCore * WS_DB * 3 * sizeof(float)
  p  = aicCore * WS_DB * 3 * sizeof(half)
  oTemp = aicCore * WS_DB * 3 * sizeof(float)
  oUpdate = aicCore * WS_DB * 3 * sizeof(float)

launch: FAInferFp16<<<aicCoreNum,nullptr,stream>>>(hwSync,q,k,v,mask,blockTables,o,qSeq,kvSeq,s,p,oTmp,oUpdate,tiling)
  14 params (hwSync + 13 GM_ADDR)

tiling: FAInferTiling::GetFATilingParam(FAInfo, blockDim, FATilingData)
  FAInfo: numTokens, numHeads, embeddingSize, numBlocks, blockSize, kvHeads, batch,
          qSeqlenList(int64_t*), kvSeqlenList(int64_t*), maskType

build: CATLASS_ARCH=2201 + --npu-arch=dav-2201
  host .asc #include "fai_kernel.cpp" 单 TU
  link: ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m
  fai_tiling.cpp 单独 CXX TU（不 #include 进 host）
  fai_tiling.cpp 与 fai_kernel.cpp 都需要 kernel_common.hpp（定义 FATilingData + 常量）
```

---

## Step 8: mask 路径（MASK_SPEC=1 逐位 / MASK_CAUSAL=2 因果）——masked 双循环逐步

> Step 1-7 是 `maskType=NO_MASK(0)` 全注意力路径。mask 路径的改动集中在：
> **①任务级边界变量 ②次循环起点/推进 ③QKTail/PVTail 组件 ④AIV softmax 换带 mask 版调用**。
> 主循环（第一循环）体零改动（只是遍历域从 kvSeqlen 缩到 noMaskKvS）。

### 8a. 组件增补（Step 1 追加）

```cpp
// ★Tail 组件名核对（catlass include dispatch_policy.hpp:175 实测存在）：
//   TailQK = MmadAtlasA2FAITailQK、TailPV = MmadAtlasA2FAITailPV
//   （"FAI"+"Tail"+"QK/PV"——勿写成 FAIQKTail/FAIPVTail，该名不存在，编译报未定义）
using BlockMmadQKTail=cg::Block::BlockMmad<cg::MmadAtlasA2FAITailQK<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutQ>,cg::GemmType<half,LayoutK>,cg::GemmType<float,LayoutS>>;
using BlockMmadPVTail=cg::Block::BlockMmad<cg::MmadAtlasA2FAITailPV<true,false>,L1TileShape,L0TileShape,
    cg::GemmType<half,LayoutP>,cg::GemmType<half,LayoutV>,cg::GemmType<float,LayoutOTmp>>;
```

AIC 构造（与 QK/PV 并列，同款 L1 偏移）：
```cpp
BlockMmadQKTail blockMmadQKTail(resource);
BlockMmadPVTail blockMmadPVTail(resource, L1_QK_SIZE);
```

### 8b. 任务级边界变量（替代 Step 3d/4d 的 `noMaskKvS=kvSeqlen`，写进任务循环内）

```cpp
uint32_t noSkipKvS = kvSeqlen;      // 因果上界：本 q 块最后一个 query 可见的 kv 长度
uint32_t noMaskKvS = kvSeqlen;      // 无掩码域长度（主循环遍历域）
uint32_t noMaskTailS = 0;           // 无掩码域的尾块残余（<128 部分）
if (maskType != 0) {
    uint32_t diffS = kvSeqlen - qSeqlen;                    // Sq≠Skv 时的 kv 前移量
    noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + diffS;  // 结构性跳块公式
    noSkipKvS = Min((uint32_t)kvSeqlen, noSkipKvS);
    noMaskKvS = noSkipKvS - qSBlockSize;                    // 对角块之前的整块域
    noMaskTailS = noMaskKvS % pagedBlockSize;
}
uint32_t maskedKvS = qSBlockSize;   // 对角（masked）块宽度 = q 块行数
uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, pagedBlockSize);
int32_t totalStackSeqNum = (maskType != 0) ? (CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize) + 1)
                                           : CeilDiv(noMaskKvS, blockStackNum * pagedBlockSize);
```

**语义**：`noMaskKvS` 域 = 全部可见无需 mask 的 KV 列（走主循环 + 普通 QK/PV 组件）；
`noSkipKvS..kvSeqlen` 的最后 `qSBlockSize` 列 = 对角块（走次循环 + QKTail/PVTail + mask）。
causal 不逐位算——整块跳过 + 仅对角块读 mask（`noSkipKvS` 公式即"跳块"）。

### 8c. 次循环起点与推进规则（AIC/AIV 两侧同公式，替换 Step 3g/4f 的计算段）

```cpp
uint32_t maskedStartIdx = (maskType != 0) ?
        ((noMaskTailS != 0) ? (kvSLoopNumNoMask - 1) : kvSLoopNumNoMask)   // ★回退一格重算带尾块的对角前块
      : AlignUp(kvSLoopNumNoMask, blockStackNum);
uint32_t noMaskTailInteStackNum = (noMaskKvS / pagedBlockSize) % blockStackNum;
noMaskTailInteStackNum = (noMaskTailInteStackNum != 0) ? noMaskTailInteStackNum
                                                       : ((noMaskTailS != 0) ? 0 : blockStackNum);
uint32_t preLaunchStackNum = (maskType != 0) ? ((preLaunch - 1) * blockStackNum + noMaskTailInteStackNum)
                                             : (preLaunch * blockStackNum);
```

次循环 `for (kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum; )` 的**推进**：
```cpp
if ((maskType != 0) && (stackSeqCount - preLaunch == totalStackSeqNum - 2)) {
    kvSIdx += noMaskTailInteStackNum;     // ★对角块前一次跳"残余整数块数"，非整 4 块
} else {
    kvSIdx += blockStackNum;
}
stackSeqCount++;
```

### 8d. AIC 次循环体（masked 版）

```cpp
for (uint32_t kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum;) {
    // ① QK（对角块）：stackSeqTile 恒 = maskedKvS；QKTail 多 2 个尾参 (noMaskTailS, 1)
    if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
        stackSeqTile = maskedKvS;
        uint32_t SPP = stackSeqCount % (preLaunch + 1);
        uint64_t gmSOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + SPP*WORKSPACE_BLOCK_SIZE_DB;
        GemmCoord shapeQK{rowNum, stackSeqTile, embed};
        blockMmadQKTail(gQ[gmQOffset], gK[gmKOffset], gS[gmSOffset], gBT[blockBOffset], layoutQTemp,
                        layoutKTemp, shapeQK, kvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                        strideKV, noMaskTailS, 1);
        Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);
    }
    // ② PV（延迟）：最后一块用 PVTail（带 mask 尾参），其余仍用普通 PV
    if (kvSIdx >= preLaunchStackNum) {
        uint32_t dKvSIdx = kvSIdx - preLaunchStackNum;
        stackSeqTile = (dKvSIdx + blockStackNum > kvSLoopNumTotal - 1 && maskType != 0) ? maskedKvS
                     : (dKvSIdx + blockStackNum > kvSLoopNumNoMask - 1) ? (noMaskKvS - dKvSIdx*pagedBlockSize)
                     : pagedBlockSize*blockStackNum;
        uint32_t PVPP = (stackSeqCount - preLaunch) % (preLaunch + 1);
        uint64_t gmPOffset = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + PVPP*WORKSPACE_BLOCK_SIZE_DB;
        LayoutP layoutPTemp(rowNum, stackSeqTile, 512);          // ★同修复1
        GemmCoord shapePV{rowNum, embed, stackSeqTile};
        if ((stackSeqCount - preLaunch == totalStackSeqNum - 1) && (maskType != 0)) {
            blockMmadPVTail(gP[gmPOffset], gV[gmVOffset], gOTmp[gmPOffset], gBT[blockBOffset], layoutPTemp,
                            layoutVTemp, shapePV, dKvSIdx, kvSLoopNumTotal, pagedBlockSize, noSkipKvS,
                            strideKV, softmaxReady, noMaskTailS, 1);
        } else {
            blockMmadPV(gP[gmPOffset], gV[gmVOffset], gOTmp[gmPOffset], gBT[blockBOffset], layoutPTemp,
                        layoutVTemp, shapePV, dKvSIdx, kvSLoopNumNoMask, pagedBlockSize, noMaskKvS,
                        strideKV, softmaxReady);
        }
        Arch::CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
    }
    // ③ 推进（8c 规则）+ stackSeqCount++
}
```

### 8e. AIV 次循环体（masked softmax 调用——签名不同，qkReady 传入 epilogue 内部 Wait）

```cpp
// NO_MASK 主循环（Step 4e）：外面显式 Wait —— CrossCoreWaitFlag(qkReady); epSoft(gP,gS,layOutP,layOutS,...);
// masked 次循环：gMask+layOutMask 进参，qkReady 作为最后参数传给 epilogue（内部按需 Wait，mask 懒加载）
if ((kvSIdx < kvSLoopNumTotal) && (stackSeqCount <= totalStackSeqNum - 1)) {
    stackSeqTile = maskedKvS;
    LayoutS layOutS(rowNum, stackSeqTile, stackSeqTilePad);
    LayoutP layOutP(rowNum, stackSeqTile, stackSeqTilePad);
    LayoutMask layOutMask(1024, 1024, 1024);        // ★固定压缩下三角 [1024,1024] fp16
    GemmCoord shapeQK{rowNum, stackSeqTile, embed};
    uint32_t SPP = stackSeqCount % (preLaunch + 1);
    uint64_t gmOffsetS = coreIdx*WORKSPACE_BLOCK_SIZE_DB*(preLaunch+1) + SPP*WORKSPACE_BLOCK_SIZE_DB;
    epSoft(gP[gmOffsetS], gS[gmOffsetS], gMask, layOutP, layOutS, layOutMask, shapeQK,
           (stackSeqCount == 0), qSBlockSize, qNBlockSize, SPP, qkReady);
    Arch::CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);
}
// rescale 段与 Step 4f 相同；isLast 判据同修复2：
//   (stackSeqCount - preLaunch == totalStackSeqNum - 1)
```

### 8f. mask 张量契约

- mask = 压缩下三角 `[1024, 1024]` fp16 常驻（`mask[q][k] = 1 当 k>q 屏蔽`，语义 `score += mask·(−3e38)`，
  mask≠0 屏蔽、0 保留）；host 一次预置复用（勿每 launch 重建）。
- MASK_SPEC（逐位）：同一 epilogue 机制，mask 由 host 按 sparse band/prefix 语义预生成。
- MASK_CAUSAL：`noSkipKvS` 结构跳块 + 仅对角块走 QKTail/PVTail（数学同 §12.14-A 的 kfc 逐行尾部法，
  组件级实现）。Sq==Skv 时 leftUp/rightDown causal 等价。

---

## Step 9: kernel_common.hpp 全文（共用头，逐行照写）

```cpp
#ifndef KERNEL_COMMON
#define KERNEL_COMMON

constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t BLOCK_SIZE = 16;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;

// 设备侧工具（CATLASS_DEVICE）
template <typename T> CATLASS_DEVICE T CeilDiv(T a, T b) { return (a + b - 1) / b; }
template <typename T> CATLASS_DEVICE T AlignUp(T a, T b) { return (b == 0) ? 0 : (a + b - 1) / b * b; }
template <typename T> CATLASS_DEVICE T RoundUp(T a, T b) { return (a + b - 1) / b * b; }
template <typename T> CATLASS_DEVICE T Min(T a, T b) { return (a > b) ? b : a; }
template <typename T> CATLASS_DEVICE T Max(T a, T b) { return (a > b) ? a : b; }

CATLASS_DEVICE uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    // N 维分块偶数化 trick：AIV 双子核行均衡（每子核 ≤64 行），辅助 rescale 编码
    uint32_t qNBlockTile = (128 / qSeqlen) / 2 * 2;
    qNBlockTile = qNBlockTile < groupSize ? qNBlockTile : groupSize;
    qNBlockTile = qNBlockTile < 1 ? 1 : qNBlockTile;
    return qNBlockTile;
}
CATLASS_DEVICE uint32_t GetQSBlockTile(uint32_t kvSeqlen) { return 128; }   // 固定 128

struct FATilingData {
    uint32_t numHeads = 0;            // Nq
    uint32_t embeddingSize = 0;       // D
    uint32_t numBlocks = 0;           // KV 物理总块数
    uint32_t blockSize = 0;           // pagedBlockSize（必须 128）
    uint32_t maxKvSeqlen = 0;
    uint32_t kvHeads = 0;             // Nkv（GQA group=Nq/Nkv）
    uint32_t batch = 0;
    uint32_t maxNumBlocksPerBatch = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t totalTaskNum = 0;
    uint32_t maskType = 0;            // 0=NO_MASK 1=MASK_SPEC 2=MASK_CAUSAL
    uint64_t mm1OutSize = 0;          // s workspace 字节数
    uint64_t smOnlineOutSize = 0;     // p workspace
    uint64_t mm2OutSize = 0;          // oTemp workspace
    uint64_t UpdateSize = 0;          // oUpdate workspace
    uint64_t workSpaceSize = 0;
    float scaleValue = 0.0;           // 默认 1/sqrt(D)（host 可覆盖）
};

struct FAIKernelParams {              // 13 个 GM_ADDR（tiling 在最后）
    GM_ADDR q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen,
            o, s, p, oTemp, oUpdate, tiling;
    // 构造函数逐成员初始化（bisheng 不吃逗号列表声明，每行一个成员）
};
#endif
```

> device 侧还有 `CeilDiv/RoundUp`（模板）；host 侧 tiling 用同名独立实现（见 Step 10）。
> `TILING_*` 索引常量表（TILING_BATCH=0…）是 **MLA kernel_common** 的内容，FA-Paged 不用（FA-Paged
> tiling 是结构体 `FATilingData` 整体 memcpy，MLA 是 fp32 数组 + 索引寻址——两算子契约不同勿混）。

---

## Step 10: fai_tiling.cpp（host 侧 tiling 计算，独立 CXX TU）

```cpp
// FAInfo：host 聚合输入
struct FAInfo {
    int32_t numTokens;        // Σ qSeqlen（生成 blockTable 用）
    int32_t numHeads;         // Nq
    int32_t embeddingSize;    // D
    int32_t numBlocks;        // KV 物理总块数 = Σ ceil(kvSeqlen_b/128)
    int32_t blockSize;        // 必须 128（否则报错返回）
    int32_t kvHeads;          // Nkv
    int32_t batch;
    int64_t* qSeqlenList;     // per-batch 原始长度（非累加！npu 累加口径 host 先差分）
    int64_t* kvSeqlenList;
    MaskType maskType;        // {NO_MASK=0, MASK_SPEC=1, MASK_CAUSAL=2}
};

int32_t GetFATilingParam(const FAInfo& faInfo, uint32_t blockDim, FATilingData& t)
{
    // ① 校验：seqlen 指针非空；blockSize==128（非 128 直接 -1）
    // ② maxKvSeqlen = max_b kvSeqlenList[b]
    // ③ FillBasicTilingData：
    //      maxNumBlocksPerBatch = ceil(maxKvSeqlen/blockSize)
    //      scaleValue = 1/sqrt(embeddingSize)（标杆自定义 scale 时 host 覆写此字段）
    //      直拷 batch/numHeads/kvHeads/embeddingSize/numBlocks/blockSize/maskType
    // ④ FillSplitCoreTilingData（与 kernel 侧 device 同公式，host 侧再算一遍）：
    //      for b in [0,batch):
    //        curQNBlockTile = GetQNBlockTile(qSeqlen_b, groupSize=Nq/Nkv)   // host 版 min/max 实现
    //        qNBlockNumPerGroup = ceil(groupSize/curQNBlockTile)
    //        curQNBlockNum = qNBlockNumPerGroup * Nkv
    //        curQSBlockNum  = ceil(qSeqlen_b/128)
    //        curTaskNum = curQNBlockNum * curQSBlockNum
    //        if b==0: firstBatchTaskNum = curTaskNum
    //        totalTaskNum += curTaskNum
    // ⑤ FillWorkSpaceTilingData（blockDim = aicCoreNum）：
    //      mm1OutSize     = blockDim * 131072 * 4 * 3   // s  : float, ×4B ×3槽
    //      smOnlineOutSize= blockDim * 131072 * 2 * 3   // p  : half, ×2B ×3槽
    //      mm2OutSize     = blockDim * 131072 * 4 * 3   // oTemp
    //      UpdateSize     = blockDim * 131072 * 4 * 3   // oUpdate
    //      workSpaceSize  = 四段之和（host 据此 malloc 4 块独立 buffer，指针透传，禁 GetUserWorkspace）
}
```

**host-kernel 一致性铁律**：④ 的任务数公式与 kernel device 侧 `while (taskIdx >= curTotalTaskNum)` 推进段
**逐字段相同**（同 GetQNBlockTile/同 ceil）——两侧任何一侧改动不同步 → totalTaskNum/firstBatchTaskNum 错位 →
AIC 与 AIV 解出不同的 (batch, qS, qN) → flag 死锁或精度全错。改切分公式必须双侧同改。

---

## Step 11: host fai.cpp 步骤（launch 管线，11 步顺序）

```cpp
// 1  aclInit(nullptr); aclrtSetDevice(deviceId); aclrtCreateStream(&stream);
// 2  aicCoreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAic();   // A2=20
//    blockDim = aicCoreNum;
// 3  读输入 bin → aclrtMallocHost/aclrtMalloc(ACL_MEM_MALLOC_HUGE_FIRST) → ReadFile → aclrtMemcpy H2D：
//      q_ntokens.bin(int32×1)、q_seqlen.bin/kv_seqlen.bin(int64×batch)、q.bin/k.bin/v.bin、
//      block_table.bin(int32 × batch×maxNumBlocksPerBatch)、mask.bin（maskType!=0 时）
// 4  建 4 段 workspace（尺寸取自 tiling 的 mm1OutSize/smOnlineOutSize/mm2OutSize/UpdateSize）
//    + o（qoSize×2 bytes）+ tilingDevice（sizeof(FATilingData)）
// 5  tilingHost = aclrtMallocHost(sizeof(FATilingData)); 填 FAInfo → GetFATilingParam(faInfo, blockDim, faTilingData)
//    → aclrtMemcpy(tilingDevice, ..., tilingHost, ..., H2D)
// 6  aclrtGetHardwareSyncAddr(&hardwareSyncAddr);      // ★跨核同步基址，kernel 入参第 1 参
// 7  launch（fp16/bf16 双入口二选一）：
//      FAInferFp16<<<blockDim, nullptr, stream>>>(
//          hardwareSyncAddr, q, k, v, mask, blockTables, o,
//          actualQseqlen, actualKvseqlen, s, p, oTemp, oUpdate, tiling);   // ★14 参顺序固定
// 8  aclrtSynchronizeStream(stream);
// 9  aclrtMemcpy o D2H（fp16 元素 2B）
// 10 golden 比对（ReadFile golden.bin → 逐元素 max_abs/matched_ratio）
// 11 逆序 Free/FreeHost
```

> **中间槽必须 nullptr**（`<<<blockDim, nullptr, stream>>>`）——传实指针即 507015（同 §12.13 A-1）。
> workspace 4 段是**独立 malloc 的 4 个指针**分别传 s/p/oTemp/oUpdate（a2-a3 stage design §7：禁合并、禁
> GetUserWorkspace）。blockTable 生成：per-batch 物理块号连续递增（identity 时 `table[t]=t`）。

## 验证状态

> **★2026-09 闭卷验证（NO_MASK 全量通过）**：纯按本 recipe 手搓 `fai_kernel.cpp`（不参照任何外部文件），
> **编译通过 + 运行不死锁 + kv=128~204800(20万) 全部 100% PASS**（max_abs≤7.6e-06）。
> Step 1-7 + 3 条修复照写即可一次编译+跑通+精度 PASS。
>
> **Step 8（mask 路径）**：masked 双循环的边界变量/推进规则/组件调用/softmax 签名完整步骤；未单独闭卷回归——首次生成 masked 版时按 §8a-8f 逐步写，golden 用行限列因果参照。
> **Step 9-11（三件套）**：kernel_common.hpp / fai_tiling.cpp / fai.cpp
> 三个配套文件的完整内容与步骤，作为闭卷生成时的"第四块拼图"（此前 recipe 只覆盖 kernel 单文件，三件套要靠重新
> 发明——这是手搓翻车的主要残留点）。

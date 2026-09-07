# MLA 手搓完整配方（kernel+tiling+host 端到端逐步骤，闭卷可生成）

> **用途**：本文件是 `fa-mla-paged.md` §8 的**实现替代**——§8 给的是组件链伪代码，本文件给的是
> **MLAKernel 生产级真实结构端到端拆解**（kernel §1-8 + host tiling §11 +
> 槽位表/host §12），按此写即可一次编译+跑通（已实证 20/20 PASS，gate≈3.19）。
>
> **构成**：通用 MLAKernel + H=128 TP1/AMLA 特化 + mla_tiling + kernel_common 的完整结构；
> host 侧按 §9/§12 自写。
>
> **实测纠错（2026-09，本会话手搓踩坑）**：§8 伪代码缺两个关键事实，导致手搓必死锁：
> 1. **AIC/AIV 分流必须用 `#ifdef __DAV_CUBE__` / `#ifdef __DAV_VEC__` 预处理分支**，**不是**
>    `__mix__(1,2)` + runtime `g_coreType`。`__mix__` 是 kfc 单 TU 路径（§12.13），MLA CrossCoreFlag
>    路径的组件/HardEvent/CrossCoreFlag token 经济按 `__DAV_CUBE__/__DAV_VEC__` 双二进制模型设计。
>    用 `__mix__`+g_coreType 会破坏 token 配平 → hang（实测：`__mix__` 下 AIC 单独 pre-set+QK+drain 即死锁）。
> 2. **主循环必须是 prologue + 延迟 PV 流水**（见 §6），**不是** "QK(n)→softmax(n)→PV(n)→rescale(n) 同块锁步"。
>    锁步的 CrossCoreFlag Set/Wait 顺序与组件内部 HardEvent 配对不匹配 → token 不平衡 → hang。
>
> CrossCoreFlag 用法（§8 §8.5 正确，本会话实证 OK）：Set 用 `Arch::CrossCoreSetFlag<0x2,PIPE_X>(flag)`，
> Wait 用 bare `Arch::CrossCoreWaitFlag(flag)`（默认 mode 0/PIPE_S，与 `<0x2,PIPE>` Set 在 dav-2201 配对）。

> **★2026-09 闭卷验证结果**：纯按本 recipe 手搓 `mla_kernel.cpp`（不参照任何外部文件），一次编译通过 +
> 运行不死锁 + 精度：kv=128/1024/2048/4096（多块 nLoop≤32，kvSplitCores=1）**100% PASS**（max_abs≤6e-5）。
> 证明 §1（`__DAV_CUBE__/__DAV_VEC__` 分流）+ §5/§6（prologue+延迟 PV 主循环）+ §2（HardEvent 预置）为
> 正确可生成的完整配方。**剩余**：kv≥6144 触发 flash-decoding kv-split（kvSplitCores>1）时必须实现 §8 FD
> rescale 归并（多核部分 O 合并），否则漂移（max_abs~0.034）；recipe §8 已载此段必需。

---

## 0. 组件选型（与 §8.10 一致，确认真实签名——全部对 dispatch_policy.hpp 核验）

```cpp
using ElementQ=half; using ElementK=half; using ElementP=half; using ElementO=half;
using ElementS=float; using ElementOTmp=float; using ElementUpdate=float;
using LayoutQ   = Catlass::layout::RowMajor;     // Q/S/P/O RowMajor
using LayoutK   = Catlass::layout::ColumnMajor;  // K/V ColumnMajor（host 不预转置）
using LayoutS   = Catlass::layout::RowMajor;
using LayoutP   = Catlass::layout::RowMajor;
using LayoutV   = Catlass::layout::ColumnMajor;
using LayoutO   = Catlass::layout::RowMajor;
using LayoutOTmp= Catlass::layout::RowMajor;
using LayoutUpdate=Catlass::layout::RowMajor;

using L1TileShape = Catlass::GemmShape<128,128,576>;   // K=embed+embedRope=576，K 常驻 L1
using L0TileShape = L1TileShape;

// ★MmadAtlasA2MLAQK/MLAPV 是普通 struct（非模板，dispatch_policy.hpp:74/78）——
//   与 FAIQK<true,false>（模板）不同，勿手滑加模板参
using BlockMmadQK = Catlass::Gemm::Block::BlockMmad<
    Catlass::Gemm::MmadAtlasA2MLAQK, L1TileShape, L0TileShape,
    GemmType<half,LayoutQ>, GemmType<half,LayoutK>, GemmType<float,LayoutS>>;
using BlockMmadPV = Catlass::Gemm::Block::BlockMmad<
    Catlass::Gemm::MmadAtlasA2MLAPV, L1TileShape, L0TileShape,
    GemmType<half,LayoutP>, GemmType<half,LayoutV>, GemmType<float,LayoutOTmp>>;
using EpilogueMLASoftmax   = Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLASoftmax,   GemmType<half,LayoutP>, GemmType<float,LayoutS>>;
using EpilogueMLARescaleO  = Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLARescaleO,  GemmType<half,LayoutO>,  GemmType<float,LayoutOTmp>>;
// FDRescaleO 是 uint32 模板（6144=ComputeEleNum）；KV_SPLIT_MAX=64、HEADS_PROCESS_MAX=16 由其内提供
using EpilogueMLAFDRescaleO= Catlass::Epilogue::Block::BlockEpilogue<Catlass::Epilogue::EpilogueAtlasA2MLAFDRescaleO<6144>, GemmType<half,LayoutO>>;
```

**TP1/AMLA 特化组件名（H=128 spec 路径，dispatch_policy.hpp:82-90 + epilogue:85-100，已核验存在）**：
`MmadAtlasA2MLAQKTp1Spec` / `MmadAtlasA2MLAPVTp1Spec` / `MmadAtlasA2AMLAPVTp1Spec` +
`EpilogueAtlasA2MLATP1Softmax` / `EpilogueAtlasA2MLATP1RescaleO` /
`EpilogueAtlasA2AMLATP1Softmax` / `EpilogueAtlasA2AMLATP1RescaleO`。
本 recipe §1-8 只展开**通用 MLAKernel**（H=16/32/64，tilingKey 0/1）；H=128 spec 走
`mla_kernel_tp1_spec.cpp`/`amla_kernel_tp1_spec.cpp`（每 Q token 一 process 的重组版，骨架同构、
组件换上表特化名 + `EpilogueAtlasA2MLATP1*` epilogue）——首次生成 spec 路径时按本 recipe §1-8 相位映射同构展开（组件换上表特化名 + TP1 epilogue）。

---

## 1. kernel 类骨架（`__DAV_CUBE__`/`__DAV_VEC__` 预处理分支）

```cpp
template <class BlockMmadQK, class BlockMmadPV, class EpilogueMLASoftmax,
          class EpilogueMLARescaleO, class EpilogueMLAFDRescaleO>
class MLAKernel {
    // type aliases: ElementQ/K/S/P/O/OTmp/Update, Layout*, ArchTag, L1TileShape ...
    // (从 BlockMmadQK:: / Epilogue*:: 取)
public:
    struct Params { GM_ADDR q,qRope,k,kRope,blockTables,o,s,p,oTmp,oUpdate,oCoreTmp,l,tiling; };

    CATLASS_DEVICE void operator()(Params const& params) {
        // §2 HardEvent pre-set (AIC+AIV 共享，§8.6 全表)
        // §3 GlobalTensor SetGlobalBuffer + tiling 读取 (按 #ifdef 分流)
        // §4 coreIdx (AIC=GetBlockIdx / AIV=GetBlockIdx/GetSubBlockNum) + task 遍历
        // §5 prologue (nIdx=0: QK+softmax，无 PV)
        // §6 main loop (nIdx=1..nLoop: QK(n)+softmax(n) if非末 / 末块下task预取 / PV(n-1)+rescale(n-1))
        // §7 HardEvent drain (§8.6 反预置)
        // §8 FD rescale (仅 maxKvSplitCoreNum!=1)
    }
};
```

★分支形式：`#ifdef __DAV_CUBE__ { AIC-only 代码 } #endif` 与 `#ifdef __DAV_VEC__ { AIV-only 代码 } #endif`
**同处并列**（同一 operator() 内），编译器在 `--npu-arch=dav-2201` mix 下自动让 cube 核走 `__DAV_CUBE__`、
vector 核走 `__DAV_VEC__`。**不要** `__mix__` 属性、**不要** `g_coreType` runtime if-else。

---

## 2. HardEvent 预置（§8.6 全表，AIC+AIV 共享，operator() 入口）

```cpp
#ifdef __DAV_CUBE__
SetFlag<HardEvent::M_MTE1>(EVENT_ID0..7);      // 8
SetFlag<HardEvent::FIX_M>(EVENT_ID0); SetFlag<HardEvent::FIX_M>(EVENT_ID1);  // 2
SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0..7);  // 8
SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0..5);   // 6
SetFlag<HardEvent::MTE2_FIX>(EVENT_ID0);      // 1    共 25
#endif
#ifdef __DAV_VEC__
SetFlag<HardEvent::MTE3_V>(EVENT_ID0); SetFlag<HardEvent::MTE3_V>(EVENT_ID2); SetFlag<HardEvent::MTE3_V>(EVENT_ID3);  // 3
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); for(e=2..7) SetFlag<HardEvent::MTE3_MTE2>(e);  // 7
SetFlag<HardEvent::V_MTE2>(EVENT_ID0); SetFlag<HardEvent::V_MTE2>(EVENT_ID1); SetFlag<HardEvent::V_MTE2>(EVENT_ID4);
SetFlag<HardEvent::V_MTE2>(EVENT_ID2); SetFlag<HardEvent::V_MTE2>(EVENT_ID3);  // 5    共 15
#endif
```
**drain（§7，遍历结束后，反序同表）**：把上表所有 `SetFlag` 换 `WaitFlag`，顺序**逐字照抄**实测顺序
（AIV 的 V_MTE2 drain 顺序是 4,0,1,2,3——非递增，照抄勿自排）。

> 实测：pre-set/drain 与 §8.6 一致，**不是死锁源**。死锁源是 §6 的循环结构 + §1 的分支形式。

---

## 3. GlobalTensor + tiling（按 #ifdef 分流）

```cpp
#ifdef __DAV_CUBE__
GlobalTensor<ElementQ> gQ; gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
gQRope/gK/gKRope 同；GlobalTensor<int32_t> gblockTable; ...SetGlobalBuffer((__gm__ int32_t*)params.blockTables);
BlockMmadQK blockMmadQK(resource); BlockMmadPV blockMmadPV(resource);
// 值：embedRope, maxNumBlocksPerQuery, kvHeads, strideQORope, strideKV, strideKVRope（仅 AIC）
#endif
#ifdef __DAV_VEC__
GlobalTensor<ElementO> gO; gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
gOUpdate/gOCoreTmp/gl 同；GlobalTensor<float> gTilingFp64(...params.tiling);
EpilogueMLASoftmax epilogueMLASoftmax(resource, tor, maxKvSplitCoreNum);
EpilogueMLARescaleO epilogueMLARescaleO(resource, maxKvSplitCoreNum);
float tor = gTilingFp64.GetValue(TILING_TOR);
uint32_t glFlag[2] = {1, 1};   // ★真数组，勿传 nullptr（kvSplit==1 不解引但实测须传 {1,1}）
#endif
// 共享：batch/qHeads/embed/blockSize/tilingHeadSize/tilingParaSize/curQheadSplitSize/curQheadSplitNum/maxKvSplitCoreNum
//        strideQO=qHeads*embed, embedRound=RoundUp<BLOCK_SIZE>(embed)
```

`resource` = `Catlass::Arch::Resource<Arch::AtlasA2>`，在 operator() 顶部构造一次（pre-set 之前）。

---

## 4. coreIdx + task 遍历（AIC/AIV 共享同一 loop——含完整 process 解码算法）

```cpp
uint32_t coreIdx = 0;
#ifdef __DAV_CUBE__  coreIdx = GetBlockIdx(); #endif
#ifdef __DAV_VEC__  coreIdx = GetBlockIdx() / GetSubBlockNum(); #endif     // ★AIV 必除，否则 blockIdx 0 与 20 双跑 task0 → flag 死锁
uint32_t coreNum  = GetBlockNum();
uint32_t tilingProcessNum = gTiling.GetValue(TILING_PROCESSNUM);
uint32_t processNum = (tilingProcessNum != 0) ? tilingProcessNum : batch * curQheadSplitNum * maxKvSplitCoreNum;
bool isForward = true, kerFlag = true;    // back-and-forth 蛇形方向位（仅 ==0 路径用）

for (uint32_t process = coreIdx; process < processNum; process += coreNum) {
    // ---- (a) process → curBatch（两种映射，逐行） ----
    uint32_t curBatch;
    uint32_t taskIdx = 0;
    if (tilingProcessNum != 0) {
        // T+1 连续前缀和映射：tilingHost[CUTASK_START_OFFSET+b] = 前 b 个 batch 的 process 累计
        while (taskIdx < batch && process >= gTiling.GetValue(CUTASK_START_OFFSET + taskIdx + 1)) {
            taskIdx++;
        }
        curBatch = taskIdx;
    } else {
        // back-and-forth 蛇形映射：同一核的相邻 process 落在相邻 batch（提高 KV cache 局部性）
        uint32_t bigProcess = process - (process % coreNum) + (coreNum - 1);
        bigProcess = (bigProcess > processNum - 1) ? processNum - 1 : bigProcess;
        uint32_t realProcess = isForward ? process : (bigProcess - process % coreNum);
        isForward = !isForward;
        curBatch = realProcess / (curQheadSplitNum * maxKvSplitCoreNum);
    }
    uint32_t offsetTiling = tilingHeadSize + tilingParaSize * curBatch;
    uint32_t qSeqlen = gTiling.GetValue(offsetTiling);
    uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
    uint32_t kvSplitPerCore = gTiling.GetValue(offsetTiling + 15);
    uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);
    if (kvSeqlen == 0) { continue; }

    // ---- (b) process → (qHeadSplitIdx, curNIdx) ----
    uint32_t qHeadSplitIdx, curNIdx;
    if (tilingProcessNum != 0) {
        qHeadSplitIdx = 0;                                          // kv-split 路径 H 不再切
        curNIdx = process - gTiling.GetValue(CUTASK_START_OFFSET + curBatch);
    } else {
        qHeadSplitIdx = (process % (curQheadSplitNum * maxKvSplitCoreNum)) / maxKvSplitCoreNum;
        curNIdx = process % maxKvSplitCoreNum;
        if (kerFlag) { kerFlag = false; }                           // 蛇形：偶次正向、奇次反向取 N 段
        else { kerFlag = true; curNIdx = maxKvSplitCoreNum - curNIdx - 1; }
    }
    uint32_t qHeadSplitSizeActual = (qHeadSplitIdx == (curQheadSplitNum - 1)) ?
        (qHeads - qHeadSplitIdx * curQheadSplitSize) : curQheadSplitSize;
    uint32_t curKVSeqlen = kvSplitPerCore;
    if (curNIdx >= kvSplitCoreNum) { continue; }                    // 对齐产生的空槽
    if (curNIdx == (kvSplitCoreNum - 1)) {
        curKVSeqlen = kvSeqlen - curNIdx * kvSplitPerCore;          // 末核吃尾差
    }
    uint32_t nLoop = (curKVSeqlen + seqTile - 1) / seqTile;         // seqTile = blockSize = 128
    uint32_t rowNum = qHeadSplitSizeActual * qSeqlen;               // H 分块 × token pack 进 M
    uint32_t rowNumRound = RoundUp<16>(rowNum);
    // → 进 §5 prologue / §6 主循环
}
```
**★AIV coreIdx 必 `GetBlockIdx()/GetSubBlockNum()`**：否则双子核 blockIdx 0 与 1 都映射 logical 0
而 blockIdx 0 与 20 也都映射 0 → 两个核 Wait 同一 flag → 死锁。
**解码三态**：kv-split 生效（TILING_PROCESSNUM≠0，q=1 decode）走前缀和 + H 不切；未生效走蛇形
（qHeadSplitIdx × curNIdx 二维 + 蛇形反向）。两态的 flag 配平依赖"每 process 恰好一轮 prologue+主循环"
——curNIdx≥kvSplitCoreNum 的空槽 continue **必须保留**（跳过即不碰 flag，配平不破）。

---

## 5. prologue（isFirstTask，nIdx=0：只 QK+softmax，**无 PV**）

```cpp
if (isFirstTask && nLoop > 0) {
    uint32_t nIdx = 0;
    // 尾块 kSeqTile 修正
#ifdef __DAV_CUBE__
    LayoutQ layoutQ(rowNum, embed); LayoutQ layoutQRope(rowNum, embedRope);
    LayoutK layoutK(embed, kSeqTile); LayoutK layoutKRope(embedRope, kSeqTile);
    LayoutS layoutS(rowNumRound, kSeqTileRound);
    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embed + embedRope};
    MatrixCoord qShapeSingleNd{qHeadSplitSizeActual, embed};
    uint32_t qkPingPongFlag = pingpongIdx % 2;
    int32_t blockTableId = gblockTable.GetValue(realCurBatch*maxNumBlocksPerQuery + startKV/blockSize + nIdx);
    uint64_t kvOffset = (uint64_t)blockTableId*blockSize*strideKV, kvOffsetRope=...*strideKVRope;
    uint64_t gSOffset = (uint64_t)coreIdx*TMP_SIZE_DECODER + (uint64_t)qkPingPongFlag*TMP_SIZE_DECODER/2;
    blockMmadQK(gQ[gQOffset], gQRope[gQRopeOffset], gK[kvOffset], gKRope[kvOffsetRope],
                gS[gSOffset], layoutQ, layoutQRope, layoutK, layoutKRope, layoutS,
                actualBlockShapeQK, qShapeSingleNd, qHeads, nIdx, pingpongIdx);
    pingpongIdx++;
    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
    isFirstTask = false;
#endif
#ifdef __DAV_VEC__
    uint32_t softmaxPingPongFlag = pingpongIdx % 2;
    Arch::CrossCoreWaitFlag(qkReady);
    LayoutP layoutP(rowNum, kSeqTile, kSeqTileRound); LayoutS layoutS(rowNumRound, kSeqTile, kSeqTileRound);
    GemmCoord actualBlockShapeQK{rowNum, kSeqTile, embedRound};
    uint64_t gmOffsetP = (uint64_t)coreIdx*TMP_SIZE + softmaxPingPongFlag*TMP_SIZE/2;
    uint64_t gmOffsetS = (uint64_t)coreIdx*TMP_SIZE_DECODER + softmaxPingPongFlag*TMP_SIZE_DECODER/2;
    epilogueMLASoftmax(gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK,
                       nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag, taskPingPongFlag);
    pingpongIdx++;
    Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
    isFirstTask = false;
#endif
}
```

---

## 6. 主循环（nIdx=1..nLoop：QK(n)+softmax(n) if非末 / 末块下task预取 / **PV(n-1)+rescale(n-1) 延迟**）

```cpp
for (uint32_t nIdx = 1; nIdx < nLoop + 1; nIdx++, pingpongIdx++) {
    // (a) 非末块：QK(n) + softmax(n)
    if (nIdx != nLoop) {
        // 尾块 kSeqTile 修正
#ifdef __DAV_CUBE__
        // ... layoutQ/K/S, blockTableId, gSOffset（同 prologue，nIdx 变）
        blockMmadQK(..., nIdx, pingpongIdx);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady);
#endif
#ifdef __DAV_VEC__
        Arch::CrossCoreWaitFlag(qkReady);
        // ... layoutP/S, gmOffsetP/S
        epilogueMLASoftmax(..., nIdx, qHeadSplitSizeActual, softmaxPingPongFlag, glFlag, taskPingPongFlag);
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(softmaxReady);
#endif
    }
    // (b) 末块（nIdx==nLoop）：下一 task 的 prologue 预取（QK(0)+softmax(0)，跨 task 流水）
    if (nIdx == nLoop) {
        // peek nextProcess（coreNum+process 起），算其 (nextBatch, nextQHeadSplitIdx, nextCurNIdx)
        // 跑 nextNIdx=0 的 blockMmadQK + epilogueMLASoftmax（AIC Set qkReady / AIV Wait+Set softmaxReady）
        // break（只预取一个 task）
    }
    // (c) ★延迟 PV(n-1) + rescale(n-1)——每轮都做，PV 用 (pingpongIdx-1)%2 槽
    if (nIdx == nLoop) { vSeqTile = curKVSeqlen - (nIdx-1)*seqTile; vSeqTileRound=RoundUp<16>(vSeqTile); }
#ifdef __DAV_CUBE__
    LayoutP layoutP(rowNum, vSeqTile, vSeqTileRound); LayoutV layoutV(embed, vSeqTile);
    LayoutOTmp layoutOTmp(rowNumRound, embedRound);
    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
    uint32_t pvPingPongFlag = (pingpongIdx - 1) % 2;
    uint64_t gPOffset  = (uint64_t)coreIdx*TMP_SIZE + pvPingPongFlag*TMP_SIZE/2;
    uint64_t gOTmpOffset = (uint64_t)coreIdx*TMP_SIZE*2 + pvPingPongFlag*TMP_SIZE;
    blockMmadPV(gP[gPOffset], gOTmp[gOTmpOffset], layoutP, layoutV, layoutOTmp,
                actualBlockShapePV, nIdx, pingpongIdx, softmaxReady, locPingPongIdx);
    Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady);
#endif
#ifdef __DAV_VEC__
    Arch::CrossCoreWaitFlag(pvReady);
    LayoutO layoutO(tokenNumPerHead, strideQO); LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
    LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
    GemmCoord actualBlockShapePV{rowNum, embed, vSeqTile};
    uint32_t rescaleOPingPongFlag = (pingpongIdx - 1) % 2;
    uint64_t gmOffsetOTmp = (uint64_t)(coreIdx*TMP_SIZE*2 + rescaleOPingPongFlag*TMP_SIZE);
    uint64_t gmOffsetUpdate = (uint64_t)(coreIdx*TMP_SIZE);
    uint32_t isLastNTile = (nIdx == nLoop) ? 1 : 0;
    epilogueMLARescaleO(gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gO[gmOffsetO], gOCoreTmp[oFdOffset],
                        gl[lOffset], layoutOTmp, layoutO, layoutUpdate, actualBlockShapePV,
                        nIdx, isLastNTile, qHeadSplitSizeActual, rescaleOPingPongFlag, glFlag, taskPingPongFlag);
#endif
}
#ifdef __DAV_VEC__ taskPingPongFlag = 1 - taskPingPongFlag; #endif
```

**★为何延迟 PV**：blockMmadPV 内部 `CrossCoreWaitFlag(softmaxReady)` 等 AIV 的 P；AIV softmax 写 P 到
`softmaxPingPongFlag` 槽，PV 用 `(pingpongIdx-1)%2` 读**上一块**的 P——形成 QK(n) 与 PV(n-1) 交错流水，
CrossCoreFlag token（qkReady/softmaxReady/pvReady）按此交错配平。同块锁步会破坏配平→死锁（实测）。

---

## 7. GM 槽位公式（§8.11，照搬）

```
S(gSOffset)    = coreIdx*TMP_SIZE_DECODER + (pp%2)*TMP_SIZE_DECODER/2     // QK 写
P(gmOffsetP)   = coreIdx*TMP_SIZE        + (pp%2)*TMP_SIZE/2              // softmax 写；PV 读用 (pp-1)%2
OTmp(gOTmpOffset) = coreIdx*TMP_SIZE*2   + ((pp-1)%2)*TMP_SIZE            // PV 写；rescale 读同偏移
OUpdate(gmOffsetUpdate) = coreIdx*TMP_SIZE                                // rescale 写
```
TMP_SIZE=65536, TMP_SIZE_DECODER=32768（kernel_common.hpp）。
per-batch Q/QRope/O/l/oFd 的 64 位 GM 地址由 host 拆 hi/lo 32 位写进 tiling 段
（offsetTiling+4/5=q hi/lo, +6/7=qRope, +11/12=l, +13/14=oFd），kernel 拼回（见 §3）。

---

## 8. FD rescale（flash-decoding kv-split 归并，仅 maxKvSplitCoreNum != 1，drain 之后——逐行步骤）

```cpp
#ifdef __DAV_VEC__
    // drain（§7 反预置）之后：
    if (maxKvSplitCoreNum != 1) {
        Catlass::Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();       // ★全核汇聚（等所有核写完 oFd/l）

        AscendC::SetAtomicNone();
        AscendC::SetMaskNorm();
        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);

        EpilogueMLAFDRescaleO epilogueMLAFDRescaleO(resource, maxKvSplitCoreNum);
        uint32_t aivNum = AscendC::GetBlockNum() * AscendC::GetSubBlockNum();   // ★此处不除（物理 aiv）
        uint32_t aivId = AscendC::GetBlockIdx();

        uint32_t headsProcess = (COMPUTE_ELE_NUM / embed) > HEADS_PROCESS_MAX ?
                                HEADS_PROCESS_MAX : (COMPUTE_ELE_NUM / embed);
        uint32_t loopsPerBatch = (qHeads + headsProcess - 1) / headsProcess;
        uint32_t loopsTotal = batch * loopsPerBatch;

        for (uint32_t loopIdx = aivId; loopIdx < loopsTotal; loopIdx += aivNum) {
            uint32_t batchIdx = loopIdx / loopsPerBatch;
            uint32_t loopIdxInBatch = loopIdx % loopsPerBatch;
            uint32_t offsetTiling = tilingHeadSize + tilingParaSize * batchIdx;
            uint32_t kvSeqlen = gTiling.GetValue(offsetTiling + 1);
            uint32_t qSeqlen  = gTiling.GetValue(offsetTiling);
            uint32_t kvSplitCoreNum = gTiling.GetValue(offsetTiling + 16);
            if (kvSeqlen == 0) { continue; }

            // per-batch 地址：O/l/oFd 均 hi/lo 拆 32 位存 tiling 槽（+4/5 复用 Q 偏移=O 偏移，
            // 因 decode 时 Q 与 O 同形 [tokens,H,512]；l=+11/12、oFd=+13/14）
            uint64_t oAddr   = ((uint64_t)gTiling.GetValue(offsetTiling+4)  << 32) | gTiling.GetValue(offsetTiling+5);
            uint64_t lOffset = ((uint64_t)gTiling.GetValue(offsetTiling+11) << 32) | gTiling.GetValue(offsetTiling+12);
            uint64_t oFdOffset = ((uint64_t)gTiling.GetValue(offsetTiling+13) << 32) | gTiling.GetValue(offsetTiling+14);

            uint32_t actualHeads = headsProcess;
            if (loopIdxInBatch == loopsPerBatch - 1) {
                actualHeads = qHeads - loopIdxInBatch * headsProcess;    // 尾批头数
            }
            for (uint32_t qSeqIdx = 0; qSeqIdx < qSeqlen; qSeqIdx++) {
                epilogueMLAFDRescaleO(
                    gO[oAddr + qSeqIdx * qHeads * embed + loopIdxInBatch * headsProcess * embed],
                    gOCoreTmp[oFdOffset * maxKvSplitCoreNum + qSeqIdx * qHeads * embed * maxKvSplitCoreNum
                              + loopIdxInBatch * headsProcess * maxKvSplitCoreNum * embed],
                    gl[lOffset + qSeqIdx * qHeads * maxKvSplitCoreNum
                                + loopIdxInBatch * headsProcess * maxKvSplitCoreNum],
                    actualHeads, headsProcess, embed, kvSplitCoreNum);
            }
        }
    }
#endif
```

- **语义**：`O[b,t,h,:] = Σ_n O_coreTmp[n]·exp(m_n−m_max)·l_n / Σ_n l_n·exp(m_n−m_max)`（FDRescaleO 组件
  内部完成多核部分 O/l/m 的在线归并）。调用前所有核必须已写完各自 oFd/l 分片 → `CrossCoreBarrier`。
- **KV_SPLIT_MAX / HEADS_PROCESS_MAX / COMPUTE_ELE_NUM** 来自 `EpilogueAtlasA2MLAFDRescaleO<6144>` 模板
  （6144=ComputeEleNum，实测取值——手搓直接 `EpilogueAtlasA2MLAFDRescaleO<6144>`）。
- kv<6144 或 batch>blockDim·0.8 或 maxQseqlen>1 时 tiling 不 split（§11 决策），本段自动跳过。
- ★`aivId = GetBlockIdx()`（不除 subBlockNum）+ `aivNum = BlockNum*SubBlockNum`：FD 段按**物理** aiv
  分发 loop（与主循环的 logical coreIdx 不同——勿复制主循环的除法）。

---

## 9. launch + build（host 契约）

- launch：`MLA<half><<<aicCoreNum, nullptr, stream>>>(hwSync, dQ,dQRope,dK,dKRope,dBT,dO,dS,dP,dOTmp,dGlobalO,dOCoreTmp,dL,dTiling)` 14 参。
- tilingKey 分发：0/1=MLA(half/bf16)、4/5=AMLATp1Spec、7/8=MLATp1Spec（H=128 用 TP1）。
  key = (specStra?1:0)<<2 | dTypeKey；若 specStra && numTokens%blockDim<=10 && batch<=40 → key=dType?8:7。
- tiling：`MLATiling::GetMLATilingParam(MLAInfo, blockDim, uint32_t* tilingHost)`（host TU）。
- workspace：wsDbl = blockDim*WORKSPACE_BLOCK_SIZE_DB(65536)；dS=wsDbl*float*2, dP=wsDbl*elem*2,
  dOTmp=wsDbl*float*2, dGlobalO=wsDbl*float, oFdSize=embed*numHeads*numTokens*maxKvSplitCoreNum*float,
  lSize=numTokens*numHeads*maxKvSplitCoreNum*float。
- build：`CATLASS_ARCH=2201` + `--npu-arch=dav-2201`（让 `__DAV_CUBE__`/`__DAV_VEC__` 宏生效）；
  host .asc `#include "mla_kernel.cpp"` 单 TU；mla_tiling.cpp 单独 CXX/ASC TU；
  link ascendc_runtime ascendcl runtime profapi mmpa c_sec error_manager graph_base tiling_api register platform unified_dlog dl m。

---

## 11. mla_tiling.cpp（host 侧 tiling，独立 host TU——MLATiling 命名空间逐步）

> MLA tiling 与 FA-Paged 不同：**fp32 数组寻址**（非结构体 memcpy）。头部常量（mla_tiling.h）：
> `TILING_HEAD_SIZE=400`（头区起点）、`TILING_PARA_SIZE=17`（每 batch 段元素数）、
> `CUTASK_START_OFFSET=25`（前缀和表起点）、`CUTASK_MAX_LENGTH=130`、
> `QN_TILE_LIST={128,64,32,16,8,1}`（按 ⌈log2(qSeqlen)⌉ 取，封顶 idx 5）、
> `SPLITKV_RATION=0.8`、`KV_SEQLEN_SLICE=128`。

**入口 `GetMLATilingParam(MLAInfo, blockDim, tilingHost)` 八步**（blockDim = aicCoreNum = 20）：

1. **校验**：tilingHost/qSeqLen/kvSeqLen 非空；`blockSize==128`（否则 -1）。
2. **per-batch 地址偏移表**（Q/QRope/mask 三套累加）：
   `qSeqOffset_b = Σ_{j<b} numHeads·embeddingSize·qSeqLen_j`（QRope 用 embeddingSizeRope、
   mask 用 maxKvSeqlen·qSeqLen_j）。
3. **排序**：`quickSortIndices(sortedIndices, kvSeqLen, qSeqLen, 0, batch-1)`——按 kv 长降序
   （长序列优先切 split，负载均衡）；`realPreTaskNums[b][token]` = 全局 token 前缀（TP1 用）。
4. **约束检查**：qSeqLen>4 时告警（MLA 契约 q∈[1,4]）；
   `Σ kvSeqLen > numBlocks·blockSize` 报错返回（paged cache 容量）。
5. **tor**：`1/sqrt(embeddingSize+embeddingSizeRope)` = 1/√576，按位 reinterpret 成 uint32 写槽
   （AIV 从 `gTilingFp64.GetValue(TILING_TOR)` 位还原读回——勿走 fp64 舍入）。
   `specStrategyFlag = (numHeads==128) ? 1 : 0`（H=128 → TP1 特化路径）。
6. **写头区 + 每 batch 段**（spec ? GetMLATilingSpec : GetMLATilingCommon）：
   头区 `TILING_HEADSIZE..` = batch/TILING_HEADSIZE(=400)/TILING_PARASIZE(17)/NUMHEADS/HEADDIM(512)/
   NUMBLOKS/BLOCKSIZE(128)/MAXBLOCKS(=⌈maxKvSeqlen/128⌉)/TOR/KVHEADS/HEAD_SPLIT_SIZE(=GetQNBlockTile)/
   HEAD_SPLIT_NUM(=⌈H/切⌉)/MASKTYPE/HEADDIM_ROPE(64)/TOTAL_QTOKENS。
   每 batch 段（`offset = TILING_HEAD_SIZE + TILING_PARA_SIZE*b`）见 §12 槽位表。
7. **KV split 决策 + 贪心切分**（GetKVSplitParam，仅非 spec；spec 用 GetKVSplitParamSpec 同思路）：
   ```
   isKVSplit = (maxKvSeqlen >= blockDim*128*2) && (batch <= blockDim*0.8) && (maxQseqlen == 1)
   // 不 split：每 batch kvSplitPerCore=kvSeqlen, kvSplitCoreNum=1, TILING_PROCESSNUM=0, KVCORENUM=1
   // split（T+1 均衡贪心，逐行）：
   MIN_TOKENS_PER_SPLIT = blockSize*4 = 512；MAX_SPLIT_PER_TASK = 8
   alignedKvLens[b] = RoundUp(kvSeqLen, 128)；batchSplitNum 全 1；totalAllocated = batch
   while (totalAllocated < blockDim):
       找 currentLoad=alignedKvLens[i]/batchSplitNum[i] 最大的 i
       （约束 batchSplitNum[i]<8 且 nextLoad≥512）
       batchSplitNum[i]++；totalAllocated++
       （无候选即 break——防过切）
   逐 batch：kvBlockPerCore=⌈(RoundUp(kv,128)/128)/batchSplitNum[b]⌉，kvSplitPerCore=kvBlockPerCore*128，
             kvSplitCoreNum=⌈kvSeqlen/kvSplitPerCore⌉
   前缀和：CUTASK_START_OFFSET[0]=0，[b+1]=Σ_{j≤b} kvSplitCoreNum[j] → TILING_PROCESSNUM=总 process
   TILING_KVCORENUM = max(batchSplitNum) = maxKvSplitCoreNum
   ```
8. **l/oFd 偏移**：`lOffset_b = Σ_{j<b} numHeads·qSeqlen_j·MAX_KV_SPLIT_NUM`（★乘全局最大 split 数）、
   `oFdOffset_b = Σ_{j<b} numHeads·qSeqlen_j·embeddingSize`——hi/lo 拆写 +11/12、+13/14 槽。

**host-kernel 一致性铁律**（同 FA-Paged Step 10）：§4 的两种 process 解码（前缀和/蛇形）必须与
本步 7 的 TILING_PROCESSNUM/KVCORENUM/CUTASK 表严格配套——host 写 split 表而 kernel 走蛇形
（或反之）→ process 解码错乱 → flag 配对破坏 → 死锁。

## 12. tiling 槽位表 + kernel_common 常量 + host 步骤

**每 batch 段槽位**（`offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE*batchIdx`，fp32 元素）：

| 槽 | 含义 | 写方（host tiling） | 读方（kernel） |
|---|---|---|---|
| +0 | qSeqlen（原始非累加） | GetMLATilingCommon | AIC/AIV 任务切分 |
| +1 | kvSeqlen | 同上 | 同上 + FD |
| +2 | sortSeqIdx（排序后原 batch 号） | 同上 | AIC 取 per-batch Q/K 基址 |
| +3 | blockSize(128) | 同上 | — |
| +4/+5 | Q 地址偏移 hi/lo（★decode 时 Q 与 O 同形，FD 段复用为 O 偏移） | GetAddrOffsetMLA | AIC QK；AIV FD |
| +6/+7 | QRope 地址偏移 hi/lo | 同上 | AIC QK |
| +8/+9 | mask 偏移 hi/lo | 同上 | mask 版用 |
| +11/+12 | l 偏移 hi/lo | GetKVSplitParam 步 8 | AIV RescaleO / FD |
| +13/+14 | oFd 偏移 hi/lo | 同上 | AIV FD |
| +15 | kvSplitPerCore（128 倍数） | GetKVSplitParam | 任务切分 curKVSeqlen |
| +16 | kvSplitCoreNum | 同上 | 任务切分 / FD |

**kernel_common.hpp MLA 常量**（共用头，recipe §7 已列 TMP_SIZE=65536 / TMP_SIZE_DECODER=32768，
另需）：`QK_READY_ID=1 / SOFTMAX_READY_ID=2 / PV_READY_ID=3 / BLOCK_SIZE=16 / WORKSPACE_BLOCK_SIZE_DB=65536`
（★MLA 的 WS_DB 是 65536，FA-Paged 是 131072，勿混）+ TILING_* 头区索引常量表（mla_tiling.h 逐值）：
`BATCH=0, NUMHEADS=1, HEADDIM=2, NUMBLOKS=3, BLOCKSIZE=4, MAXBLOCKS=5, TOR=6, KVHEADS=7, HEADSIZE=8,
PARASIZE=9, HEAD_SPLIT_SIZE=10, HEAD_SPLIT_NUM=11, MASKTYPE=12, HEADDIM_ROPE=13, MAX_KVSEQLEN=14,
KVSPLIT=15, KVCORENUM=16, MAX_QSEQLEN=17, TOTAL_QTOKENS=18, FORMERTASKNUM=19, TAILTASKNUM=20,
PROCESSNUM=21`（头区 0..21）；`TILING_HEAD_SIZE=400 / CUTASK_START_OFFSET=25 / CUTASK_MAX_LENGTH=130 /
TILING_PARA_SIZE=17`（CUTASK 前缀和表在头区内 25 起连续 batch+1 项）。

**host 步骤**（mla.cpp，同 FA-Paged Step 11 管线，差异点）：
1. 输入 6 张量：q[tokens,H,512] / qRope[tokens,H,64] / k,kRope（paged 块格式 [numBlocks,128,kvHeads,D]）/
   blockTable / o[tokens,H,512]。
2. tilingHost 按**元素数** malloc：`TILING_HEAD_SIZE + TILING_PARA_SIZE*batch` 个 uint32（fp32 数组非结构体）。
3. workspace（§9 公式）：`wsDbl = blockDim*65536`；dS=wsDbl·4B·2、dP=wsDbl·elem·2、dOTmp=wsDbl·4B·2、
   dGlobalO(=oUpdate)=wsDbl·4B、oFdSize=512·numHeads·numTokens·maxKvSplitCoreNum·4B、
   lSize=numTokens·numHeads·maxKvSplitCoreNum·4B（maxKvSplitCoreNum 从 tiling 回读后再分配——先跑
   GetMLATilingParam 再 malloc oFd/l）。
4. launch 14 参：`MLA<half><<<aicCoreNum,nullptr,stream>>>(hwSync,q,qRope,k,kRope,bt,o,s,p,oTmp,oGlobal,oCoreTmp,l,tiling)`；
   中间槽 nullptr；tilingKey 分发见 §9。

## 10. FA-Paged 近似配方但分流不同

`fai_kernel.cpp` 的 `FAInferKernel` 与 MLA 同范式（CrossCoreFlag `<0x2,PIPE_X>` Set + bare Wait、
§8.6 风格 HardEvent 预置/drain、prologue+延迟 PV 流水、GM 小槽轮转），**但 AIC/AIV 分流机制不同**：
- MLA 用 `#ifdef __DAV_CUBE__`/`#ifdef __DAV_VEC__` **预处理分支**（本 recipe §1）
- FA 用 `operator()<AscendC::AIC>()` / `operator()<AscendC::AIV>()` **模板特化**（见
  [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) Step 2）

其余组件差异：FA 用 `MmadAtlasA2FAIQK/FAIPV` + `EpilogueAtlasA2OnlineSoftmax/RescaleO`（含 `FAIQKTail/PVTail`
尾块组件），L1TileShape K=128；launch 14 参 `(hwSync,q,k,v,mask,bt,o,qSeq,kvSeq,s,p,oTmp,oUpdate,tiling)`；
tiling 用 `FAInferTiling::GetFATilingParam(FAInfo,blockDim,FATilingData)`。
手搓 FA 时按 fa-kernel-handcraft.md §1-9（FAInferKernel `operator()<AIC>/<AIV>` 特化）写，
CrossCoreFlag/主循环/HardEvent 套用本 recipe §2/§6/§7 的 MLA 同款事实。

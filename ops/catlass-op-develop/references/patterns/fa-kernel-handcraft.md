# FA kernel 手搓实现手册（从零写出 FAInferKernel 级别的 kernel）


> **定位**：这是 §0.2「FA kernel 设计经验」的**实现层**补充——§0.2 讲"怎么设计"，本手册讲"怎么写出代码"。读完本手册应能**不参考任何外部文件**，从零写出一个标准 FA（或 FA 变体）的单 kernel。
>
> **适用对象**：需要在 Ascend A2（910B/910C，AtlasA2）上手搓 FA 族 kernel 时。标准 FA 与 FA 变体（FFA/MLA/复合特性 sink/varlen/q8 等）**全部按本手册骨架（§1-11）+ 配方册/特性册（§12 系列，编号不变）手搓**（自包含，不依赖任何外部 kernel 库/示例源码）。
>
> **事实来源**：整经验证 FA 内核全量 4 文件逆向（`fai_kernel.cpp` 797 行 + `fai.cpp` 319 行 + `fai_tiling.cpp` 163 行 + `kernel_common.hpp` 154 行）+ 库层 `include/catlass/gemm/block/block_mmad_fai_qk_normal.hpp`(325 行) + `include/catlass/epilogue/block/block_epilogue_online_softmax_no_mask.hpp`(821 行) + `block_epilogue_rescale_o_no_split_row.hpp`(274 行)。

---


> **分册导航**：本册是 FAInfer 级 kernel 的**骨架手册**（§1-11 + 常量速查 + §12.14-C 全家族覆盖路由表）。
> §12 系列配方已分册（编号不变）：**§12.5（FFA 实证库）/ §12.10（s1s2 算法）/ §12.11（S1 sink）/ §12.13（kfc 终态配方）→ [fa-kernel-recipes.md](fa-kernel-recipes.md)**；
> **§12.6/§12.7/§12.12/§12.14（复合特性叠加：sink/varlen/q8/causal/paged 与测量坑）→ [fa-kernel-compound-features.md](fa-kernel-compound-features.md)**。

## 1. 你要写的 4 个文件（一个 FA kernel = 4 份代码）

> **★生成入口**：标准 FA/FA-Paged/GQA/causal（CrossCoreFlag 组件路径）**直接按
> [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) 逐行写**（四文件端到端全量提取，本节
> §1-9 是其机制解读层——recipe 写不出来时回来读这里理解为什么）；MLA 按
> [fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md)；kfc 单 TU 路径按 §12.13。

| 文件 | 内容 | 运行侧 | 行数参考 |
|---|---|---|---|
| `fai_kernel.cpp` | kernel 本体：AIC/AIV 双入口 `operator()`、任务切分、主循环、同步编排 | device | ~800 |
| `fai.cpp` | host 直调：参数解析、读数据、建 workspace、tiling、`<<<>>>` 启动 | host | ~320 |
| `fai_tiling.cpp` | host 侧 tiling：任务数/workspace 尺寸/scale 计算 | host | ~160 |
| `kernel_common.hpp` | 常量（flag ID、BLOCK_SIZE、workspace 块尺寸）、`FATilingData`、`FAIKernelParams` 结构体、对齐工具 | 共用 | ~150 |

**一个 kernel = 单一 C++ 源码里两个 `extern "C" CATLASS_GLOBAL` 入口（Fp16/Bf16），各自实例化同一个模板 `FAInferKernel<>`，模板内按 `g_coreType` 分流 AIC/AIV 的 `operator()`。** host 一次 launch 同时拉起 AIC 核（Cube）与 AIV 核（Vector）协作。

---

## 2. 任务切分算法（多核并行 + N 维 pack 进 M）

这是 FA kernel 能用普通 GEMM 组件的关键技巧。标准 FA 是 BNSD：`Q=[B,H,Sq,D], K/V=[B,H,Sk,D]`。**切分单位是 (batch, Sq 块, N 块)**：

```
每个任务 = 一个 (batch, qSBlockIdx, qNBlockIdx)
  qSBlockSize = 128（固定，Sq 按 128 分块，尾块取余）
  qNBlockTile = min(max(⌊128/qSeqlen⌋/2*2, 1), groupSize)   // N 维分块，保证偶数为 AIV 双子核负载均衡
  qNBlockNumPerGroup = ⌈groupSize / qNBlockTile⌉
  任务总数 += qNBlockNumPerGroup × kvHeads × qSBlockNum   // 逐 batch 累加
```

**核心技巧——N 维 pack 进 GEMM 的 M 维**：

```
rowNum = qSBlockSize × qNBlockSize   // 把 (S 块 × N 块) 拍平成 GEMM 的行
Q 读入后按 [qSBlockSize, qNBlockSize, D] 当作一个 rowNum×D 的矩阵做 GEMM
O 写回时按 qNBlockSize 分头拆分回 [Sq 块][H] 位置
```

**GQA（Nq>Nkv）手搓要点**：`groupSize = Nq/Nkv`（必须整除）；任务切分第 2 行的 `qNBlockTile` 按 groupSize 界定；`kvHeadIdx = qNBlockIdx / qNBlockNumPerGroup`、`qHeadIdx = kvHeadIdx·groupSize + 组内偏移`——K/V 平面用 kvHeadIdx、Q/O 平面用 qHeadIdx 索引（两套平面偏移勿混）。

多核分发（AIC/AIV 两侧相同的循环骨架）：

```cpp
uint32_t coreIdx = AscendC::GetBlockIdx();          // AIC 直接取；AIV 要 /GetSubBlockNum()
uint32_t coreNum = AscendC::GetBlockNum();
uint32_t curTotalTaskNum = 0, preTotalTaskNum = 0, curBatch = 0;
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum) {
    while (taskIdx >= curTotalTaskNum) {            // 跨 batch 推进：偏移累加
        ++curBatch; preTotalTaskNum = curTotalTaskNum;
        qBOffset += qSeqlen * strideQO;             // strideQO = H*D（BSND）
        kvBOffset += kvSeqlen * strideKV;           // strideKV = kvHeads*D
        qSeqlen = gActualQseqlen.GetValue(curBatch);
        kvSeqlen = gActualKvseqlen.GetValue(curBatch);
        curTotalTaskNum += /* 本 batch 任务数 */;
    }
    uint32_t t = taskIdx - preTotalTaskNum;
    uint32_t qSBlockIdx = t / curQNBlockNum;        // → Sq 块
    uint32_t qNBlockIdx = t % curQNBlockNum;        // → N 块（含 head 分组）
    uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup;
    uint32_t kvHeadIdx  = qNBlockIdx / qNBlockNumPerGroup;          // GQA：head 分组归属哪个 KV head
    uint32_t qHeadIdx   = kvHeadIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile;
    uint32_t rowNum = qSBlockSize * qNBlockSize;
}
```

---

## 3. AIC/AIV 双入口 + 主循环（软件流水骨架）

### 3.1 入口分流

```cpp
template <int32_t CORE_TYPE = g_coreType>
CATLASS_DEVICE void operator()(FAIKernelParams const& params);

template <> CATLASS_DEVICE void operator()<AscendC::AIC>(FAIKernelParams const& params)
{ /* Cube：loadQGM → 主循环(GEMM QK → flag → GEMM PV) */ }

template <> CATLASS_DEVICE void operator()<AscendC::AIV>(FAIKernelParams const& params)
{ /* Vector：等 qkReady → online softmax → flag → 等 pvReady → rescale O */ }
```

### 3.2 AIC 主循环（关键常量与流水结构）

```cpp
uint32_t blockStackNum = 4;      // 4 个 KV 块叠成一个 [rowNum, 512] 大块（充分利用 Cube）
uint32_t preLaunch   = 2;        // AIC 领先 AIV 2 个大块（延迟消费，软件流水深度）
int32_t  totalStackSeqNum = ⌈noMaskKvS / (4*pagedBlockSize)⌉ (+1 当 masked);
uint32_t pagedBlockSize = 128;   // KV 分块（PAGED 块大小）

// 第一循环：无 mask 的 KV 块（全注意力部分）
for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
    stackSeqTile = (尾块) ? noMaskKvS - kvSIdx*pagedBlockSize : pagedBlockSize*blockStackNum;
    // ① QK GEMM：S = Q·Kᵀ（写 workspace 槽 A）
    blockMmadQK(..., stackSeqCount % (preLaunch+1) 槽);      // 槽 = stackSeqCount%(preLaunch+1)
    CrossCoreSetFlag<0x2,PIPE_FIX>(qkReady);                 // 通知 AIV：S 就绪
    // ② PV GEMM：延迟 preLaunch 个大块消费 P（写 O_tmp 槽 B）
    if (kvSIdx >= preLaunch*blockStackNum) {
        blockMmadPV(..., (stackSeqCount-preLaunch)%(preLaunch+1) 槽);  // 读 P，等 softmaxReady
        CrossCoreSetFlag<0x2,PIPE_FIX>(pvReady);
    }
    stackSeqCount++;
}
// 第二循环：masked KV 块（causal/对角，+ 尾部 preLaunch 个块的收尾 PV）
for (kvSIdx = maskedStartIdx; kvSIdx < kvSLoopNumTotal + preLaunchStackNum; ) {
    // 同结构：QKTail（带 mask）+ 延迟 PVTail（最后一块带 mask）
    kvSIdx += (特殊边界) ? noMaskTailInteStackNum : blockStackNum;
}
```

**因果 mask 不逐位算**：`noSkipKvS = min(kvSeqlen, (qSBlockIdx+1)*qSBlockTile + diffS)` 结构性跳块；仅对角块调带 mask 的 softmax epilogue。

### 3.3 AIV 主循环（对应地消费）

```cpp
for (kvSIdx = 0; kvSIdx < kvSLoopNumNoMask; kvSIdx += blockStackNum) {
    CrossCoreWaitFlag(qkReady);                       // 等 AIC 的 S
    epilogueOnlineSoftmax(gP, gS, ..., isFirst=(stackSeqCount==0));   // S→P
    CrossCoreSetFlag<0x2,PIPE_MTE3>(softmaxReady);    // 通知 AIC：P 就绪
    if (kvSIdx >= preLaunch*blockStackNum) {
        CrossCoreWaitFlag(pvReady);                   // 等 AIC 的 O_block
        epilogueRescaleO(gO, gOTmp, ..., isFirst, isLast);            // 累加+最终÷l
    }
    stackSeqCount++;
}
```

---

## 4. 跨核同步完整编排（AIC ↔ AIV）

flag 注册在 kernel 类成员：`Arch::CrossCoreFlag qkReady{QK_READY_ID=1}, softmaxReady{2}, pvReady{3};`

| 数据流 | AIC 动作 | AIV 动作 | pipe |
|---|---|---|---|
| S（QK 输出） | `SetFlag<0x2,PIPE_FIX>(qkReady)` | `WaitFlag(qkReady)` 后读 S | AIC 写 GM workspace |
| P（softmax 输出） | `WaitFlag(softmaxReady)` 后读 P | `SetFlag<0x2,PIPE_MTE3>(softmaxReady)` | AIV 写 GM workspace |
| O_block（PV 输出） | `SetFlag<0x2,PIPE_FIX>(pvReady)` | `WaitFlag(pvReady)` 后 rescale | AIC 写 GM workspace |

- **modeId 配对**：`SetFlag<0x2, ...>` 的 modeId=2 必须与 AIV `CrossCoreWaitFlag` 内部一致（A2 支持 0/1/2）。**Set/Wait 两侧不一致 → 静默死锁 507015**。
- **硬件同步基址**：kernel 首参 `hardwareSyncAddr`（host `aclrtGetHardwareSyncAddr` 获取），入口第一行 `AscendC::SetSyncBaseAddr(hardwareSyncAddr)`。

---

## 5. HardEvent 硬件事件预置（in-pipe 流水同步）

**kernel 入口**必须预置的 `SetFlag`（漏一个或顺序错 → 流水死锁）：

```
AIC 侧： M_MTE1(EVENT0~7)×8 + FIX_M(EVENT0~1)×2 + MTE1_MTE2(EVENT0~7)×8
AIV 侧： MTE3_V(EVENT0~1)×2 + MTE3_MTE2(EVENT2~5)×4 + V_MTE2(EVENT0~3)×4 + MTE3_V(EVENT2)×1
```

**kernel 出口**对应 `WaitFlag`（逐一配平）。这些 flag 对应 A2 的物理流水依赖：
- `M_MTE1` = Cube(MMAD) 完 → L0C→UB（FixPipe→MTE1）
- `MTE1_MTE2` = UB 内操作完 → MTE2 搬运（GM/UB 间）
- `FIX_M` = FixPipe 完成 → 下一轮 MMAD 可清零 L0C
- `V_MTE2` / `MTE2_V` / `MTE3_V` / `V_MTE3` = Vector 与搬运的握手

---

## 6. BlockMmad（GEMM 块）内部机制——手搓 QK/PV 时必须复刻的流水

以 `block_mmad_fai_qk_normal.hpp`（325 行）为准，它是**一个普通 A2 GEMM + FA 专属的 KV 块堆叠**：

### 6.1 内存分级与缓冲

```
L1：l1ATensor（Q，整块一次载入，rowNum×D）
    l1BTensor[2]（K/V，2 级乒乓，每级 L1TileShape::N×K = 128×128）
L0：l0ATensor[STAGES]、l0BTensor[STAGES]（乒乓）、l0CTensor（累加器）
```

### 6.2 loadQGM：Q 的多头 Nd→L1 装载

```cpp
// Q 是 BSND [B,Sq,H,D]，一个任务取 qHeadIdx 起的 qNBlockSize 个头
// 按 [qNBlockSize, D] 一块块拷贝，nStr=H*D（跳头）
copyGmToL1A(l1ATensor, gA, layoutAInL1, layoutSingleANd, tokenNumPerGroup, qHeads*embed, ...);
SetFlag<MTE2_MTE1>(EVENT_ID3);  WaitFlag<MTE2_MTE1>(EVENT_ID3);  // 搬运完成屏障
```

### 6.3 内层计算循环（QK：M=rowNum, N=KV列, K=embed）

```cpp
for (blockStackIdx=0; blockStackIdx<4 && (nIdx+blockStackIdx)<nLoop; ++blockStackIdx) {
  for (kIdx=0; kIdx<kLoop; ++kIdx) {          // kLoop = ⌈embed/128⌉
    nowNIdx = nIdx + blockStackIdx;
    getBlockShape(...); getKVOffset(...);      // 实际 tile 尺寸 + PAGED block_table 偏移
    // 每 2 个 KV 块一个输出槽：gCOffset = (blockStackIdx/2*2)*128
    computeQK(..., firstItr=(blockStackIdx%2==0 && kIdx==0),
                  endItr  =(blockStackIdx%2==1 || nowNIdx==nLoop-1) && kIdx==kLoop-1,
                  initMmad=kIdx==0, firstQItr=blockStackIdx==0, ...);
  }
}
```

### 6.4 computeQK：单次 GEMM tile 的精确事件序列（★复刻顺序）

```cpp
// ① K 的 GM→L1（双缓冲，kIdx==1 时先发第一块，此后一直提前发下一块）
copyGmToL1B(l1BTensor[pong], gB, ...); SetFlag<MTE2_MTE1>(pong);
// ② Q 的 L1→L0（firstQItr 才做一次；此后 l0ATensor[0] 复用）
WaitFlag<M_MTE1>(0); WaitFlag<M_MTE1>(1); copyL1ToL0A(l0ATensor[0], l1ATensor, ...);
// ③ K 的 L1→L0（乒乓）
WaitFlag<MTE2_MTE1>(pong); WaitFlag<M_MTE1>(pong+2); copyL1ToL0B(l0BTensor[pong], l1BTensor[pong], ...);
SetFlag<MTE1_MTE2>(pong);
// ④ MMAD（initMmad=true 时清零 L0C；firstItr 需先等 FIX_M 0/1——等上一轮 FixPipe 写完 L0C→GM）
if (firstItr) { WaitFlag<FIX_M>(0); WaitFlag<FIX_M>(1); }
tileMmad(l0CTensor[pong*mRound*128], l0ATensor[0], l0BTensor[pong], mRound, nActual, kActual, initMmad, unitFlag);
// ⑤ L0C→GM（endItr 才搬；FIX 前必须 M_FIX 同步，写 zN→RowMajor）
if (endItr) { SetFlag<M_FIX>(pong); WaitFlag<M_FIX>(pong);
              copyL0CToGm(gC, l0CTensor, layoutC, layoutInL0C);   // zN 布局
              SetFlag<FIX_M>(0); SetFlag<FIX_M>(1); }
```

**K 每 4 个块堆成一个 512 列大块**（`gCOffset = (blockStackIdx/2*2)*128`），S 在 GM 的 stride 用 512 对齐（`LayoutC(rowNum, stackTile, 512)`）——让 AIV 侧一次处理 512 列。

---

## 7. Online softmax epilogue（AIV 核心算法）——手搓实现

`block_epilogue_online_softmax_no_mask.hpp`（821 行）。**核心是 m/l/O 状态在 UB 的布局 + 一组向量运算**。

### 7.1 UB 布局（AIV 私有 buffer）

```
LS（S 输入，fp32）  @ 0x00        ← AIC 的 QK 输出，pingpong×MAX_UB_S_ELEM_NUM(8192)
LP（P 输出，fp16）  @ 4*16384
LM/LL（本块 rowmax/rowsum）@ 10*16384+8*1024 起   （8 字节/行 × MAX_ROW_NUM_SUB_CORE=128）
HM/GM（全局 max）/ DM（diff）/ GL（全局 sum）@ 9~13*1024 偏移
TV（临时/tv）@ 10*16384
```

### 7.2 逐块计算序列（对应数学：m_new=max, P=exp, l_new 递推）

对每个输入行子块（按 `MAX_ROW_NUM_SUB_CORE=128` 行 + `preLoad=1` 乒乓）：

```
① CalcLocalRowMax    lm = rowmax(S)                    // BlockReduceMax，512 列走专用 3 级归约
② UpdateGlobalRowMax hm = max(lm, gm)                  // 第一块：hm=lm（copy）；否则：
                      dm = gm - hm; dm = exp(dm)        //   rescale 因子 = exp(m_old - m_new)
                      gm = hm
③ CalcExp            tv = Brcb(hm) 广播成块;  ls -= tv; ls = exp(ls)   // S←exp(S-m_new)
④ DownCastP          lp = cast_fp32_fp16(ls)           // P
⑤ CalcLocalRowSum    ll = rowsum(ls)                   // BlockReduceSum
⑥ UpdateGlobalRowSum gl = dm*gl + ll                   // 第一块：gl=ll（copy）；否则累加
```

**关键算子**：`BlockReduceSum/Max`（块内逐行归约，512 列 3 级 / 256 列 2 级 / 尾块 mask 处理）、`Brcb`（1×N 状态向量广播成整块）、`SetVectorMask`（尾部对齐）。

### 7.3 双子核切分（GetSubBlockIdx）

`qNBlockSize==1` 时按行对半（每子核 ≤64 行）；`>1` 时按 head 分（子核 0 拿前 `qNBlockSize/2` 头）。两子核对同一 GM S 分区消费。

### 7.4 数据流同步（AIV 内部）

```
WaitFlag<V_MTE2>(pong) → CopyS → SetFlag<MTE2_V>(pong)     // S GM→UB（preLoad 提前 1 行）
SubCoreCompute 内：      WaitFlag<MTE2_V> → 计算 → SetFlag<V_MTE3>(P写GM)
                         SetFlag<V_MTE2>(ll写GM) → WaitFlag<V_MTE3> → CopyP → SetFlag<MTE3_V>
mask 版：额外 EVENT_ID2 管 mask 的 GM→UB + ApplyMask（mask32*=-3e38 再加到 S）
```

---

## 8. Rescale O epilogue（AIV）——手搓实现

`block_epilogue_rescale_o_no_split_row.hpp`（274 行）：

```
// isFirstStackTile：O_tmp 直接拷贝进 go（不 rescale）
// 否则：go = go*dm  +  lo（lo 是上一大块 O_tmp）—— 在线累加 O_acc
// isLastStackTile：go = go / gl（Brcb 广播 gl → Div）；cast fp32→fp16；写 GM
```

```
WaitFlag<V_MTE2>(EVENT3) → DataCopy(lo) → SetFlag<MTE2_V>(EVENT0) → WaitFlag<MTE2_V>(EVENT0)
tv = Brcb(dm); go *= tv;  go += lo                    // 非首块
tv = Brcb(gl); go /= tv;  go = cast(go); CopyOToGm    // 末块（÷l 只做一次）
SetFlag<V_MTE2>(EVENT3)
```

---

## 9. Host 侧直调模板（fai.cpp 的骨架）

```cpp
auto aicCoreNum = PlatformAscendCManager::GetInstance()->GetCoreNumAic();
uint32_t blockDim = aicCoreNum;
// workspace 4 段（每段 = aicCoreNum × 131072 × 3 个元素 × 元素字节）
//   s=fp32, p=fp16, oTemp=fp32, oUpdate=fp32（§0.2.4 已有）
// tiling：FAInferTiling::GetFATilingParam(faInfo, blockDim, faTilingData) → memcpy 到 device
uint64_t hardwareSyncAddr;
aclrtGetHardwareSyncAddr((void**)&hardwareSyncAddr);
FAInferFp16<<<blockDim, nullptr, stream>>>(
    hardwareSyncAddr, q, k, v, mask, blockTables, o,
    qSeq, kvSeq, s, p, oTemp, oUpdate, tiling);
```

**kernel 入口签名（14 参数，顺序固定）**：
```cpp
extern "C" CATLASS_GLOBAL void FAInferFp16(
    uint64_t hardwareSyncAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR s, GM_ADDR p, GM_ADDR oTemp, GM_ADDR oUpdate,
    GM_ADDR tiling)
{ AscendC::SetSyncBaseAddr(hardwareSyncAddr); FAInferKernel<> kernel; kernel(params); }
```

---

## 10. 变体映射法：怎么把一个新 attention 形态手搓成 kernel

拿到一个新 FA 变体（FFA、MLA、双 KV…），按以下顺序映射到骨架：

| 步骤 | 问自己 | 对应修改点 |
|---|---|---|
| 1 数学 | 分几步 GEMM？每步 A/B/C 是什么、S 还是 P 语义？ | 新增/修改 `blockMmad*` 调用；`MmadAtlasA2FAIQK`/`FAIPV` 换成对应语义 |
| 2 flag | 数据依赖 AIC↔AIV 有几条？每步产物谁消费？ | 新增 `CrossCoreFlag`（ID 递增）；每数据流一对 Set/Wait（modeId 一致） |
| 3 workspace | 中间产物几类？fp16/fp32？ | `FATilingData` 加字段；`fai.cpp` 加段；`FillWorkSpaceTilingData` 加尺寸 |
| 4 主循环 | 每步 GEMM 的 (M,N,K)、按哪个维分块、有无跳块 | 主循环里按顺序插 QK/PV 调用的对（流水：产 S→flag→等 P→PV→flag） |
| 5 host 布局 | 公开接口 BNSD/其他？输入布局怎么转 | host 转换（§1 已有）；`FAIKernelParams` 加指针 |
| 6 epilogue | softmax 变体（分片、对角）？ | 仿 §7/§8 写新 epilogue，复用 BlockReduce/Brcb 模式 |
| 7 精度/性能 | 冻结 baseline、mixed tolerance、msprof 口径 | FA9/FA10 检查表 |

**FFA 实例**（`S=scale·(Q·K1ᵀ+Q·K2ᵀ对角)→softmax→O=W·V1+W·V2对角`）：
- 数学是 **4 个 GEMM**（QK1、QK2、PV1、PV2），其中 K2/V2 带对角语义（Q 的 M 与 K2 的 M 共享）；实现上把对角 matmul 的 `M` 维也 pack 进 GEMM 行（与 N-pack 同思路），或用 permute 消除对角（host 做）。
- flag 需要 **5 条**（QK1Ready、QK2Ready、PReady、PV1Ready、PV2Ready）而非 3 条；workspace 相应加段。
- 主循环每轮：AIC 产 QK1/QK2 → flag → AIV 先加再 softmax → flag → AIC 产 PV1/PV2 → flag → AIV rescale O。
- 其余骨架（任务切分/双循环/preLaunch/事件预置/host 模板）原样复用。

---

## 11. 手搓自查表（写完 kernel 后逐条过）

| # | 检查项 |
|---|---|
| K1 | AIC/AIV 两个 `operator()` 特化都存在；入口 18 个 SetFlag（AIC）/10 个 SetFlag（AIV）与出口 WaitFlag 一一配平 |
| K2 | `SetSyncBaseAddr(hardwareSyncAddr)` 在入口第一行；host 传对 `aclrtGetHardwareSyncAddr` |
| K3 | 每个 `CrossCoreFlag` 的 Set（AIC/AIV 侧）与 Wait（对侧）modeId 一致；ID 全局唯一且 < 硬件支持数 |
| K4 | workspace 槽位 `coreIdx×(preLaunch+1)+slot`，slot = `stackSeqCount%(preLaunch+1)`，两侧算法一致 |
| K5 | `rowNum = qSBlockSize×qNBlockSize` 的 N-pack 在 Q 装载、S 布局、O 拆分三处对称 |
| K6 | tail 块：`stackSeqTile = noMaskKvS - idx*pagedBlockSize`；非 128 对齐时 RowReduce 用 SetVectorMask 处理 |
| K7 | 除法 ÷l 只在最后一个 KV 块（isLastStackTile）；rescale 因子 exp(m_old-m_new) 对 O 和 l 都乘 |
| K8 | causal 用结构性跳块（noSkipKvS），不是逐位 mask |
| K9 | host 与 kernel 的 tiling 字段、参数顺序、bin 尺寸三方一致 |
| K10 | 精度双口径（内部 golden + aclnn 标杆）+ msprof device time 归档 |

---


---

## 12. 关键常量速查

```
BLOCK_SIZE = 16（元素对齐基数）
WORKSPACE_BLOCK_SIZE_DB = 131072（≈128K 元素槽，fp32）
L1TileShape = GemmShape<128,128,128>（K 必须 = head_dim D，D≤128）
pagedBlockSize = 128；blockStackNum = 4；preLaunch = 2
qSBlockTile = 128（GetQSBlockTile 固定）
QK_READY_ID=1, SOFTMAX_READY_ID=2, PV_READY_ID=3
MAX_UB_S_ELEM_NUM = 8192；MAX_ROW_NUM_SUB_CORE = 128
UB_UINT8_BLOCK_SIZE = 16384；UB_UINT8_VECTOR_SIZE = 1024
FLOAT_VECTOR_SIZE=64；FLOAT_BLOCK_SIZE=8；VECTOR_SIZE=128（fp16）
```


---

### C. 覆盖对照（全家族 → 手搓路径）

| 算子 | 手搓路径 |
|---|---|
| 标准 FA / FA-BNSD / **FA-Paged**（CrossCoreFlag 组件路径） | **[fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md)**（四文件端到端逐步骤：kernel Step 1-7 + mask Step 8 + 三件套 Step 9-11，NO_MASK 闭卷实证全 PASS）；§1–9 为其机制解读层 |
| 标准 FA（kfc 单 TU 直调路径） | §12.13（含 F0 十一步逐步生成流程） |
| FA-BNSD-Causal | [fa-paged-handcraft-recipe.md](fa-paged-handcraft-recipe.md) Step 8（组件级 MASK_CAUSAL）或 §12.13 + §12.14-A（kfc 逐行尾部法） |
| FA-TND varlen | **[fa-varlen-handcraft-recipe.md](fa-varlen-handcraft-recipe.md)**（varlen kernel 差异逐步骤：累加差分/TND 直读/s2End 截断/L1Carry/SameAB + 两路对照）或 §12.13 + §12.12-A2（kfc + 转块格式路径） |
| FA-paged（kfc 路径） | §12.13 + §12.14-B（block_table 查表改块级） |
| FA-sink（S1 模板路径） | **[fa-sink-handcraft-recipe.md](fa-sink-handcraft-recipe.md)**（S1 模板 kernel 逐步骤：三深流水/SoftmaxFlashV2 isUpdate 播种 sink 正确用法/workspace 精确公式/sparse 跳块）+ §12.11（算法语义/tiling 公式/基线）；kfc 路径 = §12.13（内建 sink） |
| FFA（双 KV） | §12.5.14 完整配方（生成路径）+ [ffa-handcraft-recipe.md](ffa-handcraft-recipe.md)（FFA/FIA 算子契约：TensorList/sharedPrefix/sink-in-epilogue/FD/LSE/nZ 语义与差异清单） |
| 复合（sink+varlen+q8 任意组合） | §12.6 总纲 + §12.6-E + §12.12（互不冲突，四合一 +4.3%） |
| GQA | recipe Step 3d/4d 双平面索引（kvHeadIdx/qHeadIdx；§2 机制解读） |
| fa-sageattention（A5/950 量化注意力） | **[fa-sageattention-recipe.md](fa-sageattention-recipe.md)**（Ascend950/dav-3510：开发流程七步 + architecture/rules/troubleshooting/precision 四参考，自包含） |
| MLA | **[fa-mla-handcraft-recipe.md](fa-mla-handcraft-recipe.md)**（端到端：kernel §1-8 + tiling §11 + 槽位/host §12；§12 含 FD rescale kv-split 归并全码）；fa-mla-paged.md §1-4 为 host 契约、§8 为设计依据 |

# 权重梯度分支 Ascend C Kernel 实现模板

本页落实设备入口、任务骨架、地址、搬运、计算、输出和同步。计算模型与布局见 [compute-flow](../../../generic/filter/compute-flow.md)，API 字段见 [instruction-mapping](instruction-mapping.md)。模板对应 `Conv3dDwBasicBlockMNStreamK`，当前 streamkType=0、groupEnlarge=false。

片段中的类、阶段函数和共享结构均由生成工程定义；本文给出其实现契约，不产生对参考算子仓源码的工程依赖。入口使用外层 tilingData，类内 td 引用其 dwTiling，即 TConv3DDwTiling 字段。

片段中 T 为与 DTYPE_X 一致的 half 或 bfloat16_t；省略 self->ctx 的字段仍指同一计算上下文，阶段函数名与本文各节及指令映射对应。

## 1 Kernel 概述

### 1.1 数据流图

```text
InitOutput: AIV Fill dedw -> AIC waits -> paired data paths

  +-----------------------------------+           +-----------------------------------+
  | dedy / GM                         |           | x / GM                            |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    | DataCopy                                      | DataCopy
                    |                                               |
                    |                                               |
                    v                                               v
  +-----------------+-----------------+           +-----------------+-----------------+
  | L1A / parent segment              |           | L1B / x rows                      |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    | Load2D + transpose                            | Load3D windows
                    |                                               |
                    |                                               |
                    v                                               v
  +-----------------+-----------------+           +-----------------+-----------------+
  | L0A / NZ(M,K)                     |           | L0B / ZN(K,N)                     |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            | MMAD / FP32
                                            v
                          +-----------------+-----------------+
                          | L0C / segment partial sum         |
                          +-----------------+-----------------+
                                            |
                                            |
                                            | Fixpipe / atomic add
                                            |
                                            v
                          +-----------------+-----------------+
                          | dedw / GM / FP32                  |
                          +-----------------------------------+
```

AIV 先清零 FP32 dedw，通知 AIC 后才开始主计算。普通路径中 A=dedy 直装加分形转置，B=x 正向窗口；拆核路径把原核坐标的采样前移到 MTE2。全部结果经 Fixpipe 原子累加到 dedw。

### 1.2 状态与术语

| 状态 | 含义与生产者 |
|---|---|
| mCnt_/nCnt_/totalCnt_/calRound_ | 任务计数与核轮数，CalBasicBlockCnt |
| mCoreIndx_/nCoreIndx_/cinCoreIndx_ | 当前输出通道/窗口任务，CalBasicBlock |
| groupIdx/dkIdx | 数学组与原核深度 |
| batchDoutIdx/hoIdx | 当前精度归约段起点 |
| singleShapeBatch_/singleShapeK | 当前 ND 段长度、H 段输出平面长度 |
| singleShapeM/NInCurrentHo_ | 深度有效性、组尾收缩后的任务范围 |
| baseUseM_/N_/K_ | 当前 L0 子块有效长度 |
| kal1_/kbl1_ | stepKa/stepKb 乘 baseK，父段预留长度 |
| kaStepIdx/kbStepIdx | 当前两路父段编号 |
| alignedL1UseKa | 当前 L1A 实际父跨度 |
| bL1cin1CopyLen | 当前 L1B 实际通道数 |
| offsetA_/offsetB_/offsetC_ | 当前段三路 GM 元素偏移 |
| isFirstMmad | 当前 L0C 结果是否尚无实际贡献 |
| l0aPingPongFlag_/l0cPingPongFlag_ | 输入槽事件/输出 CO1 槽的轮换状态 |
| head_/tail_/cinRemainLen_ | 非整窗 N 的核区间与有效通道尾部 |
| baseUseMBak | ComputeInit 前的有效 M；计算 M=1 时临时改为 16，发布 L0C 前恢复 |
| cinHkWkLoop_ | 一个核深度的 N 子块数；当前任务固定 kd，用于 N 起点和尾块回绕 |

singleCoreN 是任务覆盖的完整通道窗口长度，baseN 是一次 L0 计算的 N 长度，baseUseN_ 是当前子块截断后的有效长度。baseUse* 用于当前有效计算，分形对齐尺寸用于槽占用。Kernel 的 cin1G/cout1G 字段是执行组通道跨度，与 Host 的 cin1_g/cout1_g 块数分别解释，见 [Tiling 绑定](tilingkey-templates.md)。

## 2 Kernel 骨架代码

### 2.1 入口与编译实例

入口保留 x、filter_size、out_backprop、y、workspace、tiling 六个 GM 参数；filter_size 的值已经由 Host 校验并写入共享几何。当前用户 workspace 为零，不能仅因其指针为空而跳过本算子。

```cpp
template <uint32_t conv3DDWTemplateId, bool isSplitKernelHW,
          bool groupEnlarge, uint32_t winogradTilingFlag, bool winogradResidentFlag>
__global__ __aicore__ void conv3ddw_custom(
    GM_ADDR x, GM_ADDR filter_size, GM_ADDR out_backprop,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    static_assert(conv3DDWTemplateId == TPL_MN_STREAM_K);
    static_assert(!groupEnlarge && winogradTilingFlag == 0 && !winogradResidentFlag);
    REGISTER_TILING_DEFAULT(Conv3DBackpropFilterV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(Conv3DBackpropFilterV2TilingData, tilingData, tiling);
    if (workspace != nullptr) SetSysWorkspace(workspace);
    Conv3dDwInitOutput<float> init;
    init.Init(y, &tilingData);
    init.Process();
    init.Destroy();
    if ASCEND_IS_AIC {
        Conv3dDwBasicBlockMNStreamK<DTYPE_X, FORMAT_X,
            DTYPE_OUT_BACKPROP, FORMAT_OUT_BACKPROP,
            float, FORMAT_Y, isSplitKernelHW, false> op;
        op.Init(x, out_backprop, y, nullptr, &tilingData);
        op.Process(); // 内部完成任务及 End；入口不再重复 End
    }
}
```

这是生成骨架。注册模板的 MIX 核类型在共享 TPL 声明中绑定，直调由启动配置承担；两者均与初始化协议配对，见 tilingkey-templates。输入同为 half 或同为 bfloat16_t，FORMAT_Y 选择已实现的三种输出格式。

### 2.2 任务与归约骨架

任务解码沿用 CalBasicBlock。下列核心段省略类限定符，几何来自 td，当前组内实际 CinG/CoutG 由原始分组规格确定：

```cpp
__aicore__ inline void Process() {
    if (block_idx >= usedCoreNum_) return; // 初始化协议已在入口完成
    for (uint64_t idx = block_idx; idx < totalCnt_; idx += usedCoreNum_) {
        uint32_t groupIdx = idx % td.realGroup;
        uint32_t dkIdx = (idx/td.realGroup) % td.dk;
        nCoreIndx_ = (idx/td.realGroup/td.dk) % nCnt_;
        mCoreIndx_ = idx/(nCnt_*td.dk*td.realGroup);
        cinCoreIndx_ = nCoreIndx_;
        CalSingleBasicBlock(dkIdx, groupIdx);
    }
    dw_.End();
}

__aicore__ inline void CalSingleBasicBlock(uint32_t dkIdx, uint32_t groupIdx) {
    uint64_t totalND = uint64_t(td.batch)*td.dout;
    uint64_t reduceSegmentsH, reduceSegmentsND;
    CalcReduceSegments(totalND,td.ho,td.wo,reduceSegmentsH,reduceSegmentsND);
    for (uint64_t bd = 0; bd < totalND; bd += reduceSegmentsND) {
        singleShapeBatch_ = min(totalND-bd,reduceSegmentsND);
        for (uint64_t h = 0; h < td.ho; h += reduceSegmentsH) {
            CalcOffset(bd/td.dout,bd%td.dout,groupIdx,dkIdx,h);
            ReCalDkCinSingleCoreShape(bd%td.dout,groupIdx,dkIdx);
            if (singleShapeMInCurrentHo_ == 0 || singleShapeNInCurrentHo_ == 0) continue;
            uint64_t mUse = min(mCoreUse,singleShapeMInCurrentHo_);
            uint64_t nUse = min(nCoreUse,singleShapeNInCurrentHo_);
            uint64_t ciUse = min(ciCoreUse,actualCinG-ciStart);
            uint64_t kUse = min(uint64_t(td.ho)-h,reduceSegmentsH)*td.wo;
            dw_.SetOutBackprop(dedyGm_[offsetA_]);
            dw_.SetFmap(xGm_[offsetB_]);
            dw_.SetSingleShape(mUse,nUse,kUse,ciUse,singleShapeBatch_);
            dw_.SetStartIdx(bd,h,dkIdx);
            dw_.IterateAll(yGm_[offsetC_],1);
        }
    }
}
```

mCoreUse/nCoreUse/ciCoreUse 是当前任务按数学尾长截断后的范围，ciStart=cinCoreIndx_*singleCoreCin；它们在进入段循环前由任务解码产生。CalcReduceSegments 的计算见 compute-flow 第 7.3 节。每个 IterateAll 完成该段全部 M/N 子块与 batch/dout、宽片、K 的计算和写回。

### 2.3 核内计算与阶段函数

普通路径的 ComputeLoop 按 K 块号推进，两路父段更新频率独立。阶段名与实现对应：

| 阶段 | 本工程函数职责 |
|---|---|
| UpdateMNIdx | 当前 M/N 子块与尾块，确定父段是否需重载/归还 |
| ComputeLoadToA1 / PreLoad | 当前及下一 A 父段的 MTE2 搬入 |
| ComputeLoadToB1 / PreLoad | 当前及下一 B 父段；无有效 x 行带时形成跳过/零窗口 |
| LoadL12L0a / LoadL12L0b | 当前 K 子块的两路 MTE1 准备 |
| ExecuteMmad | 输入就绪、初值、有效 K、MmadLocal |
| GetTensorC / LoadL0c2Gm | 完整段结果交给 F、原子写回、归还 CO1 |

这些函数按第 4～9 节与指令映射实现。普通与拆核共用段生命周期，但拆核的核坐标循环、L0C 子片段不能删成单个普通窗口调用。

ComputeNormal 的顺序固定为：ComputeInit → batchDoutIdx 循环 → 当前宽片 → ComputeLoop → 必要的零结果 → 恢复有效 M → CO1 EnQue。isFirstMmad 在本输出子块的全部 batch/dout、宽片和 K 之间保留；只在实际 MMAD 提交后清为 false。

每次进入一个批深度先重算真实 id，即使 id 无效也完成源地址推进再跳过。每个宽片从保存的批深度基址重算，不把上一宽片的偏移重复相加。新的宽片重置 kIter_/tailK_/父段总数；每次 ComputeLoop 的父段内块号从 0 开始。普通与拆核的行带公式及例子见 [指令映射第 3 节](instruction-mapping.md#3-x-到-l1b)。

## 3 地址偏移计算

### 3.1 核间偏移计算

任务通道起点为 `coStart=mCoreIndx_*singleCoreCout`、`ciStart=cinCoreIndx_*singleCoreCin`。令 `globalCo=groupIdx*CoutG+coStart`、`globalCi=groupIdx*CinG+ciStart`，固定任务 kdIdx。

| 对象 | NCDHW 元素偏移 | NDHWC 元素偏移 |
|---|---|---|
| offsetA_ | `((n*cout+globalCo)*dout+od)*ho*wo+hoStart*wo` | `((n*dout+od)*ho+hoStart)*wo*cout+globalCo` |
| offsetB_，有效深度 | `((n*cin+globalCi)*di+id)*hi*wi+hiStart*wi` | `((n*di+id)*hi+hiStart)*wi*cin+globalCi` |
| offsetC_ | `(globalCo*CinG+ciStart)*dk*hk*wk+kdIdx*hk*wk` | `globalCo*dk*hk*wk*CinG+kdIdx*hk*wk*CinG+ciStart` |

DHWCN 的 offsetC_ 为 `kdIdx*hk*wk*CinG*cout+ciStart*cout+globalCo`。固定任务后，写回函数再加 M/N 子块与 kh/kw 的格式内偏移。

### 3.2 核内循环间偏移计算

实际 x 深度 `id=od*strideD+kdIdx*dilationD-padFront`，起始输入行 `hiStart=max(hoStart*strideH-padUp,0)`。采样坐标使用 int64_t；无效深度在搬入前跳过。

```cpp
uint64_t nd = batchDoutIdx + localND;
uint64_t n = nd/td.dout;
uint64_t od = nd%td.dout;
int64_t id = int64_t(od)*td.strideD + int64_t(dkIdx)*td.dilationD - td.padFront;
if (id < 0 || id >= td.di) continue;
// 只对有效 id 形成 GM 地址；段内 L0C 初值状态保持不变。
```

K 子块相对父段的地址见 instruction-mapping 第 2～5 节。父段切换使用真实跨度和行内起点；co/ci 的 GM 间隔保留总通道数与完整 D/H/W。offsetC_ 不加 batch/dout/ho 段起点。

本段源地址按以下层次推进，所有增量均为对应张量元素：

| 层次 | dedy / A | x / B |
|---|---|---|
| 当前 M/N 子块 | co 子块起点 | CalOut2B1Params 解出的 ci 起点 |
| 批深度 | 原 n/od 平面 | 有效 n/id 平面 |
| 宽片 | startWo，NDHWC 乘总 cout | 裁出的首个真实输入列，NDHWC 乘总 cin |
| 父段 | A 父段起点；切宽时向完整输出行回退 | 裁出的真实输入行带起点 |
| MTE1 | A 父段内 K 元素与切宽行内余数 | B 父段内 K 位置，mStartPt 补回父段行内余数 |

out2A1SrcAddr/out2B1SrcAddr 已包含外层更新的层次；内部仅添加尚未承担的增量。A/B 的原始通道、深度步幅始终使用总 cin/cout 与原 di/dout，不使用对齐维度替代。

### 3.3 Buffer 分配

weight 使用每槽一个队列的形式，不能由共享层 input 的具体队列深度推定其槽数：

| Buffer | 实现 | 单份字节数/位置 |
|---|---|---|
| L1A | a1Ping_/a1Pong_，A1 队列 | alignUp(al1Bound*sizeof(T),32) |
| L1B | b1Ping_/b1Pong_，B1 队列 | alignUp(bl1Bound*sizeof(T),32) |
| L0A/L0B | A2/B2 TBuf | 各自整个容量，按最终份数取槽 |
| L0C | l0cPing_/l0cPong_，CO1 队列 | baseM*baseN*sizeof(float) |
| 初始化 UB | InitOutput 的 TPipe 管理 | 只在初始化阶段使用 |

```text
+--------+------------------------------+--------------------------------------------+
| memory | slot 0                       | slot 1, when enabled                       |
+--------+------------------------------+--------------------------------------------+
| L1A    | [0,aAligned)                 | [halfL1,halfL1+aAligned)                   |
| L1B    | [aAligned,aAligned+bAligned) | [halfL1+aAligned,halfL1+aAligned+bAligned) |
| L0A    | first input region           | second input region                        |
| L0B    | first input region           | second region / single slot if split       |
| L0C    | current segment              | next segment output slot                   |
+--------+------------------------------+--------------------------------------------+
```

al1Bound/bl1Bound 的单位为元素。半区放置时，L1A ping=0、L1B ping=aAligned、L1A pong=L1Bytes/2、L1B pong=L1Bytes/2+aAligned；仅为实际份数创建队列，并逐段检查交叠和最大地址。

## 4 CopyIn MTE2 逻辑

### 4.1 LoadToA1：dedy 搬入

NCDHW 使用 LoadToA1Fp16Dn2Nz，NDHWC 使用 LoadToA1Fp16Nd2Nz；切宽分别使用 SplitWLoadToA1Dn2Nz/Nd2Nz。ComputeInit 将单个有效 co 的计算 M 临时扩到 16，因此本模板仍使用相同的分形转置通路；DataCopy.dValue 保留真实 tailM_。

父段以 kaStepIdx*kal1_ 定位，末段用实际 K 截断，co 尾块用 tailM_。切宽按真实覆盖行数搬入并更新 alignedL1UseKa；源矩阵步幅保留完整 wo/dout 与总 cout。

### 4.2 LoadToB1：x 搬入

CalOut2B1Params 解出当前 ci，CalB1HiCopyParams 解出输出行带及其 x 有效交集，再调用 LoadToB1Dn2Nz/Nd2Nz。bL1cin1CopyLen 保存实际通道数，不能把 Host 的块数直接写入 dValue。

拆核使用对应 SplitKernelHW 搬运：先固定原 kh/kw 的采样点，每行沿 strideW 取点，行间跨 strideH；后续窗口核长为 1。pad 区值由初始化与窗口边界产生，GM 不访问负地址。

### 4.3 父段与预取

```text
+---------+-------------------------------+---------------------------+---------------------------+
| time    | current parent                | other slot                | constraint                |
+---------+-------------------------------+---------------------------+---------------------------+
| load    | MTE2 writes parent j          | free / returned           | wait before overwrite     |
| consume | MTE1 reads parent j           | MTE2 may prefetch j+1     | next parent exists        |
| release | last MTE1 read of j submitted | hold j+1                  | publish return dependency |
| next    | consume j+1                   | reuse j only after return | separate A/B boundaries   |
+---------+-------------------------------+---------------------------+---------------------------+
```

预取前先确认下一父段存在、目的槽已归还、有效深度和行带合法；末段不生产不存在的父段。A/B 的 step 不同时，各自按边界换驻，当前计算仍消费同一 K。释放父段发生在最后一次 MTE1 读取提交后，下一次覆盖等待队列依赖。

普通与拆核共用下列 ComputeLoop。进入时两路父段内块号和父段号均为 0，isA1NormalLoad/isB1NormalLoad 为 true；p 是 UpdateMNIdx 形成的 Out2L1ScalarParams，所选 L1 槽由当前 M/N 父段基号与父段号确定。

```cpp
uint32_t kaIdx=0, kbIdx=0;
uint64_t kaStepIdx=0, kbStepIdx=0;
bool skipCurrentHiCompute=false, skipCurrentHiComputePreLoad=false;
bool isA1NormalLoad=true, isB1NormalLoad=true;
for (uint64_t k = 0; k < kIter_; ++k) {
    bool isLastKIter = k+1 == kIter_;
    bool isLastStepKa = kaIdx+1 == td.stepKa;
    bool isLastStepKb = kbIdx+1 == td.stepKb;
    baseUseK_ = isLastKIter ? tailK_ : td.baseK;
    bool isLoadA1 = kaIdx == 0, isLoadB1 = kbIdx == 0;
    bool a1Ping = td.al1Pbuffer == 1 || ((curMKL1Idx+kaStepIdx+1)&1);
    bool b1Ping = td.bl1Pbuffer == 1 || ((curNKL1Idx+kbStepIdx+1)&1);
    if constexpr (isSplitKernelHW)
        LoadToB1SplitKernelHW(self,b1Ping,p,kbStepIdx,hkIdx,isLoadB1,skipCurrentHiCompute);
    else
        ComputeLoadToB1(self,b1Ping,p,kbStepIdx,isB1NormalLoad,
                        td.bl1Pbuffer>1,isLoadB1,skipCurrentHiCompute,skipCurrentHiComputePreLoad);
    if (skipCurrentHiCompute) {
        isA1NormalLoad = isB1NormalLoad = true;
        UpdateIdx(isLastStepKa,isLastStepKb,kaIdx,kbIdx,kaStepIdx,kbStepIdx);
        continue;
    }
    ComputeLoadToA1(self,a1Ping,k,p,kaStepIdx,isA1NormalLoad,td.al1Pbuffer>1,isLoadA1);
    if constexpr (!isSplitKernelHW) {
        ComputeLoadToB1PreLoad(self,b1Ping,p,kbStepIdx,k,isB1NormalLoad,
                               td.bl1Pbuffer>1,isLoadB1,skipCurrentHiComputePreLoad);
        ComputeLoadToA1PreLoad(self,a1Ping,k,p,kaStepIdx,isA1NormalLoad,
                               td.al1Pbuffer>1,isLoadA1,skipCurrentHiComputePreLoad);
    }
    WaitFlag<HardEvent::M_MTE1>(l0aPingPongFlag_ & 1);
    ExecuteMTE1L0b(self,p,l0b,b1Ping,isLoadB1,isLastStepKb,isLastKIter,k,kbIdx);
    ExecuteMTE1L0a(self,p,l0a,a1Ping,isLoadA1,isLastStepKa,isLastKIter,k);
    ExecuteMmad(self,l0a,l0b,l0c,isFirstMmad);
    SetFlag<HardEvent::M_MTE1>(l0aPingPongFlag_);
    l0aPingPongFlag_ ^= useL0PingPong_;
    UpdateIdx(isLastStepKa,isLastStepKb,kaIdx,kbIdx,kaStepIdx,kbStepIdx);
}
```

UpdateIdx 在当前父段最后一个子块后把相应局部块号清 0、父段号加 1，其余时候只增加局部块号。isFreeAL1/BL1 的完整条件和父段复用例子见 [同步实现第 2.1 节](sync-implementation.md#21-updatemnidx首次与后续子块的复用条件)；真实搬入时 Alloc/Copy/EnQue，父段入口 DeQue，末次 MTE1 读取后按条件归还。

## 5 CopyIn MTE1 逻辑

### 5.1 LoadL12L0a：L1A 到 L0A

多通道 FP16/BF16 使用 ifTranspose=true：mStep=ceilDiv(baseUseK_,16)，kStep=ceilDiv(baseUseM_,16)，srcStride=alignedL1UseKa/16，dstStride=kStep。源偏移为 `(kPos%stepKa)*baseK*16`，切宽再加父段的行内起点。

kPos 是 LoadL12L0a 的形参，接收全局 k；源切宽余数为 `(kaStepIdx*kal1_%singleShapeWo_)*16`。srcL0aOffset_=srcL0bOffset_=0，相对本轮已选中的 L0 输入槽；物理槽基址只在构造 l0a/l0b 子视图时加一次。

### 5.2 LoadL12L0b：L1B 到 L0B

普通窗口 kExtension=baseUseN_、mExtension=baseUseK_，channelSize 按覆盖窗口数折算成实际 ci 通道跨度；Repeat 目的步幅为 ceilDiv(baseUseN_,16)。K/N 的轴义和数值例子见 instruction-mapping 第 5 节。

### 5.3 拆核通路

每轮原核高宽位置对应一个 16 通道 N 分形，Load3D 设置单位核、单位 stride/dilation、kExtension=16。原核坐标已进入 MTE2 采样与 L0C 列偏移；每轮恢复参数，避免普通路径和拆核参数互相残留。

ComputeSplitKernelHW 按以下骨架组织；其中每个原核位置保留自己的 isFirstMmad，覆盖本段全部批深度与宽片：

```cpp
LocalTensor<T> l0a, l0b;
LocalTensor<float> l0c;
uint32_t baseUseMBak;
ComputeInit(self,p,l0c,baseUseMBak); // includes ClearBaseMNL0C
uint64_t dstL0cOffsetBase = dstL0cOffset_;
uint64_t usedN = uint64_t(curNIdx_)*td.baseN;
uint64_t hwkLoopStart = usedN/16;
uint64_t hwkLoopEnd = (usedN+baseUseN_)/16;
for (uint64_t hwkLoopIdx=hwkLoopStart; hwkLoopIdx<hwkLoopEnd; ++hwkLoopIdx) {
    uint64_t hkIdx=hwkLoopIdx/td.wk, wkIdx=hwkLoopIdx%td.wk;
    dstL0cOffset_ = dstL0cOffsetBase+(hwkLoopIdx-hwkLoopStart)*td.baseM*16;
    initParasSplitKernelHW(self);
    bool isFirstMmad=true;
    uint64_t out2A1BatchDoutSrcAddrStart=p.out2A1SrcAddr;
    uint64_t out2B1BatchDoutSrcAddrStart=p.out2B1SrcAddr;
    for (uint64_t bd=batchDoutStartIdx_; bd<batchDoutStartIdx_+singleShapeBatch_; ++bd) {
        bool skipCurrentDinCompute=false;
        UpdateSrcAddrBaseOnBatchDoutIdx(self,bd,p,skipCurrentDinCompute);
        if (skipCurrentDinCompute) continue;
        uint64_t out2A1SrcAddrStart=p.out2A1SrcAddr, out2B1SrcAddrStart=p.out2B1SrcAddr;
        int32_t woIterateTimes=1;
        calculateWoIterTimes(self,woIterateTimes,td.splitWo);
        for (int32_t splitWoIdx=0; splitWoIdx<woIterateTimes; ++splitWoIdx) {
            updateSingleShapeWoI(self,p,woIterateTimes,splitWoIdx,td.splitWo);
            if (isSplitWo_)
                updateParasForSplitW(self,p,splitWoIdx*td.splitWo,out2A1SrcAddrStart,out2B1SrcAddrStart);
            updateParasForSplitKernelHW(self,p,splitWoIdx*td.splitWo,out2B1SrcAddrStart,wkIdx);
            if (!load3d_.l1W) continue;
            uint64_t curMKL1Idx=stepKaRound*curMIdx_, curNKL1Idx=stepKbRound*curNIdx_;
            ComputeLoop(self,p,l0a,l0b,l0c,isFirstMmad,curMKL1Idx,curNKL1Idx,hkIdx);
        }
        p.out2A1SrcAddr=out2A1SrcAddrStart;
        p.out2B1SrcAddr=out2B1SrcAddrStart;
    }
    p.out2A1SrcAddr=out2A1BatchDoutSrcAddrStart;
    p.out2B1SrcAddr=out2B1BatchDoutSrcAddrStart;
}
dstL0cOffset_=dstL0cOffsetBase;
baseUseM_=baseUseMBak;
```

片段按原核位置、批深度、宽片三层保存并恢复基址；curMKL1Idx/curNKL1Idx 随当前宽片的父段总数重算。拆核配置的每个任务只有一个 ci 分形，hwkLoopIdx 对应当前原核平面内的位置。

initParasSplitKernelHW 设置 Mmad.n=16、Load3D.kExtension=channelSize=16、kStartPt=0、filter/stride/dilation 均为 1、filterSize 高位为 0；每轮 K 更新 mExtension=baseUseK_。MMAD 的目标片段起点使用上述 baseM*16 步幅，因此该路径 Fixpipe 的源 M 跨度也取 baseM，实际 co 写回数仍为 baseUseM_。例如 baseM=144、有效 M=113 时，不能用 alignUp(113,16)=128 代替片段间的 144 跨度。

例如 baseM=32、baseUseN_=48、hwkLoopStart=0，三个原核位置的 L0C float 偏移为 0、512、1024。某个原核位置全无贡献时保留 ComputeInit 建立的零值；首个有效 MMAD 只初始化自己的 16 列片段。全部原核位置完成后才对同一个 L0C EnQue，交给第 8 节统一写回。

## 6 Compute 计算逻辑

ComputeInit 保存真实 M，M=1 时暂改为 16，再依次计算 L1→L0、Mmad 和 GM→L1 参数。DataCopy 的 dValue 仍由真实 tailM_ 决定。ComputeNormal/ComputeSplitKernelHW 在结束时恢复真实 M 后才发布 L0C，见 [单个有效 co](instruction-mapping.md#23-单个有效-co先扩计算-m再恢复写回-m)。

### 6.1 深度采样与完整归约

固定任务 kd 后，在段内遍历全部有效 batch/dout、ho/wo。ReCalDkCinSingleCoreShape 先检查该段是否全部处于无贡献深度，再确定通道有效范围；部分无效深度由核内循环跳过。

段内相同输出子块持有同一 L0C，isFirstMmad 跨换父段、换宽片、换 batch/dout 保留。拆核时还按目标 L0C 子片段维护对应初值状态。

### 6.2 ExecuteMmad / MmadLocal

```cpp
WaitFlag<HardEvent::M_MTE1>(l0aPingPongFlag_);
LoadL12L0a(/* 当前 A 父段及 K 子块 */);
LoadL12L0b(/* 同一 K 子块的 x 窗口 */);
SetFlag<HardEvent::MTE1_M>(l0aPingPongFlag_);
WaitFlag<HardEvent::MTE1_M>(l0aPingPongFlag_);
mmad_.m = baseUseM_;
mmad_.n = baseUseN_; // 拆核对应子片段为 16
mmad_.k = baseUseK_;
mmad_.cmatrixInitVal = isFirstMmad;
MmadLocal(l0a,l0b,l0c);
isFirstMmad = false;
SetFlag<HardEvent::M_MTE1>(l0aPingPongFlag_);
```

上面是交接顺序示意，LoadL12L0a/b 的实参类型及参数构造按第 5 节实现；完整 API 参数片段见 instruction-mapping，事件所有权见 [同步实现](sync-implementation.md#1-流水间交接)。输入槽随实际份数轮转，L0B 为单份时等待其最后消费者，不能机械使用 A 槽号推定另有 B pong。

## 7 输出初始化与主计算通路

### 7.1 输出初始化

实际输出元素数为 `outputSize=uint64_t(cout)*(cin/group)*dk*hk*wk`。清零范围来自实际 dedw，而不是 mValue*nValue 的内部补齐区。

```cpp
// AIV 分支；此代码位于 Conv3dDwInitOutput::Process。
uint64_t clearSizePerCore = alignUp(ceilDiv(outputSize,GetBlockNum()),32);
uint64_t usedClearCores = ceilDiv(outputSize,clearSizePerCore);
if (GetBlockIdx() < usedClearCores) {
    uint64_t offset = uint64_t(GetBlockIdx())*clearSizePerCore;
    uint64_t count = min(clearSizePerCore,outputSize-offset);
    Fill<float>(dedwGm[offset],count,0.0f);
}
// 所有参与 AIV 都调用，包括没有清零分片的单元。
CrossCoreSetFlag<0,PIPE_MTE3>(SYNC_AIV_FLAG);
CrossCoreWaitFlag(SYNC_AIV_FLAG);
CrossCoreSetFlag<2,PIPE_MTE3>(VEC_FALG_ID);
```

AIC 侧配对调用 `CrossCoreWaitFlag<2,PIPE_FIX>(VEC_FALG_ID)`。VEC_FALG_ID=9 沿用当前实现拼写，SYNC_AIV_FLAG 使用所选 SDK 公共常量；标识、作用域、参与单元与初始化 TPipe 生命周期保持一致。32 是 FP32 元素数，对应 128B。

### 7.2 主计算通路

```text
 +-----------------------------------------------------+
 | GET_TILING_DATA + optional system workspace binding |
 +--------------------------+--------------------------+
                            |
                            |
                            v
        +-------------------+-------------------+
        | InitOutput.Init -> Process -> Destroy |
        +-------------------+-------------------+
                            |
                            |
                            v
 +--------------------------+--------------------------+
 | AIC: Init -> Process -> CalBasicBlock -> IterateAll |
 +--------------------------+--------------------------+
                            |
                            |
                            v
+---------------------------+---------------------------+
| Process: release held buffers -> End -> TPipe cleanup |
+-------------------------------------------------------+
```

streamkType=0、groupEnlarge=false 时，AIV 清零、汇合并通知后返回；AIC 独立完成任务。主循环没有 AIC 等待 AIV 的跨核同步点，因此不需要 AIV 陪跑。若扩展到包含跨核归约的路径，应同时实现 AIV 配对路径；不能只添加 AIC 侧等待。

## 8 CopyOut 输出通路逻辑

### 8.1 LoadL0c2Gm：NZ 到外部格式

GetTensorC/LoadL0c2Gm 先从当前 CO1 队列 DeQue，再按输出格式、核体积、baseN 整窗关系选写回函数。普通 NCDHW 使用 COLUMN_MAJOR，普通 NDHWC 使用 ROW_MAJOR，DHWCN 使用 COLUMN_MAJOR；1×1×1 的简化分发单独处理。

非整窗判定为 `(baseN/16)%(hk*wk)!=0`，按 [非整窗输出](instruction-mapping.md#73-非整窗输出与归还) 的参数、地址和状态递推执行。head_/tail_ 使用核位置单位；nValueSum 使用已写核位置数；cinRemainLen_ 使用真实 ci 数。每个 IterateAll 初始化一次状态，同一 N 的不同 M 写回须恢复 nLoopHead_/nLoopCinRemainLen_，不能继承前一 M 的终态。

### 8.2 写回与归还

```cpp
auto l0c = usePing ? l0cPing_.DeQue<float>() : l0cPong_.DeQue<float>();
SetAtomicAdd<float>();
LoadL0c2GmForTransFormat(dedwGm,l0c);
SetAtomicNone();
if (usePing) l0cPing_.FreeTensor(l0c);
else l0cPong_.FreeTensor(l0c);
if (td.cl0Pbuffer > 1) l0cPingPongFlag_ = !l0cPingPongFlag_;
```

每个写回调用只合并本段结果；offsetC_ 已含任务起点，内部函数只加子块偏移。最后一次 F 读取提交后归还 CO1，后续分配等待实际完成。任务或形状在 Compute 前被跳过时不写回；已申请 L0C 而无实际贡献时，按 [零结果交接](sync-implementation.md#11-l0c-交接) 完成清零后正常交接。

## 9 流水同步

[本分支同步文档](sync-implementation.md) 按两路父段、实际输入槽和归约段定义协议：

| Buffer/阶段 | 就绪 | 归还与复用 |
|---|---|---|
| L1A/L1B | MTE2 后 EnQue，MTE1 前 DeQue | 最后一次 MTE1 读取后 FreeTensor，下次 Alloc 等待 |
| L0A/L0B | MTE1_M | M_MTE1；按实际输入槽轮转 |
| L0C | 本段全贡献后 EnQue/DeQue，M→F | F 后 FreeTensor，F→M |
| 输出初始化 | AIV 汇合后 MODE2 通知 | AIC PIPE_FIX 等待后才能原子写回 |

```text
L1:  MTE2 -- ready --> MTE1 -- return --> next MTE2 overwrite
L0:  MTE1 -- ready --> M    -- return --> next MTE1 overwrite
L0C: M    -- ready --> F    -- return --> next M output initialization

First slot: queue / verified TPipe initial state
Final use : last consumer dependency -> queue End / TPipe cleanup
```

首次空槽和固定事件初态由已核实的 TPipe/队列实现提供；生成类不重复初始化或额外等待同一令牌。父段按最后消费者归还；End 在 Process 的统一结束处调用一次，仅释放实际创建队列的事件。TPipe 生命周期覆盖末次 F 和固定事件收尾。

SetScheduleMode(1)、MIX 模板声明、实际 blockDim 与初始化参与单元作为同一启动契约记录。提交了消费者后可以发布归还依赖，但“可再次覆盖”由后续等待保证。

## 10 Kernel 产出校验

- [ ] 入口六参数、共享字段、模板组合与 Host 发布一致。
- [ ] 所有参与单元先完成初始化协议，之后无任务核才退出。
- [ ] 每个 g/kd/M/N 任务唯一归属，原始组通道步幅正确。
- [ ] batch/dout/ho 段覆盖完整，部分无效深度不会造成非法 GM 访问。
- [ ] 父段真实跨度、预留容量和本轮有效长度分别传递。
- [ ] ND2NZ/DN2NZ、Load2D、Load3D 与各自布局成对。
- [ ] 普通窗口、拆核、非整窗 N 均有对应参数与写回逻辑。
- [ ] 单个有效 co 经 ComputeInit 扩计算 M 后，DataCopy 仍只读真实通道，写回前恢复有效 M。
- [ ] 宽片尾部重算 kIter_/tailK_/父段数，各槽恢复自身真实跨度与补边。
- [ ] 非整窗写回在切换 M 时重放该 N 的初态，在新任务/段重新初始化，不产生零长度片段。
- [ ] isFirstMmad 只随实际贡献改变，段内保留、段间重建。
- [ ] FP32 原子写回只覆盖实际输出，状态恢复为 AtomicNone。
- [ ] 预取、最后消费者、槽回绕、End 和 TPipe 收尾全部闭合。

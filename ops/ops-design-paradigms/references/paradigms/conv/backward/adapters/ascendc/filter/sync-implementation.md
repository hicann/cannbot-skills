# filter 分支同步与初始化实现

本页把 [计算流程的生命周期](../../../generic/filter/compute-flow.md#8-dagbuffer-与流水) 映射为队列、事件和初始化通知。当前为 MN_StreamK、streamkType=0、groupEnlarge=false：A=dedy，B=x，累加与 dedw 均为 FP32。资源能力见 [buffer-and-types](../buffer-and-types.md)，本分支的队列实例和物理槽见 [Kernel 模板](kernel-template.md#33-buffer-分配)。

## 1 流水间交接

| 对象 | 就绪与归还机制 | 释放时机 |
|---|---|---|
| L1A | a1Ping_/a1Pong_：AllocTensor→DataCopy→EnQue→DeQue→Load2D→FreeTensor | 当前父段最后一次 MTE1 读取已提交，且 isFreeAL1 允许释放 |
| L1B | b1Ping_/b1Pong_：AllocTensor→DataCopy→EnQue→DeQue→Load3D→FreeTensor | 当前父段最后一次 MTE1 读取已提交，且 isFreeBL1 允许释放 |
| L0A/L0B | MTE1_M 表达就绪，M_MTE1 表达归还；由 TBuf 提供数据槽 | 当前 MMAD 读取已提交，后续覆盖等待归还 |
| L0C | l0cPing_/l0cPong_ 为 CO1 队列，表达 M→F 与 F→M | 当前归约段的 Fixpipe 读取已提交，下一次分配等待归还 |

两路 L1 按各自 step 换驻；L0 输入按实际 MMAD 换槽；L0C 按完整输出子块与归约段交接。三个频率分别管理。

普通窗口的 ComputeLoop 使用以下稳态顺序。k 为本次 K 循环内的全局块号，kaIdx/kbIdx 为各自父段内块号；self 为实现接口，out2L1Params、各路父段标志和 LocalTensor 已由本轮状态确定：

```cpp
WaitFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_ & 1);
ExecuteMTE1L0b(self, out2L1Params, l0b, b1PingPongFlag,
              isLoadB1, isLastStepKb, isLastKIter, k, kbIdx);
ExecuteMTE1L0a(self, out2L1Params, l0a, a1PingPongFlag,
              isLoadA1, isLastStepKa, isLastKIter, k);
ExecuteMmad(self, l0a, l0b, l0c, isFirstMmad);
SetFlag<HardEvent::M_MTE1>(self->ctx.l0aPingPongFlag_);
self->ctx.l0aPingPongFlag_ ^= self->ctx.useL0PingPong_;
```

ExecuteMTE1L0a/b 在首次使用父段时 DeQue，完成 LoadL12L0a/b 后按释放标志调用 FreeA1Tensor/FreeB1Tensor。ExecuteMmad 内部先 Set/Wait MTE1_M，再设置 cmatrixInitVal 和本轮有效 k，执行 MmadLocal。

`useL0PingPong_=(al0Pbuffer-1)&(bl0Pbuffer-1)`。两路均为两份时取 1，交替使用输入半区；任一路单份时取 0，复用同一输入槽。不能仅按 al0Pbuffer 推定 L0B 也能轮换。拆核通路使用相同的交接原则，采样和 L0C 子片段由原 kh/kw 决定。

### 1.1 L0C 交接

一个 L0C 保存一个输出子块在当前归约段内的全部贡献，跨 batch/dout、宽片和 K 父段保持累加；段结束后经 Fixpipe 原子累加到 dedw。

```text
    +-----------------------------------------+
    | AllocTensor<float>: segment output slot |
    +--------------------+--------------------+
                         |
                         |
                         v
+------------------------+------------------------+
| all actual MMADs; or ClearBaseMNL0C zero result |
+------------------------+------------------------+
                         |
                         |
                         v
             +-----------+-----------+
             | EnQue / DeQue: M -> F |
             +-----------+-----------+
                         |
                         |
                         v
       +-----------------+-----------------+
       | Fixpipe: atomicAdd to actual dedw |
       +-----------------+-----------------+
                         |
                         |
                         v
 +-----------------------+-----------------------+
 | FreeTensor: publish F -> M; next Alloc waits  |
 +-----------------------------------------------+
```

区分两种无贡献情况：

- **任务或形状在 Compute 前被跳过**：没有申请 L0C，也不生产或等待该段的 M→F 事件。
- **Compute 已申请 L0C，但段内无有效 MMAD**：普通路径调用 ClearBaseMNL0C，用零操作数 MMAD 建立零结果，再正常 EnQue/DeQue、写回和归还。拆核路径在 ComputeInit 中先清零整个 L0C，各核位置随后按自身首次贡献状态累加。

ClearBaseMNL0C 本身包含 L1 零值准备、M_MTE1 等待、MTE1_M 就绪与 MMAD，不能只发布一个没有数据生产者的 M→F 令牌。清零 dedw GM 也不能替代已分配 L0C 的初始化。

例如两个归约段分别产生 C0、C1，执行顺序为 `dedw=0 → atomicAdd(C0) → atomicAdd(C1)`。每段的首个实际乘加初始化自己的部分和；新段不能把上一段 L0C 再次当作初值。

### 1.2 初态与收尾的所有权

| 资源 | 初态与运行 | 结束处理 |
|---|---|---|
| L1A/L1B 队列 | InitBuffer 建立空槽；Alloc/EnQue/DeQue/FreeTensor 成对交接；每槽有自己的队列 | 按实际最后消费者归还张量，End 只对实际创建的队列调用 FreeAllEvent |
| L0C 队列 | 首次 AllocTensor 取得空槽，重用时等待前次 F 归还 | 每次写回后 FreeTensor，End 释放实际 CO1 队列事件 |
| M_MTE1 固定事件 | 所选 SDK 的 TPipe 建立初始令牌；每轮 Wait 后，MMAD 提交后 Set | TPipe 完成末次等待与事件释放，包括未使用槽的初始令牌 |
| MTE1_M 固定事件 | ExecuteMmad 每轮 Set/Wait；ClearBaseMNL0C 使用自己的配对调用 | 已成对消费，不再追加无生产者的末次 Wait |

当前映射依赖已核实的 TPipe 固定事件约定；更换 SDK 时先检查其构造和销毁实现。生成类不再手动 Set 同一个初始令牌，也不在 End 重复消费 TPipe 将等待的末次 M_MTE1。

Conv3dDwBasicBlockMNStreamK::Process 在任务循环完成后调用一次 dw_.End；入口不再调用第二次。父段由正常释放路径归还，End 调用 FreeAllEvent，TPipe 生命周期覆盖最后一次 F 和固定事件收尾。初始化使用自己的 TPipe，Destroy 完成后才创建主计算资源。

## 2 L1 父段驻留与预取

父段长度分别为 `kal1_=stepKa*baseK`、`kbl1_=stepKb*baseK`。当前实际长度与步幅随尾段、切宽和有效行带更新，不能只复用张量句柄而留下旧参数。

| 状态 | 作用 |
|---|---|
| k | 当前 ComputeLoop 内的全局 K 块号，元素起点为 `k*baseK` |
| kaIdx/kbIdx | 父段内块号，分别为 `k%stepKa`、`k%stepKb` |
| kaStepIdx/kbStepIdx | 父段号，分别为 `k/stepKa`、`k/stepKb`，取整数除法 |
| curMKL1Idx/curNKL1Idx | 当前 M/N 子块对应的父段基号 |
| a1PingPongFlag/b1PingPongFlag | 当前 L1 槽，true 为 ping；不是 L0 输入槽号 |
| isLoad2L1A/B | 当前父段是否需要真实搬入 |
| isFreeAL1/BL1 | 当前父段是否可在末次读取后释放 |
| isA1NormalLoad/isB1NormalLoad | 首次正常装载或已有预取的状态 |
| skipCurrentHiCompute/PreLoad | 当前/预取 x 行带没有有效贡献 |

MTE2 搬运起点由父段号乘 step*baseK 确定。实现的 LoadToA1 下层同名形参 kaIdx 接收全局 k，详见 [指令映射](instruction-mapping.md#21-loadtoa1-实现链与局部量)；父段内块号只用于局部寻址与换驻判断。

启用两份 L1 时，槽选择分别为 `(curMKL1Idx+kaStepIdx+1)&1` 与 `(curNKL1Idx+kbStepIdx+1)&1`；单份使用 ping。该奇偶关系只选择物理槽，Alloc/DeQue/FreeTensor 仍负责保证槽的可用性。

普通路径先检查 B 行带，再准备 A。当前 B 无效时跳过本轮，不执行对应的 L0 等待和 MMAD；若 B 预取已判定下一父段无效，ComputeLoadToA1PreLoad 同样停止该次 A 预取，避免产生无人消费的 A 队列元素。

跳过分支仍调用 UpdateIdx 推进 kaIdx/kbIdx 和父段号，外层 k 正常加 1；只取消本轮未生产数据的等待与消费。isA1NormalLoad/isB1NormalLoad 恢复为 true，使下一有效父段重新走正常装载。

| 时刻 | 当前槽 | 另一槽 | 必须成立 |
|---|---|---|---|
| 首次装载 | Alloc→DataCopy→EnQue | 空闲 | 只为实际存在的父段装载 |
| 首次消费 | DeQue→Load2D/Load3D | 可预取下一父段 | 等待当前 MTE2 就绪 |
| 中间复用 | 保留已出队的父张量 | 保留已预取父张量 | 两路消费同一个 K 坐标 |
| 最后消费 | FreeA1Tensor/FreeB1Tensor | 准备成为当前槽 | 满足 isFree* 与末次读取条件 |
| 槽回绕 | 下一次 Alloc 等待归还 | 按各自 step 推进 | 不覆盖仍被 MTE1 使用的数据 |

例如 stepKa=2、stepKb=4、当前有 5 个 K 子块，两路父段对应如下：

```text
+---------------------+---+---+---+---+---+
| K block             | 0 | 1 | 2 | 3 | 4 |
+---------------------+---+---+---+---+---+
| A parent / stepKa=2 | 0 | 0 | 1 | 1 | 2 |
| B parent / stepKb=4 | 0 | 0 | 0 | 0 | 1 |
+---------------------+---+---+---+---+---+
```

A 路令 `nextK=k+stepKa`，B 路令 `nextK=k+stepKb`，各自仅在 `nextK<kIter_` 时预取。末段不预取不存在的下一段；段内跨 batch/dout 或宽片时重新计算父段状态。M/N 之间允许的父段复用由 UpdateMNIdx 与 isLoad2L1*/isFree* 共同决定。

拆核路径使用 LoadToB1SplitKernelHW，固定原 kh/kw 后搬入采样行带；不直接套用普通路径的 A/B 预取调度。原核位置、组、输入通道或数据地址改变时，按该路径重建父段身份与释放条件。

### 2.1 UpdateMNIdx：首次与后续子块的复用条件

iterateOrder 的实际取值为 ORDER_M=0、ORDER_N=1。ORDER_N 先递增 M，可保留 B；ORDER_M 先递增 N，可保留 A。以下标志是 Out2L1ScalarParams 的字段，先决定本子块是否需要真实搬运，再由 ComputeLoop 的父段边界决定具体装载时刻。

当前 Host 发布 singleCoreM=baseM，有效任务的 M 不超过 baseM，因此 mIter_=1。以下保留公共状态函数的 M/N 两种推进形式；M 多子块的例子用于函数级理解与校验，不改变当前 Host 的任务范围。

首次设置 `curMIdx_=curNIdx_=0`、`isFirstIter_=false`，两路 isLoad2L1 均为 true。首次归还条件为：

| 情况 | isFreeAL1 | isFreeBL1 |
|---|---|---|
| 拆核 | true | true |
| 普通窗口 | `kIter_>stepKa*al1Pbuffer \|\| singleShapeBatch_>1 \|\| isSplitWo_ \|\| iterateOrder==1 \|\| (iterateOrder==0 && nIter_==1)` | `kIter_>stepKb*bl1Pbuffer \|\| singleShapeBatch_>1 \|\| isSplitWo_ \|\| (iterateOrder==1 && mIter_==1) \|\| iterateOrder==0` |

后续子块先重置四个标志：拆核或 isSplitWo_ 时全部置 true；其余情况，A 路 isLoad2L1A/isFreeAL1 均取 `kIter_>stepKa*al1Pbuffer || singleShapeBatch_>1`，B 路对应取 `kIter_>stepKb*bl1Pbuffer || singleShapeBatch_>1`。然后按下面的顺序推进，并补上该遍历方式必须发生的重载和末次归还：

```cpp
if (td.iterateOrder == 1) { // UpdateMNIdxOrderN
    if (++curMIdx_ >= mIter_) {
        curMIdx_ = 0;
        if (++curNIdx_ >= nIter_) { curNIdx_ = 0; return false; }
        p.isLoad2L1B = true;
    }
    p.isLoad2L1A = true;
    p.isFreeAL1 = true;
    if (curMIdx_+1 == mIter_ && curNIdx_+1 == nIter_) p.isFreeBL1 = true;
} else { // UpdateMNIdxOrderM
    if (++curNIdx_ >= nIter_) {
        curNIdx_ = 0;
        if (++curMIdx_ >= mIter_) { curMIdx_ = 0; return false; }
        p.isLoad2L1A = true;
    }
    p.isLoad2L1B = true;
    p.isFreeBL1 = true;
    if (curMIdx_+1 == mIter_ && curNIdx_+1 == nIter_) p.isFreeAL1 = true;
}
return true;
```

例如 kIter_=2、stepKa=stepKb=1、两路 L1 均双份、singleShapeBatch_=1 且不切宽，分别单独改变一个输出方向：

| 顺序 | 子块序列 | 可以保留的数据 |
|---|---|---|
| ORDER_N，mIter_=2、nIter_=1 | `(0,0)→(1,0)` | 两个 B 父段跨 M 保留，进入末子块后归还 |
| ORDER_M，mIter_=1、nIter_=2 | `(0,0)→(0,1)` | 两个 A 父段跨 N 保留，进入末子块后归还 |

若 kIter_ 超过某一路全部 L1 槽可容纳的 K 块数，该路必须在父段末次读取后释放；若 singleShapeBatch_>1 或切宽，两路源身份会在同一个 Compute 内变化，均逐父段重载。curMKL1Idx=stepKaRound*curMIdx_、curNKL1Idx=stepKbRound*curNIdx_ 与槽奇偶关系一起使用。

### 2.2 装载、消费与归还必须成对

| 阶段 | A 路条件 | B 路条件 |
|---|---|---|
| 当前父段入口 | kaIdx==0 | kbIdx==0 |
| 真实 Alloc/Copy/EnQue | 上述入口且 isLoad2L1A；已有预取时不重复装载 | 上述入口且 isLoad2L1B；先判断有效行带 |
| DeQue | 真实父段入口且 isLoad2L1A | 真实父段入口且 isLoad2L1B |
| 当前 MTE1 | 使用所选槽的缓存张量及 alignedL1UseKa | 恢复所选槽的 hiCopyLen/padUp，再使用缓存张量 |
| FreeTensor | `isFreeAL1 && (kaIdx+1==stepKa \|\| k+1==kIter_)` | `isFreeBL1 && (kbIdx+1==stepKb \|\| k+1==kIter_)` |

保留父段时缓存张量已经出队，后续 M/N 子块不能再次 DeQue。同一个宽片内，预取使用相反槽；只有下一个父段存在且该路需要真实搬入时才入队。末次使用必须产生归还依赖，包括此前保留到最后一个 M/N 子块的父段。

## 3 输出初始化

当前归约段通过 FP32 原子累加合并到 dedw，因此每次执行都先为实际 dedw 建立零基值。初始化不依赖 input 分支的 initOutputFlag。

`outputSize_=uint64_t(cout)*(cin/group)*dk*hk*wk`，单位为 FP32 元素；清零覆盖实际连续输出，不包含 mValue/nValue 的内部补齐区。NCDHW、NDHWC、DHWCN 的元素总数相同。

### 3.1 范围分配

Conv3dDwInitOutput<float> 在 AIV 执行 OutputClearZeroWithUb2Gm。下列为该函数主体，AlignUp/Ceil 为工程内的对齐与向上取整除法：

```cpp
uint32_t usedCoreNum = GetBlockNum();
uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_,usedCoreNum),32);
usedCoreNum = Ceil(outputSize_,clearSizePerCore);
if (GetBlockIdx() >= usedCoreNum) {
    SyncAllCores();
    return;
}
uint64_t offset = clearSizePerCore*GetBlockIdx();
uint64_t count = outputSize_ - offset;
count = count > clearSizePerCore ? clearSizePerCore : count;
auto yPart = yGm_[offset];
Fill<float>(yPart,count,0.0f);
SyncAllCores();
```

32 是 BEST_FIXPIPE_ELEMENTS 对应的 **32 个 FP32 元素，即 128B**；count 和 offset 均为元素数。尾片只写真实剩余长度，未分到清零数据的 AIV 也调用 SyncAllCores。

例如 outputSize_=100、清零视角 GetBlockNum()=4，clearSizePerCore=32，编号 0～3 分别清零 32、32、32、4 个元素。其他参与 AIV 无数据分片，但仍到达汇合点。GetBlockIdx 的 AIV 索引与主计算 block_idx 分开使用，不能把有计算任务的 AIC 数量当作清零参与单元集合。

### 3.2 AIV 汇合与 AIC 等待

VEC_FALG_ID=9 沿用实现拼写；SYNC_AIV_FLAG 使用 SDK 公共常量。AIV 先通过模式 0 汇合，再发布模式 2 通知：

```cpp
// Conv3dDwInitOutput::SyncAllCores，所有参与 AIV 调用。
CrossCoreSetFlag<0,PIPE_MTE3>(SYNC_AIV_FLAG);
CrossCoreWaitFlag(SYNC_AIV_FLAG);
CrossCoreSetFlag<2,PIPE_MTE3>(VEC_FALG_ID);
```

Conv3dDwInitOutput::Process 中，AIC 使用同一标识等待：

```cpp
if ASCEND_IS_AIV {
    OutputClearZeroWithUb2Gm();
}
if ASCEND_IS_AIC {
    CrossCoreWaitFlag<2,PIPE_FIX>(VEC_FALG_ID);
}
```

模式、流水、标识和参与单元作为一套协议传递。AIC 只能在全部清零完成后开始原子写回；没有主计算任务的核也先完成该初始化协议。主计算的 L0C→GM 使用 F，MTE3 在此承担初始化通知依赖。

### 3.3 入口顺序

```text
                          +-----------------------------------------------+
                          | GET_TILING_DATA; system workspace binding     |
                          | InitOutput.Init                               |
                          +-----------------------+-----------------------+
                                                  |
                       +--------------------------+--------------------------+
                       |                                                     |
                       v                                                     v
 +---------------------+---------------------+         +---------------------+---------------------+
 | AIV: InitOutput.Process                   |         | AIC: InitOutput.Process                   |
 | Fill actual dedw                          |         | wait MODE2 / PIPE_FIX                     |
 | MODE0 join -> MODE2 notify                |         | initialization complete                   |
 +---------------------+---------------------+         +---------------------+---------------------+
                       |                                                     |
                       |                                                     |
                       v                                                     v
      +----------------+----------------+                   +----------------+----------------+
      | InitOutput.Destroy              |                   | InitOutput.Destroy              |
      +----------------+----------------+                   +----------------+----------------+
                       |                                                     |
                       |                                                     |
                       v                                                     v
      +----------------+----------------+              +---------------------+---------------------+
      | AIV: return                     |              | AIC: main Init -> Process                 |
      +---------------------------------+              | CalBasicBlock -> IterateAll               |
                                                       +---------------------+---------------------+
                                                                             |
                                                                             |
                                                                             v
                                                       +---------------------+---------------------+
                                                       | Process: dw_.End once                     |
                                                       | main TPipe cleanup                        |
                                                       +-------------------------------------------+
```

入口依次调用 Init、Process、Destroy。streamkType=0、groupEnlarge=false 时，AIV 清零、汇合并通知后返回；AIC 处理拥有的输出任务，Process 内完成 End。本配置主循环没有逐基本块的 AIC↔AIV 归约等待。

启动的 MIX 核类型、AIC/AIV 配比、blockDim、SetScheduleMode(1) 与该协议一起核验，见 [模板绑定](tilingkey-templates.md)。平台能力通过 platformInfo 和 SDK 确定。

## 4 直调和注册接入

注册和直调均可使用上述融合初始化入口：每次调用清零 dedw，再做全部归约段的原子写回。用户 workspace=0 不省略初始化；平台/系统 workspace 与用户归约区分别处理。

直调也可将独立 aclrtMemsetAsync 与计算放在同一 stream。采用独立初始化时，同时移除设备内清零和 AIC 等待，按剩余执行路径重新声明核类型；跨 stream 使用明确的完成事件。不能只删 AIV 分支而保留初始化 WaitFlag。

Host Tiling 只产生计划，不提交设备清零。dedw、输入、Tiling 与 workspace 的生命周期覆盖 stream 完成；新一轮清零不能与上一轮尚未结束的 Fixpipe 重叠。同一输出缓冲区重复执行时，每次重新建立零基值。

## 5 片上补零与检查

| 初始化位置 | 作用 | 同步责任 |
|---|---|---|
| dedw GM | 全部归约段的原子累加基值 | AIV 汇合、模式 2 通知、AIC 等待 |
| L1 父段 | 拆核采样留下的零区域、ClearBaseMNL0C 的零操作数 | InitZeroValue 的 Fill 后 PIPE_MTE2 barrier；随后 EnQue/DeQue |
| L0C | 普通首个有效贡献，或已分配但无贡献的零结果；拆核先清全块 | M_MTE1/MTE1_M、MMAD 初值、CO1 队列 |

InitZeroValue 使用已分配槽的完整长度，长度按 32B 对齐；Fill 的 blockNum 单位为 32B。字段范围在构造前检查，超出单次指令范围时按相同单位分批：

```cpp
const uint64_t lenBytes = uint64_t(buf.GetSize())*sizeof(T);
InitConstValueParams<T> p{};
p.repeatTimes = 1;
p.blockNum = lenBytes/32;
p.dstGap = 0;
p.initValue = T(0);
Fill(buf, p);
PipeBarrier<PIPE_MTE2>();
```

ClearBaseMNL0C 的零操作数分别经 A1/B1 队列 Alloc→Fill→EnQue→DeQue；等待当前 M_MTE1 后，以不转置 Load2D 和单位窗口 Load3D 形成零操作数。清零 MMAD 使用 m=baseM、n=baseN、k=16、cmatrixInitVal=true，MTE1_M 成对同步；完成后发布 M_MTE1 并按实际输入槽掩码换槽。临时 A1/B1 张量在末次 MTE1 读取后归还，不能保留到主路径再次申请。

Load3D 的 padList 表达原 x 窗口的边界；通道补齐表示存储尾部。这些零各有坐标范围，不能用一次 GM 清零代替片上未初始化数据。InitZeroValue 的 PIPE_MTE2 barrier 只保证该次 Fill 先于后续 MTE2 操作，不替代 MTE2→MTE1 或 M→F 的依赖。

产出时按下列场景检查：

- [ ] 不同 stepKa/stepKb、单份/双份 L1 和输入槽组合，均无覆盖或队列元素泄漏。
- [ ] 首次消费、预取命中、B 当前/预取行带跳过、K 尾段和槽回绕都只等待实际生产的令牌。
- [ ] 普通已分配零贡献块走 ClearBaseMNL0C；拆核 L0C 初始清零与逐核位置初值保持一致。
- [ ] 每段全部贡献在发布 L0C 前完成，原子状态只包围本次写回并恢复 SetAtomicNone。
- [ ] 非整窗 N 与通道尾部只写有效 dedw，归还发生在最后一次 F 读取提交后。
- [ ] 使用非零预填输出覆盖无贡献核位置、多个归约段、尾清零分片和重复执行。
- [ ] 无清零分片的 AIV、无主计算任务的 AIC 仍按初始化协议汇合与等待。
- [ ] Process、End、FreeAllEvent 与两个 TPipe 的初态和收尾所有权唯一。

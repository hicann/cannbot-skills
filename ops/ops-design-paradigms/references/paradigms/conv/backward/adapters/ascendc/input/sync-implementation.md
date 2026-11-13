# 卷积反向算子同步与初始化实现

本页把 [计算流程的生命周期](../../../generic/input/compute-flow.md#8-dagbuffer-与流水) 映射为队列、事件和初始化通知。数据槽按 [buffer-and-types](../buffer-and-types.md) 分配。

## 1 流水间交接

| 对象 | 就绪与归还机制 | 释放时机 |
|---|---|---|
| L1A/L1B | `AllocTensor → DataCopy → EnQue → DeQue → LoadData → FreeTensor`；队列表达 MTE2→MTE1、MTE1→MTE2 依赖 | 当前驻留段最后一次 MTE1 读取已提交后 |
| L0A/L0B | `HardEvent::MTE1_M` 表达就绪，`HardEvent::M_MTE1` 表达归还 | 本轮 MMAD 读取已提交后 |
| L0C | CO1 队列在完整归约后交接，表达 M→F、F→M 依赖 | F 读取已提交后，下一次分配等待归还 |
| bias L1/BT，可选 | A1/C2 队列表达 MTE2→MTE1→M 及覆盖依赖 | 分块 L1 在复制到 BT 后归还；BT 在使用它的 MMAD 后归还；全量持有到身份变化/结束 |

L1A/L1B 分别管理驻留段和队列；更新频率来自各自的 `stepKa/stepKb`。L0A/L0B 成对轮换，稳态交接如下：

```cpp
WaitFlag<HardEvent::M_MTE1>(l0Slot);
LoadToA2(kStart,baseUseK_);
LoadToB2(kStart,baseUseK_);
SetFlag<HardEvent::MTE1_M>(l0Slot);
WaitFlag<HardEvent::MTE1_M>(l0Slot);
MmadLocal<hasBias>(useC1Buf,baseUseK_,firstContribution);
SetFlag<HardEvent::M_MTE1>(l0Slot);
l0Slot = (l0Slot+1) % inputSlotCount;
```

这里将原 ComputeForKIter 的稳态事件顺序映射到本工程阶段；L0A/L0B 使用相同份数并成对轮换。L1A/L1B 准备、L0A/L0B 句柄和参数更新见 [Kernel 模板](kernel-template.md)。`FreeTensor` 与 `SetFlag` 在最后一次读取提交后发布归还依赖，下次分配或写入等待该依赖。

### 1.1 L0C 交接

L0C 的每槽 CO1 队列按以下顺序交接：

```text
AllocTensor<float>  -- first: free slot; reuse: wait FIX_M
    |
    v
All valid MMADs     -- hold the same L0C
    |
    v
EnQue / DeQue       -- publish / wait M_FIX
    |
    v
Fixpipe
    |
    v
FreeTensor          -- publish FIX_M; slot reusable
```

首次分配由队列初态管理；不额外等待一个尚未发布的 `FIX_M`。F 内部同流水 barrier 只排序 F 指令，M→F 仍由 CO1 队列保证。初始化 L1A 的 `PIPE_MTE2` barrier、L1A/L0A 的队列/事件和 L0C 的队列各自覆盖不同依赖。

### 1.2 初态与收尾的所有权

本模板沿用 TPipe 管理固定 `M_MTE1` 事件的方式。生成前核实目标 SDK 的构造、析构与队列实现，按以下分工绑定；不在 Kernel 中再次初始化或等待同一令牌。

| 资源 | 初态与运行 | 结束处理 |
|---|---|---|
| L1A/L1B、L0C 队列 | InitBuffer 建立空槽；Alloc/EnQue/DeQue/FreeTensor 管理交接 | Process 归还持有张量；End 对各实际队列调用 FreeAllEvent |
| M_MTE1 固定事件 | TPipe 构造预留并 Set；每轮 Wait 后，MMAD 提交后重新 Set | TPipe 析构等待并释放，包括未使用槽仍保留的初始令牌 |
| MTE1_M 固定事件 | 每次 LoadToA2/L0B 后 Set，紧接着为该次 MMAD Wait | 每次就绪事件已成对消费，不再追加末次 Wait |

`FreeB1Tensor`、`End`、`FreeAllEvent` 沿用实现中的标识。FreeB1Tensor 在全部任务结束后仅归还仍持有的 L1B；End 只等待队列归还并释放队列事件。End 由入口调用一次，Process 内不重复调用；TPipe 生命周期覆盖这些操作及最后一次 F。目标 SDK 若不具备上述固定事件约定，应先完成等价同步适配，再启用本 Kernel 映射。

`UpdateL1ComputeInfo` 在申请 L0C、等待输入槽前确定 `needComputeFlag_`。空行带或没有有效核深度时，无 bias 的 NoNeedComputeHandle 归还父段后返回；带 bias 的分支通过 LoadL0Zero、MMAD 和 F 完成纯 bias 输出。非空行带中只有 stride 插零时，保留正常零窗口通路。两条路径仅等待实际生产的事件。

存在 bias 时，Process 还归还持有的 bias L1/BT，End 对实际创建的队列调用 FreeAllEvent。全量与分块模式的归还位置见 [bias](bias-bt-path.md)，同一张量只释放一次。DX 不创建或等待 bias 资源。

## 2 L1B 驻留

`l1-resident` 与其他模板使用相同的搬运、计算接口，区别在 L1B 的保留范围。完整驻留时，L1B 在首次 `DeQue` 后持续持有，多次 L0B 准备共用该张量；模板确定复用结束后再 `FreeTensor`。

Host 的 `enableFullLoad`、Kernel 的 `curEnableFullLoad_` 和 `isB1FullLoadFlag_` 分别记录策略、当前任务是否保持驻留、父段是否完整驻留。

`InitFullLoadFlag` 检查 L1B 覆盖当前任务的全部 N/K、`bl1Pbuffer=1` 和单次核深度范围。跨任务保留还要求 `dk=1`、原始单组和 `curEnableFullLoad_`；`CheckFullLoadEnable` 在权重 `offsetB_` 改变时释放旧 L1B 并关闭当前复用。完整条件见 [Kernel 流水同步](kernel-template.md#9-流水同步)。

持有 L1B 时，后续 M 块沿用该 LocalTensor，按本轮有效 N/K 更新 L0B 参数；`isLoadB1_/isFreeB1_` 决定首次出队和最终释放。L0A/L0B 每轮仍经过 MTE1→M→MTE1 交接，父布局步幅保持驻留时的值。

## 3 输出初始化

当前消费 `InitOutputFlag::NO_INIT=0`、`InitOutputFlag::L0_INIT=1`。Host 保留前置检查和后续 Tiling 的共同结果，以下条件适用于 DX 形状：

| 阶段 | 置 initOutputFlag=1 的条件 |
|---|---|
| SetInitOutput 前置检查 | `ρ_d>padBack`，或 `ρ_h>padDown`，或 `strideH>hk`，或 `dilationD>1` |
| 当前常规核窗口 Tiling | `strideD>dk` |

`ρ_d=(di+padFront+padBack-((dk-1)*dilationD+1))%strideD`，H 轴同理；形状合法性检查先保证余数口径成立。这些条件来自无贡献区域与跳过路径，不能只用其中一条代替完整判断。`SetRunInfoTiling` 写入前置结果，后续判断可以置位，`SetCommonTilingData` 发布最终值。

Transpose 按其 [前置处理](../../../generic/preprocessing.md#5-执行前置约定) 计算余数并加入 qD/qH 条件，之后复用本节执行协议。带 bias 时清零仍在主计算之前，主计算对无贡献位置写入 bias，清零本身不构成最终结果。

例如 D 轴取 `di=5、dout=2、dk=2、strideD=dilationD=2、前后pad=0`，深度1、3没有有效贡献；此时 strideD 与 dk 相等，仍由 dilationD 条件启用清零。H/W 均为单维、核长和 stride/dilation 为1时，取 dedy=[1,2]、weight=[3,4]，dedx 深度序列为 [3,0,10,0,8]。

这里的 `L0_INIT` 是枚举名称；此步骤实际通过 `Fill(GlobalTensor,...)` 清零 GM，不是主计算的 L0C 初始化。不要把这个字段改成另一个含义不同的布尔开关。

```text
outputSize_ = uint64(hi)*wi*cin*di*batch       // 输出元素数
initBytes   = outputSize_*sizeof(DTYPE_Y)    // 调用侧容量检查，字节
```

目标为本次连续 dedx 的完整有效区域。Host 检查乘积、分配容量及输出与输入的地址关系；设备按最终标志执行。`NO_INIT` 路径仍需验证主计算的有效写回范围。

### 3.1 范围分配

工程内 InitOutput 按 AIV/AIC 分支执行清零与等待，分片规则与原 ProcessWithL0 一致。FP16/BF16 的清零主体为：

```cpp
uint64_t clearSizePerCore = AlignUp(Ceil(outputSize_, GetBlockNum()), ONE_BLK_SIZE);
if (GetBlockIdx() >= Ceil(outputSize_, clearSizePerCore)) {
    SyncAllCores();
    return;
}
uint64_t offset = clearSizePerCore * GetBlockIdx();
uint64_t realClearSize = outputSize_ - offset;
realClearSize = realClearSize > clearSizePerCore ? clearSizePerCore : realClearSize;
GlobalTensor<yType> yGm;
yGm.SetGlobalBuffer((__gm__ yType*)y);
auto yPart = yGm[offset];
Fill<yType>(yPart, realClearSize, static_cast<yType>(0));
SyncAllCores();
```

本片段的 Ceil/AlignUp 和 SyncAllCores 在工程内定义，分别表示向上取整除法、对齐和第 3.2 节的通知；`offset、realClearSize、clearSizePerCore` 均为元素数。该处 `ONE_BLK_SIZE` 数值为 32，因此 FP16/BF16 的分片粒度为 32 元素，不能直接表述为 128 字节。没有数据可清的参与单元也到达同步点。

### 3.2 两个编译分支

`VEC_FALG_ID=5` 沿用实现中的拼写，`SYNC_AIV_FLAG` 使用公共同步常量。`SyncAllCores` 保留以下协议：

```cpp
#if __CUBE_VECTOR_FUSION_ONLY__
TQueSync<PIPE_MTE1, PIPE_MTE3> sync;
sync.SetFlag(VEC_FALG_ID);
#else
CrossCoreSetFlag<0, PIPE_MTE3>(SYNC_AIV_FLAG);
CrossCoreWaitFlag(SYNC_AIV_FLAG);
CrossCoreSetFlag<2, PIPE_MTE3>(VEC_FALG_ID);
#endif
```

AIC 侧使用同一编译分支等待：

```cpp
#if __CUBE_VECTOR_FUSION_ONLY__
TQueSync<PIPE_MTE1, PIPE_MTE3> sync;
sync.WaitFlag(VEC_FALG_ID);
#else
CrossCoreWaitFlag(VEC_FALG_ID);
#endif
```

一般分离执行单元路径先汇合 AIV，再通知 AIC；耦合配置使用配套的 TQueSync。启动时按目标架构保留任务类型和 Host `SetScheduleMode(1)`，具体查询与核数检查见 [Buffer 与平台](../buffer-and-types.md)。

### 3.3 入口顺序

```text
+-----------------------------------------------------+
| GET_TILING_DATA_WITH_STRUCT                         |
| GetUserWorkspace                                    |
+--------------------------+--------------------------+
                           |
                           |
                           |
                           v
          +----------------+----------------+   no
          | initOutputFlag == L0_INIT?      +----------------------------------------+
          +----------------+----------------+                                        |
                           |                                                         |
                           |                                                         |
                           | yes                                                     |
                           |                                                         |
                           v                                                         |
        +------------------+------------------+                                      |
        | InitOutput                          |                                      |
        | clear slices -> notify / wait       |                                      |
        | release initialization resources    |                                      |
        +------------------+------------------+                                      |
                           |                                                         |
                           |                                                         |
                           |                                                         |
                           +---------------------------------------------------------+
                           |
                           |
                           v
      +--------------------+--------------------+
      | Conv3DDxKernel::Init                    |
      |   -> Conv3DDxKernel::Process            |
      |      (FreeB1Tensor at end)              |
      |   -> End -> TPipe destructor            |
      +-----------------------------------------+
```

未参与主计算的核也先履行初始化协议。普通注册调用由此入口负责预清零，调用方无需预先把 dedx 置零；重复执行每次遵守相同顺序。

## 4 直调和注册接入

注册默认使用上述入口协议，按最终 initOutputFlag 执行。直调可使用同一入口，或把独立 `aclrtMemsetAsync` 与计算放在同一 stream；采用独立步骤时，同时移除计算入口的融合清零及等待。跨 stream 显式连接完成事件。

Host 只产生初始化计划，Tiling 回调不提交设备清零。未分到主计算任务的参与核先履行初始化协议；所有输入输出、Device Tiling、workspace 保持有效到 stream 完成。

## 5 片上补零与检查

L1A 使用 Fill 后的 `PIPE_MTE2` barrier 保证零填充先于真实 dedy 搬入；目的步幅留下 stride 间隔。负反向边界先裁剪，再由 `CalcSetFmatrixParams` 表达剩余补边。详细地址和字段见 [布局与指令](instruction-mapping.md) 与能力映射。

L1B 的转置加逆序路径初始化未搬入的 co 尾部，逆序路径初始化未搬入的 ci 尾部；另一条分形轴由转换补齐。复用命中时保留原 L1B 内容，仅在重新装载时执行相应补零。

用非零预填 dedx 检查全部/部分无贡献和重复执行；覆盖尾清零片段、没有清零数据的参与核、首次事件、槽回绕和最终释放。队列事件与手动事件的所有者唯一，不用 F 同流水 barrier 代替 M→F 依赖。

可选 bias 追加 MTE2→MTE1→M 的生产/覆盖检查，覆盖全量身份变化、分块归还、纯 bias MMAD 和最终 End。无贡献位置的预期值由算子语义决定：DX 为零，Transpose 为可选 bias 经输出转换后的值。

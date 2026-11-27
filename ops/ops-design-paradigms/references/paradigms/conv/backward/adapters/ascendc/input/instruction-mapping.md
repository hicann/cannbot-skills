# 卷积反向算子指令 API 落地要点

> **层级**：AscendC 语言适配层（adapters/ascendc/input/）。**指令的语义、单位与地址公式属语言中立知识**（搬运矩阵轴义、ND/DN/NZ 寻址公式、窗口网格、负边界裁剪、NZ2DN 还原寻址），见 [compute-flow 第 5 节](../../../generic/input/compute-flow.md#5-数据搬运与写回)；本页只承载 **AscendC 的 API 实体、参数结构体、位域填充与 SDK 适配检查**，按 LoadToA1、LoadToB1、LoadToA2、LoadToB2、MmadLocal、LoadL0c2Gm 展开；三种 weight 数据准备分支（VecTranspose / C04 / group enlarge）的准入、布局与搬运见 §3.4–§3.8。变量单位沿用 [language](../../../generic/language.md)，数值类型和资源对象见 [buffer-and-types](../buffer-and-types.md)。

## 1 API 总表与参数结构

| # | 阶段（通路） | Ascend C 调用形式 | 参数结构体 |
|---|---|---|---|
| 1 | LoadToA1 / LoadToB1（GM→L1 父布局） | `DataCopy(a1, dyGm[off], p)` | `Dn2NzParams / Nd2NzParams`（字段表见下） |
| 2 | LoadToA2（L1A→L0A，fmap 窗口 im2col） | `SetLoadDataRepeatWithStride(rp); LoadDataWithStride(l0a, l1A1Matrix, load3d)`；旧 SDK 分支用 `LoadData(l0a, l1A1Matrix, load3d)`（版本判定见 §4.2） | `LoadData3DParamsV2` + `LoadDataRepeatParam(WithStride)` |
| 3 | LoadToB2（L1B→L0B，weight 变换） | `LoadData(b2[dstOffset], b1, p)` | `LoadData2DParamsV2` |
| 4 | MmadLocal（L0A×L0B→L0C） | `Mmad(useC1, a2, b2, mp)`；带 bias 首步 `Mmad(useC1, a2, b2, biasBT, mp)` | `MmadParams` |
| 5 | LoadL0c2Gm（L0C→dedx GM） | `Fixpipe<OutT, float, CFG_ROW_MAJOR>(output[dst], useC1Buf, fp)` | `FixpipeParamsArch3510<ROW_MAJOR>` |
| 6 | 插零预清零（LoadToA1 前置） | `Fill(a1, zero)` + `PipeBarrier<PIPE_MTE2>()` | `InitConstValueParams<T>` |

`DataCopy(dstLocal,srcGlobal,params)` 由 `Nd2NzParams` 或 `Dn2NzParams` 选择转换。布局与寻址语义（copyN/copyD/copyIdx 轴义、ND/DN 源与 NZ 目的地址公式、补齐职责分工）见 [compute-flow 第 5.1 节](../../../generic/input/compute-flow.md#51-指令与单位总表)；本页只列参数结构体的字段与位宽：

| 参数 | 含义与单位 |
|---|---|
| `ndNum/dnNum` | 矩阵个数 |
| `nValue/dValue` | 每个矩阵的 N/D 长度，元素数 |
| `srcNdMatrixStride/srcDnMatrixStride` | 相邻源矩阵起点间隔，源元素数 |
| `srcDValue` | ND 的相邻 N 起点间隔；DN 的相邻 D 起点间隔，源元素数 |
| `dstNzC0Stride` | 目的相邻 D 通道分形起点间隔，32 字节块数 |
| `dstNzNStride` | 目的相邻 N 位置间隔，32 字节块数 |
| `dstNzMatrixStride` | 目的相邻矩阵起点间隔，目的元素数 |

参数检查先用宽整数求值，再检查结构字段和底层指令：矩阵个数、`nValue`、`dstNzC0Stride/dstNzNStride` 为 16 位；`dValue/dstNzMatrixStride` 为 32 位；两个源步幅为 64 位。多矩阵目的步幅还要检查 `dstNzMatrixStride*sizeof(T)` 为 32 的倍数，除以 32 后可由底层 16 位步幅表示。实际地址范围按 compute-flow 的地址公式核算。

## 2 dedy 到 L1A

### 2.1 LoadToA1 实现链与局部量

实现入口为 `LoadToA1`。`CalcSetFmatrixParams` 确定当前窗口与边界，`CalcCurCoutSizeA1` 确定通道范围，`CalcOutToA1DstAddr` 计算插零后的目的起点；NCDHW/NDHWC 分别由 `LoadToA1ForDn2Nz/LoadToA1ForNd2Nz` 搬入。

行带交集的几何推导（hWindowStart/hResidentStart/hSkip/copyRows、空交集跳过）见 [compute-flow 第 5.2.1 节](../../../generic/input/compute-flow.md#521-从-dedx-块反推-dedy-行带)；负反向边界裁剪的数学见同节。实现侧的通道与宽度局部量：

| 局部量 | 含义 |
|---|---|
| `curCoutSize` | 本次实际搬入的有效 dedy 通道数，尾段按 `coutG-aCoutStart` 截断 |
| `woStart/realWo` | 裁剪后的原 dedy 宽度起点、有效宽度（compute-flow 裁剪公式的结果） |
| `a1CopyOffset` | 插零后目的首元素偏移，`hSkip*woExpand*C0` |

组与深度已经定位到 GM 起点时，局部寻址只加剩余坐标。交集非空但 `copyRows=0` 时，只初始化插零区域，不提交零矩阵数的 DataCopy，后续 Load3D 仍可读取零窗口。

### 2.2 按行搬入的参数表

有 H/W stride 插零或宽度裁剪时，每条原 dedy 行作为一个搬运矩阵：

| 字段 | NCDHW：`Dn2NzParams` | NDHWC：`Nd2NzParams` |
|---|---|---|
| `dnNum/ndNum` | `copyRows` | `copyRows` |
| `nValue` | `realWo` | `realWo` |
| `dValue` | `curCoutSize` | `curCoutSize` |
| `srcDnMatrixStride/srcNdMatrixStride` | `wo` | `wo*cout` |
| `srcDValue` | `dout*ho*wo` | `cout` |
| `dstNzC0Stride` | `curHoSize*woExpand` | 同左 |
| `dstNzNStride` | `strideW` | 同左 |
| `dstNzMatrixStride` | `strideH*woExpand*C0` | 同左 |

目的 N 步幅实现宽度前放大，目的矩阵步幅实现高度前放大；`hSkip` 保留当前窗口的 stride 相位。NDHWC 的 `srcDValue` 使用**总通道数 cout**，分组仅改变起点和 `dValue`。

当 `strideH=strideW=1` 且宽度完整时，可把连续行合成一个矩阵：矩阵数取 1，`nValue=copyRows*wo`，`dstNzNStride=1`，矩阵间步幅取 0；源通道步幅与目的通道分形步幅仍按父片段计算。

插零区域在搬运前具有确定的零值。取得可写 L1A 后，当前实现按需执行以下初始化，再写入真实 dedy：

```cpp
InitConstValueParams<T> zero;
zero.repeatTimes = 1;
zero.blockNum = aSlotBytes / 32;
zero.dstGap = 0;
zero.initValue = static_cast<T>(0);
Fill(a1, zero);
PipeBarrier<PIPE_MTE2>();
DataCopy(a1[a1CopyOffset], dyGm[dyOffset], copyParams);
```

`aSlotBytes` 为该槽已分配且按 32 字节对齐的长度；`blockNum/dstGap` 单位为 32 字节。随后才 `EnQue`。DataCopy 写入的 D 尾分形与未被搬运覆盖的空间间隔分别检查，后者依靠上述初始化保持零。

## 3 weight 到 L1B

### 3.1 LoadToB1 函数选择与父片段

`LoadToB1` 经 `LoadGmDataToB1` 选择实现；父布局见 [compute-flow 第 5.3 节](../../../generic/input/compute-flow.md#53-weightgml1bl0b)。本节 `hw=hk*wk、kernelVolume=dk*hw`；`curCoutSize/curCinSize` 为实际搬入通道数。转置父布局的 `curCout1Cout0Size=alignUp(curCoutSize,16)`，仅逆序父布局的 `curCin1Cin0Size=alignUp(curCinSize,16)`；C0 按类型取 16 或 8。

DataCopy 未覆盖区域的初始化职责（与转换自动补齐的分工）：

| weight 变换方式 | DataCopy 未覆盖区域的初始化 |
|---|---|
| FP16/BF16 单位核，仅转置 | `curCoutSize<curCout1Cout0Size` 时初始化父块；D 轴 ci 的尾部由转换补齐 |
| FP32 单位核，B2 直接取块 | `curCinSize<curCin1Cin0Size` 时初始化父块；D 轴 co 的尾部由转换补齐 |
| 转置加逆序 | `curCoutSize<curCout1Cout0Size` 时初始化父块；D 轴 ci 的尾部由转换补齐 |
| 逆序 | `curCinSize<curCin1Cin0Size` 时初始化父块；D 轴 co 的尾部由转换补齐 |

### 3.2 DataCopy 参数表

| weight 格式 / weight 变换方式 | 实现函数 |
|---|---|
| NCDHW 或 NDHWC / 仅转置 | `LoadToB1Nd2NzForDkHkWkIsOne` |
| FP32 单位核 / B2 直接取块 | `LoadToB1TransposeForDkHkWkIsOne` |
| NCDHW / 转置加逆序 | `LoadToB1Dn2Nz` |
| NDHWC / 转置加逆序 | `LoadToB1Nd2Nz` |
| NCDHW / 逆序 | `LoadToB1Dn2NzTranspose` |
| NDHWC / 逆序 | `LoadToB1Nd2NzTranspose`，内部使用 `Dn2NzParams` |

| weight 格式 / weight 变换方式 | 参数结构 | `(copyIdx,copyN,copyD)` | 矩阵数 | `nValue` | `dValue` |
|---|---|---|---|---|---|
| NCDHW 或 NDHWC / 仅转置 | `Nd2NzParams` | `(0,co,ci)` | 1 | `curCoutSize` | `curCinSize` |
| FP32 单位核 / B2 直接取块 | `Dn2NzParams` | `(0,ci,co)` | 1 | `curCinSize` | `curCoutSize` |
| NCDHW / 转置加逆序 | `Dn2NzParams` | `(co,核高宽,ci)` | `curCoutSize` | `hw` | `curCinSize` |
| NDHWC / 转置加逆序 | `Nd2NzParams` | 同上 | `curCoutSize` | `hw` | `curCinSize` |
| NCDHW / 逆序 | `Dn2NzParams` | `(ci,核高宽,co)` | `curCinSize` | `hw` | `curCoutSize` |
| NDHWC / 逆序 | `Dn2NzParams` | `(核高宽,ci,co)` | `hw` | `curCinSize` | `curCoutSize` |

| weight 格式 / weight 变换方式 | 源矩阵步幅，元素 | `srcDValue`，元素 | `dstNzC0Stride`，32B | `dstNzNStride`，32B | `dstNzMatrixStride`，元素 |
|---|---|---|---|---|---|
| NCDHW 或 NDHWC / 仅转置 | 0（单矩阵） | `cinG` | `curCout1Cout0Size` | 1 | 0 |
| FP32 单位核 / B2 直接取块 | 1（单矩阵，步幅不参与寻址） | `cinG` | `curCin1Cin0Size` | 1 | 1 |
| NCDHW / 转置加逆序 | `cinG*kernelVolume` | `kernelVolume` | `hw*curCout1Cout0Size` | `curCout1Cout0Size` | C0 |
| NDHWC / 转置加逆序 | `cinG*kernelVolume` | `cinG` | `hw*curCout1Cout0Size` | `curCout1Cout0Size` | C0 |
| NCDHW / 逆序 | `kernelVolume` | `cinG*kernelVolume` | `hw*curCin1Cin0Size` | `curCin1Cin0Size` | C0 |
| NDHWC / 逆序 | `cinG` | `cinG*kernelVolume` | `hw*curCin1Cin0Size` | 1 | `curCin1Cin0Size*C0` |

**NDHWC 权重的逆序路径也使用 DN2NZ**（两轴绑定见 compute-flow 第 5.3 节的语义陷阱）。所有步幅保留原 GM 的 `cinG`，尾块只缩短有效长度。GM 起点由当前组、`bCoutStart、cinStart、kd` 代入 compute-flow 的 weight 地址式；父片段内按本表寻址，避免重复叠加组偏移。

### 3.3 仅转置：单矩阵 ND2NZ

FP16/BF16 且 `dk=hk=wk=1` 时，两种 weight 格式都连续存为 `(cout,cinG)`。LoadToB1 将当前通道矩阵按 ND2NZ 搬入，再由 B2 转置。下面的 T 为权重元素类型；FP32 单位核使用上表的 DN2NZ 路径，在 B2 直接取块。

```cpp
const uint32_t curCout1Cout0Size = (curCoutSize + 15) / 16 * 16;
if (curCout1Cout0Size != curCoutSize) {
    InitConstValueParams<T> init{};
    init.repeatTimes = 1;
    init.blockNum = useB1Buf.GetSize() * sizeof(T) / 32;
    init.dstGap = 0;
    init.initValue = static_cast<T>(0);
    Fill(useB1Buf, init);
    PipeBarrier<PIPE_MTE2>();
}
Nd2NzParams p{};
p.ndNum = 1;
p.nValue = curCoutSize;
p.dValue = curCinSize;
p.srcNdMatrixStride = 0;
p.srcDValue = td.cinG; // dkHkWk == 1
p.dstNzC0Stride = curCout1Cout0Size;
p.dstNzNStride = 1;
p.dstNzMatrixStride = 0;
DataCopy(useB1Buf, weightGm[out2B1SrcAddrOffset], p);
```

`srcDValue` 保留原组内输入通道步幅；不能换成当前 `curCinSize`。N 轴 co 的补齐区由初始化覆盖，D 轴 ci 的补齐由转换完成。传入 Fill 的 L1B 槽须可写、字节数按 32B 对齐，blockNum 在字段范围内；其容量由 Host 校验。

### 3.4 数据准备三分支与字段

普通 weight 直接搬入（§3.1–§3.3）；VecTranspose、C04、group enlarge 的准备方式见 §3.5–§3.8。类型与分形参数见 [类型表](../buffer-and-types.md#2-类型与资源对象)，编译绑定见 [Tiling 绑定](tilingkey-templates.md)，握手见 [C/V 协同](cv-collaboration.md)。FP32 的 C0=8、weightBytes=4，B2 固定仅逆序；C04 用于 FP16/BF16。

| 分支 | Host 判定 | Kernel 表达 | 参与时机 |
|---|---|---|---|
| 前置权重转置 VecTranspose | CheckVecTransEnable；enableVecTransFlag | TilingData.enableVecTrans | 主计算前准备 workspace 副本 |
| C04 | CheckC04Enable；enableC04Flag | loadB1Condition=TPL_VEC_TO_L1_C04，入口实例化 enableC04Flag=true | 每次 B1 装载请求 |
| group enlarge | CalRealG；enlarge>1 | 运行字段 enlarge/group/oriGroup；Kernel 按 enlarge>1 判定 | 每次 B1 装载请求 |

enableC04Flag 是 Host 状态和 Kernel 配置中的布尔量，原 TilingData 没有同名字段；不能额外读取一个未发布的 enableC04Flag。enableVecTranspose 不是这里的字段名，文中统一使用 VecTranspose 和上表实际标识。

### 3.5 VecTranspose：GM 权重预转置

#### 准入

常规求解器在最终 baseM/baseN/baseK、stepKb 和任务范围确定后执行 CheckVecTransEnable，条件分四组：

| 条件组 | 内容 |
|---|---|
| 前置（全满足） | groups=1，filter=NCDHW，cin≤65535；C04 关闭，tilingHkWkMode=NO_TILING_HWK；核体积大于 1，cin=16 或 cin≥32，且 cin>hk*wk |
| UB 预算 | `16*dk*hk*wk*weightBytes*2<=UBBytes`（vecIn/vecOut 队列各占半 UB；搬入的每通道核空间还按 32B 补齐，生成实现须核对补齐后的访问范围） |
| 收益判定（CheckVecTrans16bitPlus，两种 16bit 类型都执行） | dk>1、di>1；`filterCubeLoadRepeats>=cubeTotalCnt`；再满足 `vecLoopTime<=1`、`ceilDiv(filterCubeLoadRepeats,cubeTotalCnt)>1`、`baseM*5<=baseN` 中至少一项 |
| 16bit 禁用条件 | B1 需要重复装载且 MMAD 可覆盖 MTE2——重复装载条件 `ceilDiv(kValue,stepKb*baseK)>1` 或 `(dk>1且di>1)`；覆盖阈值 baseM≥512，hk≤3 且 wk≤3 时降为 384 |

其中 `filterCubeLoadRepeats=batch*ceilDiv(di,singleCoreDin)*ceilDiv(hi*wi,singleCoreM)`；cubeTotalCnt 是再乘 N 任务数后与可用 AIC 数取小的值。`vecLoopTime=ceilDiv(cout*ceilDiv(cin,16),coreNum*CVRatio)`，CVRatio 由平台核数关系取得。

FP32 在前置条件与 UB 预算通过后使用 CheckVecTransBoundary。令 `spaceTasks=batch*group*ceilDiv(di,singleCoreDin)*ceilDiv(hi*wi,singleCoreM)`、`nTasks=ceilDiv(nValue,singleCoreCin)`、`activeCores=min(spaceTasks*nTasks,coreNum)`，依次判断：

| 条件 | 选择 |
|---|---|
| `spaceTasks>=activeCores` | 启用转置 |
| `activeCores//spaceTasks>=10` | 保留原 weight 来源 |
| `baseK=8 且 baseN<320`，或 `dk=1 且 hk*wk<=4` | 保留原 weight 来源 |
| B1 重复装载且 `baseM>=256`（普通 FP32） | 保留原 weight 来源 |
| 其余 | `coutG*spaceTasks*baseN*activeCores*dk*di*hk*wk//(nTasks*nTasks)>131072` 时启用 |

上述数值为收益选择阈值，UB 容量由平台提供。

kernel-split 使用自身的简化判定：filter=NCDHW、kSCoutFullLoad=0、同一 UB 初筛成立且核体积>1。它不调用常规 CheckVecTrans16bitPlus，也不继承常规路径的 cin 经验阈值。

#### 排布与搬运

```text
+-------------------------------------------------------+
| GM filter（Cout, Cin, DkHkWk）                        |
+---------------------------+---------------------------+
                            |
                            | MTE2 / DataCopyPad：一次一个 Cin0 × DkHkWk 片
                            v
+---------------------------+---------------------------+
| UB vecIn -- V / TransDataTo5HD --> UB vecOut          |
+---------------------------+---------------------------+
                            |
                            | MTE3 / DataCopy
                            v
+---------------------------+---------------------------+
| workspace（Cout, DkHkWk, Cin1, Cin0）                 |
+---------------------------+---------------------------+
                            |
                            | AIC MTE2 / Dn2Nz（源步幅按对齐后的 Cin）
                            v
+---------------------------+---------------------------+
| L1B -- MTE1 / 核位置逆序 --> L0B                      |
+-------------------------------------------------------+
```

每个任务处理一个 `(co,ci1)`，总量为 `cout*ceilDiv(cin,16)`，由全部参与的 AIV 分工。Cin 尾块按 cin0Tail 搬入并补零；没有任务的 AIV 仍参与完成通知。核坐标在此阶段不逆序。

workspace 权重区为 `cout*dk*hk*wk*alignUp(cin,16)*weightBytes` 字节。AIC 把 filterGm_ 切到该副本，并按对齐后的 Cin 计算源步幅；kernel-split 的 kernelSplitStrideB_ 同步改为 `wk*alignUp(cin,16)`。不能只改基址而沿用原步幅。

TransDataTo5HD 的 Cin0 固定为 16。令 `kernelBytes=dk*hk*wk*weightBytes`、`kernelAlign=alignUp(kernelBytes,32)/weightBytes`，地址表与参数为：

| 项 | FP16/BF16 | FP32 |
|---|---|---|
| 32B 数据块的元素数 | 16 | 8 |
| 第 i 个源地址的元素偏移，`0<=i<16` | `i*kernelAlign` | 同左 |
| 第 i 个目的地址的元素偏移 | `i*16` | `(i>>1)*16+(i&1)*8` |
| repeatTimes | `ceilDiv(kernelBytes,32)` | 同左 |
| srcRepStride / dstRepStride，32B | repeat=1 时 0/0，其余 1/16 | 同左 |

两路 UB 槽分别覆盖 `16*alignUp(kernelBytes,32)` 字节；有效核位置写入 workspace，Cin 尾通道补零。

### 3.6 C04：配套的四通道窗口与权重

#### 准入与发布

CheckC04Enable 的准入条件（全部满足才启用）：

| 条件 | 值 |
|---|---|
| 格式 | out_backprop/filter/y 全部 NCDHW |
| dtype | weight 为 **2 字节类型（FP16 或 BF16）** |
| 通道 | 总 cout≤4 |
| 核与输出 | hk*wk>1，ho*wo>1 |
| 几何 | strideH/W、dilationH/W 均为 1；反向左右边界都非负 |
| 模式 | tilingHkWkMode=NO_TILING_HWK，groupConvMode 为原始分组模式 |
| UB 与索引 | 最小 ND 缓冲 `4*16*dk*hk*wk*weightBytes`、最小 NZ 缓冲 `alignUp(hk*wk*4,16)*16*weightBytes` 均不超过所选 C04 单槽预算；槽、索引和掩码的总分配及 gather 索引范围分别检查 |

Host 发布：`kValue=lenHkWkC0=alignUp(hk*wk*4,16)`、singleCoreCout=4、`(loadB2Condition,loadB1Condition)=(0,1)`。Kernel 的 channelSize_=4，Load3D 按四通道窗口形成 L0A；weight 整理形成配套 L0B 的 K 排列。

单槽预算由目标实现绑定：常规分支取 `(UBBytes-vRegBytes-maskBytes-blockBytes)//2`，其中 vRegBytes 来自目标 VReg 配置、maskBytes=vRegBytes/8、blockBytes 为搬运块字节数；带 gather 索引限制的分支使用对应 `BIT16_GATHER_MAX_BUF_SIZE` 约定。它是指令可寻址范围，不能当作物理 UB 容量。Host 的 CheckC04Enable、Kernel 的 InitC04Params/InitUbByteSize 必须选择同一分支，并检查总分配末地址。

当前 FP16/BF16 的 IndexT 为 uint16_t，Gather 索引以源元素计，换算字节范围时乘 weightBytes。生成索引先用宽整数计算并检查 `index<=max(IndexT)`，再窄化；同时检查基址加索引后的实际访问范围、尾掩码与对齐。vecBlockN_ 按同一单槽预算和最大 ND/NZ 片段向 16 对齐，不能因 UB 较大就扩大 gather 索引。

#### 两路配对

```text
A 路（dedy）                                                  B 路（filter）
+---------------------------+                                 +---------------------------+
| dedy / GM                 |                                 | filter / GM               |
+-------------+-------------+                                 +-------------+-------------+
              | MTE2 布局转换                                               | MTE2 / DataCopyPad
              v                                                             v
+-------------+-------------+                                 +-------------+-------------+
| L1A（四通道窗口行带）     |                                 | UB ND（拷贝 + 清零）      |
+-------------+-------------+                                 +-------------+-------------+
              |                                                             | V / gather + H/W 逆序
              |                                                             v
              |                                               +-------------+-------------+
              |                                               | UB NZ                     |
              |                                               +-------------+-------------+
              |                                                             | MTE3 / CopyUb2L14C04
              |                                                             v
              |                                               +-------------+-------------+
              |                                               | L1B（已打包、已逆序）     |
              |                                               +-------------+-------------+
              | Load3D（channelSize=4）                                     | MTE1（无额外变换）
              v                                                             v
+-------------+-------------+                                 +-------------+-------------+
| L0A                       |                                 | L0B                       |
+-------------+-------------+                                 +-------------+-------------+
              |                                                             |
              +------------------------------+------------------------------+
                                             |
                                             v
                                         +---+---+
                                         | MMAD  |
                                         +-------+
```

LoadUb4C04 用 DataCopyPad 搬入当前 Cin 块的完整核体积，cout 不足 4 的通道先清零。Dn2Nz4C04 的 gather 索引把 4 个核位置、各 4 个 cout 位置打成一个 K 分形，并完成 H/W 逆序；hk*wk 不能被 4 整除时，尾掩码使无效核位置贡献为零。CopyUb2L14C04 按 Cin 块写入 L1B。

C04 单子块配置直接执行数据准备；双子块配置按 `c04LoadToB1IterIdx_&1` 轮换，两者均履行就绪/复用握手。只有双子块配置使用第二组通知。完整参与条件见 [C/V 协同](cv-collaboration.md)，跨任务保留 B1 的普通条件排除 C04。

### 3.7 group enlarge：对角权重直写 L1B

CalRealG 先按 [scene 的候选倍数](../../../generic/input/scene.md#31-group-enlarge多个原始组合成一个计算组) 计算 enlarge。令 `windowBytes=(cout1G*C0)*hk*wk*(cin1G*16)*weightBytes`，其中 `cout1G=ceilDiv(coutG,C0)`、`cin1G=ceilDiv(cinG,16)`。Host 初筛为 `2*windowBytes+indexReserveBytes<=availableUbBytes`，索引预留与 UB 可用量由所选实现提供。切分 Hk/Wk 时使用原始组。

Kernel 的 ND/NZ 和索引区按 InitUbByteSize 的实际分支检查：Gather 路径保证每个相对索引在 IndexT 范围内；16bit 的 VSELR/scatter 分支还使用独立索引区和可选 NZ 行间 padding。后者按带 padding 窗口、紧凑窗口、运行时回退依次选择，实际槽大小由 groupNzUbBytes_ 等状态记录。CopyUb2L14Group 使用匹配的源步幅剔除内部 padding，产出同一个 L1B 布局。两份等分初筛不能替代这些实际分配和访问检查。

FP32 的 IndexT 为 uint32_t，使用 Gather；FP16/BF16 为 uint16_t，按目标实现选择 Gather 或 VSELR/scatter。索引以源元素计，地址范围乘 weightBytes 换算，索引区按 `索引个数*sizeof(IndexT)` 计入 UB。

```text
GM 原始组权重（W0、W1…）
    |
    | MTE2 / NDDMA：先整体清零，再放到对应对角块
    v
UB ND: [ W0   0 ]
       [  0  W1 ]
    |
    | V / Dn2Nz4Group（Gather 或 VSELR/scatter）
    v
UB NZ
    |
    | MTE3 / CopyUb2L14Group
    v
L1B -- LoadToB2（逆序） --> L0B
                             |
dedy --> L1A --> L0A --------+
                             |
                             v
                            MMAD --> dedx
```

GroupTransdataWeight 由 AIV 子块 0 执行。每次准备先等待 AIC 归还 B1，再清零 ND UB，用 NDDMA 把各原始组权重放到对应对角块，最后 gather 成目标布局并写 L1B。NCDHW、NDHWC、DHWCN 分别使用各自的源步幅，不能用合并后的 cinG 解释原 GM 的组内 Cin 轴。

计算组 g 中的第 t 个原始组仍是 `g*enlarge+t`。其原始 weight 坐标为 `[(g*enlarge+t)*CoutG+co,ci,kd,kh,kw]`，放入块对角矩阵的行 `t*CoutG+co`、列 `t*CinG+ci`；非对角位置保持零。当前 N/K 块从该计算组布局截取，原 weight 的 Cin 轴长度仍为 CinG。

发布字段为 `group=real_g、oriGroup=G、cinG=enlarge*CinG、coutG=enlarge*CoutG`。Kernel 用 `cin/oriGroup、cout/oriGroup` 恢复原始组内通道；curEnlarge 为当前计算组的有效原始组数，`curEnlargeCin1_=ceilDiv(curEnlarge*(cin/oriGroup),16)`。尾计算组按实际组数和通道尾长搬运、计算与写回。跨组零块只用于内部计算，不扩大外部张量。

### 3.8 数据准备分支的输出检查

- C04 的两路 K 排列、窗口尾长与 MMAD 对齐范围分别检查；单/双子块协议配对，gather 索引先检查再窄化，rot180 恰好实现一次。
- group enlarge 的非对角块为零，原始组隔离；enlarge=1、尾组合并都保持原输出布局。
- VecTranspose 的 workspace 内容、基址和对齐步幅一致，全部 AIV 完成通知后 AIC 才能读取。
- 上述路径分别核对真实 UB/L1 末地址、MTE1 最后一次读取和 MTE3 下一次写入的依赖；具体 flagId 只在 [C/V 协同](cv-collaboration.md) 维护。

## 4 L1A 到 L0A

### 4.1 LoadData3DParamsV2 字段表

`LoadData3DParamsV2<T>` 在一个 L1A 特征面上完成 H/W 窗口展开。三维卷积的深度由外层循环处理（配对条件见 [compute-flow 第 5.5 节](../../../generic/input/compute-flow.md#55-原张量地址与三维步幅)）。L1A 已按正向 stride 插零，因此 Load3D 使用单位 H/W 步长；窗口网格与 K 长度的几何见 [compute-flow 第 5.2.3 节](../../../generic/input/compute-flow.md#523-窗口网格与-k-长度)。字段设置：

| 字段 | 本路径的设置 |
|---|---|
| `l1H/l1W` | 当前父片段的 `curHoSize/woExpand` |
| `channelSize` | L1A 描述的通道范围；按驻留 K 长度推导，并截断到组内通道的对齐长度 |
| `strideH/strideW` | 1 / 1 |
| `filterH/filterW` | 原 `hk/wk` 的低 8 位；高位按 `filterSizeH/W` 的实际编码设置 |
| `dilationFilterH/W` | 原 `dilationH/dilationW` |
| `enTranspose/fMatrixCtrl/padValue` | `false / 0 / 0` |
| `mStartPt` | 当前 dedx 块的行内起点，`mStart%wi` |
| `kStartPt` | 相对**当前 L1A 驻留段**的 K 起点，`kStart-aSegmentStart` |
| `mExtension` | `baseUseM_`，即本次有效 `baseUseM_` |
| `kExtension` | 当前内部 K 块长 `baseUseK_` |

`channelSize` 是通道元素数，`mExtension` 是输出窗口行数，`kExtension` 是归约元素数；三者都不是分形个数。例如 `channelSize=32、mExtension=27、kExtension=64` 分别描述 32 个通道、27 个窗口和 64 个 K 项。分形数量另用向上取整除以 16 计算。

`aSegmentStart` 是当前 L1A 驻留段在完整 K 域中的起点。比如 `baseK=64、stepKa=2`，完整 K 起点 128 对应第二段的局部起点 0；不能把 128 再传给这段 L1A 的 `kStartPt`。

当前父片段的 `channelSize` 取 `min(curStepKa*baseK/(hk*wk),alignUp(singleCoreCout,C0))`，除法按整数处理。它描述父片段，`DataCopy.dValue` 描述实际搬入的有效通道；尾段的两者可以不同。Load3D 读取的对齐通道必须由转换补齐或显式初始化覆盖。核大小字段使用采样点数，膨胀后的有效核尺寸用于窗口范围计算。

### 4.2 FMatrix、补边与目的分形步幅

`padList` 顺序为**左、右、上、下**。左右值按 compute-flow 第 5.2.1 节的裁剪公式处理，顶部为 `max(-hWindowStart,0)`。当前路径将底部字段设为 255，用于父片段下边界外的零填充；实际提取范围由 M 起点、扩展长度和有效 dedx 区域共同限定。这个 255 是搬运配置，数学上的反向下边界仍保留在前置处理记录中。

CalcSetFmatrixParams 填写 load3d 的高宽和 padList。默认 LoadData 配置在发射窗口指令前设置 FMatrix 与 padding；只有采用关闭自动设置的重载时，才由工程显式调用 `SetFmatrix(l1H,l1W,padList,FmatrixMode::FMATRIX_LEFT)` 并设置 padding。不要把参数计算函数等同于一次硬件寄存器写入。

FMatrix 的字段对应如下，位号用于解释编码；实现使用公共 API：

| 位范围 | 内容 |
|---|---|
| 15:0 / 31:16 | l1W=woExpand / l1H=curHoSize |
| 39:32 / 47:40 | 左 / 右 pad |
| 55:48 / 63:56 | 上 / 下 pad |

每次设置时检查编码后的输出行宽与 `wi`、当前 `mStartPt` 的关系，保证展开一行后回到正确的下一行。不能仅替换某个 pad 字段而保留旧的行宽与起点。

`UpdateLoadToA2ParamsM` 设置重复参数，`LoadToA2` 使用配套重载。两个位置按同一版本宏选择：

```cpp
#if defined(ASC_DEVKIT_VERSION_NUM) && ASC_DEVKIT_VERSION_NUM >= 90000000
LoadDataRepeatParamWithStride repeatParam = {
    0, 1, 0, static_cast<uint16_t>(ceilDiv(baseUseM_, 16))};
SetLoadDataRepeatWithStride(repeatParam);
// LoadToA2
LoadDataWithStride(l0a, l1A1Matrix, load3d);
#else
LoadDataRepeatParam repeatParam = {
    0, 1, 0, static_cast<uint16_t>(ceilDiv(baseUseM_, 16))};
SetLoadDataRepeat(repeatParam);
// LoadToA2
LoadData(l0a, l1A1Matrix, load3d);
#endif
```

这是两个函数中对应片段的合并展示。两条版本分支都传入四个初始化值：重复步幅 0、次数 1、模式 0、目的分形步幅 `ceilDiv(baseUseM_,16)`。一个输入分形为 `16*C0` 个元素、512 字节；目的步幅跨过当前 L0A 的完整 M 方向。`stepKa` 控制软件 K 循环。

| Repeat 字段 | 本通路取值与含义 |
|---|---|
| repeatStride | 0，本次不配置额外重复位移 |
| repeatTime | 1，一次窗口指令 |
| repeatMode | 0，使用当前通路的模式 |
| dstStride | `ceilDiv(baseUseM_,16)`，相邻 K 分形跨过的 M 分形数 |

WithStride 重载消费显式 dstStride；支持自动目的步幅的 LoadData 重载按窗口参数补入步幅。两类调用按同一 SDK 分支成对使用，不把 repeatStride 当作 repeatMode，也不把该 dstStride 当作字节数。

实际 SDK 的类型声明和重载必须与所选版本分支一致。

FMatrix、padding 和重复参数是流水寄存器状态。切换父片段或 M 尾块时更新对应状态；预取下一段 L1A 后，发射当前段 Load3D 前要恢复当前段的配置。默认自动设置 FMatrix/padding；采用关闭自动设置的重载时，由调用方先设置相同内容。

### 4.3 地址、对齐与 SDK 适配检查

L1 源地址按 32 字节对齐，L0A 目的地址按 512 字节对齐。`padList` 单项为 8 位无符号数；M/K 起点及扩展长度为 16 位无符号数。先检查裁剪后的范围、filter 高位编码与 dilation 范围，再转换字段类型。

以 `mStart%wi` 和有效 `baseUseM_` 设置窗口，M 的物理跨度再向 16 对齐。例如 M 尾块为 72 时，`mExtension=72`、物理跨度为 80。移植时保留这两个含义，不能为了对齐而移动窗口起点或扩大有效输出。

目标 SDK 的参数检查、编译与设备执行分别核实；若检查器对 M 起点或扩展长度施加额外对齐要求，记录实际版本和触发参数，再验证对应接口。布局推导不能代替该接口的执行证据。

例如 `wi=16、mStart=16、baseUseM_=16、hk=wk=3、dilation=1、反向四侧补边=1`：`mStartPt=0`，当前三行 L1A 提供一个完整的 16 点输出行，左右 pad 均为 1，`mExtension=16`。若 `baseUseK_=16`，L0A 为一个 16×16 分形，`dstStride=1`。这个例子可以作为窗口通路的最小编译与运行检查。

## 5 L1B 到 L0B

### 5.1 LoadData2DParamsV2 字段单位

各路径最终形成 ZN(K,N) 片段 `(K1,Cin1,Cin0,K0)`。令 `blockBaseN_=ceilDiv(baseUseN_,16)`、`fractalElements=16*C0`；`curLoadKbl1_` 是当前 L1B 父段的实际 K 跨度，由本轮 step 和尾长确定。本节沿用第 3 节的 hw 与父布局跨度。

| `LoadData2DParamsV2` 字段 | 单位与用途 |
|---|---|
| `mStartPosition/mStep` | M 方向每单位 16 个输入元素 |
| `kStartPosition/kStep` | K 方向每单位 32 字节，即 C0 个输入元素 |
| `srcStride/dstStride` | 512 字节分形，即 `fractalElements` 个输入元素 |
| `ifTranspose` | 是否交换源分形内的两条通道轴 |

非空 `mStep/kStep` 取 1～255。L1 源地址按 32 字节、L0B 目的地址按 512 字节对齐。`srcStride` 为有符号数，检查底层窄化范围，并同时检查反向访问的最低与最高地址。LocalTensor 下标继续使用元素。

### 5.2 转置加逆序

本路径用于 FP16/BF16，L1B 轴序为 `(Cin1,Hk,Wk,Cout1,Cout0,Cin0)`。令 `bKStart=kStart-bSegmentStart`；第 j 个 K 分形的父段编号为 `r=bKStart/C0+j`，原核高宽位置为 `hw-1-r%hw`。

每次读取一个 K 分形、全部 `blockBaseN_` 个 dedx 通道分形：

| 参数或偏移 | 值 |
|---|---|
| `mStartPosition/kStartPosition` | 0 / 0 |
| `mStep/kStep` | 1 / `blockBaseN_` |
| `srcStride/dstStride` | `curLoadKbl1_/C0` / 1 |
| `ifTranspose` | `true` |
| L1B 源元素偏移 | `(r//hw)*fractalElements + (hw-1-r%hw)*curCout1Cout0Size*C0` |
| L0B 目的元素偏移 | `j*blockBaseN_*fractalElements` |

源偏移完成 H/W 逆序，`srcStride` 跨越 L1B 的 Cin1 层；块内 transpose 将 `(Cout0,Cin0)` 调整成 `(Cin0,Cout0)`。三者共同形成反向权重。

### 5.3 逆序：按通道窗口分段

L1B 轴序为 `(Cout1,Hk,Wk,Cin1,Cin0,Cout0)`，Cin0=16，Cout0=C0。先令 `begin=bKStart/C0、end=begin+ceilDiv(baseUseK_,C0)`，按每个 Cout1 的 hw 个位置分段：

```cpp
uint32_t dstOffset = 0;
for (uint32_t co1 = begin / hw; co1 < ceilDiv(end, hw); ++co1) {
    uint32_t windowBegin = co1 * hw;
    uint32_t windowEnd = windowBegin + hw;
    uint32_t first = begin > windowBegin ? begin : windowBegin;
    uint32_t last = end < windowEnd ? end : windowEnd;
    LoadData2DParamsV2 p{};
    p.kStartPosition = windowBegin;
    p.mStartPosition = (windowEnd - 1 - first) * blockBaseN_;
    p.mStep = blockBaseN_;
    p.kStep = last - first;
    p.srcStride = -static_cast<int32_t>(blockBaseN_);
    p.dstStride = blockBaseN_;
    p.ifTranspose = false;
    p.sid = 0;
    LoadData(b2[dstOffset], b1, p);
    dstOffset += (last - first) * blockBaseN_ * fractalElements;
}
```

这里的首个源分形位置为 `kStartPosition*abs(srcStride)+mStartPosition`；之后沿 K 按负步幅读取。Cout1 边界处重新设置起点，才能保持 dedy 通道顺序、仅逆序核高宽。H/W 线性逆序等价于两个轴各自逆序。

### 5.4 单位核：按 dtype 选择搬运

FP16/BF16 取 loadB2Condition=1。核体积为 1 时，`hw=1、curLoadKbl1_=curCout1Cout0Size`，L1B 轴序为 `(Cin1,Cout1,Cout0,Cin0)`。第 j 个 K 分形的父段编号为 `r=bKStart/C0+j`，一次读取对应的全部 Cin1：

| 参数或偏移 | 值 |
|---|---|
| `mStartPosition/kStartPosition` | 0 / 0 |
| `mStep/kStep` | 1 / `blockBaseN_` |
| `srcStride/dstStride` | `curCout1Cout0Size/C0` / 1 |
| `ifTranspose` | `true` |
| L1B 源元素偏移 | `r*fractalElements` |
| L0B 目的元素偏移 | `j*blockBaseN_*fractalElements` |

`srcStride` 跨越 L1B 的 Cin1，块内转置将 `(Cout0,Cin0)` 变为 `(Cin0,Cout0)`。Cout1 递增，源步幅为正；空间核坐标均为 0。

```text
  +---------------------+
  | weight / ND         |
  | row = co; col = ci  |
  +----------+----------+
             | ND2NZ; row stride = original cinG
             v
+------------+------------+
| L1B                     |
| (Cin1,Cout1,Cout0,Cin0) |
+------------+------------+
             | LoadData2DV2; ifTranspose=true
             v
+------------+------------+
| L0B                     |
| (Cout1,Cin1,Cin0,Cout0) |
| ZN(K,N)                 |
+-------------------------+
```

父段切换后重新以 `bSegmentStart` 求局部 K，输出通道尾部仍使用有效长度。

FP32 取 loadB2Condition=2。GM→L1B 已形成 `(Cout1,Cin1,16,8)`，LoadToB2NoTransposeNoReverse 连续抽取当前 K 子块：

| 字段 | FP32 单位核取值 |
|---|---|
| `mStartPosition/kStartPosition` | 0 / 0 |
| `mStep/kStep` | `blockBaseN_*ceilDiv(baseUseK_,C0)` / 1 |
| `ifTranspose` | false |
| L1B 源元素偏移 | `(bKStart/C0)*nCompute*C0` |
| L0B 排布 | `(K1,N1,16,8)`，每个分形 128 个 float |

该路径的核坐标恒为零，通道变换已在 GM→L1B 完成。

### 5.5 参数更新与分形单位

| 源函数 | 更新时机 | 关键结果 |
|---|---|---|
| InitLoadToB2Params | 初始化 weight 变换分支 | 起点为 0，设置 ifTranspose、初始 mStep/dstStride |
| UpdateLoadToB2ParamsN | 当前 dedx 通道块变化 | blockBaseN_=ceilDiv(baseUseN_,16)；转置路径按 Cin1 扫描，仅逆序路径使用负 srcStride |
| UpdateLoadToB2ParamsK | 当前 L1B 父段变化 | 转置路径 srcStride=curLoadKbl1_/C0 |
| LoadToB2 | 当前 K 子块 | 按第 5.2/5.3/5.4 节组织起点、分段与目的偏移 |

源码 `blockSize=c0<<4` 即 fractalElements：FP16/BF16 为 256，FP32 为 128，均占 512B。转置路径 dstStride=1，仅逆序路径 dstStride=blockBaseN_；两者单位均为 512B 分形。LocalTensor 下标和 dstB2Stride_ 使用元素。

## 6 Compute

### 6.1 MmadParams 字段表

普通三模板共用 `InitMmadParams → CalcParamsMmad → MmadLocal`。`UpdateL1KParams` 在进入本轮计算前更新 K 尾块。参数与通用术语对应如下：

FP32 的 L0A/L0B/L0C 均使用 float，Init 设置 `SetHF32Mode(false)`；Mmad 的 K 长度按 C0=8 组织，M/N 粒度仍为 16。

| `MmadParams` 字段 | 实现取值 |
|---|---|
| `m` | `baseUseM_`；当前任务末 M 块向 16 对齐 |
| `n` | `baseUseN_`，有效 dedx 通道数 |
| `k` | 普通取 `baseUseK_`；C04 尾轮取 `alignUp(baseUseK_,c0)` |
| `cmatrixInitVal` | `kPos==0 && isFirstDk` |
| `cmatrixSource`、`unitFlag`、`kDirectionAlign` | `0、0、false` |

上表为 `MmadLocal<false>`，消费当前 L0A/L0B 与 L0C。公共计算用于带 bias 的 Transpose 时，首次 MMAD 取 `cmatrixSource=1、cmatrixInitVal=false`，使用 BT 重载；后续从 L0C 累加。BT 搬运的单位、首轮条件及 LoadL0Zero 参数见 [bias 通路](bias-bt-path.md)。

`cL0` 为 `LocalTensor<float>`。`mCompute/nCompute` 用于 L0A/L0B/L0C 的物理排布和容量；实际 `Mmad.n` 使用有效 `baseUseN_`。模板中的对齐矩阵形状落到 API 时，应按本表分别设置。

例如任务末块 `baseUseM_=27`：ctx.baseUseM_ 和 Load3D.mExtension 保持 27，InitMmadParams 将本次 MmadParams.m 设为 32；Fixpipe.mSize 仍为 27，源跨度为 32。对齐只扩大内部跨度，不改变有效窗口或输出个数。

普通路径的运行 K 为对齐通道数乘核高宽；C04 的 `tmpSingleCoreK=4*singleShapeHWk_`，Host 的 `kValue` 另向 C0 对齐。两者分别派生尾长：Load3D.kExtension 使用真实 baseUseK_，C04 的 MMAD.k 再对齐，多出的乘积为零。完整定义见 [K 长度](../../../generic/input/language.md#2-mnk-三元组gemm-映射)。

### 6.2 首次有效贡献初始化

无 bias 时，每个输出块首次实际 MMAD 设置 `cmatrixInitVal=true`，后续 K 块和有效核深度保持 `false`。带 bias 时首次从 BT 引入初值，并设置 computeBiasOnce_。`firstContribution` 仅在真实计算后更新；它与核深度是否为 0 分别判断。

例如 `curDinIdx_=1、padFront=2、strideD=2、dilationD=1、dk=4、dout=2`，每个有效深度消费上述两块 K：

| 原 `kd` | `v_d=curDinIdx_+padFront-kd*dilationD` | 行为 |
|---|---:|---|
| 0 | 3 | 不满足 stride 整除，跳过 |
| 1 | 2 | `o_d=1`；K=64 初始化，K=32 累加 |
| 2 | 1 | 跳过，保留已有部分和 |
| 3 | 0 | `o_d=0`；K=64、32 均累加 |

全部深度无有效贡献时，无 bias 路径消费初始化契约并跳过；有 bias 路径执行 ComputeForBias，K 临时取 C0，随后交给 F。整个归约期间保留同一份 L0C；`unitFlag=0` 配合同步实现中的队列交接，在归约结束后交给 F。

## 7 CopyOut

### 7.1 行列方向与步幅单位

本路径用 `Fixpipe` 将 FP32 L0C 直接转换并写回 GM。目标格式决定配置、参数模板实参和步幅：

| dedx 格式 | 配置 / 参数模板实参 | `dstStride`，dedx 元素数 | 块内目标元素偏移 |
|---|---|---|---|
| NCDHW | `CFG_COLUMN_MAJOR` / `CO2Layout::COLUMN_MAJOR` | `di*hi*wi` | `nLocal*dstStride+mLocal` |
| NDHWC | `CFG_ROW_MAJOR` / `CO2Layout::ROW_MAJOR` | 总通道数 `cin` | `mLocal*dstStride+nLocal` |

两路均设置 `mSize=baseUseM_、nSize=baseUseN_、srcStride=mCompute`。Fixpipe 的源通道块为 N0=16 个 float，即 **64 字节**；`srcStride` 及源矩阵步幅使用该单位。COLUMN_MAJOR 的 `srcNzC0Stride=1` 表示相邻 M 位置跨一个源通道块。

因此 `srcStride=80` 表示相邻 N 分形起点相距 `80*64=5120` 字节。源分形跨度保留对齐尺寸，输出尺寸保留有效范围。DataCopy 目的通道步幅的 32B 单位与这里的 64B 单位分别换算。NZ2DN/NZ2ND 的逐层寻址公式见 [compute-flow 第 5.4 节](../../../generic/input/compute-flow.md#54-mmad-与-dedx-写回)。

写回范围的分组与尾块陷阱：`nSize` 若误取对齐值，会覆盖下一组；NDHWC 的 `dstStride` 若误取组内通道数 cinG，会写错下一个空间点。组起点由外层地址包含时，内层写回只加块内偏移。

### 7.2 参数构造与数值转换

LoadL0c2Gm 直接构造 Fixpipe 参数并设置转换模式。以下为 NDHWC 输出片段，OutT 为输出元素类型，useC1Buf 为 FP32 部分和：

```cpp
FixpipeParamsArch3510<CO2Layout::ROW_MAJOR> fixPipeParams{};
fixPipeParams.params.ndNum = 1;
fixPipeParams.mSize = baseUseM_;
fixPipeParams.nSize = baseUseN_;
fixPipeParams.srcStride = (baseUseM_ + 15) / 16 * 16;
fixPipeParams.dstStride = td.cin;
if constexpr (std::is_same_v<OutT, float>) {
    fixPipeParams.quantPre = QuantMode_t::NoQuant;
} else if constexpr (std::is_same_v<OutT, bfloat16_t>) {
    fixPipeParams.quantPre = QuantMode_t::F322BF16;
} else {
    fixPipeParams.quantPre = QuantMode_t::F322F16;
}
fixPipeParams.reluEn = false;
Fixpipe<OutT, float, CFG_ROW_MAJOR>(output[dstOffset], useC1Buf, fixPipeParams);
```

`dstOffset` 由当前通道、深度和空间位置计算，见 [Kernel 地址与输出通路](kernel-template.md)。FP16/BF16 分别使用 F322F16/F322BF16，FP32 使用 NoQuant；`enRelu0=0`，累加类型为 FP32。

NCDHW 分支使用 `CO2Layout::COLUMN_MAJOR`，另设 `dnNum=1`、两个矩阵间步幅为 0、`srcNzC0Stride=1`，`dstStride=di*hi*wi`。函数内参数类型和 `CFG_COLUMN_MAJOR/CFG_ROW_MAJOR` 保持配对。

上述字段构成 Fixpipe 的参数链。生成时核对目标 SDK 的构造默认值、字段范围与模板重载，保持配置与参数类型配对。

舍入中点、溢出、subnormal、±0、Inf/NaN 按目标平台转换语义验证。实际写回范围由 `mSize/nSize` 确定，源分形步幅继续使用对齐后的 M。

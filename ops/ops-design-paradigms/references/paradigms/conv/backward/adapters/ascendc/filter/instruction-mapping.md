# 权重梯度分支指令 API 落地要点

本页落实 API、参数结构、有效长度、步幅单位与数值示例。布局和地址见 [compute-flow](../../../generic/filter/compute-flow.md)，阶段组织见 [kernel-template](kernel-template.md)。A=dedy、B=x，输入为 FP16/BF16，累加与 dedw 为 FP32。队列与事件协议见 [同步实现](sync-implementation.md)。

## 1 API 总表与参数结构

| 通路 | API | 参数结构 |
|---|---|---|
| dedy GM→L1A | DataCopy，NCDHW 用 DN2NZ、NDHWC 用 ND2NZ | Dn2NzParams / Nd2NzParams |
| x GM→L1B | DataCopy，按原格式转换为通道分块父布局 | Dn2NzParams / Nd2NzParams |
| L1A→L0A | LoadData | LoadData2DParamsV2 |
| L1B→L0B | LoadDataWithStride 或配套旧重载 | LoadData3DParamsV2 + Repeat |
| 乘加 | Mmad | MmadParams |
| 写回 | Fixpipe | FixpipeParamsArch3510，ROW_MAJOR/COLUMN_MAJOR |
| 原子范围 | SetAtomicAdd<float> / SetAtomicNone | 围绕本次 Fixpipe 调用 |
| 预清零 | Fill<float>，AIV 执行 | 实际输出区域与通知协议 |

片段使用 `AscendC` 命名空间，T 为 half 或 bfloat16_t；td 为共享 Tiling 数据，a1/b1、a2/b2、l0c 分别为对应 LocalTensor。所有构造前先用宽整数检查，再赋给目标 SDK 的字段类型。

| 单位类别 | 字段或表达式 |
|---|---|
| 元素数 | nValue/dValue、srcDValue、src*MatrixStride、dstNzMatrixStride、LocalTensor 下标 |
| 32B 块 | FP16/BF16 ND2NZ/DN2NZ 的 dstNzC0Stride、dstNzNStride |
| 512B 输入分形 | Load2D 的 srcStride/dstStride；一个 16×16 输入分形 |
| 64B 源 C0 | FP32 Fixpipe 的 srcStride 及 srcNz* 步幅 |
| FP32 元素 | Fixpipe 的目的 stride、GM 下标 |

Nd2NzParams 的 nValue/dValue 是该 API 的源矩阵维度，不能直接解释为 GEMM 的 N/D。以下每条通路均说明其具体轴义。

当前只有 MN_StreamK 求解模板。isSplitKernelHW 选择普通/拆核编译实例，splitWo、iterateOrder、step 和实际尾长选择实例内部的通路：

| 判定 | 搬运与计算 | 写回 |
|---|---|---|
| isSplitKernelHW=false | 本页第 2～5 节的普通窗口 | 第 7 节按输出格式分发 |
| isSplitKernelHW=true | 第 3.3、5.3 节，原核坐标外移到 MTE2 | 同一原始 dedw 布局 |
| splitWo<wo | 第 3.2 节更新宽片、真实行带与父段元数据 | 本段 L0C 保持累加 |
| 有效 M=1 | 第 2.3 节，ComputeInit 临时扩到计算 M=16 | 恢复有效 M，只写一个 co |
| dk*hk*wk=1 | 矩阵窗口退化 | 第 7.2 节简化写回 |
| 其余且 `(baseN/16)%(hk*wk)!=0` | 保留 N 窗口起点与实际 ci 覆盖 | 第 7.3 节非整窗写回 |

上述判定可以组合；MTE2 和 MTE1 必须消费同一份父段宽高、补边与分形跨度。

## 2 dedy 到 L1A

### 2.1 LoadToA1 实现链与局部量

FP16/BF16 在 ComputeInit 后按计算 M 准备操作数，选择链为 `LoadToA1 → LoadToA1ForTransFormat`：

| dedy 格式 | 普通父段 | 切宽父段 |
|---|---|---|
| NCDHW | LoadToA1Fp16Dn2Nz | SplitWLoadToA1Dn2Nz |
| NDHWC | LoadToA1Fp16Nd2Nz / LoadToA1Nd2NzNormal | SplitWLoadToA1Nd2Nz |

kal1_=stepKa*baseK；k 为当前 ComputeLoop 内的全局 K 块号，kaIdx=k%stepKa 为父段内块号，kaStepIdx=k/stepKa 为 A 父段号；stepKaRound 是父段总数；dhwO_=dout*ho*wo。有效 M 与计算 M 的切换见第 2.3 节。

实现链 `ComputeLoadToA1(..., k, ..., kaStepIdx) → LoadToA1(..., k, ..., kaStepIdx)` 传递全局 k。LoadToA1 及其下层函数中同名形参 kaIdx 接收的是这个全局值；它与 ComputeLoop 的父段内 kaIdx 分属不同作用域。本页统一按父段号计算搬运起点，尾长不使用父段内 kaIdx。

### 2.2 按行搬入的参数表

不切宽时，两种 API 都令 nValue 为本父段有效 K、dValue 为本 co 块的有效 M：

| 字段 | NCDHW / Dn2NzParams | NDHWC / Nd2NzParams |
|---|---|---|
| dnNum / ndNum | 1 | 1 |
| nValue | 非末段 kal1_；末段 `singleShapeHo_*wo-kaStepIdx*kal1_` | 同左 |
| dValue | 非末 M 块 baseM；末块 tailM_ | 同左 |
| srcDValue | dhwO_，原通道平面步幅 | 总 cout，原空间点行步幅 |
| dstNzC0Stride | alignUp(nValue,16) | 同左 |
| dstNzNStride | 1 | 1 |
| dstNzMatrixStride | 1，单矩阵不参与跨矩阵推进 | 同左 |

以下为 NCDHW 多通道、非切宽参数片段；kaStepIdx 为本次搬运的父段号，预取下一父段时加 1：

```cpp
Dn2NzParams p{};
p.dnNum = 1;
p.nValue = (kaStepIdx + 1 == stepKaRound)
    ? uint64_t(singleShapeHo_)*td.wo - kaStepIdx*uint64_t(kal1_) : kal1_;
p.dValue = isLastMAL1 ? tailM_ : td.baseM;
p.srcDValue = uint64_t(td.dout)*td.ho*td.wo;
p.dstNzC0Stride = (uint64_t(p.nValue)+15)/16*16;
p.dstNzNStride = 1;
p.dstNzMatrixStride = 1;
DataCopy(a1, dedyGm[out2A1SrcAddrOffset], p);
```

例如当前 K 长度为 70、baseK=16、stepKa=2，末父段号为 2，起点为 64，nValue=6；此时全局 k=4、父段内 kaIdx=0，不能按后者扣除搬运起点。

例如逻辑 dedy=`(1,64,3,8,8)`、G=2，当前组处理 32 个 co、父段覆盖 64 个位置：NCDHW 的 srcDValue=192，NDHWC 的 srcDValue=64；两路均 nValue=64、dValue=32、dstNzC0Stride=64。目的通道块相距 `64*32=2048B`。

切宽时，把实际覆盖行数作为矩阵数：

| 字段 | 取值 |
|---|---|
| dnNum / ndNum | hoCopyLen |
| nValue / dValue | singleShapeWo_ / 当前有效 co |
| srcDnMatrixStride | wo，NCDHW 元素 |
| srcNdMatrixStride | wo*cout，NDHWC 元素 |
| dstNzC0Stride | alignUp(singleShapeWo_*hoCopyLen,16) |
| dstNzMatrixStride | singleShapeWo_*16，元素 |

例如原 wo=130、宽片=64、实际搬入两行，nValue=64、矩阵数=2、dstNzC0Stride=128、dstNzMatrixStride=1024。原行间隔仍为 130 或 130*cout；alignedL1UseKa 保存 128，供后续 Load2D 使用。

### 2.3 单个有效 co：先扩计算 M，再恢复写回 M

当前 MN_StreamK 的 ComputeNormal/ComputeSplitKernelHW 都先调用 ComputeInit。它保存 baseUseMBak；若有效 baseUseM_=1，计算期间改为 16，Compute 结束、CO1 EnQue 前恢复 baseUseMBak。资源仍按 baseM 预留。

| 阶段 | 单个有效 co 的取值 |
|---|---|
| ComputeInit 前 | baseUseM_=tailM_=1；保存 baseUseMBak=1 |
| MTE2 | 计算 M 为 16，但末 M 块的 DataCopy.dValue 仍为 tailM_=1；源步幅保留总 cout 或完整空间平面 |
| MTE1 | 使用第 4 节的转置通路；kStep=1，dstStride=1，ifTranspose=true |
| MMAD | m=16，k 为当前有效 K；有效 co 仍位于第 0 行 |
| CopyOut | 恢复 baseUseM_=1；Fixpipe 只写一个 co，内部对齐行不写回 |

例如某组仅有一个 co、父段有 31 个 K 元素，DataCopy.dValue=1、dstNzC0Stride=32；L1A 按一个 M 分形存放，Load2D.mStep=2、kStep=1。MMAD 使用 m=16、k=31，最终输出只含一个有效 co。数学上的单通道条件不能直接替换为 ifTranspose=false；该值由进入搬运函数时的计算 M 决定。

## 3 x 到 L1B

### 3.1 LoadToB1 函数选择与父片段

普通路径由 CalOut2B1Params、CalB1HiCopyParams 和 LoadToB1Dn2Nz / LoadToB1Nd2Nz 构成。先算正向 x 采样行带，再裁到有效 x 范围，更新本父段 l1H/l1W 与 padList。

bL1cin1CopyLen 是当前**实际通道数**；hiCopyLen 是实际行数；普通路径 curSingleCoreDk_=1。深度无效或整个行带越界时不访问 GM，使用跳过/零窗口通路。

#### 3.1.1 当前 N 子块对应哪些 ci

令 `hw=hk*wk`、`usedN=(curNIdx_%cinHkWkLoop_)*baseN`。curNIdx_ 是任务内 N 子块号，cinHkWkLoop_ 是一个核深度的 N 子块数；当前任务只含一个核深度。

当前 `nIter_=cinHkWkLoop_=ceilDiv(alignUp(singleShapeCin_,16)*hw,baseN)`；`tailN_=alignUp(singleShapeCin_,16)*hw-(nIter_-1)*baseN`。Compute 对每个 N 子块取 baseN，末子块取 tailN_；整除时 tailN_=baseN。singleShapeCin_ 是该任务实际 ci 数，不能用组内对齐通道数替代。

| 通路 | 当前父段的组内任务相对 ci 起点 b1SrCin |
|---|---|
| 整窗 | `usedN/hw` |
| 非整窗 | `(usedN/(hw*16))*16` |

整窗复制通道数取 `min(ceilDiv(baseUseN_,hw),singleShapeCin_-b1SrCin)`。非整窗先令 `bL1N=ceilDiv(baseUseN_,16)`、`copyBlocks=ceilDiv(bL1N,hw)`；singleShapeCin_>16 时，若 `hw>bL1N && hw%bL1N!=0`，或 `hw<=bL1N && (2*bL1N)%hw!=0`，copyBlocks 再加 1。实际通道数为 `min(copyBlocks*16,singleShapeCin_-b1SrCin)`。

这是 CalOut2B1Params/CalOut2B1ParamsBaseNUndivided 的两种覆盖规则。NCDHW 的父段相对 GM 偏移为 `b1SrCin*di*hi*wi`，NDHWC 为 b1SrCin；任务、组、真实深度和 H 段起点已由外层基址承担，不重复相加。实际复制通道数与 Load3D.channelSize 分别记录，后者包含分形补齐。

### 3.2 DataCopy 参数表

普通、不切宽的 FP16/BF16 片段使用布局转换 API：

| 字段 | NCDHW / DN2NZ | NDHWC / ND2NZ |
|---|---|---|
| dnNum / ndNum | curSingleCoreDk_ | curSingleCoreDk_ |
| nValue | hiCopyLen*wi | hiCopyLen*wi |
| dValue | bL1cin1CopyLen | bL1cin1CopyLen |
| srcDValue | di*hi*wi | 总 cin |
| src*MatrixStride | hi*wi | hi*wi*cin |
| dstNzC0Stride | nValue | nValue |
| dstNzNStride | 1 | 1 |
| dstNzMatrixStride | ceilDiv(dValue,16)*nValue*16 | 同左 |

```cpp
Nd2NzParams p{}; // x 为 NDHWC；已裁出真实的有效行带
p.ndNum = curSingleCoreDk_;
p.nValue = uint64_t(hiCopyLen)*td.wi;
p.dValue = bL1cin1CopyLen;
p.srcDValue = td.cin;
p.srcNdMatrixStride = uint64_t(td.hi)*td.wi*td.cin;
p.dstNzC0Stride = p.nValue;
p.dstNzNStride = 1;
p.dstNzMatrixStride = (uint64_t(p.dValue)+15)/16*p.nValue*16;
DataCopy(b1, xGm[out2B1SrcAddrOffset], p);
```

例如 x 的逻辑形状 `(1,64,3,8,8)`，本父段搬 3 行、16 通道：nValue=24、dValue=16。NCDHW 源通道步幅为 192，NDHWC 源行步幅为 64；目的通道块跨度为 `24*16=384` 个元素。

切宽后，矩阵数变为 hiCopyLen，nValue 为 `singleShapeWi-padLeft-padRight`，源矩阵步幅为原 wi 或 wi*cin；目的矩阵步幅为 nValue*16。pad 已从真实复制宽度扣除，不能让搬运越过 x 再期望 Load3D 补救。

#### 3.2.1 每个宽片都重算边界与 K 状态

第 splitWoIdx 个宽片从 `startWo=splitWoIdx*splitWo` 开始，实际宽为 `singleShapeWo_=min(splitWo,wo-startWo)`。令 `KwEff=(wk-1)*dilationW+1`：

| 量 | 切宽时的计算 |
|---|---|
| singleShapeWi | `(singleShapeWo_-1)*strideW+KwEff`，含补边的逻辑跨度 |
| b1SrcWiLeftOffGm | `startWo*strideW-padLeft`，有符号 |
| 当前左 pad | `max(-b1SrcWiLeftOffGm,0)` |
| 当前右 pad | `max(b1SrcWiLeftOffGm+singleShapeWi-wi,0)` |
| load3d.l1W | `max(singleShapeWi-当前左pad-当前右pad,0)` |
| A 宽片起点 | 当前批深度的 A 基址加 startWo；NDHWC 再乘总 cout |
| B 宽片起点 | 当前批深度的 B 基址加 `max(b1SrcWiLeftOffGm,0)`；NDHWC 再乘总 cin |

不切宽时 l1W 保留原 wi，左右 pad 保留原属性。切宽的 l1W=0 时直接跳过该宽片，不构造越界源地址。更新宽片后重算 `kIter_=ceilDiv(singleShapeHo_*singleShapeWo_,baseK)`、`tailK_=singleShapeHo_*singleShapeWo_-(kIter_-1)*baseK`，以及 stepKaRound/stepKbRound。整除时 tailK_=baseK。

每个宽片从保存的 out2A1SrcAddrStart/out2B1SrcAddrStart 计算，不能在上一宽片地址上继续累加；退出当前批深度后恢复这两个基址。

例如 wi=25、wo=13、wk=3、strideW=2、dilationW=1、左右 pad=1，splitWo=8：

| 宽片 | startWo / 实际输出宽 | 逻辑 x 区间，左闭右开 | 左/右 pad | 真实 GM 宽 |
|---|---|---|---|---|
| 0 | 0 / 8 | `[-1,16)` | 1 / 0 | `[0,16)`，16 点 |
| 1 | 8 / 5 | `[15,26)` | 0 / 1 | `[15,25)`，10 点 |

#### 3.2.2 B 父段的行带、顶部 pad 与 mStartPt

令 `kbl1_=stepKb*baseK`、`b1SrcKAlign=kbStepIdx*kbl1_`，本父段有效长度为 `L=min(kbl1_,singleShapeHo_*singleShapeWo_-b1SrcKAlign)`。仅为 L>0 的父段执行搬运。

Kernel 的 CalRows2Copy(L,W) 与 Host 最大槽预算分别使用。这里 W=singleShapeWo_：L>W 时先取 ceilDiv(L,W)，若 L 不整除 W 且 `(2*L)%W!=0` 再加 1；L=W 时为 1；L<W 时，W 能被 L 整除取 1，否则取 2。

`b1SrcHo=b1SrcKAlign/W`、`b1SrcHoGm=b1SrcHo+hoStartIdx_`、`b1SrcHiGm=b1SrcHoGm*strideH-padUp`。令 `KhEff=(hk-1)*dilationH+1`，理论覆盖长度为 `(CalRows2Copy(L,W)-1)*strideH+KhEff`。

CalB1HiCopyParams 按以下顺序形成当前父段；差值全部先在有符号宽整数中计算：

```cpp
int64_t start = b1SrcHiGm;
int64_t length = (int64_t(rows)-1)*td.strideH + KhEff;
if (start + length <= 0 || start >= td.hi) return true;
hiParams.b1SrcHi = max(start, int64_t(0)) - hiStartIdx_;
hiParams.padUp = max(-start, int64_t(0));
if (start < 0) length += start;
if (start + length > td.hi) length = int64_t(td.hi) - start;
length = min(length, int64_t(params.singleShapeHi)-hiParams.b1SrcHi);
if (length <= 0) return true;
hiParams.hiCopyLen = length;
return false;
```

rows 是上面的 CalRows2Copy 返回值；`hiStartIdx_=max(hoStartIdx_*strideH-padUp,0)`；`params.singleShapeHi=min(hi,(singleShapeHo_-1)*strideH+KhEff)`。片段返回 true 表示该父段无有效行，不分配或入队。

CalB1GmOffset 在当前任务 B 基址上加 `hiParams.b1SrcHi*wi`，NDHWC 再乘总 cin。完成复制后，把 hiCopyLen、padUp 随 ping/pong 槽分别保存；消费该槽时恢复 `load3d.l1H=hiCopyLen`、`padList[2]=padUp`。底部兼容值仍为 255。

当前 K 块在真实父行带中的起点为 `mStartPt=((k-kbIdx)*baseK)%W+kbIdx*baseK`，其中 kbIdx=k%stepKb。该起点补回父段起始 K 未落在完整行边界上的余数。

例如 hi=7、ho=4、hk=3、strideH=2、padUp=1，W=8、baseK=16、stepKb=1：两个父段的 b1SrcHiGm 分别为 −1、3，理论覆盖均为 5 行，真实复制分别为 `[0,4)`、`[3,7)`；顶部 pad 分别为 1、0。

### 3.3 拆核：按原 stride 选取采样点

isSplitKernelHW 时先固定原 hkIdx/ wkIdx，逐行复制采样点。矩阵数为 load3d.l1W，nValue=1，dValue 为实际通道数；源矩阵步幅为 strideW 或 strideW*cin，每复制完一行再跨原 `wi*strideH` 或 `wi*strideH*cin`。

普通路径的 pad 单位是原 x 空间点；拆核后 pad 单位变为采样网格中的点数。对当前宽片 W=singleShapeWo_：

| 量 | 计算 |
|---|---|
| 原始首列 x0 | `startWo*strideW+wkIdx*dilationW-padLeft` |
| 左侧无效采样数 | `ceilDiv(max(-x0,0),strideW)` |
| 右侧无效采样数 | `ceilDiv(max(x0+(W-1)*strideW+1-wi,0),strideW)` |
| load3d.l1W | W 减上述两数，截到非负 |
| 真实首列 | x0 加左侧无效采样数乘 strideW |
| 原始首行 h0 | `(hoStartIdx_+b1SrcKAlign/W)*strideH+hkIdx*dilationH-padUp` |
| 顶部无效采样数 | `ceilDiv(max(-h0,0),strideH)` |
| 底部无效采样数 | `ceilDiv(max(h0+(rows-1)*strideH+1-hi,0),strideH)` |
| hiCopyLen | rows 减顶部、底部无效采样数，截到非负 |
| 真实首行 | h0 加顶部无效采样数乘 strideH |

rows 仍由本父段有效 K 与当前 W 经 CalRows2Copy 得到。任一真实复制维度为 0 时跳过本父段；GM 起点由上表真实首行、首列代入 x 的原始步幅，先确认有效再换为无符号地址。padList[0/1/2] 分别保存左、右、顶部的无效采样数，padList[3] 保持底部兼容值。每槽保存自身的实际行高和顶部 pad，不能与上一核位置的槽元数据混用。

```cpp
Nd2NzParams p{}; // 拆核路径，x 为 NDHWC
p.ndNum = load3d.l1W;
p.nValue = 1;
p.dValue = bL1cin1CopyLen;
p.srcDValue = td.cin;
p.srcNdMatrixStride = uint64_t(td.strideW)*td.cin;
p.dstNzC0Stride = uint64_t(hiCopyLen)*load3d.l1W;
p.dstNzNStride = 1;
p.dstNzMatrixStride = 16;
InitZeroValue(b1); // 工程内实现，先初始化将被读取的父槽
for (uint32_t row = 0; row < hiCopyLen; ++row) {
    DataCopy(b1[uint64_t(row)*load3d.l1W*16], xGm[srcOffset], p);
    srcOffset += uint64_t(td.wi)*td.strideH*td.cin;
}
```

原核坐标、stride/dilation 已进入 srcOffset 与逐行采样；下一阶段使用单位窗口。InitZeroValue 与搬入之间保留 MTE2 顺序依赖。示意通路：

```text
+--------------+-----------------------------+------------------+---------------+
| path         | source mapping              | L1B              | L0B           |
+--------------+-----------------------------+------------------+---------------+
| normal       | valid x rows                | forward x region | Load3D(hk,wk) |
| split kernel | fixed kh/kw; strided points | sampled grid     | Load3D(1,1)   |
+--------------+-----------------------------+------------------+---------------+
```

例如原 x 为 5×6、输出平面 3×3、核 3×3、stride=2、dilation=1、四侧 pad=1，固定 hkIdx=wkIdx=0。原采样行、列均为 `[-1,1,3]`，实际只搬行列 `[1,3]` 形成 2×2 父网格；Load3D 的左、上 pad 各为一个采样点。得到的 3×3 窗口位置应与直接读取原 x 的结果逐点相同。

## 4 L1A 到 L0A

### 4.1 LoadData2DParamsV2 字段表

当前 ComputeInit 后的 FP16/BF16 通路，L1A `(M1,K1,K0,M0)` 转为 L0A `(K1,M1,M0,K0)`：

| 字段 | 取值与单位 |
|---|---|
| mStartPosition/kStartPosition | 0/0，相对本次传入的 LocalTensor 子视图 |
| mStep | ceilDiv(baseUseK_,16)，沿父 K 分形 |
| kStep | ceilDiv(baseUseM_,16)，沿父 M 分形 |
| srcStride | alignedL1UseKa/16，512B 分形 |
| dstStride | kStep，512B 分形 |
| ifTranspose | true，交换分形内 K0/M0 |
| 源元素偏移 | `kOffset*16*16`，`kOffset=((kPos%stepKa)*baseK)/16` |

切宽时源元素偏移额外加 `(kaStepIdx*kal1_%singleShapeWo_)*16`，alignedL1UseKa 取真实父行带跨度。kPos 是传给 LoadL12L0a 的全局 k；有效 M=1 仍按第 2.3 节的计算 M=16 使用此转置通路。

```cpp
LoadData2DParamsV2 p{};
p.mStartPosition = 0;
p.kStartPosition = 0;
p.mStep = (uint64_t(baseUseK_)+15)/16;
p.kStep = (uint64_t(baseUseM_)+15)/16;
p.srcStride = alignedL1UseKa/16;
p.dstStride = p.kStep;
p.ifTranspose = true;
uint64_t kOffset = (kPos%td.stepKa)*td.baseK/16;
LoadData(a2, a1[kOffset*256 + splitWLocalOffset], p);
```

例如 baseUseM_=32、baseUseK_=32、alignedL1UseKa=64，得到 mStep=2、kStep=2、srcStride=4、dstStride=2。第二个 32 长 K 子块的源偏移为 512 个输入元素。

### 4.2 地址、补齐与 SDK 检查

源目的分形为 16×16×2B=512B。尾 K 的有效长度可以小于 16，对齐只决定源/目的物理范围；计算长度仍用实际 baseUseK_。源父跨度、目的块大小和字段宽度都在转换为 API 字段前检查。

```text
            +---------------------+
            | L1A: (M1,K1,K0,M0)  |
            +----------+----------+
                       |
                       |
                       v
+----------------------+----------------------+
| Load2D: mStep=ceil(K/16); kStep=ceil(M/16)  |
| ifTranspose=true                            |
+----------------------+----------------------+
                       |
                       |
                       v
            +----------+----------+
            | L0A: (K1,M1,M0,K0)  |
            +---------------------+
```

## 5 L1B 到 L0B

### 5.1 LoadData3DParamsV2 字段表

本节普通窗口使用 `LoadData3DParamsV2<T>`。Load3D 的 m 是输出位置 K，k 是核窗口通道 N：

| 字段 | 普通窗口取值 |
|---|---|
| filterW/filterH | wk/hk |
| filterSizeW/filterSizeH | `(wk>>8)&255` / `(hk>>8)&255`，与 SDK 高位字段配对 |
| strideW/strideH | 原 strideW/strideH |
| dilationFilterW/H | 原 dilationW/H |
| mStartPt | `((k-kbIdx)*baseK)%singleShapeWo_+kbIdx*baseK` |
| mExtension | baseUseK_，有效输出位置数 |
| kStartPt | `((curNIdx_%cinHkWkLoop_)*baseN)%(hk*wk*16)` |
| kExtension | baseUseN_，窗口通道长度 |
| channelSize | `ceilDiv(kStartPt+baseUseN_,hk*wk*16)*16` |
| l1H/l1W | 当前实际父行带高/真实复制宽 |
| padList[0..3] | 左/右/上/下，当前父段坐标系 |
| enTranspose/fMatrixCtrl | 0/0 |

例如 hk=wk=2、baseUseN_=64、kStartPt=0，channelSize=16、kExtension=64；若 baseUseN_=48，channelSize 仍为 16。**ceilDiv 的结果是窗口数，乘 16 后才是通道数**。

### 5.2 FMatrix、边界与 Repeat

初始化沿用 `padList[3]=255` 的底部兼容边界；当前父段的真实 l1H、顶部裁边和左右边界由每次搬运更新。这个兼容字段不能当成额外的数学 padDown。x 深度越界时不构造非法 GM 访问。

FP16/BF16 普通路径使用 Repeat `{0,1,0,ceilDiv(baseUseN_,16)}`；最后一项是目的 m 方向的分形步幅。以下片段使用经 SDK 编译确认的 WithStride 接口组合：

```cpp
// load3d 已按上表设置；当前对象为 FP16/BF16。
LoadDataRepeatParamWithStride repeatParam{
    0, 1, 0, static_cast<uint16_t>((baseUseN_+15)/16)};
SetLoadDataRepeatWithStride(repeatParam);
LoadDataWithStride(b2, b1, load3d);
```

版本宏缺失不能自动回退旧接口。部分 SDK 的 `LoadDataRepeatParam` 只有三参数，不能承载上述目的步幅；选用其他重载时，重新验证参数结构、Repeat 与 LoadData 的成对关系及结果布局。

例如 x 片段为 4×4、核 2×2、stride/dilation=1、无 pad，产生 3×3 个窗口；baseUseK_=9、baseUseN_=64 时，mExtension=9、kExtension=64、channelSize=16、目的步幅=4。L0B 预留按 16×64 个输入元素计算，有效 K 仍为 9。

### 5.3 拆核窗口参数与目标 N 片段

拆核参数为 filterW/H=1、strideW/H=1、dilationFilterW/H=1、filterSizeW/H=0、kStartPt=0、kExtension=16、channelSize=16，目的 Repeat 步幅为 1。每次 MMAD 的 n 同样取 16。

原 `hkIdx=hwkIdx/wk、wkIdx=hwkIdx%wk` 在外层决定采样地址与 L0C 列偏移。不能把单位窗口解释成整个卷积核退化。

```text
+-------------------------------------------------+
| original kh/kw -> signed x sampling coordinates |
+------------------------+------------------------+
                         |
                         |
                         v
   +---------------------+---------------------+
   | MTE2: selected points -> L1B sampled grid |
   +---------------------+---------------------+
                         |
                         |
                         v
    +--------------------+--------------------+
    | MTE1: 1x1 window -> one N0=16 fragment  |
    +--------------------+--------------------+
                         |
                         |
                         v
 +-----------------------+-----------------------+
 | MMAD: n=16 -> original kernel fragment in L0C |
 +-----------------------------------------------+
```

### 5.4 参数更新与分形单位

| 更新点 | 责任 |
|---|---|
| InitLoadToB2Params | 原几何、字段高位、enTranspose/fMatrixCtrl |
| CalcParamsL12L0b | 当前 N 起点、扩展长度与通道跨度 |
| LoadL12L0b | 当前有效 K、Repeat 与目的布局 |
| 拆核循环 | 原 kh/kw、真实父段、单位窗口与 L0C 列偏移 |

所有起点在父段局部坐标系内计算。当前 baseN、有效 baseUseN_、任务 singleCoreN 和实际父通道跨度分别保留。

## 6 Compute

### 6.1 MmadParams 字段表

| 字段 | 普通路径 | 拆核路径 |
|---|---|---|
| m | baseUseM_ | baseUseM_ |
| n | baseUseN_ | 16，每次一个原核位置的通道分形 |
| k | baseUseK_ | baseUseK_ |
| cmatrixSource | 0 | 0 |
| cmatrixInitVal | 本段当前 L0C 对应结果的首次实际贡献 | 对应核位置结果的首次实际贡献 |
| unitFlag/kDirectionAlign | 0/false | 0/false |

源码更新实际 m/n/k；内部槽占用独立按 16 对齐。不能以“尾块对齐”为由把 Mmad.k 固定成 baseK。

```cpp
MmadParams p{};
p.m = baseUseM_;
p.n = baseUseN_; // 拆核阶段此处为 16
p.k = baseUseK_;
p.cmatrixSource = 0;
p.cmatrixInitVal = isFirstMmad;
p.unitFlag = 0;
p.kDirectionAlign = false;
Mmad(l0c[dstL0cOffset_], a2[srcL0aOffset_], b2[srcL0bOffset_], p);
isFirstMmad = false;
if (uint64_t(p.m)*p.n < 2560) PipeBarrier<PIPE_M>();
```

小块的 PIPE_M barrier 沿用当前实现的阈值规则；它排序 M 流水，不替代 MTE1→M 或 M→F 的交接。

### 6.2 段内累加与初值

例如本段 K=70、baseK=32，三次 k 为 32、32、6，首次 cmatrixInitVal=true，其余 false。ND 段内换 batch/dout、宽片或父段时保持初值状态；新归约段才重新建立 L0C 初值。Compute 前已跳过的任务或形状不发布 M→F 事件；已申请 L0C 而无贡献时，普通路径经 ClearBaseMNL0C 形成零结果再正常交接，拆核路径先清零整块 L0C，见 [同步实现](sync-implementation.md#11-l0c-交接)。

## 7 CopyOut

### 7.1 Fixpipe 分发、行列方向与单位

写回由输出格式、核体积和 baseN 是否整窗共同决定：

| 输出格式 | 核体积=1 | 普通整窗 | 非整窗 baseN |
|---|---|---|---|
| NCDHW | DkhkwkEqOne，ROW_MAJOR | Normal，COLUMN_MAJOR | BaseNUndivided，COLUMN_MAJOR |
| NDHWC | DkhkwkEqOne，ROW_MAJOR | NormalNz2Nd，ROW_MAJOR | BaseNUndividedNz2Nd，ROW_MAJOR |
| DHWCN | DkhkwkEqOneNz2DHWCN，COLUMN_MAJOR | NormalNz2DHWCN，COLUMN_MAJOR | BaseNUndividedNz2DHWCN，COLUMN_MAJOR |

这里的名字为 LoadL0c2Gm* 函数后缀。配置与 FixpipeParamsArch3510 的模板实参配对；FP32 输出均为 NoQuant，不做 FP16/BF16 输出舍入。

整窗且 curSingleCoreDk_=1，记 `hw=hk*wk`、`ciUse=min(当前真实剩余通道,ceilDiv(baseUseN_,hw))`。本节 mAligned 表示 L0C 相邻 N0 片段的物理 M 跨度：普通窗口取 alignUp(baseUseM_,16)，拆核取 baseM，与第 5.3 节的 L0C 片段基址配对。有效 co 数仍使用恢复后的 baseUseM_：

这是生成实现的布局一致性条件：先确认计算写入的真实片段跨度，再构造 Fixpipe。拆核 M 尾块不能仅按有效 M 重算跨度；例如 baseM=144、有效 M=113 时，计算片段间隔为 144，而 alignUp(113,16)=128。

| 字段 | NCDHW | NDHWC | DHWCN |
|---|---|---|---|
| mSize | hw | baseUseM_ | baseUseM_ |
| nSize | ciUse | ciUse | ciUse |
| 矩阵数 | dnNum=baseUseM_ | ndNum=hw | dnNum=hw |
| srcStride | hw*mAligned | hw*mAligned | hw*mAligned |
| 源矩阵步幅 | srcNzMatrixStride=1 | srcNdStride=mAligned | srcNzMatrixStride=mAligned |
| 其他源步幅 | srcNzC0Stride=mAligned | — | srcNzC0Stride=1 |
| dstStride | dk*hw | CinG*dk*hw | 总 cout |
| 目的矩阵步幅 | dstDnMatrixStride=CinG*dk*hw | dstNdStride=CinG | dstDnMatrixStride=CinG*cout |

源 stride 单位为 64B，目的 stride 单位为 float 元素。普通核的 nSize 是实际 ci 数，mSize 也可能被重解释为核位置数；不能统一写成 baseUseM_/baseUseN_。

### 7.2 参数构造与数值示例

以下为普通整窗 NCDHW 输出，dstOffset 已含当前任务和子块的 co/ci/kd 起点，l0c 使用该子块的起始地址：

```cpp
FixpipeParamsArch3510<CO2Layout::COLUMN_MAJOR> p{};
uint32_t hw = td.hk*td.wk;
uint32_t mAligned = isSplitKernelHW ? td.baseM : (baseUseM_+15)/16*16;
p.params.dnNum = baseUseM_;
p.params.srcNzMatrixStride = 1;
p.params.dstDnMatrixStride = uint64_t(cinG)*td.dk*hw;
p.params.srcNzC0Stride = mAligned;
p.mSize = hw;
p.nSize = min(cinRemainLen_, (baseUseN_+hw-1)/hw);
p.srcStride = hw*mAligned;
p.dstStride = td.dk*hw;
p.quantPre = QuantMode_t::NoQuant;
p.unitFlag = 0;
SetAtomicAdd<float>();
Fixpipe<float,float,CFG_COLUMN_MAJOR>(dedwGm[dstOffset], l0c, p);
SetAtomicNone();
```

例如 `hk=wk=2、dk=2、CinG=17、baseUseM_=16、baseUseN_=64`，当前为最后一个 ci 分形，只剩 1 个有效通道：dnNum=16、mSize=4、nSize=1、srcStride=64、dstStride=8、dstDnMatrixStride=136、srcNzC0Stride=16。nSize=16 会把补齐通道写到相邻有效区域。

1×1×1 的 NCDHW/NDHWC 简化写回使用 ROW_MAJOR：mSize=baseUseM_、nSize 为实际 ci、srcStride=mAligned、dstStride=CinG，ndNum=1。DHWCN 简化写回使用 COLUMN_MAJOR，dstStride=总 cout。

### 7.3 非整窗输出与归还

`(baseN/16) % hw != 0` 时，LoadL0c2GmBaseNUndivided、BaseNUndividedNz2Nd、BaseNUndividedNz2DHWCN 分别处理三种输出格式。当前任务固定 kd，curSingleCoreDk_=1；N 的顺序仍为 `(ci1,kh,kw,ci0)`，不得按连续 ci 平铺。

#### 7.3.1 写回状态的初始化与重放

每次 SetSingleShape、进入新的 IterateAll 前初始化 `head_=tail_=nLoopHead_=0`、`lastNIdx_=-1`，cinRemainLen_ 与 nLoopCinRemainLen_ 均为当前任务实际 singleShapeCin_。这些是 Kernel 运行量，不写入 TilingData。

CalcL0cParams 在每个输出子块写回前执行：

```cpp
bool tailCinExist = singleShapeCin_%16 != 0;
bool isNLoop = lastNIdx_ != curNIdx_;
if (tailCinExist && isNLoop && curNIdx_%cinHkWkLoop_ == 0)
    cinRemainLen_ = singleShapeCin_;
if (isNLoop) {
    nLoopHead_ = head_;
    nLoopCinRemainLen_ = cinRemainLen_;
}
head_ = nLoopHead_;
cinRemainLen_ = nLoopCinRemainLen_;
```

同一 N 子块用于另一个 M 子块时，应重放该 N 的起始 head_/剩余通道；不能使用上一次 M 写回后的终止状态。写回结束才更新 lastNIdx_=curNIdx_。ORDER_M 的 N 回绕后，同一核深度重新从完整窗口起点开始。

#### 7.3.2 每个核区间的参数与地址

令 `usedN=(curNIdx_%cinHkWkLoop_)*baseN`、`c1hkwk=baseUseN_/16`；mAligned 取第 7.1 节的实际 L0C 物理 M 跨度。nValueSum 是当前 L0 子块已写的核位置数，初值 0；j 是已跨过的 ci 分形数，初值 0。

每轮在 `nValueSum<c1hkwk` 时执行，`tail_=min(hw,head_+c1hkwk-nValueSum)`、`nValue=tail_-head_`，nValue 必须为正。只迭代真实片段，不用无符号的 `c1hkwk-(hw-head_)` 预估循环次数。

| 量 | 计算与单位 |
|---|---|
| srcOffset | `nValueSum*mAligned*16`，相对本子块 l0c 的 float 元素偏移 |
| 当前 ci | `(usedN/(hw*16)+j)*16`，相对本任务 ci 起点 |
| 当前 co | `curMIdx_*baseM`，相对本任务 co 起点 |
| nSize | `min(cinRemainLen_,16)`，实际 ci 数 |
| NCDHW dstOffset | `当前co*CinG*dk*hw+当前ci*dk*hw+head_` |
| NDHWC dstOffset | `当前co*CinG*dk*hw+head_*CinG+当前ci` |
| DHWCN dstOffset | `当前co+head_*CinG*cout+当前ci*cout` |

output 已含任务的 g/co/ci/kd 起点；上表只加子块偏移。拆核时若 baseM 不等于 alignUp(baseUseM_,16)，仍按 baseM 恢复 N0 片段间距；只用有效 M 控制真实 co 写回数。按格式设置这一片段的 Fixpipe 参数：

| 字段 | NCDHW | NDHWC | DHWCN |
|---|---|---|---|
| mSize | nValue | baseUseM_ | baseUseM_ |
| nSize | 上表实际 ci 数 | 同左 | 同左 |
| 矩阵数 | dnNum=baseUseM_ | ndNum=nValue | dnNum=nValue |
| srcStride | hw*mAligned | hw*mAligned | hw*mAligned |
| 源矩阵步幅 | srcNzMatrixStride=1 | srcNdStride=mAligned | srcNzMatrixStride=mAligned |
| 其他源步幅 | srcNzC0Stride=mAligned | — | srcNzC0Stride=1 |
| dstStride | dk*hw | CinG*dk*hw | cout |
| 目的矩阵步幅 | dstDnMatrixStride=CinG*dk*hw | dstNdStride=CinG | dstDnMatrixStride=CinG*cout |
| API 朝向 | COLUMN_MAJOR | ROW_MAJOR | COLUMN_MAJOR |

quantPre=NoQuant、unitFlag=0。按所选朝向调用 `Fixpipe(output[dstOffset],l0c[srcOffset],params)` 后，令 `nValueSum+=nValue`、j 加 1。UpdateHwKIterState 将 head_ 更新为 `tail_==hw ? 0 : tail_`；tailCinExist 且 head_ 回到 0 时，剩余通道减 16。生成代码用有符号差值并截到 0，避免最后一个不足 16 通道的分形产生无符号回绕。尚未到完整窗口末端时，剩余通道不减。

#### 7.3.3 用尾通道检查核区间连续性

取 hk*wk=4、baseN=48、singleCoreN=64；当前任务只有 1 个真实 ci，其余 ci0 为内部补齐。当前 co 为 0、dk=2、CinG=17、cout=32，output 指向当前任务 ci/kd 起点：

| 子块 | baseUseN_ | head_→tail_ | nSize | 三种格式的 dstOffset |
|---|---|---|---|---|
| 0 | 48 | 0→3 | 1 | NCDHW=0、NDHWC=0、DHWCN=0 |
| 1 | 16 | 3→4 | 1 | NCDHW=3、NDHWC=51、DHWCN=1632 |

第一块写核位置 0、1、2，第二块只写核位置 3；第二块起始 l0c 属于新的 CO1 输出子块，因此其 srcOffset 重新为 0。换 M 后重放本 N 的起点；换任务或归约段后重新执行 SetSingleShape 初始化。

### 7.4 原子范围与队列归还

SetAtomicAdd 覆盖本次输出子块的全部实际 Fixpipe 片段，结束后恢复 SetAtomicNone。CO1 队列在最后一次 Fixpipe 读取提交后发布归还依赖；下一次 AllocTensor 等待该依赖，双缓冲时再切换 l0cPingPongFlag_。Fixpipe 写回范围仅含有效 co/ci，不能把内部补齐通道作为零值原子写到相邻任务。

## 8 SDK 适配检查

- API 参数类型、字段宽度和 Repeat/LoadData 成对重载按当前头文件确认。
- 源矩阵朝向、通道总步幅与 32B/64B/512B 单位分别检查。
- 核高宽、stride/dilation、pad 和 kStartPt 超普通字段范围时选择配套拆核路径。
- 有效区域先裁出，再构造 GM 地址；负深度/负起点不直接提交搬运。
- FP16/BF16 不加载 FP32 输入通路的逐条窗口重排算法。
- Fixpipe 选型同时检查输出格式、核体积和整窗关系。
- 数值示例先复算参数，再做 SDK 编译与设备验证；参数复算不代表设备验收。

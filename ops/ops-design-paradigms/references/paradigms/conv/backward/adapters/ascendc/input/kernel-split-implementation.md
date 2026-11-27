# kernel-split Kernel 侧实现（Ascend C 适配）

> **层级**：input 分支适配层（adapters/ascendc/input/）。算法与求解见 [kernel-split 模板](../../../generic/input/tiling/templates/kernel-split/tiling.md)；AIC/AIV 协同机制见 [C/V 协同](cv-collaboration.md)。本页给出子核状态机、子核 pad 映射与输出重排的落地写法。

## 1 编译实例与入口分发

kernel-split 固定 `loadB2Condition=TPL_REVERSE_ONLY`（子核窗口由相位重建，权重只逆序不转置），`kernelSplitMode=1`（split-HW）/`2`（split-H），loadB1Condition=TPL_GM_TO_L1；key 编码与字段见 [Tiling 绑定](tilingkey-templates.md)。入口分发保持公共三实参形态：

```cpp
// Shared declarations and dtype/format macros come from the generated project.
GET_TILING_DATA(tilingData, tiling);
GM_ADDR usrWsp = workSpace ? GetUserWorkspace(workSpace) : nullptr;
if (!usrWsp && (tilingData.enableVecTrans || tilingData.kSUseWorkSpace)) {
    return;
}
#if CONV3DDX_CV_RATIO == 2
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
#elif CONV3DDX_CV_RATIO == 1
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
#else
#error "Bind the MIX execution contract"
#endif
if (tilingData.initOutputFlag == static_cast<int32_t>(InitOutputFlag::L0_INIT)) {
    Conv3dDxInitOutput<DTYPE_Y> initOutput;
    initOutput.Init(y, tilingData);
    initOutput.Process(y);
    initOutput.Destroy();
}
if ASCEND_IS_AIV_SCALAR {
    if (tilingData.enableVecTrans) {
        DxVecTranspose::Conv3dDxVecTranspose<DTYPE_FILTER> transpose;
        transpose.Init(filter, usrWsp, tilingData);
        transpose.Process();
        transpose.Destroy();
    }
}
Conv3dDxKsBlock<DTYPE_FILTER, FORMAT_FILTER, DTYPE_OUT_BACKPROP,
    FORMAT_OUT_BACKPROP, DTYPE_Y, FORMAT_Y, DTYPE_BIAS, FORMAT_BIAS,
    TPL_REVERSE_ONLY, kernelSplitMode> op;
op.Init(filter, out_backprop, y, usrWsp, tilingData);
op.Process();
```

与公共入口的差异：

| 差异点 | 说明 |
|---|---|
| 核类型 | 按目标配置使用 MIX（AIC_1_1 或 AIC_1_2）；split-HW 的 AIV 承担输出重排，split-H 的 AIV 仅参与初始化/前置转置 |
| workspace | `enableVecTrans/kSUseWorkSpace` 任一置位时强制要求用户 workspace，缺失直接返回 |
| kernel 类 | `Conv3dDxKsBlock`（基于公共滑动窗块扩展）；split-H 模式 AIV 在 Init/Process 即返回 |
| 主循环分发 | 公共 `IterateAll` 检测 `kernelSplitMode==SPLIT_KERNEL_HW` 时改走 `IterateAllForKernelSplit` |

Host 可按 [H/HW 成本比较](../../../generic/input/tiling/templates/kernel-split/tiling.md#23-hhw-的成本比较) 调整模式尝试顺序。Kernel 只消费最终 kernelSplitMode、kSUseWorkSpace 和子核几何；优先选择 H 不改变其错行写回，选择 HW 则仍完成对应的 UB/workspace 交织协议。收益模型放宽的是宽度门槛，不能放宽指令、地址或同步约束。

### 1.1 执行路径

初始化和可选 VecTranspose 完成后，`Conv3dDxKsBlock` 按以下路径执行。`CalBasicBlock` 调度任务，`IterateAll*` 推进任务内的输出块和子核；每个子核完成全部有效 K/kd 贡献后才交接结果。

```text
               Init: InitTilingData; dedx_.Init; InitMixCoreBuffer
                                        |
                                        v
                             Process / CalBasicBlock
                                        |
                  +---------------------+---------------------+
                  |                                           |
                  v                                           v
               split-H                                    split-HW
     CalBasicBlockCoreForSplitH                       CalBasicBlockCore
        SetSingleShapeParams                      IterateAllForKernelSplit
        fixed H kernel phase                        IterateForKernelSplit
                  |                                           |
                  +---------------------+---------------------+
                                        |
                                        v
                       LoadToA1/B1 -> LoadToA2/B2 -> Mmad
                                        |
                                  complete K/kd
                                        |
                  +---------------------+---------------------+
                  |                                           |
                  v                                           v
               split-H                                    split-HW
        Fixpipe -> dedx rows                       Fixpipe -> UB/workspace
                  |                                           |
                  |                               AIV: Rearrange2Gm -> dedx
                  |                                           |
                  +---------------------+---------------------+
                                        |
                           more phases/blocks/tasks?
                                        |
                                        +-- yes --> repeat task/phase setup
                                        |
                                        | no
                                        v
                            FreeB1Tensor (when held)
                                        |
                                        v
                            dedx_.End inside Process
```

`FreeB1Tensor` 的跨轮保留条件见第 4 节；split-H 在自己的任务/父段边界释放。`End` 归还剩余事件与队列资源，随后由所属 TPipe 完成收尾。

## 2 任务解码与基本块循环

空间总量按子核口径重建：`kSM_=ceilDiv(hi,strideH)*wi`（split-H）或 `ceilDiv(hi,strideH)*ceilDiv(wi,strideW)`（split-HW）；子核循环数 `kSCnt=strideH`（split-H）或 1（split-HW，相位在核内交织）。`totalTasks=kSCnt*batch*group*ceilDiv(di,singleCoreDin)*ceilDiv(kSM,singleCoreM)*ceilDiv(cinG,singleCoreCin)`。轴序：`dk>1 → M→D→N`；`hk/strideH>1 或 mCnt>nCnt → D→M→N`；其余 `D→N→M`。

**split-H 的子核号与输出相位**：`kSCoreIdx_` 是从任务号最外层解码得到的子核号，用于定位 weight；`kernelIdx` 是 dedx 输出行相位。由 `ih=oh*strideH-padUp+kh` 得到：`kernelStartIdx=(strideH-padUp%strideH)%strideH`，`kernelIdx=(kSCoreIdx_+kernelStartIdx)%strideH`。

例如 strideH=3、padUp=1 时，kernelStartIdx=2，子核号 0、1、2 分别写输出行相位 2、0、1。`splitHStartIndex=padUp%strideH` 表示与输出相位 0 配对的子核号，不能替代上述输出相位偏移。

`kSTailIdx=hi%strideH` 非零时，尾部相位的空间总量收缩为 `kSTailM_=(hi/strideH)*wi`；解码后按 `kernelIdx>=kSTailIdx` 切换 `mCnt/mCoreTail`，超出 `kSTailMCnt_` 的任务直接跳过。

**子核偏移**（`CalcBlockKSOffset`，在公共块偏移之上追加）：

| 对象 | 追加偏移 | 说明 |
|---|---|---|
| weight | `kSCoreIdx_*kernelSplitStrideB_` | `kernelSplitStrideB_`：NDHWC 为 `wk*cin`，NCDHW 为 `wk`，前置转置态为 `wk*alignUp(cin,16)`（见 [Kernel 模板](kernel-template.md) 第 4.2.1 节） |
| dedy | `alignedHoStartIdx*wo`（NCDHW）或 `*wo*cout` | 行相位起点，负起点截断到 0 |
| dedx | `kernelIdx*wi`（NCDHW）或 `*wi*cin` | 输出行相位 |

split-H 的结果已经包含完整宽度，只需将相邻子问题行写到间隔 strideH 的 dedx 行。下面用 `kernelIdx=1` 示意行地址；每个方括号是一整行的有效宽度：

```text
subproblem rows                       dedx rows
[row 0] -- Fixpipe ------------------> ih = 1
                                      ... other H phases ...
[row 1] -- Fixpipe ------------------> ih = 1 + strideH
                                      ... other H phases ...
[row 2] -- Fixpipe ------------------> ih = 1 + 2*strideH
```

NCDHW 使用 `LoadL0c2GmDnForKernelSplitH`，NDHWC 使用 `LoadL0c2GmNdForKernelSplitH`；二者都从 L0C 直接写 GM，并按首块、整行和尾块恢复行间距。

## 3 子核状态机

### 3.1 split-HW：四相位子核分解

子核按 H/W 相位各拆两半：`wkLeft=ceilDiv(wk,strideW)、wkRight=wk-wkLeft`（为 0 时取 wkLeft），`hkUp/hkDown` 同理。列表 `[L,R,L,R]×[U,U,D,D]`，`splitIndex=splitHIndex*strideW+splitWIndex`：

| 原核 | 4 个子核 | 说明 |
|---|---|---|
| 4×4 | 2×2 × 4 | 子核全等 |
| 3×3 | 2×2 / 2×1 / 1×2 / 1×1 | 子核不等，K 长度随相位变化 |
| 2×2 | 1×1 × 4 | 子核全等 |
| 1×1, stride2 | 列表为 1×1 × 4 | 仅一个相位有乘积，其余相位按零或 bias 语义处理；不是把同一乘积算四次 |

初值相位由 pad 决定：`splitWStartIndex=padLeft%strideW`、`splitHStartIndex=padUp%strideH`，首个 `splitIndex=splitHStartIndex*strideW+splitWStartIndex`。`splitWi_=ceilDiv(wi,strideW)` 为子核空间宽度。

以下把**输出相位**和**核采样相位**分开画。S00/S01/S10/S11 分别对应原 weight 的 `(kh%2,kw%2)`；`splitIndex=0/1/2/3` 与其同序，`rearrangeHIndex_/rearrangeWIndex_` 始终按输出位置推进：

```text
+---------------------+-------+-------+-------+-------+
|  output phase (H,W) | (0,0) | (0,1) | (1,0) | (1,1) |
| padUp%2=padLeft%2=0 |  S00  |  S01  |  S10  |  S11  |
|      splitIndex     |   0   |   1   |   2   |   3   |
| padUp%2=padLeft%2=1 |  S11  |  S10  |  S01  |  S00  |
|      splitIndex     |   3   |   2   |   1   |   0   |
|  W pair (output H)  |   0   |   0   |   1   |   1   |
+---------------------+-------+-------+-------+-------+
```

每一对 W 输出相位完成后才通知 AIV 的 subblock 0 重排。pad 的商影响 dedy 起点和子核边界，不能只按上表切换 weight。

每次 `Iterate` 推进一轮相位轮转：W 相位最快（`splitWIndex/rearrangeWIndex` 在 strideW 处回绕），W 回绕时推进 H 相位；全部相位完成即一轮空间块。切换时依次刷新：

| 刷新项 | 内容 |
|---|---|
| `RecalcBaseUseMForKernelSplit` | split-HW 的第二个 H 输出相位下 `hi%strideH≠0` 时两个子核 M 总量不同：`subKernelM=hi/strideH*splitWi_`，curM 超界置 `needComputeFlag_=false`，尾块收缩 `baseUseM_/mmad.m` |
| `UpdateHoIdxForKernelSplit` | `backpropPadUp==2` 时 rearrangeH 推进令 `curHoIdx_++`；`hk∈{2,4}` 子核等高跳过；其余重算 `curHoSize_`（行带随相位移动） |
| `RecalcStepForKernelSplit` | `curStepKa=ceilDiv(channelSize_*splitHkWkList[splitIndex],baseK)`，curStepKb 同；`stepKb%curStepKb==0` 时直接取 stepKb（子核间整除可放大驻留） |
| `kIter/tailK` | `tmpSingleCoreK=(group==1?cout1:cout1G)*splitHkWkC0List[splitIndex]`，按 baseK 重算迭代数与尾段 |
| `load3d_.filterW/filterH` | 当前子核核几何 |

`channelSize_/coutChannelSize_` 在初始化时按最大子核求出：cout 全载取 `alignUp(singleCoreCout,C0)`，否则 `min(curStepKa*baseK/splitHkWkList[0], alignedCout)`。

### 3.2 split-H：单子核刷新

split-H 每个任务固定一个子核（`splitIndex=0`），进入任务时用 `SetSingleShapeParams(curSplitHk,curBackPropPadUp)` 重设：`splitHkList[0]=kSCoreIdx_<baseHkIter?ceilDiv(hk,strideH):hk/strideH`、`splitWkList[0]=wk`，随之刷新 splitHkWk（C0）、curStepKa/Kb 与 load3d 参数。`curBackPropPadUp` 由第 3.3 节 H 轴公式给出。

### 3.3 子核 pad 映射

子核的反向边界由原边界按相位重算。W 轴（split-HW，当前仅 stride=2，位运算简化）：

| 量 | 公式 |
|---|---|
| 子核长 | `subK=(kernelSize+stride-idx-1)>>1` |
| 左补边 | `subPadBegin=(subK-1)-((kernelSize-idx-backPad)>>1)` |
| 右补边 | `subPadEnd=(subK-1)-((kernelSize-idxReverse-backPad)>>1)`，`idxReverse=kernelSize-((subK-1)*stride+1+idx)` |

H 轴（split-H，通用 stride）：先取 `subK=ceilDiv(kernelSize-idx,stride)`，再算 `missPoint=(kernelSize-idx-backPad-1)>0 ? (kernelSize-idx-backPad-2)/stride+1 : 0`，`subPadUp=(subK-1)-missPoint`。**奇数 wi 修正**：`(backpropPadLeft∈{0,2} 且 wk∈{4,2}) 或 (backpropPadLeft==1 且 wk==3)` 时子核 0/2 右补边 +1，否则子核 1/3 右补边 +1——对应子核输出交换顺序的场景；下补边由 load3d 固定 pad 表达，不需要显式计算。

**同一相位，窗口仍随 pad 移动。** 固定 `ih=0`，只看核高 4、strideH=2 的 H 轴；`u0/u1` 为 dedy 第 0/1 行，越界行取零。两种 pad 都选择原核的偶数行，逆序后的子核相同，但配对的 dedy 行不同：

```text
original kh (reversed):        2       0
subkernel weight:             w2      w0
                               |       |
ih=0, padUp=0:                 0      u0
ih=0, padUp=2:                u0      u1
```

因此，相位编号相同不能直接复用旧滑窗；`UpdateHoIdxForKernelSplit` 与子核 pad 共同确定实际行带，负补边按公共裁剪协议处理。

## 4 K 循环与驻留

K 循环结构与公共 [Kernel 模板](kernel-template.md) 一致（父段、尾段、槽轮转），差异只在三处：

1. **父段长度按子核窗口**：L1A/L1B 的驻留通道数（channelSize_/coutChannelSize_）与窗口 `splitHkWkC0List[splitIndex]` 相乘得到实际 K 跨度；子核不等长时每次切换重算 curStep（3.1 节）。
2. **B1 跨轮保留**：`dk==1` 且 B1 全载（`isB1FullLoadFlag_`）时整个任务循环复用同一份 L1B，循环结束后统一 `FreeB1Tensor`；该保留属于 split-HW；split-H 按自身任务与父段边界释放，不能套用 split-HW 的跨轮保留条件。
3. **dedy 驻留行带**：`curHoSize_` 由 `CalFmapHForKernelSplit`（模板第 5 节同式）按当前子核核高计算，相位切换时随行带起点更新。

### 4.1 输入重排与子核滑窗

固定 co、ci 和 kd，原 weight 的空间采样如下。图只画普通非 cout 全载的一个子核；`a/c/i/k` 等均是原权重值，箭头末端才是逆序后的核空间次序，物理分形轴序见图后说明：

```text
weight / GM                         L1B: splitIndex=0           L0B: reverse only
+---+---+---+---+                   +---+---+                   +---+---+
| a | b | c | d |                   | a | c |                   | k | i |
| e | f | g | h |-- MTE2 ---------->| i | k |-- MTE1 ---------->| c | a |
| i | j | k | l |                   +---+---+                   +---+---+
| m | n | o | p |
+---+---+---+---+
select: kh=0,2; kw=0,2
```

| 路径 | 子核选择的位置 | 搬运与布局 |
|---|---|---|
| split-HW，非 cout 全载 | `LoadToB1` | NCDHW 使用 `LoadToB1Dn2NzTransposeForKernelSplitHW`；NDHWC/前置转置态使用 `LoadToB1Nd2NzForKernelSplit`；按 strideH/strideW 跳读后紧凑放入 L1B |
| split-H | `LoadToB1` | H 跳读步长为 strideH，W 保留完整行；NCDHW 使用 `LoadToB1Dn2NzTransposeForKernelSplitH`，其余使用 `LoadToB1Nd2NzForKernelSplit` |
| split-HW，cout 全载 | `LoadToB2ForKernelSplitB1FullLoad` | L1B 保留原 hk/wk 网格；MTE1 按当前 splitIndex 选择子核并逆序，跨相位复用这份父段 |

普通紧凑 L1B 的空间轴是 `(splitHk,splitWk)`，完整轴序为 `(Cout1,splitHk,splitWk,Cin1,Cin0,Cout0)`；cout 全载父段保留 `(Cout1,Hk,Wk,Cin1,Cin0,Cout0)`。因此，全载路径的 B2 起点和跨步按原核网格计算。

dedy 一路对应如下。图中的零仅表示 stride 插零，边界和通道补齐仍按当前窗口处理：

```text
GM dedy / real rows
  [a b c ...]
  [d e f ...]
        |
        +-- split-HW / MTE2 --> L1A: [a b c ...]
        |                           [d e f ...]
        |                                |
        |                                v
        |                         Load3D -> L0A
        |                         H/W unit-step windows
        |
        +-- split-H / MTE2 ---> L1A: [a 0 ... b 0 ... c ...]
                                    [d 0 ... e 0 ... f ...]
                                         |
                                         v
                                  Load3D -> L0A
                                  H unit-step; W expanded
```

`LoadToA1` 装入当前真实 dedy 行带，`LoadToA2` 用子核尺寸和子核 pad 取窗。split-H 的 W 前放大已在 L1A 表达，Load3D 在展开视图上仍按单位窗口步长执行。`curHoIdx_/curHoSize_` 改变时同步更新行带与 padList；K 内继续保持 Cout 分形、子核高宽、Cout0 的配对顺序。

允许双份 buffer 时，下一父段的 MTE2 可与当前 L0 操作数的搬运/计算重叠；同一 L1 槽须等末次 MTE1 读取完成后复用。相位切换本身不强制重新搬运，是否保留由全载标志和实际父段身份决定。

## 5 输出重排通路（split-HW）

### 5.1 两个 W 相位恢复一行

split-HW 的 L0C 内容是当前子核完成核深度与通道归约后的结果，使用**子核坐标**（宽 `splitWi_=ceilDiv(wi,strideW)`），写回前需交织回原 stride 网格。写回链替换公共 `LoadL0c2Gm`：

```text
AIC:  L0C --Fixpipe--> UB（或 workspace）   # 按相位暂存，不直接写 GM
AIV:  UB/workspace --Interleave--> UB       # 两个 W 相位交错
      UB --DataCopyPad--> GM dedx           # 首块/中间行/尾块拼接
```

```text
+---------------------+       +---------------------+
| W 相位 0: a0 a1 a2  |       | W 相位 1: b0 b1 b2  |
+----------+----------+       +----------+----------+
           |                             |
           +--------------+--------------+
                          v
             +------------+------------+
             | 交织: a0 b0 a1 b1 a2 b2 |
             +------------+------------+
                          |
                          v
            +-------------+-------------+
            | dedx 行: 只保留 wi 个元素 |
            +---------------------------+
```

图中 a/b 表示同一 H 输出相位的两个 W 输出相位；它们已按 rearrangeWIndex 绑定，不按 splitIndex 的数值直接排列。wi 为奇数时，最后一个补齐位置不写回。

H 方向也按相位还原。固定一个通道和深度，以下只画有效元素，内部对齐间隔在下一节处理；a/b 属于 H 相位 0，c/d 属于 H 相位 1：

```text
Fixpipe 后的相位平面                                  dedx / GM
+---------------------------+                         +---------------------------+
| H=0, W=0       H=0, W=1   |                         | ih=0: [a00 b00 a01 b01]   |
| [a00 a01]      [b00 b01]  |                         | ih=1: [c00 d00 c01 d01]   |
| [a10 a11]      [b10 b11]  |                         | ih=2: [a10 b10 a11 b11]   |
|                           |-- Interleave ---------->| ih=3: [c10 d10 c11 d11]   |
| H=1, W=0       H=1, W=1   |                         |                           |
| [c00 c01]      [d00 d01]  |                         |                           |
| [c10 c11]      [d10 d11]  |                         |                           |
+---------------------------+                         +---------------------------+
```

AIV 每次只交织同一 H 相位的两个 W 平面，再由目的行跨步隔行写回；两个 H 相位都完成后，dedx 的行顺序完整。

### 5.2 首块、中间行与尾块

**相位切分（CalcCutInWIndex）**：`doubleBaseUseM=baseUseM_*2`（两个 W 相位），`wiUsed=alignUp(wi,strideW)`；上一块遗留的行内剩余 `curWiPos=wiUsed-(mSize%wiUsed)` 决定三段：

| 段 | 条件与长度 |
|---|---|
| 首块 `headWi_` | `0<curWiPos<=doubleBaseUseM` 且非整行时为 curWiPos，否则无首块 |
| 中间块 `midHi_` | `(doubleBaseUseM-headWi_)/wiUsed` 个整行 |
| 尾块 `tailWi_` | 剩余宽度 |

一个 M 块可能从行中部开始，也可能在下一行中部结束。每个相位先独立按行段对齐，再交织；图中 `pad` 是内部占位，不属于有效 dedx：

```text
+---------------------------------------------------------------+
| L0C，单个 W 相位:                                             |
|   [真实首块][srcWi][srcWi] ... [真实尾块]                     |
+-------------------------------+-------------------------------+
                                |
                                | Fixpipe: 首块 / 整行 / 尾块
                                v
+-------------------------------+-------------------------------+
| UB/workspace，同一 W 相位:                                    |
|   [首块|补齐][整行|补齐][整行|补齐] ... [尾块|补齐]           |
+-------------------------------+-------------------------------+
                                |
                                | 交织配对的 W 平面
                                v
+-------------------------------+-------------------------------+
| 拷贝有效首块 / 整行 / 尾块到 dedx                             |
+---------------------------------------------------------------+
```

**Fixpipe 三段搬出**：`dstStride=hwSize`（对齐后单相位平面大小），中间块按 `dnNum=midHi_` 循环。两种目的存储的 mSize 不同：

| 行段 | 直接写 UB 的 mSize | 写 workspace 的 mSize |
|---|---|---|
| 首块 | `alignUp(realHeadWi,16)` | `realHeadWi` |
| 中间行 | `alignUp(srcWi,16)` | `srcWi` |
| 尾块 | `alignUp(realTailWi,16)` | `realTailWi` |

两路的目的行段起点均按 16 元素对齐；L0C 源偏移始终按实长×16 推进，如 `srcOffset+=realHeadWi<<4`。UB 路径会搬出源端对齐区域，最终 DataCopyPad 只写有效宽度。workspace 模式目的偏移为 `hwSize*baseUseN*rearrangeWIndex+aicBlockIdx*(hwAlign*2)`（每核独立槽位）；UB 模式直接进 `vecOutBuf_`。

### 5.3 重排存储与 AIV 分段

一次 AIV 分段取 `L=curUseN*hwSize` 个 DstT 元素。`InterleaveUbOutForKernelSplit` 的两个输入各占 L，交织输出占 2L：

```text
vecOutBuf_，元素偏移
0                  L                  2*L                                  4*L
+------------------+------------------+------------------------------------+
| src0: W 相位 0   | src1: W 相位 1   |  交织输出: 2*L 个元素              |
+------------------+------------------+------------------------------------+
输入区保持存活直至 Interleave 完成
```

输出区从 `2*L` 开始，`dst0Ptr/dst1Ptr` 是每轮向量交织的两段写指针；`dst1Ptr` 起点为 `2*L+vfLen`。资源预算的“四份”来自两份输入加两份输出长度，不能把它们当成四个独立 workspace。向量寄存器宽度从目标平台取得，对齐覆盖按实际指令范围检查。

workspace 路径中，`hwAlign=alignUp(splitWi_,16)*ceilDiv(baseM,splitWi_)*baseN`，单位为 DstT 元素；一个 AIC 的重排槽基址为 `aicBlockIdx*(2*hwAlign)`，当前两个完整相位平面各占 `hwSize*baseUseN_`。AIV 按 N 分段读入 UB：第二相位的 GM 源起点跨过完整平面，UB 目的起点只跨过本段的 L，两种跨度分别计算。

**AIV 交织与写回（Rearrange2Gm）**：

| 路径 | 流程 |
|---|---|
| UB 直达 | `Interleave`（两个相位平面按向量寄存器宽度交叉）→ `V_MTE3` 事件 → `LoadUbToGm` |
| workspace 中转 | 按 `ceilDiv(dataLen*4*sizeof(DstT),UB)` 分段（UB 需 4 块等大区域）；从第二段起等待 `MTE3_MTE2` → workspace→UB（MTE2，两个相位平面各一块，1×1 无 bias 只搬一块）→ `MTE2_V` → Interleave → `V_MTE3` → `LoadUbToGm` → 非末段发布 `MTE3_MTE2` |

1×1 stride2 且无 bias 时第二相位平面全零，Interleave 走清零特例。`LoadUbToGm` 的目的偏移：`curNIdx*baseN*diHiWi + curDinIdx*hiWi + 行内换算(隔行空开) + rearrangeHIndex*wi`；首块在 `rearrangeWIndex==1 且 wi 奇数` 时末元素不搬（避免写出界），中间块 `blockCount=midHi、blockLen=wi`，循环模式 `loop1Size=curUseN、srcStride=(hwSize*2)*sizeof(DstT)、dstStride=diHiWi*sizeof(DstT)`。

AIC 在 rearrangeWIndex=0 时等待槽可用，在最后一个 W 相位完成后通知 AIV；AIV 交织写回后回告。1×1 无 bias 跳过无贡献 H 相位，通知与等待随同一状态机跳过。时序见 [C/V 协同](cv-collaboration.md)。

### 5.4 计算与重排的流水交接

输入侧的 L1/L0 ping-pong、L0C 槽轮换和输出重排槽分别管理。当前输出协议按一个 W 相位对交接；AIC 可以在 AIV 重排上一对时计算下一对，但覆盖同一暂存槽前必须等回告：

```text
+-------------------------------+                                       +-------------------------------+
| AIC / M + FIX                 |                                       | AIV / MTE2 + V + MTE3         |
+---------------+---------------+                                       +---------------+---------------+
                |                                                                       |
                |<----------------------- initial FLAG_FIXP_ID -------------------------|
                | compute W0; wait free; Fixpipe W0                                     |
                | compute W1; Fixpipe W1                                                |
                |-------------------------- FLAG_MTE2_VEC_ID -------------------------->|
                |                                                                       | load workspace if used
                | compute next W0                                                       | Interleave
                |                                                                       | DataCopyPad -> dedx
                |<--------------------------- FLAG_FIXP_ID -----------------------------|
                | Fixpipe next W0; compute/Fixpipe next W1                              |
                |-------------------------- FLAG_MTE2_VEC_ID -------------------------->|
                |                                                                       | next W pair
```

首令牌由 `CrossCoreSetHeadForMix` 发布，末次回告由 `CrossCoreWaitTailForMix` 消费。workspace 路径就绪通知为 FIX→MTE2，UB 直达为 FIX→V；回告均为 MTE3→FIX。图中每次通知代表两个 W 相位已备齐。1×1 无 bias 时，缺失 W 平面由交织特例补零；无贡献 H 相位不产生额外重排回合。

## 6 与公共 Kernel 流程的差异清单

| 维度 | 公共流程 | kernel-split |
|---|---|---|
| M 尺度 | `hi*wi` | `splitHiWi`（相位折算后） |
| K 窗口 | `hk*wk*C0` | `splitHkWkC0List[splitIndex]`（随相位） |
| 权重变换 | 三分支按格式/块形选择 | 固定 REVERSE_ONLY |
| 权重来源 | GM filter | 可选前置转置态：AIV 预转置进 workspace，AIC 改读 workspace 且 `kernelSplitStrideB_` 切 16 对齐口径（见 [Kernel 模板](kernel-template.md) 第 4.2.1 节） |
| 写回 | L0C→GM（Fixpipe 直达） | split-HW 经重排（UB/workspace 中转 + Interleave）；split-H 常规写回 + 行相位偏移 |
| 反向边界 | 原 backpropPad | 子核 pad（3.3 节）按相位重算 |
| 核型 | AIC-only 或按初始化协议 | 固定 MIX，AIV 承担重排/前置转置；与 C04、group enlarge 互斥（无 key 组合，见 [Tiling 绑定](tilingkey-templates.md) 第 1 节） |
| 任务总数 | batch*group*D*M*N | 追加 `kSCnt`（split-H） |

## 7 检查要点

- [ ] 相位起点（pad 取模）、子核分解表与 load3d 参数刷新逐相位核对；3×3 子核不等长时 kIter/curStep 随相位变化
- [ ] 非全载 B1 的紧凑子核与 cout 全载 B1 的原核布局分别寻址；子核采样和空间逆序各完成一次
- [ ] `hi%strideH≠0`（split-H）/`wi%strideW≠0`（split-HW）的尾部收缩与跳过任务不产生空写回
- [ ] 重排三段切分（head/mid/tail）覆盖全部有效 M；奇数 wi 的首块/右补边修正恰好一次
- [ ] workspace 分段循环的 `MTE3_MTE2/MTE2_V/V_MTE3` 事件成对；1×1 清零特例不读未初始化 UB
- [ ] UB 两份输入与 2L 输出互不覆盖；跨相位平面的 GM 跨度和分段 UB 跨度分别使用 baseUseN_ 与 curUseN
- [ ] B1 跨轮保留仅在身份不变的连续任务间发生；相位/组边界释放
- [ ] 子核 pad 的符号与上/下边界方向与公共裁剪口径一致（先裁剪真实 dedy，再表达剩余补边）

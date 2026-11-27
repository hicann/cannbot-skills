# kernel-split Tiling 模板（核窗口拆分）

kernel-split 对应 KernelSplit。把 H/W 核窗口按 stride 相位拆分为多个子核，每个子核按缩减后的核几何独立做 K 归约；拆分轴的子问题按单位步长执行，保留未拆分轴的原 stride；子核结果按输出相位写回。准入还包含 1×1 stride2 的专门条件，不能只按“大核”判断。共用计算见 [公共 Tiling](../../tiling.md)。

本模板是**改写问题几何**的模板：M/K 按子核重建，N 沿用 dedx 通道定义（第 3 节），并引入子核轮转与输出重排两个执行期概念；Kernel 侧落地见 [kernel-split 实现](../../../../../adapters/ascendc/input/kernel-split-implementation.md)，AIC/AIV 协同见 [C/V 协同](../../../../../adapters/ascendc/input/cv-collaboration.md)。

## 1 模式与登记

| 模式 | 拆分方式 | 空间改写 | 输出通路 |
|---|---|---|---|
| split-HW | H、W 两轴都按 stride 相位拆 | `hi×wi → ceilDiv(hi,s_h)×ceilDiv(wi,s_w)`，W 相位交织 | 输出重排（L0C→UB/GM 交织） |
| split-H | 仅 H 轴按 stride 相位拆 | `hi×wi → ceilDiv(hi,s_h)×wi`，W 维持原 stride | 常规写回，行起点带相位 |
| cout 全载变体 | split-HW/H 的 L1A、L1B 同时全载 dedy 通道 | 同所属模式 | 同所属模式 |

模板在公共选择器的**最高优先序**登记（kernel-split → small-shape → l1-resident → standard）：准入不通过时回落公共选择器，不改写标志扩大准入。cout 全载变体作为独立求解器注册在 kernel-split 之前，仅当共同准入已置 `kSCoutFullLoad` 且 `coutG<=256` 时接管；否则回落普通 kernel-split。

## 2 选择条件

当前浮点路径（FP16/BF16/FP32）要求原始 groups=1；FP32 下重排暂存按 4B 复核。通过公共规格和目标执行能力检查后，CheckKernelSplitEnable 可先比较 H/HW 的计算与搬运成本；比较模型未启用或不适用时，按 split-HW、split-H 的顺序尝试。以下条件只描述当前浮点路径。

| 公共准入 | 条件 |
|---|---|
| dilation | `dilationH=dilationW=1` |
| filter 格式 | NCDHW 或 NDHWC |
| pad 对称 | 上下相等且左右相等 |
| offsetX | 等于 0 |
| 任务量 | 非 `hi*wi<256 且 cout1G=1`（拆分收益不足） |

**split-HW 准入**（双输出算子不进入）：

| 条件 | 说明 |
|---|---|
| 格式 | out_backprop 与 y 均为 NCDHW（重排搬出按 32B 对齐要求） |
| stride | `strideW=strideH=2` |
| 核尺寸 | `kernelW=kernelH` 且 ∈ {1,2,3,4} |
| 退化排除 | `cinG=coutG=1 且 kernel=2×2` 不准入（子核 1×1 算力浪费） |
| 奇宽排除 | `wi%2≠0 且 wi>512` 不准入（scalar 过重） |
| 1×1 stride2 | 满足第 2.1 节专项准入 |
| 经验阈值 | 非 cout 全载时 `wi>=12`，且 `wi<=16` 时要求 `splitHiWi>256`；cout 全载时 `wi>2` |

**split-H 准入**：

| 条件 | 说明 |
|---|---|
| stride/核 | `strideH>=2 且 kernelH>=strideH`，`kernelH<=16` |
| 通道 | `cinG>=16 且 coutG>=16` |
| 宽度 | 通常 `wi>=32`；第 2.3 节模型判 H 更优时，可放宽该收益门槛，仍检查全部能力及容量条件 |

### 2.1 1×1 stride2 专项准入

`kernelH=kernelW=1` 且 stride=2 时，使用专项准入处理公共 L1 容量估算的偏差。按未拆分几何计算 `Mraw=hi*wi、Nalign=alignUp(cinG,16)、Kalign=alignUp(coutG,C0)`，再依次检查以下排除条件：

| 顺序 | 排除条件 | 说明 |
|---|---|---|
| 1 | `Mraw<=256` | 无基本块轮次收益 |
| 2 | `batch*di*ceilDiv(Mraw,256)*ceilDiv(Nalign,256)<=coreNum` | 分不满核 |
| 3 | `cinG<=32 或 coutG<=32` | 通道规模较小 |
| 4 | L0A、L0B 均可整块容纳 | 无需切块的小 shape |
| 5 | `dk=1、Mraw>Nalign、wi<=512`，收益条件满足，且 `Kalign*min(256,Nalign)*weightBytes*2<=L1Bytes` | 命中本函数的全载排除条件，N 上限为 256 |
| 6 | `hi>wi 或 cinG>coutG` | MTE2 bound 的经验判断 |

全部排除项均未命中时准入，并设置 `kSCoutFullLoad=0`。

### 2.2 L1 初筛与 cout 全载判定

split-HW 先执行容量初筛；split-H 的最小容量检查见本节末。当前浮点路径的 bias 预算为 `hasBias ? alignUp(cinG*4,64) : 0`，scale 预算为 0；扣除后得到 availableL1。取 `curBaseM=min(256,splitHiWi)`、`curBaseN=min(256,nValue)`，按 [驻留行数](#5-l1-驻留容量模型) 求 `curHo`，则：

| 量 | 公式 |
|---|---|
| A1 单通道分形 | `a1Size=dedyBytes*curHo*Wo*C0` |
| B1 单通道分形 | `b1Size=weightBytes*min(256,nValue)*splitHkWkC0` |
| A1 可驻留分形数 | `cout1A1=min(max(1,availableL1/2/a1Size),cout1G)` |
| B1 可驻留分形数 | `cout1B1=(availableL1-cout1A1*a1Size)/b1Size` |

**AB 全载条件**：`cout1B1>=cout1G*strideHW 且 cout1A1>=cout1G 且 curBaseN>=cinG 且 (dout=1 或 dk=1)`。令 bestBlockEnable 为：分离执行模式取 `bestBlockCnt>=coreNum 且 nValue<=128 且 cinG>=16`，融合执行模式取 `nValue<=64 且 cinG>=16`。只有 AB 全载、group=1、coutG<=256 且 **bestBlockEnable=false** 时才置 kSCoutFullLoad=1。bestBlockCnt 使用 `floor(splitHiWi/512)*di*batch`。split-HW 的 1×1 专项准入通过时强制 kSCoutFullLoad=0。分离执行模式还排除 `filter=NDHWC 且 (kSCoutFullLoad 或 cinG=1)`；融合执行模式按 dedy 通道分形下限过滤。该初筛属于 split-HW；split-H 单独检查最小 A1+B1 容量，不能直接套用本节全载判定。

### 2.3 H/HW 的成本比较

split-HW 的子核更短，但可能需要输出交织；split-H 保留全部核宽，增加 K 归约，却可直接写回。选择取决于计算和输出搬运哪一项占主导，不能仅按子核大小决定。

先对 `hk=strideH、wk=strideW` 的完美切分建立模型；在当前 split-HW 范围内，对应 2×2 核、两轴 stride=2。其他核形状需要另行推导 K 放大倍数，沿用原顺序。两种模式分别通过规格、能力与容量检查，模型只决定尝试顺序。

令 `batchDepth=N*Di`、`outVol=batchDepth*Hi*Wi*cinG`、`splitHiWiHW=ceilDiv(Hi,strideH)*ceilDiv(Wi,strideW)`。outVol 是有效输出元素数；nValue 是对齐后的组内 dedx 通道数。以下量均为估算周期：

| 管道 | split-HW 估计 | split-H 估计 |
|---|---|---|
| MMAD | `macCycles=outVol*coutG*Dk/macRate` | `macCycles*Wk` |
| dedy 的 GM→L1 | `aGmCycles=batchDepth*splitHiWiHW*coutG*Dk*dedyBytes/gmLoadRate` | `aGmCycles*Wk` |
| 窗口的 L1→L0A | `aL0Cycles=outVol*coutG*Dk*dedyBytes/(nValue*l0LoadRate)` | `aL0Cycles*Wk` |
| 输出通路 | `outCyclesHW`：按实际交织通路估计 | `outCyclesH=outVol*dstBytes/directOutRate` |

macRate 的单位为 MAC/周期；各 rate 为目标策略采用的有效字节/周期，dstBytes 为 dedx 元素字节数。它们是模型参数，结合目标能力和测量校准；不把经验带宽当成 platformInfo 的容量字段。

workspace 通路含 Fixpipe 写中转、MTE2 读中转、MTE3 写 dedx 三段。粗略模型可将三段有效带宽统一为 outRate，取 `outCyclesHW=3*outVol*dstBytes/outRate`；逐段模型则分别估计并求和。直接 UB 重排路径使用自身的 Fixpipe、UB 交织和 MTE3 成本，不能套用三趟 GM 搬运。两种估计都须采用同一时延口径；未建模的权重、同步与任务开销通过收益余量约束。

按最慢管道比较：`hwDuration=max(macCycles,aGmCycles,aL0Cycles,outCyclesHW)`，`hDuration=max(macCycles*Wk,aGmCycles*Wk,aL0Cycles*Wk,outCyclesH)`。仅当 hDuration 更小，且输出节省超过 minSavedCycles 时优先 H。minSavedCycles 是目标策略的固定开销余量；可按每核优选输出块的搬运成本设置，不是片上容量。

```text
if modelEnabled and ModelApplicable() and IsPreferKernelSplitHOverHW():
    if TryKernelSplitH(relaxSmallShapeGate=true): return true
if TryKernelSplitHW(): return true
return TryKernelSplitH(relaxSmallShapeGate=false)
```

每次尝试前从原始几何重建子核参数，失败候选的 kSCoutFullLoad、mValue/kValue 和 workspace 状态不得带入下一模式。最终输出通路与模型假设不同、或完整预算失败时，重新评价候选或回到原顺序。

## 3 子核几何与 M/N/K 重定义

### 3.1 子核与输出位置示例

**拆分依据是插零位置与核采样相位的对应关系。** 固定一个 dedx 位置，前放大窗口中只有部分位置来自真实 dedy，其余是 stride 引入的零。两路同时保留有效采样位置，就能缩短归约窗口。下图只画 H/W；`a/b/c/d` 是 dedy 值，`w00` 等是已经逆序的 weight_rot 窗口元素：

```text
fmap 窗口                           weight_rot 窗口
+---+---+---+---+                   +-----+-----+-----+-----+
| a | 0 | b | 0 |                   | w00 | w01 | w02 | w03 |
| 0 | 0 | 0 | 0 |                   | w10 | w11 | w12 | w13 |
| c | 0 | d | 0 |                   | w20 | w21 | w22 | w23 |
| 0 | 0 | 0 | 0 |                   | w30 | w31 | w32 | w33 |
+---+---+---+---+                   +-----+-----+-----+-----+
        |                                       |
        | 保留真实 dedy                         | 保留配对权重
        v                                       v
    +---+---+                             +-----+-----+
    | a | b |                             | w00 | w02 |
    | c | d |                             | w20 | w22 |
    +---+---+                             +-----+-----+

同一贡献: a*w00 + b*w02 + c*w20 + d*w22
```

split-HW 在 H/W 两轴上消除 stride 插零；split-H 只消除 H 插零，W 仍采用前放大的窗口。两者都保留边界补零、通道对齐补零及原有核深度归约。

**子核按原 weight 的采样相位编号。** 对某个 H 相位，取 `kh=p、p+strideH、p+2*strideH...`；split-HW 对 W 同样取样，split-H 保留完整 W 行。各子核的采样行数相差至多一行，多出的行分给前 `hk%strideH` 个核采样相位，W 同理。子核内部仍按 weight_rot 的方向逆序读取。

固定深度、batch 和通道，只看原 weight 的 H/W 位置。3×3、strideH=strideW=2：

```text
原 weight         按 (kh%2, kw%2) 拆出的子核
+---+---+---+     S00       S01       S10       S11
| a | b | c |     [a c]     [b]       [d f]     [e]
+---+---+---+     [g i]     [h]
| d | e | f |
+---+---+---+     输出相位网格（padUp=padLeft=0）
| g | h | i |     S00 S01 S00 S01 ...
+---+---+---+     S10 S11 S10 S11 ...
                  S00 S01 S00 S01 ...
```

输出坐标满足 `ih=oh*strideH-padUp+kh`，W 同理。因此输出相位是 `(kh-padUp)%strideH` 与 `(kw-padLeft)%strideW`；pad 为奇数时，上图的子核与输出相位对应关系随之交换。子核编号 splitIndex 与输出相位 rearrangeH/WIndex 不能混用。

例如两轴 pad 的余数同时从 0 变为 1，输出网格中的子核身份如下变化。格内的 S 编号沿用上图，格的位置才是 dedx 输出相位：

```text
padUp%2=0, padLeft%2=0          padUp%2=1, padLeft%2=1
+-----+-----+                   +-----+-----+
| S00 | S01 |                   | S11 | S10 |
+-----+-----+                   +-----+-----+
| S10 | S11 |                   | S01 | S00 |
+-----+-----+                   +-----+-----+
```

pad 相差一个完整 stride 时，相位对应关系相同，但 dedy 行带起点和子核补边仍会改变；滑窗参数须随实际 pad 重算。

split-H 只拆 kh，相同子核直接写入对应的隔行位置，W 仍按原 strideW 采样。split-HW 的两个 W 相位需交织后搬出。

### 3.2 几何与尺度

子核几何由 `SetParamForKernelSplit` 一次性派生（`C0` 随 dtype：FP16/BF16=16、FP32=8；`blockSize=C0`）：

| 字段 | split-HW | split-H |
|---|---|---|
| `curWi` | `wi`，`wi%strideW≠0` 时 +1（对齐相位） | `wi` |
| `curStrideW` | 1 | `strideW` |
| `strideHW` | `strideH*strideW` | `strideH` |
| `maxSplitKh` | `ceilDiv(hk,strideH)` | `ceilDiv(hk,strideH)` |
| `maxSplitKw` | `ceilDiv(wk,strideW)` | `wk` |
| `splitHiWi` | `ceilDiv(hi,strideH)*ceilDiv(wi,strideW)` | `ceilDiv(hi,strideH)*wi` |
| `splitHkWkC0` | `maxSplitKh*maxSplitKw*C0` | 同左 |
| `subSplitHkWkC0` | `(hk/strideH)*(wk/strideW)*C0` | `(hk/strideH)*wk*C0` |

`splitHStartIndex=padUp%strideH`、`splitWStartIndex=padLeft%strideW` 表示与输出 H/W 相位 0 配对的核采样相位；split-HW 的首个子核号为二者组合。split-H 从子核号反求输出行相位时使用补数偏移，见 [kernel-split 实现](../../../../../adapters/ascendc/input/kernel-split-implementation.md#2-任务解码与基本块循环)。

进入公共求解前重定义三个尺度（与 [公共 Tiling](../../tiling.md#3-输入与输出) 的差异）：

| 方向 | kernel-split 定义 | 公共定义 |
|---|---|---|
| M | `alignUp(splitHiWi,m0)` | `alignUp(hi*wi,m0)` |
| N | `alignUp(cinG,n0)`（不变） | 同左 |
| K | `splitHkWkC0*cout1G` | `alignUp(coutG,C0)*hk*wk` |
| 核深度 | `dk` 独立循环，不占 L0 的 K | 同左 |

核深度贡献在 K 循环之外完整归约的公共约定不变；变化的是 K 内的核高宽窗口长度（子核窗口取代原窗口）。`lenHkWkC0=maxSplitKh*maxSplitKw*C0` 记录对齐用最大子核窗口，`subSplitHkWkC0` 是按向下取整得到的最小窗口整除辅助量（1×1 stride2 时可为 0），各相位的实际窗口另由 splitHkWkC0List 记录，二者的整除关系参与 baseK 约束（第 8 节）。

## 4 基本块起点与 L0C 份数

`InitBaseMNK` 初始 `al0Pbuffer=bl0Pbuffer=2、cl0Pbuffer=1`：

| 条件 | baseM/baseN/baseK 起点 |
|---|---|
| `dk*maxSplitKh*maxSplitKw>1`（格式转换 bound） | `(512,128, 64/weightBytes)` |
| 其余（带宽优先） | `(256,256, 128/weightBytes)` |

耦合架构（CV 合体）关闭 L0 pingpong 掩盖 scalar。随后执行公共 `AdjustBaseMNK()`。

`UpdateL0CBufferMode` 默认关闭 L0C 双 buffer；`baseM*baseN*accBytes*2<L0CBytes` 时开启；仅 split-HW 接受等号，split-H 等号时保持单份。

## 5 L1 驻留容量模型

预算与容量回调使用 dedy 通道分形数 cout1A1/cout1B1，分别描述 L1A/L1B 的驻留范围。传给 Kernel 的参数仍为公共 stepKa/stepKb（stepKa=ceilDiv(cout1A1*splitHkWkC0,baseK)，stepKb 同理）。

驻留行数由 `CalFmapHForKernelSplit` 给出。令 `splitW=splitWi`（split-HW 已按相位折算，split-H 为 `wi`）：

| 量 | 公式 |
|---|---|
| 覆盖行数 | `mL1%splitW=0 → ceilDiv(mL1,splitW)`；`mL1>splitW → mL1/splitW+2`；否则 2 |
| 有效核高 | `khDilation=(ceilDiv(hk,strideH)-1)*dilationH+1` |
| 驻留行数 | `min(覆盖行数-1+khDilation, ho)` |

`mL1` 取 `min(baseM,splitHiWi)`；末尾的 ho 截断保证驻留不超过真实 dedy 高度。`IsL1ParamsValid` 在此模型上检查：

| 量 | 公式 |
|---|---|
| 驻留通道数 | `coutNum=max(stepKa*baseK/(maxSplitKh*maxSplitKw),1)` |
| A1 像素量 | `驻留行数(baseM)*Wo*curStrideW*coutNum` |
| 合法性 | `A1字节*al1Pbuffer + B1字节 + bias(64B对齐) + scale <= L1Bytes` |

B1 字节沿用公共 `CalcBL1Size`，并先通过公共 `IsHkWkAligned`（K 窗口配对）检查。

**step 求解特例**：split-HW 且 kernelH=3 时，四个子核为 2×2/2×1/1×2/1×1。令 `minCoutA1=baseK/C0`，按候选分别处理：

| 候选 | 调整 |
|---|---|
| 等分容量 | 仅在 cout1A1>minCoutA1 时向下对齐 cout1A1，再按公共规则调整两路分形数和 step 整除关系；两路 step 不要求相等 |
| 阶梯搜索 | 公共搜索后恢复 cout1A1；仅在 cout1A1>minCoutA1 时向下对齐，并重算 stepKa、令 stepKb=stepKa；否则保留原结果 |

调整后仍检查窗口配对和容量，不能把小于 minCoutA1 的分形数对齐成零。

## 6 容量回调

`LegalProtection` 只在 kernel-split 自身模型内收缩，不切换模板，两级顺序：

| 级 | 动作 | 说明 |
|---|---|---|
| 1. 减基本块 | `ShrinkBaseMN`：按两路预算及当前块长选择收缩 baseM（下限 `max(curWi,m0)`）/baseN（下限 n0），直到 `baseA1Size*驻留行数 + baseB1Size*baseN <= L1Bytes` | 先收缩占 L1 更大的一路；成功后重算 stepK 与 L0C 份数并复查 |
| 2. 减驻留通道 | `ShrinkCoutA1B1`：从 `L1/2` 等分起步，优先保大的一路，交替递减 `cout1A1/cout1B1` 至放得下；递减到 1 后仍按完整容量检查决定是否合法 | 先使用单份 L1；`UpdateBaseKParams` 对分形数大于 1 的一路将其减半并重开 DB，再对齐两路分形数、按 L0 容量重设 baseK，执行整除收缩（第 8 节） |

两级回调后仍按完整容量与搬运范围检查；未通过时返回失败。

## 7 任务划分（SetSingleCoreInfo）

`singleCoreDin=1、singleCoreCout=coutG` 与公共口径一致。

**split-H** 复用公共任务恢复，但空间总量换成 `splitHiWi`，且子核循环数 `kSCnt=strideH` 计入总任务数（`totalTasks=batch*group*depth分段*空间分段*通道分段*strideH`）。`hi%strideH≠0` 时尾部子核的 M 总量更小（`kSTailM_`），任务解码按子核号切换尾块口径。

**split-HW** 不乘 strideH（H/W 相位在核内交织轮转），`totalCnt=batch*realG*ceilDiv(di,singleCoreDin)*ceilDiv(splitHiWi,baseM)*ceilDiv(nValue,baseN)`。通道划分满足任一条件时重取 `baseN=alignUp(nValue/nCnt,n0)`：任务数不超过核数、`nCnt` 与 coreNum 互有整除关系、或 `baseN` 超阈（kernelDHW=1 时 256，否则 128）。耦合架构任务数不足核数一半时把 baseM 收到 128。空间任务长度按 `splitWi=ceilDiv(wi,strideW)` 对齐取整行，`singleCoreM<baseM` 时收缩 `baseM=alignUp(singleCoreM,m0)`。

两个模式的相位归属不同，决定了任务数与写回方式：

```text
split-H
  任务: batch/group/D/M/N + kSCoreIdx
    -> 固定 H 核相位 -> 全部 K/kd -> kernelIdx 的行

split-HW
  任务: batch/group/D/M/N
    -> 输出相位 (0,0) -> (0,1) -> (1,0) -> (1,1)
    -> 每个输出 H 相位一对 W 平面 -> 交织 -> dedx
```

图中 split-HW 按输出相位排序；实际 splitIndex 由 pad 映射。每个有效 dedx 元素归属唯一相位，不在多个子核之间再做输出累加。

## 8 子核对齐、workspace 与 vector 资源

**baseK 整除约束**：求解完成后，若 `lenHkWkC0≠subSplitHkWkC0` 且 `baseK≠C0`，说明子核窗口与公共窗口不对齐，需要收缩 baseK 使 `min(cout1A1,cout1B1)*splitHkWkC0` 与 `min(cout1A1,cout1B1)*subSplitHkWkC0` 都能被 baseK 整除（从当前 baseK 按 C0 递减搜索），再按整除结果重算 stepKa/stepKb——保证一次 L1 装载覆盖整数个（子）核窗口。

**workspace 判定**（split-HW 输出重排）：输出的 `dstBytes` 随 dtype（FP16/BF16=2、FP32=4），L0C 的 `accBytes=4`。源码 dtypeByteL0c_ 表示输出元素字节数，不能按名字误认成累加器大小。令 `evenWi=ceilDiv(curWi,strideW)`、`alignWi=alignUp(evenWi,16)`，暂存行数为 `ceilDiv(baseM,evenWi)`，原 wi 为奇数时再加一行：

| 用途 | 字节预算或条件 |
|---|---|
| 单核重排暂存 | `curL0cSize=暂存行数*alignWi*dstBytes*baseN*4` |
| 启用 GM 中转 | `curL0cSize>UBBytes 或 curWi%16!=0`；置 kSUseWorkSpace=1 |
| 重排 workspace 预留 | `curL0cSize*可用AIC核数`；实际启动核号不能超出预留槽数 |
| 关闭 L0C 双份 | `暂存行数*alignWi*baseN*accBytes*2>L0CBytes` |

**vector 判定**：enableVecTransFlag 在 filter=NCDHW、非 cout 全载、`16*dk*hk*wk*weightBytes*2<=UBBytes` 且核体积>1 时启用；使用 [指令映射 §3.5](../../../../../adapters/ascendc/input/instruction-mapping.md) 的转置路径，workspace 前段追加转置权重字节数。kSCoutFullLoad 单独置位不消费 workspace。

**coreNum 发布**：只有 enableVecTransFlag=true 时，才将 cube 任务数与 vector 所需 AIC 数 `ceilDiv(cout*ceilDiv(cin,16),CVRatio)` 取大，再限制到可用 AIC 数。当前浮点 kernel-split 准入 groups=1，因此源码使用的 cinG/coutG 与这里的 cin/cout 等值；统一总通道口径以匹配转置任务和 workspace 布局。CVRatio 动态获取；发布前按目标平台分支执行 CheckVectorCoreNum，见 [C/V 协同](../../../../../adapters/ascendc/input/cv-collaboration.md)。

## 9 cout 全载变体（KernelSplitFullLoad）

共享 kernel-split 的全部准入与子核几何，接管条件：共同准入已置 `kSCoutFullLoad`（第 2.2 节）且 `coutG<=256`；`coutG>256` 时基本块被过度压缩，回落普通 kernel-split 并清除该标志。求解差异：

| 项 | 差异 |
|---|---|
| 基本块 | 固定 `(256,256)`；`baseK=max(128/weightBytes, subSplitHkWkC0)`，覆盖版 AdjustBaseMNK 从最优块向下找满足 `maxBaseK%lenHkWkC0=0` 的最大 K，且 baseK 不低于子核窗口 |
| L1 策略 | L1A、L1B 一次装载全部 dedy/weight 通道（`cout1A1=cout1B1=cout1G`），初始化使用公共 InitL1Params；CalStepK 最终令 bl1Pbuffer=1，kSCoutFullLoad=1 时 al1Pbuffer=1；stepK 起点为 1，走公共 `LadderMatchStepKWithFullLoad` |
| 其余 | 任务划分、容量回调、workspace/vector 判定复用 kernel-split |

## 10 Host 求解顺序

```text
GetShapeAttrsInfo:
    GetPublicShapeAttrsInfo()            # 公共规格
    enableSplitKernelFlag = CheckKernelSplitEnable()
        # 公共准入 -> 可选 H/HW 成本比较 -> 按第 2.3 节尝试
        # 期间完成 L1 初筛与 kSCoutFullLoad 判定，并重定义 mValue/kValue
DoLibApiTiling:
    InitBaseMNK(l0)
    InitL1Params(l1,l0)
    SetSingleCoreInfo(core,l0)
    CalStepK(l1,l0)                      # 公共分块驻留
    if not IsL1ParamsValid(l1,l0):
        LegalProtection(l1,l0)           # 减基本块 -> 减驻留通道（含 baseK 整除收缩）
        if not IsL1ParamsValid(l1,l0): return false
        SetSingleCoreInfo(core,l0)
    if lenHkWkC0 != subSplitHkWkC0 and baseK != C0:
        ShrinkBaseKForKernelSplit(l1,l0) # 子核窗口整除约束
    UpdateWorkSpaceSize(l0)              # workspace 与 L0C 份数
    CheckVectorCoreNum()
    SetTilingData(core,l1,l0)            # 含 kSCoutFullLoad/kSUseWorkSpace 与 coreNum 取大
```

`SetTilingCondition` 固定 loadB2Condition=仅逆序（子核窗口由相位重建，权重只逆序不转置）；key 与字段编码见 [Tiling 绑定](../../../../../adapters/ascendc/input/tilingkey-templates.md)。

## 11 检查要点

两路在同一个子核 K 坐标上配对。图中的驻留范围和 buffer 份数由最终 Tiling 决定；子核紧凑布局可在搬运中形成，全载权重也可在取出时选择子核：

```text
dedy   -> L1A -> 子核窗口 -> L0A ---+
                                    +-> Mmad -> L0C
weight -> L1B -> 子核权重 -> L0B ---+           |
                                                | 完整 K/kd 归约
                                                v
                            +-------------------+-------------------+
                            |                                       |
                            v                                       v
                         split-H                                split-HW
                     按相位错行写回                          保留两个 W 相位
                            |                                       |
                            |                                     交织
                            |                                       |
                            +-------------------+-------------------+
                                                v
                                              dedx
```

- [ ] 子核有效贡献与原始连接逐项一致；仅移除 stride 插零项，边界与通道尾补零仍有明确责任
- [ ] pad 改变后，核采样相位、输出相位、dedy 起点和子核补边同步更新
- [ ] 相位起点由 pad 取模得到；子核窗口、sub pad 与 load3d 参数随 splitIndex 刷新
- [ ] `lenHkWkC0/subSplitHkWkC0/k0` 三种窗口长度分别记录，baseK 整除约束在最终块上复核
- [ ] split-H 的 `hi%strideH≠0` 尾部子核任务数收缩（kSTailM_），不产生空任务
- [ ] workspace 启用条件、容量与 L0C 份数联动正确；`kSUseWorkSpace` 与 AIV 通路一致
- [ ] H/HW 模型开启、关闭及不适用均有确定顺序；H 优先失败后重建 HW 参数，模型采用最终输出通路
- [ ] `kSCoutFullLoad` 的置位/清除路径完整（1×1 强制清除、coutG>256 回落清除）
- [ ] 模板失败回落公共选择器，不改写准入标志；与 small-shape/l1-resident 同时满足时本模板优先

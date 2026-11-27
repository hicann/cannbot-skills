# 卷积反向 input 分支：分支记号

> **层级**：通用知识层（generic/input/），语言中立。
>
> **定位**：本文件收录 **input 分支（Conv3DBackpropInput / DX、Conv3DTranspose）专属记号**：核心记号速查、张量与属性补充、M/N/K 与 GEMM 映射、坐标映射、缓存内容与阶段名称、切分与执行坐标、模板与状态、bias 词汇。公共术语（分形粒度、维度、卷积属性、缓存层级与流水、格式、数学工具）见 [../language.md](../language.md)，规格归一见 [../preprocessing.md](../preprocessing.md)。
>
> 与 filter 分支的 M/N/K 等**同名不同义**（filter：M=dedy 通道、N=x 通道×核窗口、K=输出平面），见 [../filter/language.md](../filter/language.md)，两侧文档不得混用记号。

---

## 0 核心记号速查（先背这张表）

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **mValue / nValue / kValue** | — | `alignUp(hi*wi,m0)` / `alignUp(cinG,n0)` / `alignUp(coutG,C0)*hk*wk`，Host 选块的对齐长度 |
| **baseM / baseN / baseK** | — | 一次计算的空间、dedx 通道、K 容量（L0 基本块） |
| **singleCoreM / Cin / Din** | — | 一个任务的空间、dedx 通道、深度范围；一个核可处理多个任务 |
| **stepKa / stepKb** | — | 一份 L1A/L1B 驻留供给的 K 计算块数；完整 K 驻留仍按 stepKb*baseK 表达 |
| **loadB2Condition** | — | B2 装载分支：1=转置加逆序，2=仅逆序，0=不转置不逆序；FP32 固定为 2，单位核的空间逆序为恒等 |
| **K 遍历顺序** | — | 普通布局为 `Cout1 → r_h → r_w → Cout0`；C04 使用四通道窗口，kernel-split 使用子核窗口，两路须使用同一排列 |

> 分形粒度 M0/N0/C0/K0 与公共记号见 [../language.md](../language.md) §0。

## 1 张量与属性的分支补充

本分支的张量绑定（在公共 §1 的 X/Y/weight/dedy 之外）：

| 张量 | 别名 / 接口字段 | 语义 |
|------|---------|------|
| **dedx** | y | 正向输入梯度（当前输出） |
| **input_size** | 输入张量或规格属性 | dedx 目标形状，按 data_format 解读 |
| **bias** | 可选输入 | 按 dedx 总通道排列的 FP32 向量（转置卷积） |

**转置卷积的对象绑定**：Conv3DTranspose 的 x/filter/y 分别绑定 dedy/weight/dedx。**Transpose 输入通道对应 Cout，输出通道对应 Cin**——这是复用计算层的命名约定；weight 第一轴保持 Cout，第二轴为 CinG。入口只绑定对象，通道转置与空间逆序由公共 weight 变换完成。

本分支的专属属性与坐标记号：

| 记号 | 含义 |
|------|------|
| **output_padding** | 记号 `outputpadD/outputpadH/outputpadW`（数学简写 `q_d/q_h/q_w`）；仅转置卷积，N/C 分量为 0；`q>=0` 且 `(q<stride or q<dilation)` |
| **enable_hf32 / hf32Flag** | 接口精度属性 / Host–Kernel 运行字段；本分支的 FP32 分别取 false / 0，使用普通 FP32 MMAD |
| **b_a^-、b_a^+** | 反向有符号边界，`b_a^-=K_a^eff-1-p_a^-`、`b_a^+=K_a^eff-1-p_a^++ρ_a`；正值补零、负值裁剪（backpropPad* 与裁剪起点共同表达） |
| **r_d/r_h/r_w** | 逆序核坐标，`r_a=K_a-1-k_a`，反向窗口配对 |

**group enlarge 的发布口径**：数学上的 G/CinG/CoutG 不变。Host 运行信息及 TilingData 中的小写 `cinG/coutG` 已是合并后的 `enlarge*CinG/enlarge*CoutG`，`group=real_g`，`oriGroup=G`。原始组内通道应由 `cin/oriGroup、cout/oriGroup` 恢复，不能再把发布后的 cinG/coutG 乘一次 enlarge。前置权重转置统一称 VecTranspose；Host 开关名为 `enableVecTransFlag`，Kernel 字段为 `enableVecTrans`，不引入 enableVecTranspose 新字段。

## 2 M/N/K 三元组（GEMM 映射）

dedy 经插零、边界映射与窗口展开（img2col）形成 L0A，weight_rot 形成 L0B，矩阵乘 `L0C[M,N] = L0A[M,K] × L0B[K,N]` 得到 dedx：

```
dedy (N,Cout,Dout,Ho,Wo)            weight (Cout,CinG,Dk,Hk,Wk)                     dedx (N,Cin,Di,Hi,Wi)
    |                                   |                                               ^
    | 插零 + 反向边界（fmap）           | rot180 + 通道角色交换（weight_rot）           |
    v                                   v                                               |
A 矩阵 [M, K]                   ×   B 矩阵 [K, N]                               =   Y 矩阵 [M, N]
行 = dedx 空间位置 m                列 = 组内 dedx 通道 ci                          行×列 = dedx 输出元素
列 = (co1,r_h,r_w,co0) 展平         行 = 同一 (co1,r_h,r_w,co0)
```

| 记号 | 定义 | 说明 |
|------|------|------|
| **M** | `m = h_i*Wi+w_i`，展平 dedx 高宽 | 有效长度 `hi*wi`，选块长度 `mValue=alignUp(hi*wi,m0)` |
| **N 方向** | 组内 dedx 通道 ci | 与 batch N 分开；选块长度 `nValue=alignUp(cinG,n0)` |
| **K** | `kLinear = ((co1*Hk+r_h)*Wk+r_w)*C0+co0` | 有效归约项 `coutG*hk*wk`；选块长度 `kValue=alignUp(coutG,C0)*hk*wk` |
| **核深度** | kd | 不混入 K，独立于 K 循环完整归约 |

> filter 分支的 M/N/K 与此**同名不同义**（M=dedy 通道、N=x 通道×核窗口、K=输出平面），见 [../filter/language.md](../filter/language.md)。

派生对齐量（L0 物理轴 (K1,M1,M0,K0)/(K1,N1,N0,K0)/(N1,M1,M0,N0) 中的块数轴）：

| 记号 | 定义 | 含义 |
|------|------|------|
| M1 / N1 / K1 | `CeilDiv(选块 M/N/K, M0/N0/K0)` | 各矩阵方向的分形块数 |
| Cout1 / Cout0 | `ceilDiv(coutG,C0)` / `C0` | dedy 侧 K 通道块数 / 块内长度 |
| Cin1 / Cin0 | `ceilDiv(cinG,N0)` / `N0` | L0B、L0C 及 VecTranspose 的 dedx 通道块数 / 块内长度 |

FP16/BF16 的 C0=K0=16，FP32 的 C0=K0=8；M0/N0 恒为 16。FP32 的 K 通道按 8 对齐，Cin 仍按 16 分块。参数见 [类型表](../../adapters/ascendc/buffer-and-types.md#2-类型与资源对象)。

固定 `(n,g,curDinIdx_)` 后：L0A 的一行是一个 dedx 空间位置所需的 fmap 窗口；L0B 的一列是一个 dedx 通道所需的 weight_rot；L0A 的 K 列与 L0B 的 K 行必须对应同一个 dedy 通道和逆序核位置。补齐通道参与计算时贡献为零。

K 长度按用途区分，核深度不计入下表：

| 记号 / 用途 | 普通布局 | C04 |
|---|---|---|
| Host `kValue`，选块与预算 | `alignUp(coutG,C0)*hk*wk` | `alignUp(4*hk*wk,C0)` |
| Kernel `tmpSingleCoreK`，当前窗口的归约长度 | `alignUp(singleCoreCout,C0)*singleShapeHWk_` | `4*singleShapeHWk_` |
| `baseUseK_` / Load3D.kExtension | `min(baseK,tmpSingleCoreK-kStart)` | 同左 |
| MMAD.k | `baseUseK_` | 尾 K 块取 `alignUp(baseUseK_,c0)` |

`singleShapeHWk_` 是当前核高宽采样点数；kernel-split 使用当前子核窗口。槽容量按选块与驻留预留长度计算，运行循环按 `tmpSingleCoreK` 计算。C04 的指令补齐位置由两路准备保证零贡献。

## 3 坐标映射（dedx 位置 → dedy 采样）

直接沿正向连接求梯度：

| 项 | 公式 | 说明 |
|----|------|------|
| 直接求和 | `dedx_g[n,ci,i] = Σ_co Σ_k weight_g[co,ci,k] · dedy_g[n,co,(i+p^- - k⊙δ)/s]` | 仅累加分子非负、能被 stride 整除且 dedy 坐标在界内的项 |
| 反向边界 | `b_a^- = K_a^eff-1-p_a^-`，`b_a^+ = K_a^eff-1-p_a^++ρ_a` | `K_a^eff=(K_a-1)*δ_a+1`；ρ_a 为尾部余数；正值补零、负值裁剪 |
| 等价窗口模型 | `fmap = dedy_expand 按反向边界补零/裁剪后的视图`，与 `weight_rot` 单位步长互相关 | `dedy_expand` 为按 stride 插零视图 |

窗口模型的采样几何（与正向镜像：窗口步进与采样间隔互换角色）：

```
dedx 元素 dedx(n,ci,i)            fmap 感受野（Hk×Wk 窗口，单位步长）
    |                                 |
    | 对每个 (co, r_h, r_w)：         v
    | fmap(n,co, i+r⊙δ)           +--------+--------+-----+
    | × weight_rot[ci,co,r]       | rh=0   | rh=1   | ... |
    v 累加到 dedx                 +--------+--------+-----+
                                  | 采样点 | 采样点 | ... |
                                  +--------+--------+-----+
                                  采样点间隔 = dilation（保留正向原值）
                                  窗口间步进 = 1（正向 stride 已由插零表达）
```

这条映射是 img2col、反向边界、golden 生成（[verification.md](verification.md) 的坐标归一与基准实现）的共同数学依据；完整模型推导见 [scene.md](scene.md)。

**视图记号**：

| 名称 | 含义 | 实现落点 |
|------|------|---------|
| **dedy_expand** | dedy 按 stride 插零的视图 | L1A 中保存当前驻留片段 |
| **fmap** | dedy_expand 按反向边界补零/裁剪后的视图 | Load3D 取窗口 |
| **weight_rot** | 三个空间轴逆序、组内通道角色交换后的 weight 视图 | L0B 的计算语义 |

这三个都是数学视图，不要求分别分配中间张量：L1A 保存驻留内容，Load3D 结合 padList 形成 fmap 窗口。

## 4 缓存内容与阶段名称（input 分支）

| 层级 | 存储内容 | 阶段名称 | SDK 位置 |
|------|---------|---------|---------|
| **L1A** | L1 的 dedy 驻留片段，供多个窗口复用 | LoadToA1 | `TPosition::A1` |
| **L1B** | L1 的原核顺序权重父段 | LoadToB1 | `TPosition::B1` |
| **L0A** | fmap 窗口，M×K，NZ | LoadToA2 / Load3D（img2col） | `TPosition::A2` |
| **L0B** | 转置逆序后的 weight，K×N，ZN | LoadToB2 / Load2D | `TPosition::B2` |
| **L0C** | M×N 部分和，NZ，FP32 | MmadLocal / MMAD | `TPosition::CO1` |
| **dedx** | GM 中的有效输出 | LoadL0c2Gm / Fixpipe | GM |

层级与流水的公共语义（MTE2/MTE1/M/F 分工、槽与 pbuffer、单位记账、容量来源）见 [../language.md](../language.md) §4。

## 5 切分与执行坐标

### 5.0 记号层次图

三个尺度各管一层：**任务级**决定核间分配，**L1 父段级**决定驻留复用，**L0 基本块级**决定一次 MMAD：

```
切分对象：
+-------------------------+   +-------------------------+   +-----------------------------+
| 任务级（核间分配）      |   | L1 父段级（驻留复用）   |   | L0 基本块级（一次 MMAD）    |
| singleCoreM/Cin/Din     |   | stepKa / stepKb         |   | baseM / baseN / baseK       |
| 一个输出区域的完整归约  +-->+ 一份 L1A/L1B 驻留覆盖的 +-->+ 一次乘加的空间×通道×K；     |
| batch×group×D段×M段     |   | K 计算块数；kLocalA/B、 |   | baseUse*/mCompute 分别是    |
| × Cin 段（任务解码见    |   | curStepKa/b 定位父段内  |   | 本轮有效长度与片上对齐跨度  |
| compute-flow §6）       |   | K 位置与换驻            |   | （尾块按剩余量收缩）        |
+-------------------------+   +-------------------------+   +-----------------------------+
```

### 5.1 单核规模与多核维度

| 记号 | 含义 |
|------|------|
| baseM/baseN/baseK | 一次计算的空间、dedx 通道、K 容量（L0 基本块） |
| singleCoreM/Cin/Din | 一个任务的空间、dedx 通道、深度范围 |
| singleCoreCout | 当前任务归约的 dedy 通道数（coutG），单位为通道 |
| taskId/coreId/coreNum | 输出任务号、核号、实际启动计算核数 |
| depthStart/spaceStart/channelStart | 任务起点，按 singleCore* 恢复 |
| curDinIdx_、mStart、cinStart | 当前输出块的全局 dedx 深度、平面空间起点、组内通道起点 |
| baseUseM_/baseUseN_/baseUseK_ | 当前块有效空间点数、dedx 通道数、K 长度；K 长度含通道补齐 |
| mCompute/nCompute | `alignUp(baseUseM_,m0)`、`alignUp(baseUseN_,n0)`，片上布局跨度 |
| isLastMBlock | 本任务末个 M 块，`mStart+baseUseM_=spaceStart+spaceUse` |

`mStart=spaceStart+curMIdx_*baseM`，`cinStart=channelStart+curNIdx_*baseN`，其中 curMIdx_/curNIdx_ 为任务内块号；恢复 GM 地址时只加一次任务起点。

### 5.2 迭代变量与父段状态

| 记号 | 含义 |
|------|------|
| kIter/kIterIdx | `ceilDiv(tmpSingleCoreK,baseK)`、当前 K 轮号 |
| kStart | 当前完整 K 起点 |
| kLinear/kLocal | 完整 K 元素坐标、当前 L0 块内坐标 |
| segmentStart/segmentUse | 当前 L1 父段完整 K 起点、实际长度 |
| kLocalA/kLocalB | 当前 K 起点减去各自父段起点 |
| curStepKa/curStepKb | 当前父段实际覆盖的 K 轮数，末段按剩余量截断 |
| aCoutStart/bCoutStart、bCinStart | L1A/L1B 各自的 dedy 通道起点，以及 L1B 的 dedx 通道起点 |
| curCoutSize/curCinSize | 本次搬入的有效 Cout/Cin 通道数 |
| curCout1Cout0Size/curCin1Cin0Size | 转置父布局的 Cout、仅逆序父布局的 Cin 分别向 16 对齐后的长度；FP32 使用后者 |
| blockBaseN_ | `ceilDiv(baseUseN_,16)`，当前 dedx 通道分形数 |
| curLoadKbl1_ | 当前 L1B 父段的实际 K 跨度 |
| curHoSize/woExpand | L1A 当前实际高宽，另记 dedy_expand 中原点 |
| channelSize_ | L1A 父片段描述的通道长度；普通路径为 `min(curStepKa*baseK/(hk*wk),alignUp(singleCoreCout,C0))`，C04 为 4 |

`C0` 是通道分形粒度，`channelSize_` 是父片段通道长度，`DataCopy.dValue` 是本次实际搬入的通道数，三者分别使用。

### 5.3 尾块与有效范围

| 记号 | 含义 |
|------|------|
| 尾块 | 任务/块尾部按剩余量收缩：`baseUse* = min(base*, 总长-起点)` |
| 有效 M 与对齐 M | 分别传递：Load3D.mExtension 和 Fixpipe.mSize 使用 baseUseM_；MmadParams.m 在任务末 M 块使用 mCompute |

## 6 模板、分支与状态

| 记号 | 含义 |
|------|------|
| standard | 经验选块和常规任务分配；求解逻辑对应 InnerProduct |
| small-shape | 小任务量 M/N 均衡；对应 SmallShape |
| l1-resident | 完整 K 权重驻留；对应 FullLoad |
| kernel-split | 核窗口按 stride 相位拆分为子核族；对应 KernelSplit（split-H/split-HW 及 cout 全载变体） |
| templateId | 四个 Tiling 模板之一，只用于 Host 选择及检查 |
| loadB2Condition | 1=转置加逆序，2=仅逆序，0=不转置不逆序（C04 的 B2 装载阶段）；普通单位核复用 1，kernel-split 固定为 2 |
| kernelSplitMode | 0=不拆分；1=split-HW（H/W 相位交织，输出重排）；2=split-H（H 相位、行起点带相位） |
| splitIndex/splitHkWkC0List | kernel-split 当前子核相位号与各相位的子核窗口长度表 |
| enlarge、real_g | group 放大倍数与放大组数：enlarge>1 时 real_g=ceilDiv(groups,enlarge)，N/K 按放大组口径（cinOpt/coutOpt）重算 |
| groupConvMode/loadB1Condition | groupConvMode 是 Host 分组状态，0=原始分组、1=group enlarge，Kernel 按 enlarge 判定；loadB1Condition 是编译参数，0=GM→L1，1=VEC_TO_L1_C04 |
| tilingKey | 适配层按有序模板声明编码的编译分支；普通三模板可命中同一个 key，kernel-split 另编码 |
| enableFullLoad | 公共流程的跨任务权重驻留策略；实际保留仍检查完整 K 与权重身份 |
| isB1FullLoadFlag | 当前 L1B 是否满足完整驻留范围及份数条件 |
| curEnableFullLoad | Kernel 当前跨任务保留权限，失效后按实现规则关闭 |
| needComputeFlag_ | 行带与 dedy_expand 有交集且存在有效 dedy 深度时进入计算；由 UpdateL1ComputeInfo 更新 |
| firstContribution | 当前输出块是否尚无实际 MMAD；跳过深度不改变它 |
| initOutputFlag | 1=需要输出初始化，0=主计算完整写回；执行方式在适配层确定 |

普通布局的 K 顺序为 `Cout1 → r_h → r_w → Cout0`；C04 与 kernel-split 使用各自窗口排列。原核位置与逆序位置成对出现，空间反向关系每轴表达一次。现有示例的私有分发枚举见其 README，与本节编译参数分开解释。

## 7 bias 词汇（转置卷积可选）

DX 入口使用无 bias 配置；Transpose 的输出形状由其前置处理确定后，进入同一计算层。

| 记号 | 含义 |
|------|------|
| bias、biasBytes | 按 dedx 总通道排列的 FP32 向量；元素为 4B |
| hasBiasFlag_ / hasBias | Host 是否存在非空 bias / Device 编译分支，必须一致 |
| isBiasFullLoad | bias 按任务通道段整体驻留，或按当前 N 子块准备；与 enableFullLoad 分开 |
| offsetBias_ / preOffsetBias_ | 当前/前次 bias 的全局通道起点，用于检查复用 |
| biasL1Que_ / biasBTQue_ | bias 的 L1 / BT 队列；SDK 位置为 A1 / C2，各一份 |
| computeBiasOnce_ | 当前输出块是否已通过 MMAD 引入 bias，每个输出块重置 |
| FullLoadBias / LoadBiasToBT | 整个任务通道段 / 当前 N 子块的 bias 搬入 |
| ComputeBTBias | 全量模式沿用已持有 BT；分块模式准备当前 bias |
| ComputeForBias / LoadL0Zero | 无有效乘积时，用全补零操作数与 BT 产生 bias 输出 |
| FreeBiasTensor | 归还仍持有的全量 bias L1/BT，随后由 End 收尾队列事件 |

`needComputeFlag_` 描述卷积乘积是否需要计算。带 bias 时它为 false 仍须写回 bias；`firstContribution` 记录实际 MMAD，纯 bias MMAD 也计入。公共无 bias 行为保持原定义。

## 8 与外部知识的术语对接

公共对接表见 [../language.md](../language.md) §7。

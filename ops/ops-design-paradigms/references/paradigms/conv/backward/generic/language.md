# 卷积反向算子统一语言

> **层级**：通用知识层（generic/），语言中立。
>
> §0–§12 保留 input 分支术语；filter 分支使用 §13。通道轴、卷积属性和数学工具共用，M/N/K 与缓存内容按分支绑定。
>
> **目的**：为输入梯度、权重梯度与转置卷积的设计、tiling、kernel、验证统一轴义、属性和单位。M/N/K、缓存与任务字段按当前分支解释；具体计算与实现资料从 [input 索引](input/index.md) 或 [filter 索引](filter/index.md) 进入。

---

## 0 核心记号速查（先背这张表）

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **M0 = m0** | — | 16（dedx 空间方向分形基粒度） |
| **N0 = n0** | — | 16（dedx 通道方向分形基粒度） |
| **C0 = c0** | — | 16（dedy 通道分形基粒度；FP16/BF16 下与 m0/n0 同值） |
| **mValue / nValue / kValue** | — | `alignUp(hi*wi,m0)` / `alignUp(cinG,n0)` / `alignUp(coutG,C0)*hk*wk`，Host 选块的对齐长度 |
| **baseM / baseN / baseK** | — | 一次计算的空间、dedx 通道、K 容量（L0 基本块） |
| **singleCoreM / Cin / Din** | — | 一个任务的空间、dedx 通道、深度范围；一个核可处理多个任务 |
| **stepKa / stepKb** | — | 一份 L1A/L1B 驻留供给的 K 计算块数 |
| **loadB2Condition** | — | weight 变换分支：1=转置加逆序，2=仅逆序，3=仅转置 |
| **五路 pbuffer** | al1/bl1/al0/bl0/cl0Pbuffer | L1A/L1B/L0A/L0B/L0C 的物理槽份数，取 1 或 2 |
| **K 遍历顺序** | — | `Cout1 → r_h → r_w → Cout0`，固定不变 |
| **kernel（实现语境）** | 内核 | 设备侧程序，与 tiling（host 侧）相对 |

> ⚠️ 布局记号 ND/DN/NZ/ZN 里的 "N"、"D"、"Z" 只是快轴/慢轴记号，与 Cube 的 M0/N0/C0 **完全无关**，详见 [input/compute-flow.md](input/compute-flow.md) 的格式与轴序。

---

## 1 张量命名

| 张量 | 别名 / 接口字段 | 语义 |
|------|---------|------|
| **X / Y** | 正向输入、输出 | 数学定义用（正向卷积的输入/输出），不作为生成入口的给定对象；正向 Y 与接口输出 y 分开 |
| **weight** | filter | 正向权重；数据图中的 wght 也指该对象 |
| **dedy** | out_backprop | 正向输出梯度（数值输入） |
| **dedx** | y | 正向输入梯度（当前输出） |
| **input_size** | 输入张量或规格属性 | dedx 目标形状，按 data_format 解读 |
| **bias** | 可选输入 | 按 dedx 总通道排列的 FP32 向量（转置卷积） |

逻辑形状：X/dedx 为 `(N,Cin,Di,Hi,Wi)`，Y/dedy 为 `(N,Cout,Dout,Ho,Wo)`，weight 为 `(Cout,CinG,Dk,Hk,Wk)`。Cin/Cout 始终按正向卷积命名；代码字段保持 `cin/cout、cinG/coutG、di/hi/wi、dout/ho/wo、dk/hk/wk` 的实际拼写。

**转置卷积的对象绑定**：Conv3DTranspose 的 x/filter/y 分别绑定 dedy/weight/dedx。**Transpose 输入通道对应 Cout，输出通道对应 Cin**——这是复用计算层的命名约定；weight 第一轴保持 Cout，第二轴为 CinG。入口只绑定对象，通道转置与空间逆序由公共 weight 变换完成。

## 2 维度命名

| 记号 | 含义 |
|------|------|
| **N** | batch 维 |
| **Cin / Cout** | 正向输入 / 输出总通道数 |
| **G；CinG、CoutG** | 分组数；`CinG=Cin/G`、`CoutG=Cout/G` |
| **Di/Hi/Wi** | dedx 深度、高、宽 |
| **Dout/Ho/Wo** | dedy 深度、高、宽 |
| **Dk/Hk/Wk** | 核采样点数（dk/hk/wk 为核坐标轴名） |
| **s_a、δ_a** | a 轴的正向 stride、dilation（strideD/H/W、dilationD/H/W） |
| **p_a^-、p_a^+** | 正向前、后两侧 pad（padFront/Back、padUp/Down、padLeft/Right） |
| **b_a^-、b_a^+** | 反向有符号边界，正值补零、负值裁剪（backpropPad* 与裁剪起点共同表达） |
| **ci / co** | 组内通道坐标；全局通道加 `groupIdx*cinG/coutG` |
| **kd/kh/kw** | 原 weight 的核坐标 |
| **r_d/r_h/r_w** | 逆序核坐标，`r_a=K_a-1-k_a`，反向窗口配对 |
| **Cout1/Cout0、Cin1/Cin0** | 通道块轴和块内轴，块内长度 C0=16；坐标 co1/co0、ci1/ci0 与轴长分开 |

> ci/co 始终是组内坐标，换成全局通道需加组起点 `g×CinG` 或 `g×CoutG`。

## 3 卷积属性词汇

| 属性 | 统一形式 | 约定 |
|------|---------|------|
| strides | `(s_d,s_h,s_w)` | 批次和通道分量为 1；按张量轴序给出的先取空间分量 |
| dilations | `(δ_d,δ_h,δ_w)` | 卷积核采样点间隔 |
| pads | `(p_d^-,p_d^+,p_h^-,p_h^+,p_w^-,p_w^+)` | 依次为前后、上下、左右 |
| groups | 正整数 G | 每组通道数为 CinG、CoutG；G 整除 Cin、Cout |
| output_padding | `(q_d,q_h,q_w)` | 仅转置卷积；N/C 分量为 0；`q>=0` 且 `(q<stride or q<dilation)` |
| data_format | NCDHW / NDHWC | dedy/dedx 配对使用；weight 独立选择两种之一 |

## 4 M/N/K 三元组（GEMM 映射）

dedx 经窗口展开（img2col）映射为矩阵乘 `L0C[M,N] = L0A[M,K] × L0B[K,N]`：

```text
dedy -- stride insert zeros + boundary --> fmap [M,K] ------+
                                                            |
weight -- rot180 + channel role swap --> weight_rot [K,N] --+
                                                            |
                                                            |
                                                            v
                                           +----------------+----------------+
                                           | L0C[M,N] = L0A[M,K] * L0B[K,N]  |
                                           +----------------+----------------+
                                                            |
                                                            |
                                                            v
                                                +-----------+-----------+
                                                | dedx (N,Cin,Di,Hi,Wi) |
                                                +-----------------------+

L0A: row = dedx spatial m; col = K = (co1,r_h,r_w,co0)
L0B: row = the same K; col = group-local dedx channel ci
L0C: row = m; col = ci
```

| 记号 | 定义 | 说明 |
|------|------|------|
| **M** | `m = h_i*Wi+w_i`，展平 dedx 高宽 | 有效长度 `hi*wi`，选块长度 `mValue=alignUp(hi*wi,m0)` |
| **N 方向** | 组内 dedx 通道 ci | 与 batch N 分开；选块长度 `nValue=alignUp(cinG,n0)` |
| **K** | `kLinear = ((co1*Hk+r_h)*Wk+r_w)*C0+co0` | 有效归约项 `coutG*hk*wk`；选块长度 `kValue=alignUp(coutG,C0)*hk*wk` |
| **核深度** | kd | 不混入 K，独立于 K 循环完整归约 |

固定 `(n,g,curDinIdx_)` 后：L0A 的一行是一个 dedx 空间位置所需的 fmap 窗口；L0B 的一列是一个 dedx 通道所需的 weight_rot；L0A 的 K 列与 L0B 的 K 行必须对应同一个 dedy 通道和逆序核位置。补齐通道参与计算时贡献为零。

## 5 坐标映射（dedx 位置 → dedy 采样）

直接沿正向连接求梯度：

| 项 | 公式 | 说明 |
|----|------|------|
| 直接求和 | `dedx_g[n,ci,i] = Σ_co Σ_k weight_g[co,ci,k] · dedy_g[n,co,(i+p^- - k⊙δ)/s]` | 仅累加分子非负、能被 stride 整除且 dedy 坐标在界内的项 |
| 反向边界 | `b_a^- = K_a^eff-1-p_a^-`，`b_a^+ = K_a^eff-1-p_a^++ρ_a` | `K_a^eff=(K_a-1)*δ_a+1`；ρ_a 为尾部余数；正值补零、负值裁剪 |
| 等价窗口模型 | `fmap = dedy_expand 按反向边界补零/裁剪后的视图`，与 `weight_rot` 单位步长互相关 | `dedy_expand` 为按 stride 插零视图 |

这条映射是 img2col、反向边界、golden 生成（[input/verification.md](input/verification.md) 的坐标归一与基准实现）的共同数学依据；完整模型推导见 [input/scene.md](input/scene.md)。

**视图记号**：

| 名称 | 含义 | 实现落点 |
|------|------|---------|
| **dedy_expand** | dedy 按 stride 插零的视图 | L1A 中保存当前驻留片段 |
| **fmap** | dedy_expand 按反向边界补零/裁剪后的视图 | Load3D 取窗口 |
| **weight_rot** | 三个空间轴逆序、组内通道角色交换后的 weight 视图 | L0B 的计算语义 |

这三个都是数学视图，不要求分别分配中间张量：L1A 保存驻留内容，Load3D 结合 padList 形成 fmap 窗口。

## 6 缓存层级词汇

| 层级 | 存储内容 | 阶段名称 | SDK 位置 |
|------|---------|---------|---------|
| **L1A** | L1 的 dedy 驻留片段，供多个窗口复用 | LoadToA1 | `TPosition::A1` |
| **L1B** | L1 的原核顺序权重父段 | LoadToB1 | `TPosition::B1` |
| **L0A** | fmap 窗口，M×K，NZ | LoadToA2 / Load3D（img2col） | `TPosition::A2` |
| **L0B** | 转置逆序后的 weight，K×N，ZN | LoadToB2 / Load2D | `TPosition::B2` |
| **L0C** | M×N 的 FP32 部分和，NZ | MmadLocal / MMAD | `TPosition::CO1` |
| **dedx** | GM 中的有效输出 | LoadL0c2Gm / Fixpipe | GM |
| **父布局** | L1 本次驻留段的实际形状和步幅 | 子块按父布局寻址 | — |
| **槽 / slot** | 一份物理存储 | aSlot/bSlot/l0Slot/cSlot 分别轮转 | — |
| **al1/bl1/al0/bl0/cl0Pbuffer** | 五路物理槽份数，1 或 2 | 与队列模板深度分别使用 | — |

GM→L1 由 MTE2 完成，L1→L0A/L0B 由 MTE1 完成，矩阵乘加用 M，L0C→GM 用 F（Fixpipe）。SDK 中对应的流水枚举仍是 PIPE_FIX，队列事件仍使用 M_FIX/FIX_M；MTE3 用于适配层明确采用的辅助步骤。

元素偏移用于 Tensor 下标；`*Bytes`、容量和槽基址使用字节；分形步幅另标单位。`dedyBytes=weightBytes=dedxBytes=2`，`accBytes=4`。布局中的 ND/DN/NZ/ZN 不代表 batch 或空间深度。

## 7 格式词汇

详细排布图与偏移公式见 [input/compute-flow.md](input/compute-flow.md) 的格式与轴序，此处只列术语：

| 格式 | 一句话定义 |
|------|-----------|
| NCDHW / NDHWC | dedy/dedx 的两种存储轴序：通道在外 / 通道在内 |
| DHWCN | weight 的推导视角（dedy 通道连续），仅用于比较排布 |
| ND | 逻辑矩阵中空间点的通道连续 |
| DN | 通道的空间点连续 |
| NZ | 最内轴按分形块排：块间列优先、块内行优先 |
| ZN | 块间行优先、块内列优先 |

> NZ/ZN 必须说明相对哪一个矩阵，不能仅凭缩写推断通道朝向。

## 8 切分与执行坐标

### 8.1 单核规模与多核维度

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

### 8.2 迭代变量与父段状态

| 记号 | 含义 |
|------|------|
| kIter/kIterIdx | `ceilDiv(kValue,baseK)`、当前 K 轮号 |
| kStart | 当前完整 K 起点 |
| kLinear/kLocal | 完整 K 元素坐标、当前 L0 块内坐标 |
| segmentStart/segmentUse | 当前 L1 父段完整 K 起点、实际长度 |
| kLocalA/kLocalB | 当前 K 起点减去各自父段起点 |
| curStepKa/curStepKb | 当前父段实际覆盖的 K 轮数，末段按剩余量截断 |
| aCoutStart/bCoutStart、bCinStart | L1A/L1B 各自的 dedy 通道起点，以及 L1B 的 dedx 通道起点 |
| curCoutSize/curCinSize | 本次搬入的有效 Cout/Cin 通道数 |
| curCout1Cout0Size/curCin1Cin0Size | 当前通道数向 16 对齐后的长度 |
| blockBaseN_ | `ceilDiv(baseUseN_,16)`，当前 dedx 通道分形数 |
| curLoadKbl1_ | 当前 L1B 父段的实际 K 跨度 |
| curHoSize/woExpand | L1A 当前实际高宽，另记 dedy_expand 中原点 |

### 8.3 尾块与有效范围

| 记号 | 含义 |
|------|------|
| 尾块 | 任务/块尾部按剩余量收缩：`baseUse* = min(base*, 总长-起点)` |
| 有效 M 与对齐 M | 分别传递：Load3D.mExtension 和 Fixpipe.mSize 使用 baseUseM_；MmadParams.m 在任务末 M 块使用 mCompute |

## 9 数学记号

| 记号 | 定义 |
|------|------|
| `alignUp(x,a)` | 向上对齐到 a 的倍数 |
| `alignDown(x,a)` | 向下对齐 |
| `ceilDiv(x,a)` | 向上取整除法 |
| `gcd` | 最大公因数（绑核候选因子生成） |

字段改动时，同时核对其生产者、消费者和单位。

## 10 模板、分支与状态

| 记号 | 含义 |
|------|------|
| standard | 经验选块和常规任务分配；求解逻辑对应 InnerProduct |
| small-shape | 小任务量 M/N 均衡；对应 SmallShape |
| l1-resident | 完整 K 权重驻留；对应 FullLoad |
| templateId | 三个 Tiling 模板之一，只用于 Host 选择及检查 |
| loadB2Condition | 1=转置加逆序，2=仅逆序，3=仅转置 |
| groupConvMode/loadB1Condition | 当前分别为原始分组寻址 0、GM→L1 搬入 0 |
| tilingKey | 适配层按有序模板声明编码的编译分支；三个 Tiling 模板可命中同一个 key |
| enableFullLoad | Host 给出的权重驻留策略 |
| isB1FullLoadFlag | 当前 L1B 是否满足完整驻留范围及份数条件 |
| curEnableFullLoad | Kernel 当前跨任务保留权限，失效后按实现规则关闭 |
| needComputeFlag_ | 行带与 dedy_expand 有交集且存在有效 dedy 深度时进入计算；由 UpdateL1ComputeInfo 更新 |
| firstContribution | 当前输出块是否尚无实际 MMAD；跳过深度不改变它 |
| initOutputFlag | 1=需要输出初始化，0=主计算完整写回；执行方式在适配层确定 |

K 的遍历顺序固定为 `Cout1 → r_h → r_w → Cout0`。原核位置与逆序位置成对出现，空间反向关系每轴表达一次。

## 11 bias 词汇（转置卷积可选）

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

## 12 与外部知识的术语对接

| 外部知识 | 路径 | 对接说明 |
|---------|------|---------|
| 精度标准 | `cannbot-skills/ops/ops-precision-standard` | atol/rtol/ptol 与混合容差的系统性定义 |


## 13 filter 分支记号


> 以下记号用于 filter 分支，计算与实现资料分别从 [filter 通用索引](filter/index.md) 和 [filter 适配索引](../adapters/ascendc/filter/index.md) 进入。命名与 Host/Device 共享的 TilingData 字段对齐。

### 13.1 核心记号速查（先背这张表）

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **M** | `mValue = cout1_g·C0` | 组内 dedy 通道（coutG）对齐 C0=16；**输出块第一维**；`cout1_g=ceilDiv(coutG,C0)` 为组内 dedy 通道分形数 |
| **N** | `nValue = hk·wk·cin1_g·C0` | 组内 x 通道 × 核高宽窗口；**输出块第二维**（与 batch 维 N 区分，由上下文确定）；`cin1_g=ceilDiv(cinG,C0)` |
| **K** | `kValue = ho·wo` | 单层输出平面（dout 内），归约维 |
| **K 之外的独立归约维** | batch×dout | 基本块内分段进 L0C 累加，多段 Fixpipe 原子累加出 GM |
| **A 路 / B 路** | — | **A = dedy**（`[M,K]`，直装不窗口化）；**B = x**（`[K,N]`，Load3D 窗口化） |
| **basicBlockIdx 任务四元组** | `(mIdx, nIdx, dkIdx, gIdx)` | 一个基本块 = M 块 × N 块 × 一个核深度 × 一个组；totalCnt=mCnt·nCnt·dk·G |
| **reduceSegmentsND / reduceSegmentsH** | — | batch×dout / ho 的精度分段长度（累加轴阈值 `SINGLE_BLOCK_ADD_THRESHOLD=65536` 触发） |
| **大核拆分** | `isSplitKernelHW` | kh/kw/dilation/stride/pad 超限或 `kh·wk·C0>65536` 时的核窗口拆分路径 |
| **x_win** | fmap 窗口视图 | x 按正向 pad 补零、stride/dilation 采样形成的窗口矩阵（Load3D 语义） |
| **dedy_flat** | 输出网格视图 | dedy 按输出平面 (ho,wo) 展平后的 `[M,K]` 矩阵（无插零、无反向边界） |

### 13.2 M/N/K 与 GEMM 映射

```text
dedy -- flatten output plane --> dedy_flat [M,K] -----+
                                                      |
x -- pad + stride/dilation --> x_win [K,N] (Load3D) --+
                                                      |
                                                      |
                                                      v
                                     +----------------+----------------+
                                     | L0C[M,N] = L0A[M,K] * L0B[K,N]  |
                                     +----------------+----------------+
                                                      |
                                                      |
                                                      v
                                     +----------------+----------------+
                                     | dedw (Cout,CinG,Dk,Hk,Wk) FP32  |
                                     +---------------------------------+

L0A: row = group-local co; col = K = flattened (ho,wo)
L0B: row = the same K; col = ci / (kh,kw) blocks; N binds channels and kernel windows
L0C: row = co; col = ci*(kh,kw); one dedw output block
```

| 记号 | 定义 | 说明 |
|------|------|------|
| **M** | `mValue = cout1_g·C0` | 组内 dedy 通道对齐 16；单块 ≤ 256 |
| **N** | `nValue = hk·wk·cin1_g·C0` | 有效长度 `cinG·hk·wk`；任务 singleCoreN 保持完整通道窗口，L0 baseN 按 C0 对齐，可使用非整窗块 |
| **K** | `kValue = ho·wo` | 每层输出平面；baseK 由 L0A/L0B 容量推出并按 16 对齐，优先对齐 splitWo |
| **核深度 kd** | 任务轴 | 不混入 K/N；每个基本块固定一个 dkIdx（cin 与 dk 分轴） |
| **batch×dout** | 核内归约轴 | 基本块内沿 batch·dout 分段，段内 L0C 累加，段间 Fixpipe 原子累加出 GM |

### 13.3 切分与执行坐标

| 记号 | 含义 |
|------|------|
| baseM / baseN / baseK | L0 基本块（TilingData 字段 baseM/baseN/baseK）；M/N ≤ 256 且按 C0 对齐；普通整窗判定为 `(baseN/C0) % (hk*wk) == 0`，其余使用非整窗通路 |
| singleCoreM / singleCoreN / singleCoreK | 一个基本块任务的 M/N/K 范围（`singleCoreM=baseM`、`singleCoreN` 为 cin1_g 的完整核窗口数×C0、`singleCoreK=kValue`） |
| singleCoreCout / singleCoreCin | 任务在 co / ci 方向的通道跨度 |
| singleCoreBatchDout / singleCoreHo | batch·dout 基值 / singleCoreK 折算的输出行数 |
| mCnt / nCnt / totalCnt | M 块数 / N 块数 / 基本块总数 `mCnt·nCnt·dk·G` |
| usedCoreNum / calRound | 启动核数 / 每核轮数 `ceil(totalCnt/usedCoreNum)`；`basicBlockIdx += usedCoreNum` 轮转 |
| stepKa / stepKb | L1A（dedy）/ L1B（x 窗口父段）驻留覆盖的 K 块数；取彼此整数倍 |
| k | 当前 ComputeLoop 内的全局 K 块号，`kStart=k*baseK`；每次进入该 K 循环从 0 开始 |
| kaIdx / kbIdx | 父段内 K 块号，分别为 `k%stepKa`、`k%stepKb`；用于父段内寻址与换驻判断 |
| kaStepIdx / kbStepIdx | 父段号，分别为 `k/stepKa`、`k/stepKb`，均取整数除法；父段起点为各自父段号乘 step 与 baseK |
| depthA1 / depthB1 | L1 配比深度（求解器内部）；16 起按 2 阶梯衰减，≤2 时双缓冲降为 1（发布为 al1Pbuffer/bl1Pbuffer） |
| splitWo / splitWi / tailWo / tailWi | 输出宽分段长度（wo>512 时按 128 分段）及对应输入宽 / 尾段（发布字段仅 splitWo，其余为派生量） |
| al1Bound / bl1Bound | L1A / L1B 的驻留上界（元素数） |
| isSplitKernelHW | 大核拆分开关（核窗口超过 Load3D/搬运字段上限） |
| iterateOrder | `mValue>nValue` 时为 1，否则 0；控制任务内 M/N 子块遍历与父段复用。任务号仍按 group→kd→N→M（快→慢）解码 |
| singleShapeK / singleShapeBatch_ | 运行时段内 K 长度（`min(ho−hoStart, reduceSegmentsH)·wo`）/ 段内 batch·dout 长度（kernel 运行量，区别于 TilingData 字段） |

### 13.4 状态与执行判定

| 记号 | 含义 |
|------|------|
| SINGLE_BLOCK_ADD_THRESHOLD | 累加轴阈值 65536；`batch·dout·ho·wo` 超过它则对 batch·dout 分段 |
| initOutput | AIV 输出预清零阶段（Fill 清零 dedw GM，AIC WaitFlag 等待）；原子累加写回的前置义务 |
| needCompute / invalidDIndex | 段内 dout 全部落在 pad 外时该段无有效贡献，跳过计算（对应"无贡献核位置"的执行侧表达） |
| ReCalDkCinSingleCoreShape | 按 dout 段与 group 尾块收缩当前段 singleCoreM/N 有效范围的运行时重算 |
| l0cPingPongFlag_ | L0C 双缓冲乒乓标志，M/N 子块迭代间翻转 |
| offsetA_ / offsetB_ / offsetC_ | 当前段三路 GM 偏移；每段重算，batch/dout/ho 变化影响 A/B，不影响 C |

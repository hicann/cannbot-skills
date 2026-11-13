# 卷积反向算子场景与计算模型

> **层级**：通用知识层（generic/input/），语言中立。
>
> **分工**：本文件回答"**算什么**"——反向梯度/转置卷积的数学模型与视图定义（前放大、rot180、反向边界、乘加归约、output_padding 与 bias）。"数据怎么流、任务怎么分、循环怎么写、同步怎么插"见 [compute-flow.md](compute-flow.md)；规格整理见 [../preprocessing.md](../preprocessing.md)。后续文档沿用这里的对象、符号与步骤名称。

---

## 1 场景准入

本范式处理三维卷积的反向计算：给定正向权重 weight、输出梯度 dedy、目标输出形状及卷积属性，计算输入梯度 dedx；转置卷积是同一计算模型的正向语义接口。两类生成请求在这里统一：

| 进入场景 | 给定信息 | 整理方式 |
|---|---|---|
| 输入梯度 dedx（Conv3DBackpropInput） | weight、dedy 的规格，目标输入形状（dedx 规格或 input_size），以及 strides/pads/dilations/groups | input_size 按 data_format 解读为 dedx 形状；逐轴套正向形状公式 `输出长度 = floor((输入长度 + 两侧 pad − 有效核长度)/stride) + 1`（输入长度取 dedx 轴，输出长度取 dedy 轴）校验三者对应同一个卷积问题 |
| 转置卷积（Conv3DTranspose） | x、filter、可选 bias 的规格与卷积属性（含 output_padding），目标 y 形状或 input_size | 按对象绑定 x→dedy、filter→weight、y→dedx 转入同一计算层；y 的空间尺寸由属性推导或显式 input_size 确定（q 与 rho 的形状关系见 [../preprocessing.md](../preprocessing.md)） |

**转置卷积的进入绑定**：Conv3DTranspose 提供的是正向语义接口，对象按下表绑定后进入同一个计算层：

| 外部对象 | 公共计算对象 | 逻辑形状 |
|---|---|---|
| x | dedy | `(N,Cout,Dout,Ho,Wo)` |
| filter | weight | `(Cout,CinG,Dk,Hk,Wk)` |
| y | dedx | `(N,Cin,Di,Hi,Wi)` |
| bias | bias | `(Cin,)`，按输出通道广播 |
| input_size | dedx 目标形状 | 按 data_format 排列的五个尺寸 |

满足以下条件后，进入计算模型与方案设计：

- **语义明确**：求三维卷积关于输入的梯度或转置卷积，各组使用本组的通道与权重连接。
- **规格完整**：能够确定 weight、dedy、dedx 的 shape、dtype、format，以及各空间轴的步长、膨胀率和两侧补边。
- **形状一致**：批次数、通道数、分组数、卷积核尺寸和空间尺寸相互匹配；目标形状由 dedx 规格或按 format 解读的 input_size 明确给出。
- **数据规格匹配**：采用 FP16 或 BF16；dedy、dedx 使用一致的 NCDHW 或 NDHWC 格式，weight 使用 NCDHW 或 NDHWC 格式；转置卷积的 bias 为长度 Cin 的 FP32 向量。具体组合与生成参数满足所选适配层契约。

生成阶段使用上述规格；执行阶段使用 weight、dedy 的数值及相应几何信息。目标输入形状统一记为 `S_X`，后续所有输出范围都以它为依据。

## 2 计算模型

### 2.1 统一符号

设标量目标为 L，卷积属性统称为 θ：

$$
Y=\operatorname{Conv3D}(X,\mathrm{weight};\theta),
\qquad
\mathrm{dedy}=\frac{\partial L}{\partial Y},
\qquad
\mathrm{dedx}=\frac{\partial L}{\partial X}.
$$

dedy 是已知输出梯度，dedx 是待求输入梯度；weight 保持正向权重的含义。正向采用空间互相关约定，即按原权重坐标采样。

固定权重与属性后，正向卷积是关于 X 的线性映射。dedx 沿正向连接把每个输出梯度乘以对应权重，再累加到相应输入位置。下面将这一过程改写为"梯度展开＋反向权重＋单位步长互相关"。

### 2.2 前放大：对 dedy 插零

沿每个空间轴，在相邻 dedy 采样点之间插入"步长减一"个零，得到展开视图 dedy_expand。这样，原步长造成的采样间隔就被显式表达，后续计算按单位步长移动。

例如，某一行 dedy 为 `[a,b,c]`，该方向步长为 2，则展开为 `[a,0,b,0,c]`。步长为 1 时，该方向保持原有采样间隔。

### 2.3 权重变换：rot180 与通道交换

三维 rot180 的含义是：**每个高度×宽度切片旋转 180°，同时反转深度方向的切片顺序**。随后交换组内输入、输出通道的角色，得到反向权重视图：

$$
\mathrm{weight}_{g}^{\mathrm{rot}}[ci,co,r_d,r_h,r_w]
=
\mathrm{weight}_g[co,ci,Dk-1-r_d,Hk-1-r_h,Wk-1-r_w].
$$

这个公式同时表达空间逆序与组内通道交换。例如，核尺寸为 2×3×3 时，原空间位置 (0,0,0) 对应逆序后的 (1,2,2)，原位置 (0,1,2) 对应 (1,1,0)。

固定一对组内通道，取 2×2×2 核；数字标识原权重元素：

```text
weight: kw ->    weight_rot: r_w ->
kd=0             r_d=0
+---+---+        +---+---+
| 0 | 1 |        | 7 | 6 |
| 2 | 3 |        | 5 | 4 |
+---+---+        +---+---+
kd=1             r_d=1
+---+---+        +---+---+
| 4 | 5 |        | 3 | 2 |
| 6 | 7 |        | 1 | 0 |
+---+---+        +---+---+

kd=1 -> r_d=0; kd=0 -> r_d=1
Reverse depth order and rotate each H/W slice by 180 degrees.
```

通道交换表示：反向计算从 dedy 的 Cout 个通道中归约，产生 dedx 的 Cin 个通道。这里先定义逻辑关系，具体物理布局和矩阵转置方式由布局与适配文档确定。

### 2.4 边界处理：确定反向补边与输出范围

对空间轴 a，有效核长度 `K_a^eff` 等于"核采样点数减一，乘 dilation，再加一"。记尾部余数为 ρ_a：它是"正向输入长度加两侧正向补边、减去有效核长度"除以步长后的余数，由目标输入形状确定。

反向两侧补边为：

$$
b_a^-=K_a^{\mathrm{eff}}-1-p_a^-,
\qquad
b_a^+=K_a^{\mathrm{eff}}-1-p_a^++\rho_a.
$$

正值表示补零，负值表示裁去对应边界。对展开视图 dedy_expand 应用这组边界映射，得到 fmap；其采样起点由前侧补边确定，采样终点结合目标形状确定。

例如，只看高度轴：核高为 3、步长为 2、dilation 为 1、正向上下各补 1 时，输入高度 7 和 8 都得到输出高度 4。反向展开后高度为 7；目标高度为 7 时，反向上下各补 1；目标高度为 8 时，尾部余数为 1，下侧反向补边增至 2。因此，目标形状参与确定整个计算范围。

### 2.5 乘加归约：生成 dedx

使用 fmap 和反向权重 weight_rot 做单位步长互相关，保留原 dilation 与分组关系：

$$
\mathrm{dedx}_g[n,ci,\mathbf i]
=
\sum_{co=0}^{(CoutG)-1}
\sum_{\mathbf0\le\mathbf r<\mathbf K}
\mathrm{weight}_{g}^{\mathrm{rot}}[ci,co,\mathbf r]\,
\mathrm{fmap}_g[n,co,\mathbf i+\mathbf r\odot\boldsymbol\delta].
$$

其中 ⊙ 表示逐轴相乘。对每个 dedx 位置，遍历同组的 dedy 通道和核位置，把有效乘积累加起来；无有效贡献时，数学结果为零。最终输出形状为 S_X。

后续统一用 dedy_expand 表示插零视图、fmap 表示处理边界后的视图、weight_rot 表示反向权重视图。直接沿正向连接求梯度时，原权重与反向索引成对使用；采用本节互相关模型时，逆序权重与正向移动的索引成对使用，空间反向关系在各自形式中表达一次。直接求和形式及窗口化见 [compute-flow.md](compute-flow.md)。

### 2.6 核为 1×1×1 的退化情形

三个核坐标均为 0，rot180 保持权重值不变，反向权重只交换组内通道角色。每个 dedx 位置按 `o_a=(i_a+p_a^-)/s_a` 查找 dedy；三个轴均满足整除且坐标有效时，归约同组 dedy 通道，否则该位置为零。

例如 stride 为 1、pad 为 0，`dedy=(320,64,1,1,1)`、`weight=(64,32,1,1,1)`，则 `dedx=(320,32,1,1,1)`。每个批次用 64 个 dedy 通道计算 32 个 dedx 通道。布局层将这一权重准备方式称为**仅转置**。核体积为 1 不改变 stride、边界和目标形状的校验规则。

### 2.7 转置卷积的 output_padding 与 bias

转置卷积复用上述窗口模型：x 按 stride 形成 dedy_expand，结合有符号边界形成 fmap；filter 构造 weight_rot，与 fmap 窗口相乘。它是散射关系的等价改写，每轴只处理一次。

**output_padding（q）**：属性推导模式中，每个空间轴的输出长度为

$$
L_y=(L_x-1)s-p^- -p^+ +(K-1)\delta+1+q.
$$

q 非负，且满足 `q < stride 或 q < dilation`。q 改变输出范围，不改变 weight 的核采样点数和 K 归约长度。新增输出位置仍按相同采样关系计算，可能有非零乘积。

例如宽度方向 `x=[1,2]、weight=[1,2,3]、stride=2、左右 pad=1`：q=0 时输出 `[2,5,4]`，q=1 时输出 `[2,5,4,6]`。最后的 6 来自已有输入和权重。带显式目标形状时的余量 ρ 与 q 的关系见 [../preprocessing.md](../preprocessing.md)。

**bias**：对每个 x 元素，按原核坐标将乘积累加到输出位置——某轴的目标位置为 `i_a=o_a s_a-p_a^-+k_a δ_a`，仅对目标范围内的位置累加，同组内连接。完成全部输入通道与核位置的归约后，对每个输出通道加一次 bias；未提供 bias 时该项为零。**每个输出位置都具有此语义，包括没有有效乘积的位置**。

## 3 能力边界

本模型的适用条件和实现约束按以下维度说明：

| 维度 | 条件与约定 |
|---|---|
| 数学语义 | 三维输入梯度/转置卷积、正向零填充、各组独立；批次、通道与空间尺寸为正整数，分组数为正整数并整除输入和输出通道数 |
| 几何关系 | 步长、膨胀率、核尺寸为正整数；正向补边为非负整数，形状关系成立；反向边界量允许用有符号方式表达补边或裁剪；output_padding 满足 q 的取值约束 |
| 数据表达 | 使用入口约定的 dtype、format；逻辑形状与实际存储映射一致，尾部和无有效贡献位置都有明确的结果语义（DX 为零，Transpose 为可选 bias 经输出转换后的值） |
| 实现能力 | 目标语言、平台和 SDK 能表达所需的采样、布局转换、乘加累加、结果写回与同步，且切分方案满足资源和对齐条件；带 bias 时额外核对 BT、带 bias 的 Mmad 及其同步通路 |
| 验证依据 | 用独立的梯度求和验证数学关系，再按 dtype 检查实际数值结果；生成方案还需满足对应适配层的 Tiling 准入条件 |

dedy_expand、fmap、weight_rot 都是数学视图。实现可将这些变换落实为索引映射、搬运过程中的布局处理，或显式中间数据；具体选择以能力映射、资源预算和数据生命周期为依据。

切分尺度与资源约束见 [tiling/tiling.md](tiling/tiling.md)，目标平台和 API 参数由适配层提供。

## 4 数据流 backbone

以一个任务负责的 dedx 输出区域为单位，主数据流分为两路操作数准备、乘加累加和结果写回：

```text
Fixed batch, group and dedx output region

dedy -- stride insert zeros --> dedy_expand -- pad/crop --> fmap --+
                                                                   |
weight -- rot180 + channel role swap --> weight_rot ---------------+
                                                                   |
                                                                   |
                                                                   v
                                                        +----------+----------+
                                                        | paired multiply-add |
                                                        +----------+----------+
                                                                   |
                                                                   |
                                                                   v
                                                 +-----------------+-----------------+
                                                 | full co / kd / kh / kw reduction  |
                                                 +-----------------+-----------------+
                                                                   |
                                                                   |
                                                                   v
                                                        +----------+----------+
                                                        | dedx output region  |
                                                        +---------------------+
```

带 bias 的转置卷积在乘加首步引入 bias 通路：

```text
x / dedy --> dedy_expand --> fmap window --> L0A --+
                                                   |
filter / weight --> weight_rot --> L0B ------------+
                                                   |
bias --> L1 bias --> BT (first MMAD only) ---------+
                                                   |
                                                   |
                                                   v
                                            +------+------+
                                            | MMAD / L0C  |
                                            +------+------+
                                                   |
                                                   |
                                                   v
                                              +----+----+
                                              | Fixpipe |
                                              +----+----+
                                                   |
                                                   |
                                                   v
                                             +-----+-----+
                                             | y / dedx  |
                                             +-----------+
```

| 阶段 | 输入与输出 | 关键约定 |
|---|---|---|
| 前置处理与任务建立 | 原始规格 → 统一几何、目标形状与任务范围 | 两类进入场景在此统一，任务范围与目标输出坐标对应 |
| dedy 路准备 | dedy → 当前计算需要的 fmap 片段 | 按步长定位有效采样点，应用边界映射，并形成计算所需布局 |
| weight 路准备 | weight → 当前计算需要的 weight_rot 片段 | 空间逆序、组内通道映射与归约范围保持一致 |
| bias 路准备（可选） | bias → L1 → BT | 首次有效乘加经 BT 引入 L0C 初值；无贡献输出块单独产生 bias |
| 乘加累加 | 两路操作数 → 当前输出区域的部分和或完整结果 | 首次贡献前建立累加器初值，后续贡献继续累加 |
| 结果写回 | 完整结果 → dedx | 完成该区域的全部有效贡献后，按目标坐标和有效范围写回 |

对于同一输出区域，按归约切分重复两路操作数准备和乘加，直到完成全部同组通道及核位置。操作数就绪后才能计算，读取完成后才能复用对应 buffer；数据流和同步设计据此确定各阶段的依赖。

这里描述的是逻辑依赖。两路准备的执行顺序、数据复用、buffer 份数及流水重叠，在具体能力和资源条件下确定；具体 DAG 与四级流水见 [compute-flow.md](compute-flow.md)。

## 5 检查点

- [ ] 两类进入场景推导出的 dedy/dedx/weight 逻辑形状一致，与 data_format 解读无关
- [ ] 反向补边 `b^-、b^+` 保留有符号值，未转成无符号大数；目标形状参与确定 ρ 与后侧补边
- [ ] weight_rot 的三维 rot180（H/W 切片旋转 + 深度切片逆序）与原核寻址是同一反向关系的两种表达，每轴只表达一次
- [ ] 1×1×1 核退化时仅转置分支仍检查 stride/pad 采样
- [ ] 转置卷积的 q 满足 `q<stride or q<dilation`；q 只改变输出范围，不改变核采样点数与 K 长度
- [ ] 带 bias 时无贡献位置写 bias 值，无 bias 时写零；两者都不依赖 weight/dedy 数值
- [ ] dedy_expand/fmap/weight_rot 均为数学视图，未要求物化中间张量

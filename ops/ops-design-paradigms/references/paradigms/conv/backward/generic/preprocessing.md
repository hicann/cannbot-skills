# 卷积反向算子规格归一（前置处理）

> **层级**：通用知识层（generic/），语言中立。
>
> **分工**：本文件回答"**原始规格如何变成统一的 3D 反向问题**"——对象识别、属性归一、升维、形状校验与派生量、输出形状模式、执行前置约定。归一化统一形状、轴语义和属性，张量数值保持不变；清零、数据重排等执行步骤形成约定，由后续方案确定实施阶段。计算与布局从 [input 索引](input/index.md) 或 [filter 索引](filter/index.md) 进入。

第 1–7 节保留 input 的规格绑定。filter 共用属性展开、单维插入与正向形状公式，具体对象、目标形状和执行前置按第 8 节绑定。

---

## 1 处理顺序

```text
     +---------------------------------------+
     | Conv3DBackpropInput / Conv3DTranspose |
     +-------------------+-------------------+
                         |
                         |
                         v
+------------------------+------------------------+
| Resolve tensor roles, formats and spatial rank  |
+------------------------+------------------------+
                         |
                         |
                         v
     +-------------------+-------------------+
     | Normalize attributes and target shape |
     +-------------------+-------------------+
                         |
                         |
                         v
               +---------+---------+
               | Map 1D / 2D to 3D |
               +---------+---------+
                         |
                         |
                         v
       +-----------------+-----------------+
       | Validate shapes; derive geometry  |
       +-----------------+-----------------+
                         |
                         |
                         v
 +-----------------------+-----------------------+
 | Normalized problem + execution prerequisites  |
 +-----------------------------------------------+
```

设计前先完成规格整理。

## 2 统一输入说明

### 2.1 两种进入场景

| 输入场景 | 整理规则 |
|---|---|
| 输入梯度 dedx（Conv3DBackpropInput） | input_size 按 data_format 解读为 dedx 形状（或直接给定 dedx 规格），随后按第 4.1 节的正向形状公式逐轴校验 dedx/dedy/weight/属性对应同一个卷积问题；校验通过后 dedy、dedx 的角色即数值输入/输出 |
| 转置卷积（Conv3DTranspose） | x/filter/y 按对象绑定（x→dedy、filter→weight、y→dedx）进入，output_padding 并入属性统一；y 的空间尺寸按第 4.2 节两种模式确定（属性推导或显式 input_size），随后与输入梯度场景执行同一套形状校验（绑定表见 [input/scene.md](input/scene.md)） |

`input_size` 先按声明的轴序解释，再与 dedx 的逻辑形状比较。两者同时给出时必须一致。冲突字段保留原值并说明冲突关系。梯度的 dtype、format 单独确定。

### 2.2 张量说明

每个对象保留原始规格，并建立以下统一说明：

| 字段 | 约定 |
|---|---|
| 对象角色 | weight 为正向权重，dedy 为数值输入，dedx 为数值输出；X、Y 提供形状依据；转置卷积的 bias 为可选 FP32 向量 |
| 逻辑形状 | dedx：`(N,Cin,Di,Hi,Wi)`；dedy：`(N,Cout,Dout,Ho,Wo)`；weight：`(Cout,CinG,Dk,Hk,Wk)`；bias：`(Cin,)` |
| 存储描述 | 实际 shape、format、轴映射、元素步幅及基址偏移；连续张量按格式推导步幅，视图使用实际步幅 |
| dtype | 已确认的 `fp16/float16` 统一为 `float16`，`bf16/bfloat16` 统一为 `bfloat16`；分别记录 weight、dedy、dedx 的类型；bias 为 FP32 |
| 字段来源 | 标明调用方给定、接口默认或公式推导，保留对应原始字段 |

逻辑形状统一后，NCDHW、NDHWC 仍保留各自的存储顺序。weight 格式中的 N 表示 Cout，C 表示组内输入通道数。dtype 组合由适配层核对，类型转换须单独声明。

### 2.3 属性说明

先按原始空间维数和接口约定解析，再升维为以下形式：

| 属性 | 统一形式 |
|---|---|
| strides | `(s_d,s_h,s_w)` |
| dilations | `(δ_d,δ_h,δ_w)` |
| pads | `(p_d^-,p_d^+,p_h^-,p_h^+,p_w^-,p_w^+)`，依次为前后、上下、左右 |
| groups | 正整数 G；每组通道数为 `CinG`、`CoutG` |
| output_padding | `(q_d,q_h,q_w)`；N/C 分量为 0，缺省为五个 0 |

- 标量按接口约定扩展到**原始空间维数**，再补新增轴。缺省值注明来源。
- 按张量轴序给出的 strides、dilations，先取空间分量；批次和通道分量须为 1。例如 NDHWC 的 `(1,2,3,4,1)` 归一为 `(2,3,4)`。
- 对称 pad 展开为两侧值；非对称 pad 按轴和侧别重排。符号 padding 按已明确的规则求出两侧值。
- output_padding 按轴序取空间分量：NCDHW 为 `[0,0,qD,qH,qW]`，NDHWC 为 `[0,qD,qH,qW,0]`；内部使用 `qD/qH/qW` 与空间轴逐一对应。
- 属性轴序、pad 侧别或默认规则不明确时，列出缺项；不能仅凭数组长度猜测含义。

## 3 1D / 2D 升维

1D 的宽度映射到 W，2D 的高、宽映射到 H、W。新增空间轴长度为 1：

| 原空间维数与格式 | 原 shape | 3D shape 与格式 |
|---|---|---|
| 1D，NCW | `(N,C,W)` | `(N,C,1,1,W)`，NCDHW |
| 1D，NWC | `(N,W,C)` | `(N,1,1,W,C)`，NDHWC |
| 2D，NCHW | `(N,C,H,W)` | `(N,C,1,H,W)`，NCDHW |
| 2D，NHWC | `(N,H,W,C)` | `(N,1,H,W,C)`，NDHWC |
| 3D | 原 5D shape | 保持原轴映射 |

X、Y、dedy、dedx 使用同一空间映射。weight 在对应核空间轴补 1，保留 Cout 和 CinG。

| 新增轴的项目 | 取值 |
|---|---|
| 输入、输出及核尺寸 | 1 |
| stride、dilation | 1 |
| 前后 pad、output_padding | 0 |
| 有效坐标 | 0 |

例如 1D 的 `stride=2` 升维后为 `(1,1,2)`；1D 的 `q=1` 升维后为 `(0,0,1)`。原有空间轴即使长度为 1，也保留其轴名和属性。

**升维保持元素数和地址映射。** 连续张量可通过插入单维建立视图；已有轴保留原存储步幅。若需要改变通道存储顺序，另行记录实际重排步骤。正式生成接口统一传入 3D 规格；1D/2D 的外层包装按所选工作流接入。

产出保留原空间维数、插入轴及逆映射。还原低维 shape 时，只移除本次插入的轴。

## 4 形状校验与派生量

### 4.1 输入梯度场景：逐轴校验正向形状关系

```text
有效核长度 = (核尺寸 - 1) × dilation + 1
输出长度 = floor((输入长度 + 前 pad + 后 pad - 有效核长度) / stride) + 1
```

这里输入长度来自 dedx，输出长度来自 dedy。检查：

- shape、核尺寸、stride、dilation、G 均为正整数，正向 pad 为非负整数。
- dedy、dedx 批次数一致；G 整除 Cin、Cout；weight 通道为 Cout 和 CinG。
- 三个空间轴满足上式，目标形状与 input_size 一致。
- 尺寸乘积、元素偏移和字节数计算不溢出，目标接口字段可表示对应值。

有 stride 时，同一 dedy 尺寸可能对应多个 dedx 尺寸。目标形状决定尾部余数和反向后侧补边，必须保留。

### 4.2 转置卷积场景：两种输出形状模式

令 `Keff=(K-1)*dilation+1`，`base=(Lx-1)*stride-pMinus-pPlus+Keff`。形状计算使用有符号宽整数，并检查乘积与最终字段范围。

| 模式 | y 的空间长度 | 校验 |
|---|---|---|
| 属性推导 | `Ly=base+q` | Ly 为正；input_size 全零哨兵在推导后替换成真实尺寸 |
| 显式 input_size | 使用其元素值作为 Ly | `rho=Ly-(base+q)`，要求 `0<=rho<stride` |

显式模式对应 Host 的反算检查 `Lx=(Ly+pMinus+pPlus-(Keff+q))//stride+1`，检查前保证分子非负。它允许 stride 造成的尺寸余量；rho 是该模式的形状余量，属性推导模式为 0。两种模式都要求 `q>=0` 且 `(q<stride or q<dilation)`。

input_size 全零表示属性推导；非零模式的五个尺寸均为正，N 与 x 相同，Cin 等于 `weight.CinG*G`。调用方另给 y 规格时校验一致。部分为零的形状报告具体字段错误。推导和实际 Tiling 共用这套规则。

例如 `Lx=2、K=3、stride=2、pad=(1,1)、dilation=1、q=0`，属性模式得到 Ly=3；显式 Ly=4 的 rho=1，也满足本接口的形状关系。若业务要求唯一的属性推导输出，则在包装层选择属性模式。Host 形状检查使用的 `filter_*_dilation=Keff+q` 只服务于形状检查和初始化判定；Kernel 的真实核几何与 K 归约长度不受 q 影响。

### 4.3 派生量

随后记录：

| 派生量 | 求法与用途 |
|---|---|
| 有效核尺寸 | 按上式求 D、H、W 三个值 |
| 尾部余数 ρ | "输入长度加两侧 pad、减有效核长度"对 stride 取余；每轴满足 `0 ≤ ρ < stride`；转置卷积的 ρ 按其形状关系（含 q）计算 |
| dedy_expand 的空间尺寸 | 每轴为 `(dedy 轴长 - 1) × stride + 1` |
| 反向补边 `b^-、b^+` | 按 [input/scene.md](input/scene.md) 计算；保留有符号值，供 fmap 的补零或裁剪使用；转置卷积等价地 `bPlus=Keff-1-pPlus+q+rho` |
| 分组映射 | 组内通道加上 `g×CinG` 或 `g×CoutG` 后得到全局通道 |

## 5 执行前置约定

### 5.1 视图与数据处理

dedy_expand、fmap、weight_rot 沿用 [input/scene.md](input/scene.md) 的逻辑视图。前置处理先给出映射，后续方案选择索引计算、随搬运转换或显式缓冲区。实际重排须记录源/目标布局、有效范围、存储大小和完成依赖。

各种零分别说明：

| 零的来源 | 责任 |
|---|---|
| dedy_expand 的步长插零、fmap 的边界补零 | 对应采样位置提供零值 |
| 通道或分形对齐补零 | 补齐项参与计算时贡献为零 |
| dedx 输出预清零 | 为主计算跳过的输出位置建立零结果 |
| 局部累加器初值 | 每个新 dedx 计算块首次累加前初始化 |

### 5.2 dedx 输出预清零

无贡献位置按连接关系判定，与 weight、dedy 的数值无关。后续 Tiling 和 Kernel 方案须说明写回覆盖范围：

- 主计算跳过无贡献位置时，先为这些位置建立零值。可将整个 dedx 预清零，再覆盖有效计算结果；有贡献位置须完整计算并写回。
- 主计算完整写入所有 dedx 位置，并为无贡献位置写零时，可省去独立预清零。
- 全部输出均无贡献时，仍须产出全零 dedx。局部累加器按计算块独立初始化。
- 带 bias 时清零仍在主计算之前；主计算对无贡献位置写入 bias，清零本身不构成最终结果。

需要独立清零时，执行顺序为：

```text
         +-------------------+
         | Output allocated  |
         +---------+---------+
                   |
                   |
                   v
+------------------+------------------+
| Required output region initialized  |
+------------------+------------------+
                   |
                   |
                   v
   +---------------+---------------+
   | Main computation writes dedx  |
   +---------------+---------------+
                   |
                   |
                   v
        +----------+----------+
        | dedx ready to read  |
        +---------------------+
```

每次调用都遵守此顺序。清零区与仍需读取的 weight、dedy 不重叠；后续清零不得覆盖已写回结果。连续 dedx 记录起点和有效字节数，带间隔存储记录有效地址区间。

初始化判定（Host 的 initOutputFlag）按算子场景分别汇总：

| 场景 | 置 initOutputFlag=1 的条件 |
|---|---|
| 输入梯度（DX 形状） | `ρ_d>padBack`，或 `ρ_h>padDown`，或 `strideH>hk`，或 `dilationD>1`；当前常规核窗口 Tiling 追加 `strideD>dk` |
| 转置卷积（rho 按其形状关系计算） | `rhoD>padBack`、`rhoH>padDown` 或 `strideH>hk`；`qD>0` 或 `qH>0`，以及 `dilationD>1`；所选 Tiling 中 `strideD>dk` 的补充条件 |

例如 D 轴取 `di=5、dout=2、dk=2、strideD=dilationD=2、前后pad=0`，深度1、3没有有效贡献；此时 strideD 与 dk 相等，仍由 dilationD 条件启用清零。这些条件来自无贡献区域与跳过路径，不能只用其中一条代替完整判断；qW 通过输出宽度和窗口边界参与处理。最终标志还与主计算覆盖检查一同发布。

通用层产出清零条件、范围和依赖。适配层落实 API、执行单元及两种调用方式的接入，具体执行见 [adapters/ascendc/input/sync-implementation.md](../adapters/ascendc/input/sync-implementation.md)。写回方案未确定时，将清零决策标为待确定。

### 5.3 bias 前置（转置卷积）

bias 缺省时 `hasBiasFlag_=false`。非空 bias 须为一维 FP32，长度 Cin；无元素的可选 Tensor 由包装层统一成缺省指针。输出通道的全局偏移为 `groupIdx*cinG+cinStart`，与 weight 的组内通道字段区分。带 bias 时，主计算还须为无贡献位置产生 bias：使用带 BT 的零乘积 MMAD，再按正常输出通路写回；预清零、L0C 首次初始化、bias 加一次分别承担各自职责，具体执行见 [adapters/ascendc/input/bias-bt-path.md](../adapters/ascendc/input/bias-bt-path.md)。

## 6 示例

以下只保留语义关键的小规模示例（转置卷积的 output_padding/bias 与输出预清零）；升维、属性归一与形状校验的规则见前文各表，完整规格推导属验证期工作，不在范式文档展开。

### 6.1 转置卷积：output_padding 与 bias

取 1D 问题 `x=[1,2]、weight=[3]、stride=2、pad=0、q=1、bias=5`，升维后核为 `(1,1,1)`、q 为 `(0,0,1)`，y 宽度为 4。乘积贡献为 `[3,0,6,0]`，最终 y 为 `[8,5,11,5]`。

另取 `stride=1、dilation=2、q=1、核宽=2`：q 虽不小于 stride，但小于 dilation，满足属性条件。x 宽为 2 时，属性模式输出宽为 5；不能再用普通 DX 的正向形状式否定该目标。

### 6.2 输出预清零

取 3D NCDHW 问题，仅宽度方向非单维：

```text
dedx.shape   = (1,1,1,1,5)
dedy.shape   = (1,1,1,1,3)
weight.shape = (1,1,1,1,1)
strides      = (1,1,2)
dilations    = (1,1,1)
pads         = (0,0,0,0,0,0)
G            = 1

dedy         = [1,2,3]
weight       = [2]
dedx before  = [9,9,9,9,9]

Main computation writes contributing positions only:
after zero   = [0,0,0,0,0]
after write  = [2,0,4,0,6]
```

宽度位置 1、3 无贡献，必须为零。若主计算负责完整写回，也应得到同一结果。重复调用时仍执行相应初始化协议。

## 7 产出与校验

向 Tiling 和 Kernel 交付：统一的五维逻辑形状、三路原始格式与存储步幅、各轴属性、组内通道、有效核尺寸、dedy_expand 尺寸、反向边界、bias 记录及初始化条件。

- 两种进入场景得到相同问题；升维前后的元素数、步幅与计算关系一致。
- 正向形状公式、批次、组关系和 input_size 一致；有符号边界不被转成无符号大数。
- 转置卷积保留形状模式（属性推导/显式 input_size）、q 与 rho 的记录；input_size 全零哨兵正确替换。
- bias 存在性、FP32、长度 Cin、全局通道寻址记录完整。
- 从原始数据中选取元素，按统一轴恢复后仍指向同一数值。
- 用非零预填 dedx 检查无贡献位置、部分无贡献区域和重复执行；初始化由生成工程完成。

布局和窗口继续进入 [input/compute-flow.md](input/compute-flow.md)，资源和字段进入 [input/tiling/tiling.md](input/tiling/tiling.md)。

## 8 filter 分支的规格归一

### 8.1 对象与格式

以 x、dedy 为数值输入，dedw 为 FP32 输出；目标形状来自 filter_size 的**元素值**。从正向规格进入时，weight 只提供目标形状，数值不参与计算。

| 对象 | 统一逻辑轴 | 存储约定 |
|---|---|---|
| x | `(batch,Cin,Di,Hi,Wi)` | NCDHW 或 NDHWC |
| dedy / out_backprop | `(batch,Cout,Dout,Ho,Wo)` | 与 x 相同格式、相同 FP16/BF16 类型 |
| dedw / y | `(Cout,CinG,Dk,Hk,Wk)` | FP32；格式独立选择 NCDHW、NDHWC 或 DHWCN |
| filter_size | 目标 dedw 的物理 shape | 一维 5 元素整数张量；值按输出格式解释 |

filter_size 的三种轴序分别为 `(Cout,CinG,Dk,Hk,Wk)`、`(Cout,Dk,Hk,Wk,CinG)`、`(Dk,Hk,Wk,CinG,Cout)`。先解释轴再校验 Cout、CinG、核尺寸，不能把张量自身的 shape `(5,)` 当成 dedw 的形状。注册推导声明 filter_size 值依赖，输出 dtype 固定 FP32；直调包装传入等价的目标形状值。

### 8.2 属性、升维与形状

复用 §2.3 的空间属性展开与 §4.1 的正向尺寸公式，输入长度取 x，输出长度取 dedy。检查批次相同、G 整除 Cin/Cout、`dedw.Cout=Cout`、`dedw.CinG=Cin/G`，并核对每轴输出长度。

当前 filter 模型是普通正向卷积的权重梯度，接口 `output_padding` 缺省或显式全零，`enable_hf32=false`。保留属性的原始值和来源；带非零 output_padding 的转置卷积权重梯度需要独立的几何绑定。

原始低维规格按 §3 插入空间单维，x、dedy 与 dedw 核轴使用同一映射，新增轴取长度 1、stride/dilation 1、pad 0。**调用方已给出的五维规格保持原轴名**；例如以 D 轴承载的一维问题，不能重新解释为 W 轴问题。

例如原始 NWC 的 x=`(1,5,1)`、dedy=`(1,2,1)`，核宽 3、stride 2、pad 0：插入 D/H 单维后 x=`(1,1,1,5,1)`、dedy=`(1,1,1,2,1)`，NDHWC dedw=`(1,1,1,3,1)`。若原规格已是 x=`(1,5,1,1,1)`、dedy=`(1,2,1,1,1)`，则核沿 D，dedw=`(1,3,1,1,1)`；两种记录各自保留轴映射。

### 8.3 派生量与执行前置

| 派生项 | 交付内容 |
|---|---|
| 通道与核 | CinG、CoutG、各轴有效核长 `(核长-1)*dilation+1` |
| 输出网格 | ho×wo，batch×dout；核深度 kd 是输出任务轴 |
| 正向采样 | `id=od*strideD+kd*dilationD-padFront`，H/W 同理；越界 x 贡献为零 |
| 布局 | 原始格式、实际元素步幅、升维轴及逆映射 |
| 初始化 | MN_StreamK 的每段结果原子累加到 dedw；每次调用先将实际 dedw 全部置零，并完成清零到首次写回的依赖 |

filter 不生成 dedy_expand 或 weight_rot；x_win 与 dedy_flat 的布局由 [filter/compute-flow](filter/compute-flow.md) 定义。输出清零与 L0C 首次乘加初始化各自保留；无贡献核位置保持零。融合 AIV 清零和独立清零是两套完整执行协议，适配层选定其一后连同 Kernel 等待条件一起实现。

### 8.4 产出校验

产出保留原始与统一规格、filter_size 值及轴序、空间属性、分组关系、输出网格、实际存储步幅和初始化协议。检查：

- 五维目标形状与 filter_size 一致，所有正向形状关系成立；尺寸和地址乘积不溢出。
- 从原始 x/dedy 各取一个坐标，升维后的坐标指向同一数值；五维输入不二次换轴。
- 用 x=`[1,2,3,4,5]`、dedy=`[10,20]` 验证上述核宽 3 的例子，dedw=`[70,100,130]`，核顺序不翻转。
- 非零预填 dedw 后重复执行，结果不带上次残留；初始化由生成工程完成。

随后进入 [filter 场景](filter/scene.md)、[filter 切分](filter/tiling/tiling.md) 和对应语言适配层。

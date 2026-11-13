# 卷积反向权重梯度分支：场景与计算模型

本页建立权重梯度 dedw 的计算模型。规格归一见 [前置处理](../preprocessing.md)，共享记号见 [language.md](../language.md) 的 §13.1–§13.4；布局、任务与流水在 [compute-flow.md](compute-flow.md) 展开。

## 1 场景准入

本分支接受两种规格来源，最终统一为以 x、dedy 为数值输入、以 dedw 为输出的权重梯度问题：

| 场景 | 调用方提供 | 整理结果 |
|---|---|---|
| 从正向卷积参数生成权重梯度 | x、weight 的规格和 strides/pads/dilations/groups | 按正向公式确定 Y/dedy 规格，以 weight 的形状确定 dedw 与 filter_size |
| 直接给定权重梯度参数 | x、dedy、目标 dedw 或 filter_size 的规格及卷积属性 | 校验三者对应同一个正向卷积问题 |

生成阶段消费张量规格；执行阶段消费 x、dedy 的数值。weight 的原始数值不参与 dedw 计算。filter_size 按目标输出格式解读；它的值与 dedw 规格同时给出时应一致。1D/2D 按共享前置处理升维到 3D。

准入需要以下信息完整且相互一致：

- 批次数一致，groups 为正整数并整除 Cin、Cout，组内通道为 CinG、CoutG。
- x 与 dedy 同为 FP16 或同为 BF16，并同为 NCDHW 或同为 NDHWC；dedw 为 FP32，独立选择 NCDHW、NDHWC 或 DHWCN。
- 每轴空间形状满足正向卷积公式，核尺寸、stride、dilation 为正整数，正向两侧 pad 为非负整数。
- 当前计算模型采用 `output_padding=0`、`enable_hf32=false`；属性缺省值和调用方显式值一同记录。
- 目标平台与 SDK 能提供两路布局搬运、窗口提取、矩阵乘加、FP32 原子累加写回和初始化同步；资源由 platformInfo 与能力记录确定。

属性先按数学关系校验，再按 [Tiling](tiling/tiling.md) 和适配层检查指令范围。普通窗口与拆核窗口使用同一数学模型，具体参数与生成路径在求解后确定。

## 2 计算模型

### 2.1 统一符号

正向采用空间互相关约定，标量目标为 L：

$$
Y=\operatorname{Conv3D}(x,\mathrm{weight};\theta),\qquad
\mathrm{dedy}=\frac{\partial L}{\partial Y},\qquad
\mathrm{dedw}=\frac{\partial L}{\partial\mathrm{weight}}.
$$

逻辑形状使用以下轴义，实际存储顺序由各张量的 format 决定：

| 对象 | 逻辑轴 |
|---|---|
| x | `(batch,Cin,Di,Hi,Wi)` |
| dedy | `(batch,Cout,Dout,Ho,Wo)` |
| dedw | `(Cout,CinG,Dk,Hk,Wk)` |
| g、co、ci | 组号、组内输出通道、组内输入通道 |
| kd/kh/kw | 原正向核坐标，各自从 0 起 |
| od/oh/ow | dedy 的空间坐标，各自从 0 起 |

后续代码中 di/hi/wi、dout/ho/wo、dk/hk/wk 表示这些轴的长度；带 Idx 或下标的量表示坐标。矩阵 N 表示输出块列方向，批次数使用 batch。

### 2.2 乘加归约：从正向连接求 dedw

固定组、通道对和核位置，沿所有批次及输出空间位置累加：

$$
\mathrm{dedw}_g[co,ci,kd,kh,kw]
=\sum_{n,od,oh,ow}\mathrm{dedy}_g[n,co,od,oh,ow]\,
x_g[n,ci,\ od\,s_d+kd\,\delta_d-p_d^-,\
oh\,s_h+kh\,\delta_h-p_h^-,\
ow\,s_w+kw\,\delta_w-p_w^-].
$$

x 坐标越界时，该项取零。核坐标保持正向顺序，stride/dilation 直接作用于 x 的采样位置。先固定核位置，再遍历其所有连接，是本分支的基本计算过程。

例如只看宽度：`x=[1,2,3,4,5]`、`dedy=[10,20]`、核宽为 3、stride 为 2、pad 为 0。两个输出位置分别连接 x 的 `[1,2,3]` 和 `[3,4,5]`：

```text
+-----------------+----------+------+
| output position | x window | dedy |
+-----------------+----------+------+
| ow=0            | [1,2,3]  | 10   |
| ow=1            | [3,4,5]  | 20   |
+-----------------+----------+------+

kw=0: 10*1 + 20*3 = 70
kw=1: 10*2 + 20*4 = 100
kw=2: 10*3 + 20*5 = 130
```

结果为 `dedw=[70,100,130]`。这里每个核位置的输出是一个独立的内积，不能把核顺序倒置。

### 2.3 GEMM 映射

固定组 g、核深度 kd、批次 n 和输出深度 od。将输出高宽按 `k=oh*wo+ow` 展平：

- dedy_flat 的每一行对应一个 co，列对应输出位置 k。
- x_win 的每一行对应同一个 k，列对应当前 ci 与核高宽位置。
- 两者相乘得到该批深度对当前 dedw 块的贡献。

```text
dedy -> dedy_flat [M,K] --------------+
                                      |
x -> forward windows -> x_win [K,N] --+
                                      |
                                      |
                                      v
                    +-----------------+-----------------+
                    | L0C [M,N] = L0A [M,K] * L0B [K,N] |
                    +-----------------+-----------------+
                                      |
                                      |
                                      v
                  +-------------------+-------------------+
                  | contribution to dedw at fixed g / kd  |
                  +---------------------------------------+
```

| 方向 | 有效含义 | 内部选块长度 |
|---|---|---|
| M | 组内 dedy 通道 co | `mValue=alignUp(CoutG,16)` |
| N | 组内 x 通道与核高宽位置 | `nValue=alignUp(CinG,16)*hk*wk` |
| K | 一个 dedy 深度平面的输出高宽 | `kValue=ho*wo` |
| kd | 固定原核深度的任务轴 | `dk` 个切片 |
| batch×dout | K 之外的归约轴 | 在核内分段处理 |

N 的线性顺序为 `(ci1,kh,kw,ci0)`，ci0 长度为 16。任务的 singleCoreN 包含完整通道窗口；L0 的 baseN 是一次计算长度，普通整窗候选可一次或多次覆盖任务。发生非整窗分块时，按 compute-flow 中的核位置与通道状态继续寻址，不能将任务尺度与计算尺度混用。

### 2.4 视图定义与边界

| 视图 | 含义 | 实现位置 |
|---|---|---|
| dedy_flat | 当前输出平面或行带的 `[M,K]` 视图 | GM→L1A 布局转换，L1A→L0A 转置分形 |
| x_win | 按正向 stride/dilation/pad 提取的 `[K,N]` 窗口 | L1B 保存原 x 片段，L1B→L0B 提取窗口 |
| dedw 块 | 固定 g、kd 和 M/N 范围的梯度 | L0C 保存一个归约段的 FP32 部分和 |

两个操作数的同一 K 必须指向相同的 `(n,od,oh,ow)`。窗口视图不要求在 GM 物化；相邻窗口中的重复 x 元素可复用同一 L1 父段。

合法规格也可能出现无贡献核位置。例如 `wi=1、wk=3、padLeft=padRight=1、stride=1` 时，两个边缘核位置始终读取零，中心核位置读取唯一的 x 元素。另有运行时的无贡献深度段：固定 kd 后，该段所有 od 对应的输入深度均越界，整段可跳过。两种情况的输出均由零基值保持正确。

### 2.5 核为 1×1×1 的退化情形

三个核坐标固定为 0，x_win 退化为按 stride/pad 对齐的逐点采样。每个通道对的梯度仍需归约 `batch*dout*ho*wo` 个位置。

例如 `x=[1,2,3]、dedy=[4,5,6]`，两个张量的空间长度均为 3，stride=1、pad=0，则唯一的 dedw 为 32。多通道时同时计算多对通道的外积归约；长归约仍使用分段与 GM 累加。

### 2.6 分组

第 g 组只使用 x 的 `[g*CinG,(g+1)*CinG)` 通道和 dedy 的 `[g*CoutG,(g+1)*CoutG)` 通道。dedw 的 ci 始终是组内编号，co 的外部编号包含组起点。

例如 `Cin=Cout=4、G=2`：组 0 只连接通道 0、1，组 1 只连接通道 2、3。给组 1 的 x 加常量，组 0 的 dedw 应不变。内部尾通道补齐不增加数学分组或实际通道。

### 2.7 拆核窗口与分段累加

普通路径一次描述 H/W 窗口；isSplitKernelHW 路径按原 kh/kw 分别搬入采样行带，再使用单位核窗口提取。两者对应相同的 x 坐标，核位置顺序保持不变。

长归约拆成 batch×dout 段和 ho 段。每段在 L0C 完成自身全部贡献后，原子累加到同一 dedw 区域；新段建立新的 L0C 初值。阈值、段长与执行顺序见 compute-flow 第 7 节。

## 3 能力边界

| 维度 | 条件与约定 |
|---|---|
| 数学语义 | 正向零填充的卷积权重梯度，各组独立；越界 x 采样为零 |
| 形状 | filter_size、x、dedy 对应同一卷积问题；kernel 与属性合法性由共享前置处理及适配检查共同确定 |
| 数值与布局 | FP16/BF16 输入，FP32 累加和输出；x/dedy 格式配对，dedw 按目标格式恢复 |
| 执行 | 两路源数据及窗口能合法装载，部分和与原子写回能力可用；初始化先于全部写回 |
| 资源 | 基本块、父段、buffer 份数和物理地址同时满足平台与 API 限制 |
| 验证 | 独立直接求和与框架权重梯度对照，逐层检查布局、归约覆盖和输出边界 |

数值输出与逻辑布局是算法契约；API 宏、核类型、字段宽度和资源容量由所选语言适配层落实，不以固定芯片名称替代能力检查。

## 4 数据流 backbone

```text
dedw GM: initialize to zero before all segment writes

dedy -> dedy_flat -> L0A ----------------------------+
                                                     |
x -> forward pad + stride/dilation -> x_win -> L0B --+
                                                     |
                                                     |
                                                     v
                              +----------------------+----------------------+
                              | multiply-add: full reduction within segment |
                              +----------------------+----------------------+
                                                     |
                                                     |
                                                     v
                                      +--------------+--------------+
                                      | Fixpipe: atomic add to dedw |
                                      +--------------+--------------+
                                                     |
                                                     |
                                                     v
                           +-------------------------+-------------------------+
                           | next segment: same output, new L0C initial state  |
                           +---------------------------------------------------+
```

| 阶段 | 输入与输出 | 关键约定 |
|---|---|---|
| 前置处理 | x/dedy/filter_size → 统一几何 | 格式轴义、分组和形状一致 |
| 输出初始化 | dedw GM → 零基值 | 在任何原子累加写回前完成 |
| dedy 路 | dedy → dedy_flat → L0A | 同组 co 与输出网格配对 |
| x 路 | x → x_win → L0B | 正向边界、stride/dilation 与原核坐标配对 |
| 段内累加 | L0A×L0B → L0C | 首个有效乘加初始化，后续贡献累加 |
| 段末写回 | L0C → dedw | 仅写有效通道和核位置；归约段写同一结果区域 |

一个任务拥有固定 g、kd 和通道窗口范围。任务间输出互斥；同一任务的多段结果在 GM 合并。操作数就绪后才能计算，最后一次消费者完成后才可覆盖相应槽。

## 5 检查点

- [ ] 两种规格来源得到相同的 x/dedy/dedw 逻辑轴与属性。
- [ ] dedy_flat 的 K 列与 x_win 的 K 行对应同一输出位置。
- [ ] N 轴使用 `(ci1,kh,kw,ci0)`，任务与 L0 计算块分别记录。
- [ ] 固定 kd 的输入深度为 `od*strideD+kd*dilationD-padFront`。
- [ ] 组内 ci 与全局输入通道明确区分，组间贡献不混合。
- [ ] 无贡献采样、无贡献核位置和跳过段都有零结果语义。
- [ ] 每段独立初始化 L0C，段间通过已清零的 dedw 原子累加。

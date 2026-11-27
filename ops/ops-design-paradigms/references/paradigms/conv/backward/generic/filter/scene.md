# 卷积反向权重梯度分支：场景与计算模型

本页建立权重梯度 dedw 的计算模型。规格归一见 [前置处理](../preprocessing.md)，共享记号见 [language.md](../language.md)，分支记号见 [filter/language.md](language.md) §1–§4；布局、任务与流水在 [compute-flow.md](compute-flow.md) 展开。

## 1 场景准入

本分支接受两种规格来源，最终统一为以 x、dedy 为数值输入、以 dedw 为输出的权重梯度问题：

| 场景 | 调用方提供 | 整理结果 |
|---|---|---|
| 从正向卷积参数生成权重梯度 | x、weight 的规格和 strides/pads/dilations/groups | 按正向公式确定 Y/dedy 规格，以 weight 的形状确定 dedw 与 filter_size |
| 直接给定权重梯度参数 | x、dedy、目标 dedw 或 filter_size 的规格及卷积属性 | 校验三者对应同一个正向卷积问题 |

生成阶段消费张量规格；执行阶段消费 x、dedy 的数值。weight 的原始数值不参与 dedw 计算。filter_size 按目标输出格式解读；它的值与 dedw 规格同时给出时应一致。1D/2D 按共享前置处理升维到 3D。

准入需要以下信息完整且相互一致：

- 批次数一致，groups 为正整数并整除 Cin、Cout，组内通道为 CinG、CoutG。
- x 与 dedy 同为 FP16、BF16 或 FP32，并同为 NCDHW 或同为 NDHWC；dedw 为 FP32，独立选择 NCDHW、NDHWC 或 DHWCN。
- 每轴空间形状满足正向卷积公式，核尺寸、stride、dilation 为正整数，正向两侧 pad 为非负整数。
- 普通权重梯度使用 output_padding=0；源码接口中的非零 output_padding 还涉及转置卷积的形状绑定，不能仅检查取值范围后套用本页的普通卷积模型与 Golden。
- FP32 的 enable_hf32 决定 hf32Flag；MN_StreamK 也消费此标记。Winograd/DLoad 还需各自的平台门禁，hf32 是计算模式，输入输出仍按 FP32 存储。
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

以 H/W 平面为例直观理解（stride=1、单位 dilation 示意，·为 pad 零填充）：

```
x（pad 后）                 正向核窗口（2×2）
+---------------------+
| .  .  .  .  .  .  . |     +---------+
| .  a  b  .  .  .  . |     | k00 k01 |
| . [c  d] e  .  .  . |     | k10 k11 |
| . [f  g] .  .  .  . |     +---------+
+---------------------+     窗口按正向 stride 滑动；
                            [c,d] 与下排同列是 (oh,ow) 的采样窗
dedw[co,ci,kh,kw] += dedy(n,co,od,oh,ow) × x(oh·s_h+kh·δ_h−p_h⁻, ow·s_w+kw·δ_w−p_w⁻)
```

```text
固定 (g,co,ci,kd,kh,kw)：

每个输出位置 (n,od,oh,ow)
  -> 该位置的 dedy 值
  -> 对应正向采样的 x 值
  -> 乘积 -> 累加到同一个 dedw 元素
```

每个原核位置对应独立的归约，核位置始终保持正向顺序。

### 2.3 GEMM 映射

固定组 g、核深度 kd、批次 n 和输出深度 od。将输出高宽按 `k=oh*wo+ow` 展平：

- dedy_flat 的每一行对应一个 co，列对应输出位置 k。
- x_win 的每一行对应同一个 k，列对应当前 ci 与核高宽位置。
- 两者相乘得到该批深度对当前 dedw 块的贡献。

```
dedy (N,Cout,Dout,Ho,Wo)              x (N,Cin,Di,Hi,Wi)                     dedw (Cout,CinG,Dk,Hk,Wk)
    |                                     |                                      ^
    | 展平输出平面（dedy_flat）           | 正向 pad + stride/dilation           | FP32 输出
    v                                     v（x_win，LOAD3D 窗口化）              |
A 矩阵 [M, K]                     ×   B 矩阵 [K, N]                      =   Y 矩阵 [M, N]
行 = 组内 co                          列 = (ci1,kh,kw,ci0)                   行×列 = 固定 g/kd 的
列 = 展平的 (oh,ow)                   行 = 同一 (oh,ow)                      一个 dedw 输出块
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

合法规格也可能有始终落在 pad 区的核位置，其梯度为零。固定 kd 后，若一个归约段的全部 od 都对应无效输入深度，该段可跳过；只有部分无效时仍归约其余有效批深度。两种情况均由输出零基值保持正确。

### 2.5 核为 1×1×1 的退化情形

三个核坐标固定为 0，x_win 退化为按 stride/pad 对齐的逐点采样。每个通道对的梯度仍需归约 `batch*dout*ho*wo` 个位置。

多通道时同时计算多对通道的外积归约；核为单位核也不改变长归约分段与 GM 累加规则。

### 2.6 分组

第 g 组只使用 x 的 `[g*CinG,(g+1)*CinG)` 通道和 dedy 的 `[g*CoutG,(g+1)*CoutG)` 通道。dedw 的 ci 始终是组内编号，co 的外部编号包含组起点。

改变某一组的输入，不应影响其他组的 dedw。内部尾通道补齐不增加数学分组或实际通道。

### 2.7 拆核窗口与分段累加

普通路径一次描述 H/W 窗口；isSplitKernelHW 路径按原 kh/kw 分别搬入采样行带，再使用单位核窗口提取。两者对应相同的 x 坐标，核位置顺序保持不变。

长归约拆成 batch×dout 段和 ho 段。每段在 L0C 完成自身全部贡献后，原子累加到同一 dedw 区域；新段建立新的 L0C 初值。阈值、段长与执行顺序见 compute-flow 第 7 节。

### 2.8 Winograd 等价模型（F(2,3) 变换域求导）

3×3 的 H/W 核、stride/dilation=1 的规格可使用 F(2,3) 变换域求导；完整准入见 [Winograd 模板](tiling/templates/winograd/tiling.md)。输出按 2×2 tile 划分，每 tile 对应 x 的 4×4 感受野。固定 co、ci、kd，t 遍历 batch、od 及输出 tile：

$$
\mathrm{dedw}[co,ci,kd,:,:]
= T_w\!\left(\left\{\sum_t
T_{\mathrm{dedy}}(\mathrm{dedy}_t)[p]\,
T_x(x_{t,kd})[p]\right\}_{p=0}^{15}\right).
$$

T_dedy 与 T_x 是两种不同的正变换，T_w 将 16 个变换坐标还原为 3×3 梯度。p 是变换坐标，不称为频率。od 折叠进归约，kd 保持输出核轴；x tile 的输入深度为 `id=od+kd-padFront`。

```text
固定 co / ci / kd：
x 的 4x4 tile ------> T_x ------+
                               |
                               +--> 16 点分别乘加 --> T_w --> 3x3 dedw
                               |
dedy 的 2x2 tile --> T_dedy ----+
                         沿所有 (batch,od,tile) 归约
```

对完整单 tile，直接法每通道对需要 4×9=36 次乘法，变换域为 16 次，即 4/9；正逆变换、搬运与边缘开销另计，这一比例不是整体加速比。当前实现仅放通 FP32，hf32 与 kd=3 各有平台能力条件。

变换域求和与逆变换在线性关系下等价于 §2.2。设备还会分段：段内累加 16 点，段末逆变换，首段覆写结果、后续段原子合并；不是整次调用只做一次逆变换。独立 Golden 仍用直接求和。

D 轴边缘及尾 tile 仍有无效采样。无效 D 步保持跨核握手；全无效段向逆变换提供零输入。尾轮切 K 后先写部分和 workspace，再聚合到 dedw，不能沿用 MN_StreamK 的清零与树形归约协议。执行、排布和同步图见 [Winograd 实现](../../adapters/ascendc/filter/winograd-implementation.md)。

## 3 能力边界

| 维度 | 条件与约定 |
|---|---|
| 数学语义 | 正向零填充的卷积权重梯度，各组独立；越界 x 采样为零 |
| 形状 | filter_size、x、dedy 对应同一卷积问题；kernel 与属性合法性由共享前置处理及适配检查共同确定 |
| 数值与布局 | FP16/BF16/FP32 输入、FP32 输出；格式与 hf32 按所选模板准入，dedw 按目标格式恢复 |
| 执行 | 搬运、计算和同步满足所选模板；MN_StreamK 原子段写回前清零，Winograd 首段覆写、后段合并 |
| 资源 | 基本块、父段、buffer 份数和物理地址同时满足平台与 API 限制 |
| 验证 | 独立直接求和与框架权重梯度对照，逐层检查布局、归约覆盖和输出边界 |

数值输出与逻辑布局是算法契约；API 宏、核类型、字段宽度和资源容量由所选语言适配层落实，不以固定芯片名称替代能力检查。

## 4 数据流 backbone

以下图示为 MN_StreamK 主轮 im2col 路径。尾轮切 K 见 [MN_StreamK](tiling/templates/mn-streamk/tiling.md#6-确定性归约与-workspace)，Winograd 使用 §2.8 的独立变换流水。

```
dedy --展平输出网格--> dedy_flat --> L0A -----+
                                              |
x --正向窗口--> x_win --> L0B ----------------+--> L0C --Fixpipe--> 原子累加 --> dedw GM

前置：dedw GM 预清零（在全部段写回之前完成）。
段内乘加归约（完整归约）；下一段：同一输出区域，新的 L0C 初值。
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
- [ ] MN_StreamK 各段独立初始化 L0C，主轮合并到已清零 dedw；尾轮遵守独立 workspace 协议。
- [ ] Winograd 等价模型与直接求和逐元素相等可手工验证（小规模单 tile 例子）；两模型互斥、由 tiling 准入二选一。

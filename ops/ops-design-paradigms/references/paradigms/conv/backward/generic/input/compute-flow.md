# 卷积反向算子计算流程

> **层级**：通用知识层（generic/input/），语言中立。以达芬奇架构 Cube 通路为基准描述。
>
> **分工**：本文件回答"**数据怎么流、任务怎么分、循环怎么写、同步怎么插**"。算什么（数学模型与视图定义）见 [scene.md](scene.md)；切分大小如何决策见 [tiling/tiling.md](tiling/tiling.md)。
>
> 术语记号见 [../language.md](../language.md)（公共）与 [language.md](language.md)（本分支）。dedy_expand、fmap、weight_rot 是逻辑视图；L1A/L1B、L0A/L0B、L0C 是实际片上 Buffer。轴序均从慢到快，小分形用于展示元素顺序；具体 API 字段由语言适配层落实。

> 下文逐级父段循环以普通三模板为主；kernel-split 的差异在第 8.4 节汇总。完整 N/K 驻留复用公共父段状态机，L1A 按当前 M 块的局部窗口更新。
>
> Conv3DTranspose 完成自身形状处理后（`x→dedy、filter→weight、y→dedx` 绑定）复用本页布局、窗口、任务和流水；output_padding 与 bias 对流程的影响在对应章节给出。真实核长和 K 归约长度始终由 weight 决定。

---

## 1 计算模型速览

### 1.1 从正向连接得到 dedx

固定批次和组。dedx 的一个位置由同组 dedy 通道和卷积核位置共同贡献：

$$
\mathrm{dedx}_g[n,ci,\mathbf i]
=\sum_{co,\mathbf k} \mathrm{weight}_g[co,ci,\mathbf k]\,
\mathrm{dedy}_g\!\left[n,co,
\frac{\mathbf i+\mathbf p^- -\mathbf k\odot\boldsymbol\delta}{\mathbf s}\right].
$$

只累加分子各轴非负、能被 stride 整除且 dedy 坐标在界内的项。这个直接求和式用于独立 Golden。实现采用等价的插零视图 dedy_expand、边界视图 fmap 和反向权重 weight_rot：窗口按单位 stride 移动，保留 dilation。



```text
dedy --stride 插零--> dedy_expand --反向边界--> fmap -----+
                                                          |
weight --rot180 + 通道角色交换--> weight_rot -------------+--> 窗口归约 --> dedx
```

img2col 把每个 dedx 位置的 fmap 窗口展成矩阵的一行。窗口提取随 L1→L0A 完成，GM 中仍保存原始 dedy；视图定义见 [scene](scene.md)。

---

## 2 数据流总览

### 2.1 阶段与指令职责

| 阶段 | 通路 | 完成的转换 |
|---|---|---|
| LoadToA1 / MTE2 | dedy GM→L1A | 按原格式寻址、通道分块、保留 stride 插零间隔 |
| LoadToB1 / MTE2 | weight GM→L1B | 选择原核深度，形成所选父布局 |
| LoadToA2 / MTE1 | L1A→L0A | fmap 边界、H/W img2col、NZ 分形 |
| LoadToB2 / MTE1 | L1B→L0B | 核位置逆序、按需转置分形，形成 ZN |
| MmadLocal / M | L0A×L0B→L0C | FP32 部分和，完整 K 与核深度归约 |
| LoadL0c2Gm / F | L0C→dedx GM | NZ2DN/NZ2ND 还原 + 按 dedx dtype 转换（FP16/BF16 做 FP32→窄精度；FP32 直出无转换）、有效范围写回 |
| bias 装载（可选） | bias GM→L1→BT | FP32 通道段搬入 BT，首次乘加作 L0C 初值 |

指令的语言中立语义、单位与地址公式见第 5.1 节；各语言的 API 实体与参数结构体见 [指令映射（Ascend C）](../../adapters/ascendc/input/instruction-mapping.md) / [DSL](../../adapters/dsl/input/instruction-mapping.md)；bias 通路的装载与 MMAD 细节见 [bias 通路（Ascend C）](../../adapters/ascendc/input/bias-bt-path.md) / [DSL](../../adapters/dsl/input/bias-bt-path.md)。两路的 K 位置一致；改变某一父布局时，同时重算子块定位。

各阶段使用的长度口径如下，实际字段由适配层映射：

| 阶段 | 有效范围 | 内部跨度与寻址 |
|---|---|---|
| GM→L1A/L1B | 本段实际 dedy、dedx 通道及原张量区域 | 目的通道按 C0 补齐，源步幅保留总通道和完整深度 |
| L1A→L0A | `baseUseM_、baseUseK_` 个窗口行与内部 K 项 | M 方向按 `mCompute` 排布；K 起点为 `kLocalA` |
| L1B→L0B | 当前 dedx 通道与相同的 `baseUseK_` | 按 `nCompute` 排布；K 起点为 `kLocalB`，沿实际父布局取块 |
| MMAD | n 取 `baseUseN_`；普通 k 取 `baseUseK_`，C04 尾 k 向 c0 对齐 | m 在任务末 M 块取 `mCompute`，其余按当前有效 M；L0C 保持完整归约 |
| F | `mSize=baseUseM_、nSize=baseUseN_` | 源跨度取 `mCompute`，目的步幅取原 dedx 步幅 |

`baseUseK_` 按 Kernel 的 `tmpSingleCoreK` 截取；普通布局为对齐通道数乘当前核高宽，C04 为 `4*singleShapeHWk_`。Host 的 C04 `kValue` 还向 C0 对齐，用于选块与预算，不能直接替代运行终点。Load3D.kExtension 保留实际尾长，C04 的 MMAD.k 单独对齐；补齐位置贡献为零。各长度定义见 [分支记号](language.md#2-mnk-三元组gemm-映射)。

### 2.2 dedy 侧：GM → L1A → L0A

```
dedy(GM, NCDHW/NDHWC)
  │  DN2NZ / ND2NZ 布局转换搬运（随路 ND→NZ；目的步幅保留 stride 插零间隔）
  ▼
L1A [Cout1, curHoSize, woExpand, Cout0]    <-- dedy_expand 驻留行带（未做滑窗展开！）
  │  LOAD3D（随路完成反向边界 + 滑窗展开 = 硬件 im2col，窗口单位步长）
  ▼
L0A [K1, M1, M0, K0]                       <-- 展开后的 fmap 窗口分形
     其中 K 沿 (Cout1→r_h→r_w→Cout0) 展平，M1 = CeilDiv(baseUseM_, M0)
```

要点：

1. **L1 里存的是 dedy_expand 的驻留片段**（一段行带 × 若干通道分形），不是展开后的矩阵——展开发生在 L1→L0A 的 Load3D 里；
2. 驻留行带按"从 dedx 块反推 dedy 行带"的交集计算确定；负反向边界先裁剪真实 dedy，剩余补边由窗口参数表达；
3. stride 插零不搬全零行：Fill 预置零 + 目的步幅保留间隔，真实行按相位写入。

### 2.3 weight 侧：GM → L1B → L0B

```text
weight(GM, NCDHW/NDHWC)
  │  MTE2 装载（选定原核深度 kd 与父布局；GM→L1 不展开、不逆序）
  ▼
L1B 父布局三选一（见 §5.3）    <-- 原核顺序的权重父段
  │  LOAD2D（H/W 核空间逆序 + 按需转置，形成 ZN 分形）
  ▼
L0B [K1, N1, N0, K0]           <-- weight_rot[K,N]：K 与 L0A 同序配对
```

普通路径中，weight 的 GM→L1 保留原核坐标，核高宽逆序发生在 L1→L0B 的 Load2D；C04 则在 AIV gather 时提前完成该逆序。核深度配对由外层采样检查完成。每个空间轴的反向关系只表达一次。

可选前置支路：Host 启用前置权重转置时，GM→L1B 的源不是原始 GM weight，而是 AIV 预先整理过的 workspace 副本；C04 与 group enlarge 态由 AIV pass 直写 L1B（B 路来源四态，见 [5.3](#53-weightgml1bl0b)）；归约语义不变；VecTranspose 改变源地址与步幅，C04 改变两路 K 排列，group enlarge 改变计算组与 N/K 尺度。

### 2.4 输出侧：MMAD → L0C → dedx GM

```text
L0A [K1,M1,M0,K0] -----------+
                             |
L0B [K1,N1,N0,K0] -----------+
                             |
BT（bias，仅转置卷积）-------+
                             |
                             v
     +-----------------------+-----------------------+
     | MMAD -> L0C [N1,M1,M0,N0]（FP32 累加）        |
     | FIXPIPE: NZ2DN/NZ2ND + dtype（FP32 直出）     |
     | dedx(GM, NCDHW/NDHWC，只写有效 M/N)           |
     +-----------------------------------------------+

bias 只进 L0C 初值。
```

要点：同一个输出块的所有 K 与有效核深度累加到同一 L0C；完整归约后一次 Fixpipe 写回，只写有效 M/N，保留原张量全局步幅。

### 2.5 两路数据流总图

```text
C 路（dedx）：bias 经 BT -> L0C 初值，仅转置卷积

A 路（dedy -> fmap）                          B 路（weight -> weight_rot）
+-----------------------------------+         +-----------------------+
| GM -> L1A（dedy_expand 驻留行带） |         | GM -> L1B（权重父段） |
+-----------------+-----------------+         +-----------+-----------+
                  | LOAD3D 单位步长窗口                   | LOAD2D 逆序/转置
                  v                                       v
+-----------------+-----------------+         +-----------+-----------+
| L0A（NZ 分形）                    |         | L0B（ZN 分形）        |
+-----------------+-----------------+         +-----------+-----------+
                  |                                       |
                  +-------------------+-------------------+
                                      |
                                      v
                       +--------------+--------------+
                       | MMAD -> L0C（FP32 部分和）  |
                       +--------------+--------------+
                                      | FIXPIPE / 有效 M/N 写出
                                      v
                               +------+------+
                               | GM（dedx）  |
                               +-------------+
```

L1A 保存可复用的图像行带，L0A 才展开窗口；L1B 保存当前父布局，L0B 组织计算所需的 weight_rot。C04、group enlarge、VecTranspose 的准备方式见第 5.3 节。

---

## 3 数据格式与存储轴序

先看原张量和各级 Buffer 的轴序，再展开窗口与搬运关系：

| 对象 | 外部格式与轴序（慢到快） | 说明 |
|---|---|---|
| dedy/dedx | NCDHW：`(N,C,D,H,W)`；NDHWC：`(N,D,H,W,C)` | 两种通道位置 |
| weight | NCDHW：`(Cout,CinG,Dk,Hk,Wk)`；NDHWC：`(Cout,Dk,Hk,Wk,CinG)`；DHWCN 推导视角：`(Dk,Hk,Wk,CinG,Cout)` | 权重格式中的 N 对应 Cout，C 对应 CinG |
| L1A | `(Cout1,curHoSize,woExpand,Cout0)` | 一个深度的 dedy_expand 驻留片段 |
| L0A / L0B / L0C | `(K1,M1,M0,K0)` / `(K1,N1,N0,K0)` / `(N1,M1,M0,N0)` | M0/N0=16；K0 为 FP16/BF16 的 16、FP32 的 8 |

**格式转换全景**（GM 原始排布 → L1 驻留布局 → L0 计算分形 → 输出还原）：

```text
dedy / NCDHW -- DN2NZ --+
                        +--> L1A 通道分块 --Load3D--> L0A / NZ
dedy / NDHWC -- ND2NZ --+

weight / GM --> 选定 L1B 父布局 --Load2D--> L0B / ZN

L0A × L0B --> L0C / NZ --+-- NZ2DN --> dedx / NCDHW
                         |
                         +-- NZ2ND --> dedx / NDHWC
```

- dedy 侧的布局转换（DN2NZ/ND2NZ）在 GM→L1 随路完成，滑窗展开（img2col）延迟到 L1→L0A；
- weight 侧 GM→L1 不展开不逆序，逆序/转置在 L1→L0B 完成；
- 输出按 dedx 格式选择 NZ2DN（NCDHW）/NZ2ND（NDHWC）随路还原。

### 3.1 NCDHW、NDHWC 与 DHWCN

轴序从最慢到最快变化。一般特征坐标为 `(n,c,d,h,w)`，连续张量的元素偏移由轴序逐层展开：

| 对象与格式 | 存储轴序 | 连续方向 |
|---|---|---|
| dedy/dedx，NCDHW | `(N,C,D,H,W)` | 走完一个通道的空间体，再换通道 |
| dedy/dedx，NDHWC | `(N,D,H,W,C)` | 走完一个空间点的通道，再换空间点 |
| weight，NCDHW | `(Cout,CinG,Dk,Hk,Wk)` | 核宽连续 |
| weight，NDHWC | `(Cout,Dk,Hk,Wk,CinG)` | 组内 dedx 通道连续 |
| weight，DHWCN 推导视角 | `(Dk,Hk,Wk,CinG,Cout)` | dedy 通道连续 |

DHWCN 用于比较排布；生成接口按入口约定使用 weight 格式。权重格式中的 N 对应 Cout，C 对应 CinG。

```text
特征张量（固定 batch）：
NCDHW: [c: d -> h -> w] [下一 c: d -> h -> w] ...
NDHWC: [d,h,w: c ->]    [下一空间点: c ->] ...

权重（固定 group）：
NCDHW: (co,ci,kd,kh,kw)    核宽连续
NDHWC: (co,kd,kh,kw,ci)    dedx 通道连续
DHWCN: (kd,kh,kw,ci,co)    dedy 通道连续
```

格式转换改变存放位置；核逆序改变反向窗口读取的原核坐标，两者分别追踪。

特征元素偏移为：

$$
\begin{aligned}
\mathrm{off}_{\mathrm{NCDHW}} &= ((((nC+c)D+d)H+h)W+w), \\
\mathrm{off}_{\mathrm{NDHWC}} &= ((((nD+d)H+h)W+w)C+c).
\end{aligned}
$$

组内 ci/co 先加组起点，再使用原张量总通道及完整空间步幅。NDHWC 相邻空间点仍隔总通道数，NCDHW 相邻通道仍隔完整 D×H×W。

### 3.2 ND、DN、NZ 与 ZN

先固定同一个逻辑矩阵 `T[R,S]`。ND 沿列连续，DN 沿行连续；NZ 在分形之间按列走、分形内按行走，ZN 在分形之间按行走、分形内按列走。N/D 记号与批次、卷积深度无关。

```text
ND: 列连续（通道在内）              DN: 行连续（通道在外）
+-----+-----+-----+                 +-----+-----+-----+
| T00 | T01 | ... |                 | T00 | T10 | ... |
+-----+-----+-----+                 +-----+-----+-----+
| T10 | T11 | ... |                 | T01 | T11 | ... |
+-----+-----+-----+                 +-----+-----+-----+
内存序: T00,T01,...,T10,T11,...     内存序: T00,T10,...,T01,T11,...
```

将行、列拆成块号 `(r1,s1)` 和块内坐标 `(r0,s0)`；分形大小为 bR×bS，块数为 `R1=ceilDiv(R,bR)`、`S1=ceilDiv(S,bS)`。

| 格式 | 慢到快轴序 | 连续方向 |
|---|---|---|
| ND | `(R,S)` | 列元素 |
| DN | `(S,R)` | 行元素 |
| NZ | `(S1,R1,bR,bS)` | 分形内的列元素 |
| ZN | `(R1,S1,bS,bR)` | 分形内的行元素 |

```text
分形网格；F00/F01/F10/F11 是块坐标，不是数据值：
+---------+---------+
| F00     | F01     |
+---------+---------+
| F10     | F11     |
+---------+---------+

NZ: F00 -> F10 -> F01 -> F11；块间按列走、块内按行走
ZN: F00 -> F01 -> F10 -> F11；块间按行走、块内按列走

pad (R,S) -> 拆 (R1,bR,S1,bS) -> 重排 -> 目的布局
```

布局转换保持元素的数学坐标。紧致存储的偏移为：

$$
\begin{aligned}
\mathrm{off}_{NZ}&=((s_1R_1+r_1)b_R+r_0)b_S+s_0,\\
\mathrm{off}_{ZN}&=((r_1S_1+s_1)b_S+s_0)b_R+r_0.
\end{aligned}
$$

真实搬运可同时完成补齐、拆轴和重排；带间隔布局使用实际步幅。先说明矩阵的行列含义，再选择转换方向。

本分支固定 dedx 深度，以 HW 为行、CinG 为列时，NCDHW 是 DN、NDHWC 是 ND；L0A 的行列为 M/K，L0B 为 K/N，L0C 为 M/N。各对象的五维映射和具体分形轴序见后两节。

### 3.3 五维张量与片上视图

NCDHW 固定 `(n,d)` 后，以 H×W 为矩阵行、C 为列，原存储是 DN 视角；NDHWC 是 ND 视角。若为整个五维张量定义分形视图：

| 视图 | 慢到快轴序 | 说明 |
|---|---|---|
| 固定深度、HW×C 的 NZ | `(N,D,C1,HW1,M0,C0)` | 各深度平面分别分形 |
| 固定深度、HW×C 的 ZN | `(N,D,HW1,C1,C0,M0)` | 同一矩阵的另一朝向 |
| DHW×C 合轴 NZ | `(N,C1,DHW1,M0,C0)` | 空间体整体分形，与逐深度视图分别使用 |
| 通道分块 NDC1HWC0 | `(N,D,C1,H,W,C0)` | 每个空间点保留一组 C0，不把它等同于完整矩阵分形 |

实际 L1A 为一个深度的 dedy_expand 驻留片段，轴序 `(Cout1,curHoSize,woExpand,Cout0)`。L0A 的轴序为 `(K1,M1,M0,K0)`，L0B 为 `(K1,N1,N0,K0)`，L0C 为 `(N1,M1,M0,N0)`。M0/N0 取 16，K0 随 dtype 取 16 或 8。

Cout1/Cin1 是 dedy/dedx 通道分形轴，Cout0/Cin0 是块内轴；M1/N1/K1 是矩阵分形轴，M0/N0/K0 是对应块内轴。

### 3.4 L0A、L0B 与 L0C 的计算分形

L0A/L0B 用同一 K 配对（K0 随 dtype：FP16/BF16=16、FP32=8），L0C 的行列分别继承 dedx 空间和 dedx 通道：

```text
+-----------------+----------+----------------+---------------+
| Buffer          | 逻辑分形 | 分形内连续方向 | 物理轴序      |
+-----------------+----------+----------------+---------------+
| L0A / NZ        | M0 x K0  | K0（行方向）   | (K1,M1,M0,K0) |
| L0B / ZN        | K0 x N0  | K0（列方向）   | (K1,N1,N0,K0) |
| L0C / NZ / FP32 | M0 x N0  | N0（行方向）   | (N1,M1,M0,N0) |
+-----------------+----------+----------------+---------------+

L0A / NZ:       L0B / ZN:      L0C / NZ:
K0 ->           N0 ->          N0 ->
+---------+     +--+--+--+     +---------+
| ----->  |     |  |  |  |     | ----->  |
| ----->  |     | v| v| v|     | ----->  |
|  ...    |     |  |  |  |     |  ...    |
+---------+     +--+--+--+     +---------+

L0C[M,N] += L0A[M,K] * L0B[K,N]
```

L0A 的逻辑分形为 M0×K0，L0B 为 K0×N0；物理最内轴分别为 `(M0,K0)`、`(N0,K0)`。FP16/BF16 的输入分形为 16×16，FP32 为 16×8，均占 512B。L0C 为 16×16 个 float，占 1024B。

---

## 4 img2col 与窗口归约

### 4.1 M/N/K 与窗口归约顺序

固定 `(n,g,curDinIdx_)`，M 展开 dedx 高宽，N 方向取组内 dedx 通道；每个核深度片独立形成一次矩阵乘。归约顺序为 `Cout1 → r_h → r_w → Cout0`，其中 `co=co1*C0+Cout0`：

```text
M: m = h_i*Wi+w_i
N: ci
K: kLinear = ((co1*Hk+r_h)*Wk+r_w)*C0+co0

L0A[m,:]  = fmap window for one dedx spatial position
L0B[:,ci] = weight_rot for one dedx channel
L0C[m,ci] += L0A[m,K] * L0B[K,ci]
```

L0A 的 K 列与 L0B 的 K 行必须对应同一个 dedy 通道和逆序核位置。有效归约项为 `coutG*hk*wk`；计算 K 使用对齐通道，补齐项为零。核深度不混入这个 K。

上述顺序适用于普通 16 通道窗口。C04 将每个核位置的通道容量改为 4，A 路 Load3D 与 B 路 gather 必须使用配套的新 K 排列，不能声称物理遍历顺序不变；示意见 [scene](scene.md#32-c04四通道窗口参与-k-排列)。group enlarge 使用合并后的计算组通道，数学原始组之间仍通过对角权重隔离。

### 4.2 前放大与边界视图

先按每个空间轴的 stride 将真实 dedy 放到展开坐标，采样点之间为零；再按有符号反向边界补零或裁剪，得到 fmap。

```text
dedy 轴:      [o 处的值]                  [o+1 处的值]
                   |                            |
                   | stride 间隔                |
dedy_expand:     [值]     [零间隙 ...]     [下一个值]
                   |
                   | 有符号边界映射
                   v
fmap:          [前侧补零/裁剪] [展开采样点] [后侧补零/裁剪]
                   |
                   | 单位步长窗口（保留原 dilation）
                   v
dedx 位置:     每个目标位置取一个窗口
```

三维的 D/H/W 独立采用上述映射。dedy_expand 的轴长为 `(dedy轴长-1)*stride+1`；fmap 的前侧边界决定采样起点，后侧边界和目标形状决定终点。正向 stride 已体现在插零间隔里，Load3D 不再重复使用它。

L1A 只保存当前深度的必要行带。整行插零和边界零分别由准备步骤与窗口参数实现，不要求在 GM 物化整个 fmap。

### 4.3 三维 rot180 与核深度配对

rot180 在三个空间轴上逆序，并交换组内通道的角色。普通路径保留原 kd，通过深度采样反解配对 dedy，再由 Load2D 逆序 H/W；这与先构造完整 weight_rot 的数学关系相同。

```text
原 weight（固定 co,ci）:
  深度 kd:                 H/W 坐标 (kh,kw)
      |                         |
      | 深度逆序                | H/W 逆序
      v                         v
  深度 rd=dk-1-kd:         (rh,rw)=(hk-1-kh,wk-1-kw)

weight[co,ci,kd,kh,kw] -> weight_rot[ci,co,rd,rh,rw]

实现侧深度配对：
  dTmp = curDinIdx_ + padFront - kd*dilationD
  有效当且仅当 dTmp >= 0, dTmp % strideD == 0, dTmp/strideD < dout
```

无效深度配对不提交搬运或计算；有效深度的部分和继续保留在当前 L0C。C04 的 H/W 逆序在前置 gather 中完成，各空间轴的反向关系只表达一次。

### 4.4 LOAD3D（L1A→L0A，硬件 im2col）

**im2col 原理**——把滑窗展开写成矩阵：每个 dedx 位置的感受野展平成 A 矩阵的一行：

```
L1A 中的 fmap 行带（curHoSize × woExpand）   感受野(2×2)     L0A 上的 A 矩阵（M × K）
+---------------------+                      -- im2col -->   行 = dedx 空间位置 m
| .  .  .  .  .  .  . |                                      列 = (co, r_h, r_w) 展平
| . [.  .] .  .  .  . |                                      +--------+-----------------------------------+
| . [.  .] .  .  .  . |                                      | 窗口行 | 展开后的元素                      |
| .  .  .  .  .  .  . |                                      +--------+-----------------------------------+
+---------------------+                                      | m      | fmap(m,co,0,0) fmap(m,co,0,1) ... |
窗口沿 W、H 以单位步长滑动                                   | m+1    | fmap(m+1,co,0,0) ...              |
                                                             +--------+-----------------------------------+
```

图中按 fmap 逻辑坐标显示窗口，L1A 实际只存 dedy_expand 驻留行带，边界零由参数表达。LOAD3D 一条指令完成"取窗 → 展开 → 摆成 M0×K0 分形"三件事。

**相邻窗口的元素复用**（单位 stride、单位 dilation、单通道小图；下一窗口右移一列，与前窗口共享一列元素）：

```text
L1A / fmap 行带       L0A / NZ：同一通道的选中元素
+---------------+     行 = dedx 空间 m；列标签 = (r_h,r_w)
| .  .  .  .  . |     +--------+-------+-------+-------+-------+
| . [a  b] e  . |     | window | (0,0) | (0,1) | (1,0) | (1,1) |
| . [c  d] f  . |     +--------+-------+-------+-------+-------+
| .  .  .  .  . |     | 当前 m | a     | b     | c     | d     |
+---------------+     | 下一 m | b     | e     | d     | f     |
窗口 -> 下一窗口      +--------+-------+-------+-------+-------+
                      完整 K 序：(Cout1,r_h,r_w,Cout0)
                      逻辑窗口 -> 物理分形
```

实际指令保留规格中的 dilation，同一核位置的 Cout0 连续。L0A 的 K 列与 L0B 的 K 行必须配对到相同的通道和逆序核位置。

配置分两组（位域下发）：

| 配置组 | 内容 | 语义 |
|---|---|---|
| 窗口几何 | 当前 L1A 行带高宽、剩余反向补边、hk/wk、dilation、channelSize | "在哪张行带上取什么窗口" |
| 块定位 | mStartPt、kStartPt、mExtension、kExtension、目的分形步幅 | "这次展开出哪个 [M0,K0] 子块" |

**块定位参数的精确语义**（单位错误是滑窗类 bug 的高发点）：

| 参数 | 单位/语义 | 取值规则 |
|------|----------|---------|
| mStartPt | **dedx 行内列号**（本 M 块首个窗口对应的输出 w 坐标，`= mStart mod wi`），硬件按行带行宽折行定位。**不是** GM 元素偏移，**也不是** fmap 展开坐标，**不要**手工换算 | 传行内列号正确；误传展开坐标会整块错位 |
| mExtension | 本步展开的 M 行数（dedx 空间窗口数，用真值不对齐到 16） | `baseUseM_`，允许跨多条 dedx 行 |
| kStartPt | 相对**当前 L1A 父段**的 K 起点（`kLocalA`），父段重装时按新段原点重算。**不是**全局 K 坐标 | 传段内相对起点正确；误传全局坐标越界读行带 |
| kExtension | 本步内部 K 长度 | `baseUseK_`（含通道补齐，补齐项零贡献） |
| channelSize | 当前父段的通道元素数（C0 对齐后的跨度） | **不是** K 长度，也**不是**分形数 |
| H/W stride | 窗口滑动步长，恒为 1（正向 stride 已由插零表达） | **不要**再乘 stride——重复乘 stride 是本通路特有事故源 |
| repeat/dstStride | 四字段 `{repeatStride, repeatTime, repeatMode, dstStride}` | repeatTime=1（一次展开 mExtension 个窗口）、dstStride=**M0 分形个数**（`alignUp(baseUseM_,M0)/M0`，L0A 内相邻 M0 分形的排布步进）、repeatStride/repeatMode 恒 0 |
| FMatrix padList | **元素序固定 {left, right, top, bottom}**，单位=行带内列/行数，即剩余反向补边（负边界先裁剪真实 dedy 后的剩余值） | bottom 恒填上限值 255 表示**下边界不设防**——窗口越过行带底的行为由驻留行带覆盖（§5.2.1 反推），上界则必须精确 |

**M 块跨多输出行的语义**：

- mExtension 允许 **> wi**：窗口序列跨 dedx 行，行切换由硬件在 fmap 行带内自动完成，软件只需保证行带反推（§5.2.1）覆盖本块全部窗口的输入行；
- mStartPt 仍取**本块首输出的行内列号**（跨行不改变取值规则）；
- 跨行与 W 尾块叠加（本块末行不满）时：mExtension 取本块 M 真值，MMAD 的 m 取 `alignUp(真值, m0)`（任务末 M 块为 mCompute），FIXPIPE 的 mSize 取真值——对齐规则同 §2.1 汇总表，无额外修正。

**参数限制（超限先收缩再复检，仍超限由 Tiling 准入拒绝）**：

| 限制 | 值 | 超限出路 |
|------|----|---------|
| 反向补边 padList（左/右/上） | ≤ 255（8bit 搬运字段；负边界先裁剪真实 dedy 再表达剩余补边） | maxM 收紧到 512 后复检（[tiling](tiling/tiling.md)） |
| padList（下） | 恒 255（下边界不设防，兼容配置不是数学 pad） | — |
| H/W stride | 恒 1（指令上限 63 天然满足） | — |
| dilation | 保留正向原值；指令上限随适配层核实 | 超限由 Tiling 准入拒绝 |
| dedy 驻留长度（stepKa×baseK） | ≤ 65535（搬运字段） | 缩小 stepKa |

窗口跨行时仍按 dedx 的 wi 折行；实际驻留行带按第 5.2.1 节反推。Load3D 的窗口范围使用 dilation 扩大后的有效核长，K 长度仍只使用 hk×wk 个采样点。所有 K 和有效 kd 完成后才写回 dedx。

具体 API 字段、编码范围和配套重载见 [指令映射](../../adapters/ascendc/input/instruction-mapping.md#4-l1a-到-l0a)。

### 4.5 output_padding 到窗口的对应（转置卷积）

Conv3DTranspose 的 output_padding 改变目标输出范围，相应扩展后侧边界和可选窗口数。新增位置仍按相同输入、weight 和采样关系计算，可能有非零贡献。

```text
同一 dedy_expand 与 weight 核
              |
              v
目标轴: [公共输出窗口] [q 选择的新增窗口]
              |                 |
              +---- 同一 K 配对-+

每个输出位置：乘积和 + 其通道 bias
```

q 不增加真实核长或 K 项，也不表示在输出末尾直接拼零。bias 沿 N 通道方向广播，同一输出块只在首次乘加引入一次；无贡献块的 bias 输出见第 8.6 节。

---

## 5 数据搬运与写回

### 5.1 指令与单位总表

本节给出**语言中立的指令语义、单位与地址公式**；API 实体、参数结构体与位域填充见 [Ascend C 指令映射](../../adapters/ascendc/input/instruction-mapping.md) 或 [DSL 指令映射](../../adapters/dsl/input/instruction-mapping.md)。

| 指令 | 作用 | 关键单位 |
|------|------|---------|
| GM→L1 布局转换搬运（DN2NZ / ND2NZ 模式） | 随路转换源矩阵朝向；DN2NZ 读 DN，ND2NZ 读 ND | 有效长度、源步幅和目的矩阵步幅为元素；目的 C0/N 步幅为 32B 块 |
| Load3D（滑窗搬运） | L1A→L0A，随路完成反向边界 + img2col 滑窗展开 | 窗口/起点/扩展均为元素数 |
| Load2D（子块抽取） | L1B→L0B，转置/逆序形成 ZN 分形 | 步幅以 512B 分形计（FP16/BF16=256 元素、FP32=128 元素） |
| MMAD | L0A×L0B→L0C 矩阵乘累加 | m/n/k 为元素数；有效范围与内部计算跨度按第 2.1 节分别确定 |
| FIXPIPE | L0C→GM，NZ2DN/NZ2ND 还原，按输出 dtype 转换或 FP32 直出 | 源步幅以 64B 通道块计 |
| Fill（置零） | 片上区域预置零（插零间隔、尾块） | 32B 块 |

**搬运矩阵轴义**：布局转换搬运以"矩阵"为单位组织数据。对 dedy，copyD 是当前 Cout 通道，copyN 是一行 Wo 或连续的 Ho×Wo，copyIdx 在逐行搬运时表示行号——copyN 不是 batch，copyD 也不是卷积深度 D。对 weight，三轴按 weight 变换分支分别绑定（见 [kernel 侧参数表（Ascend C）](../../adapters/ascendc/input/instruction-mapping.md) / [DSL](../../adapters/dsl/input/instruction-mapping.md)）。

**逐元素地址公式**（忽略已加基址；copyIdx 为矩阵号，C0=32/元素字节数）：

```text
ND source = copyIdx*srcNdMatrixStride + copyN*srcDValue + copyD
DN source = copyIdx*srcDnMatrixStride + copyD*srcDValue + copyN
NZ target = copyIdx*dstNzMatrixStride
          + (copyD//C0)*dstNzC0Stride*C0 + copyN*dstNzNStride*C0 + copyD%C0
```

目的通道拆为 `copyD=copyD1*C0+copyD0`：copyD0 在一个 32B 块内连续；copyD1、copyN、copyIdx 分别使用各自目的步幅。矩阵之间可以交错存放，实际轴序由步幅决定，不能把矩阵个数机械地当成目的最外轴。

**补齐职责分工**：布局转换搬运只补齐 D 轴尾部；N 轴尾部、矩阵间空洞及 stride 前放大留下的空间位置，由调用方的 Fill 覆盖。元素偏移用于张量下标；Buffer 分配和平台容量使用字节。源矩阵步幅保留原张量完整几何（分组、深度不参与本次矩阵循环，但 GM 起点与通道步幅仍包含它们）。

---

### 5.2 dedy：GM→L1A→L0A

#### 5.2.1 从 dedx 块反推 dedy 行带

固定批次、组和有效深度配对。当前输出块从 `mStart` 开始，覆盖 `baseUseM_` 个空间点；设 `Hd=(hk-1)*dilationH+1`、`hoExpand=(ho-1)*strideH+1`。先求完整 H/W 窗口的包围行带，区间均为左闭右开：

| 范围 | 计算 |
|---|---|
| dedx 首、末行 | `hFirst=mStart//wi`，`hLast=(mStart+baseUseM_-1)//wi` |
| 窗口在 dedy_expand 中的行区间 | `hWindowStart=hFirst-backpropPadUp`，`hWindowEnd=hLast-backpropPadUp+Hd` |
| 有效驻留行区间 | `hResidentStart=max(hWindowStart,0)`，`hResidentEnd=min(hWindowEnd,hoExpand)` |
| 非空区域的真实 dedy 行 | `[ceilDiv(hResidentStart,strideH),ceilDiv(hResidentEnd,strideH))` |

交集非空时，`curHoSize=hResidentEnd-hResidentStart`；首条真实 dedy 行在 L1A 内的行偏移为 `hSkip=alignUp(hResidentStart,strideH)-hResidentStart`。交集为空时转入初始化/跳过协议。窗口和交集先按有符号整数计算，再生成搬运字段。

```text
+-----------------------------------------+
| dedx M 块 -> 首/末行 -> 所需展开行区间  |
+--------------------+--------------------+
                     |
                     v
      +--------------+--------------+
      | 与有效 dedy_expand 行取交集 |
      +--------------+--------------+
                     |
                     v
         +-----------+-----------+
         | 按正确相位拷贝真实行  |
         +-----------------------+

L1A: [零间隙] [真实 dedy 行] [零间隙] [真实 dedy 行] ...
```

预算使用最坏跨行范围，实际搬入范围按当前块重算。strideH 大于 1 时，首条真实行在 L1A 中的相位由 hSkip 保留；不能把它移到父段首行。

负左右边界对应裁剪 dedy_expand，先保留仍参与计算的真实 dedy 列（有符号宽整数计算，先检查非空及源范围）：

```text
# expandedW=(wo-1)*strideW+1
woStart = ceilDiv(-backpropPadLeft,strideW) if backpropPadLeft < 0 else 0
woEnd   = ceilDiv(expandedW+backpropPadRight,strideW) if backpropPadRight < 0 else wo
realWo  = woEnd-woStart
# If nonempty: woExpand=(realWo-1)*strideW+1
```

搬运起点移动 `woStart` 个原 dedy 列；指令剩余左补边为 `woStart*strideW+backpropPadLeft`，剩余右补边为 `expandedW+backpropPadRight-((woEnd-1)*strideW+1)`。非负边界直接保留原值。将负数直接转成无符号 pad 字段会改变边界；空区域单独进入初始化/跳过协议，不提交零长度或越界搬运。

#### 5.2.2 布局转换与窗口提取

```text
GM 源:                       同一 L1A 目的布局:
NCDHW / DN -- DN2NZ --+
                      +-----> (Cout1,curHoSize,woExpand,Cout0)
NDHWC / ND -- ND2NZ --+                      |
                                             | Load3D
                                             v
                                       (K1,M1,M0,K0)

每个 Cout1 内:
  [u_w 处的 Cout0 值] [零通道 ...] [下一真实 u_w 处的 Cout0 值]
      真实采样点       stride 间隔       真实采样点

末个 Cout1: 有效通道 + 通道对齐零
```

两种源格式只改变 GM 寻址，目的轴序一致。源地址保留完整深度、总通道和原行步幅；目的步幅保留 stride 插零间隔。通道尾部补零、插零间隔和边界零分别检查，不能互相替代。

#### 5.2.3 窗口网格与 K 长度

滑窗指令先按有效核覆盖范围确定窗口网格，再按原核采样点形成 K；dilation 增大覆盖范围，不增加采样点数：

| 量 | 计算 |
|---|---|
| Hd、Wd | `(hk-1)*dilationH+1`、`(wk-1)*dilationW+1` |
| Hp、Wp | `curHoSize+顶部补边+底部补边`、`woExpand+左补边+右补边` |
| Hstep、Wstep | `(Hp-Hd)//stride+1`、`(Wp-Wd)//stride+1`（本通路 stride=1） |
| 窗口网格 M | Hstep*Wstep；当前输出块只取 mStartPt 起的 baseUseM_ 行 |
| 父片段 K | channelSize*hk*wk；channelSize 已按 C0 对齐 |

窗口按 Wstep 折行，可以跨多个输出行；须保持编码后的 Wstep 与原 wi 一致，不能任意改变 L1A 行宽再继续使用原 mStartPt。Dk 在外层单独归约，不乘入本次 H/W 窗口的 K。

有效核尺寸决定窗口的覆盖范围，`hk*wk` 决定采样点数；dilation 增大前者，不增加后者。

### 5.3 weight：GM→L1B→L0B

L1B 中的 hk/wk 是原核顺序；L0B 按 r_h/r_w 配对窗口：

| loadB2Condition | L1B 轴序 | L1B→L0B |
|---|---|---|
| 1，转置加逆序 | `(Cin1,Hk,Wk,Cout1,Cout0,Cin0)` | 核空间逆序，并转置分形 |
| 2，仅逆序 | `(Cout1,Hk,Wk,Cin1,Cin0,Cout0)` | 保留通道朝向，核空间逆序 |
| 1，单位核仅转置 | `(Cin1,Cout1,Cout0,Cin0)` | 核体积为 1，Cout1 正序，转置分形 |
| 2，FP32 单位核直接取块 | `(Cout1,Cin1,16,8)` | GM→L1B 已形成目标朝向，连续抽取 |

分支 1 用于 FP16/BF16，表中两种通道块长度均为 16。FP32 固定选择分支 2：Cin0=16、Cout0=8；普通核在 B2 逆序核高宽，单位核直接取块。

```text
                              +-----------------------+
                              | weight: 固定 g、kd    |
                              | MTE2: 选定 L1B 父布局 |
                              +-----------+-----------+
                                          |
            +-----------------------------+-----------------------------+
            v                             v                             v
+-----------+-----------+     +-----------+-----------+     +-----------+-----------+
| loadB2Condition=1     |     | loadB2Condition=2     |     | loadB2Condition=1     |
| Cin1,Hk,Wk,           |     | Cout1,Hk,Wk,          |     | Cin1,Cout1,           |
| Cout1,Cout0,Cin0      |     | Cin1,Cin0,Cout0       |     | Cout0,Cin0            |
| H/W 逆序              |     | H/W 逆序              |     | 核体积=1              |
| 转置                  |     | 保持通道朝向          |     | 转置                  |
+-----------+-----------+     +-----------+-----------+     +-----------+-----------+
            |                             |                             |
            +-----------------------------+-----------------------------+
                                          v
                             +------------+------------+
                             | L0B: (K1,N1,N0,K0)      |
                             | K: Cout1,r_h,r_w,Cout0  |
                             | N: 当前 dedx 通道       |
                             +-------------------------+
```

```text
loadB2=1: 父段(Cout0,Cin0)     -- 转置 + H/W 逆序 ---+
                                                     |
loadB2=2: 父段(Cin0,Cout0)     -- H/W 逆序 ----------+-> L0B(K,N)
                                                     |
loadB2=1: 点核父段(Cout0,Cin0) -- 转置 --------------+

L0B 物理轴序: (K1,N1,N0,K0)
同一 K -> L0A 与 L0B 中同一 (Cout1,r_h,r_w,Cout0)
```

图中的转置支路用于 FP16/BF16；FP32 沿中间支路准备 `(Cout1,Hk,Wk,Cin1,16,8)`，单位核省去 Hk/Wk 轴和空间逆序。各路径的 L0B 均以 K 为行、dedx 通道为列。GM 的通道步幅保留完整 dk。

**NDHWC 权重的逆序路径也使用 DN2NZ 语义**：此时组内输入通道 ci 是连续的 N 轴，dedy 通道 co 是跨 `cinG*kernelVolume` 的 D 轴——按张量格式名称选择转换方向会把两轴放反。所有源步幅保留原 GM 的 `cinG`，尾块只缩短有效长度；具体参数结构选择见适配层 [instruction-mapping（Ascend C）](../../adapters/ascendc/input/instruction-mapping.md#3-weight-到-l1b) / [DSL](../../adapters/dsl/input/instruction-mapping.md)。

**数据准备分支**：普通 weight 直接搬入；VecTranspose 从预转置的 workspace 读取；C04 与 group enlarge 由 AIV 准备 L1B。普通父布局表不能直接套给 C04：它使用四通道 K 排列，而且 H/W 逆序已经在 gather 中完成。group enlarge 的块对角结果按原始组通道拼接，不是跨组求和。数学关系见 [scene](scene.md#3-数据准备与分组合并)，准入、缓冲与 API 见 [指令映射 §3.4–§3.8](../../adapters/ascendc/input/instruction-mapping.md)。

weight 变换按 dtype 和最终块形状选择，见 [Ascend C 分支判定](../../adapters/ascendc/input/tilingkey-templates.md#2-loadb2condition-判定与-key) 或 [DSL 分支判定](../../adapters/dsl/input/tiling.md#3-weight-变换分支)。单位核沿用原 stride、边界和目标形状寻找 dedy，无贡献位置按初始化契约处理。

### 5.4 MMAD 与 dedx 写回

同一个输出块的所有 K 与有效核深度累加到同一 L0C；归约结束后执行一次写回。

L0C 的行对应 dedx 高宽，列对应 dedx 通道。NCDHW 采用 NZ2DN，空间步幅为 1、通道步幅为 `di*hi*wi`；NDHWC 采用 NZ2ND，空间步幅为 cin、通道步幅为 1。写回只取 baseUseM_/baseUseN_，内部对齐区留在片上。

```text
L0C / NZ: (N1,M1,M0,N0)
  |
  +-- NZ2DN --> dedx / NCDHW
  |               空间步幅 = 1；通道步幅 = di*hi*wi
  |
  `-- NZ2ND --> dedx / NDHWC
                  空间步幅 = cin；通道步幅 = 1

源: 对齐 M/N 分形 -> 目的: 只写有效 M/N
```

逐层理解 NZ2DN/NZ2ND 还原（以 NCDHW 为例，四层：矩阵 → Cin1 → M 位置 → Cin0；单矩阵时最外层步幅为 0）。令 `ci1=nLocal//16、ci0=nLocal%16`，忽略两侧已加基址：

```text
srcElement = (ci1*srcStride + mLocal*srcNzC0Stride)*16 + ci0
dstElement = nLocal*dstStride + mLocal
```

`srcStride=mCompute` 跨过一个完整 Cin1 的源数据（单位 16 个 float，即 64B）；相邻 M 位置跨一个 Cin0。`dstStride=di*hi*wi` 保留完整 dedx 深度跨度，不能取当前块的 baseUseM_。NDHWC（NZ2ND）使用相同源排布，目的改为 `mLocal*cin+nLocal`。

### 5.5 原张量地址与三维步幅

先恢复当前全局坐标，再代入「NCDHW、NDHWC 与 DHWCN」的偏移式。若改用任务基址加增量，明确基址已包含哪些轴。

| 张量 | 完整坐标 |
|---|---|
| dedy | `(batchIdx,groupIdx*coutG+co,curDoutIdx,h_o,w_o)` |
| weight | `(groupIdx*coutG+co,ci,kd,kh,kw)`，按 weight 格式展开 |
| dedx | `(batchIdx,groupIdx*cinG+cinStart+nLocal,curDinIdx_,(mStart+mLocal)//wi,(mStart+mLocal)%wi)` |

权重令 `coGlobal=groupIdx*coutG+co`，其元素偏移为：

```text
NCDHW: ((((coGlobal*cinG+ci)*dk+kd)*hk+kh)*wk+kw)
NDHWC: ((((coGlobal*dk+kd)*hk+kh)*wk+kw)*cinG+ci)
```

当前 dedx 深度 curDinIdx_ 与原核深度 kd 通过 `dTmp=curDinIdx_+padFront-kd*dilationD` 配对；仅 dTmp 非负、能被 strideD 整除且商小于 dout 时参与计算。原深度索引已经表达反向关系，不再额外反转一次。

---

## 6 多核任务与输出归属

### 6.1 按 dedx 区域分配任务

任务范围由 singleCoreM/Cin/Din 决定，任务内再按 baseM/baseN 计算：

```text
  +-----------------------+
  | dedx 任务             |
  | batch × group × D 段  |
  | × M 段 × Cin 段       |
  +-----------+-----------+
              |
              v
+-------------+-------------+
| 一个任务归属一个输出区域  |
| 完整 co/kd/kh/kw 归约     |
+-------------+-------------+
              |
              v
+-------------+-------------+
| 输出块 -> 有效 kd -> K 块 |
+---------------------------+
```

下图为任务的组成维度（非解码轴序）；解码轴序见下表，按条件三选一。

任务计数：`mCount=ceilDiv(hi*wi,singleCoreM)`、`nCount=ceilDiv(cinG,singleCoreCin)`、`dCount=ceilDiv(di,singleCoreDin)`，`totalTasks=batch*group*dCount*mCount*nCount`。核按 `taskId=coreId+taskRound*coreNum` 取任务。group enlarge 打包时 group 以放大组数 real_g 参与计数（见 [场景分支](scene.md#31-group-enlarge多个原始组合成一个计算组)）。

批次、组在外层，内层从慢到快为：

| 条件 | 轴顺序 |
|---|---|
| dk>1 | 空间段 → 深度段 → dedx 通道段 |
| dk=1 且 hk>1，或其余情况下 mCount>nCount | 深度段 → 空间段 → dedx 通道段 |
| 其余 | 深度段 → dedx 通道段 → 空间段 |

解码从最快轴取余并整除；起点为段号乘任务长度，尾长为 `min(任务长度,总长-起点)`。任务内按照 dedx 深度、dedx 通道块、dedx 空间块推进，空间块变化最快。

Dk>1 决定任务轴顺序，核号本身不固定对应某个空间段。每个任务的每个输出块都要完成有效核深度和 K 归约。

### 6.2 输出互斥与尾任务

dedy/weight 只读，多个核可以访问同一输入；每个有效 dedx 坐标属于唯一输出任务。K 与核深度在任务内归约，主计算不依赖其他核的部分和。

| 范围 | 起点 | 本次有效长度 |
|---|---|---|
| dedx 深度 | `depthIdx*singleCoreDin` | `min(singleCoreDin,di-depthStart)` |
| dedx 空间 | `spaceIdx*singleCoreM` | `min(singleCoreM,hi*wi-spaceStart)` |
| dedx 组内通道 | `channelIdx*singleCoreCin` | `min(singleCoreCin,cinG-channelStart)` |

各核独立管理自己的 L1A/L1B/L0A/L0B/L0C。输出预清零的先后依赖另按 [前置处理](../preprocessing.md) 和调用适配执行。

---

## 7 单核分块与迭代

### 7.1 任务、输出块与完整归约

下面用**语言中立的伪代码**表达有效任务与完整归约，只体现执行逻辑，不绑定任何开发语言；阶段名与状态记号见 [../language.md](../language.md)（第 8/10 节）。实际实现还按「DAG、Buffer 与流水」一章维护槽状态，各语言的 API 实体见适配层。

```text
for taskId = coreId to totalTasks-1, step coreNum:      # 每核按核数步长认领任务
    DecodeTask(taskId)
    for d = 0 to depthUse-1:                             # dedx 深度
        for n = 0 to channelUse-1, step baseN:           # dedx 通道块
            for m = 0 to spaceUse-1, step baseM:         # dedx 空间块，最快变化
                SetOutputBlock(depthStart+d, channelStart+n, spaceStart+m)
                UpdateL1ComputeInfo()
                if not needComputeFlag_:                 # 行带交集或有效深度为空
                    NoNeedComputeHandle()                # 归还持有父段后跳过本块
                    continue
                tmpSingleCoreK = 4*singleShapeHWk_ if enableC04Flag else
                                 alignUp(singleCoreCout,C0)*singleShapeHWk_
                firstContribution = true
                for kd = 0 to dk-1:                      # 原核深度
                    dTmp = curDinIdx_ + padFront - kd*dilationD    # 有符号
                    if dTmp < 0 or dTmp % strideD != 0 or dTmp // strideD >= dout:
                        continue                         # 该 kd 无有效 dedy 深度配对
                    curDoutIdx = dTmp // strideD
                    for kStart = 0 to tmpSingleCoreK-1, step baseK: # K 分块
                        baseUseK_ = min(baseK, tmpSingleCoreK-kStart)
                        LoadToA1(curDoutIdx, kStart)
                        LoadToB1(kd, kStart)
                        LoadToA2(kStart, baseUseK_)
                        LoadToB2(kStart, baseUseK_)
                        MmadLocal(baseUseK_, firstContribution)      # 首次乘加初始化 L0C
                        firstContribution = false
                if not firstContribution:                # 本块有实际乘加才写回
                    LoadL0c2Gm()
    FinishTaskB1()                                       # 任务末按条件保留 L1B
FreeB1Tensor()                                           # 全部任务后归还剩余持有的 L1B
```

`singleShapeHWk_` 是当前核高宽采样点数。LoadToA1/L1B 仅在需要新父段时装载；允许预取后继段。每个输出块的第一个实际乘加初始化 L0C，后续所有 K 和有效核深度更新同一部分和。

kernel-split 模板在此循环之外增加一层**子核相位循环**：M/K 按子核口径重建，相位切换刷新父段长度、反向边界与窗口参数，split-HW 的写回经重排阶段按相位交织（详见 [kernel-split 模板](tiling/templates/kernel-split/tiling.md) 与其适配层实现）。

UpdateL1ComputeInfo 先确定有效输出范围、dedy_expand 行带交集及深度采样。行带为空或没有有效核深度时，needComputeFlag_=false，NoNeedComputeHandle 按持有状态归还父段后跳过；不申请结果槽或等待计算事件。行带非空但仅含 stride 插零时，仍可按零窗口计算。跳过位置的零值由前置处理保证。

全部任务结束后，FreeB1Tensor 归还仍持有的 L1B；调用方再执行一次 End，完成队列事件收尾。固定事件的初态与收尾由适配层的队列/事件机制承载，循环体内不重复收尾。

### 7.2 L1 父段与 K 子块

令 `j=kIterIdx`，每路分别定位当前父段：

```text
segmentFirstIter = (j // stepK) * stepK
segmentStart     = segmentFirstIter * baseK
curStep          = min(stepK, kIter-segmentFirstIter)
segmentUse       = min(curStep*baseK, tmpSingleCoreK-segmentStart)

kLocalA = kStart - a1.segmentStart
kLocalB = kStart - b1.segmentStart
```

`kIter=ceilDiv(tmpSingleCoreK,baseK)`。L1A/L1B 的 step 可以不同，两路最终取出的完整 K 区间必须一致。段内寻址使用实际父形状；`stepK*baseK` 仅为槽预留上限。K 尾部 `baseUseK_=min(baseK,tmpSingleCoreK-kStart)`；C04 最后一个窗口与 MMAD 的补齐长度分别记录。

```text
+--------------+-------------+------------------+-------------+
| K 块号       | ... j-1 ... | ... j ...        | ... j+1 ... |
+--------------+-------------+------------------+-------------+
| L1A 生命周期 | 父段 A_i    | 父段 A_(i+1) ... | ...         |
| L1B 生命周期 | 父段 B_t    | 父段 B_t         | 下一 B 父段 |
+--------------+-------------+------------------+-------------+

每个 j: 同一完整 K 区间 -> 各自独立的 kLocalA 与 kLocalB
A 仅在其父段变化时重装；B 仅在其父段变化时重装
```

### 7.3 从 dedx 平面观察任务与计算块

固定批次、组、dedx 深度和一个通道，按 `m=h_i*wi+w_i` 展平输出平面。多核任务切分连续 M 段，任务内再按 baseM 切成计算块；两种边界都可能落在一行中部。

示例设定（块大小取易画的小值；实际由 tiling 决策，数值不改变投影规则）：`hi=4, wi=8 → M=32`；`singleCoreM=11`（任务0 M[0,11)、任务1 M[11,22)、任务2 M[22,32) 尾任务收缩为 10）；`baseM=6`（跨行计算块）：

```text
+---+--------------------------+----------+-----------------------------------------+
| h | 空间点所在块             | m 范围   | 说明                                    |
+---+--------------------------+----------+-----------------------------------------+
| 0 | [0][0][0][0][0][0][1][1] | m=0..7   | 块1 从 (0,6) 起：中间列起块             |
| 1 | [1][1][2][2][2][2][2][2] | m=8..15  | 任务1 从 (1,3) 起：任务边界不保证行对齐 |
| 2 | [2][3][3][3][3][4][4][4] | m=16..23 | 块2 wrap 至 (2,0)；任务2 从 (2,6) 起    |
| 3 | [4][4][4][4][5][5][5][5] | m=24..31 | 块5 从 (3,4) 起（任务2 的 M 尾块）      |
+---+--------------------------+----------+-----------------------------------------+

任务0：M[0,11)；任务1：M[11,22)；任务2：M[22,32)
```

要点：块边界可以落在行中任意列（跨行块在行切换处 wrap）——这正是 L1A 行带要按块反推（§5.2.1）、tiling 预算要用最坏跨行覆盖的空间原因。**任务按 `taskId = coreId + round*coreNum` 轮转认领**：同一物理核的多个任务在 M 轴上不必连续（与正向"连续段绑核"不同），但**任务内计算块连续、块边界不跨任务**（尾块按任务剩余 M 收缩）。

```text
dedx M 轴 ----------------------------------------------------------->
任务:       [         任务 i         ][下一任务 ...]
计算块:     [baseM 块][baseM 块][尾 M][baseM 块] ...
行:         [行 h ...][行 h+1 ...][行 h+2 ...] ...
                         ^ 块可能起始于一行中部

一个输出块:
  有效 kd -> 全部 K 块 -> 同一 L0C -> Fixpipe -> 归属的 dedx 区域
```

任务起点为 `spaceIdx*singleCoreM`，块起点为任务起点加 `curMIdx_*baseM`；行列由 `h_i=m//wi、w_i=m%wi` 恢复。有效块长不超过当前任务剩余 M，计算块不能越过任务边界。

K 和 kd 只更新当前输出块的部分和，不推进空间位置。定位差异时，从 dedx 坐标依次反查 m、任务、计算块及其 L1A 行带。

### 7.4 有效范围、物理跨度与槽容量

有效 M/N 决定窗口和写回范围，mCompute/nCompute 表示分形对齐跨度。单份 L0C 槽预先覆盖 `baseM*baseN*accBytes`；尾块缩短使用范围，不改变槽容量，也不扩大任务归属。

```text
L0C 槽:     [ 有效 M/N 区域 ][ 对齐补齐 ]
Fixpipe:    [ 有效 M/N 区域 ] -> 只写归属的 dedx 坐标

K/kd 循环保持同一输出块；下一输出块独立初始化
```

| 尾部 | 必须更新 |
|---|---|
| M | 当前 dedy_expand 驻留行、窗口起点、baseUseM_/mCompute、Fixpipe 源跨度及写回长度 |
| dedx 通道 | L1B/L0B 有效通道、分形补齐、baseUseN_ 与写回通道数 |
| K 父段/子块 | 当前父跨度、kLocalA/B、baseUseK_；补齐 dedy 通道贡献为零 |
| 核深度 | 只发射有效深度配对；首次实际 MMAD 初始化，全部跳过时由初始化协议产出零 |

### 7.5 循环切换与状态生命周期

| 切换位置 | 更新或重建 | 继续保留 |
|---|---|---|
| 同一父段内推进 K | `kStart/baseUseK_、kLocalA/B`，按实际 MMAD 轮转输入槽 | L1A/L1B 父布局、当前 L0C、firstContribution |
| 某一路更换父段 | 本路原点、segmentUse、实际父形状；局部 K 从新段起点计算 | 另一路仍有效的父段、当前 L0C |
| 下一个有效原 kd | dedy 深度、weight 核深度；K 从0开始，两路段内 K 重新定位 | 当前输出块的 L0C 与 firstContribution，输入槽轮转序列 |
| 跳过无效 kd | 不发起该深度的搬运、计算或等待 | 已有部分和及槽状态；尚无贡献时仍保持 firstContribution=true |
| 下一个 dedx 输出块 | 新的深度/M/N坐标、有效范围、L0C 槽；firstContribution=true | 仅保留身份和覆盖范围仍匹配的父段 |
| 下一个任务 | 解码新坐标，检查 L1B 跨任务保留权限 | 获准且权重身份未变的 L1B；其他句柄按最后消费者归还 |

L1A 的身份包含批次、组、dedy 深度、空间原点及通道范围；L1B 包含权重地址、原核深度、通道范围和父布局。K 起点相同不代表父段相同。预取下一段时，当前 Load3D 仍须使用当前段的 FMatrix、补边和重复参数。

所有 K 和有效核深度完成后才交接 L0C；新输出块的初始化与已有输出块的继续累加分别判断。各级物理槽沿实际提交顺序轮转，不能在深度切换时直接归零槽号。

---

## 8 DAG、Buffer 与流水

### 8.1 四条主流水的职责

| 流水 | 操作 | 消费的数据 |
|---|---|---|
| MTE2 | dedy/weight 的 GM→L1 | 生成 L1A/L1B 父段 |
| MTE1 | Load3D / Load2D | 读取 L1A/L1B，生成 L0A/L0B |
| M | MMAD | 读取 L0A/L0B，更新 L0C |
| F | Fixpipe | 读取完整 L0C，写回 dedx |

同一流水内按提交顺序执行，跨流水通过就绪和归还依赖衔接。两路 MTE2、两路 MTE1 是数据依赖分支，同一硬件流水上的指令仍顺序提交。上表为普通主通路，L0C→GM 使用 F。C04/group enlarge 的 B1 准备使用 AIV 的 V/MTE3；split-HW 的结果由 F 写中转区，再由 AIV 交织并通过 MTE3 写回。新增依赖分别见权重准备和 kernel-split 文档。

### 8.2 四级流水图

四级流水沿用"搬入父段 → 准备操作数 → 乘加 → 写回"的顺序，四条流水（MTE2/MTE1/M/F）用双槽缓冲错开一级，使装载与计算重叠：

```text
+----------------+--------+-------------+--------------------+--------------+---------------+
| 流水 / 时间 -> | 阶段 0 | 阶段 1      | 阶段 2             | 阶段 3       | 阶段 4 / 后续 |
+----------------+--------+-------------+--------------------+--------------+---------------+
| MTE2           | 装A0B0 | 装A1B1      | 装A2B2             | ...          |               |
| MTE1           |        | A0B0:L1->L0 | A1B1:L1->L0        | A2B2         | ...           |
| M              |        |             | MMAD(0,首步初始化) | MMAD(1,累加) | ...           |
| F              |        |             |                    |              | dedx(0)->GM   |
+----------------+--------+-------------+--------------------+--------------+---------------+
每级错开一步，四条流水并行流动。
```

下图用两个输出块表示重叠，每块有两条实际贡献：j00/j01 属于输出 0，j10/j11 属于输出 1；各贡献可来自 K 分块或有效 kd。图中各级使用双槽，仅用于解释依赖。

```text
+------+--------+--------+----------+--------+---------+--------+-------+
| 流水 | t0     | t1     | t2       | t3     | t4      | t5     | t6    |
+------+--------+--------+----------+--------+---------+--------+-------+
| MTE2 | 装 j00 | 装 j01 | 装 j10   | 装 j11 |         |        |       |
| MTE1 |        | j00    | j01      | j10    | j11     |        |       |
| M    |        |        | 初始化 0 | 累加 0 | 初始化1 | 累加 1 |       |
| F    |        |        |          |        | dedx0   |        | dedx1 |
+------+--------+--------+----------+--------+---------+--------+-------+
```

各级槽的轮转（每格为可重叠的逻辑阶段，不代表固定周期或实测耗时）：

| contribution | j00 | j01 | j10 | j11 |
|---|---|---|---|---|
| L0 input slot | 0 | 1 | 0 | 1 |
| L0C slot | 0 | 0 | 1 | 1 |
| L0C operation | init | add | init | add |

输出 0 的 F 等待 j01 完成；此时 M 可以使用另一 L0C 槽计算输出 1。存在更多 K/核深度贡献时，相应延长 M 的累加区间，F 仍只在该输出块完整归约后发生。

L1 父段实际按各自 stepKa/stepKb 轮转，l1-resident 的 L1B 可以跨多个输出块保持不动；不能把图中“每次贡献重装父段”的示意套到驻留路径。

### 8.3 DAG 与 Buffer 存活区间

```text
MTE2(A1) -> MTE1(A2) -----+
                          |
MTE2(B1) -> MTE1(B2) -----+--> MMAD -> 完整 K/kd 归约 -> F -> dedx

同一槽: 末次消费者完成 -> 下一次覆盖
```

| Buffer | 持有范围 | 份数与轮转单位 |
|---|---|---|
| L1A | 当前 dedy 父段到最后一次窗口提取 | al1Pbuffer，按 L1A 段 |
| L1B | 当前权重父段到最后一次 L0B 准备 | bl1Pbuffer，按 L1B 段或驻留身份 |
| L0A/L0B | 当前乘加的输入 | al0/bl0Pbuffer，成对按实际 MMAD 轮转 |
| L0C | 当前输出块的完整归约与写回 | cl0Pbuffer，按完整输出块轮转 |

L1 容量含同时存活的 L1A/L1B 及槽间隔；L0 三块分别核算。Host 数据预算、Kernel 物理末地址和指令访问范围分别检查。

### 8.4 各模板的差异

| 模板 | DAG 与 Buffer 行为 |
|---|---|
| kernel-split | M/K 按子核口径重建，相位切换刷新父段与窗口参数；split-HW 的 F 输出经重排阶段（UB/workspace 中转、AIV 交织）写回，DAG 形态见 [kernel-split 模板](tiling/templates/kernel-split/tiling.md) |
| standard | 输出任务独立；父段按各自 step 推进，任务内可保留已覆盖完整 K 的 L1B |
| small-shape | 主通路相同；任务范围与 baseM 可以不同，短任务不一定使用第二份槽 |
| l1-resident | L1B 对多个输出块形成多消费者节点；L1A 继续更新，每个输出块仍独立初始化 L0C |

普通三模板的完整 L1B 要求 `baseN>=singleCoreCin`、`baseK*curStepKb>=singleCoreCout*hk*wk`、`bl1Pbuffer=1`、`dk=singleIterateDk=1`。跨任务保留还要求 `enableFullLoad` 对应的当前权限有效、group=1、weight 地址和通道/核/父布局身份一致，并排除 C04 与 vector group enlarge。当 `baseN=singleCoreCin=nValue` 时，当前组的 N 方向也全载；任务不再切 N，L1A 仍按 M 窗口更新，L0B 仍逐 K 块取数。该分支复用同一 L1B 多消费者节点，不增加独立布局或同步协议。

下面沿用两输出块、每块两次 K 贡献，取 dk=1。L1B_R 表示覆盖完整 K 的同一份 L1B；L1A 每次贡献更新父段，L1A 和 L0 输入使用双槽，L0C 按输出块轮转：

```text
+------+---------+-------+-------+---------+--------+---------+--------+-------+
| 流水 | t0      | t1    | t2    | t3      | t4     | t5      | t6     | t7    |
+------+---------+-------+-------+---------+--------+---------+--------+-------+
| MTE2 | B1 全量 | A1_00 | A1_01 | A1_10   | A1_11  |         |        |       |
| MTE1 |         |       | j00   | j01     | j10    | j11     |        |       |
| M    |         |       |       | 初始化0 | 累加 0 | 初始化1 | 累加 1 |       |
| F    |         |       |       |         |        | dedx0   |        | dedx1 |
+------+---------+-------+-------+---------+--------+---------+--------+-------+

B1: 装载一次 -> 跨多个 Load2D 消费者保持 -> 末次读取后释放
A1: 当前父段 -> Load3D 消费者 -> 释放
L0C: 每个输出块独立初始化
```

MTE2 行先装 L1B_R，再依次装 L1A，表示同一流水上的顺序。L1B 的释放等待最后一次 L0B 准备，不必等该输出的 MMAD 或 F；继续复用时仍保持持有。图中输出0与输出1各自初始化 L0C，驻留权重不传递部分和。

权重身份改变时，先结束旧版本读取再释放，关闭本核后续跨任务保留权限；新任务按实际条件装载。任务内复用和跨任务保留分别判断，最终 FreeB1Tensor 只释放实际持有的数据。

### 8.5 就绪、归还与首末轮

| 存储 | 就绪方向 | 可覆盖方向 |
|---|---|---|
| L1A/L1B | MTE2→MTE1 | 最后一次 MTE1→MTE2 |
| L0A/L0B | MTE1→M | M→MTE1 |
| L0C | 完整归约 M→F | F→M |

| 阶段 | 发起与等待 |
|---|---|
| 首轮填充 | 建立可写槽，完成需要的补零，再装载首段；等待实际生产者就绪，不等待尚未发布的历史归还 |
| 稳态 | 在可写的另一槽准备后继段；MTE1 准备操作数、M 累加，F 只消费完整输出块；父段在最后一次读取后归还 |
| 末轮排空 | 停止不存在后继段的预取；完成最后的 MMAD/F，释放仍持有的父段及事件 |

单槽复用按下面的依赖等待；双槽交替选择可写槽，仍分别等待该槽的上次读取完成。单槽时允许不同存储级之间重叠，但不能同时读写同一槽。发布归还可跟在读取提交后，下一次覆盖仍必须等待读取完成。

各级槽按自己的频率轮转。K 增加推进输入槽，L0C 按完整输出块轮转；SDK 队列和事件见 [同步实现（Ascend C）](../../adapters/ascendc/input/sync-implementation.md) / [DSL](../../adapters/dsl/input/sync-implementation.md)。

```text
同一 L1A/L1B 槽: MTE2 写入 -> 全部 MTE1 读取完成 -> 下一次 MTE2 写入
同一 L0A/L0B 槽: MTE1 写入 -> MMAD 读取完成       -> 下一次 MTE1 写入
同一 L0C 槽:     首次 MMAD -> 其余 MMAD -> F 读取完成 -> 下一输出块
```

就绪防止读到未完成的数据，归还防止覆盖仍被消费的数据。首次令牌由队列/事件所有者建立，末次只等待实际提交的操作；空父段、跳过的深度和无后继输出不产生虚假的等待或预取。

当前 Ascend C 映射采用 CO1 队列交接 M→F、F→M；其 MMAD `unitFlag=0`。同步参数和事件资源统一按适配层实现。

### 8.6 可选 bias 通路与无贡献输出（转置卷积）

带 bias 的 DAG 在两路操作数之外增加 bias 支路：

```text
dedy / GM --MTE2--> L1A --MTE1 / img2col--> L0A ---------+
                                                         |
weight/GM --MTE2--> L1B --MTE1 / weight_rot--> L0B ------+
                                                         |
bias / GM --MTE2--> biasL1Que_ --MTE1--> biasBTQue_ -----+
                                                         |
                                                         v
                                                 +-------+-------+
                                                 | M -> L0C      |
                                                 | F -> dedx/GM  |
                                                 +---------------+

bias 只进入每个输出块的首次实际 MMAD。
```

| Buffer | 内容 | 有效期与复用 |
|---|---|---|
| L1A/L1B、L0A/L0B/L0C | 公共计算对象 | 使用公共 DAG、五路槽和事件 |
| biasL1Que_ | 当前 bias 通道段，64B 对齐 | 全量模式保留到身份变化；分块模式在复制到 BT 后归还 |
| biasBTQue_ | MMAD 可读取的 FP32 bias | 覆盖所用通道段；最后一次 MMAD 读取后才归还 |

`isBiasFullLoad=1` 时搬入当前任务的 dedx 通道范围，N 子块从 BT 取相应偏移；为 0 时按当前 N 块搬入。该标志与权重 `enableFullLoad` 分别管理。standard/small-shape/l1-resident 三种模板均计入 bias L1 和 BT 预算；kernel-split 在自身容量模型中计入 bias/scale 预算。

带 bias 的流水在公共四级流水之上增加 bias 首次准备与纯 bias 输出：

```text
bias 准备:
GM --MTE2--> L1 --MTE1--> BT（就绪）

输出块 0: L0 就绪 -> M(首步带 bias) -> M(其余 K/kd) -> F
输出块 1: L0 就绪 -> M(首步带 bias) -> M(其余 K/kd) -> F
BT 可复用；每个输出块重置"仅一次 bias"状态。

纯 bias 块: MTE1(LoadL0Zero) -> M(0*0 + bias) -> F

bias 段切换:
末次 M 读取完成 -> 释放旧 BT/L1 bias -> 装载下一 bias 段
```

`computeBiasOnce_` 每个输出块开始时为 false，首次带 bias 的 MMAD 后为 true。父段切换、K 轮转和核深度跳过均保留该状态。

若行带为空或所有深度都没有有效贡献，带 bias 的路径仍执行 `ComputeForBias` 并写回。LoadL0Zero 通过合法的全补零窗口准备 L0A/L0B，再使用带 BT 的 MMAD；它不读取未初始化的输入槽。没有 bias 时沿用公共跳过与预清零协议。

行带裁剪只决定输入搬运范围。输出任务的有效 M/N 必须全部覆盖；若指令范围使一个输出块无法完整表达，先在 Tiling/适配层调整块长。含 bias 的输出不能因输入窗口为空而漏写。

---

## 9 检查点

- [ ] 阶段表六条通路（含可选 bias）与指令职责一一对应，两路 K 位置始终一致
- [ ] 同一元素在 NCDHW/NDHWC 下的偏移可互推；NZ/ZN 相对的矩阵明确
- [ ] K 遍历序 `Cout1 → r_h → r_w → Cout0` 在 L0A 列与 L0B 行配对，补齐通道零贡献
- [ ] 反向边界先裁剪真实 dedy，再表达剩余补边；stride 相位保留
- [ ] 每个有效 dedx 坐标唯一归属一个任务；任务内完成全部有效核深度与 K 归约
- [ ] 差异元素 `(h_i,w_i)` 可反查 `m=h_i*Wi+w_i`、所属空间段与计算块（衔接 verification 失败定位）
- [ ] 首末轮事件按实际提交判断；驻留 L1B 的多消费者释放等待最后一次 MTE1 读取
- [ ] 带 bias 时每个输出块恰好引入一次 bias；无贡献块仍写回完整输出

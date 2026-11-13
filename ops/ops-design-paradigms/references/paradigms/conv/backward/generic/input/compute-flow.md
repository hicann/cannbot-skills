# 卷积反向算子计算流程

> **层级**：通用知识层（generic/input/），语言中立。以达芬奇架构 Cube 通路为基准描述。
>
> **分工**：本文件回答"**数据怎么流、任务怎么分、循环怎么写、同步怎么插**"。算什么（数学模型与视图定义）见 [scene.md](scene.md)；切分大小如何决策见 [tiling/tiling.md](tiling/tiling.md)。
>
> 术语记号见 [../language.md](../language.md)。dedy_expand、fmap、weight_rot 是逻辑视图；L1A/L1B、L0A/L0B、L0C 是实际片上 Buffer。轴序均从慢到快，小分形用于展示元素顺序；具体 API 字段由语言适配层落实。
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
dedy -- stride insert zeros --> dedy_expand -- pad/crop --> fmap --+
                                                                   |
weight -- rot180 + channel role swap --> weight_rot ---------------+
                                                                   |
                                                                   |
                                                                   v
                                                        +----------+----------+
                                                        | window * weight_rot |
                                                        +----------+----------+
                                                                   |
                                                                   |
                                                                   v
                                                         +---------+---------+
                                                         | sum(co,kd,kh,kw)  |
                                                         +---------+---------+
                                                                   |
                                                                   |
                                                                   v
                                                               +---+---+
                                                               | dedx  |
                                                               +-------+
```

img2col（窗口展开，也称 img2col）把每个 dedx 空间位置需要的 fmap 窗口排成矩阵的一行。相邻窗口共享的元素会出现在多行中；窗口提取可随 L1→L0A 搬运完成，不要求先在 GM 物化展开矩阵。

完整模型推导（前放大、rot180、反向边界、转置卷积的 output_padding 与 bias）见 [scene.md](scene.md)。

---

## 2 数据流总览

### 2.1 阶段与指令职责

| 阶段 | 通路 | 完成的转换 |
|---|---|---|
| LoadToA1 / MTE2 | dedy GM→L1A | 按原格式寻址、通道分块、保留 stride 插零间隔 |
| LoadToB1 / MTE2 | weight GM→L1B L1 | 选择原核深度，形成所选父布局 |
| LoadToA2 / MTE1 | L1A→L0A | fmap 边界、H/W img2col、NZ 分形 |
| LoadToB2 / MTE1 | L1B→L0B | 核位置逆序、按需转置分形，形成 ZN |
| MmadLocal / M | L0A×L0B→L0C | FP32 部分和，完整 K 与核深度归约 |
| LoadL0c2Gm / F | L0C→dedx GM | 类型转换、NZ2DN/NZ2ND、有效范围写回 |
| bias 装载（可选） | bias GM→L1→BT | FP32 通道段搬入 BT，首次乘加作 L0C 初值 |

指令的语言中立语义、单位与地址公式见第 5.1 节；各语言的 API 实体与参数结构体见 [指令映射](../../adapters/ascendc/input/instruction-mapping.md)；bias 通路的装载与 MMAD 细节见 [bias 通路](../../adapters/ascendc/input/bias-bt-path.md)。两路的 K 位置一致；改变某一父布局时，同时重算子块定位。

各阶段使用的长度口径如下，实际字段由适配层映射：

| 阶段 | 有效范围 | 内部跨度与寻址 |
|---|---|---|
| GM→L1A/L1B | 本段实际 dedy、dedx 通道及原张量区域 | 目的通道按 C0 补齐，源步幅保留总通道和完整深度 |
| L1A→L0A | `baseUseM_、baseUseK_` 个窗口行与内部 K 项 | M 方向按 `mCompute` 排布；K 起点为 `kLocalA` |
| L1B→L0B | 当前 dedx 通道与相同的 `baseUseK_` | 按 `nCompute` 排布；K 起点为 `kLocalB`，沿实际父布局取块 |
| MMAD | n 取 `baseUseN_`，k 取 `baseUseK_` | m 在任务末 M 块取 `mCompute`，其余按当前有效 M；L0C 保持完整归约 |
| F | `mSize=baseUseM_、nSize=baseUseN_` | 源跨度取 `mCompute`，目的步幅取原 dedx 步幅 |

`baseUseK_` 包含 dedy 通道补齐项。例如 `coutG=18、hk*wk=3`，数学贡献项为 54，内部 `kValue=96`；`baseK=64` 时计算 64、32 两轮。槽预留长度由 baseK 和 step 决定，末轮只消费实际剩余 K。分形数、元素数和字节数在适配层分别换算。

### 2.2 dedy 侧：GM → L1A → L0A

```text
      +---------------------------+
      | dedy / GM: NCDHW or NDHWC |
      +-------------+-------------+
                    |
                    | MTE2: DataCopy + Fill; stride zeros
                    v
+-------------------+-------------------+
| L1A: (Cout1,curHoSize,woExpand,Cout0) |
+-------------------+-------------------+
                    |
                    | MTE1: Load3D; boundary + img2col
                    v
         +----------+----------+
         | L0A: (K1,M1,M0,K0)  |
         +---------------------+

L0A row = dedx spatial m
L0A col = K = (co1,r_h,r_w,co0)
```

要点：

1. **L1 里存的是 dedy_expand 的驻留片段**（一段行带 × 若干通道分形），不是展开后的矩阵——展开发生在 L1→L0A 的 Load3D 里；
2. 驻留行带按"从 dedx 块反推 dedy 行带"的交集计算确定；负反向边界先裁剪真实 dedy，剩余补边由窗口参数表达；
3. stride 插零不搬全零行：Fill 预置零 + 目的步幅保留间隔，真实行按相位写入。

### 2.3 weight 侧：GM → L1B → L0B

```text
+-----------------------------+
| weight / GM: NCDHW or NDHWC |
+--------------+--------------+
               |
               | MTE2: DataCopy; original kd/kh/kw
               v
+--------------+--------------+
| L1B: selected parent layout |
+--------------+--------------+
               |
               | MTE1: Load2D; transpose/reverse
               v
    +----------+----------+
    | L0B: (K1,N1,N0,K0)  |
    +---------------------+

L1B, loadB2Condition:
1: (Cin1,Hk,Wk,Cout1,Cout0,Cin0)
2: (Cout1,Hk,Wk,Cin1,Cin0,Cout0)
3: (Cin1,Cout1,Cout0,Cin0)

L0B row = K = (co1,r_h,r_w,co0)
L0B col = dedx channel ci
```

要点：weight 与 dedy 不同——**GM→L1 时保留原核坐标**，仅形成父布局；rot180 的核空间逆序发生在 L1→L0B 的 Load2D 里；核深度配对由外层的采样检查完成，不在搬运中反转。

### 2.4 输出侧：MMAD → L0C → dedx GM

```text
L0A: (K1,M1,M0,K0) --------------------+
                                       |
L0B: (K1,N1,N0,K0) --------------------+
                                       |
bias / BT: optional, first MMAD only --+
                                       |
                                       |
                                       v
                                   +---+---+
                                   | MMAD  |
                                   +---+---+
                                       |
                                       |
                                       v
                         +-------------+-------------+
                         | L0C: (N1,M1,M0,N0), FP32  |
                         +-------------+-------------+
                                       |
                                       |
                                       v
                           +-----------+-----------+
                           | FIXPIPE: NZ2DN/NZ2ND  |
                           | FP32 -> FP16/BF16     |
                           +-----------+-----------+
                                       |
                                       |
                                       v
                         +-------------+-------------+
                         | dedx / GM: NCDHW or NDHWC |
                         +---------------------------+
```

要点：同一个输出块的所有 K 与有效核深度累加到同一 L0C；完整归约后一次 Fixpipe 写回，只写有效 M/N，保留原张量全局步幅。

### 2.5 两路数据流总图

```text
       +-----------+                             +-------------+
       | dedy / GM |                             | weight / GM |
       +-----+-----+                             +------+------+
             |                                          |
             | MTE2: DataCopy + Fill                    | MTE2: parent layout
             v                                          v
+------------+------------+              +--------------+--------------+
| L1A: (Cout1,H,W,Cout0)  |              | L1B: original kernel order  |
+------------+------------+              +--------------+--------------+
             |                                          |
             | MTE1: Load3D / img2col                   | MTE1: Load2D
             v                                          v
    +--------+--------+                        +--------+--------+
    | L0A: M x K, NZ  |                        | L0B: K x N, ZN  |
    +-----------------+                        +-----------------+

L0A ----------+
              |
L0B ----------+
              |
              |
              v
  +-----------+-----------+
  | MMAD: sum over K / Dk |
  +-----------+-----------+
              |
              |
              v
  +-----------+-----------+
  | L0C: M x N, FP32, NZ  |
  +-----------+-----------+
              |
              |
              v
       +------+------+
       | F: Fixpipe  |
       +------+------+
              |
              |
              v
 +------------+------------+
 | dedx / GM: valid M/N    |
 | Original tensor strides |
 +-------------------------+
```

L1A 保存可复用的图像片段，L0A 才展开每个 dedx 位置的窗口；相邻窗口可重复读取 L1A。L1B 保留原核坐标，L0B 读取时组织 weight_rot；原深度索引与 H/W 逆序合起来实现三个轴的反向关系。

---

## 3 数据格式与存储轴序

先给速览表与格式转换全景，详细推导与示例见后续小节：

| 对象 | 外部格式与轴序（慢到快） | 说明 |
|---|---|---|
| dedy/dedx | NCDHW：`(N,C,D,H,W)`；NDHWC：`(N,D,H,W,C)` | 两种通道位置 |
| weight | NCDHW：`(Cout,CinG,Dk,Hk,Wk)`；NDHWC：`(Cout,Dk,Hk,Wk,CinG)`；DHWCN 推导视角：`(Dk,Hk,Wk,CinG,Cout)` | 权重格式中的 N 对应 Cout，C 对应 CinG |
| L1A | `(Cout1,curHoSize,woExpand,Cout0)` | 一个深度的 dedy_expand 驻留片段 |
| L0A / L0B / L0C | `(K1,M1,M0,K0)` / `(K1,N1,N0,K0)` / `(N1,M1,M0,N0)` | 粒度均为 16 |

**格式转换全景**（GM 原始排布 → L1 驻留布局 → L0 计算分形 → 输出还原）：

```text
dedy / NCDHW (DN) -- DN2NZ --+
                             |
dedy / NDHWC (ND) -- ND2NZ --+
                             |
                             |
                             v
                +------------+------------+
                | L1A: (Cout1,H,W,Cout0)  |
                +------------+------------+
                             |
                             |
                             v
                   +---------+---------+
                   | Load3D / img2col  |
                   +---------+---------+
                             |
                             |
                             v
                  +----------+----------+
                  | L0A: (K1,M1,M0,K0)  |
                  +---------------------+

        +-------------+
        | weight / GM |
        +------+------+
               |
               | DataCopy
               v
+--------------+--------------+
| L1B: selected parent layout |
+--------------+--------------+
               |
               | Load2D
               v
    +----------+----------+
    | L0B: (K1,N1,N0,K0)  |
    +---------------------+

        +-----------+
        | L0A * L0B |
        +-----+-----+
              |
              |
              v
          +---+---+
          | MMAD  |
          +---+---+
              |
              |
              v
+-------------+-------------+
| L0C: (N1,M1,M0,N0), FP32  |
+---------------------------+

FIXPIPE:
NZ2DN -> dedx / GM: NCDHW
NZ2ND -> dedx / GM: NDHWC
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
n fixed; C=2, D=2, H=1, W=2; element = (c,d,w)

NCDHW:
  c=0: 000 001 010 011
  c=1: 100 101 110 111
  order: c=0 -> c=1

NDHWC:
  d0,w0: 000 100
  d0,w1: 001 101
  d1,w0: 010 110
  d1,w1: 011 111
  order: d0,w0 -> d0,w1 -> d1,w0 -> d1,w1

element (1,0,1): NCDHW offset=5; NDHWC offset=3
```

权重取单组 `Cout=cinG=2、dk=hk=1、wk=2`，元素编号为 `100*co+10*ci+kw`。同一组数值按三种格式排列：

```text
NCDHW: (co,ci,kw)
+----+---------+---------+
| co | ci=0    | ci=1    |
+----+---------+---------+
| 0  | 0 1     | 10 11   |
| 1  | 100 101 | 110 111 |
+----+---------+---------+
Memory: 0 1 10 11 100 101 110 111

NDHWC: (co,kw,ci)
+----+---------+---------+
| co | kw=0    | kw=1    |
+----+---------+---------+
| 0  | 0 10    | 1 11    |
| 1  | 100 110 | 101 111 |
+----+---------+---------+
Memory: 0 10 1 11 100 110 101 111

DHWCN: (kw,ci,co)
+----+-------+--------+
| kw | ci=0  | ci=1   |
+----+-------+--------+
| 0  | 0 100 | 10 110 |
| 1  | 1 101 | 11 111 |
+----+-------+--------+
Memory: 0 100 10 110 1 101 11 111
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

固定一个 dedx 深度平面，将空间作为行、dedx 通道作为列，得到逻辑矩阵 T。记行数 R、列数 S，ND 按行连续，DN 按列连续。将行列分别分成 bR/bS 大小的块，块数为 R1=ceilDiv(R,bR)、S1=ceilDiv(S,bS)：

```text
+------------+--------+--------+--------+-----+   +------------+--------+--------+-----+
| ND: m / ci | c0     | c1     | c2     | ... |   | DN: ci / m | h0w0   | h0w1   | ... |
+------------+--------+--------+--------+-----+   +------------+--------+--------+-----+
| h0w0       | T[0,0] | T[0,1] | T[0,2] | ... |   | c0         | T[0,0] | T[1,0] | ... |
| h0w1       | T[1,0] | T[1,1] | T[1,2] | ... |   | c1         | T[0,1] | T[1,1] | ... |
+------------+--------+--------+--------+-----+   +------------+--------+--------+-----+

NDHWC: adjacent ci contiguous
NCDHW: adjacent m contiguous; channel stride = Di*Hi*Wi
Fixed (n,g,curDinIdx_): current Hi*Wi plane
```

| 格式 | 慢到快轴序 | 遍历方式 |
|---|---|---|
| ND | `(R,S)` | 逐行 |
| DN | `(S,R)` | 逐列 |
| NZ | `(S1,R1,bR,bS)` | 块间按列，块内按行 |
| ZN | `(R1,S1,bS,bR)` | 块间按行，块内按列 |

```text
T[m,ci]: Hi=1, Wi=4, cinG=4; bR=bS=2

  0  1 |  2  3
  4  5 |  6  7
-------+-------
  8  9 | 10 11
 12 13 | 14 15

ND:  0  1  2  3   4  5  6  7   8  9 10 11  12 13 14 15
DN:  0  4  8 12   1  5  9 13   2  6 10 14   3  7 11 15
NZ:  0  1  4  5 | 8  9 12 13 | 2  3  6  7 |10 11 14 15
ZN:  0  4  1  5 | 2  6  3  7 | 8 12  9 13 |10 14 11 15
```

把四个分形标为 F00/F01/F10/F11，下标分别为行块、列块：

```text
+-----------------------+-----+-----+
| row block / col block | 0   | 1   |
+-----------------------+-----+-----+
| 0                     | F00 | F01 |
| 1                     | F10 | F11 |
+-----------------------+-----+-----+

NZ block order: F00 -> F10 -> F01 -> F11
ZN block order: F00 -> F01 -> F10 -> F11

Inside F00:
+--------+------------------+---------------+
| layout | in-block order   | axes          |
+--------+------------------+---------------+
| NZ     | 0 -> 1 -> 4 -> 5 | (S1,R1,bR,bS) |
| ZN     | 0 -> 4 -> 1 -> 5 | (R1,S1,bS,bR) |
+--------+------------------+---------------+
```

ND→NZ/ZN 的推导分三步，原矩阵坐标始终不变：

```text
    +-------+
    | (R,S) |
    +---+---+
        |
        | Pad tail rows/columns with zeros
        v
+-------+-------+
| (R1*bR,S1*bS) |
+-------+-------+
        |
        | Split block and in-block indices
        v
+-------+-------+
| (R1,bR,S1,bS) |
+---------------+

Reorder and store:
NZ: (S1,R1,bR,bS)
ZN: (R1,S1,bS,bR)
```

真实搬运可同时完成多个步骤。NZ/ZN 必须说明相对哪一个矩阵，不能仅凭缩写推断通道朝向。

设行坐标拆成 `(r1,r0)`，列坐标拆成 `(s1,s0)`，紧致布局的元素偏移为：

$$
\begin{aligned}
\mathrm{NZ} &= ((s_1R_1+r_1)b_R+r_0)b_S+s_0, \\
\mathrm{ZN} &= ((r_1S_1+s_1)b_S+s_0)b_R+r_0.
\end{aligned}
$$

### 3.3 五维张量与片上视图

NCDHW 固定 `(n,d)` 后，以 H×W 为矩阵行、C 为列，原存储是 DN 视角；NDHWC 是 ND 视角。若为整个五维张量定义分形视图：

| 视图 | 慢到快轴序 | 说明 |
|---|---|---|
| 固定深度、HW×C 的 NZ | `(N,D,C1,HW1,M0,C0)` | 各深度平面分别分形 |
| 固定深度、HW×C 的 ZN | `(N,D,HW1,C1,C0,M0)` | 同一矩阵的另一朝向 |
| DHW×C 合轴 NZ | `(N,C1,DHW1,M0,C0)` | 空间体整体分形，与逐深度视图分别使用 |
| 通道分块 NDC1HWC0 | `(N,D,C1,H,W,C0)` | 每个空间点保留一组 C0，不把它等同于完整矩阵分形 |

实际 L1A 为一个深度的 dedy_expand 驻留片段，轴序 `(Cout1,curHoSize,woExpand,Cout0)`；它不是所有 fmap 窗口的拼接。L0A 的轴序为 `(K1,M1,M0,K0)`；L0B 的轴序为 `(K1,N1,N0,K0)`；L0C 为 `(N1,M1,M0,N0)`，粒度均为 16。

Cout1/Cin1 是 dedy/dedx 通道分形轴，Cout0/Cin0 是块内轴；M1/N1/K1 是矩阵分形轴，M0/N0/K0 是对应块内轴。

### 3.4 L0A、L0B 与 L0C 的计算分形

L0A/L0B 用同一 K 配对，L0C 的行列分别继承 dedx 空间和 dedx 通道：

```text
+-----------------+--------------+------------------------+---------------+
| buffer          | logical tile | contiguous inside tile | axes          |
+-----------------+--------------+------------------------+---------------+
| L0A / NZ        | M0 x K0      | K0 (row-wise)          | (K1,M1,M0,K0) |
| L0B / ZN        | K0 x N0      | K0 (column-wise)       | (K1,N1,N0,K0) |
| L0C / NZ / FP32 | M0 x N0      | N0 (row-wise)          | (N1,M1,M0,N0) |
+-----------------+--------------+------------------------+---------------+

L0A / NZ:                 L0B / ZN:              L0C / NZ:
K0 ->                     N0 ->                  N0 ->
+--------+                +--+--+--+             +--------+
| -----> |                |  |  |  |             | -----> |
| -----> |                | v| v| v|             | -----> |
|  ...   |                |  |  |  |             |  ...   |
+--------+                +--+--+--+             +--------+

L0C[M,N] += L0A[M,K] * L0B[K,N]
```

L0A/L0B 的 FP16/BF16 分形为 16×16、512 字节；L0C 的 FP32 分形为 16×16、1024 字节。块长与元素字节数分别记账。

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

例如 `C0=2、coutG=3、hk=1、wk=2`：先读通道 0/1 的窗口位置 0，再读通道 0/1 的窗口位置 1，然后读通道 2/补齐通道的两个位置。两路同时采用这个顺序。

```text
C0=2; pad = aligned channel, zero contribution

+-----+---+---+---+---+---+-----+---+-----+
| K   | 0 | 1 | 2 | 3 | 4 | 5   | 6 | 7   |
+-----+---+---+---+---+---+-----+---+-----+
| co1 | 0 | 0 | 0 | 0 | 1 | 1   | 1 | 1   |
| r_w | 0 | 0 | 1 | 1 | 0 | 0   | 1 | 1   |
| co  | 0 | 1 | 0 | 1 | 2 | pad | 2 | pad |
+-----+---+---+---+---+---+-----+---+-----+
L0A column K and L0B row K refer to the same coordinates.
```

### 4.2 三维例子：先插零，再建立边界视图

取 FP16、NCDHW、单组问题，所有 shape 均为逻辑轴序：

| 对象或属性 | 取值 |
|---|---|
| dedx | `(1,16,3,3,4)` |
| dedy | `(1,16,2,2,2)` |
| weight | `(16,16,2,2,2)` |
| D/H/W strides | `(1,1,2)` |
| dilations、pads | 全 1、全 0 |

正向尺寸为 `(3-2)+1=2`、`(3-2)+1=2`、`floor((4-2)/2)+1=2`，与 dedy 一致。为逐元素追踪，只令 dedy 通道 0 非零；以下展示 dedx 通道 0，其他输出通道按同样规则计算。

```text
+------------+-----+-----+   +-------------+-------+-------+-------+
| dedy, co=0 | w=0 | w=1 |   | dedy_expand | u_w=0 | u_w=1 | u_w=2 |
+------------+-----+-----+   +-------------+-------+-------+-------+
| d=0,h=0    | a   | b   |   | d=0,h=0     | a     | 0     | b     |
| d=0,h=1    | c   | d   |   | d=0,h=1     | c     | 0     | d     |
| d=1,h=0    | e   | f   |   | d=1,h=0     | e     | 0     | f     |
| d=1,h=1    | g   | h   |   | d=1,h=1     | g     | 0     | h     |
+------------+-----+-----+   +-------------+-------+-------+-------+

dedy_expand: (2,2,3); strideW=2
pad: 1 on both sides of D/H/W; fmap: (4,4,5)
fmap[d=0] = fmap[d=3] = all zeros

+----------+-----+-----+-----+-----+-----+   +----------+-----+-----+-----+-----+-----+
| fmap d=1 | w=0 | w=1 | w=2 | w=3 | w=4 |   | fmap d=2 | w=0 | w=1 | w=2 | w=3 | w=4 |
+----------+-----+-----+-----+-----+-----+   +----------+-----+-----+-----+-----+-----+
| h=0      | 0   | 0   | 0   | 0   | 0   |   | h=0      | 0   | 0   | 0   | 0   | 0   |
| h=1      | 0   | a   | 0   | b   | 0   |   | h=1      | 0   | e   | 0   | f   | 0   |
| h=2      | 0   | c   | 0   | d   | 0   |   | h=2      | 0   | g   | 0   | h   | 0   |
| h=3      | 0   | 0   | 0   | 0   | 0   |   | h=3      | 0   | 0   | 0   | 0   | 0   |
+----------+-----+-----+-----+-----+-----+   +----------+-----+-----+-----+-----+-----+
```

这里的外圈零来自反向边界，中间零来自正向宽度 stride。dedy_expand/fmap 表达采样关系；实际 L1A 只驻留本轮需要的 dedy_expand 区域，边界由窗口参数表达。

### 4.3 三维 rot180 与核深度配对

固定 `co=ci=0`，原权重与反向视图为：

```text
+-----------+------+------+   +-------------+-------+-------+
| weight    | kw=0 | kw=1 |   | weight_rot  | r_w=0 | r_w=1 |
+-----------+------+------+   +-------------+-------+-------+
| kd=0,kh=0 | w0   | w1   |   | r_d=0,r_h=0 | w7    | w6    |
| kd=0,kh=1 | w2   | w3   |   | r_d=0,r_h=1 | w5    | w4    |
| kd=1,kh=0 | w4   | w5   |   | r_d=1,r_h=0 | w3    | w2    |
| kd=1,kh=1 | w6   | w7   |   | r_d=1,r_h=1 | w1    | w0    |
+-----------+------+------+   +-------------+-------+-------+

r_d=Dk-1-kd; reverse H/W inside each slice
```

实现保留原 `kd`，按 `curDinIdx_+padFront-kd*dilationD` 选择 dedy 深度；Load2D 只完成该片的 H/W 逆序。上图完整 rot180 与下表的原核寻址是同一个反向关系的两种表达。

| dedx 深度 curDinIdx_ | 原 kd=0 | 原 kd=1 |
|---|---|---|
| 0 | dedy 深度 0 | 越界，跳过 |
| 1 | dedy 深度 1 | dedy 深度 0 |
| 2 | 越界，跳过 | dedy 深度 1 |

### 4.4 img2col：从 fmap 的窗口到 L0A

固定 `curDinIdx_=1`。反向视图 `r_d=0` 对应 fmap 深度 1、原 `kd=1`。H/W 窗口为 2×2，按单位步长移动，形成 12 个 dedx 空间位置：

```text
fmap, d=1; [ ] marks the 2x2 window for m=5:
+-----------------+
|  0  0  0  0  0  |
|  0 [a  0] b  0  |
|  0 [c  0] d  0  |
|  0  0  0  0  0  |
+-----------------+

unit-stride windows: W first, then H
  |
  v
L0A rows; only co=0 shown; column labels = (r_h,r_w)
+----+-----+-----+-------+-------+-------+-------+
| m  | h_i | w_i | (0,0) | (0,1) | (1,0) | (1,1) |
+----+-----+-----+-------+-------+-------+-------+
| 0  | 0   | 0   | 0     | 0     | 0     | a     |
| 1  | 0   | 1   | 0     | 0     | a     | 0     |
| 2  | 0   | 2   | 0     | 0     | 0     | b     |
| 3  | 0   | 3   | 0     | 0     | b     | 0     |
| 4  | 1   | 0   | 0     | a     | 0     | c     |
| 5  | 1   | 1   | a     | 0     | c     | 0     |
| 6  | 1   | 2   | 0     | b     | 0     | d     |
| 7  | 1   | 3   | b     | 0     | d     | 0     |
| 8  | 2   | 0   | 0     | c     | 0     | 0     |
| 9  | 2   | 1   | c     | 0     | 0     | 0     |
| 10 | 2   | 2   | 0     | d     | 0     | 0     |
| 11 | 2   | 3   | d     | 0     | 0     | 0     |
+----+-----+-----+-------+-------+-------+-------+
```

图中只显示 `co=0` 的四列，它们在完整 K 中的坐标为 `0、16、32、48`。每个窗口位置还包含其他 15 个 dedy 通道；本例这些通道的数值设为零，与通道对齐补零分别理解。完整 L0A 的逻辑形状为 `12×64`，M 的物理跨度按 16 对齐。

第二个核深度片的窗口矩阵相同，只将 `a/b/c/d` 换成 `e/f/g/h`。两片分别乘对应的反向权重，并累加到同一个 L0C：

```text
fmap[d=1] -> L0A * L0B[:,ci=0]=[w7,w6,w5,w4] --+
                                               |
fmap[d=2] -> L0A * L0B[:,ci=0]=[w3,w2,w1,w0] --+
                                               |
                                               |
                                               v
                                         +-----+-----+
                                         | L0C[:,0]  |
                                         +-----------+

L0B: only the four K positions for co=0 are shown.
dedx[0,0,1,0,0] = a*w4 + e*w0
dedx[0,0,1,1,1] = a*w7 + c*w5 + e*w3 + g*w1
```

第一项可沿原连接复算：dedx 深度 1 分别接收“dedy 深度 0 × 原核深度 1”和“dedy 深度 1 × 原核深度 0”。上图按反向视图列出两项；实际 Kernel 按原 kd 正序，先计算原核深度 0 的贡献，再累加原核深度 1 的贡献。整个例子同时覆盖宽度插零、H/W 窗口和核深度累加。

### 4.5 output_padding 到窗口的对应（转置卷积）

固定批次、组、通道、深度和高度，取宽度例子：x 为 `[a,b]`，weight 为 `[w0,w1,w2]`，stride=2，左右 pad=1，dilation=1，qW=1，输出宽度为 4。

```text
x / dedy:    [a,b]
dedy_expand: [a,0,b]
fmap:        [0,a,0,b,0,0]

Each y row selects three consecutive fmap positions:
+------------+---+---+---+---+---+---+
| fmap index | 0 | 1 | 2 | 3 | 4 | 5 |
+------------+---+---+---+---+---+---+
| y[0]       | 0 | a | 0 |   |   |   |
| y[1]       |   | a | 0 | b |   |   |
| y[2]       |   |   | 0 | b | 0 |   |
| y[3]       |   |   |   | b | 0 | 0 |
+------------+---+---+---+---+---+---+

img2col / L0A     weight_rot / L0B     y, product only
[ 0 a 0 ]         [ w2 ]               [ a*w1      ]
[ a 0 b ]         [ w1 ]               [ a*w2+b*w0 ]
[ 0 b 0 ]         [ w0 ]               [ b*w1      ]
[ b 0 0 ]                              [ b*w2      ]

y = L0A * L0B (before bias)
```

qW=0 时只取前三个输出窗口；qW=1 增加第四个窗口，真实核长和 K 均未改变。图中窗口为逻辑视图，实际按块装入 NZ/ZN；生成工程按目标范围取窗口，不要求在 GM 中生成整个插零张量。

每行再加对应通道 bias。多通道时，L0A 的列与 L0B 的行按公共 K 顺序配对；深度核在外层累加。bias 广播方向为矩阵 N 方向，对同一输出通道的所有 M 行使用同一个值。

---

## 5 数据搬运与写回

### 5.1 指令与单位总表

搬运与计算由五类指令构成。本节给出**语言中立的指令语义、单位与地址公式**；各开发语言的 API 实体、参数结构体与位域填充在适配层（AscendC 见 [instruction-mapping](../../adapters/ascendc/input/instruction-mapping.md)）。

| 指令 | 作用 | 关键单位 |
|------|------|---------|
| GM→L1 布局转换搬运（DN2NZ / ND2NZ 模式） | GM→L1，随路完成布局转换。**DN2NZ = 源为 DN**（通道在外，NCDHW 类输入用）；**ND2NZ = 源为 ND**（通道在内，NDHWC 类输入用）。前缀指源格式 | 目的步幅以 32B 块计 |
| Load3D（滑窗搬运） | L1A→L0A，随路完成反向边界 + img2col 滑窗展开 | 窗口/起点/扩展均为元素数 |
| Load2D（子块抽取） | L1B→L0B，转置/逆序形成 ZN 分形 | 步幅以 512B 分形（256 元素）计 |
| MMAD | L0A×L0B→L0C 矩阵乘累加 | m/n/k 为元素数（m/n 对齐、k 真值） |
| FIXPIPE | L0C→GM，随路 NZ2DN/NZ2ND 还原 + FP32→FP16/BF16 | 源步幅以 64B C0 计 |
| Fill（置零） | 片上区域预置零（插零间隔、尾块） | 32B 块 |

**搬运矩阵轴义**：布局转换搬运以"矩阵"为单位组织数据。对 dedy，copyD 是当前 Cout 通道，copyN 是一行 Wo 或连续的 Ho×Wo，copyIdx 在逐行搬运时表示行号——copyN 不是 batch，copyD 也不是卷积深度 D。对 weight，三轴按 weight 变换分支分别绑定（见 [kernel 侧参数表](../../adapters/ascendc/input/instruction-mapping.md)）。

**逐元素地址公式**（FP16/BF16，忽略已加基址；copyIdx 为矩阵号）：

```text
ND source = copyIdx*srcNdMatrixStride + copyN*srcDValue + copyD
DN source = copyIdx*srcDnMatrixStride + copyD*srcDValue + copyN
NZ target = copyIdx*dstNzMatrixStride
          + (copyD//16)*dstNzC0Stride*16 + copyN*dstNzNStride*16 + copyD%16
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

以「从 dedx 平面观察任务与计算块」的 4×8 平面为例，`baseM=16、mStart=18、baseUseM_=14、hk=2、dilationH=1、backpropPadUp=1`，dedy_expand 高度为 3、strideH 为 1：

```text
dedx rows:        [2,4); hFirst=2, hLast=3
required window:  [1,4)
L1A resident:     [1,3)

+-----------------+-----------+-----------+---------------+
| dedy_expand row | 1         | 2         | 3             |
+-----------------+-----------+-----------+---------------+
| LoadToA1        | copy      | copy      | outside       |
| Load3D          | L1A row 0 | L1A row 1 | boundary zero |
+-----------------+-----------+-----------+---------------+
```

该块实际驻留 2 行；同一 baseM 的容量预算可按 3 行保留。预算、实际搬入区域、虚拟补边分别记录。实现按指令需要扩大驻留范围时，同步更新原点、FMatrix 和容量检查。

strideH 大于 1 时还要保留相位。例如 dedy_expand 行带从 1 开始、strideH=2，L1A 首行是插零，dedy 行1写入 L1A 行1：`dedy_expand[1:5]=[0,dedy行1,0,dedy行2]`。宽度同样先保留插零位置；负反向边界的裁剪编码见下文。

负左右边界对应裁剪 dedy_expand，先保留仍参与计算的真实 dedy 列（有符号宽整数计算，先检查非空及源范围）：

```text
# expandedW=(wo-1)*strideW+1
woStart = ceilDiv(-backpropPadLeft,strideW) if backpropPadLeft < 0 else 0
woEnd   = ceilDiv(expandedW+backpropPadRight,strideW) if backpropPadRight < 0 else wo
realWo  = woEnd-woStart
# If nonempty: woExpand=(realWo-1)*strideW+1
```

搬运起点移动 `woStart` 个原 dedy 列；指令剩余左补边为 `woStart*strideW+backpropPadLeft`，剩余右补边为 `expandedW+backpropPadRight-((woEnd-1)*strideW+1)`。非负边界直接保留原值。将负数直接转成无符号 pad 字段会改变边界；空区域单独进入初始化/跳过协议，不提交零长度或越界搬运。

例如 `wo=4、strideW=2`，dedy_expand 为 `[a,0,b,0,c,0,d]`；反向左右边界为 `-1、-2` 时：

```text
fmap after crop: [0,b,0,c]
GM columns:      b,c          woStart=1, woEnd=3
L1A:             [b,0,c]      woExpand=3
Load3D pad:      left=1, right=0
```


#### 5.2.2 布局转换与窗口提取

```text
GM: NCDHW / DN -- DN2NZ --+
                          |
GM: NDHWC / ND -- ND2NZ --+
                          |
                          |
                          v
      +-------------------+-------------------+
      | L1A: (Cout1,curHoSize,woExpand,Cout0) |
      | Fill + destination strides            |
      +-------------------+-------------------+
                          |
                          |
                          v
                +---------+---------+
                | Load3D / img2col  |
                +---------+---------+
                          |
                          |
                          v
               +----------+----------+
               | L0A: (K1,M1,M0,K0)  |
               +---------------------+
```

固定批次、组、深度和一行，取三个 dedy 通道、两个宽度位置、`strideW=2`。`u_w` 为 L1A 内的展开宽度坐标（dedy_expand 的列号），插零位占一个 u_w。示意通道块长取 2；实际取 16。两种源格式形成相同的 L1A 排布：

```text
+-----------+------+----------+
| GM format | axis | values   |
+-----------+------+----------+
| NDHWC     | w=0  | a0 a1 a2 |
| NDHWC     | w=1  | b0 b1 b2 |
| NCDHW     | co=0 | a0 b0    |
| NCDHW     | co=1 | a1 b1    |
| NCDHW     | co=2 | a2 b2    |
+-----------+------+----------+

ND2NZ / DN2NZ -> same L1A:
+-----+-----------------+-----------------+
| u_w | Cout1=0, co0 -> | Cout1=1, co0 -> |
+-----+-----------------+-----------------+
| 0   | a0 a1           | a2 0            |
| 1   | 0 0             | 0 0             |
| 2   | b0 b1           | b2 0            |
+-----+-----------------+-----------------+

Address order:
Cout1=0: [a0 a1 0 0 b0 b1]
Cout1=1: [a2  0 0 0 b2  0]

u_w=1: stride zeros; after co=2: channel alignment zeros
```

GM 左图只列出选定行的元素，跨行、跨通道仍使用原张量步幅；L1A 按预置零和目的步幅保留插零间隔。高度 stride 同样留下整行零。负反向边界先裁剪搬入区域，剩余补边由窗口提取表达。

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

例如 Hk=3、Wk=2、dilationH=2、dilationW=3，一个窗口覆盖 5×4 空间，但只取六个位置。固定一个 dedy 通道，局部 fmap 为：

```text
+---+-----+-----+-----+-----+---------------+
| h | w=0 | w=1 | w=2 | w=3 | sampled (h,w) |
+---+-----+-----+-----+-----+---------------+
| 0 | 1   | 2   | 3   | 4   | (0,0), (0,3)  |
| 1 | 5   | 6   | 7   | 8   |               |
| 2 | 9   | 10  | 11  | 12  | (2,0), (2,3)  |
| 3 | 13  | 14  | 15  | 16  |               |
| 4 | 17  | 18  | 19  | 20  | (4,0), (4,3)  |
+---+-----+-----+-----+-----+---------------+
Window values for this channel: 1,4,9,12,17,20
```

channelSize=16 时，该窗口有 96 个 K 项；通道 0 的这六个值位于 K=0、16、32、48、64、80，其余列对应同一采样位置的其他通道。**K 使用 Hk×Wk（采样点数），不使用 Hd×Wd（覆盖范围）**。


### 5.3 weight：GM→L1B→L0B

L1B 中的 hk/wk 是原核顺序；L0B 按 r_h/r_w 配对窗口：

| loadB2Condition | L1B 轴序 | L1B→L0B |
|---|---|---|
| 1，转置加逆序 | `(Cin1,Hk,Wk,Cout1,Cout0,Cin0)` | 核空间逆序，并转置分形 |
| 2，仅逆序 | `(Cout1,Hk,Wk,Cin1,Cin0,Cout0)` | 保留通道朝向，核空间逆序 |
| 3，仅转置 | `(Cin1,Cout1,Cout0,Cin0)` | 核体积为 1，Cout1 正序，转置分形 |

```text
                          +---------------------------+
                          | weight: fixed g, kd       |
                          | MTE2: selected L1B parent |
                          +-------------+-------------+
                                        |
                                        |
            +---------------------------+---------------------------+
            |                           |                           |
            v                           v                           v
+-----------+-----------+   +-----------+-----------+   +-----------+-----------+
| loadB2Condition=1     |   | loadB2Condition=2     |   | loadB2Condition=3     |
| Cin1,Hk,Wk,           |   | Cout1,Hk,Wk,          |   | Cin1,Cout1,           |
| Cout1,Cout0,Cin0      |   | Cin1,Cin0,Cout0       |   | Cout0,Cin0            |
| H/W reverse           |   | H/W reverse           |   | Kernel volume=1       |
| Transpose             |   | Keep orientation      |   | Transpose             |
+-----------+-----------+   +-----------+-----------+   +-----------+-----------+
            |                           |                           |
            |                           |                           |
            +---------------------------+---------------------------+
                                        |
                                        v
                          +-------------+-------------+
                          | L0B: (K1,N1,N0,K0)        |
                          | K: Cout1,r_h,r_w,Cout0    |
                          | N: current dedx channels  |
                          +---------------------------+
```

沿用「NCDHW、NDHWC 与 DHWCN」的权重编号，取示意通道块长 2。两种含空间核的路径最终得到同一 L0B 物理序列：

```text
condition=1, L1B (co0,ci0 inside each block):
kw=0 [0 10 100 110] | kw=1 [1 11 101 111]
  | H/W reverse + in-block transpose
  v
L0B (ci0,co0 inside each block):
r_w=0 [1 101 11 111] | r_w=1 [0 100 10 110]
  ^
  | H/W reverse
condition=2, L1B (ci0,co0 inside each block):
kw=0 [0 100 10 110] | kw=1 [1 101 11 111]

condition=3, wk=1:
L1B [0 10 100 110] -- in-block transpose --> L0B [0 100 10 110]
```

转置改变分形内的存储朝向；两条路径的逻辑 L0B 均以 K 为行、dedx 通道为列。固定核深度处理时，GM 权重的通道步幅仍包含完整 dk，不能套用二维核面积代替。

**NDHWC 权重的逆序路径也使用 DN2NZ 语义**：此时组内输入通道 ci 是连续的 N 轴，dedy 通道 co 是跨 `cinG*kernelVolume` 的 D 轴——按张量格式名称选择转换方向会把两轴放反。所有源步幅保留原 GM 的 `cinG`，尾块只缩短有效长度；具体参数结构选择见适配层 [instruction-mapping](../../adapters/ascendc/input/instruction-mapping.md#3-weight-到-l1b)。

三种 weight 变换 的选择使用最终块形状，见 [分支判定](../../adapters/ascendc/input/tilingkey-templates.md#2-loadb2condition-判定与-key)。`1×1×1` 仅使空间逆序成为恒等映射；仅转置路径仍按 stride、边界和目标形状寻找 dedy。例如 `dedy=[1,2,3]、weight=[2]、strideW=2、pad=0`，目标 dedx 宽度为 5 时得到 `[2,0,4,0,6]`，其余空间轴为单维。

### 5.4 MMAD 与 dedx 写回

同一个输出块的所有 K 与有效核深度累加到同一 L0C；归约结束后执行一次写回。

L0C 的行对应 dedx 高宽，列对应 dedx 通道。NCDHW 采用 NZ2DN，空间步幅为 1、通道步幅为 `di*hi*wi`；NDHWC 采用 NZ2ND，空间步幅为 cin、通道步幅为 1。写回只取 baseUseM_/baseUseN_，内部对齐区留在片上。

取单批次、单组，`di=2、hi=1、wi=3、cin=cinG=2`，当前计算深度为 0。以编号表示结果，`x` 表示另一深度的元素：

```text
L0C: valid logical matrix
+---+------+------+
| m | ci=0 | ci=1 |
+---+------+------+
| 0 | 0    | 1    |
| 1 | 10   | 11   |
| 2 | 20   | 21   |
+---+------+------+

NZ2ND -> dedx / NDHWC (memory order):
+-------+-----------------+
| depth | values          |
+-------+-----------------+
| d=0   | 0 1 10 11 20 21 |
| d=1   | x x x x x x     |
+-------+-----------------+

NZ2DN -> dedx / NCDHW (memory order):
+---------+---------+-------+
| channel | d=0     | d=1   |
+---------+---------+-------+
| ci=0    | 0 10 20 | x x x |
| ci=1    | 1 11 21 | x x x |
+---------+---------+-------+
```

NCDHW 同一深度的两个通道在 GM 中并不紧邻；写回必须保留完整深度步幅。

逐层理解 NZ2DN/NZ2ND 还原（以 NCDHW 为例，四层：矩阵 → Cin1 → M 位置 → Cin0；单矩阵时最外层步幅为 0）。令 `ci1=nLocal//16、ci0=nLocal%16`，忽略两侧已加基址：

```text
srcElement = (ci1*srcStride + mLocal*srcNzC0Stride)*16 + ci0
dstElement = nLocal*dstStride + mLocal
```

`srcStride=mCompute` 跨过一个完整 Cin1 的源数据（单位 16 个 float，即 64B）；相邻 M 位置跨一个 Cin0。`dstStride=di*hi*wi` 保留完整 dedx 深度跨度，不能取当前块的 baseUseM_。例如 baseUseM_=27、baseUseN_=19、mCompute=32：对 `(mLocal,nLocal)=(26,18)`，源元素偏移 930；L0C 的 32×32 源区只写 27×19 个有效结果。NDHWC（NZ2ND）使用相同源排布，目的改为 `mLocal*cin+nLocal`。

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
    +-----------------------------+
    | dedx tasks                  |
    | batch x group x D segments  |
    | x M segments x Cin segments |
    +--------------+--------------+
                   |
                   |
                   v
 +-----------------+-----------------+
 | One task owns one output region   |
 | Full co / kd / kh / kw reduction  |
 +-----------------+-----------------+
                   |
                   |
                   v
+------------------+------------------+
| Output block -> valid kd -> K block |
+-------------------------------------+
```

下图为任务的组成维度（非解码轴序）；解码轴序见下表，按条件三选一。

任务计数：`mCount=ceilDiv(hi*wi,singleCoreM)`、`nCount=ceilDiv(cinG,singleCoreCin)`、`dCount=ceilDiv(di,singleCoreDin)`，`totalTasks=batch*group*dCount*mCount*nCount`。核按 `taskId=coreId+taskRound*coreNum` 取任务。

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
                firstContribution = true
                for kd = 0 to dk-1:                      # 原核深度
                    dTmp = curDinIdx_ + padFront - kd*dilationD    # 有符号
                    if dTmp < 0 or dTmp % strideD != 0 or dTmp // strideD >= dout:
                        continue                         # 该 kd 无有效 dedy 深度配对
                    curDoutIdx = dTmp // strideD
                    for kStart = 0 to kValue-1, step baseK:          # K 分块
                        baseUseK_ = min(baseK, kValue-kStart)
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

LoadToA1/L1B 仅在需要新父段时装载；允许预取后继段。每个输出块的第一个实际乘加初始化 L0C，后续所有 K 和有效核深度更新同一部分和。

UpdateL1ComputeInfo 先确定有效输出范围、dedy_expand 行带交集及深度采样。行带为空或没有有效核深度时，needComputeFlag_=false，NoNeedComputeHandle 按持有状态归还父段后跳过；不申请结果槽或等待计算事件。行带非空但仅含 stride 插零时，仍可按零窗口计算。跳过位置的零值由前置处理保证。

全部任务结束后，FreeB1Tensor 归还仍持有的 L1B；调用方再执行一次 End，完成队列事件收尾。固定事件的初态与收尾由适配层的队列/事件机制承载，循环体内不重复收尾。

### 7.2 L1 父段与 K 子块

令 `j=kIterIdx`，每路分别定位当前父段：

```text
segmentFirstIter = (j // stepK) * stepK
segmentStart     = segmentFirstIter * baseK
curStep          = min(stepK, kIter-segmentFirstIter)
segmentUse       = min(curStep*baseK, kValue-segmentStart)

kLocalA = kStart - a1.segmentStart
kLocalB = kStart - b1.segmentStart
```

L1A/L1B 的 step 可以不同，两路最终取出的完整 K 区间必须一致。段内寻址使用实际父形状；`stepK*baseK` 仅为槽预留上限。K 尾部 `baseUseK_=min(baseK,kValue-kStart)`，补齐通道为零贡献。

例如 `kValue=144、baseK=32、stepKa=2、stepKb=4`，L1A 父段为 `[0,64)、[64,128)、[128,144)`，L1B 为 `[0,128)、[128,144)`。末轮长度 16；不能把 L1A 段内起点直接用于 L1B。

```text
+------------+---------+---------+----------+----------+-----------+
| j          | 0       | 1       | 2        | 3        | 4         |
+------------+---------+---------+----------+----------+-----------+
| K interval | [0,32)  | [32,64) | [64,96)  | [96,128) | [128,144) |
| L1A parent | [0,64)  | [0,64)  | [64,128) | [64,128) | [128,144) |
| L1B parent | [0,128) | [0,128) | [0,128)  | [0,128)  | [128,144) |
| kLocalA    | 0       | 32      | 0        | 32       | 0         |
| kLocalB    | 0       | 32      | 64       | 96       | 0         |
+------------+---------+---------+----------+----------+-----------+
Each j: L0A and L0B select the same full K interval.
```

### 7.3 从 dedx 平面观察任务与计算块

另取满足正向形状关系的 3D 问题：`dedx=(1,16,3,4,8)`、`dedy=(1,16,2,3,4)`、`weight=(16,16,2,2,2)`，stride 为 `(1,1,2)`，dilation=1、pad=0。

固定批次、组、dedx 深度和一个 dedx 通道，平面为 4×8。用 `singleCoreM=18、baseM=16` 展示两级边界；这是坐标示意，实际组合由 Tiling 求解。空间段号 0/1 仅用于该平面，不是全局 taskId；网格数字标识计算块 0/1/2：

```text
+-------+---+---+---+---+---+---+---+---+------------+
| h / w | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | M interval |
+-------+---+---+---+---+---+---+---+---+------------+
| 0     | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | [0,8)      |
| 1     | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | [8,16)     |
| 2     | 1 | 1 | 2 | 2 | 2 | 2 | 2 | 2 | [16,24)    |
| 3     | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | [24,32)    |
+-------+---+---+---+---+---+---+---+---+------------+

M       0                               16  18                          32
block   +-------------------------------+---+---------------------------+
        |               0               | 1 |             2             |
        +-------------------------------+---+---------------------------+
task    +-----------------------------------+---------------------------+
        |                 0                 |             1             |
        +-----------------------------------+---------------------------+

+-----------+-------+-------+-------+-------+--------------+--------------+
| m         | 16    | 17    | 18    | 19    | 20..23       | 24..31       |
+-----------+-------+-------+-------+-------+--------------+--------------+
| (h_i,w_i) | (2,0) | (2,1) | (2,2) | (2,3) | (2,4)..(2,7) | (3,0)..(3,7) |
| block     | 1     | 1     | 2     | 2     | 2            | 2            |
| task      | 0     | 0     | 1     | 1     | 1            | 1            |
+-----------+-------+-------+-------+-------+--------------+--------------+
```

每个计算块内部都遍历有效核深度与 K，再写回自己的有效 M/N。K 循环改变部分和，不推进图中的空间位置。由错误位置 `(h_i,w_i)` 可反查 `m=h_i*Wi+w_i`、所属空间段及计算块，再定位 L1A 起点和窗口参数。

### 7.4 有效范围、物理跨度与槽容量

以一组示例值 `hi=8、wi=9、singleCoreM=45、baseM=48` 投影到一个 dedx 通道平面：

```text
+------+-----------+----------+---------+----------+-------+
| task | dedx rows | target M | valid M | mCompute | pad M |
+------+-----------+----------+---------+----------+-------+
| 0    | h=[0,5)   | [0,45)   | 45      | 48       | 3     |
| 1    | h=[5,8)   | [45,72)  | 27      | 32       | 5     |
+------+-----------+----------+---------+----------+-------+

L0C task 0: [ 45 valid | 3 pad ] -> write M[0,45)
L0C task 1: [ 27 valid | 5 pad ] -> write M[45,72)
```

每份 L0C 槽由 Buffer 计划预先分配，容量须覆盖 `baseM*baseN*accBytes`。任务 1 使用的 M 跨度缩为 32，已分配的槽容量保持不变。对齐位置只在片上占位，不延伸到下一个任务。K/核深度循环只更新这片 dedx 的部分和，不改变任务归属。

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

同一流水内按提交顺序执行，跨流水通过就绪和归还依赖衔接。两路 MTE2、两路 MTE1 是数据依赖分支，同一硬件流水上的指令仍顺序提交。MTE3 用于适配层明确采用的辅助步骤；本主计算的 L0C→GM 通路为 F。

### 8.2 四级流水图

取两个输出块，每块有两次实际 MMAD 贡献。`j00/j01` 属于输出块 0，`j10/j11` 属于输出块 1；贡献可来自 K 分块或有效核深度。为展示重叠，假设两路 L1 每次贡献更新父段、各级均使用两份槽。

```text
+-------+---------+---------+---------+---------+---------+---------+---------+
| stage | t0      | t1      | t2      | t3      | t4      | t5      | t6      |
+-------+---------+---------+---------+---------+---------+---------+---------+
| MTE2  | j00     | j01     | j10     | j11     |         |         |         |
| MTE1  |         | j00     | j01     | j10     | j11     |         |         |
| M     |         |         | j00     | j01     | j10     | j11     |         |
| F     |         |         |         |         | dedx0   |         | dedx1   |
+-------+---------+---------+---------+---------+---------+---------+---------+

t0..t1: fill; t2..t4: overlap; t5..t6: finish pending operations

+---------------+------+------+------+------+
| contribution  | j00  | j01  | j10  | j11  |
+---------------+------+------+------+------+
| L0 input slot | 0    | 1    | 0    | 1    |
| L0C slot      | 0    | 0    | 1    | 1    |
| L0C operation | init | add  | init | add  |
+---------------+------+------+------+------+
```

每格表示可重叠的逻辑阶段，不表示固定周期或实测耗时。输出 0 的 F 等待 j01 完成；此时 M 可以使用另一 L0C 槽计算输出 1。存在更多 K/核深度贡献时，相应延长 M 的累加区间，F 仍只在该输出块完整归约后发生。

L1 父段实际按各自 stepKa/stepKb 轮转，l1-resident 的 L1B 可以跨多个输出块保持不动；不能把图中“每次贡献重装父段”的示意套到驻留路径。

### 8.3 DAG 与 Buffer 存活区间

```text
MTE2_L1A --> MTE1_L0A --+
                        |
MTE2_L1B --> MTE1_L0B --+
                        |
                        |
                        v
                    +---+---+
                    | MMAD  |
                    +---+---+
                        |
                        |
                        v
            +-----------+-----------+
            | Full K / Dk reduction |
            +-----------+-----------+
                        |
                        |
                        v
                  +-----+-----+
                  | F -> dedx |
                  +-----------+

Same slot: last consumer completes --> next overwrite
L1A/L1B consumer: MTE1
L0A/L0B consumer: MMAD
L0C consumer: F
```

| Buffer | 持有范围 | 份数与轮转单位 |
|---|---|---|
| L1A | 当前 dedy 父段到最后一次窗口提取 | al1Pbuffer，按 L1A 段 |
| L1B | 当前权重父段到最后一次 L0B 准备 | bl1Pbuffer，按 L1B 段或驻留身份 |
| L0A/L0B | 当前乘加的输入 | al0/bl0Pbuffer，成对按实际 MMAD 轮转 |
| L0C | 当前输出块的完整归约与写回 | cl0Pbuffer，按完整输出块轮转 |

L1 容量含同时存活的 L1A/L1B 及槽间隔；L0 三块分别核算。Host 数据预算、Kernel 物理末地址和指令访问范围分别检查。

### 8.4 三个模板的差异

| 模板 | DAG 与 Buffer 行为 |
|---|---|
| standard | 输出任务独立；父段按各自 step 推进，任务内可保留已覆盖完整 K 的 L1B |
| small-shape | 主通路相同；任务范围与 baseM 可以不同，短任务不一定使用第二份槽 |
| l1-resident | L1B 对多个输出块形成多消费者节点；L1A 继续更新，每个输出块仍独立初始化 L0C |

完整 L1B 要求 `baseN>=singleCoreCin`、`baseK*curStepKb>=singleCoreCout*hk*wk`、`bl1Pbuffer=1`、`dk=singleIterateDk=1`。跨任务保留还要求 `enableFullLoad` 对应的当前权限有效、group=1、weight 地址和通道/核/父布局身份一致。

下面沿用两输出块、每块两次 K 贡献，取 dk=1。L1B_R 表示覆盖完整 K 的同一份 L1B；L1A 每次贡献更新父段，L1A 和 L0 输入使用双槽，L0C 按输出块轮转：

```text
+-------+---------+---------+---------+---------+---------+---------+---------+---------+
| stage | t0      | t1      | t2      | t3      | t4      | t5      | t6      | t7      |
+-------+---------+---------+---------+---------+---------+---------+---------+---------+
| MTE2  | L1B_R   | L1A_00  | L1A_01  | L1A_10  | L1A_11  |         |         |         |
| MTE1  |         |         | j00     | j01     | j10     | j11     |         |         |
| M     |         |         |         | j00     | j01     | j10     | j11     |         |
| F     |         |         |         |         |         | dedx0   |         | dedx1   |
+-------+---------+---------+---------+---------+---------+---------+---------+---------+

L1B_R: load at t0; hold at t1..t5; release after the last MTE1 read at t5.
L1A:   load/extract/release each parent segment independently.
L0B:   Load2D selects the current K block from the same L1B_R.
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

各级槽按自己的频率轮转。K 增加推进输入槽，L0C 按完整输出块轮转；SDK 队列和事件见 [同步实现](../../adapters/ascendc/input/sync-implementation.md)。

```text
Same L1A/L1B slot: MTE2 write -> all MTE1 reads complete -> next MTE2 write
Same L0A/L0B slot: MTE1 write -> MMAD read completes     -> next MTE1 write
Same L0C slot:    first MMAD -> remaining MMADs -> F read completes -> next output
```

就绪防止读到未完成的数据，归还防止覆盖仍被消费的数据。首次令牌由队列/事件所有者建立，末次只等待实际提交的操作；空父段、跳过的深度和无后继输出不产生虚假的等待或预取。

当前 Ascend C 映射采用 CO1 队列交接 M→F、F→M；其 MMAD `unitFlag=0`。同步参数和事件资源统一按适配层实现。

### 8.6 可选 bias 通路与无贡献输出（转置卷积）

带 bias 的 DAG 在两路操作数之外增加 bias 支路：

```text
dedy / GM --MTE2--> L1A --MTE1 / img2col--> L0A ------+
                                                      |
weight/GM --MTE2--> L1B --MTE1 / weight_rot--> L0B ---+
                                                      |
bias / GM --MTE2--> biasL1Que_ --MTE1--> biasBTQue_ --+
                                                      |
                                                      |
                                                      v
                                                +-----+-----+
                                                | M -> L0C  |
                                                +-----+-----+
                                                      |
                                                      |
                                                      v
                                             +--------+--------+
                                             | F -> dedx / GM  |
                                             +-----------------+
Bias enters the first actual MMAD of each output block only.
```

| Buffer | 内容 | 有效期与复用 |
|---|---|---|
| L1A/L1B、L0A/L0B/L0C | 公共计算对象 | 使用公共 DAG、五路槽和事件 |
| biasL1Que_ | 当前 bias 通道段，64B 对齐 | 全量模式保留到身份变化；分块模式在复制到 BT 后归还 |
| biasBTQue_ | MMAD 可读取的 FP32 bias | 覆盖所用通道段；最后一次 MMAD 读取后才归还 |

`isBiasFullLoad=1` 时搬入当前任务的 dedx 通道范围，N 子块从 BT 取相应偏移；为 0 时按当前 N 块搬入。该标志与权重 `enableFullLoad` 分别管理。三种 Tiling 模板均计入 bias L1 和 BT 预算。

带 bias 的流水在公共四级流水之上增加 bias 首次准备与纯 bias 输出：

```text
Bias preparation:
GM --MTE2--> L1 --MTE1--> BT (ready)

Output block 0: L0 ready -> M(first, bias) -> M(other K/kd) -> F
Output block 1: L0 ready -> M(first, bias) -> M(other K/kd) -> F
BT may be reused; reset the bias-once state for each output block.

Bias-only block: MTE1(LoadL0Zero) -> M(0*0 + bias) -> F

Bias segment change:
last M read completes -> release old BT/L1 bias -> load next bias segment
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

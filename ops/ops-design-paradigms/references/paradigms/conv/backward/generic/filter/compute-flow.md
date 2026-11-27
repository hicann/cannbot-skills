# 卷积反向权重梯度分支：计算流程

本文回答数据怎样存放、窗口怎样展开、任务怎样分配、归约怎样执行和 Buffer 怎样交接。计算模型见 [scene.md](scene.md)，切分规则见 [tiling/tiling.md](tiling/tiling.md)，API 字段见 [指令映射（Ascend C）](../../adapters/ascendc/filter/instruction-mapping.md) / [DSL](../../adapters/dsl/filter/instruction-mapping.md)。

沿用 x、dedy、dedw、dedy_flat、x_win。A 路为 dedy，B 路为 x；L1A/L1B 保存父段，L0A/L0B 为本次计算操作数，L0C 保存 FP32 部分和。轴序均从慢到快，示意小分形只说明顺序。

本页主要展开 MN_StreamK 的主轮 im2col 数据流，具体分形图采用 FP16/BF16。FP32 的物理通道块及专用搬运见 [指令映射 §9](../../adapters/ascendc/filter/instruction-mapping.md#9-fp32-输入与-hf32-计算模式)；尾轮 K 切分见 [MN_StreamK](tiling/templates/mn-streamk/tiling.md)，Winograd 使用[独立变换流水](../../adapters/ascendc/filter/winograd-implementation.md)。

## 1 计算模型速览

### 1.1 从正向连接得到 dedw

固定 g、co、ci 和原核坐标，每个梯度元素归约全部 batch 和 dedy 空间位置：

$$
\mathrm{dedw}_g[co,ci,\mathbf k]
=\sum_{n,\mathbf o}\mathrm{dedy}_g[n,co,\mathbf o]\,
x_g[n,ci,\mathbf o\odot\mathbf s+\mathbf k\odot\boldsymbol\delta-\mathbf p^-].
$$

越界 x 采样为零。固定 kd 后，dedy_flat 的 K 列和 x_win 的 K 行按同一个 `(oh,ow)` 配对；batch×dout 在矩阵乘外累加。矩阵映射图（A=dedy_flat、B=x_win、Y=dedw 块）见 [language.md](language.md) §2 的 GEMM 映射。

img2col 将 x 的每个正向窗口排成 x_win 的一行。相邻窗口可重复读取同一 x 元素；窗口矩阵可以在搬运时形成，不要求预先存入 GM。

## 2 数据流总览

### 2.1 阶段与指令职责

| 阶段 | 通路 | 转换 |
|---|---|---|
| LoadToA1 / MTE2 | dedy GM→L1A | 当前 co/输出行带，ND2NZ 或 DN2NZ |
| LoadToB1 / MTE2 | x GM→L1B | 原 x 深度与行带，通道分块及布局转换 |
| LoadL12L0a / MTE1 | L1A→L0A | 2D 分形转置，形成 NZ(M,K) |
| LoadL12L0b / MTE1 | L1B→L0B | 正向窗口提取，形成 ZN(K,N) |
| ExecuteMmad / M | L0A×L0B→L0C | 段内 FP32 乘加 |
| LoadL0c2Gm / F | L0C→dedw GM | 按输出格式恢复核/通道轴并原子累加 |
| InitOutput / AIV | 零→dedw GM | 在主计算前建立原子累加基值 |

物理容量、有效长度和地址单位分别记录：

| 对象 | 有效范围 | 存储范围 |
|---|---|---|
| L1A | 本次 co 与输出行带 | M/K 分形补齐；切宽可能多搬整行 |
| L1B | 有效 x 行带与 ci | C0 补齐、父行带跨度和裁边后的宽度 |
| L0A/L0B | `baseUseM_/baseUseN_/baseUseK_` | 各轴分形对齐后的占用 |
| L0C | 当前段的有效输出块 | M/N 对齐跨度；跨本段全部 K 保留 |
| dedw | 实际 Cout、CinG 和核位置 | 无内部补齐通道写回 |

### 2.2 dedy 侧：GM → L1A → L0A

```text
dedy(GM, NCDHW/NDHWC)
  │  DN2NZ / ND2NZ 布局转换搬运（随路 ND→NZ，装入当前 co 段与输出行带）
  ▼
L1A [Cout1, K1, K0, Cout0]    <-- dedy_flat 父段（不做滑窗展开）
  │  LOAD2D（分形内两轴交换 + 分形重排，完成转置）
  ▼
L0A [K1, Cout1, Cout0, K0]    <-- NZ(M,K)：行=co（M）、列=输出位置（K）
```

FP16/BF16 的常规 L1A 父布局为 `(Cout1,K1,K0,Cout0)`；Load2D 交换分形内两轴并重排分形，得到 `(K1,Cout1,Cout0,K0)`。K 指输出位置，不是核位置。

### 2.3 x 侧：GM → L1B → L0B

```text
x(GM, NCDHW/NDHWC)
  │  DN2NZ / ND2NZ 布局转换搬运（只搬有效深度/行带；拆核路径按原核坐标采样分段）
  ▼
L1B [Cin1, hiCopyLen, realWi, Cin0]    <-- 原始 x 行带（未做滑窗展开！）
  │  LOAD3D（按正向 stride/pad/dilation 取窗 = 硬件 im2col）
  ▼
L0B [K1, Cin1, kh, kw, Cin0, K0]       <-- x_win：ZN(K,N)，K=输出位置、N=(ci1,kh,kw,ci0)
```

固定实际 x 深度，L1B 的典型视图为 `(Cin1,hiCopyLen,realWi,Cin0)`。Load3D 按正向 stride/dilation 和边界提取窗口，组织为本轮 x_win。普通路径和拆核路径的差别在窗口由哪一阶段展开。

### 2.4 输出侧：MMAD → L0C → dedw GM

```text
L0A [K1,M1,M0,K0] -----+
                       |
L0B [K1,N1,N0,K0] -----+--> MMAD -> L0C（FP32 段内部分和）

段内首次乘加初始化；L0C --FIXPIPE（原子累加，只写有效 M/N）--> dedw(GM, FP32)
```

每个归约段第一次实际 MMAD 初始化 L0C，其余贡献继续累加。段末写回时 GM 中已有先前段结果，因此使用原子累加；新段不沿用旧 L0C。

### 2.5 两路数据流总图

```text
前置：AIV Fill 清零有效 dedw -> 全员汇合 -> 发 MODE2 旗标 -> AIC 等待
（两阶段衔接见 §8.2；段内首次乘加初始化 L0C）

A 路（dedy）                                B 路（x）
+-----------------------------+             +-------------------------+
| GM -> L1A（dedy_flat 父段） |             | GM -> L1B（x 有效行带） |
+--------------+--------------+             +------------+------------+
               | LOAD2D 转置                             | LOAD3D 正向窗口 im2col
               v                                         v
+--------------+--------------+             +------------+------------+
| L0A（NZ[M,K] 分形）         |             | L0B（ZN[K,N] 分形）     |
+--------------+--------------+             +------------+------------+
               |                                         |
               +--------------------+--------------------+
                                    |
                                    v
                    +---------------+---------------+
                    | MMAD -> L0C（FP32 段部分和）  |
                    +---------------+---------------+
                                    | FIXPIPE / 原子累加
                                    v
                           +--------+--------+
                           | GM（dedw,FP32） |
                           +-----------------+
```

初始化与主计算分阶段衔接。MN_StreamK、streamkType=0、原始组模式下，AIV 清零、汇合并通知后不做主轮数值工作；段内乘加与段间原子合并由 AIC 完成。

## 3 数据格式与存储轴序

### 3.1 NCDHW、NDHWC 与 DHWCN

NCDHW 使同一通道的空间体连续，NDHWC 使同一空间点的通道连续。dedw 的 N/C 分别对应 Cout/CinG，DHWCN 则让全局 co 连续：

```text
特征张量（固定 batch）：
NCDHW: [c: d -> h -> w] [下一 c: d -> h -> w] ...
NDHWC: [d,h,w: c ->]    [下一空间点: c ->] ...

dedw 轴序:
NCDHW: (co,ci,kd,kh,kw)
NDHWC: (co,kd,kh,kw,ci)
DHWCN: (kd,kh,kw,ci,co)
```

连续存储的元素步幅如下；每项与该行的轴序一一对应：

| 对象/格式 | 轴序 | 元素步幅 |
|---|---|---|
| x / NCDHW | `(batch,cin,di,hi,wi)` | `(cin*di*hi*wi,di*hi*wi,hi*wi,wi,1)` |
| x / NDHWC | `(batch,di,hi,wi,cin)` | `(di*hi*wi*cin,hi*wi*cin,wi*cin,cin,1)` |
| dedy / NCDHW | `(batch,cout,dout,ho,wo)` | `(cout*dout*ho*wo,dout*ho*wo,ho*wo,wo,1)` |
| dedy / NDHWC | `(batch,dout,ho,wo,cout)` | `(dout*ho*wo*cout,ho*wo*cout,wo*cout,cout,1)` |
| dedw / NCDHW | `(cout,cinG,dk,hk,wk)` | `(cinG*dk*hk*wk,dk*hk*wk,hk*wk,wk,1)` |
| dedw / NDHWC | `(cout,dk,hk,wk,cinG)` | `(dk*hk*wk*cinG,hk*wk*cinG,wk*cinG,cinG,1)` |
| dedw / DHWCN | `(dk,hk,wk,cinG,cout)` | `(hk*wk*cinG*cout,wk*cinG*cout,cinG*cout,cout,1)` |

元素偏移为各坐标与对应步幅乘积之和；字节地址再乘该对象的元素字节数。x/dedy 当前为 2B，dedw 为 4B。

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

读取 dedy 时，以输出空间为行、Cout 为列：NDHWC 为 ND，NCDHW 为 DN。计算的 dedy_flat 以 Cout 为行、输出空间 K 为列，由 Load2D 完成相应分形转置。x_win 以 K 为行、窗口通道 N 为列，由 Load3D 形成 ZN。

filter 的 K 是输出位置，不采用 input 的“通道×核高宽”定义。

### 3.3 五维张量与片上视图

固定组、任务 kd 与实际 batch/od 后，两路五维张量分别选择深度切片，再形成父段：

```text
dedy[batch,Cout,Dout,Ho,Wo]
  | 选择 batch / group / od / co 范围
  v
L1A(Cout1,K1,K0,Cout0) --Load2D--> L0A(K1,Cout1,Cout0,K0)

x[batch,Cin,Di,Hi,Wi]
  | id = od*strideD + kd*dilationD - padFront
  v
L1B(Cin1,hiCopyLen,realWi,Cin0) --Load3D--> L0B(K1,Cin1,kh,kw,Cin0,K0)
```

父段带有自己的起点、有效尺寸和实际跨度。L1A 的 alignedL1UseKa 可能大于本轮 kal1 的有效部分；L1B 中 bL1cin1CopyLen 在 Kernel 中表示实际搬入的通道数，Host 预算中的 bL1Cin1CopyLen 表示通道块数，二者相差 C0，须按各自名称与单位换算。

### 3.4 L0A、L0B 与 L0C 的计算分形

FP16/BF16 下 `M0=N0=K0=C0=16`。下表均为物理轴序：

| 对象 | 逻辑矩阵 | 慢→快物理轴序 |
|---|---|---|
| L0A | dedy_flat[M,K] | `(K1,M1,M0,K0)`，NZ |
| L0B | x_win[K,N] | `(K1,N1,N0,K0)`，ZN |
| L0C | dedw 段部分和[M,N] | `(N1,M1,M0,N0)`，NZ |

将 N1 展开，L0C 对应 `(Cin1,kh,kw,Cout1,Cout0,Cin0)`；固定 kd 的输出还须恢复外部格式中的 kd 轴。逻辑矩阵 `[M,N]` 的书写顺序不能替代物理分形顺序。

```text
+----------+--------+---------------------+---------------+
| Buffer   | 行     | 列                  | 物理轴序      |
+----------+--------+---------------------+---------------+
| L0A / NZ | M = co | K = oh*wo+ow        | (K1,M1,M0,K0) |
| L0B / ZN | 同一 K | N = (ci1,kh,kw,ci0) | (K1,N1,N0,K0) |
| L0C / NZ | M = co | 同一 N              | (N1,M1,M0,N0) |
+----------+--------+---------------------+---------------+
```

这里的 `(ci1,kh,kw,ci0)` 为 N 的逻辑顺序。非整窗 baseN 可以截在相邻核位置或通道块之间，必须保留 head_/tail_/cinRemainLen_，由写回逻辑拆成合法片段。

## 4 img2col 与窗口归约

### 4.1 M/N/K 与窗口归约顺序

固定 kd 时，K 位置为 `k=oh*wo+ow`。N 位置为：

`n=((ci1*hk+khIdx)*wk+kwIdx)*C0+ci0`。

解码 N 时，先取 `ci0=n%C0`，再将 `n//C0` 解为 kwIdx、khIdx、ci1。A 的 K 列与 B 的 K 行必须对应同一批次、同一输出深度和同一输出平面位置。

```text
固定任务 (g,kd,co 范围,ci 窗口范围):

  batch / od 段
    -> ho 段
      -> 实际 batch / od
        -> 输出宽片
          -> K 块: oh / ow 位置

N 位置: ci1 -> kh -> kw -> ci0（最快）
A 的 K 列与 B 的 K 行对应同一实际 batch / od / oh / ow。
```

batch×dout 为外层归约域；ho 分段和 wo 切宽只是将同一输出网格分成不重叠区域。补齐通道只占内部存储，不增加有效乘积。

### 4.2 三维窗口与输出平面

固定组、kd、batch 和输出深度 od。每个 `(oh,ow)` 同时提供一个 dedy 值和一个 x 窗口；将这些位置按 K 排列后，两路相乘得到当前批深度对 dedw 的贡献。

```text
dedy 平面: (co,oh,ow) -> dedy_flat[M,K]
x 平面: 同一 (oh,ow) 的正向窗口 -> x_win[K,N]

dedw[M,N] += dedy_flat[M,K] * x_win[K,N]

一个 K 位置:
  dedy_flat[:,K]  外积  x_win[K,:]  -> 对 dedw 的贡献

下一 batch / od:
  新输入值，同一输出核坐标
```

M 对应 co，N 按 `(ci1,kh,kw,ci0)` 排列。核位置保持正向顺序，batch 和 dout 在矩阵乘外累加；没有插零或 rot180。

### 4.3 LOAD3D（L1B→L0B，硬件 im2col）

**im2col 原理**——按正向 stride/pad/dilation 在 x 上取窗，把滑窗展开写成矩阵：每个输出位置的窗口展平成 B 矩阵的一行：

```
L1B 中的 x 行带（hiCopyLen × realWi）   正向感受野(2×2)   L0B 上的 B 矩阵（K × N）
+---------------------+                 -- im2col -->     行 = 输出位置 k = oh·wo+ow
| .  .  .  .  .  .  . |                                   列 = (ci, kh, kw) 展平
| . [.  .] .  .  .  . |                                   +--------+-----------------------------+
| . [.  .] .  .  .  . |                                   | 窗口行 | 展开后的元素                |
| .  .  .  .  .  .  . |                                   +--------+-----------------------------+
+---------------------+                                   | k      | x(k,ci,0,0) x(k,ci,0,1) ... |
窗口沿 W、H 按正向 stride 滑动                            | k+1    | x(k+1,ci,0,0) ...           |
                                                          +--------+-----------------------------+
```

LOAD3D 一条指令完成"取窗 → 展开 → 摆成 ZN 分形"三件事，目标在 L0B。

**相邻窗口的元素复用**（单位 stride/dilation、单通道小图；下一窗口右移一列，与前窗口共享一列元素）：

```text
L1B / x 行带          L0B / ZN：同一通道的选中元素
+---------------+     行 = 输出位置 K；列标签 = (kh,kw)
| .  .  .  .  . |     +--------+-------+-------+-------+-------+
| . [a  b] e  . |     | window | (0,0) | (0,1) | (1,0) | (1,1) |
| . [c  d] f  . |     +--------+-------+-------+-------+-------+
| .  .  .  .  . |     | 当前 k | a     | b     | c     | d     |
+---------------+     | 下一 k | b     | e     | d     | f     |
窗口 -> 下一窗口      +--------+-------+-------+-------+-------+
                      完整 N 序：(Cin1,kh,kw,Cin0)
                      逻辑窗口 -> 物理分形
```

实际指令保留规格中的 stride/dilation。N 中还包含同一核位置的其他 Cin0。dedy_flat 的 K 列与 x_win 的 K 行对应同一个输出位置，随后在 L0C 中累加。

配置分两组（位域下发）：

| 配置组 | 内容 | 语义 |
|---|---|---|
| 窗口几何 | 当前 x 行带高宽、pad、原 stride/dilation、hk/wk、channelSize | "在哪张输入行带上取什么窗口" |
| 块定位 | mStartPt/mExtension、kStartPt/kExtension、目的分形步幅 | "这次取出哪个 [K0,N0] 子块" |

**块定位参数的精确语义**——⚠️ 本指令的 m/k 参数与 GEMM 的 M/K **同名不同义**（沿用窗口指令的行命名：m 组作用于**输出位置（K 方向）**、k 组作用于**窗口通道/核位置（N 方向）**，单位错误是本分支滑窗类 bug 的高发点）：

| 参数 | 单位/语义 | 取值规则 |
|------|----------|---------|
| mStartPt / mExtension | **输出网格局部起点** / 本轮展开的输出位置数 | 起点=本段首输出位置的行内列号（跨行不改变取值规则，**不要**手工乘 stride）；长度=本轮有效 `baseUseK_`（真值） |
| kStartPt / kExtension | 相对**当前父段**的窗口通道起点 / 本轮窗口通道数 | `baseUseN_`；父段切换时按新段原点重算，**不是**全局 N 坐标 |
| channelSize | 当前窗口覆盖的实际 ci 通道跨度 | **不是**窗口元素总数，也**不是**分形数 |
| padList | 当前 x 父段的左、右、上、下边界 | 元素序固定 {left, right, top, bottom}；bottom 恒填兼容值 255 表示**下边界不设防**（真实行数由每次搬运更新的 l1H 覆盖），上界必须精确 |
| filter/stride/dilation | 原正向采样几何 | 拆核路径由 MTE2 定位原 kh/kw，MTE1 使用单位窗口 |
| repeat / dstStride | 目的步幅按 **N0 分形数**，由本轮 N 范围换算 | dstStride=本轮 N0 分形个数；repeatTime=1 |

**K 位置跨多输出行的语义**（对应本分支的"一般 K 模式"）：

- mExtension 允许 **> 当前宽片宽度**：输出位置序列跨输出行，行切换由硬件在 x 行带内自动完成，软件只需保证行带反推（§5.3.1）覆盖本段全部窗口的输入行；
- mStartPt 仍取**本段首输出位置的行内列号**（跨行不改变取值规则）；
- 跨行与宽片尾叠加时：mExtension 取本段真值，MMAD 的 k 取真值，FIXPIPE 按有效 M/N 写出——有效扩展长度与分形存储跨度分别传递，无额外修正。

**参数限制（超限走拆核/切宽路径）**——范围来自当前窗口指令映射（[tiling](tiling/tiling.md) 与适配层），目标 SDK 更换时重新核实：

| 限制 | 值 | 超限出路 |
|------|----|---------|
| stride（H/W 每维） | ≤ 63 | 拆核路径（isSplitKernelHW） |
| dilation / hk / wk（每维） | ≤ 255 | 拆核路径 |
| pad（H/W 四侧） | ≤ 255 | 拆核路径 |
| hk·wk·C0 | ≤ 65536 | 拆核路径 |
| wo | ≤ 512 | 切宽（splitWo 按 128 分段，见 §5.2.1） |
| hi / wi（行带高宽） | ≤ 32767 | 行带预算上限，超限由 Tiling 准入拒绝 |

FP16/BF16 使用配套的一组 Repeat/LoadData 调用；具体字段和原窗口、拆核窗口的差异见 [指令映射](../../adapters/ascendc/filter/instruction-mapping.md#5-l1b-到-l0b)。

### 4.4 核深度与批深度的处理

对当前 batchDoutIdx，`n=batchDoutIdx//dout`、`od=batchDoutIdx%dout`；固定任务 kd 后，x 深度为 `od*strideD+kd*dilationD-padFront`。跨 dout 边界时 batch 增 1，x 深度重新从该 batch 的输出深度关系计算。

深度索引无效时，不提交越界 GM 访问。整段无效时跳过整段，部分无效时只跳过对应批深度；段内后续有效批深度继续归约。isFirstMmad 只在实际提交计算后变为 false。

## 5 数据搬运与写回

### 5.1 指令与单位总表

| 阶段 | 长度或步幅单位 | 本分支约定 |
|---|---|---|
| GM/LocalTensor 下标 | 该张量元素 | x/dedy 2B，L0C/dedw 4B |
| ND2NZ/DN2NZ | 有效维度为元素；各 stride 按相应 API 定义 | 保留原 cin/cout 与完整空间步幅 |
| Load2D | m/k 步数为分形坐标；src/dstStride 为分形间距 | 16×16 输入分形为 512B |
| Load3D | 窗口位置与扩展长度为元素/采样点 | K 输出位置与 N 核窗口分别设置 |
| MMAD | 本轮有效 M/N/K | L0 占用另外按分形对齐 |
| Fixpipe | 目的步幅为 FP32 元素；源步幅按 64B 单位 | 64B 为 16 个 float，不沿用 32B 搬运单位 |

具体字段与数值例子见适配层。分形数、元素数、字节数在每一处转换前明确写出。

### 5.2 dedy：GM→L1A→L0A

#### 5.2.1 输出行带与切宽

一次父段的理论 K 长度为 `kal1=stepKa*baseK`。Host 的覆盖行数预算使用：

| kal1 与当前输出宽 W 的关系 | 覆盖行数 |
|---|---|
| 互有整除关系 | `ceilDiv(kal1,W)` |
| 不整除且 kal1>W | `kal1//W+2` |
| 不整除且 kal1<W | 2 |

kaStepIdx 为 A 父段号，其元素起点为 `kaStepIdx*kal1`；kal1Use 为本父段在当前宽片内的有效 K 长度。Kernel 按实际起点计算 `hoStart=floor(kaStepIdx*kal1/singleShapeWo)`、`hoEnd=ceilDiv(kaStepIdx*kal1+kal1Use,singleShapeWo)`，只搬 `hoEnd-hoStart` 行。预算与实际长度分别保存。

预算按最坏跨行覆盖保留，实际搬运按当前父段的起点与 kal1Use 重算。切宽后的行跨度使用当前 singleShapeWo，原 GM 行步幅仍保留完整 wo。

#### 5.2.2 布局转换与计算分形

NCDHW 使用 DN2NZ，源通道步幅是 `dout*ho*wo`；NDHWC 使用 ND2NZ，源行步幅是总 cout。两者形成相同 L1A 父布局，再由 Load2D 转置进入 L0A。

```text
NCDHW / DN -- DN2NZ --+
                      +--> L1A(Cout1,K1,K0,Cout0)
NDHWC / ND -- ND2NZ --+            |
                                   | Load2D: 转置并重排
                                   v
                        L0A(K1,Cout1,Cout0,K0)
```

切宽后 alignedL1UseKa 取实际搬入行数与宽度形成的父跨度，Load2D 再加行内 K 起点。不能用最大槽长或原完整 wo 代替该跨度。

### 5.3 x：GM→L1B→L0B

#### 5.3.1 从输出行带反推 x 区域

普通路径的覆盖范围为：

`hiCal=(hoCal-1)*strideH+(hk-1)*dilationH+1`，
`splitWi=(splitWo-1)*strideW+(wk-1)*dilationW+1`。

拆核路径每次处理一个原核坐标，Host 预算为 `hiCal=hoCal、splitWi=splitWo`，原 stride/dilation 和原核起点由分段搬运表达。

```text
输出行带
  | stride、核跨度与 dilation
  v
所需 x 行带: [逻辑起点, 逻辑终点)
  | 与 [0,hi) 取交集
  v
GM 拷贝: 只搬有效行；缺失的上下行由窗口补边表达
```

起点和交集先按有符号宽整数计算。只提交有效 x 区域的搬运；pad 用于窗口补零，不能用负 srcStride 修补越界。

#### 5.3.2 原窗口与拆核窗口

```text
+----------+----------------------------+-------------------------------+
| 路径     | MTE2 / L1B                 | MTE1 / L0B                    |
+----------+----------------------------+-------------------------------+
| 普通窗口 | 拷贝有效原始 x 行带        | Load3D(hk,wk,stride,dilation) |
| 拆核     | 选原 kh/kw；按 stride 采样 | Load3D(1,1,1,1)               |
+----------+----------------------------+-------------------------------+

两条路径 -> 同一 (K,N) 坐标上的相同 x_win 值。
```

普通路径在 Load3D 中完成整个核窗口；拆核路径在 GM→L1B 的选择中落实原 kh/kw，Load3D 只提取单位核。更换路径时，原核位置、源地址、窗口参数和目标 N 分形位置必须一起更新。

普通父段保留原 x 的连续行带，pad 使用原空间点单位；拆核父段只保存固定核位置的有效采样点，pad 改用采样网格的点数。两条路径最终对应相同的 `(K,N)`，具体裁剪、行内起点和数值示例见 [指令映射（Ascend C）](../../adapters/ascendc/filter/instruction-mapping.md#3-x-到-l1b) / [DSL](../../adapters/dsl/filter/instruction-mapping.md)。

### 5.4 MMAD 与 dedw 写回

#### 5.4.1 段内初值与段间合并

```text
dedw GM 初始: 0

段 0: 初始化 L0C -> 累加全部有效贡献 -> 原子累加到 dedw
段 1: 初始化 L0C -> 累加全部有效贡献 -> 原子累加到 dedw
段 2: 初始化 L0C -> 累加全部有效贡献 -> 原子累加到 dedw

dedw GM 最终: 段0 + 段1 + 段2
```

MMAD 使用当前计算 baseUseM_/N_/K_；内部槽占用另行对齐。适配实现会在单个有效 co 时临时把计算 M 扩到 16，写回前恢复有效 M；数学输出范围始终不变。isFirstMmad 跨本段的 batch/dout、切宽和 K 循环保留；段末将完整部分和原子累加到 dedw，下一段重新置 true。

#### 5.4.2 从 L0C 恢复三种输出格式

```text
+--------------+-------------------------------+
| 阶段/格式    | 慢 -> 快轴序（固定任务 kd）   |
+--------------+-------------------------------+
| L0C          | (Cin1,kh,kw,Cout1,Cout0,Cin0) |
| dedw / NCDHW | (co,ci,kd,kh,kw)              |
| dedw / NDHWC | (co,kd,kh,kw,ci)              |
| dedw / DHWCN | (kd,kh,kw,ci,co)              |
+--------------+-------------------------------+
```

L0C 的 N 包含 `(ci1,kh,kw,ci0)`，不能将它作为连续 CinG 直接平铺到外部权重格式。普通核 NCDHW 写回需交换核位置与 ci0；NDHWC 保持核位置在 ci 之外；DHWCN 还使全局 co 成为最快轴。1×1×1 使用简化矩阵写回，普通核和非整窗 N 另有分发。

ci 尾分形按 `min(C0,CinG-ciStart)` 写回每个有效核位置，补齐通道不落 GM；dk>1 时保留完整核深度步幅，避免写到相邻任务。

### 5.5 原张量地址与三维步幅

用 `c_o=g*CoutG+co`、`c_i=g*CinG+ci` 表示全局通道；以下为元素偏移：

| 张量 | NCDHW | NDHWC |
|---|---|---|
| dedy | `((((n*cout+c_o)*dout+od)*ho+oh)*wo+ow)` | `((((n*dout+od)*ho+oh)*wo+ow)*cout+c_o)` |
| x | `((((n*cin+c_i)*di+id)*hi+ih)*wi+iw)` | `((((n*di+id)*hi+ih)*wi+iw)*cin+c_i)` |
| dedw | `((((c_o*cinG+ci)*dk+kdIdx)*hk+khIdx)*wk+kwIdx)` | `((((c_o*dk+kdIdx)*hk+khIdx)*wk+kwIdx)*cinG+ci)` |

DHWCN 的 dedw 偏移为 `((((kdIdx*hk+khIdx)*wk+kwIdx)*cinG+ci)*cout+c_o)`。

```text
offsetA = batch 基址 + group/co 基址 + od 平面 + ho 行
offsetB = batch 基址 + group/ci 基址 + 有效 id 平面 + 有效 ih 行
offsetC = group/co 基址 + ci 基址 + 固定 kd 切片

段切换 batch / od / ho -> 重算 offsetA 与 offsetB
同一任务              -> 保持 offsetC
```

offsetC_ 只随任务通道、组和核深度变化；batch/dout/ho 段切换重算 offsetA_/offsetB_。带 pad 的采样坐标先按有符号宽整数判断，只有确认有效后才能提交 GM 访问。

## 6 多核任务与输出归属

### 6.1 按 M/N 基本块分配任务

`mCnt=ceilDiv(mValue,singleCoreM)`、`nCnt=ceilDiv(nValue,singleCoreN)`，`totalCnt=mCnt*nCnt*dk*G`。每个基本块绑定一个 g、kd 和 M/N 范围：

| 任务 | co 范围（M） | ci/核窗口范围（N） | 固定轴 |
|---|---|---|---|
| (m0,n0) | [0, singleCoreM) | 首组完整通道窗口 | g、kd |
| (m0,n1) | 同一 co 段 | 下一组完整通道窗口 | g、kd |
| (m1,n0) | 下一 co 段 | 首组完整通道窗口 | g、kd |

任务四元组在 M×N 输出平面上的投影（固定 g、kd 后的一层；每个格子 = 一个基本块任务，M/N 尾任务按"总长−起点"收缩）：

```text
M(co)
  ^        n0              n1              n2
  |
m1|   task(m1,n0)    task(m1,n1)    task(m1,n2) ...
m0|   task(m0,n0)    task(m0,n1)    task(m0,n2) ...
  +------------------------------------------------> N(ci×核窗口)

idx = (((mIdx*nCnt+nIdx)*dk+dkIdx)*G+gIdx)
core assignment: idx = block_idx + round*coreNum
```

```cpp
for (uint64_t idx = block_idx; idx < totalCnt; idx += coreNum) {
    uint32_t gIdx  = idx % G;
    uint32_t dkIdx = (idx / G) % dk;
    uint64_t nIdx  = (idx / G / dk) % nCnt;
    uint64_t mIdx  = idx / (G * dk * nCnt);
    mCoreIndx_ = mIdx;
    nCoreIndx_ = cinCoreIndx_ = nIdx;
    CalSingleBasicBlock(dkIdx, gIdx);
}
```

这是 MN_StreamK、streamkType=0 的任务解码。iterateOrder 控制任务内部 M/N 子块遍历和父段复用，不改变上述四元组解码。

### 6.2 输出互斥与尾任务

同一 dedw 元素只归属一个任务；每个任务自行归约所有 batch×dout×ho×wo。原子累加用于合并任务自己的多段结果，不用于替代错误的跨任务重叠。

尾块长度使用“总长减起点”，整除时仍为完整块。co/ci 的数学尾长按实际 CoutG/CinG 收缩，内部 mValue/nValue 的补齐范围不写回。物理核可以处理多轮任务，空任务核在完成初始化协议后退出。

## 7 单核分块与迭代

### 7.1 任务、输出块与完整归约

```text
dedw
`-- 任务: g / kd / co 范围 / ci 窗口范围
    `-- 归约段: batch*dout 范围 / ho 范围
        `-- M/N 输出子块: baseM x baseN
            `-- 实际 batch / od 与输出宽片
                `-- K 子块: 配对的 L0A / L0B
                    -> 累加到同一归约段的 L0C
        -> Fixpipe: 把完整段结果原子累加到 dedw
```

singleCoreN 描述任务通道窗口，baseN 描述 L0 子块；一项任务可以有多个 N 子块。每个归约段的 L0C 必须包含当前输出块在该段内的全部有效贡献，再进入 Fixpipe。

### 7.2 L1 父段与 K 子块

两路分别按 stepKa/stepKb 换驻，父段状态包括范围、实际行带、槽号和最后一次消费者。较大的 step 为较小 step 的整数倍，末段实际轮数另算。

```text
+----------------------+----+----+----+----+----+
| K 块号               | 0  | 1  | 2  | 3  | 4  |
+----------------------+----+----+----+----+----+
| A 父段 / stepKa=2    | A0 | A0 | A1 | A1 | A2 |
| B 父段 / stepKb=4    | B0 | B0 | B0 | B0 | B1 |
| A 内局部 K / baseK   | 0  | 1  | 0  | 1  | 0  |
| B 内局部 K / baseK   | 0  | 1  | 2  | 3  | 0  |
+----------------------+----+----+----+----+----+
```

每轮 k 表示当前 ComputeLoop 内的全局 K 块号：`kStart=k*baseK`，`baseUseK_=min(baseK,段内K-kStart)`。父段号为 `kaStepIdx=k/stepKa`、`kbStepIdx=k/stepKb`，父段内块号为 `kaIdx=k%stepKa`、`kbIdx=k%stepKb`；相对父段起点为 `kaIdx*baseK` 与 `kbIdx*baseK`。切宽的行内起点另行折算。

### 7.3 归约分段：精度与资源

CalcReduceSegments 使用 `SINGLE_BLOCK_ADD_THRESHOLD=65536`：

| 量 | 计算 |
|---|---|
| segmentsNDH | `ceilDiv(65536,wo)` |
| reduceSegmentsH | `min(segmentsNDH,ho)`；拆核路径再与 256 取小 |
| reduceSegmentsND | `min(ceilDiv(segmentsNDH,reduceSegmentsH),batch*dout)` |

阈值是段长选择依据，不是严格上限。wo 大于阈值时，一行本身就会超过它；使用向上取整也可能使段乘积略大于 65536。

```cpp
for (uint64_t bd = 0; bd < uint64_t(batch)*dout; bd += reduceSegmentsND) {
    singleShapeBatch_ = min(uint64_t(batch)*dout-bd, reduceSegmentsND);
    for (uint64_t h = 0; h < ho; h += reduceSegmentsH) {
        singleShapeK = min(uint64_t(ho)-h, reduceSegmentsH)*wo;
        CalcOffset(bd/dout, bd%dout, groupIdx, dkIdx, h);
        ReCalDkCinSingleCoreShape(bd%dout, groupIdx, dkIdx);
        if (singleShapeMInCurrentHo_ == 0 || singleShapeNInCurrentHo_ == 0) continue;
        // SetOutBackprop / SetFmap / SetSingleShape / SetStartIdx
        dw_.IterateAll(yGm_[offsetC_], 1);
    }
}
```

IterateAll 覆盖该段全部有效 batch/dout、切宽和 K 子块；第二参数 1 表示原子累加。段内累计所有有效贡献，段间只通过 GM 的原子累加合并。

### 7.4 有效范围、物理跨度与槽容量

| 量 | 含义 | 用途 |
|---|---|---|
| baseUseM_/N_/K_ | 本轮实际计算长度 | MMAD/窗口参数 |
| alignUp(baseUse*,16) | 当前分形存储跨度 | 地址和占用 |
| al1Bound/bl1Bound | 单份 L1 父段最大元素数 | InitBuffer 字节数 |
| alignedL1UseKa | 当前 L1A 的实际父跨度 | Load2D 源步幅 |
| bL1cin1CopyLen | 当前 L1B 实际通道数 | DataCopy/窗口 channelSize |
| cinRemainLen_ | 当前 ci 尾部剩余通道 | Fixpipe 有效 nSize |

```text
tail ci block: [ valid ci lanes ][ alignment padding ]
each original kernel position writes valid ci lanes only
stored width and effective output width are separate
```

MMAD 使用源码更新后的有效 m/n/k，分形容量另外对齐。源地址对齐不等于扩大有效输出，补齐槽也不能作为额外通道写回。

这里区分数学有效范围与适配层的临时计算范围；例如有效 M=1 的计算可暂扩到 16，CopyOut 仍恢复为 1。各阶段实际尺寸及恢复顺序由 [Kernel 模板（Ascend C）](../../adapters/ascendc/filter/kernel-template.md#6-compute-计算逻辑) / [DSL](../../adapters/dsl/filter/kernel-template.md) 定义。

### 7.5 循环切换与状态生命周期

| 切换点 | 必须更新 | 可以保留 |
|---|---|---|
| 任务切换 | g/kd/co/ci、offsetC_、输出尾长 | 资源配置 |
| 归约段切换 | batchDout/ho 起点、offsetA_/B_、L0C 初值 | 当前输出任务范围 |
| ND 段内批深度切换 | 真实 batch/od/id、源父段与深度有效性 | 当前段 L0C |
| 切宽片切换 | 实际宽、左右 pad、父行带、K 长度 | 当前段 L0C |
| K 块切换 | baseUseK_、父段内起点、输入槽 | 非末次消费者仍需的 L1 父段 |

```text
+-------------------------------------------------------+
| 任务开始: 输出归属 / 尾块                             |
+---------------------------+---------------------------+
                            |
                            v
+---------------------------+---------------------------+
| 段开始: 源身份 / 首次贡献                             |
+---------------------------+---------------------------+
                            |
                            v
+---------------------------+---------------------------+
| batch-depth / 宽片 / K 循环: 保持段 L0C               |
+---------------------------+---------------------------+
                            |
                            v
+---------------------------+---------------------------+
| 段结束: F -> 原子累加 -> 释放 L0C                     |
+---------------------------+---------------------------+
                            |
                            v
+---------------------------+---------------------------+
| 下一段: 新的 L0C 初始化                               |
+-------------------------------------------------------+
```

跨段不得误复用已经变化的 GM 父段；段内也不能在换 batch、换宽片时重置 isFirstMmad。

## 8 DAG、Buffer 与流水

### 8.1 四条主流水的职责

| 流水 | 生产与消费 |
|---|---|
| MTE2 | 生产 L1A/L1B 父段 |
| MTE1 | 消费父段，生产 L0A/L0B |
| M | 消费 L0A/L0B，累加当前 L0C |
| F | 消费完整归约的 L0C，写入 dedw GM |

AIV 的 Fill 清零作为执行前置阶段，不计为主计算的一条常驻流水。

### 8.2 两阶段执行：预清零与主计算

```text
             +-----------------------------------+
             | AIV: Fill entire actual dedw = 0  |
             +-----------------+-----------------+
                               |
                               v
     +-------------------------+-------------------------+
     | all participating AIVs join -> publish MODE2 flag |
     +-------------------------+-------------------------+
                               |
                               v
               +---------------+---------------+
               | AIC: wait initialization flag |
               +---------------+---------------+
                               |
                               v
+------------------------------+------------------------------+
| AIC: execute owned output tasks; no per-block AIV reduction |
+-------------------------------------------------------------+
```

清零覆盖实际 dedw 的全部 FP32 元素。没有清零分片的 AIV 也参加汇合；没有主计算任务的 AIC 先等待初始化通知。streamkType=0、原始分组模式下，AIV 清零、汇合并通知后不做主轮数值工作；统一 Process 中仅保持任务推进。

AIV 是否需要参与主循环，取决于主循环是否存在跨核同步。本配置的 AIC 在初始化后不再等待 AIV；后续增加跨核归约路径时，必须同时实现 AIV 的配对路径，不能沿用本配置的提前返回规则。

### 8.3 四级流水图

四级流水沿用"搬入父段 → 准备操作数 → 乘加 → 段末原子写回"的顺序，四条流水（MTE2/MTE1/M/F）用双槽缓冲错开一级，使装载与计算重叠：

```text
+----------------+--------+-------------+----------------+---------------+-----------------+
| 流水 / 时间 -> | 阶段 0 | 阶段 1      | 阶段 2         | 阶段 3        | 阶段 4 / 后续   |
+----------------+--------+-------------+----------------+---------------+-----------------+
| MTE2           | 装A0B0 | 装A1B1      | 装A2B2         | ...           |                 |
| MTE1           |        | A0B0:L1->L0 | A1B1:L1->L0    | A2B2          | ...             |
| M              |        |             | MMAD(段0,init) | MMAD(段0,add) | ...             |
| F              |        |             |                |               | 原子累加段0->GM |
+----------------+--------+-------------+----------------+---------------+-----------------+
每级错开一步，四条流水并行流动。
```

取两个归约段示意流水，段内各有两次实际 MMAD。两路父段准备好后，MTE1 提取当前 K 子块；M 完成该段全部贡献后，F 才把段部分和原子累加到 dedw。

```text
+-------+----------+----------+--------+-------+-----------+-------+-----------+
| stage | t0       | t1       | t2     | t3    | t4        | t5    | t6        |
+-------+----------+----------+--------+-------+-----------+-------+-----------+
| MTE2  | seg0 A/B | seg1 A/B |        |       |           |       |           |
| MTE1  |          | j00      | j01    | j10   | j11       |       |           |
| M     |          |          | init 0 | add 0 | init 1    | add 1 |           |
| F     |          |          |        |       | atomic F0 |       | atomic F1 |
+-------+----------+----------+--------+-------+-----------+-------+-----------+
```

| 贡献 | j00 | j01 | j10 | j11 |
|---|---|---|---|---|
| 归约段 | 0 | 0 | 1 | 1 |
| L0 输入槽 | 0 | 1 | 0 | 1 |
| L0C 槽 | 0 | 0 | 1 | 1 |
| L0C 操作 | 初始化 | 累加 | 初始化 | 累加 |

每格表示逻辑可重叠阶段，不代表固定周期。F0 必须等 j01 完成，F1 必须等 j11 完成；F0 可与另一 L0C 槽上的 j10 重叠。若只有一份 L0C，则后段先等前段 F 归还。

### 8.4 DAG 与 Buffer 存活区间

```text
MTE2(dedy) -> MTE1(Load2D) -----+
                                |
MTE2(x) ----> MTE1(Load3D) -----+--> MMAD -> complete segment -> atomic F -> dedw
```

| Buffer | 生产者→消费者 | 保留到何时 | 轮换依据 |
|---|---|---|---|
| L1A | MTE2→MTE1 | 最后一次 Load2D 读取 | A 路父段 |
| L1B | MTE2→MTE1 | 最后一次 Load3D 读取 | B 路父段 |
| L0A/L0B | MTE1→M | 本次 MMAD 读取结束 | 实际输入槽，成对交接 |
| L0C | M→F | 当前段全部贡献与 Fixpipe 读取结束 | 当前输出子块/归约段 |

```text
+--------+------------------------------+--------------------------------------------+
| 存储   | 槽 0                         | 槽 1（启用时）                             |
+--------+------------------------------+--------------------------------------------+
| L1A    | [0,aAligned)                 | [halfL1,halfL1+aAligned)                   |
| L1B    | [aAligned,aAligned+bAligned) | [halfL1+aAligned,halfL1+aAligned+bAligned) |
| L0A    | 首个输入区                   | 第二输入区                                 |
| L0B    | 首个输入区                   | 第二区 / 拆核时单槽                        |
| L0C    | 当前归约段                   | 下一段输出槽                               |
+--------+------------------------------+--------------------------------------------+
```

Host 预算包含 buffer 份数，Kernel 还检查固定槽基址与最大末地址。若 pong 放在 L1 半区，只有总字节数合法仍不够，ping/pong 区间必须实际互不重叠。

### 8.5 就绪、归还与首末轮

```text
L1:  MTE2 ──就绪──► MTE1 ──归还──► 下一次 MTE2 覆盖
L0:  MTE1 ──就绪──► M    ──归还──► 下一次 MTE1 覆盖
L0C: M    ──就绪──► F    ──归还──► 下一输出块的 M 初始化

首槽: 队列 / 已核实的 TPipe 初态
末次使用: 最后消费者依赖 -> 队列 End / TPipe 收尾
```

首次空槽由队列或所选 SDK 的事件初态提供；每次 Load/Compute/CopyOut 成对发布和消费令牌。最后一次提交消费者之后发布归还依赖，下一次覆盖前等待该依赖；提交不等于已完成。

全部任务结束后归还仍持有的父段和 L0C，End 释放队列事件，TPipe 完成固定事件收尾。空段不等待未生产的就绪事件。混合核声明、AIV→AIC 初始化通知、零贡献 L0C 交接和 SetScheduleMode 按 [filter 同步实现（Ascend C）](../../adapters/ascendc/filter/sync-implementation.md) / [DSL](../../adapters/dsl/filter/sync-implementation.md) 配对。

## 9 检查点

- [ ] 外部三种输出布局、两种输入布局与实际元素步幅一致。
- [ ] L1A 的转置分形与 L1B 的窗口分形在 K 上配对。
- [ ] x 的 pad 区只影响窗口值，GM 搬运始终在有效区域。
- [ ] N 使用 `(ci1,kh,kw,ci0)`；非整窗块保留核位置和通道状态。
- [ ] kd 为任务轴，batch×dout 与 ho 分段覆盖完整归约域。
- [ ] 每个实际 dedw 元素唯一归属，补齐通道和相邻 kd 区域不被多写。
- [ ] 首次实际乘加初始化；段内各循环保留部分和，段间 GM 原子合并。
- [ ] 主计算发生在预清零完成之后，重复执行无旧输出残留。
- [ ] 四级流水的就绪、归还和最终收尾都有对应依赖。

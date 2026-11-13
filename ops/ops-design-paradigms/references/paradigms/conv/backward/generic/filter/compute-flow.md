# 卷积反向权重梯度分支：计算流程

本文回答数据怎样存放、窗口怎样展开、任务怎样分配、归约怎样执行和 Buffer 怎样交接。计算模型见 [scene.md](scene.md)，切分规则见 [tiling/tiling.md](tiling/tiling.md)，API 字段见 [指令映射](../../adapters/ascendc/filter/instruction-mapping.md)。

沿用 x、dedy、dedw、dedy_flat、x_win。A 路为 dedy，B 路为 x；L1A/L1B 保存父段，L0A/L0B 为本次计算操作数，L0C 保存 FP32 部分和。轴序均从慢到快，示意小分形只说明顺序。

## 1 计算模型速览

### 1.1 从正向连接得到 dedw

固定 g、co、ci 和原核坐标，每个梯度元素归约全部 batch 和 dedy 空间位置：

$$
\mathrm{dedw}_g[co,ci,\mathbf k]
=\sum_{n,\mathbf o}\mathrm{dedy}_g[n,co,\mathbf o]\,
x_g[n,ci,\mathbf o\odot\mathbf s+\mathbf k\odot\boldsymbol\delta-\mathbf p^-].
$$

越界 x 采样为零。固定 kd 后，dedy_flat 的 K 列和 x_win 的 K 行按同一个 `(oh,ow)` 配对；batch×dout 在矩阵乘外累加。

```text
dedy -- flatten output grid --> dedy_flat [M,K] --+
                                                  |
x -- sample forward windows --> x_win [K,N] ------+
                                                  |
                                                  |
                                                  v
                                  +---------------+---------------+
                                  | multiply-add at fixed g / kd  |
                                  +---------------+---------------+
                                                  |
                                                  |
                                                  v
                                       +----------+----------+
                                       | sum(batch,od,oh,ow) |
                                       +----------+----------+
                                                  |
                                                  |
                                                  v
                                    +-------------+-------------+
                                    | dedw [Cout,CinG,Dk,Hk,Wk] |
                                    +---------------------------+
```

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
+---------------------------+
| dedy / GM: NCDHW or NDHWC |
+-------------+-------------+
              |
              | MTE2: DN2NZ / ND2NZ
              v
+-------------+-------------+
| L1A: (Cout1,K1,K0,Cout0)  |
+-------------+-------------+
              |
              | MTE1: Load2D, transpose fractals
              v
+-------------+-------------+
| L0A: (K1,Cout1,Cout0,K0)  |
| NZ(M,K)                   |
+---------------------------+
```

FP16/BF16 的常规 L1A 父布局为 `(Cout1,K1,K0,Cout0)`；Load2D 交换分形内两轴并重排分形，得到 `(K1,Cout1,Cout0,K0)`。K 指输出位置，不是核位置。

### 2.3 x 侧：GM → L1B → L0B

```text
     +-------------------------+
     | x / GM: NCDHW or NDHWC  |
     +------------+------------+
                  |
                  | MTE2: crop valid source + DN2NZ / ND2NZ
                  v
+-----------------+-----------------+
| L1B: (Cin1,hiCopyLen,realWi,Cin0) |
+-----------------+-----------------+
                  |
                  | MTE1: Load3D, forward windows
                  v
  +---------------+---------------+
  | L0B: (K1,Cin1,kh,kw,Cin0,K0)  |
  | ZN(K,N)                       |
  +-------------------------------+
```

固定实际 x 深度，L1B 的典型视图为 `(Cin1,hiCopyLen,realWi,Cin0)`。Load3D 按正向 stride/dilation 和边界提取窗口，组织为本轮 x_win。普通路径和拆核路径的差别在窗口由哪一阶段展开。

### 2.4 输出侧：MMAD → L0C → dedw GM

```text
      +-------------------------------------+
      | L0C segment partial sum / NZ / FP32 |
      +------------------+------------------+
                         |
                         | wait for all segment contributions
                         v
+------------------------+------------------------+
| SetAtomicAdd<float> -> Fixpipe -> SetAtomicNone |
+------------------------+------------------------+
                         |
                         | valid channels and kernel positions only
                         v
      +------------------+------------------+
      | dedw / GM: NCDHW or NDHWC or DHWCN  |
      +-------------------------------------+
```

每个归约段第一次实际 MMAD 初始化 L0C，其余贡献继续累加。段末写回时 GM 中已有先前段结果，因此使用原子累加；新段不沿用旧 L0C。

### 2.5 两路数据流总图

```text
AIV: clear dedw -> notify -> AIC wait -> main computation

  +-----------------------------------+           +-----------------------------------+
  | dedy / GM                         |           | x / GM                            |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    | DataCopy                                      | DataCopy
                    |                                               |
                    |                                               |
                    v                                               v
  +-----------------+-----------------+           +-----------------+-----------------+
  | L1A / parent segment              |           | L1B / x rows                      |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    | Load2D + transpose                            | Load3D windows
                    |                                               |
                    |                                               |
                    v                                               v
  +-----------------+-----------------+           +-----------------+-----------------+
  | L0A / NZ(M,K)                     |           | L0B / ZN(K,N)                     |
  +-----------------+-----------------+           +-----------------+-----------------+
                    |                                               |
                    |                                               |
                    +-----------------------+-----------------------+
                                            |
                                            | MMAD / FP32
                                            v
                          +-----------------+-----------------+
                          | L0C / segment partial sum         |
                          +-----------------+-----------------+
                                            |
                                            |
                                            | Fixpipe / atomic add
                                            |
                                            v
                          +-----------------+-----------------+
                          | dedw / GM / FP32                  |
                          +-----------------------------------+
```

预清零在两路主计算之前完成。streamkType=0、groupEnlarge=false 时，AIV 完成初始化协议后不承担每个计算块的乘加或归约。

## 3 数据格式与存储轴序

### 3.1 NCDHW、NDHWC 与 DHWCN

对一般特征张量 T，固定 batch，示例取 `C=2、D=2、H=1、W=2`。标号为 `(c,d,w)`：

```text
Logical coordinates (c,d,w), H=1, fixed batch:
+-------+-------------------+
| c / d | w=0       w=1     |
+-------+-------------------+
| 0 / 0 | (0,0,0) (0,0,1)   |
| 0 / 1 | (0,1,0) (0,1,1)   |
| 1 / 0 | (1,0,0) (1,0,1)   |
| 1 / 1 | (1,1,0) (1,1,1)   |
+-------+-------------------+

NCDHW: 000 001 010 011 | 100 101 110 111
NDHWC: 000 100 001 101 | 010 110 011 111
```

NCDHW 先走完一个通道的空间体，NDHWC 先走完一个空间点的通道。处理组内片段时，原张量空间步幅仍使用总 Cin 或 Cout。

dedw 的格式字符串中 N 对应正向 Cout、C 对应 CinG。用 `100*co+10*ci+kw` 给同一组元素编号，取 `Cout=CinG=2、dk=hk=1、wk=2`：

```text
+--------+-------------------+---------------------------+
| format | slow -> fast axes | memory order              |
+--------+-------------------+---------------------------+
| NCDHW  | (co,ci,kd,kh,kw)  | 0 1 10 11 100 101 110 111 |
| NDHWC  | (co,kd,kh,kw,ci)  | 0 10 1 11 100 110 101 111 |
| DHWCN  | (kd,kh,kw,ci,co)  | 0 100 10 110 1 101 11 111 |
+--------+-------------------+---------------------------+
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

固定逻辑矩阵 T[R,S]，将 R/S 分别拆为块号 R1/S1 与块内坐标 R0/S0：

| 格式 | 慢→快轴序 | 遍历顺序 |
|---|---|---|
| ND | `(R,S)` | 行连续 |
| DN | `(S,R)` | 列连续 |
| NZ | `(S1,R1,R0,S0)` | 块间列优先，块内行优先 |
| ZN | `(R1,S1,S0,R0)` | 块间行优先，块内列优先 |

同一个 4×4 矩阵，示意块为 2×2：

```text
+-------+----+----+----+----+
| r / c | 0  | 1  | 2  | 3  |
+-------+----+----+----+----+
| 0     | 0  | 1  | 2  | 3  |
| 1     | 4  | 5  | 6  | 7  |
| 2     | 8  | 9  | 10 | 11 |
| 3     | 12 | 13 | 14 | 15 |
+-------+----+----+----+----+

ND: 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15
DN: 0 4 8 12 | 1 5 9 13 | 2 6 10 14 | 3 7 11 15
NZ: 0 1 4 5 | 8 9 12 13 | 2 3 6 7 | 10 11 14 15
ZN: 0 4 1 5 | 2 6 3 7 | 8 12 9 13 | 10 14 11 15

Each group of four numbers is one 2x2 fractal in NZ / ZN.
```

布局转换只改变地址，不改变 T 的数学坐标。例如 `T[2,1]=9` 在 ND/DN/NZ/ZN 中的元素偏移分别为 9/6/5/10。

```text
 +-------------------------+
 | (R,S) / row-contiguous  |
 +------------+------------+
              |
              |
              v
  +-----------+-----------+
  | pad -> (R1*R0,S1*S0)  |
  +-----------+-----------+
              |
              |
              v
+-------------+-------------+
| reshape -> (R1,R0,S1,S0)  |
+-------------+-------------+
              |
              |
              v
    +---------+---------+
    | NZ: (S1,R1,R0,S0) |
    | ZN: (R1,S1,S0,R0) |
    +-------------------+
```

上述 pad/reshape/轴序重排是推导步骤，实际搬运可一次实现多步。ND2NZ/DN2NZ 的命名按其源矩阵朝向理解；先写清行列是谁，再配置 API。

### 3.3 五维张量与片上视图

固定 g、kd 和一个 batch/dout 后，两条五维张量通路变为：

```text
    +---------------------------------------------+
    | dedy [batch,Cout,Dout,Ho,Wo] / logical axes |
    +----------------------+----------------------+
                           |
                           |
                           v
  +------------------------+------------------------+
  | select batch / group / output depth / co block  |
  +------------------------+------------------------+
                           |
                           |
                           v
+--------------------------+--------------------------+
| L1A [Cout1,K1,K0,Cout0] -> L0A [K1,Cout1,Cout0,K0]  |
+-----------------------------------------------------+
```

```text
  +---------------------------------------+
  | x [batch,Cin,Di,Hi,Wi] / logical axes |
  +-------------------+-------------------+
                      |
                      |
                      v
+---------------------+---------------------+
| id = od*strideD + kd*dilationD - padFront |
+---------------------+---------------------+
                      |
                      |
                      v
    +-----------------+-----------------+
    | L1B [Cin1,hiCopyLen,realWi,Cin0]  |
    +-----------------+-----------------+
                      |
                      |
                      v
  +-------------------+-------------------+
  | x_win -> L0B [K1,Cin1,kh,kw,Cin0,K0]  |
  +---------------------------------------+
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
+----------+------------+---------------------+---------------+
| buffer   | row        | col                 | physical axes |
+----------+------------+---------------------+---------------+
| L0A / NZ | M = co     | K = oh*wo+ow        | (K1,M1,M0,K0) |
| L0B / ZN | the same K | N = (ci1,kh,kw,ci0) | (K1,N1,N0,K0) |
| L0C / NZ | M = co     | the same N          | (N1,M1,M0,N0) |
+----------+------------+---------------------+---------------+
```

这里的 `(ci1,kh,kw,ci0)` 为 N 的逻辑顺序。非整窗 baseN 可以截在相邻核位置或通道块之间，必须保留 head_/tail_/cinRemainLen_，由写回逻辑拆成合法片段。

## 4 img2col 与窗口归约

### 4.1 M/N/K 与窗口归约顺序

固定 kd 时，K 位置为 `k=oh*wo+ow`。N 位置为：

`n=((ci1*hk+khIdx)*wk+kwIdx)*C0+ci0`。

解码 N 时，先取 `ci0=n%C0`，再将 `n//C0` 解为 kwIdx、khIdx、ci1。A 的 K 列与 B 的 K 行必须对应同一批次、同一输出深度和同一输出平面位置。

```text
Fixed task (g,kd,co-range,ci-window-range):

  batch / od segment
    -> ho segment
      -> actual batch / od
        -> output width slice
          -> K block: oh / ow positions

N position: ci1 -> kh -> kw -> ci0 (fastest)
A K column and B K row refer to the same actual batch / od / oh / ow.
```

batch×dout 为外层归约域；ho 分段和 wo 切宽只是将同一输出网格分成不重叠区域。补齐通道只占内部存储，不增加有效乘积。

### 4.2 三维例子：x 窗口与输出平面

取 `x=(1,1,3,3,3)`、`dedw=(1,1,2,2,2)`、stride/dilation 均为 1、pad=0，得到 `dedy=(1,1,2,2,2)`。x 按深度顺序取 1～27，dedy 取 1～8：

```text
x, depth id=0       x, depth id=1       x, depth id=2
+----+----+----+    +----+----+----+    +----+----+----+
|  1 |  2 |  3 |    | 10 | 11 | 12 |    | 19 | 20 | 21 |
|  4 |  5 |  6 |    | 13 | 14 | 15 |    | 22 | 23 | 24 |
|  7 |  8 |  9 |    | 16 | 17 | 18 |    | 25 | 26 | 27 |
+----+----+----+    +----+----+----+    +----+----+----+

dedy, od=0    dedy, od=1
+---+---+     +---+---+
| 1 | 2 |     | 5 | 6 |
| 3 | 4 |     | 7 | 8 |
+---+---+     +---+---+
```

先固定 `kd=0、od=0`。四个输出位置需要四个 2×2 窗口：

```text
+---+---------+------------------+--------------+
| K | (oh,ow) | x window at id=0 | row in x_win |
+---+---------+------------------+--------------+
| 0 | (0,0)   | [1 2; 4 5]       | 1 2 4 5      |
| 1 | (0,1)   | [2 3; 5 6]       | 2 3 5 6      |
| 2 | (1,0)   | [4 5; 7 8]       | 4 5 7 8      |
| 3 | (1,1)   | [5 6; 8 9]       | 5 6 8 9      |
+---+---------+------------------+--------------+
```

将它们排入 x_win 的四行，同时将 dedy 平面排成 dedy_flat 的一行：

```text
dedy_flat: [1 2 3 4]

+-------+-------+-------+-------+-------+
| K / N | (0,0) | (0,1) | (1,0) | (1,1) |
+-------+-------+-------+-------+-------+
| 0     | 1     | 2     | 4     | 5     |
| 1     | 2     | 3     | 5     | 6     |
| 2     | 4     | 5     | 7     | 8     |
| 3     | 5     | 6     | 8     | 9     |
+-------+-------+-------+-------+-------+

dedy_flat [1x4] * x_win [4x4] = [37 47 67 77]
Columns above show (kh,kw), fixed ci=0.
```

这一批深度产生 `[37,47,67,77]`。继续归约 `od=1`，对 kd=0 得到最终 `[356,392,464,500]`；kd=1 的最终结果为 `[680,716,788,824]`。这些数值只用于验证连接、核顺序和矩阵化，实际计算的通道维按 C0 补齐。

### 4.3 img2col：从 x 的窗口到 L0B

Load3D 的 m 方向表示当前输出位置 K，k 方向表示窗口/通道 N。它们与 GEMM 的 M/K 不是同一命名：

| 窗口参数 | 本分支含义 |
|---|---|
| mStartPt / mExtension | 输出网格起点 / 当前有效 K |
| kStartPt / kExtension | 当前核窗口通道起点 / 当前 N 长度 |
| channelSize | 当前提取覆盖的 ci 通道跨度 |
| padList | 当前 x 父段的左、右、上、下边界 |
| filter/stride/dilation | 原正向 H/W 采样几何；拆核时改为单位窗口并由外层定位原核坐标 |

```text
          +---------------------------------+
          | L1B: x row band [Cin1,H,W,Cin0] |
          +----------------+----------------+
                           |
                           | select one output position
                           v
+--------------------------+--------------------------+
| window origin: oh*strideH-padUp, ow*strideW-padLeft |
+--------------------------+--------------------------+
                           |
                           | extract forward kernel
                           v
+--------------------------+--------------------------+
| sample kh/kw with dilation; pack N=(ci1,kh,kw,ci0)  |
+--------------------------+--------------------------+
                           |
                           | repeat for each valid K
                           v
              +------------+------------+
              | L0B [K1,N1,N0,K0] / ZN  |
              +-------------------------+
```

例如 `hk=wk=2、C0=16、baseUseN_=64、kStartPt=0`：覆盖一个完整的核高宽通道块，channelSize 为 16 个通道；kExtension 为 64 个窗口元素，不能把 64 误当成通道数。FP16/BF16 使用一条配套 Repeat/LoadData 调用；按分形循环多条命令是另一种 dtype 通路，不能直接移用。

### 4.4 核深度与批深度的处理

```text
+---------+---------+-----------------------------------+
| task kd | dedy od | x id (strideD=dilationD=1, pad=0) |
+---------+---------+-----------------------------------+
| 0       | 0       | 0                                 |
| 0       | 1       | 1                                 |
| 1       | 0       | 1                                 |
| 1       | 1       | 2                                 |
+---------+---------+-----------------------------------+
```

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

```text
Current output width = 100, kal1 = 128

row 0: [ K   0 .............................................. K  99 ]
row 1: [ K 100 .... K 127 | remaining positions in this output row   ]
row 2: [ Host budget reserves crossing-row coverage when needed     ]

At parent start 0: copy 2 real rows, keep actual parent stride.
At a row-aligned final 64 positions: copy 1 real row.
Host upper bound: floor(128/100)+2 = 3 rows.
```

例如 `kal1=128、singleShapeWo=100`：预算覆盖 3 行；从平面起点搬入首父段时实际覆盖 2 行。如果末父段只剩 64 个位置且从新行开始，实际只搬 1 行。128 不能被 100 整除。

#### 5.2.2 布局转换与计算分形

NCDHW 使用 DN2NZ，源通道步幅是 `dout*ho*wo`；NDHWC 使用 ND2NZ，源行步幅是总 cout。两者形成相同 L1A 父布局，再由 Load2D 转置进入 L0A。

```text
     +-------------------------------+
     | NCDHW: source [co,K] / DN2NZ  |
     | NDHWC: source [K,co] / ND2NZ  |
     +---------------+---------------+
                     |
                     |
                     v
        +------------+------------+
        | L1A [Cout1,K1,K0,Cout0] |
        +------------+------------+
                     |
                     |
                     v
+--------------------+--------------------+
| Load2D: swap K0 / Cout0, reorder blocks |
+--------------------+--------------------+
                     |
                     |
                     v
        +------------+------------+
        | L0A [K1,Cout1,Cout0,K0] |
        +-------------------------+
```

切宽后 alignedL1UseKa 取实际搬入行数与宽度形成的父跨度，Load2D 再加行内 K 起点。不能用最大槽长或原完整 wo 代替该跨度。

### 5.3 x：GM→L1B→L0B

#### 5.3.1 从输出行带反推 x 区域

普通路径的覆盖范围为：

`hiCal=(hoCal-1)*strideH+(hk-1)*dilationH+1`，
`splitWi=(splitWo-1)*strideW+(wk-1)*dilationW+1`。

拆核路径每次处理一个原核坐标，Host 预算为 `hiCal=hoCal、splitWi=splitWo`，原 stride/dilation 和原核起点由分段搬运表达。

```text
hoCal=2, strideH=2, hk=3, dilationH=1

output row 0 -> input rows [0 1 2]
output row 1 -> input rows     [2 3 4]
                              ^ shared row
union       -> input rows [0 1 2 3 4] = 5 rows

If origin is -1: logical rows [-1 0 1 2 3]
  pad top -> [-1]
  GM copy -> [ 0 1 2 3 ]
```

例如 `hoCal=2、strideH=2、hk=3、dilationH=1`，普通路径需覆盖 5 行。若起点为 −1，则真实复制从 x 第 0 行开始，顶部缺失一行由 pad 表达。搬运只覆盖 x 的有效交集，不能将负 srcStride 当作越界修正。

#### 5.3.2 原窗口与拆核窗口

```text
+--------------+-----------------------------------------+-------------------------------+
| path         | MTE2 / L1B                              | MTE1 / L0B                    |
+--------------+-----------------------------------------+-------------------------------+
| normal       | copy valid original x row band          | Load3D(hk,wk,stride,dilation) |
| split kernel | select original kh/kw; sample by stride | Load3D(1,1,1,1)               |
+--------------+-----------------------------------------+-------------------------------+

Both paths -> same x_win values at the same (K,N) coordinates.
```

普通路径在 Load3D 中完成整个核窗口；拆核路径在 GM→L1B 的选择中落实原 kh/kw，Load3D 只提取单位核。更换路径时，原核位置、源地址、窗口参数和目标 N 分形位置必须一起更新。

普通父段保留原 x 的连续行带，pad 使用原空间点单位；拆核父段只保存固定核位置的有效采样点，pad 改用采样网格的点数。两条路径最终对应相同的 `(K,N)`，具体裁剪、行内起点和数值示例见 [指令映射](../../adapters/ascendc/filter/instruction-mapping.md#3-x-到-l1b)。

### 5.4 MMAD 与 dedw 写回

#### 5.4.1 段内初值与段间合并

```text
dedw GM initially: 0

segment 0: init L0C -> add all valid contributions -> atomic add to dedw
segment 1: init L0C -> add all valid contributions -> atomic add to dedw
segment 2: init L0C -> add all valid contributions -> atomic add to dedw

dedw GM finally: segment0 + segment1 + segment2
```

MMAD 使用当前计算 baseUseM_/N_/K_；内部槽占用另行对齐。适配实现会在单个有效 co 时临时把计算 M 扩到 16，写回前恢复有效 M；数学输出范围始终不变。isFirstMmad 跨本段的 batch/dout、切宽和 K 循环保留；段末将完整部分和原子累加到 dedw，下一段重新置 true。

#### 5.4.2 从 L0C 恢复三种输出格式

```text
+----------------+----------------------------------+
| stage / format | slow -> fast axes, fixed task kd |
+----------------+----------------------------------+
| L0C            | (Cin1,kh,kw,Cout1,Cout0,Cin0)    |
| dedw / NCDHW   | (co,ci,kd,kh,kw)                 |
| dedw / NDHWC   | (co,kd,kh,kw,ci)                 |
| dedw / DHWCN   | (kd,kh,kw,ci,co)                 |
+----------------+----------------------------------+
```

L0C 的 N 包含 `(ci1,kh,kw,ci0)`，不能将它作为连续 CinG 直接平铺到外部权重格式。普通核 NCDHW 写回需交换核位置与 ci0；NDHWC 保持核位置在 ci 之外；DHWCN 还使全局 co 成为最快轴。1×1×1 使用简化矩阵写回，普通核和非整窗 N 另有分发。

例如 CinG=17 时，第二个 ci 分形只有一个有效通道。每个核位置只写这一个通道，余下 15 个内部补齐位置都不能写回；dk>1 时扩大 nSize 会落到相邻深度或通道区域。

### 5.5 原张量地址与三维步幅

用 `c_o=g*CoutG+co`、`c_i=g*CinG+ci` 表示全局通道；以下为元素偏移：

| 张量 | NCDHW | NDHWC |
|---|---|---|
| dedy | `((((n*cout+c_o)*dout+od)*ho+oh)*wo+ow)` | `((((n*dout+od)*ho+oh)*wo+ow)*cout+c_o)` |
| x | `((((n*cin+c_i)*di+id)*hi+ih)*wi+iw)` | `((((n*di+id)*hi+ih)*wi+iw)*cin+c_i)` |
| dedw | `((((c_o*cinG+ci)*dk+kdIdx)*hk+khIdx)*wk+kwIdx)` | `((((c_o*dk+kdIdx)*hk+khIdx)*wk+kwIdx)*cinG+ci)` |

DHWCN 的 dedw 偏移为 `((((kdIdx*hk+khIdx)*wk+kwIdx)*cinG+ci)*cout+c_o)`。

```text
offsetA = batch base + group/co base + od plane + ho row
offsetB = batch base + group/ci base + valid id plane + valid ih row
offsetC = group/co base + ci base + fixed kd slice

segment changes batch / od / ho -> recalculate offsetA and offsetB
same task                     -> keep offsetC
```

offsetC_ 只随任务通道、组和核深度变化；batch/dout/ho 段切换重算 offsetA_/offsetB_。带 pad 的采样坐标先按有符号宽整数判断，只有确认有效后才能提交 GM 访问。

## 6 多核任务与输出归属

### 6.1 按 M/N 基本块分配任务

`mCnt=ceilDiv(mValue,singleCoreM)`、`nCnt=ceilDiv(nValue,singleCoreN)`，`totalCnt=mCnt*nCnt*dk*G`。每个基本块绑定一个 g、kd 和 M/N 范围：

```text
+---------+-----------------+----------------------------+------------+
| task    | co range        | ci/window range            | fixed axes |
+---------+-----------------+----------------------------+------------+
| (m0,n0) | [0,singleCoreM) | first full channel windows | g,kd       |
| (m0,n1) | same co range   | next full channel windows  | g,kd       |
| (m1,n0) | next co range   | first full channel windows | g,kd       |
+---------+-----------------+----------------------------+------------+

idx = (((mIdx*nCnt+nIdx)*dk+dkIdx)*G+gIdx)
core assignment: idx = block_idx + round*usedCoreNum
```

```cpp
for (uint64_t idx = block_idx; idx < totalCnt; idx += usedCoreNum) {
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
`-- task: g / kd / co-range / ci-window-range
    `-- reduction segment: batch*dout range / ho range
        `-- M/N output subblock: baseM x baseN
            `-- actual batch / od and output width slice
                `-- K subblock: paired L0A / L0B
                    -> accumulate in the same segment L0C
        -> Fixpipe: add complete segment result to dedw
```

singleCoreN 描述任务通道窗口，baseN 描述 L0 子块；一项任务可以有多个 N 子块。每个归约段的 L0C 必须包含当前输出块在该段内的全部有效贡献，再进入 Fixpipe。

### 7.2 L1 父段与 K 子块

两路分别按 stepKa/stepKb 换驻，父段状态包括范围、实际行带、槽号和最后一次消费者。较大的 step 为较小 step 的整数倍，末段实际轮数另算。

```text
+----------------------+----+----+----+----+----+
| K block number       | 0  | 1  | 2  | 3  | 4  |
+----------------------+----+----+----+----+----+
| A parent, stepKa=2   | A0 | A0 | A1 | A1 | A2 |
| B parent, stepKb=4   | B0 | B0 | B0 | B0 | B1 |
| local K / baseK in A | 0  | 1  | 0  | 1  | 0  |
| local K / baseK in B | 0  | 1  | 2  | 3  | 0  |
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

IterateAll 覆盖该段全部有效 batch/dout、切宽和 K 子块；第二参数 1 表示原子累加。示例 `batch*dout=128、ho=wo=32` 得到 H 段长 32、ND 段长 64，共两段，每段归约 65536 个输出位置。

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
Example ci tail: actual CinG=17

ci block 0: [0 1 ... 15]       -> 16 valid channels
ci block 1: [16 | pad ... pad] ->  1 valid channel, 16 stored lanes

MMAD / buffer addressing: use actual extent + fractal storage contract
Fixpipe: write the one valid ci for each original kernel position
```

MMAD 使用源码更新后的有效 m/n/k，分形容量另外对齐。源地址对齐不等于扩大有效输出，补齐槽也不能作为额外通道写回。

这里区分数学有效范围与适配层的临时计算范围；例如有效 M=1 的计算可暂扩到 16，CopyOut 仍恢复为 1。各阶段实际尺寸及恢复顺序由 [Kernel 模板](../../adapters/ascendc/filter/kernel-template.md#6-compute-计算逻辑) 定义。

### 7.5 循环切换与状态生命周期

| 切换点 | 必须更新 | 可以保留 |
|---|---|---|
| 任务切换 | g/kd/co/ci、offsetC_、输出尾长 | 资源配置 |
| 归约段切换 | batchDout/ho 起点、offsetA_/B_、L0C 初值 | 当前输出任务范围 |
| ND 段内批深度切换 | 真实 batch/od/id、源父段与深度有效性 | 当前段 L0C |
| 切宽片切换 | 实际宽、左右 pad、父行带、K 长度 | 当前段 L0C |
| K 块切换 | baseUseK_、父段内起点、输入槽 | 非末次消费者仍需的 L1 父段 |

```text
        +---------------------------------------+
        | task starts: output ownership / tails |
        +-------------------+-------------------+
                            |
                            |
                            v
+---------------------------+---------------------------+
| segment starts: source identity / first contribution  |
+---------------------------+---------------------------+
                            |
                            |
                            v
  +-------------------------+-------------------------+
  | batch-depth / width / K loops: retain segment L0C |
  +-------------------------+-------------------------+
                            |
                            |
                            v
  +-------------------------+-------------------------+
  | segment finishes: F -> atomic add -> release L0C  |
  +-------------------------+-------------------------+
                            |
                            |
                            v
        +-------------------+-------------------+
        | next segment: new L0C initialization  |
        +---------------------------------------+
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
                               |
                               v
     +-------------------------+-------------------------+
     | all participating AIVs join -> publish MODE2 flag |
     +-------------------------+-------------------------+
                               |
                               |
                               v
               +---------------+---------------+
               | AIC: wait initialization flag |
               +---------------+---------------+
                               |
                               |
                               v
+------------------------------+------------------------------+
| AIC: execute owned output tasks; no per-block AIV reduction |
+-------------------------------------------------------------+
```

清零覆盖实际 dedw 的全部 FP32 元素。没有清零分片的 AIV 也参加汇合；没有主计算任务的 AIC 先等待初始化通知。streamkType=0、原始分组模式下，AIV 清零、汇合并通知后返回，不进入主循环。

AIV 是否需要参与主循环，取决于主循环是否存在跨核同步。本配置的 AIC 在初始化后不再等待 AIV；后续增加跨核归约路径时，必须同时实现 AIV 的配对路径，不能沿用本配置的提前返回规则。

### 8.3 四级流水图

取两个归约段，每段有两轮 MMAD，输入和 L0C 均可双缓冲。seg0/seg1 的两路父段分别先完成，再消费相应的 j00/j01 和 j10/j11：

```text
+-------+----------+----------+-----+-----+-----+-----+----+
| stage | t0       | t1       | t2  | t3  | t4  | t5  | t6 |
+-------+----------+----------+-----+-----+-----+-----+----+
| MTE2  | seg0 A/B | seg1 A/B |     |     |     |     |    |
| MTE1  |          | j00      | j01 | j10 | j11 |     |    |
| M     |          |          | j00 | j01 | j10 | j11 |    |
| F     |          |          |     |     | F0  |     | F1 |
+-------+----------+----------+-----+-----+-----+-----+----+

+---------------+------+-----+------+-----+
| contribution  | j00  | j01 | j10  | j11 |
+---------------+------+-----+------+-----+
| segment       | 0    | 0   | 1    | 1   |
| L0 input slot | 0    | 1   | 0    | 1   |
| L0C slot      | 0    | 0   | 1    | 1   |
| L0C action    | init | add | init | add |
+---------------+------+-----+------+-----+
```

每格表示逻辑可重叠阶段，不代表固定周期。F0 必须等 j01 完成，F1 必须等 j11 完成；F0 可与另一 L0C 槽上的 j10 重叠。若只有一份 L0C，则后段先等前段 F 归还。

### 8.4 DAG 与 Buffer 存活区间

```text
MTE2 / L1A -> MTE1 / L0A --+
                           |
MTE2 / L1B -> MTE1 / L0B --+
                           |
                           |
                           v
                  +--------+--------+
                  | MMAD: paired K  |
                  +--------+--------+
                           |
                           |
                           v
           +---------------+---------------+
           | full segment reduction in L0C |
           +---------------+---------------+
                           |
                           |
                           v
              +------------+------------+
              | F -> atomic add -> dedw |
              +-------------------------+
```

| Buffer | 生产者→消费者 | 保留到何时 | 轮换依据 |
|---|---|---|---|
| L1A | MTE2→MTE1 | 最后一次 Load2D 读取 | A 路父段 |
| L1B | MTE2→MTE1 | 最后一次 Load3D 读取 | B 路父段 |
| L0A/L0B | MTE1→M | 本次 MMAD 读取结束 | 实际输入槽，成对交接 |
| L0C | M→F | 当前段全部贡献与 Fixpipe 读取结束 | 当前输出子块/归约段 |

```text
+--------+------------------------------+--------------------------------------------+
| memory | slot 0                       | slot 1, when enabled                       |
+--------+------------------------------+--------------------------------------------+
| L1A    | [0,aAligned)                 | [halfL1,halfL1+aAligned)                   |
| L1B    | [aAligned,aAligned+bAligned) | [halfL1+aAligned,halfL1+aAligned+bAligned) |
| L0A    | first input region           | second input region                        |
| L0B    | first input region           | second region / single slot if split       |
| L0C    | current segment              | next segment output slot                   |
+--------+------------------------------+--------------------------------------------+
```

Host 预算包含 buffer 份数，Kernel 还检查固定槽基址与最大末地址。若 pong 放在 L1 半区，只有总字节数合法仍不够，ping/pong 区间必须实际互不重叠。

### 8.5 就绪、归还与首末轮

```text
L1:  MTE2 -- ready --> MTE1 -- return --> next MTE2 overwrite
L0:  MTE1 -- ready --> M    -- return --> next MTE1 overwrite
L0C: M    -- ready --> F    -- return --> next M output initialization

First slot: queue / verified TPipe initial state
Final use : last consumer dependency -> queue End / TPipe cleanup
```

首次空槽由队列或所选 SDK 的事件初态提供；每次 Load/Compute/CopyOut 成对发布和消费令牌。最后一次提交消费者之后发布归还依赖，下一次覆盖前等待该依赖；提交不等于已完成。

全部任务结束后归还仍持有的父段和 L0C，End 释放队列事件，TPipe 完成固定事件收尾。空段不等待未生产的就绪事件。混合核声明、AIV→AIC 初始化通知、零贡献 L0C 交接和 SetScheduleMode 按 [filter 同步实现](../../adapters/ascendc/filter/sync-implementation.md) 配对。

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

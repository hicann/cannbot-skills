# Convolution 范式：卷积正向计算流程

> **层级**：通用知识层（generic/）。以达芬奇架构 Cube 通路为基准描述（AscendC / DSL 最终都落到同一套硬件指令，仅封装名不同）。
>
> 术语记号见 [language.md](language.md)；切分大小如何决策见 [tiling.md](tiling.md)；本文件回答"**数据怎么流、指令怎么用、循环怎么写、同步怎么插**"。

---

## 1 卷积计算公式

### 1.1 计算公式

以 2D 为例（1D/3D 是空间维数量的增减，见 §1.3）：

```
y(n, co, h, w) = bias(co) + Σ_{ci=0}^{Ci-1} Σ_{kh=0}^{Kh-1} Σ_{kw=0}^{Kw-1}
                   x(n, ci, h·sH - padTop + kh·dH, w·sW - padLeft + kw·dW) · filter(co, ci, kh, kw)
```

- 语义为**互相关**（与 PyTorch/ONNX 一致，kernel 不翻转）；
- 输入采样坐标落到 `[0, Hi)` 之外（pad 区）时，x 取 0（zero padding）；
- bias 是可选项：长度 Co，逐输出通道一个标量。

### 1.2 输出形状怎么算

每个空间维**独立**套同一个公式（以 H 为例）：

```
dkH = (Kh - 1)·dH + 1                    # 膨胀后的卷积核尺寸
Ho  = floor((Hi + padTop + padBottom - dkH) / sH) + 1
```

| 算子 | 输入 | 卷积核 | 输出 |
|------|------|--------|------|
| Conv1D | (N, Ci, Wi) | (Co, Ci, Kw) | (N, Co, Wo) |
| Conv2D | (N, Ci, Hi, Wi) | (Co, Ci, Kh, Kw) | (N, Co, Ho, Wo) |
| Conv3D | (N, Ci, Di, Hi, Wi) | (Co, Ci, Kd, Kh, Kw) | (N, Co, Do, Ho, Wo) |

**公式使用的三条边界语义**（生成代码不得自行加保护性截断）：

1. **输入小于卷积核时输出仍可能非空**——pad 参与公式：`Hi + padTop + padBottom ≥ dkH` 即有输出。形如 "Hi < dkH 则输出取 1" 的截断是**错误**的（带 pad 时会算丢输出行），必须直接套公式；
2. **公式结果 ≤ 0 表示空输出**（感受野连 pad 都放不下）——这是合法规格：算子无计算语义，输出为空张量，验证时按空输出处理而非报错；
3. **全 pad 窗口只可能出现在 D 方向**——H/W 方向在 pad<kernel 约束下**数学上不可能**整窗全 pad（上方全 pad 需 `ho·sH + dkH ≤ padT`，而 padT < dkH 蕴含 ho·sH < 0；下方全 pad 需 `ho·sH ≥ Hi + padT`，而输出行存在蕴涵 `ho·sH ≤ Hi + padT + padB − dkH < Hi + padT`）。D 方向逐平面取样（每次 kd 取一个输入平面），平面整体越界是常态，**必须保留全零平面处理**。**触发实例**：Di=1、Kd=11、padHead=padTail=5（sD=1）时，输出 Do=1 的感受野覆盖 [−5, 5]，11 个 kd 平面中仅 kd=0（diIdx=−5+0·1=−5，越界→全零）、…、kd=5（diIdx=0，有效）、…、kd=10（diIdx=5，越界→全零）——两侧各 5 个平面走全零分支。

### 1.3 卷积滑动原理图

卷积 = **卷积核在输入图上按 stride 滑窗，每个窗口位置做一次乘累加**。以 `Hi=Wi=5, Kh=Kw=3, sH=sW=1, pad=1` 为例（输出 5×5）：

```
输入 x（pad 后 7×7，·为pad零填充）        卷积核 filter（3×3）
┌───────────────────────┐
│ ·  ·  ·  ·  ·  ·  ·  · │
│ ·  a  b  c  d  e  ·   │        ┌─────────┐
│ ·  f  g  h  i  j  ·   │        │ w0 w1 w2│
│ ·  k  l [m  n  o] ·   │        │ w3 w4 w5│   窗口初始位置
│ ·  p  q [r  s  t] ·   │        │ w6 w7 w8│   （虚线框 m,n,o,r,s,t）
│ ·  u  v  w  x  y  ·   │        └─────────┘
│ ·  ·  ·  ·  ·  ·  ·  · │
└───────────────────────┘        y(0,0) = Σ 窗口元素 × 对应核元素 (+ bias)

窗口滑动：先沿 W 每步移 sW=1 → 移到行尾后沿 H 移 sH=1 回到行首 → 遍历完全部 Ho×Wo 个位置
```

**滑窗的本质**：把每个窗口位置的 `Ci×Kh×Kw` 个元素展平成一行，卷积就变成了矩阵乘（im2col + GEMM，见 §4.3）。

---

## 2 通用卷积在 kernel 中的数据流

数据在 NPU 上的流动主线：**GM → L1 → L0 →（计算）→ L0C → GM**。四个角色（fmap / weight / bias / output）各有各的通路。

**本章总览**（各侧详情见对应小节）：

| 侧 | 小节 | GM → L1 指令 | L1 内容 | L1 → L0 指令 | L0 内容 | 出口 |
|----|------|-------------|---------|-------------|---------|------|
| fmap（A） | §2.1 | DN2NZ | 原始 2D 图（未展开） | LOAD3D | M0×K0 分形（已展开） | — |
| weight（B） | §2.2 | DN2NZ | NZ 分形块（已展开） | LOAD2D | N0×K0 分形 | — |
| bias | §2.3 | 普通搬运 | 全量驻留 | L1→BT 搬运 | BT 段 | Mmad 初值 |
| output（C） | §2.4 | — | — | — | L0C fp32 累加 | FIXPIPE→GM |

### 2.1 fmap 侧：GM → L1 → L0A

```
fmap(GM, NCHW: [N, Ci, Hi, Wi])
  │  DN2NZ 布局转换搬运（随路 ND→NZ，整行搬运）
  ▼
fmap(L1, [Ci1_L1, Hi_L1, Wi_L1, C0])          ← 原始 2D 特征图（未做滑窗展开！）
  │  LOAD3D（随路完成滑窗展开 = 硬件 im2col）
  ▼
fmap(L0A, [K1_L0, M1_L0, M0, K0])             ← 展开后的 A 矩阵分形
     其中 K1_L0 = Ci1_L0·Kh·Kw（K 侧展开），M1_L0 = CeilDiv(M_L0, M0)
```

要点：

1. **L1 里存的是"原始 2D 图"**（一段连续输入行 × 若干通道），不是展开后的矩阵——展开发生在 L1→L0A 的 LOAD3D 里；
2. Hi_L1 由行推理决定（见 §6.3），一般小于 Hi；
3. 备选路径 **GM → UB → L1 → L0**：当 LOAD3D 指令限制不满足（超大 stride/dilation/pad/kernel）或特定优化场景时，退化用普通搬运指令经 UB 中转，软件做滑窗索引。

### 2.2 weight 侧：GM → L1 → L0B

```
weight(GM, NCHW: [Co, Ci, Kh, Kw])
  │  DN2NZ 布局转换搬运（随路 ND→NZ）
  ▼
weight(L1, [Ci1·Kh·Kw, Co1, N0, C0])     ← NZ 分形块（K 侧已展开）
  │  LOAD2D（按子块取，无滑窗语义）
  ▼
weight(L0B, [K1_L0, N1_L0, N0, K0])
```

要点：weight 与 fmap 不同——**GM→L1 时就完成了展开排布**（把 (Kh·Kw, Ci) 视作矩阵做 NZ 转换），L1 里已经是分形块。

**DN2NZ 的 GM 寻址公式**（按 kd 切片逐片装载，2D 视作 kd=1 的特例）：

```
切片基址   = coStart·(Ci·Kd·Kh·Kw) + kd·(Kh·Kw)            ← 3D 时先跳到本 co 的 kd 片
通道步进   = Kd·Kh·Kw                                         ← ⚠️ 3D 与 2D 不同！
通道块偏移 = kC·cinChunk·(Kd·Kh·Kw)                          ← K 在 L1 切块时同样要乘 Kd
搬运形态   = dnNum=co真值, nValue=Kh·Kw, dValue=cin真值（C0 尾自动补零）
```

> ⚠️ **3D 通道步进陷阱**：weight GM 布局是 `[Co][Ci][Kd][Kh][Kw]`，kd 片在通道方向**不连续**——同一 kd 片的相邻通道相距 `Kd·Kh·Kw` 元素而非 `Kh·Kw`。若按 2D 公式寻址，读到的 weight 是错乱的通道混合。

### 2.3 bias 侧：GM → L1 → BT

```
bias(GM, [Co])
  │  GM→L1 普通搬运（全量一次；尾块零填充到 32B 对齐）
  ▼
bias(L1, [Co^align])
  │  L1→BT 搬运（逐 N 段：nBlk 对齐段，长度按 bias dtype 字节计）
  ▼
bias(BT, [nBlk 段]) ──► MMAD 首步 cmatrixSource=true 读 BT 作 L0C 初值
```

bias 不参与 MMAD 的矩阵乘，而是作为 **L0C 的累加初值**参与（见 §4.5）。通路分三步（详细配方与越界适配见适配层 bias-bt-path）：

1. **GM→L1 全量一次**：bias 全量（Co^align 个元素，host 侧零填充到 32B 对齐）拷入 L1，全程驻留；
2. **L1→BT 逐 N 段**：每个 N 块装载对齐段，按 bias dtype 字节折算 32B 块数；
3. **MMAD 初值**：带 bias 首步 `cmatrixInitVal=false、cmatrixSource=true`（L0C 从 BT 取初值）；无 bias 首步 `cmatrixInitVal=true`（零初始化）。

### 2.4 cube 计算与 output 搬出

```
fmap(L0A, [K1, M1, M0, K0]) ─────┐
                                 ├──► MMAD ──► out(L0C, [Co1, M1, M0, N0], fp32累加)
weight(L0B, [K1, Co1, N0, K0]) ──┤                    │
                                 │                    │ FIX_L0C_TO_OUT（列主序，NZ→ND 还原）
bias(BT, [Co1, N0]) ─────────────┘（只进 L0C 初值）    ▼
                                                 out(GM, NCHW: [N, Co, Ho, Wo])
```

### 2.5 数据流总图

```
              A 侧（fmap）              B 侧（weight）         C 侧（output）
           ┌──────────────────┐    ┌──────────────────┐   ┌──────────────────┐
   GM ─────► L1 (2D原图, NZ)  │ GM─►│ L1 (NZ分形块)    │   │  L0C (fp32累加)  │
           │   │ LOAD3D       │    │   │ LOAD2D       │   │   │ FIXPIPE     │
           │   ▼              │    │   ▼              │   │   ▼            │
           │  L0A (M0×K0分形) │    │  L0B (N0×K0分形) │◄──┼───┘  GM (NCHW)  │
           └───────┬──────────┘    └───────┬──────────┘   └──────────────────┘
                   └───────► MMAD ◄────────┘   （bias 经 BT → L0C 初值）
```

---

## 3 数据格式：ND、DN、NZ、FRACTAL_Z、FRACTAL_Z_3D

> ⚠️ **先澄清一个命名陷阱**：ND/DN/NZ 里的 "D"、"N"、"Z" 只是**快轴/慢轴的记号**，与 Cube 最小单元的 N0/M0/K0 **完全无关**。

### 3.1 ND 格式（channel 在内）

把 D 理解为 channel，**先排 D（channel 连续），再排 N（空间方向）**，即 `[H*W, C]`——NHWC 就是 ND：

```
      ------ D (channel) ------>
      c0 c1 c2 ... c(Ci-1)  ← 一行 = 一个空间位置的所有通道（连续）
      c0 c1 c2 ...
  N   c0 c1 c2 ...
  │   ...
  ▼   c0 c1 c2 ...
```

### 3.2 DN 格式（channel 在外）

**先排 N（空间连续），再排 D（channel）**，即 `[C, H*W]`——NCHW 就是 DN：

```
      ------ N (空间) ------>
      h0w0 h0w1 h0w2 ...    ← 一行 = 一个通道的整张图（连续）
      h1w0 h1w1 ...
  D   ...
  │   h(H-1)w0 ...
  ▼
```

### 3.3 NZ 格式（最内轴按 C0 分块）

Z = C0（一个 32 字节的通道块，元素个数 `C0 = 32B/dtypesize`）。**先排 Z（C0 内元素连续），再排 N，再排下一个 Z**：

```
   ----C0--→   ----C0--→        C0 块内连续；块沿 N 方向排满后，
   ----C0--→   ----C0--→        再排下一个 C0（通道下一组）
   |                        |
   N          ...           N
   |                        |
   ----C0--→   ----C0--→
```

### 3.4 FRACTAL_Z 格式（权重的标准分形排布）

**FRACTAL（分形）** 是最小物理块，大小 `N0 × C0`（N0=16 行 × C0 列，N0 方向是 cout、C0 方向是 Cin）：

```
   ---C0--->
   ---C0--->      ×N0 行 = 一个 FRACTAL
   |  ...
   N0
   ---C0--->
```

**FRACTAL_Z = 把 FRACTAL 按 NZ 方式排布**：分形沿 N（cout）方向排成一列，列沿 C1（Cin 分形块）方向展开，**C1（K 侧）必须为最外轴**：

```
   [分形]  [分形]         物理轴序（2D 权重 [Co, Ci, Kh, Kw]）：
   N0×C0   N0×C0          [Ci1·Kh·Kw, Co1, N0, C0]
     |       |                  ↑K侧展开(最外)  ↑N1  ↑fractal
     N       N
     |       |            C0 = 32B / dtypesize（元素个数）：
     [分形]  [分形]           dtypesize=1 → C0=32   (int8/fp8)
        |                    dtypesize=2 → C0=16   (fp16/bf16)
        ---C1--->            dtypesize=4 → C0=8    (fp32)
                             dtypesize=8 → C0=4    (fp64)
       C1 = CeilDiv(C, C0)（注意：C1 是通道的分形块数，C0 才是块内元素个数）
```

### 3.5 FRACTAL_Z_3D 格式

与 FRACTAL_Z 基本相同，只是 **N 方向多一根 D 轴**（D 在 N 的外层：先排 N，再排 D）：

```
物理轴序（3D 权重 [Co, Ci, Kd, Kh, Kw]）：
[Kd·Ci1·Kh·Kw, Co1, N0, C0]      ← K 侧展开（含 Kd）仍为最外轴；Co1 内再按 D 分组
```

> 精确定义：`(G, Kd, Ci1, Kh, Kw, Co, C0)`，即 K 侧展开 `[Kd·Ci1·Kh·Kw]` 最外，然后 Co1，然后分形 `[Co方向16, C0]`。groups>1 时还有按对角阵打包的 mag_factor 排布（当前范围 groups=1，不展开）。

### 3.6 NC1HWC0 格式（5HD）

即 NZ 排布：Z=C0，"N" 换成复合轴 **W → H → C1 → N**（W 最快）：

```
物理轴序：[N, C1, H, W, C0]
```

### 3.7 达芬奇架构的最小 cube 计算

一次最小的 MMAD：`(M0×K0, NZ) × (K0×N0, NZ) → (M0×N0)`：

```
   A (L0A)              B (L0B)               C (L0C)
   --K0-->              ┌─┐┌─┐┌─┐┌─┐
   ------>  M0 行       │ ││ ││ ││ │  N0 列       ------N0------>
   ------>              └─┘└─┘└─┘└─┘            -------------->
   |                    └──K0──┘                 -------------->
   M0                     每列一个                 --------------
                                                  （M0×N0 个累加值，
                                                   沿 K0 全深度乘累加）
```

M0=N0=16；K0=C0=32B/dtypesize（fp16 时三者都是 16）。

---

## 4 通用卷积 kernel 需要使用哪些指令 / API

### 4.1 指令总表

卷积 kernel 由下列**硬件指令**（操作语义，反汇编中亦以对应助记符出现）构成。各开发语言对这些指令的 API 形态与参数结构体**在语言适配层定义**（AscendC 见 [../adapters/ascendc/instruction-mapping.md](../adapters/ascendc/instruction-mapping.md)），本节及后续小节只描述**指令的语义、参数含义与限制**（语言中立）：

| 指令 | 作用 |
|------|------|
| GM→L1 布局转换搬运（DN2NZ / ND2NZ 模式） | GM→L1，随路完成布局转换。**DN2NZ = 源为 DN**（channel 在外，NCHW 类输入用）；**ND2NZ = 源为 ND**（channel 在内，NHWC 类输入 / HWCN 类权重用）。前缀指源格式 |
| LOAD3D | L1→L0A，**随路滑窗展开（im2col）** |
| LOAD2D | L1→L0B，分形子块搬运 |
| MMAD | L0A×L0B→L0C 矩阵乘累加 |
| FIX_L0C_TO_OUT（FIXPIPE） | L0C→GM，随路 NZ→ND 还原（可融合量化/激活） |
| L1→BT 搬运 | bias 装载进 BT 缓冲 |
| GM→L1 普通搬运 | 无布局转换的直通搬运（bias 全量、对齐段） |

**参数取值规则汇总（对齐值 vs 真值）**——同一循环体内各指令的"长度/步进"参数取值口径不同，尾块场景（M/N/K 任一非整除）是错配高发点：

| 指令 | 取**真值**（未对齐） | 取**对齐值/分形数** |
|------|---------------------|---------------------|
| MMAD | k（本步 K 深度，尾步=尾值） | m=AlignB(m,M0)、n=AlignB(n,N0) |
| FIXPIPE | nSize、mSize（写出真长度，多余分形列/行不落 GM） | srcStride=AlignB(m,M0) |
| LOAD3D | mExtension（窗口数）、currentKL0（本步 K） | dstStride=AlignB(m,M0)/M0（M0 分形个数） |
| LOAD2D | — | MStep=CeilDiv(n,N0)、KStep=k/K0（一律按分形数） |
| DN2NZ | dValue（通道真值，尾块依赖 C0 补零） | — |

**各指令参数的约束与限制汇总**（超限即需退化软件搬运或改切分；详细语义见对应小节）：

| 指令 | 参数 | 约束/限制 | 超限后果与出路 |
|------|------|----------|---------------|
| DN2NZ / ND2NZ | burst 长度 | ≤ 65535 | 分段搬运 |
| | 源/目的地址 | 目的 32B 对齐（源不对齐带宽降档） | 调整切分基址 |
| | dValue 尾块 | 非 C0 整数倍由指令自动补零，无需目的区预清零 | — |
| LOAD3D | stride（每维） | ≤ 63 | 退化软件搬运（GM→UB→L1） |
| | dilation（每维） | ≤ 255 | 同上 |
| | pad（每维） | ≤ 255 | 同上 |
| | 膨胀后核尺寸 | ≤ 511 | 同上 |
| | posK / M 起点累计 | < 65536（16bit） | 缩小 K/M 块 |
| | L1 图 Hi_L1·Wi_L1 相关积 | ≤ 32767（15bit） | 缩小行推理窗口 |
| LOAD2D | MStartPosition / MStep | N0 分形数口径，≤ 紧邻上限（16bit） | 缩小 nBlk/nL0 |
| | KStartPosition / KStep | K0 分形数口径，≤ 紧邻上限（16bit） | 缩小 kChunk/kL0 |
| MMAD | m / n | 对齐值（AlignB(·,M0/N0)），≤ 65535（16bit） | 缩小块 |
| | k | 真值，≤ 65535（16bit） | K 再切步 |
| FIXPIPE | burst | ≤ 65535 | 分段写出 |
| | srcStride 及子参数 | 16bit（srcNzMatrixStride 按 N0 元素组计） | 缩小块/分段 |
| | dstStride / dstDnMatrixStride | 32bit（元素） | 大跨度按 32bit 用，见 scene §3.3 |

### 4.2 DN2NZ / ND2NZ（GM→L1 的布局转换）

以单 batch 的 `Ci×Hi×Wi` fmap 为例，NCHW 输入走 DN2NZ：

```
搬运前 GM（ND：channel 连续）            搬运后 L1（NZ：C0 块连续）
(w0,c0)(w0,c1)...(w0,c15)(w0,c16)...    [w0..wHi*Wi-1 的 c0..c15]   ← 第 0 个 C0 块
(w1,c0)(w1,c1)...                       [w0..wHi*Wi-1 的 c16..c31]  ← 第 1 个 C0 块
（空间位置为行、通道为列）                （按通道 C0 分块，块内整图空间连续）
```

关键参数（DN2NZ 指令参数，以 fmap 为例）：

| 参数 | 含义 | fmap 取值示例 |
|------|------|--------------|
| dnNum | N 方向（源矩阵份数）重复次数 | fmap 全宽整段搬运=1；**weight 侧=本段输出通道数 coReal**（每个 co 一份矩阵） |
| nValue | 每份矩阵的 N 长度 | fmap=Hi_L1·Wi（整段空间）；weight=Kh·Kw（核空间位置） |
| dValue | 每份矩阵的 D（通道）长度 | 本段通道**真值**（尾块取余数，尾段由 C0 补零，见下注） |
| srcDnMatrixStride | 源相邻矩阵步进 | fmap=0（单矩阵）；weight=Ci·Kd·Kh·Kw（GM 里跨一个输出通道） |
| srcDValue | 源 D 步进（跨一个通道） | fmap=Hi·Wi（3D 为 Di·Hi·Wi）；weight=Kd·Kh·Kw（**3D 与 2D 不同**，见 §2.2） |
| dstNzC0Stride | 目标 C0 块步进 | fmap=Hi_L1·Wi；weight=Kh·Kw·nBlk |
| dstNzNStride | 目标 N 步进 | fmap=1；weight=nBlk（nBlk 份 co 矩阵在目标内交错） |
| dstNzMatrixStride | 目标相邻矩阵步进 | fmap=0；weight=K0（矩阵间以 K0 分形为单位排布） |

**weight 侧目标布局推导**（与 §2.2 的 L1 `[Ci1·Kh·Kw, Co1, N0, K0]` 分形布局对应）：weight 的每份源矩阵是 `[N=Kh·Kw, D=Ci]`；DN2NZ 把 D 方向按 C0 分块后，nBlk 份 co 矩阵以 dstNzNStride=nBlk **交错**排布（K0 分形内 N 连续、跨分形按 dstNzMatrixStride=K0 推进、通道方向按 dstNzC0Stride 推进）——三个步进共同定义了 LOAD2D 可按 (N0 分形, K0 分形) 二维寻址的规则布局。**改动任一步进（nBL1 切块、K 侧重排）必须重推导此布局**，写错表现为通道错乱（输出整体量级正常但数值大面积不符）。

> **C0 尾自动补零的保证范围**：dValue 非 C0 整数倍时，目标尾段由指令自动补零，无需目的区预清零

参数限制：burst 长度 ≤ 65535、地址 32B 对齐。**带宽注意**：NCHW 用 DN2NZ 带宽最高约 32B/cycle，DHWCN 用 ND2NZ 可达 128B/cycle（相差 4 倍，tiling 代价模型要计入，见 [tiling.md](tiling.md) §3.7）。

### 4.3 LOAD3D（L1→L0A，硬件 im2col）

**im2col 原理**——把滑窗展开写成矩阵：每个输出位置的感受野展平成 A 矩阵的一行：

```
L1 中的 2D 图（Hi_L1 × Wi_L1）             L0A 上的 A 矩阵（M × K）
┌───────────────┐
│ □ □ □ □ □ □ □ │   感受野(2×2)          行 = 输出位置 m
│ □ [■ ■] □ □ □ │   ────────►            列 = (ci, kh, kw) 展平
│ □ [■ ■] □ □ □ │                          ┌───────────────┐
│ □ □ □ □ □ □ □ │                          │ x(m,ci,0,0) x(m,ci,0,1) ... │  ← m 行 = 窗口m
└───────────────┘                          │ x(m+1,ci,0,0)   ...         │
  窗口沿 W、H 滑动                          └───────────────┘
```

LOAD3D 一条指令完成"取窗 + 展开 + 摆成 M0×K0 分形"三件事。配置分两组（位域下发）：

| 配置组 | 内容 | 语义 |
|-------|------|------|
| 窗口几何 | L1 图形状(Hi_L1×Wi_L1)、padLeft/Right、strideH/W、kernelH/W、dilationH/W、channelSize | "在 L1 的 2D 图上怎么滑窗" |
| 块定位 | currentKL0（本步 K 长）、posK（K 累计）、mStartPt（M 起点列偏移）、repeat/stride | "这次展开出哪个 [M0,K0] 子块" |

**块定位参数的精确语义**（单位错误是滑窗类 bug 的高发点）：

| 参数 | 单位/语义 | 取值规则 |
|------|----------|---------|
| mStartPt | **输出列号**（本 M 块首个窗口对应的输出 w 坐标，`= mStart mod Wo`），硬件内部按 strideW 换算到图内列。**不是**图内元素偏移，**不要**手工乘 stride | 传起始列号×strideW 会导致整块错；传输出列号正确 |
| posK | K 展平坐标（(ci, kh, kw) 展平后的元素偏移），每步累加 currentKL0，AL1 重装时归零 | |
| mExtension | 本步展开的 M 行数（输出窗口数，用真值不对齐到 16） | |
| repeat/dstStride | 四字段 `{repeatStride, repeatTime, repeatMode, dstStride}` = `{0, 1, 0, AlignB(m,M0)/M0}`：repeatTime=1（一次展开 mExtension 个窗口，不重复搬运）、dstStride=**M0 分形个数**（L0A 内相邻 M0 分形的排布步进）、repeatStride/repeatMode 恒 0 | |
| FMatrix padList | **元素序固定 {left, right, top, bottom}**，单位=图内列/行数；bottom 恒填上限值（255）表示**下边界不设防**——窗口越过图底的行为由行推理保证（装载行数已覆盖全部窗口），上界则必须精确（越界窗口会读到 L1 之外） | |

**M 块跨多输出行（一般 M 模式）的语义**：

- mExtension 允许 **> Wo**：窗口序列跨输出行，行切换由硬件在 L1 图内自动完成，软件只需保证行推理公式（§6.3）覆盖本块全部窗口的输入行；
- mStartPt 仍取**本块首输出的列号**（跨行不改变取值规则）；
- 跨行与 W 尾块叠加（本块末行不满）时：mExtension 取本块 M 真值，MMAD 的 m 取 AlignB(真值)，FIXPIPE 的 mSize 取真值——对齐规则同 §4.1 汇总表，无额外修正。

**参数限制（超限必须退化软件搬运）**：

| 限制 | 值 |
|------|----|
| stride（每维） | ≤ 63 |
| dilation（每维） | ≤ 255 |
| pad（每维） | ≤ 255 |
| 卷积核尺寸（膨胀后） | ≤ 511 |
| posK / M 起点 累计值 | < 65536（16bit） |
| L1 图 Hi_L1·Wi_L1 相关 | ≤ 32767（15bit） |

### 4.4 LOAD2D（L1→L0B，B 侧子块抽取）

从 L1 的 weight 分形布局中按 (N0 分形 × K0 分形) 二维抽取 `[kL0, nL0]` 子块进 L0B。**无滑窗语义**，是纯分形块的定位搬运：

```
L1 [Ci1·Kh·Kw, Co1, N0, K0]          L0B [kL0, nL0]
（§4.2 的 DN2NZ 目标布局）    ──抽取──►  [K1_L0, N1_L0, N0, K0]
```

参数单位语义（**全部按分形数计，不按元素计**——单位错一位即全图错）：

| 参数 | 单位/语义 | 卷积取值 |
|------|----------|---------|
| MStartPosition | **N（输出通道）方向起点**，N0 分形数 | nL0Iter·(nL0/N0) |
| MStep | N 方向长度，N0 分形数 | CeilDiv(curN, N0) |
| KStartPosition | K 方向起点，K0 分形数 | kIter·(kL0/K0)（BL1 重装时归零） |
| KStep | K 方向长度，K0 分形数 | curK/K0 |
| SrcStride | 源内相邻 N0 分形组的步进，N0 分形数 | nBlk/N0 |
| DstStride | 目的相邻分形步进，N0 分形数 | =MStep |
| IfTranspose | 转置开关 | false |

> ⚠️ **命名陷阱**：参数名里的 "M"（MStartPosition/MStep）实际作用在 B 矩阵的 **N（输出通道）方向**——LOAD2D 的视角里 B 矩阵的行是 K、列是 N，"M 方向"沿用了其矩阵行命名，与卷积的 M（输出空间）无关。

### 4.5 MMAD（L0A × L0B → L0C）

```
L0A [K1, M1, M0, K0]     L0B [K1, N1, N0, K0]        L0C [Co1, M1, M0, N0]
（A 矩阵分形）            （B 矩阵分形）               （累加结果，fp32）
   ┌────┐ ┌────┐           ┌───┬───┬───┬───┐
   │M0× │ │M0× │    ×      │N0×│N0×│N0×│N0×│    =     每 (M0, N0) 对一个
   │K0  │ │K0  │  (沿 K    └───┴───┴───┴───┘          M0×N0 累加块
   └────┘ └────┘   归约累加)   ↑沿 K1 逐块累加
```

关键参数（MMAD 指令参数）：

| 参数 | 含义 | 卷积取值 |
|------|------|---------|
| k | 本次乘加的 K 深度 | currentKL0（尾块取 kL0Tail） |
| cmatrixInitVal | L0C 初始化开关 | 无 bias：首步=true（零初始化 L0C），之后=false。**带 bias：首步=false**（初值取 BT） |
| cmatrixSource | 初始化数据源 | bias（经 BT）；无 bias 时初始化为 0 |
| unitFlag | L0C 单元占用标志：0=不使能；2=仅置位（本块 L0C 被 FIXPIPE 消费前保持占用）；3=置位+翻转（占用切换到另一份 L0C，双缓冲乒乓的基础） | K 归约中间步=2（还剩后续 MMAD 要写同一块 L0C）；**末步=3**；FIXPIPE 同步置 3。⚠️ 末步误留在 2 的后果**未定义**（可能不翻转导致下一次写撞未消费的 L0C），须在循环界与 isLast 判定中一致使用逐层取余口径杜绝此场景 |

**bias 通路**：bias 不进 MMAD 操作数，而是经 BT 作为 L0C 的**累加初值**——首个 k 步 `L0C = bias + A×B`，其余步 `L0C += A×B`。

**L0C 累加类型（架构语义，生成代码的默认取值）**：

| 输入 dtype | L0C 累加类型 | 说明 |
|------|------|------|
| fp16 / bf16 | **fp32** | 窄精度输入按宽精度累加，避免长 K 链的累加误差；输出时由 FIXPIPE 随路转换回目标 dtype |
| int8 类（量化） | int32 | 定点累加，还原（×scale）在 FIXPIPE 随路或 Vector 侧完成（见 [scene.md](scene.md) §2） |

> ⚠️ 设计约束：design 与生成代码**一律按"输入 dtype → 累加类型"的基线组合书写**，不引入定点移位类的非基线机制。各语言实现该组合的 API 形态与版本注意事项见适配层（ascendc：[../adapters/ascendc/buffer-and-types.md](../adapters/ascendc/buffer-and-types.md)）。

### 4.6 FIX_L0C_TO_OUT（L0C → GM）

以搬出 `Co1 × (M1·M0) × N0` 的 output tile 为例（NCHW 输出走**列主序**配置）：

```
L0C（NZ 分形）                          GM（NCHW）
┌──────┐ ┌──────┐ ┌──────┐             out(n, co_start..co_start+nL0,
│ M0×N0│ │ M0×N0│ │ M0×N0│  ──FIXPIPE──►        ho..ho+mL0 行连续段)
│      │ │      │ │      │             dstStride = Ho·Wo（跨一个输出通道）
└──────┘ └──────┘ └──────┘             srcStride = AlignB(m, M0)
（M1 个分形沿 M 排）                    一次完成 fp32→fp16 转换 + NZ→ND 还原
```

**子参数（NZ→ND 列主序还原的源/目的步进）单位语义**：

| 子参数 | 单位/语义 | NCHW 列主序取值 |
|--------|----------|----------------|
| dnNum | 写出的 dn 矩阵份数（多次 loop 迭代） | 1（单矩阵）；**>1 用于多个 dn 矩阵一次写出**（见下"dnNum>1 的用法"） |
| srcNzMatrixStride | 源 L0C 内相邻 dn 矩阵步进，**N0 元素组**（= 元素数 ÷ N0） | 单个 dn 矩阵的对齐 L0C 跨度 ÷ N0（见下） |
| dstDnMatrixStride | 目的 GM 内相邻 dn 矩阵步进，**元素** | 相邻 dn 矩阵在 GM 的跨度（见下） |
| srcNzC0Stride | 源 L0C 内 C0 方向步进 | 1（N0 分形内列连续） |

**dnNum>1 的用法**——多个 dn 矩阵一次写出。dn 矩阵是对输出的一次划分，两种典型形态：

| 形态 | dn 矩阵划分 | dnNum | dstDnMatrixStride |
|------|------------|-------|-------------------|
| batch 批写 | **每个 batch** 一份 [n × m] 矩阵（M 块几何相同） | batch 数 | Co·Do·Ho·Wo（batch 在 GM 的跨度） |
| 行窗口批写 | 同一 N 块的**每行窗口**一份 [n × 行宽] 矩阵 | 行数 | Wout（相邻行窗口起点距） |

batch 批写的参数口径（源侧）：

- 各 dn 矩阵在 L0C 按**对齐跨度**（mAlign × nAlign 元素）依次堆叠，MMAD 分别写入各矩阵基址（`cl0[矩阵序号 × mAlign·nAlign]`）；
- `srcNzMatrixStride` = 单个 dn 矩阵的对齐 L0C 跨度 ÷ N0（**N0 元素组**口径）；
- **每个 dn 矩阵的首个 MMAD 都必须独立做 L0C 初始化**（cmatrixInitVal=true 零初始化，或 cmatrixSource=true 读 BT）——dn 矩阵之间不会自动传递初值，漏初始化表现为部分矩阵带垃圾初值（数值量级正常但大面积不符）。

行窗口批写（HW 行切分）的源侧口径与 batch 批写不同：dn 矩阵不是整矩阵堆叠，而是 NZ 分形内的行方向迭代，`srcNzMatrixStride` 取单行对齐跨度（`AlignB(行宽, M0)`，N0 组口径）。落地该形态时按内置 conv 实现的取值公式核对，不要套用 batch 批写的堆叠公式。

参数限制：burst ≤ 65535；**srcStride 及子参数为 16bit、dstStride 为 32bit**（大 Ho·Wo 场景按 32bit 使用，见 [scene.md](scene.md) §3.3 的 `Ho·Wo ≤ 2^32` 约束）；支持随路量化/激活融合（当前范围仅 dtype 转换）。

### 4.7 GM→L1 搬运的通用形态

用法：给定 GM 源地址 + L1 目的地址 + 上述 DN2NZ/ND2NZ 转换参数（或普通对齐搬运参数），完成一段数据的搬运与随路格式转换。fmap 侧（§2.1）与 weight 侧（§2.2）共用，区别只在转换参数。要点：

- 目的地址 32B 对齐；源地址建议对齐（不对齐时带宽降档）；
- 每次搬运的最优粒度是"整行 × 若干通道块"（连续地址最大化）。

### 4.8 L1→BT 搬运

用法：把 L1 中缓存的 bias 段拷入 BT。每次 `AlignB(curN, N0)` 个元素（按 32B 块计 burst）。BT 容量小（KB 级），N 块超出 BT 容量时按 nL0 分段。详见适配层 bias-bt-path。

---

## 5 NPU 是多核的，卷积 kernel 一定是分核计算的

### 5.1 对 fmap、weight 的哪些轴分核？依据是什么？

**绑核永远面向 output 切**（output 的各块之间无数据依赖，天然可并行；fmap/weight 只读共享）：

| 模式 | 绑核轴（output 视角） | 说明 |
|------|---------------------|------|
| M 模式 | N(batch)、C(cout)、D(dout)、**HW 合轴成 M** | H、W 合成一根 M 轴统一切分，切分更简单、负载更均衡 |
| HW 模式 | N(batch)、C(cout)、D(dout)、**H(hout)** | H、W 保持独立轴，只切 H |
| groups>1 时 | 再加 groupDim | 例：groups=8、4 个核 → 每核 2 个 groups，核内再按 groups=2 分配 cout/cin |

**为什么 HW 模式不切 W？** 从搬运出发：fmap GM→L1 用 DN2NZ 时，**最优方案是 W 方向整行搬运**；若切 W，每个核只搬行的一部分，相邻核会重复搬相同的行（h 方向重复搬运）。

**什么时候可以切W?**：Conv1d 场景使用（此时 ho≡1，只有 W 能提供空间并行度）。

### 5.2 如何保证分核后计算不冲突？

1. **输出互斥**：各核的 output 区域按轴切分后天然不重叠——`blockIdx` 经**除法链**反解为本核各轴索引。反解规则：**从最内维往外在链尾取模**。以 (batch, co, m) 三维绑核为例（batchDim×nDim×mDim = blockDim）：
   ```
   mIdx  =  blockIdx % mDim                            # 最内维：直接取模
   nIdx  = (blockIdx / mDim) % nDim                    # 中间维：先除掉内维积
   bIdx  =  blockIdx / (mDim × nDim)                   # 最外维：除掉内二维积
   ```
   **worked example**（32 核，batch=2、Co=96、M=1280，取 batchDim=2、nDim=4、mDim=4）：
   ```
   blockIdx=13 → mIdx=13%4=1, nIdx=(13/4)%4=3, bIdx=13/16=0
   每核规模: batch=1, Co=CeilDiv(96,4)=24, M=CeilDiv(1280,4)=320
   本核起点: batch=0, co=3×24=72, m=1×320=320
   尾核处理: blockIdx=31 → bIdx=1,nIdx=3,mIdx=3；若 Co=90 则该核 co 段=90−72=18（尾核余数）
   ```
   **N0 对齐修正的统一做法**：切分在**对齐后的规模**上进行（N 方向用 Co^align=AlignB(Co,N0) 参与切分与反解），则每核 co 起点天然是 N0 的倍数，无需事后修正；写出时按真值截断（FIXPIPE nSize=真值）。若直接用真值 Co 切分，则每核起点须补一步 `AlignB(coStart, N0)` 修正。

2. **launch 形态**：blockDim = 各维绑核数之积（batchDim × nDim × mDim × doDim × groupDim）；简化实现常取二维（batch × M），一般式含 N/Do/Group 维。**输出块是最小分配单位**——每核分到整数个块，块边界不跨核。反解顺序固定**从最内维取模、最外维做除**。

**两种切分口径的关系**（先选维数、再算规模，两个层次不冲突）：

| 口径 | 用途 | 做法 | 适用场景 |
|------|------|------|---------|
| **因子切分** | 选 dimDim 组合 | 取"各维规模与核数的公因子"，保证各维整除、无尾核 | 形状与核数有公因子时优先 |
| **CeilDiv + 尾核** | 算每核规模 | 非整除维按 CeilDiv 切，余数归尾核 | 因子搜索失败时退化使用 |

执行顺序：先用因子法搜索（绑满 + 均衡），搜索失败退化为 CeilDiv 口径——两口径不会同时作用于同一维的规模计算。

| 约束 | 说明 |
|------|------|
| 输出互斥 | 各核 output 区域按轴切分后天然不重叠 |
| 输入只读 | fmap/weight/bias 全部只读，多核读同一份数据无冲突（L2 天然共享） |
| 无核间同步 | 正向卷积核间零依赖、零通信——这是"绑核面向 output"的根本好处 |

### 5.3 尾核与正常核的差异

各轴规模除不尽核数时产生尾核（最后一个核负载小）：

| 差异点 | 正常核 | 尾核 |
|--------|--------|------|
| 负载规模 | maxDimPerCore | `wholeDim - (realDim-1)·maxDimPerCore`（余数） |
| 迭代次数 | 完整循环次数 | 部分循环可能只剩 1 次甚至跳过 |
| 尾块叠加 | 可能遇到尾块 | 尾核 + 尾块双重叠加，最易出越界/漏算 |
| N 方向 | 对齐到 N0 | 尾核 N 起点需按 N0 对齐修正（保证 FIXPIPE 分形对齐） |

设计检查：`每核规模 × 核数 ≥ 总规模` 恒成立（穷尽性），尾核按余数收缩。

---

## 6 单核中的分块计算：切分与迭代

### 6.1 各张量的轴定义

| 张量 | 轴 | 说明 |
|------|----|------|
| fmap | N(batch)、C(channel=Ci)、H(Hin)、W(Win) | 输入 |
| weight | N(cout=Co)、C(channel=Ci)、H(kh)、W(kw) | 注意 weight 的 N 是输出通道 |
| output | N(batch)、C(cout)、H(Hout)、W(Wout) | 绑核/切分都面向它 |

### 6.2 对哪些轴切分？优先级是什么？

切分优先级（依据详见 [tiling.md](tiling.md)）：

```
① K 轴（Ci×Kh×Kw，3D 含 Kd，reduce 轴）── 最优先考虑"全载"：
     reduce 轴切了就要多次装载 A/B 重复参与计算，全载能最大化减少重复搬运
② L0 层级先决策（mL0/nL0/kL0）── 保 cube 利用率：
     L0 块越大，MMAD 次数越少、cube 效率越高；在 L0A/L0B/L0C 容量约束下取尽量大
③ L1 层级后决策（mAL1/nBL1/kAL1/kBL1）── 减少重复搬运：
     L1 块由 L0 块向上推导；fmap/weight 哪侧能全载哪侧全载（fullLoad 四态）
④ M/N 轴 —— 空间维（M=Ho·Wo）与 cout（N）互补切分，平衡负载与搬运
```

### 6.3 单核 M 块与 fmap 行推理（A 侧占用模型）

M 模式下 mAL1 块覆盖输出行范围 `[hoStart, hoEnd)`（块可跨多输出行，跨行窗口语义见 §4.3），需要的输入行数（**行推理**）：

```
hiLoad  = (hoEnd - hoStart - 1)·sH + dkH          # dkH = (Kh-1)·dH + 1
hiIdx   = hoStart·sH - padTop                       # 起始输入行（可为负 → 折算 padTopL1）
hiLoad -= padTopL1                                  # hiIdx<0 时：hiIdx=0、扣掉顶部 pad 行，交给 FMatrix padTop 虚拟化
hiLoad  = min(hiLoad, Hi - hiIdx)                   # ⚠️ GM 下边界钳制：装载行数不得超过 [hiIdx, Hi)
```

**下边界钳制不可省略**：块窗口伸入底部 pad 时 `hiIdx + hiLoad` 会超过 Hi；超出的行属底部 pad，由 FMatrix `padBottom=255` 虚拟填零。若不钳制而按未截断行数从 GM 装载，既**越界读 GM**，又把越界数据当有效图像行参与计算（末尾输出行数值错误，故障特征见 §7.2 经验表）。

A 侧 L1 占用预算 = `cinA × min(行推理最坏值, Hi) × Wi × dtypesize × 乒乓份数`（最坏情形口径见 [tiling.md](tiling.md) §3.3，为逐块精确装载的安全上界）。**stride 越大，切 M 对 A 侧占用的收益越差**（hiLoad 被 dkH 撑住下限）。pad 处理：

| pad 方向 | 处理方式 | 占 L1？ |
|---------|---------|--------|
| W 左右 | LOAD3D 窗口参数虚拟化（L1 只存有效列） | 否 |
| H 上 | 块边界折算 padTopL1 后虚拟化 | 否 |
| H 下 | 行推理下边界钳制 + FMatrix padBottom=255 虚拟填零 | 否 |
| 全 pad 窗口 | 显式填零块 | 是 |
| D 头尾（3D） | **软件显式填零进 AL1** | 是 |

### 6.4 切分后 kernel 如何迭代各个轴（伪代码）

M-first 顺序（外 → 内；N-first 把两对 M/N 循环各自交换）：

```cpp
for (nBL1Iter  = 0; nBL1Iter  < CeilDiv(singleCoreCo, nBL1);   ++nBL1Iter)   // N 大块(L1)
for (batchIter = 0; batchIter < singleCoreBatch;                ++batchIter)
for (dOutIter  = 0; dOutIter  < ddr2l1LoopD;    ++dOutIter)   // 仅 3D
for (mAL1Iter  = 0; mAL1Iter  < CeilDiv(singleCoreM,  mAL1);   ++mAL1Iter)   // M 大块(L1)
for (nL0Iter   = 0; nL0Iter   < CeilDiv(nBL1, nL0);            ++nL0Iter)    // N 子块(L0)
for (mL0Iter   = 0; mL0Iter   < CeilDiv(mAL1,  mL0);           ++mL0Iter) {  // M 子块(L0)
    for (kIter = 0; kIter < CeilDiv(kAL1, kL0); ++kIter) {                   // K 归约
        if (kIter % (kAL1/kL0) == 0)  A侧L1装载();   // DN2NZ 布局转换搬运
        if (kIter % (kBL1/kL0) == 0)  B侧L1装载();
        LOAD3D(al0, al1, 窗口几何+块定位);           // L1→L0A，随路 im2col
        LOAD2D(bl0, bl1, N/K子块定位);               // L1→L0B
        MMAD(cl0, al0, bl0, {k=本步KL0, 首步含bias初值});
    }
    FIX_L0C_TO_OUT(outputGm + tile偏移, cl0, 列主序参数);   // 每个tile算完立即搬出
}
```

循环次数记号（TilingData 下发）：`ddr2l1LoopM = CeilDiv(singleCoreM, mAL1)`、`l12l0LoopM = CeilDiv(mAL1, mL0)`、`l12l0LoopN = CeilDiv(nBL1, nL0)`、`ddr2l0LoopK = CeilDiv(kAL1, kL0)`。

### 6.5 从 output（N=1,Cout=1, H, W）的视角观测各个切分轴的迭代过程（图示）

固定 N=1、Cout=1，剥掉 batch 与通道维，output 退化为一张 `H×W` 单通道特征图——§6.4 的六层循环骨架在这张图上的投影一目了然，也是失败定位时"差异元素 → 迭代步"反解的底图。

示例设定（块大小取易画的小值；实际 M0=16、块大小由 tiling 决策，数值不改变投影规则）：`H=4, W=8 → M=32`；多核 `mDim=3`（`singleCoreM = CeilDiv(32,3) = 11`，每核 2 个 M 块，尾核收缩为 10）；`mBlk=6`（跨行 M 块）；`mL0=3`（L0 子块）；K 侧任意（本视图不可见）。

**图 a：三级切分在 output 平面上的投影**（每格标注所属块号，`m = h·W + w`）：

```
h=0:  [0][0][0][0][0][0][1][1]      m=0..7     ← 块1 从 (0,6) 起：中间列起块
h=1:  [1][1][1][2][2][2][2][2]      m=8..15    ← 块1 wrap 至 (1,2)；块2 从 (1,3) 起
h=2:  [2][3][3][3][3][3][4][4]      m=16..23   ← 块2 wrap 至 (2,0)；块3 从 (2,1) 起；块4 从 (2,6) 起
h=3:  [4][4][4][4][5][5][5][5]      m=24..31   ← 块5 从 (3,4) 起（核2 的 M 尾块）

      └── 核0：M[0,11) ──┘└── 核1：M[11,22) ──┘└─ 核2：M[22,32) ─┘
```

要点：块边界可以落在行中任意列（跨行块在行切换处 wrap）——这正是 A 侧行推理要按块重算（§6.3）、tiling 预算要用最坏情形跨行数（tiling §3.3）的空间原因。**核间按 CeilDiv 切连续段、段内再切 mBlk 块，块边界不跨核**（§5.2/§5.3）；核边界同样不保证行对齐（上图中落在 (1,3) 与 (2,6)）；`mStart ≥ M` 的空尾核直接跳过（§5.3 判界在前，防无符号下溢）。

**放大一个块看 L0 子块**（块2 = M[11,17)，mL0=3）：

```
  m:      11     12     13 │ 14     15     16
  (h,w): (1,3)  (1,4)  (1,5) │ (1,6)  (1,7)  (2,0)
          └──── 子块0 ─────┘ └───── 子块1 ─────┘
          ↑ mStart=11（mStart%W=3≠0，中间列起块；子块1 内部跨行 wrap）
```

**图 b：单核迭代时间线（M-first；Cout=1 → N 侧两层循环各退化 1 次）**：

```
核0：
for 块0..块1 (mAL1Iter)
  ├─ A 侧按行推理装载本块输入行带（§6.3）
  └─ for L0 子块 (mL0Iter)
       ├─ for kIter = 0..CeilDiv(kAL1,kL0)      ← output 上【不可见】：
       │     LOAD3D → LOAD2D → MMAD                K 在 L0C 深度方向累加，
       │     （每步只推进 L0C 的部分积）            output 元素尚未定型
       └─ 末条 MMAD(unitFlag=3) → FIXPIPE ──► y[mStart..mStart+mL0) 落 GM
                                                   ← output 上【可见】：
                                                      tile 成型的最小事件
```

写出顺序投影到展平 M 轴：`M[0,3) → M[3,6) → M[6,9) → M[9,11) → …` 递增推进（M-first，尾子块按余量收缩）；N-first（Cout>1 时）改为同一 m 段先扫完全部 N 块再前进——**单通道视图下两种迭代序的轨迹相同**，只有多通道 dump 才能区分。

**各切分轴在本视图上的可见性**：

| 切分轴 | 单通道视图上的投影 | 可见性 |
|--------|------------------|--------|
| batch / dOut | 视图外（更外层循环，每片一张独立 H×W 图） | 不在本视图 |
| N（Cout） | 循环退化为 1 次；但 Co^align=N0=16 → L0C/Fixpipe 仍按 16 通道对齐块工作、nSize=1 真值写出——**N 尾块常态存在** | 残留可见 |
| M 多核块 | 展平 M 上的连续段（CeilDiv 切分、尾核收缩、空尾核跳过）；核边界可跨行（wrap） | 可见（各核写出交错落 GM） |
| mAL1 / mL0 | 块内二级细分；每个 (mL0×nL0) tile 一次 FIXPIPE——output 最小成型单位 | 可见 |
| K 归约 | 无投影：深度方向在 L0C 内累加，元素只在末条 MMAD + FIXPIPE 后定型 | 不可见 |
| 乒乓 / L1 / L0 中间态 | 无投影 | 不可见 |

**与失败定位的衔接**（verification §7.2）：差异元素 `(h,w)` → `m = h·W + w` → 所属核（m 所在的块段）→ 块号与子块号 `（m − mStart）/mL0`——即把精度差异的**空间分布**反解为**迭代步坐标**，直接指向对应的装载与指令参数。

### 6.6 各切分轴的尾块与正常块的差异

| 尾块 | 定义 | 差异点 |
|------|------|--------|
| mAL1Tail | singleCoreM % mAL1 | A 侧行推理按尾块重算（hiLoadL1 变小）；mStartPt 修正 |
| mAL0Tail | mAL1Tail % mL0 | LOAD3D 的 repeat/M 参数按尾块收缩；FIXPIPE srcStride 修正 |
| nBL1Tail / nL0Tail | Co 侧两级取余 | LOAD2D 的 N 长度、BT 装载长度、FIXPIPE 通道数按尾块收缩 |
| kL0Tail | K % kL0 | MMAD 的 k 参数取尾值；A 侧通道对齐补零段在此步生效 |
| dOut 尾片（3D） | Do % 单片 | D 向 pad 尾部填零参与 |

**总原则**：尾块不影响"装多少"的容量判断（按正常块预算），但影响"最后一步指令参数"——装载与写出的长度、步进、repeat 全部要按尾块修正，这是越界/漏算事故的高发区。

**嵌套循环的逐层取余原则**：当同一根轴被切分为多级嵌套循环（如 K → L1 分块 × L0 步进两层）时，**内层循环界必须按当前外层块的剩余量计算**，不能复用"整块口径"的外层循环界：

```
外层（L1 分块）: for kC in [0, CeilDiv(kSlice, kChunk)):
    本块余量 = min(kChunk, kSlice − kC·kChunk)
    内层（L0 步进）: for kI in [0, CeilDiv(本块余量, kL0)):    // ✗ 不能用 CeilDiv(kChunk, kL0)
        本步长度 = min(kL0, 本块余量 − kI·kL0)                  // ✗ 不能用 kChunk − kI·kL0
```

违反此原则的后果：末尾块内层"多跑"若干步，剩余量减法发生**无符号下溢**，被钳位后以越界坐标发起指令——设备 fault（流污染），host 侧表现为输出回读失败。这类故障的隐蔽性在于：仅在"L1 预算迫使外层块缩小 **且** 剩余量非整除"的组合下触发，窄输入/小通道数的常见用例不会命中。

**末步标志同步原则**："末条 MMAD"的判定（unitFlag 置 3 的条件）也必须基于逐层取余后的实际最后一步，不能用整块口径的循环界末端——否则尾块跳步场景下末条 MMAD 留在 unitFlag=2，正确性后果取决于硬件对"无翻转结束"的容忍度（当前未定义，见适配层 [../adapters/ascendc/sync-implementation.md](../adapters/ascendc/sync-implementation.md)）。

### 6.7 L1 驻留与逐块抽取（全载态的生命周期）

fullLoad 判定（tiling §3.3）说"某侧驻留 L1"，其代码形态是**装载一次、跨块复用、期间不回收**：

- **驻留侧**（如 AL1 全载态的 fmap）：块迭代前 Alloc→搬运→EnQue→DeQue 一次，**持有该 LocalTensor 跨越全部对侧块**；期间不再 Alloc/Free（每次 Free+Alloc 都会触发一轮装载，驻留失效）；多个 M 块从同一份驻留数据以不同 mStartPt 反复 LOAD3D 抽取（§4.3）；
- **流式侧**（对侧）照常逐块 Alloc→搬运→消费→Free 轮转；
- **同步**：驻留侧只需装载完成一次的 RAW 事件（对首个消费块生效），后续块免同步；流式侧事件照常；
- **代价收益**：驻留侧装载次数从"每 M 块一次"降为 1，A 侧（fmap 主导）或 B 侧（weight 主导）的 GM→L1 流量等比例下降。

### 6.8 迭代序与重装触发（M-first / N-first 与 K 双周期）

**迭代序由全载侧决定**（§6.4 的循环骨架按此交换）：

| 全载态 | 迭代序 | 慢循环（外层，驻留侧） | 快循环（内层，流式侧重装） |
|--------|--------|----------------------|--------------------------|
| AL1 全载（fmap 主导） | **N-first** | mAL1 | nBL1（B 侧每 N 块重装） |
| BL1 全载（weight 主导） | **M-first** | nBL1 | mAL1（A 侧每 M 块重装） |

交换 = 把骨架里两对 M/N 循环（L1 级与 L0 级）各自内外互换；重装触发跟随流式侧的循环变量。

**K 双周期重装（kAL1 ≠ kBL1 时）**：K 归约循环以 kL0 统一步进，两侧各自独立判断重装：

```
for kIter in [0, CeilDiv(kSlice, kL0)):
    if kIter % (kAL1/kL0) == 0:  重装 A 侧 L1 块; posK = 0            # A 侧周期
    if kIter % (kBL1/kL0) == 0:  重装 B 侧 L1 块; KStartPosition = 0   # B 侧周期
    LOAD3D(…, posK);  posK += kL0
    LOAD2D(…, KStartPosition);  KStartPosition += kL0/K0
    MMAD(k = kL0，尾步取尾值)     # MMAD 的 k 只跟 kL0，与两侧周期无关
```

两侧周期互不重叠时重装可交错发起；严格串行写法（先判 A 再判 B 再装载 L0）亦正确。

---

## 7 fmap、weight、bias、output 的搬入、搬出和cube计算必须同步进行

### 7.0 MTE 队列是什么：MTE1 / MTE2 / MTE3 与 FIX

达芬奇架构中，**搬运与计算分属独立硬件队列**，各自异步执行、并行流动。卷积通路涉及四条队列：

| 队列 | 全称 | 负责的搬运 | 卷积中的用途 |
|------|------|-----------|-------------|
| **MTE2** | Memory Transfer Engine 2（入侧 DMA） | **GM → 片上存储**（L1/UB） | fmap/weight/bias 的 GM→L1 装载（DN2NZ/普通搬运）；备选 UB 中转路径的 GM→UB |
| **MTE1** | Memory Transfer Engine 1（片间 DMA） | **L1 → L0A/L0B**（含 BT 装载） | LOAD3D（A 侧滑窗展开）、LOAD2D（B 侧子块抽取）、L1→BT bias 装载 |
| **MTE3** | Memory Transfer Engine 3（出侧 DMA） | **片上存储 → GM**（从 L0C/UB） | FIXPIPE 的 L0C→GM 结果搬出（含 NZ→ND 还原与随路 dtype 转换） |
| **FIX** | Fixpipe 队列 | 与 MTE3 配合的 L0C 消费通道 | FIXPIPE 指令所属队列；与 MMAD 之间靠 unitFlag 传递占用关系 |
| （M） | Cube 计算队列 | L0A/L0B → L0C 乘累加 | MMAD |

要点：

1. **队列间无隐式依赖**——MTE2 装完数据，MTE1 不会自动知道；所有跨队列的"生产者→消费者"关系都必须显式同步（§7.2）；
2. **队列内按序执行**——同一条队列内的指令天然保序，不需要额外同步；
3. **四级流水的本质**：MTE2 / MTE1 / M（MMAD）/ MTE3(FIX) 四条队列用乒乓缓冲错开一级，使装载与计算重叠（§7.1）。

### 7.1 四级流水总体图

计算（MAD）与搬运（MTE2/MTE1/FIX）是**异步硬件队列**，靠乒乓缓冲让它们重叠：

```
时间 ─────────────────────────────────────────────────────►
MTE2:  │ 装A0B0 │ 装A1B1 │ 装A2B2 │ 装A3B3 │ ...
MTE1:          │ A0B0:L1→L0 │ A1B1:L1→L0 │ A2B2 │ ...
MAD :                │ MMAD(0) │ MMAD(1) │ MMAD(2) │ ...
FIX :                      │ OUT(0)→GM │ OUT(1) │ ...
       └────────── 每级错开一步，四条队列并行流动 ──────────┘
```

### 7.2 什么位置插入同步？谁等谁？

| 同步点 | 谁等谁 | 防的问题 |
|--------|--------|---------|
| MTE2 → MTE1 | LOAD3D/LOAD2D 等 GM→L1 装载完成 | 读到未装好的 L1（RAW） |
| MTE1 → MAD | MMAD 等 LOAD3D/LOAD2D 完成 | 用到未展开好的 L0A/L0B（RAW） |
| MAD → FIX | FIXPIPE 等 MMAD 完成 | 搬出未算完的 L0C（RAW） |
| FIX/MAD → MTE2 | L1 重装等当前 L0 计算释放 | 覆盖还在用的 L1（WAR） |
| MTE1（BT 装载）→ MAD | MMAD 等 BT 装载完成 | L0C 初值读到未装好的 BT（RAW） |

实现上用队列式 buffer 管理原语（自动插入依赖事件）或显式事件对原语（Set/Wait）；各语言的原语形态见适配层。事件必须**成对且同序**（Set(X) 与 Wait(X) 的顺序在全路径一致）。

**双缓冲（乒乓）下的事件扩展原则**：

- 每份缓冲一套独立事件对（ping/pong 各自的 RAW 与 WAR），生产者写 pong 前只等 pong 的 WAR 事件——两套事件交错即形成流水；沿用单事件串行写法在双缓冲下会造成假依赖（性能退化）或事件错配（死锁）；
- L1/L0C 级若用队列原语，把缓冲份数设为 2 即由队列自动管理两套事件；L0 级（手写事件对）须显式维护乒乓标志交替选用事件；
- CL0 双缓冲与 unitFlag 配合：末步 MMAD 与 FIXPIPE 的 unitFlag=3（置位+翻转，见 §4.5）驱动 L0C 两份交替占用；
- 双缓冲的事件配对数翻倍，尾块早退路径的 Set/Wait 平衡性须逐路径复核（§7.3 第 4 条在双缓冲下更易违例）。

### 7.3 如何防止同步冲突导致卡死

1. **乒乓是流水的前提**：单缓冲时"装下一块"必须等"当前块用完"，流水退化成串行（空转但不会死锁）；双缓冲才能重叠。**禁止在单缓冲上假设重叠收益**；
2. **事件成对**：每个 Set 必有对应 Wait，且两者顺序在所有路径上一致（含尾块路径、早退路径）；
3. **循环边界先处理再进循环**：首轮装载（prime）在循环外发起，末轮等待不重复；
4. **尾块路径单独走查**：尾块分支里的 Free/Alloc 次数必须与正常块平衡，否则队列泄漏 → 死锁；
5. **多核无栅栏**：正向卷积核间零同步（§5.2），任何核间等待都是设计错误。

### 7.4 检查点

- [ ] 四侧通路（A/B/bias/C）的数据流图与格式标注齐全
- [ ] 每条指令的"前后格式图 + 参数限制"已核对（尤其 LOAD3D 限制表与块定位参数单位）
- [ ] 分核反解公式与尾核规模正确
- [ ] 迭代伪代码覆盖全部循环层，尾块修正点已标注
- [ ] 同步插入点成对、有序，单/双缓冲行为已区分
- [ ] L0C 累加类型按"输入 dtype → 累加类型"基线组合取值

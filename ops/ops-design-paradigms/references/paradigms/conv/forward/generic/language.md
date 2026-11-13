# Convolution 范式：卷积正向算子统一语言

> **层级**：通用知识层（generic/），语言中立。
>
> **目的**：为卷积正向算子的设计、tiling、kernel、验证建立**一套统一术语与记号**。后续所有卷积文档（compute-flow / tiling / verification / 语言适配层）均以本表为准，避免同义异名造成设计歧义。

---

## 0 核心记号速查（先背这张表）

| 记号 | 等价写法 | 值 / 定义 |
|------|---------|----------|
| **K0 = k0 = C0 = c0** | — | **32B / dtypesize**（元素个数）：fp16/bf16→16，fp32→8，int8/fp8→32，fp64→4 |
| **M0 = m0** | — | 16（M 方向分形基粒度） |
| **N0 = n0** | — | 16（N 方向分形基粒度） |
| **C1 = c1** | — | CeilDiv(C, C0)，通道的分形块数（**不是** 32/dtypesize，那是 C0） |
| **db = pingpong = double_buffer** | — | 双缓冲开关 |
| **kernel（张量语境）** | weight / filter | 卷积核张量 |
| **kernel（实现语境）** | 内核 | 设备侧程序，与 tiling（host 侧）相对 |

> ⚠️ **NZ/ND 格式里的 "N"、"D"、"Z" 与 Cube 的 N0/M0/K0 完全无关**——那只是快轴/慢轴记号，详见 [compute-flow.md](compute-flow.md) §3。

---

## 1 张量命名

| 张量 | 别名 | 语义 |
|------|------|------|
| **x** | fmap / feature map / 特征图 | 卷积输入 |
| **filter** | weight / 权重 | 卷积核张量 |
| **bias** | 偏置 | 每输出通道一个标量，可选 |
| **y** | output | 卷积结果 |

> ⚠️ "kernel" 一词有歧义（卷积核 vs 内核），本范式约定：谈**尺寸**用"卷积核尺寸"或记号 Kd/Kh/Kw；谈**设备侧程序**写"kernel（内核）"，一般与 tiling 对举（tiling 部分 vs kernel 部分）。

## 2 维度命名

### 2.1 基本维度

| 记号 | 含义 | 备注 |
|------|------|------|
| **N** | batch 维 | |
| **Ci / Co** | 输入 / 输出通道数 | |
| **D / H / W** | 空间维：深 / 高 / 宽 | 1D 只有 W；2D 有 H、W；3D 全有 |
| **Di / Hi / Wi** | 输入空间维大小 | |
| **Do / Ho / Wo** | 输出空间维大小 | |
| **Kd / Kh / Kw** | 卷积核空间尺寸 | |
| **dkH / dkW / dkD** | 膨胀后卷积核尺寸 | `dkH = (Kh-1)·dH + 1` |

### 2.2 空间维统一表达（1D/2D/3D）

```
Ho = floor((Hi + padTop + padBottom - dkH) / sH) + 1     # 每个空间维独立套用
```

| 场景 | 空间维 | 输入格式 | 等价高维视角 |
|------|--------|---------|-------------|
| Conv1D | {W} | NCW | H=1 的 2D：(N,Ci,1,Wi)，filter (Co,Ci,1,Kw) |
| Conv2D | {H,W} | NCHW | D=1 的 3D |
| Conv3D | {D,H,W} | NCDHW | 原生 |

> 1D 按"升维 H=1 的 2D"处理（aclnn 层即如此实现），tiling/kernel 复用 conv2d。

## 3 卷积属性词汇

| 属性 | 记号 | 约定 |
|------|------|------|
| 步长 stride | `sD/sH/sW` | 每空间维独立；属性列表长度跟随算子原型（NCHW 下 4 元组取后两位） |
| 填充 pad | `padHead/padTail`（D）、`padTop/padBottom`（H）、`padLeft/padRight`（W） | **头/上/左在前，尾/下/右在后**；2D 四元组 `[top, bottom, left, right]`，3D 六元组 `[head, tail, top, bottom, left, right]` |
| 空洞率 dilation | `dD/dH/dW` | 卷积核采样点间隔 |
| 分组数 groups | `g` | 当前范围 `g=1` |
| pad_mode | — | SPECIFIC / VALID / SAME / SAME_UPPER / SAME_LOWER；非 SPECIFIC 时 pad 由模式推导：`pad = (ceil(in/stride)−1)·stride + dilation·(k−1) − in + 1`（负值截 0，SAME 余数给尾/头）——与 verification.md §4.3 同一口径 |

## 4 M/N/K 三元组（GEMM 映射）

卷积经滑窗展开（im2col）映射为矩阵乘 `Y[M,N] = A[M,K] × B[K,N]`：

```
  fmap (N,Ci,Hi,Wi)                weight (Co,Ci,Kh,Kw)              output (N,Co,Ho,Wo)
       │                                  │                                │
       │ LOAD3D 滑窗展开                 │ DN2NZ 布局转换                  │
       ▼                                  ▼                                ▼
  A 矩阵 [M, K]              ×     B 矩阵 [K, N]              =     Y 矩阵 [M, N]
  行 = 输出位置 m                  列 = 输出通道 co                  行×列 = 输出元素
  列 = (ci,kh,kw) 展平              行 = (ci,kh,kw) 展平
```

| 记号 | 1D | 2D | 3D | 说明 |
|------|----|----|----|------|
| **M** | Wo | Ho×Wo | Ho×Wo | 输出空间维展平；**3D 的 Do 不并入 M**（独立外层循环） |
| **N** | Co | Co | Co | 输出通道 |
| **K** | Ci^align×Kw | Ci^align×Kh×Kw | Ci^align×Kd×Kh×Kw | `Ci^align = CeilDiv(Ci,K0)·K0` |

派生对齐量：

| 记号 | 定义 | 含义 |
|------|------|------|
| M1 | CeilDiv(M, M0) | M 方向分形块数 |
| N1 | CeilDiv(Co, N0) | N 方向分形块数 |
| Ci1 | CeilDiv(Ci, K0) | 通道方向分形块数 |

> M0/N0/K0 随 dtype 变化（§0 速查表），通用层算法一律用符号，真值在语言适配层 buffer-and-types（平台能力查询位置节）。

## 5 坐标映射（输出 → 输入感受野）

```
输出元素 y(n, co, h, w)                      输入感受野（Kh×Kw 窗口）
    │                                              │
    │  对每个 (ci, kh, kw)：                       ▼
    │                          ┌── kh=0 ──┬── kh=1 ──┬── ... ──┐
    │         x(n,ci,hi(kh),wi(kw))        │          │          │
    │         × filter(co,ci,kh,kw)        │  采样点   │  采样点   │
    │         累加到 y                       │  (hi,w0)  │  (hi,w1)  │
    ▼                                       └──────────┴──────────┘
                                              采样点间隔 = dilation
                                              窗口间步进 = stride
```

| 项 | 公式 | 说明 |
|----|------|------|
| 输出计算 | `y(n,co,h,w) = bias(co) + Σ_ci Σ_kh Σ_kw x(n,ci,hi(kh),wi(kw)) · filter(co,ci,kh,kw)` | 互相关语义 |
| H 向采样 | `hi(kh) = h·sH − padTop + kh·dH` | 越界（pad 区）时 x 取 0 |
| W 向采样 | `wi(kw) = w·sW − padLeft + kw·dW` | 同上 |

这条映射是 im2col、pad 处理、行推理、golden 生成的共同数学依据。

## 6 缓存层级词汇

| 层级 | 卷积语境职责 | 典型容量（Ascend 950，dav-3510） |
|------|-------------|----------------------|
| **GM** | x/filter/bias/y，ND 布局 | — |
| **L1** | A 侧驻留原始 2D 图、B 侧驻留 NZ 分形块 | 512 KB |
| **L0A / L0B** | A/B 分形缓冲（MMAD 操作数） | 64 KB / 64 KB |
| **L0C** | MMAD 累加结果（fp32），FIXPIPE 直出 | 256 KB |
| **BT** | bias 专用缓冲 | 4 KB |
| **UB** | Vector 侧缓冲（卷积主通路不经过） | 248 KB |

## 7 格式词汇

详细排布图与示例见 [compute-flow.md](compute-flow.md) §3，此处只列术语：

| 格式 | 一句话定义 |
|------|-----------|
| ND | channel 在内（NCHW 即 ND） |
| DN | channel 在外（NHWC 即 DN） |
| NZ | 最内轴按 C0 分块（Z=C0） |
| FRACTAL_Z | 权重标准分形排布：[K 侧展开, N1, N0, C0]，C1 最外轴 |
| FRACTAL_Z_3D | FRACTAL_Z + D 轴（在 N 外层） |
| NC1HWC0（5HD） | [N, C1, H, W, C0] |

## 8 切分记号体系

### 8.0 记号层次图

```
                     ┌── L1 级块（大块） ──┐    ┌── L0 级子块 ──┐
                     │  mAL1  nBL1  kAL1  kBL1 │  mL0  nL0  kL0 │
切分方向 ──┬── M ────┤  (fmap 侧 M 方向)      │  (MMAD 块内)   │
           ├── N ────┤  nBL1                  │  nL0           │
           │         │  (weight 侧 N 方向)     │                │
           └── K ────┤  kAL1  kBL1            │  kL0           │
                     │  (reduce 轴)           │  (K 归约步进)   │
                     └───────────────────────┘  └──────────────┘

HW 模式空间维：hoL1  woL1  hoL0  woL0（不合轴，H/W 独立切）
```

### 8.1 命名总规则

```
<方向><粒度><层级>：mAL1 / nBL1 / kAL1 / kBL1（L1 级块）
                    mL0 / nL0 / kL0（L0 级子块）
                    hoL1 / woL1 / hoL0 / woL0（HW 模式空间维）
```

- **A** = fmap 侧，**B** = weight 侧：`mAL1` 读作"A 侧（fmap）的 M 方向 L1 块"；
- ⚠️ **kAL1/kBL1 是乘过 Kh·Kw 的展开 K 口径**；A 侧 L1 实际驻留字节数要除回 Kh·Kw（见 [tiling.md](tiling.md) §3.3）。

### 8.2 单核规模与多核维度

| 记号 | 含义 |
|------|------|
| singleCoreM / singleCoreCo / singleCoreCi / singleCoreBatch / singleCoreHo / singleCoreWo / singleCoreDo | 单核负责的规模（多核切分后） |
| batchDim / nDim / mDim（复用 hoDim）/ hoDim / woDim / doDim / groupDim | 各维绑核数 |
| **blockDim** | 总核数 = 各 Dim 之积 |
| mTile / nTile / mBlkNum / nBlkNum / fDim | 基本块模式专用：块型 (M×N)、块网格数、沿 batch/M 块方向的绑核数（tiling §1.3/§2.4） |

### 8.3 迭代变量与循环计数

| 记号 | 含义 |
|------|------|
| *Iter（mAL1Iter / nBL1Iter / mL0Iter / nL0Iter / kIter / batchIter / dOutIter） | kernel 侧迭代游标 |
| ddr2l1LoopM = CeilDiv(singleCoreM, mAL1) | GM→L1 的 M 大块循环次数 |
| ddr2l1LoopN = CeilDiv(singleCoreCo, nBL1) | GM→L1 的 N 大块循环次数 |
| l12l0LoopM = CeilDiv(mAL1, mL0) | L1→L0 的 M 子块循环次数 |
| l12l0LoopN = CeilDiv(nBL1, nL0) | L1→L0 的 N 子块循环次数 |
| ddr2l0LoopK = CeilDiv(kAL1, kL0) | K 归约循环次数 |

### 8.4 尾块

`*Tail` 后缀：`Tail = 总量 % 块大小`（余 0 取块大小）。

```
mAL1Tail / mAL0Tail / nBL1Tail / nL0Tail / kL0Tail
```

### 8.5 全载与乒乓

| 记号 | 含义 |
|------|------|
| FULLLOAD_AL1 / BL1 | fmap / weight 整体驻留 L1 |
| kAL1fullload / kBL1fullload | 仅 K 方向全载 |
| pBufferFlag | 乒乓位域：bit0=BL1, bit1=AL1, bit2=CL0, bit3=BL0, bit4=AL0 |
| **db / pingpong / double_buffer** | 同一个东西：双缓冲（开启 = 搬运与计算重叠） |

## 9 数学记号

| 记号 | 定义 | 示例 |
|------|------|------|
| `CeilDiv(a, b)` | 向上取整除 | CeilDiv(35,16)=3 |
| `AlignB(a, b)` | 向上对齐到 b 的倍数（= CeilAlign / RoundUp） | AlignB(35,16)=48 |
| `FloorAlign(a, b)` | 向下对齐 | FloorAlign(35,16)=32 |
| `Gcd / Lcm` | 最大公约 / 最小公倍 | 绑核候选因子生成 |

## 10 dtype 与精度词汇

| 术语 | 含义 |
|------|------|
| 存储 dtype | GM 中的类型 |
| 计算 dtype | L0C 累加类型：fp16 输入按 fp32 累加 |
| A16W16 / A16W8 | A 侧×B 侧位宽组合描述（扩展场景） |

## 11 与外部知识的术语对接

| 外部知识 | 路径 | 对接说明 |
|---------|------|---------|
| 通用 tiling 方法论 | `cannbot-skills/ops/ascendc-tiling-design/references/` | 其多核/UB 切分词汇与本范式 §8 对齐；其 Convolution 类以本范式为准 |
| 精度标准 | `cannbot-skills/ops/ops-precision-standard/references/float_compute.md` | 混合容差（atol/rtol/matched_ratio）的系统性定义 |

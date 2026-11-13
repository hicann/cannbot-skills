# Convolution 范式：卷积正向算子场景扩展

> **层级**：通用知识层（generic/），通用卷积的**补充知识**。
>
> **定位**：主干文档（[compute-flow.md](compute-flow.md)、[tiling.md](tiling.md)）描述的是**标准路径**——fp16、NCHW/NCDHW、groups=1、pad<kernel。本文件逐项描述每个扩展维度对主干的**改造点**：只写"与主干差在哪、为什么、怎么实现"，不重复主干内容。主干覆盖的场景不需要读本文件。

| 扩展维度 | 对主干的改造 | 章节 |
|---------|-------------|------|
| 空间维 1D/3D | 3D 增加 D 轴机制（软件 pad、D 片打包） | §1 |
| dtype 量化（int8/fp8 + scale） | pad 填充值、FIXPIPE/UB 反量化、定点移位 | §2 |
| format NHWC | 搬运指令与 FIXPIPE 主序切换、带宽差异 | §3 |
| groups>1 | weight 对角阵化打包 / 独立分组 | §4 |
| pad ≥ kernel | 全零窗口、小核路径禁用、超限回退 DMA | §5 |
| 小通道 Ci≤4 | C04 紧排模式，省 4 倍 L1/L0 资源 | §6 |

---

## 1 1D / 2D / 3D 卷积的区别

| 项 | 1D | 2D | 3D |
|----|----|----|----|
| 空间维 | W | H、W | D、H、W |
| format | NCW | NCHW | NCDHW |
| K 组成 | Ci^align×Kw | Ci^align×Kh×Kw | Ci^align×Kd×Kh×Kw |
| 实现方式 | **视作 H=1 的 2D，完全复用 2D 的切分与 kernel** | 基准形态 | 在 2D 骨架上扩 D 维（执行框架共用，见下） |

### 1.1 1D：H=1 的 2D

- 2D 卷积 NCHW 中 H=1，即退化为 1D（NCW）。**升维处理**：输入 NCW→NCHW、卷积核 (Co,Ci,L)→(Co,Ci,1,L)、stride/dilation 补 1、padding 仅 W 方向，tiling/kernel 完全复用 2D 方案；
- 存在 **H 维退化快路径**：`Hi==Kh==sH==dH==1 且无上下 pad` 时 H 维完全退化，滑窗只沿 W 方向进行，可省去 H 向的窗口几何配置；
- 多核切分上，HW 模式的 **woDim 只在 1D 场景使用**（此时 ho≡1，只有 W 能提供空间并行度）。

### 1.2 3D：单独实现 + 部分共用

理论上 3D 也可以收编 2D（把 D 并进 M 或 K），但工程上通常**以 2D 骨架为基础单独扩展 D 维机制**：迭代调度、指令封装、迭代推进等执行框架与 2D 共用，差异集中在：

| 差异点 | 3D 的做法 |
|--------|----------|
| D 维迭代 | 输出 D 不并入 M：**每次迭代只处理 1 层输出 D**（循环次数 = 单核 Do），M 恒为 Ho×Wo |
| D 向 K 展开 | fmap 侧每次 AL1 装载打包 **kd 或多倍 kd** 个 D 片（打包数记 aL1Dk），权重侧同理（§1.3） |
| D 向 pad | **软件显式填零**（2D 的 H/W pad 是硬件虚拟化，见 [compute-flow.md](compute-flow.md) §6.3） |
| 多核切分 | 增加 doDim（绑核面向 output 的 dout 轴） |

### 1.3 3D 的 D 轴机制（tiling/kernel 如何处理）

**装载打包**（cinA 为单次装载的通道跨度）：

```
aL1Dk  = cinA ≤ AlignB(Ci,k0) ? 1 : cinA / AlignB(Ci,k0)    # 每次 AL1 装载打包的 kernel-D 片数
aL1Cin = cinA / aL1Dk                                          # 每次装载覆盖的 CI 通道数
kAL1 一趟的 K 跨度 = aL1Cin × aL1Dk × Kh × Kw（展开口径）
```

**K 迭代序号解码为 (dk, ci)**（aL1CinLoadNum = 单核通道数 / aL1Cin）：

```
cinIdx  = (kAL1Iter % aL1CinLoadNum) × aL1Cin     # ci 为内层
kdL1Idx = (kAL1Iter / aL1CinLoadNum) × aL1Dk      # dk 为外层
diStartWithPad = diStartPos + dOutIter·sD + kdL1Idx·dD   # 本趟 D 窗起点（含 pad）
```

**D 向 pad 软件填充**：头部整块落在 pad 区的平面数 `kdTmp = CeilDiv(padHead - diStartWithPad, dD)`，这些平面用 `常量填充` 显式填 **0（非量化）/ offset_x（量化）**，有效 D 片数 `dkLoadL1` 相应收缩。Load3D 读到的 AL1 是"[D(含软件 pad), C0, H×W] 的连续虚拟体"——**D 向 pad 占 L1 空间**，L1 预算必须计入（与 2D 的 H/W pad 虚拟化不同）。

---

## 2 浮点卷积和量化卷积的区别

### 2.1 通路对比

```
浮点：  GM(fp16 x/w) → L1 → L0A/L0B ──MMAD(浮点, L0C=fp32累加)──► L0C ──FIXPIPE(dtype转换)──► GM(fp16 y)
量化：  GM(int8 x/w) → …同主干… ──MMAD(定点, L0C=int32累加)──► L0C ──×scale + bias 还原──► GM(fp16 y)
                                                ↑ 差异全部在 L0C 之后的"还原"环节和 pad 语义
```

| 差异点 | 浮点卷积 | 量化卷积（int8/fp8 输入 + scale 类） |
|--------|---------|--------------------------------------------------|
| L0C 累加类型 | fp32 | int32（int8×int8 定点）或 fp32（hifloat8/fp8 输入） |
| scale 输入 | 无 | 有（per-channel 向量或 scalar，int64/uint64 编码的 fp32） |
| 还原环节 | FIXPIPE 仅做 dtype 转换 | FIXPIPE 随路反量化（主流）或 Vector 侧 UB 反量化（特例） |
| pad 填充值 | 0 | **offset_x**（量化输入的有效零点） |
| bias dtype | fp16/fp32 | fp32（int8 场景 golden/input 插件强制修正） |

### 2.2 还原环节的三条路径

**① FIXPIPE 随路反量化（主路径）**：scale 先 GM→L1（按当前 nBL1/nL0 通道段，`通道维 L1 装载`），FIXPIPE 带 `scaleL1` 与 `quantPre` 参数，在 L0C→GM 的同时完成 ×scale+bias。quantPre 按输出 dtype 选：

| L0C 类型 | 输出 dtype | quantPre |
|----------|-----------|----------|
| int32 | fp16 | VDEQF16（向量）/ DEQF16（标量/定点） |
| int32 | int8 | VREQ8 / REQ8 |
| fp32（hifloat8/fp8 输入） | fp16/bf16/int8 | F322F16 / VQF322B8_PRE 等 |

**② Vector 侧 UB 反量化（conv3d 特例）**：`int8 输入 + 浮点 bias`（Conv3d→Dequant 融合来源）时启用（`DeQuant 标志`）：L0C ──FIXPIPE(NoQuant)──► UB ──Cast(fp32)→Mul(scale)→Add(bias)──► GM。走 UB 是因为该场景需要 int32→float 的 cast + 逐通道 mul + 浮点 bias add，FIXPIPE 硬件单元做不了组合。

**③ 后处理融合变体**：scale、relu、clip 等后处理融合进 FIXPIPE 随路完成，可支持双输出（一次 MMAD 结果两路量化分别写出）。

### 2.3 offset_x 与定点移位

- **offset_x**：量化算法中的偏移，**pad 区的填充值**（int8 输入表示的真实值是 x+offset_x，pad 必须填 offset_x 才语义等价于 pad 0）。Load3D 硬件 pad 值、allPad 整块填充、3D 的 D 向软件填充全部用它；hifloat8/fp8 输入时仅支持 0。
- **A16W8 定点移位**：fp16 fmap × int8 weight 的乘加在 L0C 以定点 int32 累积——MMAD 先**左移** `fixedShiftValue` 位保精度，FIXPIPE 输出时**右移** `总位宽 − fixedShiftValue` 位还原（A16W8 总位宽 29、A16W16 为 58）。

---

## 3 NCHW 和 NHWC 格式在 kernel 实现上的区别

### 3.1 三处指令差异

| 环节 | NCHW | NHWC |
|------|------|------|
| fmap GM→L1 | **DN2NZ**：以"通道平面 H×W"为搬运单元，每通道一次大块连续拷贝（`dnNum=1, nValue=hiLoadL1·wiLoadL1, srcDValue=H·W`） | **ND2NZ**：逐 H 行搬运，行内 C 维连续（`ndNum=hiLoadL1, srcDValue=Ci`） |
| weight GM→L1 | NCHW：DN2NZ，innermost 段仅 kh·kw 元素、跨 cout 平面跳转（**小 burst，带宽最低 32B/cycle**） | 配 HWCN：ND2NZ 以输出通道为内层（**128B/cycle，相差 4 倍**）；FRACTAL_Z 则直搬（最优） |
| FIXPIPE 输出 | **CFG_COLUMN_MAJOR**：每输出通道是连续 Ho×Wo 平面（`dstStride=Ho·Wo`） | **CFG_ROW_MAJOR**：每行 Wo×Co 连续、C 在最内（`dstStride=Co`） |

### 3.2 带宽代价（tiling 计入）

多核切分代价模型对 weight 搬运量乘带宽系数（`权重带宽系数查询`）：

| format 组合 | 带宽系数 |
|------------|---------|
| NCHW fmap + NCHW weight（非 NZ） | **4×**（DN2NZ 小 burst） |
| NCHW + FRACTAL_Z / NHWC + HWCN | 1× |
| C04 + NHWC（weight 需 vec 转 C04 分形） | 10× |

### 3.3 指令约束差异

- NCHW 输出：`Ho×Wo ≤ 2^32`（FIXPIPE dstStride 32 位）；
- NHWC 输出：`Wo×Co ≤ 2^32`。

> 结论：**NHWC 对搬运更友好**（带宽 4 倍差），NCHW 是框架原生格式。选型时 tiling 代价模型已把该差异折算进去。

---

## 4 groups=1 和 groups>1 的区别

### 4.1 三种模式

| 模式 | 条件 | 思路 |
|------|------|------|
| NORMAL（主干） | groups=1 | 普通卷积 |
| **ORI_GROUP 原始分组** | groups>1，OPT 不可用（weight 装不下 UB / DMA 回退） | 每个 group 独立算一个小卷积 |
| **OPT_GROUP 优化分组** | groups>1 且打包 weight 能装进 UB | weight **对角阵化**，enlarge 个 group 打包成一个放大卷积 |
| OPT_SIMPLIFIED | OPT_GROUP 且 ci==groups（depthwise）等 | depthwise 简化 kernel |

### 4.2 ORI_GROUP：原始group卷积
**原理**：groups=g 时通道按组划分——`ciPerGroup = Ci/g`、`coPerGroup = Co/g`，第 i 组的输出只依赖第 i 组的输入与权重：

```
y_i(n, co, h, w) = bias_i(co) + Σ_cii Σ_kh Σ_kw x_i(n, cii, hi(kh), wi(kw)) · W_i(co, cii, kh, kw)
x_i = x 的第 i 组输入通道段 [i·ciPerGroup, (i+1)·ciPerGroup)
W_i = filter 的第 i 组输出通道行 × 第 i 组输入通道列
```

每个 group 就是一个 **M/N/K 缩小的普通卷积**：M = Ho×Wo 不变（空间维与分组无关）、N = coPerGroup、K = CeilDiv(ciPerGroup, K0)·K0 × Kh×Kw；组间读不同的输入通道段、写不同的输出通道段，**零依赖、天然可并行**。

与 OPT_GROUP 的取舍：ORI_GROUP 不做 weight 预处理（无对角阵化、无 Vector 核参与），实现最简单、任何规格可用；代价是 per-group 的 N/K 小，凑不满 N0/K0 分形时 cube 利用率低——这正是 OPT_GROUP 的收益场景。选用条件即 §4.1：OPT 不可用（打包 weight 装不下 UB / DMA 回退路径）。

**tiling 侧怎么实现**（四项要点）：

| # | 要点 | 做法 |
|---|------|------|
| 1 | M/N/K 按组重算 | M 不变；N = coPerGroup、K = CeilDiv(ciPerGroup,K0)·K0 × Kh×Kw；容量不等式、fullLoad 四态全部按 per-group 口径套用 [tiling.md](tiling.md) §3 的算法（各组规格相同，决策一次、参数复用） |
| 2 | 绑核 | groupDim 进入 `blockDim = batchDim × nDim × mDim × doDim × groupDim`（tiling §2.1）；每核 `singleGroups = CeilDiv(groups, groupDim)` 个完整 group |
| 3 | 代价模型 | 沿用 tiling §2.3 四笔账，各项乘 `CeilDiv(groups, groupDim)`；weight 装载按 per-group 段计 |
| 4 | 尾块 | `groups % groupDim ≠ 0` 时尾核 group 数少；ciPerGroup/coPerGroup 非 K0/N0 整数倍的通道尾块照常处理（K 尾补零、N 尾收缩） |

**kernel 侧怎么实现**：循环骨架与普通卷积完全一致，仅在 N 迭代外增加一层 group 循环（或将 group 并入 blockIdx 反解，分解顺序 `batch → group → n → m`，§4.4）：

```cpp
for (groupIter = 0; groupIter < singleGroups; ++groupIter) {
    gi = groupStart + groupIter;                          // 本核起始组 + 迭代号
    xOff    = gi * ciPerGroup * Di*Hi*Wi;                 // NCHW 下 group 维与 C 维连续
    wOff    = gi * coPerGroup * ciPerGroup * Kd*Kh*Kw;
    yOff    = gi * coPerGroup * Do*Ho*Wo;
    biasOff = gi * coPerGroup;
    ...  // 普通卷积的 M/N/K 迭代（compute-flow §6.4 骨架）：
         // A/B/bias/output 的 GM 基址加上述组偏移，通道上下界换成组内范围
}
```

寻址要点：NCHW 下 fmap 的 group 段是**连续通道**（`[N, G, ciPerGroup, H, W]`），A 侧装载只需 GM 基址加组偏移，DN2NZ 参数形态不变（dValue 用组内通道真值，尾组依赖 C0 补零）；weight/bias/output 同理按组偏移。与 OPT_GROUP 的实现差异：无 UB 对角化、无 AIC↔AIV 旗标同步；代价是逐组装载，小通道场景 weight 装载次数多。

### 4.3 OPT_GROUP：weight 对角阵化

**原理**：把 `enlarge` 个 group 的卷积打包成一个放大的普通卷积（K = cinOpt×kh×kw，N = coutOpt）：

```
普通分组（2 个 group）                     对角阵化打包（enlarge=2）
y0 = x0 × W00                             y = [x0 x1] × ┌W00  0 ┐
y1 = x1 × W11                                          └ 0  W11┘
                                          非对角块置 0 ⇒ 一次 MMAD 算完 2 个 group
```

**为什么可行**：NCHW fmap 布局 `[N, G, ciPerGroup, H, W]` 中 group 维与 C 维连续——第 g 个大组的 fmap 就是**天然的连续 cinOpt 个通道**，不用动；只有 weight 需要构造成 K×N 分块对角阵。

**参数计算**：

```
enlarge = min( Lcm(ciPerGroup,k0)/ciPerGroup , Lcm(coPerGroup,n0)/coPerGroup , groups )
          ↑ 取 Lcm 是为了让打包后的通道数能对齐到 k0/n0 分形
groupOpt = CeilDiv(groups, enlarge)     # 打包后的大组个数
cinOpt   = ciPerGroup × enlarge          # 打包后单大组输入通道
coutOpt  = coPerGroup × enlarge          # 打包后单大组输出通道
```

**kernel 侧流程**（ND 布局权重需经 Vector 核在 UB 内完成对角阵化再入 BL1）：

```
GM weight ──NDDMA──► UB：先整块清 0，再把每个 group 的 weight 放到对角块位置
            （dstStride 跳过 coPerGroup 行 + ciPerGroup 列 → 下一个对角块）
UB ──ND→NZ 转换──► UB(NZ) ──► BL1；AIC 侧按旗标同步等待
```

**收益**：小通道 per-group（如 ciPerGroup=4、coPerGroup=8）时，独立分组的 M/N/K 都凑不满分形，Cube 利用率极低；对角阵化后 M/N/K 放大 enlarge 倍，一次 Cube pass 捞回利用率。

**尾块**：groups 不被 enlarge 整除时 `enlargeTail = groups % enlarge`，最后一个 groupOpt 上收缩 singleGroups/singleCoOpt/singleCoreCi。

### 4.4 groupDim 在绑核中的参与

- blockDim = `batchDim × mDim(或 hoDim×woDim) × nDim × doDim × groupDim`；
- **group 可视为 n（cout）的切分轴**：blockIdx 分解顺序 NCHW 下为 `batch → group → n → ho → wo`；OPT_GROUP 时 groupDim 切的是 groupOpt（大组数）、nDim 切 coutOpt；ORI_GROUP 时切 groups、coPerGroup；
- 代价模型各项乘 `CeilDiv(curGroups, groupDim)`；OPT_GROUP 下 weight 常驻 UB，权重装载代价 不再乘 N 切分。

---

## 5 pad ≥ kernel 时 kernel 侧怎么处理


**语义后果**：stride 较大或 pad ≥ kernel 时，部分输出位置的**感受野完全落在 pad 区**（全零窗口）→ 这些输出只等于 bias。

**kernel 侧处理（逐 L1 tile 判定，非全局）**：

```
判定：padTopL1 ≥ paddingHiAL1（该 tile 的输入窗跨度）或 padBottom/padLeft/padRight 任一超限
      → allPadFlag = true
处理：跳过 GM 搬运，整块 AL1 用 常量填充 显式填充：
      - 非量化：填 0
      - 量化：填 offset_x（两个 int8 打包成 uint16）
      同时 FMatrix 设为 1×1 + pad=255，让 Load3D 全窗口读 pad 值
```

**tiling 侧影响**：

| 影响 | 内容 |
|------|------|
| 小核路径禁用 | 任一维 pad > kernel 即排除小核优化路径，回退普通流水线 |
| 指令上限 | LOAD3D_MAX_PAD=255；allPad 时虚拟尺寸 = 255+1+255 = **511**，该方向 dilatedKernel > 511 则 Load3D 溢出 |
| 回退软件搬运 | 滑窗指令限制（stride>63 / dilation>255 / pad>255 / 膨胀核>511 / kh·kw·k0>65535）任一超限时，A 侧改走普通搬运 + 软件滑窗索引，性能显著下降但功能正确 |

判定链（按序）：滑窗指令路径（默认）→ H 维退化路径（Hi=Kh=sH=dH=1 且无上下 pad）→ 软件搬运路径（滑窗指令限制不过）。

---

## 6 C04 优化是什么

### 6.1 是什么、为什么

C04 = **小通道紧排模式**，面向 `Ci ≤ 4` 的场景（典型：RGB 3 通道的首层卷积）。

普通模式下 Ci=4 的浪费：fmap 通道要对齐到 K0=16（fp16），AL1/L0A 中每 16 个通道元素里 12 个是补零；K = `CeilDiv(4,16)×k0×kh·kw = 16·kh·kw`，实际有效仅 `4·kh·kw`——**AL1 面积、GM→L1 带宽、L0B、MMAD K 循环全部 4 倍浪费**。

C04 模式：C 维只按 **4** 紧排，`K = AlignB(4·kh·kw, k0)`，资源占用约降为 1/4。

### 6.2 触发条件

全部满足才启用：`Ci ≤ 4` 且 `groups==1` 且 weight 为浮点 dtype（或已是 FRACTAL_Z_C04 格式）且非 DMA/非 extend，且 UB 能装下至少一个 N0 块。

### 6.3 落地要点

| 环节 | C04 的特殊处理 |
|------|---------------|
| tiling | `kAL1 = kBL1 = C04 对齐 K = AlignB(4·kh·kw, k0)`，**K 方向强制全载**；`singleCi1 = 1` |
| fmap GM→L1 | C 维只搬真实 ci（≤4）个通道、不 pad 到 16，沿 H×W 方向做 32B 对齐（`dstNzMatrixStride = nValue×4`） |
| weight | ND 格式（NCHW/HWCN）需 **Vector 核在 UB 转 FRACTAL_Z_C04**：GM→UB（不足 4 通道由 DMA 补 0）→ K 尾清零对齐 → C04 分形转换 组分形 → UB→L1；已是 FRACTAL_Z_C04 则直搬 |
| Load3D | `channelSize = 4`（每步只读 4 通道）；K 尾块用 `C04 尾步 K = (4·kh·kw) % kL0` |
| 代价 | weight 的 UB 转换有开销，绑核代价模型乘带宽系数 **C04 带宽系数 = 10**——所以只有 fmapC≤4 时才划算 |

### 6.4 普通模式 vs C04 对比（Ci=4, fp16）

| 项 | 普通模式 | C04 模式 |
|----|---------|---------|
| fmap C 维占用 | 对齐到 16（4/16 有效，**75% 浪费**） | 紧排 4 |
| K | 16·kh·kw | AlignB(4·kh·kw, 16) |
| MMAD K 循环 | 4 倍冗余 | 有效 |
| weight 格式 | NCHW/FRACTAL_Z | 需 FRACTAL_Z_C04（ND 要 vec 转换，10× 带宽系数） |

---

## 7 检查点

- [ ] 每个扩展章节只描述"与主干差异"，主干行为未被重复或改写
- [ ] 3D 的 D 向 pad"软件填充占 L1"与 2D H/W pad"虚拟化不占 L1"的口径区分清楚
- [ ] 量化场景的 pad 填充值（offset_x）与 golden 的 pad 处理对齐
- [ ] groups>1 的 enlarge 公式与对角阵等价性可手工验证（小规模 2-group 例子）
- [ ] ORI_GROUP 的组寻址偏移（x/w/y/bias 四侧）与 per-group 三元组重算可手工验证（小规模 2-group 例子）
- [ ] pad≥kernel 的 allPad 判定是逐 tile 的，不是全局开关
- [ ] C04 的收益/代价（4 倍资源 vs 10 倍 weight 带宽系数）都已呈现

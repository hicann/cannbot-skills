# 基本块与 Roofline 分析

> 承接 S3 的公式化 buffer 清单。本文推导的 `mBaseSize` / `s2BaseSize` / 性能区制，是 S5 流水、S6 负载均衡的输入，需代入 S3 buffer 清单以硬件内存为约束迭代收敛。

## §1 为什么 FA 需要专门的 Roofline

FA 是混合算子：矩阵乘（GEMM）+ softmax（归约 / 指数），跨核 stage 化。性能瓶颈**不能用单一算术强度（AI）概括**，三个独立吞吐通道各自可能成为瓶颈：

1. 计算算力（Cube 矩阵乘单元吞吐）
2. 内存带宽（K / V / Q / O 的加载与写回）
3. 跨核 / 片上吞吐（Vector softmax 吞吐 + Cube↔Vector 片上握手）

> 常见建模错误：把片上握手与 HBM 加载混进同一个 AI 分母，再用"混合带宽"求单一 AI——会把 AI 人为封顶在 `D/sizeof(T)`，并掩盖真正的第三瓶颈。
>
> 正确做法：**三通道各自独立建模，性能 = 三个 ceiling 取 min**；瓶颈 = 限制项。

## §2 三通道 Roofline

### §2.1 mBaseSize：GQA 多头合轴后的 M 维分块

同一 kvHead 下的 G 个 qHead 与 Sq **合轴**成一条 M 轴（`M = Sq × G`，I2），矩阵乘的有效 m 维就是 M 轴的分块大小：

```
mBaseSize = 合轴 M 轴上的分块行数
```

- **M 轴 = Sq×G（GQA 合轴）**，mBaseSize 的取值原则：UB 容量允许的前提下尽量取大，目标是把 `AI_HBM ∝ mBaseSize` 推到 ridge_HBM 以上——此时 Cube 是瓶颈、HBM 带宽被喂满，即"以 Cube-bound 为目标"。**基本块套数由 spec 驱动**——spec 未给场景典型值时归一为单一常量（prefill/decode 共享）；spec 给了明确典型值时可按典型值分多套常量。
- **M 轴小（decode 短行）**：mBaseSize 有效值受 M 轴锁死、低于块常量属固有 HBM-bound，基本块改不动 AI，优化下沉到 S6 split-KV（见下）。

**mBaseSize 是核心自变量**，同时决定 AI 和三通道的相对松紧。

### §2.2 单 task 的 FLOPs 与双 GEMM 映射

两个 GEMM 各遍历全 Sk：

```
FLOPs_total = 4 × mBaseSize × Sk × D
```

| GEMM | 计算 | M | K | N | L1 关键输入 |
|---|---|---|---|---|---|
| Bmm1 | Q·K^T | mBaseSize | D | s2BaseSize | Q(D×mBaseSize) + K(D×s2BaseSize) |
| Bmm2 | P·V | mBaseSize | s2BaseSize | D | P(mBaseSize×s2BaseSize) + V(s2BaseSize×D) |

> **causal 的影响不在 AI**：被完全 mask 的 tile 被跳过，FLOPs 与 K/V 加载同比例减少，AI ≈ 不变；causal 真正影响的是**绝对耗时**与**核间负载均衡**，不是算术强度。

### §2.3 HBM 带宽与 AI_HBM（主 Roofline）

单 task 的 HBM 流量（遍历全 Sk）：

| 张量 | 流量 | 说明 |
|---|---|---|
| K 加载 | `Sk × D × sizeof(T)` | 全序列加载一次 |
| V 加载 | `Sk × D × sizeof(T)` | 全序列加载一次 |
| Q 加载 | `mBaseSize × D × sizeof(T)` | 驻留，摊薄 |
| O 写回 | `mBaseSize × D × sizeof(T)` | 末块写出 |
| O_acc streaming（仅 streaming 模式） | `2 × mBaseSize × D × sizeof(acc) × (Sk/s2BaseSize)` | 每个 chunk 读回 + 写出，共 Sk/s2BaseSize 次 |

`Sk ≫ mBaseSize` 时 K/V 主导：

```
Bytes_HBM ≈ 2 × Sk × D × sizeof(T) + 2 × mBaseSize × D × sizeof(acc) × (Sk/s2BaseSize)
AI_HBM = FLOPs_total / Bytes_HBM
```

忽略 O_acc streaming 项的清晰上界：

```
AI_HBM ≈ 2·mBaseSize / sizeof(T)
```

关键结论：
- **AI_HBM ∝ mBaseSize**（不封顶在 D/sizeof(T)）
- **M 轴小（decode 短行，mBaseSize 受 M 锁死）**：`AI_HBM = 2·mBaseSize/sizeof(T)` 被行数压死，GQA 合轴（M = Sq×G）是提升 AI 的唯一杠杆；基本块改不动，由 split-KV 补核数
- **s2BaseSize 通过 O_acc streaming 项影响 AI**：streaming 下增大 s2BaseSize 减少 O_acc 往返；非 streaming（O_acc 驻留）下 s2BaseSize 对 AI 无直接影响

HBM ceiling：

```
Perf_HBM = Peak_BW × AI_HBM
ridge_HBM = Peak_FLOPS / Peak_BW    (FLOPs/Byte)
AI_HBM < ridge_HBM → HBM-bound
```

### §2.4 跨核 / 片上吞吐（cross-core-bound 校核）

若 **Vector 端（softmax）吞吐 < Cube 端（GEMM）吞吐**，即使带宽与算力都不饱和，算子仍被 Vector 拖住。

Vector 端每 tile 工作量：
- softmax over `[mBaseSize, s2BaseSize]`：ReduceMax / Exp / ReduceSum，∝ mBaseSize·s2BaseSize
- 片上握手（不占 HBM，占片上带宽）：各 ∝ mBaseSize·s2BaseSize 或 mBaseSize·D

校核判据（吞吐平衡）：

```
T_cube(tile)  ≈ FLOPs(tile) / (Peak_FLOPS × util)
T_vec(tile)   ≈ softmax_ops(mBaseSize·s2BaseSize) / (vec_throughput × r_cv) + onchip_bytes / onchip_BW
cross-core-bound ⟺ T_vec > T_cube
```

`r_cv` = Vector:Cube 核数配比（1:2 型架构 r_cv=2）：同一 task 的 V1/V2 由搭档的 r_cv 个 Vector 核按 mBaseSize 行维分摊并行，Vector 有效吞吐按 `vec_throughput × r_cv` 折算。**禁止默认 r_cv=1**，否则会把 Vector 端低估一倍、误判 cross-core-bound。

> **算力口径警示（`vec_throughput` 与 `Peak_FLOPS` 的单核化）**：芯片规格里的 Cube / Vector 算力通常是**全芯片合计**。代入上式前必须先单核化（以下数字为**示例**，仅示意换算方法，禁止代入方案）：
>
> ```
> # 示例芯片：Cube 总算力 432T / 32 核，Vector 总算力 54T / 64 核（54T = 基准 27T × 双发 2）
> 单 Cube 算力  = Cube总算力 / CubeCore核数            # 432T/32 = 13.5T
> 单 AIV 算力   = Vector总算力 / VectorCore核数         # 54T/64 ≈ 0.84T（勿按 CubeCore 核数均摊）
> 执行组 vec_throughput × r_cv = 单 AIV × r_cv          # 0.84T×2 ≈ 1.69T
> ```
>
> 两类典型错误：① 把 Vector 总算力按 **CubeCore 核数**均摊（`54T/32`），再乘 r_cv → 单执行组算力被**高估一倍**（3.4T vs 实际 1.69T）；② 只取单 AIV 不乘 r_cv → **低估一倍**。Cube 与 Vector 核数不同（1:2），均摊除数必须与该算力所属核类一致。

主杠杆：
- 提高 Cube:Vector 配比（即利用芯片固有的 r_cv 个 Vector 核分摊，而非只用 1 个）
- 减小握手粒度、per-stage 同步、双缓冲让两单元重叠
- 减少冗余同步屏障

### §2.5 计算算力（compute ceiling）

```
Perf_compute = Peak_FLOPS × util
```

util 主要由矩阵乘的 m / n / k 对齐决定。

### §2.6 三通道取 min

```
Perf ≈ min(Perf_compute, Perf_HBM, Perf_crosscore)
瓶颈 = 三者中最小的通道
```

| 区制 | 判据 | ceiling | 主杠杆 |
|---|---|---|---|
| Compute-bound | AI_HBM ≥ ridge 且算力最低 | `Peak_FLOPS × util` | 矩阵乘对齐、Cube/Vector 重叠、减同步 |
| HBM-bound | AI_HBM < ridge 且带宽最低 | `Peak_BW × AI_HBM` | 增大 mBaseSize、Q 驻留、增大 s2BaseSize 减 O_acc 往返 |
| Cross-core-bound | T_vec > T_cube | 片上吞吐 | 提配比、减握手粒度、双缓冲 |

### §2.7 搬运 / 计算比 R_compute（等价精确判定）

Roofline 用 AI vs ridge 判区制，工程上更直接的是**搬运/计算比**：

```
R_compute = 搬运耗时 / 计算耗时
```

- `R < 1`：compute-bound；`R = 1`：临界；`R > 1`：memory-bound
- 优化目标 `min max_i R_compute^(i)`（最差核的比值最小）

**D 的影响**：
- GM→L1 搬运主导的边界：R_compute 中 D 被约去，**与 D 无关**
- L0C→输出搬运的边界：`R ∝ (1/s2BaseSize + 1/D)`，D 减小时输出搬运更易成为瓶颈

## §3 基本块可行域：下界（性能）与上界（硬件）

### §3.1 下界 — Roofline 反推最小 mBaseSize

HBM-bound 下 `Perf ∝ AI_HBM ∝ mBaseSize`，需要 mBaseSize 足够大才能达到可接受性能：

```
目标 AI = α × ridge_HBM    (α 为可接受 ridge 占比，如 0.3~0.5)
→ mBaseSize_min = ceil(α × ridge_HBM × sizeof(T) / 2)
```

M 轴小（decode 短行）时 mBaseSize 受 M 轴锁死：若 `mBaseSize_min > M`，说明该 shape 下 HBM-bound 无法靠增 mBaseSize 突破 → 依赖流水隐藏延迟或 split-KV reduce（见 S6 负载均衡），**不改基本块**。

### §3.2 上界 — 硬件容量约束

基本块受 UB / L1 / L0 多层片上容量约束，取交集。softmax 状态 buffer（max / sum）随 mBaseSize 线性增长，是常驻项。

矩阵乘侧容量约束：
- L1：`subK × (subM + subN) × d_s ≤ S_L1 / 2`（÷2 为输入双缓冲）
- L0A：`2 × subM × subK × d_s ≤ S_L0A_avail`（×2 为 pingpong，硬约束）
- L0B：`2 × subK × subN × d_s ≤ S_L0B_avail`（×2 为 pingpong，硬约束）
- L0C：`2 × max(subM × subN, subM × D) × d_FP32 ≤ S_L0C`（两 GEMM 复用同一 L0C，×2 为 pingpong，硬约束）

> **L0 的 ×2 不可省**：L0A / L0B / L0C 份数恒 2（单缓冲会使装载 / 计算 / 搬出串行、流水阻塞）。容量放不下时**缩小 subM / subN / subK（或减小 s2BaseSize）**，禁止退回单缓冲；L0 容量与 K/V 槽池的 L1 余量（决定 r）互不占用，不得"省 L0 换 r"。

### §3.3 硬件常量来源

所有 Peak 值与容量必须**运行时查询（来自芯片规格），禁止硬编码**。

## §4 场景特化：Prefill / Decode / Causal（同一基本块下的瓶颈对照）

| 场景 | M 轴 | 典型瓶颈 | 首要杠杆 |
|---|---|---|---|
| Prefill（M=Sq×G 大） | 大，切满 mBaseSize 块 | compute / cross-core | 增大 mBaseSize 提 AI（受容量约束）；对齐；流水重叠 |
| Decode（M 轴小，Sq=1 / MTP 短行） | 小，mBaseSize 受 M 锁死 | HBM / cross-core | 基本块改不动 AI；核数富余时 split-KV（S6） |
| Causal（叠加） | 同上 | 同上 + 负载不均 | AI 近似不变；zigzag / 剪枝均衡 |

> 三场景共用同一套基本块（Cube-bound 目标统一切分），差别只在 M 轴有效行数与 S6 任务分发（split-KV / 剪枝 / zigzag），**不在基本块层分流**。

## §5 量化路径的 AI 修正

量化 dtype 下 K/V 的 HBM 流量按量化字节宽计：

```
Bytes_HBM ≈ 2 × Sk × D × sizeof(quant_T)
AI_HBM = 2·mBaseSize / sizeof(quant_T)
```

fp16（2B）→ fp8（1B）：`sizeof` 减半 → **AI_HBM 翻倍（约 +100%）**。

## §6 产出清单

进入 S5 流水前，方案必须展示以下推导（**每项展示算式，禁止只给结论**）：

逐阶段计算及搬运按 [FA 阶段成本](stage-cost.md) 代入；Vector 负载包含 V1 与 V2，Cube 负载包含 C1 与 C2，并关联各自共享资源与状态依赖。

- [ ] AI_HBM 公式 + 代入 mBaseSize/D/sizeof 的数值 + ridge_HBM
- [ ] 三通道 ceiling 各自估算 + min 取哪个 → 瓶颈区制
- [ ] cross-core 校核：T_vec vs T_cube 的定性 / 定量判断（含 r_cv 配比折算）
- [ ] mBaseSize_min（下界）+ mBaseSize_max / s2BaseSize_max（上界）
- [ ] 候选 mBaseSize、s2BaseSize + 在可行域中的相对位置
- [ ] 代入 S3 buffer 清单：Σ 总占用 ≤ 硬件容量（UB / L1 / L0）校验；L0 按 ×2（pingpong）计入，逐端口独立校验
- [ ] 容量不满足时的调整路径（减小 s2BaseSize / 转 streaming / split-KV）+ 迭代收敛后的最终 tiling 块
- [ ] 场景：prefill / decode /（是否 causal）
- [ ] 性能上限估算 = min(三通道 ceiling)
- [ ] 基本块套数由 spec 驱动：spec 未给场景典型值时默认 Cube-bound 归一为单一编译期常量切分（decode 短行优化下沉 S6 split-KV）；给了典型值时可分多套、每套仍为编译期写死常量并按 shape 属性静态分流，无运行时动态块值

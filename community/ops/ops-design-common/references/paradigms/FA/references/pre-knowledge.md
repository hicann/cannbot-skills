# FlashAttention 常识与硬件基础

> 前置知识：开始出方案前先加载本文档，建立 FlashAttention 的公共认知。所有变体（GQA / MHA / MLA / 量化 / 稀疏）共享本节内容。

## §1 FlashAttention 是什么

FA（FlashAttention）类算子是**分块化、不显式实例化 attention score 矩阵**的 attention 计算族。三个输入 Q / K / V，一个输出 O，沿 KV 维度分块流式累积 softmax 与加权和，**不在显存中构建完整的 \[Sq, Sk] score 矩阵**。

输入语义（维度名）：

- **Q**：query，含 batch / Sq / qHead / D 维
- **K / V**：key / value，含 batch / Sk / kvHead / D 维
- **O**：输出，维度与 Q 同

D 为 head dim，Hq / Hkv 为 query / kv head 数。变体在 Hq 与 Hkv 的关系、KV 是否压缩、Sq 是否退化为 1 等维度特化。

字节宽记号（全文档通用）：`d_s` = 输入张量字节宽（fp16/bf16 = 2，fp8 = 1，int8 = 1），`d_FP32` = 4（GEMM 累加器与 softmax 状态）。

与其他算子族的关键区别：

| 维度        | FA 类                                     | 其他族     |
| --------- | ---------------------------------------- | ------- |
| 计算类型      | 矩阵乘 + softmax 混合                         | 多为单一    |
| 计算流       | 跨核 stage 化（Cube 算 GEMM，Vector 算 softmax） | 单核内完成   |
| 跨 KV 分块状态 | 必须在线 softmax 累积（max / sum / O\_acc）      | 多数无在线状态 |

因此 FA 不能套用 reduction / elementwise / matmul 的设计模式。

### 全局符号表

| 符号                          | 含义                           | 首次出现 |
| --------------------------- | ---------------------------- | ---- |
| `Sq` / `Sk`                 | query / key-value 序列长度       | §1   |
| `D`                         | head dim                     | §1   |
| `Hq` / `Hkv`                | query / kv head 数            | §1   |
| `G`                         | `Hq / Hkv`（GQA 合轴倍数）         | §1   |
| `mBaseSize`                 | 任务/L1 级 M 轴分块（合轴后）           | S4   |
| `s2BaseSize`                | 任务/L1 级 KV 轴分块               | S4   |
| `subM` / `subN` / `subK`    | L0 级 M/N/K 子块                | §5   |
| `r_cv`                      | Vector:Cube 核数配比（如 2 表示 1:2） | §5   |
| `r`                         | 预取深度（GM→L1 槽池深度）             | S5   |
| `d_s`                       | 输入张量字节宽（fp16=2, fp8=1）       | §1   |
| `d_FP32`                    | fp32 字节宽（=4，累加器与 softmax 状态） | §1   |
| `S_UB` / `S_L1`             | UB / L1 容量                   | §5   |
| `S_L0A` / `S_L0B` / `S_L0C` | L0A / L0B / L0C 容量           | §5   |

和arena仓一致

## §2 数学公式（Online Softmax）

对每个 KV 分块 `j = 0, 1, ..., (Sk/s2BaseSize)-1` 顺序累积：

```
S_j   = (Q · K_j^T) * scale                       # [mBaseSize, s2BaseSize]
m_j   = max(m_{j-1}, rowmax(S_j))                 # [mBaseSize]   (m_{-1} = -∞)
sum_j = exp(m_{j-1} - m_j)*sum_{j-1} + rowsum(exp(S_j - m_j))   # [mBaseSize]
O_j   = exp(m_{j-1} - m_j)*O_{j-1} + exp(S_j - m_j)·V_j         # [mBaseSize, D]

最终: O = O_last / sum_last
```

关键性质：

- `S_j` 仅 `[mBaseSize, s2BaseSize]` 大小，与 Sk 无关 —— FA"不实例化全矩阵"由此实现。
- 跨块累积需 **rescale 旧状态**（`exp(m_{j-1} - m_j)`），这是 online softmax 的本质。
- 末块归一化（除以 `sum_last`）只做一次；**中间步骤的 P 不做归一化**。

## §3 数据流四阶段

每个 KV 分块的计算切成 4 个 stage，跨计算单元协同：

| 阶段     | 操作                               | 产出                             |
| ------ | -------------------------------- | ------------------------------ |
| **C1** | Q·K^T 矩阵乘                        | `[mBaseSize, s2BaseSize]` 中间矩阵 |
| **V1** | scale + softmax（未归一化 `exp(S-m)`） | 未归一化 P                         |
| **C2** | P·V 矩阵乘                          | `[mBaseSize, D]` 中间矩阵          |
| **V2** | 跨块累积（rescale + Add）+ 末块归一化       | 写回 O                           |

C1 → V1 → C2 → V2 是同一 KV 分块内的**严格数据流方向**（正确性约束 I1）。不同 KV 分块之间如何交错（流水排布、几级流水）由设计阶段决定。

## §4 跨 KV 分块状态

**task（任务）**：多核并行的最小分发单元，粒度为二维分块 **\[M 分块，S2 分块]**——M 轴（合轴后的 Sq×G，见 I2）上的一个分块 × KV 轴（S2/Sk）上的一个分块范围。task 内部沿 KV 维顺序遍历自己的分块、持有并累积自己的 max / sum / O\_acc。

- 默认（不切 S2）：task 的 S2 范围覆盖全 Sk，跨块状态 task 内闭环，末块归一化后直接写回 O。
- 切 S2（split-KV / FlashDecoding）：同一 M 分块的多个 task 各自累积 partial max / sum / O\_acc，须由跨任务 reduce 合并（rescale 后求和）再归一化。

跨 KV 分块需持有三类量：

- **max**：`[mBaseSize]`，每行当前最大 score
- **sum**：`[mBaseSize]`，每行归一化 sum
- **O\_acc**：`[mBaseSize, D]`，累积的未归一化输出

并行策略不同，这三类量的物理实现与生命周期不同（默认 task 级顺序累积 vs split-KV reduce）。

## §5 硬件基础

### 计算单元

- **Cube 单元**：矩阵乘（GEMM），吞吐以 FLOPS 计。
- **Vector 单元**：softmax 的 ReduceMax / Exp / ReduceSum 等归约与指数。
- 两者跨核协同（stage 化），需片上握手同步。

### CV 配比（Cube:Vector 核数比）

芯片规格给出的核数分 Cube 核与 Vector 核两类，二者配比不恒为 1:1：

- **1:2 型**（如 DAV\_2201 / DAV\_3510，型号仅为**示例**）：**1 个 Cube 核固定搭档 2 个 Vector 核（AIV）**，构成一个执行组。
- **其他型**：Cube/Vector 集成于同一核（无独立配比），或配比非 1:2——以芯片规格为准，禁止默认 1:1。

> 记号约定：`r_cv` = **Vector:Cube 配比**（每个 Cube 核搭档的 Vector 核数）。1:2 型（Cube:Vector = 1:2）架构下 `r_cv = 2`——注意两个比值的表达方向相反，禁止把 r\_cv 误取为 1/2。

对 FA 设计的影响（贯穿 S2–S5）：

- **UB 每 Vector 核私有**：规格里的 UB 容量是**单个 Vector 核**的值；1:2 下一个执行组有 2 份独立 UB。
- **V1/V2 按 mBaseSize 行维分摊**：同一 task 的 softmax / 累积由搭档的多个 Vector 核按行对半并行，每核只驻留 `mBaseSize/r_cv` 行（`r_cv` = Vector:Cube 配比）。max / sum / O\_acc 每核只持自己那部分行，行间无依赖，无需跨核归约。
- **容量校验口径分层**：UB 层按「每 Vector 核占用（mBaseSize/r\_cv 行）≤ 单核 UB」校验；L1 / L0 挂 Cube 侧，按全量 mBaseSize 校验。**禁止把全量 UB buffer 直接对比单核 UB**（等效按 1:1 计，会把 mBaseSize 压低一档）。
- **Vector 有效吞吐** = 单核 Vector 算力 × r\_cv。

### 片上缓存层级

- **UB**：最靠近计算单元的统一缓冲，容量最受限。
- **L1**：片上次级缓冲。
- **L0**：矩阵乘专用分块缓冲，含三个端口：
  - **L0A**：左输入缓冲
  - **L0B**：右输入缓冲
  - **L0C**：累加输出缓冲
  - **FA 中 L0 三个端口均需 pingpong（各 2 份 slot，结构硬约束）**
- **HBM / GM**：片外显存，带宽相对低。

### 多级搬运数据流

矩阵乘的数据流：`GM(HBM) → L1 → L0 → L0C → 输出`。各级带宽与容量来自芯片规格，禁止硬编码。链路上每一跳由独立搬运 / 计算单元完成，只有两端各有 2 份 slot 才能让相邻块重叠。

### 切分粒度两级

- **任务/L1 级**：M 轴分块 `mBaseSize`、KV 轴分块 `s2BaseSize`、N 轴（D 维，不切）
- **L0 级**：`subM / subN / subK`，层级约束 `subX ≤ 任务级对应维度`

两个 GEMM 的 L0 子块映射：

| GEMM    | M         | K          | N          | L0 子块约束                                       |
| ------- | --------- | ---------- | ---------- | --------------------------------------------- |
| C1 QK^T | mBaseSize | D          | s2BaseSize | subM ≤ mBaseSize, subK ≤ D, subN ≤ s2BaseSize |
| C2 PV   | mBaseSize | s2BaseSize | D          | subM ≤ mBaseSize, subK ≤ s2BaseSize, subN ≤ D |

### 对齐约束

subM / subN / subK 必须对齐到矩阵乘硬件粒度（如 16），否则利用率下降。

### 数据格式（ND / NZ / DN）与通路转换约束

数据在片上流转时分三种存储格式，**不同 buffer 的格式不同直接决定能否原位写 / 同槽复用**，buffer 规划（S3）与计算流（S2）必须逐 buffer 标注：

| 格式     | 含义                         | 典型所在                                          |
| ------ | -------------------------- | --------------------------------------------- |
| **ND** | 行优先常规连续布局（每行 D 个元素连续）      | GM 上的 Q/K/V/O；Vector 在 UB 的输入输出               |
| **NZ** | 分形（fractal）块化布局，矩阵乘装载要求的格式 | **L1 上所有 GEMM 操作数（Q/K/V/P，L1 只存 NZ）**；L0A/L0B |
| **DN** | 分形转置布局                     | 特定转置装载通路（按需）                                  |

**通路转换能力（硬件事实，方案不得违反）**：

```
GM ──ND──>(GM→L1 搬运完成 ND→NZ 转换)──> L1(NZ) ──NZ──> L0A/L0B ──(矩阵乘)──> L0C ──ND──> UB
UB ──仅 NZ──> L1        （UB→L1 无格式转换能力，写入 L1 的数据必须已是 NZ）
```

- **L1 上格式必须全是 NZ**：GM→L1 的搬运通路自带 ND→NZ 转换，数据一进入 L1 即为 NZ；L1→L0 装载通路不做格式转换，要求 L1 侧已是 NZ。**方案中任何标注为 L1 的 buffer 格式一律 NZ，禁止标注 ND**。
- **L0C→UB 输出为 ND**：Vector 拿到的 S / O\_tile 是 ND。
- **UB→L1 必须 NZ**：Vector 产出要回交 Cube 的中间量（典型即 P）**必须在 UB 侧先排成 NZ** 再搬入 L1——「UB 上 cast 出 P」与「P 排成 NZ」是同一 V1 stage 内必须完成的事，不是可选优化。
- **原位写 / 同槽复用前提：格式一致 + dtype 一致 + 生命周期不重叠**，三者缺一禁止同槽。反例：S（UB，ND，fp32）与 P（UB→L1，NZ，fp16）格式与 dtype 双不同，**禁止原位复用 / 共享槽**——不考虑格式会错误地把两者划入同一 slot 池，物理上写出的 NZ P 会按 ND 解读，直接错数。

### 性能主线

三个独立吞吐通道：Cube 算力 / HBM 带宽 / 片上跨核吞吐，性能取三者最小。

> 芯片规格（UB / L1 / L0 容量、核数、Cube 算力、Vector 算力、带宽）来自 patterns.md 输入的第二个输入，Peak 值与容量一律运行时查询，禁止硬编码。

## §6 正确性约束 I1-I5

任何 FA 实现都必须满足，**违反必崩**（输出全零 / NaN / 精度崩溃 / 死锁）。

- **I1 数据流方向**：同一 KV 分块内 stage 依赖严格有序 C1→V1→C2→V2，禁止重排。
- **I2 同 kvHead 同任务**：同一 kvHead 的 KV 数据只加载一次、由同一任务消费。实现上把该 kvHead 下的 G 个 qHead 与 Sq **合轴**成一条 M 轴（M = Sq × G），M 轴分块大小即 `mBaseSize`；M 轴分块全部留在同一任务内共享 KV 加载，禁止按 qHead 逐头拆任务（否则 KV 重复搬运 G 次）。
- **I3 跨任务状态隔离**：跨任务持有的状态（max / sum / O\_acc）按任务分槽，禁止共享同一物理位置。
- **I4 状态累积正确性**：跨块累积的状态必须完整正确。沿累积维（S2）切分出的多个 task 各持 partial max / sum / O\_acc，**必须配跨任务 reduce 合并**（rescale 后求和）再归一化，禁止只累积不合并。
- **I5 量化 scale 轴对齐**：量化 scale 沿被消费矩阵乘的 reduction 轴量化，禁止轴错位。**反直觉点**：V 在 `[B, Sk, Hkv, D]` 布局下 innermost 是 D，但 P·V 的 reduction 轴是 S\_k，故 **V 的 scale 沿 S\_k**（不是 D）。

## §7 softmax 数值陷阱

除 I1-I5 结构约束外，softmax 数值计算还有以下易错点：

- **mask 用大有限负数**（如 -1e30），**禁用 -∞**：整行被屏蔽时 -∞ 会使 `rowmax = -∞`，`exp(-∞-(-∞)) = NaN`，并沿 S2（KV 轴）跨块累积传播。
- **完全屏蔽的 tile**：跳过其矩阵乘的同时，也**必须跳过**该 tile 的 softmax 与状态更新（贡献为 0）；只跳一半会得到 NaN。
- **padding 行**：softmax 前写屏蔽值，输出最终强制清零，不参与有效累积。
- **首块初始化**：running max = -∞、sum = 0、O\_acc 清零；rescale 因子等价于 1（首块不做 rescale）。
- **末块除零保护**：`O = O_acc / sum` 前对 sum 做下限保护（加 eps / clamp），避免 padding 或极端输入导致 inf / NaN。

# Buffer 规划（公式化清单）

> 本步骤（S3）只产出**公式化 buffer 清单**：单份大小用符号（mBaseSize / s2BaseSize / D / d_s 等）表达，份数 N 用公式表达（预取深度作参数）。**大小数值由 S4 代入、份数数值由 S5 回填**（预取深度来自流水级数），两者齐备后以硬件内存为约束迭代校验。

**回答：需要哪些 buffer？各自大小的符号公式？每个 buffer 的「生命周期」「物理份数 N」「跨 buffer 复用」分别是什么？**

以下共享 K/V 槽池、P 的预填份数对应各自说明的候选模型。选定其他装载或交接排布时，按 [流水编排](pipeline.md) 回填逐对象的实际生命周期。max/sum、工作项缩放系数与 O_acc 分别记账。

- 输入 buffer / 输出 buffer / 中间计算 buffer / 状态 buffer（max / sum / O_acc）
- 各 buffer 单份大小公式（符号化，用 mBaseSize / s2BaseSize / D 等变量）
- **数据生命周期**：常驻（KV 遍历期间数据不变）还是轮转（逐 KV 块替换覆写）
- **物理份数 N**：分配几份 slot（1 份 / pingpong 2 份 / ring N 份），与生命周期正交
- **Cube 侧 L0**（L0A / L0B / L0C）：份数**恒 pingpong（N = 2）**，不接受单缓冲（见下「L0」）
- **跨 buffer 复用**：生命周期不重叠的不同 buffer 时分复用同一物理内存
- 总占用 Σ = Σ(单份大小 × 份数)（符号化，大小待 S4 代入、份数待 S5 回填；L0 份数在 S3 即定死为 2）

产出：
- buffer 清单（名称 + 用途 + 符号化单份大小 + 生命周期 + 物理份数 N + 轮转方式）
- 逐块占用分解（每个 buffer：单份大小 × N 份 = 小计）
- 跨 buffer 复用方案
- 总占用与容量校验的符号化公式

## 三个正交维度：数据生命周期 × 物理份数 × 跨 buffer 复用

「常驻」「轮转」「复用」分属三个正交维度，分开决策、分别标注：

- **数据生命周期**：该 buffer 的数据在 KV 遍历过程中是否保持不变（数据特性，不涉及内存复用）。
- **物理份数 N（slot 轮转）**：分配几个物理 slot 做预取 / 读写重叠。
- **跨 buffer 复用**：生命周期不重叠的**不同 buffer** 时分复用同一块物理内存。**复用硬前提：格式（ND/NZ/DN）一致 + dtype 一致 + 生命周期不重叠**。

> 例：Q_tile 生命周期上是**常驻**（切 KV 时，不同 KV 块与同一个 Q 块相乘，KV 在遍历、Q 块不动），物理上**仍可 pingpong 轮转**（Sq 分块时，算 Q 块 i 的同时预取 Q 块 i+1 到另一 slot）。二者可共存。S（fp32）消费完后其物理内存可被后续 tile 的别的 buffer **时分复用**——这是第三维，与前两维独立。

### 维度一：数据生命周期

| 生命周期 | 定义 | 命中 buffer |
|---|---|---|
| 跨块保留 | Q 保持不变，累计状态按 KV 顺序更新并承接旧值 | Q_tile、O_acc、max、sum |
| 轮转 | 数据逐 KV 块替换（旧块数据失效、新块覆写） | K_tile、V_tile、S（UB）、P（UB 与 L1） |

> **Q tile 块常驻是结构决策**：Q 在整个 Sk 遍历期间常驻片上（L1/UB），切 KV 块时不重复加载 Q，禁止每个 KV 块重搬 Q。
>
> 注：「轮转」描述的是**数据**逐块失效被覆写（生命周期特性），与「物理份数 N 的 slot 轮转」（维度二）是两层——前者说数据能活多久，后者说给几个物理槽。

### 维度二：物理份数 N（轮转方式）

| 份数 | 轮转方式 | 含义 |
|---|---|---|
| N = 1 | 单份 | 无读写重叠 |
| N = 2 | pingpong | `slot = j % 2`，读写交替 |
| N ≥ 2 | ring（预取深度 r） | `slot = j % N`，`r = N - 1`，更深预取 |

### 交叉表：每个 buffer 的（数据生命周期, 物理份数）

| Buffer | 数据生命周期 | 物理份数 N | 说明 |
|---|---|---|---|
| Q_tile `[mBaseSize, D]` | 常驻 | 1 或 ≥ 2 | 跨 KV 块复用不动；Sq 分块时可 pingpong 预取下一 Q 块 |
| K/V 共享槽池 `[s2BaseSize, D]` | 轮转 | N_KV ≥ r+1 | K、V 单份同构，共享一个槽池按 `K(j),V(j),K(j+1),V(j+1)…` 交替承载（见下「K/V 共享槽池」） |
| S `[mBaseSize, s2BaseSize]`（UB） | 轮转 | **恒 2** | L0C→UB 通路 pingpong，Cube 写 / Vector 读交替 |
| P `[mBaseSize, s2BaseSize]`（UB vec1res） | 轮转 | **恒 2** | Vector 生产 / 搬出 L1 交替 |
| P `[mBaseSize, s2BaseSize]`（L1） | 轮转 | **N_P_L1 = r + 1** | preload 阶段 Vector 连续产出多块 P 囤于 L1，与 K/V 槽池同构 |
| O_acc `[mBaseSize, D]` fp32 | 跨块保留 | 尚未完成最后读取的 O_acc 份数 | 首次 V2 到输出读取完成，按 task 归属 |
| max `[mBaseSize]` fp32 | 跨块保留 | 尚未完成最后读取的 max 份数 | 首次 V1 到最后使用，按 task 归属 |
| sum `[mBaseSize]` fp32 | 跨块保留 | 尚未完成最后读取的 sum 份数 | 首次 V1 到最后使用，按 task 归属 |
| alpha `[mBaseSize]` fp32 | 逐工作项保留 | 尚未被对应 V2 使用完成的版本数 | V1 产生，V2 延迟读取，与 task 槽分别编号 |
| L0A `[subM, subK]` NZ | 轮转（逐 base 子块替换） | **恒 2** | 左操作数装载槽（Q / P）；单份则装载与矩阵乘串行 |
| L0B `[subK, subN]` NZ | 轮转 | **恒 2** | 右操作数装载槽（K / V）；同上 |
| L0C `[subM, subN]` fp32 | 轮转 | **恒 2** | 累加结果搬出槽（S / O_tile）；单份则矩阵乘与搬出串行 |

### K/V 共享槽池

当 `K_tile` 与 `V_tile` 的物理布局、类型与单份容量一致，且完成协议允许其存储复用时，可选择共享槽池；C1 消费 K(j)，C2 消费 V(j)。本模型在消费期间保留一个槽，其他槽用于预取：

```
N_KV ≥ r + 1        # r = 预取深度（已加载未消费的槽数）；+1 = 正在消费的槽
```

推导：C1 与 C2 在同一 Cube 单元上**串行**（Cube 核一次只跑一个 GEMM），稳态下任意时刻最多 1 个槽在被消费；其余槽全部可用于预取。槽位按数据流顺序 `K(j), V(j), K(j+1), V(j+1)…` 依次承载，物理槽在 K/V 间交替复用。

- **r 由 S5 的实际预取排布决定**；份数覆盖尚未释放的预取与消费对象，选择更深预取时重新校验 Σ 容量、同步与收益。
- 相比 K、V 各自独立 ring（各自需 r+1 槽、共 2r+2 槽），共享池只需 r+1 槽：省下的槽来自「同一时刻只有一个槽在被消费」——独立 ring 中未消费的 ring 槽无法跨张量挪用，共享池可以。
- 约束：槽的释放须等其承载的 K 被 C1（或 V 被 C2）消费完，同步按槽序保证，**禁止 K/V 混用导致 C1/C2 读错槽**。

### 状态 buffer 轮转语义（max / sum / O_acc）

同一 task 的递推保持稳定逻辑归属，每次更新承接该 task 的正确旧值。max/sum 由 V1 更新，O_acc 由 V2 更新；初始化和最终释放按各自阶段定位。alpha 按工作项保留旧版本，直到所属 V2 使用完成。

分别计算各状态的最大同时占用，再校验容量。多个短 task 交错时，max/sum 可以超过两份；V2 按 task 顺序推进且最终输出完成后再初始化下一 task 时，O_acc 可以独立用一份。具体推导见 [四阶段示例](pipeline-example.md)。容量不足时调整排布或块大小，并同步修改等待和释放。

### L0：恒 pingpong（结构硬约束）

Cube 处理一个 base 子块要串行走完三条通路，三条各由**不同硬件单元**承担：

```
L1 ──(装载单元)──> L0A / L0B ──(矩阵乘单元)──> L0C ──(搬出单元)──> UB
```

单元不同 ⇒ 三者落在**不同 slot 上**可以真正并行。因此相邻块重叠的前提是每个端口各留 2 份 slot：块 j 在算的同时，块 j+1 的装载（或块 j−1 的搬出）在另一个 slot 上进行。

| 端口 | 单缓冲（N=1）的后果 | pingpong（N=2）后 |
|---|---|---|
| L0A / L0B | 装载块 j+1 必须等块 j 的矩阵乘读完，否则覆写正在被乘的操作数 → **装载与矩阵乘串行** | 装载块 j+1 与块 j 的矩阵乘并行 |
| L0C | 块 j+1 的累加结果无处可写，必须等块 j 的 L0C→UB 搬完 → **矩阵乘与搬出串行** | 搬出块 j 与块 j+1 的矩阵乘并行 |

结论：

```
N_L0A = N_L0B = N_L0C = 2      # 结构硬约束，不接受 N = 1
```

- **单缓冲 = 流水阻塞**：任一端退化到单份，该端两侧立刻串行，Cube 有效利用率腰斩（加载 / 搬出时间被完全暴露在关键路径上）。这属于 patterns.md 纪律中"把退化配置作为设计目标"的禁止项。
- **与流水级数 r 正交**：r 描述 GM→L1 的 KV 槽池深度（S5 回填）；L0 是 **Cube 内部块级重叠**，份数恒 2，不随 r 增长、不经 S5 回填——S3 直接定死为 2。
- **容量是唯一上界**：pingpong 使 L0A / L0B / L0C 各按 2 份计入 L0 容量（S4 校验，`2 × 单份 ≤ S_L0X`）。容量放不下时**缩小 subM / subN / subK（或减小 s2BaseSize）**，**禁止**退回单缓冲。
- **三个端口独立成槽**：L0A / L0B / L0C 是三个独立物理缓冲，左操作数、右操作数、累加结果的角色与格式均不同，**禁止跨端口共用 slot**。

## 份数决策规则

```
# 各状态分别按尚未完成最后读取的对象计数
N_ms = max_t active_tasks_with_unreleased_max_sum(t)
N_Oacc = max_t active_tasks_with_unreleased_Oacc(t)
N_alpha = max_t active_workitems_with_unconsumed_alpha(t)

# 常驻 buffer：份数与生命周期无关，按预取需求独立定
N_Q = 1（不预取） 或 ≥ 2（Sq 分块时预取下一 Q 块）

# 轮转 buffer：K/V 共享槽池 ≥ 预取深度 + 1；S（UB）恒 2；P（UB）恒 2；P（L1）= r + 1
N_KV ≥ r + 1          # K、V 共享槽池，交替承载；C1/C2 同 Cube 串行故同时仅 1 槽在消费；取更大值加深预取
N_S = 2               # L0C→UB 通路 pingpong，Cube 写 / Vector 读交替
N_P_UB = 2            # Vector 生产 / 搬出 L1 交替
N_P_L1 = r + 1        # preload 阶段 Vector 连续产出多块 P 囤于 L1，与 K/V 槽池同构

# L0：恒 pingpong，与 r 无关、不经 S5 回填（S3 定死）
N_L0A = N_L0B = N_L0C = 2     # 硬约束：单缓冲则装载/计算/搬出三者串行 → 流水阻塞
```

> 唯一「禁止轮转」的硬约束只落在**跨块累积状态的 task 内期间**（max / sum / O_acc，I3）；其跨 task 份数与其余 buffer 的 N 一样由流水 / 预取 / 容量独立决定，与「常驻 / 轮转」生命周期正交，**不因常驻就必须 N=1**。公式中的「预取深度」参数由 S5 流水编排提供，本步骤只出规则。

## 总占用公式（含份数）

```
单份大小 size_i（符号化，代入 mBaseSize/s2BaseSize/D/d_s）
Σ_total = Σ_i (size_i × N_i)
        = size_Q×N_Q + size_KV×N_KV + size_S×2 + size_P×(2 + N_P_L1)
        + size_Oacc×N_Oacc + size_max×N_ms + size_sum×N_ms + size_alpha×N_alpha
        # K/V 与 P 的份数对应所选预取模型；各状态份数按各自最后读取推导

Σ_L0（逐端口独立校验，不合并为一个池）
     = 2 × size_L0A ≤ S_L0A      # size_L0A = subM × subK × d_s
     = 2 × size_L0B ≤ S_L0B      # size_L0B = subK × subN × d_s
     = 2 × size_L0C ≤ S_L0C      # size_L0C = subM × max(subN, D) × d_FP32（C1/C2 共用该端口）
```

容量校验对 UB / L1 / L0 分层，每层只累加驻留在该层的 buffer 的 `size_i × N_i`（S4 代入大小、S5 代入份数后校验）。

- **以 N 表达流水策略，不省略**：份数 N 就是流水的物理体现（N=1 是单缓冲串行，N=2 是 pingpong，N=3 是更深预取）。只写"双缓冲"不落到 N 等于没定份数。
- **约束归属按 buffer 生命周期 × 驻留层对位**：轮转 buffer（K/V 槽池、S、P、O_tile，块内一拍生产即消费、旧块失效被覆写）份数由流水级数/双缓冲定，其预取放大部分（K/V 槽池与 P 的 L1 份数，均 = r+1）驻留 L1、只受 L1 约束，UB 中的 S/P/O_tile 份数不随 r 增长；常驻状态 buffer（max/sum/O_acc，跨块保留、按累积域（task）各持一份）份数 = 同时活跃累积域数，活跃块（已进入流水、stage 序列未完成）跨多个 task 时该数随 r 增长、受 UB 约束；L0（L0A / L0B / L0C）份数恒 2、**与 r 无关**，只受 L0 容量约束（且各端口独立校验，不合并）。禁止跨层级 / 跨属性错配论证（如一律用 UB 余量限制 r，或忽略状态 buffer 的活跃域多份，或把 L0 容量省下来换 r）。

> **校验口径必须按 CV 配比分层**：
> - **UB 层（Vector 侧）**：1:2 型架构下 V1/V2 按 mBaseSize 行维在搭档 Vector 核间分摊，每核只驻留 `mBaseSize/r_cv` 行 ⇒ 按「每 Vector 核占用 ≤ 单核 UB」校验，即 UB 各 buffer 的 mBaseSize 以 `mBaseSize/r_cv` 代入。Cube 写入 UB 的 S / O_tile 同样按行切片落各核，适用同一口径。
> - **L1 / L0 层（Cube 侧）**：按全量 mBaseSize 校验，不分摊。
>
> 若直接把全量 UB buffer 对单核 UB 校验（等效 1:1），会把 mBaseSize 人为压低一档（例如 D=128 时 256→128），丢失近一半 prefill 性能上限。

## 实例：典型 FA 的 buffer 分层

> 一份标准 FlashAttention 片上 buffer 分层参考，展示「复用语义 × 物理份数 × 层级 × dtype」如何落到真实算子。Q / K / V / P 用 fp16/bf16；GEMM 输出与 softmax 状态用 fp32。

### 分层总览

| 层 | buffer | 形状 | dtype | 格式 | 生命周期 | 物理份数 N |
|---|---|---|---|---|---|---|
| L1 | Q | `[mBaseSize, D]` | fp16/bf16 | NZ（GEMM 操作数） | 常驻 | 1 或 ≥2 |
| L1 | K | `[s2BaseSize, D]` | fp16/bf16 | NZ | 轮转 | ≥2 |
| L1 | V | `[s2BaseSize, D]` | fp16/bf16 | NZ | 轮转 | ≥2 |
| L1 | P | `[mBaseSize, s2BaseSize]` | fp16/bf16 | **NZ（UB→L1 仅 NZ 通路）** | 轮转 | r + 1 |
| L0A | 左操作数（Q / P） | `[subM, subK]` | fp16/bf16 | NZ | 轮转 | **2（硬约束）** |
| L0B | 右操作数（K / V，转置形式） | `[subK, subN]` | fp16/bf16 | NZ | 轮转 | **2（硬约束）** |
| L0C | 累加结果（S / O_tile） | `[subM, subN]` | fp32 | —（累加器） | 轮转 | **2（硬约束）** |
| UB | bmm1res = S | `[mBaseSize, s2BaseSize]` | fp32 | ND（L0C 输出原生 ND） | 轮转 | 2 |
| UB | max | `[mBaseSize]` | fp32 | —（一维） | 常驻 | 1 |
| UB | sum | `[mBaseSize]` | fp32 | —（一维） | 常驻 | 1 |
| UB | vec1res = P | `[mBaseSize, s2BaseSize]` | fp16 | **NZ（须在 UB 侧排成 NZ 再搬 L1）** | 轮转 | 2 |
| UB | bmm2res = O_tile | `[mBaseSize, D]` | fp32 | ND | 轮转 | 2 |
| UB | vec2res = O_acc | `[mBaseSize, D]` | fp32 | ND | 常驻 | 1 |

> 格式列是**复用决策的硬输入**（前提见本文「三个正交维度」维度三）。S（ND, fp32）与 vec1res P（NZ, fp16）格式与 dtype 双不同，**禁止同槽**；O_tile（ND, fp32）与 S 格式与 dtype 一致、生命周期不重叠（S(j) 消费完后 O_tile(j) 才产生），**可同槽时分复用**。

### 分层规则

- **L1 放 GEMM 输入操作数**：Cube 只从 L1→L0A/L0B 取数，Q / K / V / P 都是矩阵乘输入，故都经 L1。
- **UB 放 GEMM 输出 + Vector 工作区**：S / O_tile 由 L0C 搬入 UB，softmax / 累积（Vector）在 UB 上完成。
- **P 在 L1 的原因**：PV 相乘在 Cube 核，P 是 BMM2 左输入，必须经 L1→L0A。
- **L0 全 pingpong**：L0A / L0B / L0C 各 2 份（**不可省**，论证见本文「L0」）。

### 数据流（C1 → V1 → C2 → V2）

```
C1: Q,K(L1 NZ→L0A/L0B) → Cube S=QK^T → L0C → UB(bmm1res, fp32, ND)
V1: UB 上 reduceMax→max、exp→cast fp16 并按 NZ 排布存 vec1res、reduceSum→sum
    vec1res(P, NZ) 经 UB→L1（仅 NZ 通路）搬成 L1 的 P
C2: P,V(L1 NZ→L0A/L0B) → Cube O_tile=PV → L0C → UB(bmm2res, fp32, ND)
V2: O_acc = exp(m_prev−m)·O_acc + bmm2res（UB 上，vec2res=O_acc, ND）；末块 O = O_acc/sum
```

### dtype 约定

- **vec1res（P）是 UB 上唯一 fp16**：softmax 算出 P 后在 UB 上 cast 成 fp16 再存 vec1res，搬去 L1 的 P 已是 fp16。
- **其余 UB buffer 全 fp32**：S、O_tile、max、sum、O_acc 保持 fp32，与「中间 softmax 状态（max/sum/O_acc）以 fp32 计算」一致 —— P 不属于 fp32 状态。

### 命名映射

| CANN 常见命名 | 含义 |
|---|---|
| bmm1res | S = QK^T 结果 |
| bmm2res | O_tile = PV 单块结果 |
| vec1res | P = softmax 结果（cast 后 fp16） |
| vec2res | O_acc = 跨块累积输出 |

> bmm1res（S, fp32, ND）用毕后其空间可复用给 bmm2res（O_tile）；vec1res（P, fp16, NZ）必须与 S 独立分块——复用前提与反例见本文「三个正交维度」维度三及上表注。

> 量化变种需额外的 scale buffer 与回传路径，按量化 trait 规则处理。
> 稀疏变种块长不定，buffer 按最坏块长分配。

## 结构决策：∝ m×D 的 buffer 是否 streaming

若存在尺寸 ∝ m × D 的常驻 buffer（典型如 O_acc `[mBaseSize, D]`），大 D 下必然溢出 → **必须**移出片上、改存片外 workspace，按 chunk 流式读回：

```
chunk 行数 = 预算字节 / (D_align × 字节宽)
```

这是**结构决策**（非数值标定）：定死后须有显式字段承载，禁止由实现自行改回常驻。

streaming 到片外的 buffer（及 split-KV 的 partial max / sum / O_acc）落在一块**片外 workspace（GM）** 中，其总量按并发 task 槽位估算：

```
workspaceTotal = Σ(per-slot 段大小) × 并发槽位数
并发槽位数 = min(totalTasks, usedCoreNum)   // ≈ 核数，不是总任务数
```

`totalTasks` = 任务总数、[M 分块 × S2 分块] 展开；`usedCoreNum` = 实际参与计算的核数（运行时输入）。按 `totalTasks` 分配会放大数百倍，推理小 batch 场景易超内存。

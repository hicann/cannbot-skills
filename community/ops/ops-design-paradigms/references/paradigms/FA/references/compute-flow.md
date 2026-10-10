# 单核计算流

> 本步骤（S2）定义**核内数据流**与**核间流水**两个层次，是 S3 Buffer 规划、S4 性能建模、S5 流水编排的逻辑前置。
>
> **边界**：本步骤只描述数据流方向与 stage 接力（数据沿哪些缓存层级流动、跨核如何传递），**不决定切块**（tile 大小 / subM / subN / subK 等）；切块是 S4 性能建模的职责。

**回答两个问题：**

1. **核内**：单个 Cube 核 / Vector 核内部，数据在 GM / L1 / L0A / L0B / L0C / UB 之间如何搬运（通道连接）？
2. **核间**：Cube 与 Vector 两个计算单元如何接力（stage 序列 + 跨核通道）？

产出：

- 核内数据流：Cube 核内通路 + Vector 核内通路
- 核间 stage 序列 + 跨核通道

下面的 C1 → V1 → C2 → V2 四阶段、核内搬运路径与跨核通道，只是\*\*标准 FlashAttention（无 feature 的 GQA / MHA）\*\*的一个典型示例，并非所有变体都必须遵守的唯一形式。

## 核内数据流

数据以「通道」为单元流动：一条通道连接两个缓存层级，数据沿通道单向搬运；通道的份数（pingpong / ring）由 S3 Buffer 规划确定，本步骤只标通道方向。

### Cube 核内（矩阵乘 stage）

矩阵乘沿专用硬件通路流动：

```
GM ──ND──>(GM→L1 搬运自带 ND→NZ)──> L1(NZ) ──NZ──> L0A / L0B(NZ) ──(矩阵乘)──> L0C ──ND──> UB
```

- 输入操作数先驻留 L1，再送入 L0A（左矩阵）/ L0B（右矩阵，转置形式）
- 矩阵乘在 L0C 累加出结果，从 L0C 搬到 UB（ND），交给 Vector 侧
- L0A / L0B / L0C 恒 pingpong（N=2，结构硬约束）

C1 / C2 两个 stage 的核内通路：

| stage | 左操作数                    | 右操作数                         | 结果                 |
| ----- | ----------------------- | ---------------------------- | ------------------ |
| C1 QK | Q：GM(ND)→L1(NZ)→L0A(NZ) | K：GM(ND)→L1(NZ)→L0B(NZ)（K^T） | S：L0C→UB(ND)       |
| C2 PV | P：L1(NZ)→L0A(NZ)        | V：GM(ND)→L1(NZ)→L0B(NZ)（V^T） | O\_tile：L0C→UB(ND) |

> 数据流层面的复用语义：Q 常驻 L1（KV 遍历期间不重载，GM→L1 一次），K / V 流式（随 KV 遍历加载）。各 buffer 的份数与大小分别在 S3 / S4 决定，本步骤仅标数据流方向。

### Vector 核内（softmax / 累积 stage）

数据在 UB 上做逐元素计算，通路：

```
UB(ND)（读入 S / P@V）──(逐元素计算)──> UB（P / O_acc）
UB ──仅 NZ──> L1（P 交接给 Cube）     UB(ND) ──> GM（最终 O 写回）
```

- **softmax**：UB 上读 S（ND），做 scale / mask / rowmax / exp，产出 P（cast 后）到 UB，再搬到 L1 供 Cube 的 PV 用。**P 必须在 UB 侧就排成 NZ 布局**——「cast 出 P」与「排成 NZ」是同一 V1 stage 内必须完成的事
- **累积**：UB 上读 P\@V（ND），累积到 O\_acc（UB，ND），末块归一化后 cast 写回 GM（ND）

## 核间流水（跨 Cube / Vector 的 stage 接力）

同一 KV 分块内严格 `C1 → V1 → C2 → V2`（I1，禁止重排）；末块归一化只在最后 KV 块做一次。

| stage | 计算单元   | 操作                                                | 输入 → 输出          |
| ----- | ------ | ------------------------------------------------- | ---------------- |
| C1    | Cube   | `S = Q·K^T × scale`                               | Q, K → S         |
| V1    | Vector | softmax：`reduceMax` / `exp` / `reduceSum`（含 mask） | S → P, max, sum  |
| C2    | Cube   | `O_tile = P·V`                                    | P, V → O\_tile   |
| V2    | Vector | 跨块累积 `O_acc = exp(m_prev−m)·O_acc + O_tile`       | O\_tile → O\_acc |
| 末块    | Vector | `O = O_acc / sum`（仅一次）                            | O\_acc → O 写回 GM |

跨核通过**跨核通道**传递中间结果（生产者写、消费者读）：

| 交接   | 生产者        | 消费者        | 通道所在层级 | 格式                                |
| ---- | ---------- | ---------- | ------ | --------------------------------- |
| S    | Cube（C1）   | Vector（V1） | UB     | ND（L0C 输出原生 ND）                   |
| P    | Vector（V1） | Cube（C2）   | L1     | **NZ（UB→L1 仅 NZ，P 须在 UB 侧排成 NZ）** |
| P\@V | Cube（C2）   | Vector（V2） | UB     | ND                                |

> **CV 配比与行维分摊**：1:2 型架构下 1 个 Cube 搭档 r\_cv=2 个 Vector 核，V1/V2 按 mBaseSize 行维在搭档核间对半并行；L0C→UB 的 S / O\_tile 按行切片落入对应 Vector 核的 UB，UB→L1 的 P 由各核分别搬出自己那半行（各核 NZ 分形独立成块）。状态 max/sum/O\_acc 每核只持自己行段，无跨核归约。

> P 是唯一「Vector 产出、Cube 消费」的中间量：Cube 的矩阵乘输入只从 L1→L0A 取数，P 须搬到 L1，且 NZ 布局由 Vector 在 UB 侧完成。

## 完整数据流（核内 + 核间串起来）

| stage      | 完整搬运路径                                                                   |
| ---------- | ------------------------------------------------------------------------ |
| C1 QK      | Q：GM(ND)→L1(NZ)→L0A(NZ)；K：GM(ND)→L1(NZ)→L0B(NZ)；L0A×L0B→L0C；S：L0C→UB(ND) |
| V1 softmax | S：UB(ND) 读入→(scale/mask/rowmax/exp/cast)→P(UB, NZ)；P：UB→L1（仅 NZ 通路）      |
| C2 PV      | P：L1(NZ)→L0A(NZ)；V：GM(ND)→L1(NZ)→L0B(NZ)；L0A×L0B→L0C；O\_tile：L0C→UB(ND)  |
| V2 累积      | P\@V：UB(ND) 读入→(rescale+add)→O\_acc(UB, ND)；末块 O\_acc→(cast)→GM(ND)      |

## 与 S5 流水编排的边界

- 本步骤定**逻辑计算流**：核内搬运通路 + 核间 stage 序列（做什么、谁做、数据怎么流）。
- S5 流水编排定**物理排布**：stage 如何重叠 / 几级流水 / 同步语义（怎么排）。
- 本步骤的 stage 顺序与跨核通道方向是 S5 的输入约束（I1 不可重排）。

## 特性对 stage 序列的改造

上面的 `C1 → V1 → C2 → V2` 只是无 feature 的基线。命中 trait 后 stage 序列按需改造，本步骤须按实际 trait 重新梳理 stage 与跨核通道，禁止照搬基线：

| trait | stage 序列变化 | 新增 / 修改的跨核通道 |
| --- | --- | --- |
| 稀疏（token/paged 粒度） | C1 前插入 **V0 gather** stage → `V0 → C1 → V1 → C2 → V2` | 聚合结果经片外周转交回 Cube，槽池份数与 K/V 一致（≥ r+1） |
| 量化（Vector 预反量化） | V1 内新增 dequant 子步骤，或新增独立 dequant stage | 反量化结果 UB→L1 回灌；P scale 随 P 写回 L1 |
| 量化（Cube 硬件反量化） | stage 序列不变，scale 作为 GEMM 额外输入 | scale 驻留 L1 与数据同层 |
| MLA | GEMM 语义改变（latent absorption 进 Q 侧），D 方向超宽需内迭代 | Q 沿 D 切分多次载入，latent 维决定 L0 子块约束 |

各 trait 的具体 stage 定义见对应 trait 文件；多 trait 叠加时按各自插入位置组合（如稀疏+量化 = `V0 → dequant → C1 → …`），并重新校验跨核通道的格式与驻留层约束。

## 检查点

- [ ] stage 序列完整（覆盖计算图全部节点），数据流方向与 I1 一致（无 stage 重排）
- [ ] 计算图逐 buffer 标注格式（ND / NZ / DN）；所有 UB→L1 搬运的 buffer（如 P）在 UB 侧即为 NZ

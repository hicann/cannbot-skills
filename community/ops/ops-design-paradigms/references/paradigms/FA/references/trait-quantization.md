# 量化 trait

> 触发条件：Q / KV dtype ∉ {fp16 / bf16}。量化 trait 改变核心计算路径，在 S3 / S4 额外加载。

## 核心影响

量化 trait 在乘法前对操作数做量化（量化乘法），引入 scale，需在方案中额外决策。

## S3 scale buffer 与回传

- scale 的存储 layout（沿被消费乘法的 reduction 轴）
- scale 是否需要反量化（AntiQuant）回传路径
- scale 额外 buffer 的大小公式与复用

### AntiQuant 位置决策（决定全链路）

| 位置 | 含义 | scale 驻留层 | 适用 |
|---|---|---|---|
| **Vector 侧反量化** | Cube 做纯量化 matmul，descale 融合进 V1 softmax / V2 累积 | **UB**（跟随消费方 Vector） | 对称 8-bit（per-token / per-tensor 细粒度 scale） |
| **Cube 侧硬件反量化** | scale 作为矩阵乘输入，dequant 在 GEMM 内完成，Vector 零 dequant | **L1**（与量化数据同层） | per-block power-of-two scale（MXFP8 类） |
| **Vector 预反量化（非对称必选）** | 量化权重先在 Vector 反量化成 16-bit，**经 UB→L1** 再以 16-bit 喂 Cube GEMM | 量化数据进 UB，反量化结果 UB→L1 | 非对称量化（A16W8 / A16W4） |

- **平台硬约束：Ascend950 系列 Cube 不支持非对称矩阵计算**——A16W8 / A16W4 无法作为 GEMM 输入，必须先在 Vector 侧把权重反量化回 16-bit，再走 UB→L1 通路送入 Cube。即非对称量化的 AntiQuant 位置**没有选择权**，恒为"Vector 预反量化 + UB→L1 回灌"，其代价是 L1 上驻留的仍是 2B 数据（L1 容量不省，只省 HBM 带宽），且 Vector 侧新增一段 dequant 计算须计入 cross-core-bound 校核。
- 对称 8-bit 量化才有上表前两行的二选一：Cube 支持 A8W8（32B=32 个数，峰值 2 倍），故可在"Vector 融合 descale"与"Cube 硬件 dequant（MXFP8 类）"之间按 scale 粒度选择。
- **规则：scale 跟随消费方驻留**——Vector 反量化则 scale 进 UB，Cube 反量化则 scale 与数据同层驻留 L1；禁止跨层搬运（如 scale 进 UB 却给 Cube 用）。
- **P 的 scale 反向回传**：量化路径下 P 也须量化后才能进 C2，其 scale 由 V1 在 UB 侧生成，随 P 一起写回 L1 供 Cube 消费；P scale 的大小计入 L1P 的分配（P 数据 + scale 一体）。
- 量化粒度谱系（由粗到细）：per-tensor < per-token-per-head < per-block（power-of-two）。粒度越细 scale buffer 越大、搬运开销越高；Cube 硬件反量化可把细粒度的计算成本归零，是细粒度可行的前提。

## S4 基本块算术强度修正

量化乘法的算力消耗与标准 fp16 / bf16 乘法不同，Roofline 计算通道的算术强度需按量化位宽修正，可能导致基本块取值改变。

实测修正规律（蒸馏自量产算子）：

- **计算通道峰值按量化对称性分情况**（Cube 每拍以 K 方向 32B 计：`16 × 32B × 16`）：
  - **非量化（fp16/bf16）**：32B = 16 个数 → 每拍 16×16×16 = **4096 MAC**。
  - **非对称量化（A16W8 / A16W4）**：activation 仍是 16-bit，K 方向 32B 仍是 16 个数 → 峰值不变（4096/拍）。量化只省 HBM/L1 字节，**不省算力**。
  - **对称量化（A8W8，双侧 8-bit，如 FP8 / HIF8 / MXFP8）**：32B = 32 个数 → 每拍 16×32×16 = **8192 MAC，峰值 2 倍**。ridge 点右移，打满 Cube 需要 2 倍算术强度。
- **HBM 通道 AI 按字节宽翻倍**：fp16(2B)→fp8(1B) 使 `AI_HBM ≈ 2·mBaseSize/sizeof(T)` 翻倍，且 K/V 驻留 L1 的容量压力减半——这与计算通道的修正独立，A16W8 同样享受。
- **只放大 s2BaseSize，不动 mBaseSize / D**：A8W8 场景下峰值翻倍 + AI 翻倍叠加，s2BaseSize 可从 128 放大到 256（Vector 反量化）或 512（Cube 硬件反量化，dequant 免费故可喂更大块）；mBaseSize 由 UB/ridge 决定、与量化无关，保持不变。
- **s2BaseSize 放大受 L0/UB 再约束**：放大后若 L0B / UB 放不下，在基本块内部再切子循环（sub-loop）控制驻留，而非回退 s2BaseSize；L0 某端口因此退化单缓冲时须显式标注为容量所迫的退化。

## 正确性约束

- **I5 量化 scale 轴对齐**：scale 沿被消费乘法的 reduction 轴量化，禁止轴错位（否则输出 NaN / inf）。

## 常见变体

- **MXFP8**：power-of-two scale，per-block 布局
- **INT8**：integer scale，per-tile 布局
- **AntiQuant**：反量化后走原路径 + scale 回传

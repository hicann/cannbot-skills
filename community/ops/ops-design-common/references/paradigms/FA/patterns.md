# FlashAttention 范式入口（语言中立）

针对 Attention 及其变种（GQA、MHA、MLA、量化、稀疏等）生成语言无关的 Tiling 设计方案。用于用户描述 attention 算子并要求产出 Tiling 设计或方案文档时。

进入本模块前核对实际公式中的分数计算、行归一化与加权值聚合。C1/V1/C2/V2 是相应计算图的一种阶段划分，具体排布由数据依赖、状态版本与目标资源推导。

芯片规格缺失时向用户补齐；无法提供则在方案中标注"待提供"，不猜测。

## 前置知识

先加载 [pre-knowledge.md](./references/pre-knowledge.md)，建立 FlashAttention 常识与硬件基础（数学公式、数据流四阶段、片上缓存层级、正确性约束 I1-I5）。

## 方案步骤

按顺序执行 S1→S7。每个步骤只加载该步骤所需的 reference（见"加载协议"）。

| 步骤           | 做什么                                                                  | 基础加载                                                                        |
| ------------ | -------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| S1 选型        | 定基础形态 + 正交 trait + Feature                                           | [selection.md](./references/selection.md)                                   |
| S2 单核计算流     | 定义单核内 stage 序列 + 数据流方向                                               | [compute-flow.md](./references/compute-flow.md)                             |
| S3 Buffer 规划 | 出公式化 buffer 清单（大小符号化 + 份数公式化，预取深度作参数；含 L0A/L0B/L0C pingpong 硬约束） | [buffer.md](./references/buffer.md)                                         |
| S4 基本块与性能建模  | Roofline 与逐阶段成本定候选块，分析两次矩阵计算、向量阶段及搬运，代入 buffer 大小 | [roofline.md](./references/roofline.md)、[stage-cost.md](./references/stage-cost.md)、[buffer.md](./references/buffer.md) |
| S5 流水编排      | 流水级数 + stage 顺序 + 同步语义 + 回填 buffer 份数                                | [pipeline.md](./references/pipeline.md)                                     |
| S6 负载均衡策略    | 多核切分 + 负载均衡                                                          | [load-balancing.md](./references/load-balancing.md)                         |
| S7 自检        | 跑 Self-Check 清单                                                      | [advanced.md](./references/advanced.md)                                     |

> **基本块与 buffer 的迭代闭环（S3↔S4↔S5）**：S3 出公式化 buffer 清单（大小、份数与生命周期）；S4 由 Roofline 与阶段成本定候选块并代入大小；S5 按实际流水、预取及最后读完成推导份数；以硬件内存（UB / L1 / L0）对 Σ(size×N) 校验。不满足则调整基本块、复用或排布并重新代入，直到收敛到可行的 tiling 块。

> **自检不通过**：定位失败项，回退到对应步骤重新决策（对应关系见 Self-Check），修正后重新自检，直至全部通过；未通过不允许产出方案。

## 加载协议（按需加载）

**基础加载**：S1→S7 依上表逐步加载对应文件。

**四阶段排布示例**：S5 采用 C1→V1→C2→V2 并交错推进多个 KV 工作项时，按需读取 [完整时序与状态示例](./references/pipeline-example.md)。其中的阶段偏移、队列和状态份数只对应所示排布，实际路径结合 S3、S4、S5 重新代入。

**特性加载**：命中下列特征时，在相关步骤额外加载对应 trait 文件：

| 特征  | 触发条件                                                         | 额外加载                                                        | 涉及的步骤                       |
| --- | ------------------------------------------------------------ | ----------------------------------------------------------- | --------------------------- |
| 量化  | Q / KV dtype ∉ {fp16 / bf16}                                 | [trait-quantization.md](./references/trait-quantization.md) | S4 算术强度修正 / S3 scale buffer |
| 稀疏  | KV 稀疏 pattern 由外部索引 / 掩码数据给定（规则掩码如 causal / 滑窗属 Feature，不触发） | [trait-sparse.md](./references/trait-sparse.md)             | S6 块跳过 / S3 不定长 buffer      |
| MLA | 数学公式含 latent absorption                                      | [trait-mla.md](./references/trait-mla.md)                   | S3–S6 全链路                   |

**样例加载**（arch35 / Ascend950PR / DAV_3510 量产算子蒸馏的完整 S1–S7 走查）：spec 芯片为 arch35 时按 trait 命中加载对应 best-practices，作为 S3–S6 的参考实现锚点（基本块取值、buffer 份数、流水深度、均衡手段均以样例为实测基准，roofline 推导用于解释而非推翻样例值）：

| 算子 | 形态 | 加载文件 |
| --- | --- | --- |
| flash_attn | 非量化 GQA（ND/DN 双模板） | [flash_attn-arch35-best-practices.md](./design_sample/flash_attn-arch35-best-practices.md) |
| quant_flash_attn | 量化 GQA（FP8 / HIF8 / MXFP8 三套） | [quant_flash_attn-arch35-best-practices.md](./design_sample/quant_flash_attn-arch35-best-practices.md) |
| sparse_flash_attention | MLA + token 稀疏（不切 G、Vec0 gather） | [sparse_flash_attention-arch35-best-practices.md](./design_sample/sparse_flash_attention-arch35-best-practices.md) |
| sparse_flash_mla | MLA + 稀疏双 trait（split-G、CSA/SWA 五模式） | [sparse_flash_mla-arch35-best-practices.md](./design_sample/sparse_flash_mla-arch35-best-practices.md) |

> 样例符号与规则文件一致（`mBaseSize` / `s2BaseSize`）。注意样例 kernel 侧存在 `mBaseSize_effective = mBaseSize × r_cv` 的生效值换算（UB 分摊口径），读 buffer 份数与容量校验时区分。样例值是 arch35 特定芯片（L1 512KB / L0A/B 64KB / L0C 256KB / UB 248KB）下的实测最优点，迁移到其他芯片时须重新走 S3↔S4↔S5 闭环推导，禁止直接搬运数值。

## 纪律

- 语言无关：方案中不出现任何编程语言、框架或硬件 API 符号，只描述切分维度、buffer 公式、并行策略、流水级数、同步语义等 Tiling 决策。
- 信息来源可信：芯片容量 / 核数 / 算力等参数未经确认时标注"待验证"，不编造。
- 设计即最优：每个决策（基本块、流水级数、并行度等）必须取 Roofline + 容量推导的最优点，可用优化手段有软流水 / 双缓冲 / 负载均衡 / 编译期分流；其中 Cube 侧 **L0A / L0B / L0C 恒 pingpong（各 2 份），单缓冲属退化配置、不接受**；低于最优须显式标注其绑定的硬件 / 容量约束；禁止"先跑通后优化"式性能推迟、把退化配置作为设计目标。
- **基本块必须是静态常量切分，套数由 spec 驱动**：mBaseSize / s2BaseSize / D / r 都是**编译期写死的常量**，**禁止输出运行时计算的动态块值**（如 `min(mBaseSize×G, cap)` 作为块大小）。**套数取决于 spec 是否给出场景典型值**——
  - **spec 未指定场景 / 未给典型值**：默认把 mBaseSize 取为 UB 容量允许下的最大值（目标 `AI_HBM ≥ ridge_HBM`，Cube 吃满），prefill 与 decode 归一为同一基本块常量。不为 decode 短行单设第二套块的原因：decode 的 HBM-bound 由 M 轴行数锁死、基本块改不动 AI，优化下沉到 S6 split-KV（任务分发层）解决。
  - **spec 给了明确的场景典型值**（如明确 prefill 主打 Sq=X、decode 主打 Sq=1）：可按典型值分**多套常量基本块**，每套针对一类典型 shape 调优，但每套内部仍为编译期写死常量、按 shape 属性静态分流，**不得退化为运行时动态块值**。
  - 运行时只允许"有效行数 ≤ 常量块、不满则 padding"，块本身不变。
- 设计期 vs 验收边界：本 skill 只产出设计期可校核的决策（结构决策 + 公式推导 + 容量校验）；实测验证（profiling 采集、反汇编检查、性能对标）属于性能验收阶段，不属于设计期。

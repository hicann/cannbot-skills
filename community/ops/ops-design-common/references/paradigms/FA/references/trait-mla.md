# MLA trait

> 触发条件：数学公式含 latent absorption。MLA 沿 KV 侧做 latent 压缩，kvHeadNum = 1。

## 核心影响

MLA 把标准 KV 投影成低维 latent，再做 latent absorption，影响全链路 S3–S6。

## 关键决策

- latent 压缩：KV 投影到低维 latent 的维度选择
- latent absorption：吸收进 Q 侧的矩阵乘
- 中间精度：latent 路径的中间段是否需要更高精度
- kvHeadNum = 1 对负载均衡（S6）和 buffer（S3）的影响

## 涉及影响点

### S3 buffer：D 超宽是全链路第一约束

- **latent 维 D_latent（如 512，或 nope 512 + rope 64 = 576）远大于常规 head dim**——单块 KV（`s2BaseSize × D_latent`）占 L1 的比例被 D 放大数倍，**K/V 槽池份数的上界由 D_latent 锁死**（D=512、s2BaseSize=128 时单块即 128KB，512KB L1 下槽池只能开 3 份，正好压在下界 r+1 上）。
- **D 不进基本块常量**：基本块仍写死 `mBaseSize × s2BaseSize`，D 方向由固定的子块宽（如 128）在 GEMM 内部迭代消化；Q 沿 D 切半/分段多次载入。
- rope 拆分与否是**模板级分流**：带 rope（D=576）与不带（D=512）按模板参数静态分开，rope 维作为独立常量，不混入主 D 的动态计算。

### S6 负载均衡：kvHeadNum=1 使 G 维成为唯一杠杆

- I2 合轴后 M = Sq × G 中 **G 可极大（全部 qHead 压在一个 kvHead 上，G 可达 128）**——G 既是 AI 的放大器，也是任务粒度失衡的来源。
- **split-G**：G 超过阈值（如 64）时把 G 对折进 M 维、由相邻核分摊同一 M 分块的 G 前后各半，以核数换任务粒度——这是 MLA 特有的均衡手段，GQA 不需要（G 小）。
- **split-KV（FlashDecoding）仍是 decode 短行的承载点**：kvHead=1 时 totalTasks = B × M分块 更少、核更富余，split-KV 判据（`totalTasks < usedCoreNum` 且 Sk 够长）更容易命中。
- 与之正交的另一路线：**不切 G、每个基本块只处理一行 query**（逐行 M=1 任务）——任务是够细了，但放弃 G 合轴的 AI 增益，仅当 G 维无法合轴共享 KV（如逐行独立稀疏索引）时才选。

### S4/S5 联动

- D 超宽不改 AI 公式（`AI_HBM ≈ 2·mBaseSize/sizeof(T)` 与 D 无关），但**绝对耗时中 D 方向迭代次数翻倍**，Cube:Vector 的节拍比须按 `D_latent/128` 折算校核 cross-core-bound。
- 流水级数与常规 FA 相同（r=2、槽池 ≥3）；D 超宽挤压的是槽池**上界**，不是流水结构本身。

产出：latent 维度 / absorption 位置 / 中间精度选择。

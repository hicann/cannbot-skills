---
name: vision-mlp
description: 视觉MLP类算子（WeightedPermuteMLP 等 ViP/Permutator 加权置换 MLP）的 Triton Ascend 优化经验合集，按算子分章节组织，含通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 视觉MLP类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：跨算子重复的工程约束（已提取，各算子章节不再重复；与其他类别共用的通用约束见 `tensor-transform.md` G1-G8 / `transformer-inference.md` T1-T6 等，此处只写本类别特有的）
- **§2 65_WeightedPermuteMLP**（ViP 加权置换 MLP）
- **§3 常见陷阱**（按算子分小节）

> ⚠️ **关键区分**：本类别的核心优化哲学是 **"host 只做布局路由，device 侧模式特化转置 + 纯 GEMM 分带 tile + 小 softmax 消费端内联"**。生成时**禁止混用**其他类别的经验（如不要套用 attention 的在线 softmax 分块、不要套用搬运类的逐元素 gather）。

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| 65_WeightedPermuteMLP | `vision-mlp` | x(B,H,W,C) 三路 Linear（h/w 路需 permute 路由）→ mean→fc1→GELU→fc2→softmax(3) 重加权组合 → proj | 特化转置 kernel + 纯 GEMM slim-tile 分带 + 按 B 分路的 fc2 + softmax 内联消费端 |

---

## §1 通用经验（跨算子，首次生成必须遵守）

以下约束是本类别算子**共有**且**未在其他 template 文件覆盖**的工程约束。其他文件已提取的通用约束（如 `tensor-transform.md` 的 G1 动态 num_cores / G2 pow2 BLOCK / G4 grid 不超核数 / G7 contiguous 等）此处不再重复，各算子章节引用时标注。

### V1 禁止在 GEMM k-loop 内做 gather/加权 prologue
- **禁止** 把逐元素加权组合（combine 类）折进下游 GEMM 的 A-tile load 端。
- **Why:** Cube kernel 的输入 tile 必须连续；gather 加权 prologue 比 "独立向量 kernel + 纯 GEMM" 慢 13x（实测同规模 GEMM 0.086ms vs 融合 1.128ms）。
- **典型应用**：三路 embed 加权合并后再 proj 的结构。
- 与 `tensor-transform.md` G7（contiguous）的差异：G7 管输入布局，本条管**计算融合位置**——即使输入连续也不许在 k-loop 里做加权。

### V2 禁止 store 侧置换折叠
- **禁止** GEMM 输出直接按置换偏移 scatter store。
- **Why:** Cube 路径上 computed-offset scatter 是灾难级劣化（实测 geomean 掉到 0.0681）。
- **典型应用**：GEMM 输出需要 permute 回 (B,H,W,C) 的场景。

### V3 禁止向量整数除法做索引分解
- **禁止** `offs // N`、`(x//C)*C` 作用于 tensor；块索引分解只在标量 program_id 上做（`i = pid - (pid//n)*n`）。
- **Why:** 向量整除触发标量降级 7.7x。

### V4 host `.permute().contiguous()` 必须替换为模式特化转置 kernel
- **必须** 每种置换模式独立 kernel（tile-based 连续访存）；同 grid 的转置合并为单 kernel（双输出指针或展平 grid + 标量 if/else）。
- **Why:** aclnn Transpose 每个 8-13μs device 时间，小 shape 下占 66%。特化 kernel 单次 4-5μs。

### V5 UB 预算按活 tile 数推导
- **必须** 向量 kernel 活 tile 数 × 单 tile 字节 ≤ 1572864 bits。
- **How:** 7 个活 tile（3 输入 + 3 权重 + 1 store）时单 tile ≤ 32×128 fp32。

### V6 隔离微基准 ≠ in-situ
- **必须** tile/融合决策以 in-situ 全 forward A/B 定夺；隔离扫描只筛候选。
- **Why:** 隔离循环里操作数 L2 常驻、同一 kernel 连跑多次；in-situ 每个 kernel 只跑一次、L2 由前序 kernel 决定（本项目三次隔离赢面 in-situ 回退）。

### V7 交错读的优化出路是"置换生产者"，不是"改读法"
- **必须** 下游 kernel 需按分支维度切分读取上游 GEMM 交错输出时，把上游权重/bias 列在 host 构建期一次性置换为 branch-major，使消费端变连续读。
- **禁止** 消费端 stride-3 向量 load（实测劣化 3x：3.3→15.2μs）。
- **Why:** 置换在权重构建期完成，运行时零开销（65_WPM opt_iter_31: sm3 10.8→4.0μs）。

### V8 device kernel 时间是评分口径
- **必须** 所有融合/分裂决策按 device 时间实测，不按 launch 数。
- **Why:** benchmark 只计 device kernel 时间，launch 开销不计入。

---

## §2 65_WeightedPermuteMLP 算子（ViP 加权置换 MLP）

**算子类别**: `vision-mlp`
**典型特征**: x (B,H,W,C) fp32，B∈{1,2,4}，C∈{64,128,256,512}，H,W∈{4..11}；seg_dim 经 resolve 后多数为 1；hidden=C//4
**性能基准**: 50/50 pass，几何平均加速比 **2.3449x** vs torch eager（基线 0.2463 → 9.5x）

### §2.0 首次生成必读：为什么必须把主要框架写对

本算子是 "6 GEMM + 2 转置对 + mean/softmax/combine" 的复合结构，首次生成最容易写偏的两处：
1. **把 combine 折进 proj GEMM**（违反 V1）——首版 geomean 0.0681，后续所有调优都救不回架构性劣势；
2. **host 侧 permute 保留 aclnn Transpose**（违反 V4）——小 shape 下 66% 时间花在搬运。
框架必须第一步就是：特化转置 + 独立向量 combine + 纯 GEMM 分带。

### §2.1 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

#### L1.1 分块参数上界按最坏维度推导
- **必须** S=C/seg_dim 的 clamp 上界取 C（seg_dim 可被 resolve 到 1，S 可达 C）。
- **Why:** BLOCK clamp 不足 → tile 静默半覆盖，只炸大 C case（精度错且难定位）。
- **How to apply:** 见 L3.1 `_tp_fwd_kernel` 的 BLOCK_S。

#### L1.2 mean 分解与权重角标顺序
- **必须** fc1 输入为多分支之和的 mean 时用 `mean((c+h+w)) = mean(c)+mean(h)+mean(w)`；权重角标按参考实现核对（c*w0 + w*w1 + h*w2 类映射容易写反）。

#### L1.3 GELU 用 tanh 近似
- **必须** tanh 近似（与 NPU nn.GELU golden 位对齐），**禁止** erf 精确式。

#### L1.4 权重复刻用 hash 派生 seed + RNG save/restore
- **必须** 模块级缓存权重，seed 由 `hash((dim, seg_dim, H, W, qkv_bias, proj_drop)) & 0xFFFFFFFF` 派生，外层 `torch.get_rng_state()/set_rng_state()` 保护；kaiming_uniform_ bound 公式照抄（`std = gain/sqrt(fan_in)`，`wbound = sqrt(3)*std`）。
- **Why:** 验证框架以隐藏状态探针比对逐位权重，公式推导对但顺序/dtype 不对即全错。

#### L1.5 禁止 torch.zeros 预清零走 aclnn ZerosLike
- **禁止** split-K/atomic 路径的 `torch.zeros((M,N))`（每次派发 1.2-2.3μs 的 ZerosLike device kernel）。
- **必须** split_k 只在 `nblocks*10 <= ncores and K >= 512` 这类严重饥饿且足够大时启用；否则宁可用 `torch.empty` 单 program 路径。

#### L1.6 fc2（小 M branch GEMM+softmax）按 B 分路
- **必须** B=1 保持融合单 kernel（fc2sm 直接产 wgt）；B≥2 一律拆为 K1 GEMM (B,3C,hidden) 产 branch-major y2 + softmax 内联进 combine。
- **Why:** 融合 kernel 在 B*num_cb programs 上有 ~2.9μs/program 的 latency floor（B=2/C=64 时 11.7μs vs 拆分 ~5μs）；但 B=1 两跳 launch 不划算。
- **How to apply:** 见 L2.2 分派决策树 + L3.3。

#### L1.7 转置 kernel 的 grid 必须填核
- **必须** tp_fwd 按输出行并行 `(B*SEG*(W+H),)` 每 program 单 tile 拷贝。
- **禁止** `(B*SEG,)` 每 program 串行 H+W 次拷贝（2-4 program，20 核闲 18 个）。
- **Why:** 串行版 6.9-14.5μs，行并行版 3.8-4.5μs。
- **How to apply:** 见 L3.1。

#### L1.8 小 softmax 可内联进纯向量消费 kernel
- **必须** softmax(3) 每列仅 3 元素、消费 kernel（combine 类）本就要按列加载这 3 个权重时，直接内联计算（max/exp/sum 就地），省 1 次 launch + 中间 wgt 张量往返。用 constexpr 分支兼容 "生产者已直接输出权重" 的旧路径。
- **Why:** 65_WPM 实测 combine 4.3→4.9μs 但消除 sm3 10.8μs，净 -10μs/case，geomean +17.5%。
- **禁止** 内联分支的列索引用块内相对索引再叠加块号——必须用**全局列索引**（如 rc）。曾因 `cs` 已含 `i_cb*BLOCK_CN` 又叠加一次导致 OOB 读（数值发散 0.67）。
- **How to apply:** 见 L3.3 `_combine` SM3_INLINE 分支。

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧布局路由 + 数据流总体设计

**权重构建（一次性，缓存复用）**：复刻参考实现的 RNG 派生顺序；fc2 权重按消费路径准备两种布局——B=1 融合路用 branch-major 三平面布局，B≥2 GEMM 路用**列置换为 branch-major 的转置布局**（置换在构建期完成，运行时零开销）。

**forward 数据流（8 个 device kernel）**：

1. **tp_fwd 模式特化转置**：x → (xh, xw)。把 h/w 路需要的两个置换合到一个 kernel、双输出；**grid 按输出行并行**（每个 program 拷贝一个 tile），填满 20 核。
2. **三路投影 GEMM**：c 路直接用 x 的 2D 视图；h/w 路用转置结果。全部是纯 GEMM、连续 tile，按 M/N/K 分带选 tile（见 L2.2）。split_k 只在"输出 tile 数远小于核数且 K 足够大"时启用。
3. **tp_inv 逆转置**：h/w 路 GEMM 输出转回 (B,H,W,C) 嵌入布局，同样单 kernel 双输出。
4. **fc1（mean→Linear→GELU）**：利用 mean 可分解性——每列的 mean 只依赖该列的三路 embed，无需先物化 mean 向量；按 (b, k-chunk) 切分原子累加（并行度足够时）或整批单 kernel（B=1 小 C 时）。
5. **fc2 按 B 分路**：B=1 融合单 kernel 直接产出三平面 softmax 权重 wgt；B≥2 走纯 GEMM 产 branch-major logits y2，**softmax 不落盘**。
6. **combine + 内联 softmax**：B≥2 时在 combine kernel 内从 y2 就地算三路 softmax 权重再加权合成（constexpr 编译期分支）；B=1 从 wgt 平面加载。
7. **proj GEMM**：标准带 bias 纯 GEMM。

#### L2.2 GEMM tile 分带原理与 split_k 门控

**分带依据**：h/w 路径的 GEMM 是 "M 很小（≤48）、N=K 可达 5632、权重矩阵可达 127MB" 的 **slim 访存受限型**——性能由权重流的带宽/并行度决定，而非 cube 吞吐。因此 tile 选择的核心目标是**让 N 维产生足够多的输出 tile 去填核**，同时保持单 tile 的权重访问长连续：

- **M≤20（slim 带）**：bm=32 固定；bn 随 N 增大逐级放宽（小 N 用 64 换 tile 数，大 N 用 128/256 换连续性）；bk=256 拉长每次权重读。
- **M≤48 且 N≥512（中带）**：bn=128 主力（N≤2048）或 bn=256（N≥2560 时 tile 数已够）；bm 在 N 最大档升到 64。
- **其余（fat 带：c 路 / proj，M≥100）**：输出 tile 本来就多，回到常规 clamp 策略，bn 收到 128 保 UB。

**split_k 门控原理**：原子累加路径必须先清零输出（见 L1.5 的 ZerosLike 代价），只有当"输出 tile 数 ×10 仍 ≤ 核数"（即严重饥饿）**且** K≥512（切分后每段仍有足够工作量）时，用清零代价换并行度才划算。

### §2.3 Layer 3: 关键算法设计与原理（优化重点与易错点）

#### L3.1 模式特化转置（tp_fwd / tp_inv）

**设计原理**：host 的 `permute().contiguous()` 会派发 aclnn Transpose（8-13μs）；本算子的两个置换模式固定，可推导出精确的源/目标地址映射，用 tile 拷贝实现（4-5μs）。

**并行化设计**：grid 不按 (B, SEG) 组织（那会让每个 program 串行做 H+W 次拷贝，20 核闲置 18 个），而是**按输出行展开**——grid = B×SEG×(W+H)，每个 program 恰好拷贝一个 (行 × S) tile。kernel 内用**标量 pid** 与阈值 NWH 做两路分支（oh 侧/ow 侧），分支内的索引分解全部在标量上做（避免向量整除降级）。

**易错点**：
- tile 的 BLOCK_S 上界必须按 S=C 的最坏情形取（seg 可能被 resolve 成 1）；
- 目标地址的每个维度步长（含 w×SEG×S 这类复合步长）必须逐一对照 torch permute 单测，漏一个维度 B>1 全错。

#### L3.2 combine 向量 kernel 的块索引分解

**设计原理**：combine 是纯向量 kernel，输入三路 embed + 三路权重、输出一个合成矩阵。持久化 grid 下每个 program 循环处理多个 (行块, batch, 列块) 线性编号，需把线性编号分解回三维。**分解必须全部在标量块编号上用 `减法+乘法` 实现（`i = bi - (bi//n)*n` 形式），且分解顺序是 列块 → 行块 → batch**（除最后一次外都不能跳层）——分解顺序写错（如先除行块数）会得到错位索引，数值全错但 shape 不报错。

#### L3.3 权重列置换 + softmax 内联消费（opt_iter_31 核心）

**问题本质**：fc2 输出的三路 logits 按列交错存放（每通道 3 个 branch 相邻）。下游若按 branch 切分读取就是 stride-3 交错 load——在 Ascend 向量核上劣化 3x。

**解法一：置换生产者**。与其改消费端读法，不如把 fc2 权重/bias 的**列序**在 host 构建期重排为 branch-major（三个 branch 各占连续的 C 列段）。GEMM 是"列序无关"的——重排列序后输出 y2 自然按 branch 分段连续，下游读全连续。置换成本为零（构建期一次性），收益在所有 B≥2 case 上兑现（sm3 10.8→4.0μs）。

**解法二：小 softmax 内联消费**。softmax 只在每列 3 个元素间归一化；而 combine kernel 本来就要按列加载这三个权重。因此把 max/exp/sum/除法就地内联进 combine（编译期 constexpr 开关），权重即算即用——省掉独立 sm3 kernel 的整次 launch + wgt (3,B,C) 中间张量的写/读往返。combine 自身略涨（多 3 组向量化 exp），但净省约 10μs/case。

**易错点（曾致数值发散 0.67）**：内联分支的列地址必须由**全局列索引**一步构造（该索引天然含列块偏移）；若再用"块内相对索引 + 手工叠加块号"会重复叠加偏移，末 batch 的尾部块读到越界地址。此类 OOB 定位手段：用 torch 按 kernel 的地址公式逐块重放，首个越界块即暴露。

**适用边界**：branch 数 >3 或每列归一化元素多时，内联的向量开销可能超过省下的 launch——需 in-situ A/B 决定；生产者已直接输出 softmax 权重的路径（如 B=1 融合 fc2sm）保持加载模式，不做内联。

### §2.4 65_WeightedPermuteMLP 性能基准

| 维度 | cases | 加速比区间 | 备注 |
|------|-------|-----------|------|
| B≥2 & C=256/512（sm3 内联受益类） | 30+ | 1.93 ~ 3.23 | opt_iter_31 主增益 +0.6~+1.0/case |
| B=1（融合 fc2sm 路径） | ~10 | 1.54 ~ 2.1 | 不受内联影响 |
| C=512 seg=1 大 H/W（带宽墙类） | 3 | 1.54 ~ 1.66 | 84-127MB 权重 DRAM 流读，结构性 |
| 全量 | 50 | **2.3449x**（geomean） | 40/50 ≥ 2.0x，达标 |

**关键结论**:
1. 优化过程：0.2463 基线 → 1.1637 首轮 → 1.5066 → 1.9955 → **2.3449**（32 轮 opt_iter）
2. 结构性上限：C=512 大 H/W 的 h/w 权重 84-127MB 以 1.7-1.9TB/s DRAM 流读——带宽墙；(288,288) 类小 N GEMM tile 在 in-situ 不敏感（±1μs）
3. 剩余瓶颈：K1 GEMM MTE2 52% / MMAD 11%（msprof simulator 实测）；手写 k-loop prefetch 被 BiShengIR 阻断（cbuf memref 动态/静态类型不匹配），编译器限制暂无修复路径

---

## §3 常见陷阱与避免方法

### §3.1 65_WeightedPermuteMLP 陷阱

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 融合 gather-GEMM（k-loop 内加权 prologue） | Cube tile 必须连续，13x 劣化 | V1：独立向量 kernel + 纯 GEMM |
| store 侧置换折叠 | Cube computed-offset scatter 灾难级 | V2 |
| 向量整除索引 | 标量降级 7.7x | V3：标量 pid 分解 |
| host permute 派发 aclnn Transpose | 小 shape device 时间 66% 是 Transpose | V4：特化转置 kernel + 两两合并 |
| BLOCK_S clamp < max(S)（seg→1 时 S=C） | 只炸 C=512 case，静默半错 | L1.1：clamp 上界 = C；单测用 resolve 后的 seg |
| 转置 dst 漏某维 stride（如 w*SEG*S） | B>1 全错 | 每个转置 kernel 逐一对照 torch permute 单测 |
| k-loop 手写 prefetch（a=a_next） | BiShengIR scf.for 类型不匹配编译错 | 本编译器版本不可行 |
| 合并 kernel 漏传分支维度（B） | NameError 编译错 | pid 分解用到的维度全部传参 |
| 合并 kernel 单输出指针 | 下游要独立指针时 NameError | 双输出指针 |
| split-K 的 torch.zeros 预清零 | 每次 1.2-2.3μs ZerosLike，叠加 10-16% | L1.5：收紧 split_k 门控 |
| 融合 fc2+softmax 用于 B≥2 | latency floor 2.9μs/program，B=2/C=64 时 11.7μs | L1.6：B≥2 拆 GEMM，B=1 保持融合 |
| 转置 kernel grid=B*SEG（串行 H/W 循环） | 20 核闲 18 个，6.9-14.5μs | L1.7：按输出行并行 grid=B*SEG*(W+H) |
| 隔离微基准 tile 直接上线 | L2 常驻假象，in-situ 回退（三次教训） | V6：隔离只筛候选，in-situ A/B 定夺 |
| 读 kernel_details.csv 前几行当稳态 | 首 rep 冷启动值虚高 5-10x | 稳态看 groupby(Name).sum()/reps |
| softmax 消费端 stride-3 交错 load | sm3 kernel 7.6-12.8μs/case（占类时长 15-25%） | V7：上游权重列置换 branch-major |
| 内联分支列索引用块内相对索引再叠加块号 | OOB 读、数值发散 0.67（b 末块越界） | L1.8：全局列索引一步到位；torch 逐块模拟可定位 |

---
name: sparse_mask
description: Sparse-掩码型注意力算子（窗口局部 / 块对角 / 元素周期 stride / 复合 window∪global / 轴向分解 / causal / padding / topk-selected KV / 结构化加性偏置）的 Triton Ascend 优化经验。以"掩码或稀疏在哪一层生效 + 掩码几何形态"为核心路由轴，按算子分章节，含通用经验 + 各算子专属杠杆/陷阱/证伪。已落地成员：op14 Window / op23 Local / op24 BlockSparse / op25 Strided / op51 Longformer / op37 BigBird / op35 Axial（0818，2.4388）。online softmax / UB 预算 / fp32 契约 / 投影 GEMM 等共享机制继承自 flash_attention.md，本文件只写 delta。
metadata:
  type: reference
---

# Sparse-掩码型注意力算子优化经验

> **本类算子**：attention 的 score 矩阵**不是**朴素稠密 `Q@Kᵀ`，而是被**结构性调制**——掩码（causal / padding / 窗口带 / 块对角 / 元素周期 / 复合 / topk 的 `-inf` 门控）、**结构化加性偏置**（相对位置偏置 / ALiBi），或**结构局部化**（窗口分区、轴向分解）。共同点：score 模式有**可利用的结构**。
>
> ⚔️ **与 [`flash_attention.md`](flash_attention.md) 的分工**：本类算子**底层仍是 online softmax + KV 分块**，FA 的 §1（F1 ghost 列 / F2 有限极小值 / F3 ceil16 / F4 UB 预算 / F5 改结构后重扫 / F6 大中小三档）、§2 Layer 1（L1.1 tl.dot 契约 / L1.3 online softmax 同步缩放 / L1.4 精度契约 / L1.13 BLOCK 面积预算 / L1.14 propagate_nan 分档 / L1.16 逐步舍入回输入 dtype / L1.17 strided 索引掩码 507015）、§4（§4.3 BLOCK 推 UB / §4.4 权重重排 / §4.5 投影分块+并行度保护 / §4.7 CV 重排）、§5.2 证伪表**全部继承，不在此重复**。本文件**只写 delta**：掩码 / 稀疏 / 结构化偏置带来的**额外**约束、杠杆与陷阱。
>
> **判别顺序**：按本文件 §0.1 确认是否为 Sparse-掩码型注意力。满足任一特征 → 用本文件。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足以下任意一条 → 用本文件（Sparse-掩码型注意力优化经验）：

1. forward 含 `mask` / `attn_mask` 参数，或 score 注入 `-inf` / 有限极小值门控（causal / padding / 自定义布尔掩码）；
2. score 叠加**结构化加性偏置**（相对位置偏置 `rel_bias`、ALiBi、可学习 bias），形状/索引由几何或学习参数决定；
3. attention **结构局部**：token 分进窗口/块，只窗口/块内 attend（Swin 窗口、局部滑窗），或 **Axial 轴向分解**（2D 局部化被消费为两次 1D 稠密 attention）；
4. KV 由 `indices` / `topk` / block-sparse pattern 选择性参与（只算被选中的 key/value）；
5. score 掩码为**元素级周期**（`j%stride == i%stride` 残基类掩码，op25）；
6. **复合掩码**：多种成分叠加（窗口带 ∪ 全局行/列 ∪ 随机 gather，op51 Longformer / op37 BigBird）。

### §0.2 ★ 掩码/稀疏的「生效层」三分（**本类核心路由，Phase 2 第一步必须定位**）

掩码/稀疏的**优化杠杆完全取决于它在哪一层生效**。先定位，再选杠杆——**攻错层 = no-op 或回归**（op14 连续两轮回归的根因，§3.1-R1/R2）。

| 生效层 | 含义 | 典型算子 | 可用杠杆 | 不可用杠杆 |
|--------|------|----------|---------|-----------|
| **(a) host 侧结构稀疏** | 稀疏在算子边界外已被消费（窗口预切分 / 轴向 reshape），kernel 看到稠密**小 N** | 单窗口 WindowAttention（op14）、AxialAttention（op35，待跑） | 投影 GEMM + attention tile 推 UB + 小 N launch 开销意识 | ❌ 块跳过 / mask 特化 / KV 区间收缩 |
| **(b) kernel 内 score 掩码/偏置** | 掩码（`-inf`）或加性偏置在 kernel 内、softmax 前注入 score | causal / padding / rel-bias / ALiBi / 真·shifted-window；窗口带（op23）、元素周期（op25）、复合 window∪global（op51、op37） | mask 注入精度对齐、mask 分支 constexpr 特化、**带区间收缩**（op23，kernel 内块跳过）、加性偏置 gather 路径、**复合成分分解 OR 合并**（op51，M10） | ❌ 把加性偏置当掩码"跳块" |
| **(c) 块级跳过** | 稀疏 pattern 决定**哪些 QK 块根本不算** | 块对角小块（op24）、topk-KV | program=掩码块粒度、**单 tile 一次性 softmax**（M8）、块边界对齐 tile、索引 gather | ❌ 朴素稠密 tile（白算被掩块） |

> **op14 = (a) + (b) 的加性偏置子集**：窗口稀疏 host-batched（N=Wh·Ww），rel_bias 是 kernel 内加性注入。故制胜靠 (a)/(b) 杠杆（投影 + tile 推 UB），**不是 (c) 块跳过**——这决定它停在 2.2×、以及哪些方向是死路。

**已落地成员速查**：

| 算子 | 生效层 | 掩码几何 | 对角保底（M9） | 成绩（geomean，50/50） |
|------|--------|----------|:---:|------|
| op14 WindowAttention | (a)+(b) 加性 | host 窗口 + rel_bias | 免（加性无 `-inf`） | 2.1968（旧跑）→ 0817 复跑 3.3972→**4.0054** |
| op23 LocalAttention | (b) 门控 | 窗口带 \|i−j\|<w | ✅ 含对角 | 0815 首版 **3.2810**；0817 复跑 2.7557 |
| op24 BlockSparseAttention | (c) | 块对角，block≤31 | ✅ 块内全 attend | 0815 首版 **3.6387**；0817 复跑 2.6048 |
| op25 StridedAttention | (b) 元素 | j%s==i%s | ✅ 含对角 | 0815 首版 2.4029；0817 2.4683→**3.4325** |
| op51 LongformerAttention | (b) 复合 | 带 ∪ 全局行列 | ✅ 自窗口保底 | 0.4114→**2.7685**（6 轮 Phase4） |
| op35 AxialAttention | (a) | 轴向分解稠密小 N | 免（无掩码） | 0818 重跑新标杆（标准 scale）iter_2 首过 **2.4269**，#12 分档后 **2.4388**（50/50，NPU5） |
| op37 BigBirdAttention | (b) 复合+随机 | 带 ∪ 全局 ∪ random gather | ✅ 自窗口保底 | 0817 iter_2 首过即 **67.0198**（host mask cache 通吃，Phase 4 数值锁定零优化） |

> 0815↔0817 复跑差异（±10~30%）为芯片间 framework 侧波动，优化判定一律用 impl_ms 紧邻 A/B（[[feedback-crossattn-honest-benchmark]]，§6）。

### §0.3 形态补充四问（在 FA §0.2 四问之上追加）

| # | 问题 | 影响 |
|---|------|------|
| **M-Q1** | 掩码/稀疏在哪层生效（§0.2 a/b/c）？ | 决定整篇适用章节；答错攻 no-op |
| **M-Q2** | 掩码固定还是运行时可变？ | 可变 → mask 分支 constexpr 特化 +（causal/带）KV 区间收缩是大杠杆（FA §3.2/§5.1-#4）；固定 → 这两笔拿不到 |
| **M-Q3** | score 调制是**门控**（`-inf`，改 softmax 分母）还是**加性偏置**（不改分母）？ | 门控须处理 ghost 列/整行被掩（§M3、§2）；加性偏置不需要，但精度仍须对齐 softmax 域（§M2） |
| **M-Q4** | 掩码是否**含对角/自窗口保底**？ | 是（窗口带 / strided / 块对角 / longformer 自窗口）→ 每行至少一个有限 score，**豁免整行被掩 fallback**（M9）；否（纯 padding 全掩行等）→ 必须 fallback |

---

## §1 通用经验（跨成员，首次生成必须遵守）

> 本类**特有**、FA 未覆盖。FA F1~F6 / L1.1~L1.17 不重复。

### M1 ★ 先定位「生效层」，再选杠杆

- **必须** 写 kernel 前按 §0.2 确定 a/b/c。
- **禁止** 对 (a) host-batched 稀疏算子套 (c) 块跳过/mask 特化——稀疏已不在 kernel 内，no-op；去动偏置反而回归（op14 opt4 −5.8%）。
- **Why**：op14 窗口稀疏 host-batched 后 kernel 内是稠密小 N；连续两轮"再榨掩码/偏置"（opt3/opt4）全回归。攻错层是本类最大浪费。

### M2 掩码/偏置注入精度必须与 softmax 归约域对齐

- **必须** 掩码值（有限极小值，FA F2）与加性偏置都在 **softmax 累加域**（通常 fp32）注入，再进 online softmax。
- **禁止** 低精度注入 `-inf`/偏置后升精度；禁止偏置降到与累加域不同 dtype 再相加。
- **Why**：op14 参考全程 fp32，rel_bias 须 fp32 域相加才 50/50；降 fp16 相加则 dtype 转换+行对齐劣化抵消流量（opt4 −5.8%）。门控掩码同理：低精度 `-inf` 经 cast 可能变 NaN。
- **fp16 特例（op37 实测）**：参考链本身在 fp16 域（NPU `F.softmax(fp16)` 全链）时，注入与归约**必须在 fp16 域逐位复刻**，而非笼统升 fp32——完整三连见 M11。

### M3 门控掩码的"幽灵列/整行被掩"必须处理（**加性偏置不触发**）

- **门控型**（`-inf`）必守：① 被掩 key 在分母贡献为 0（继承 FA F1：`sum` 前 `tl.where(kv_valid, p, 0.0)`，或利用 `exp(NEG−m_new)` 精确下溢为 0 省掉这次 where，见 §3.4-S2）；② **整行被掩** → `softmax(全 -inf)=NaN`，须 fallback（该行取 `mean(V)` 或零）——**M-Q4 含对角保底者豁免**（M9）。
- **幽灵列**：masked load `other=0.0` 的行/列 score=0，**会混过窗口带/残基类判定**（op23 实测），keep 掩码必须显式 `& (offs_kv < S)`。
- **加性偏置例外**：op14 的 rel_bias 加性、无 `-inf`，每 key 都 attend，**不触发** M3——这是区分 op14 与真掩码算子的关键边界。
- **验证集**：门控算子主动构造"整行被掩"case（M-Q4 保底者改构造 ghost 列 case）；加性偏置算子构造偏置量级远大于 score 的 case（考注入精度）。

### M4 tile 策略必须与掩码/稀疏结构协同

- **(a)/(b) 稠密窗口内**：直接把 attention tile 推到 UB 上限（FA §4.3/L1.13），按 D 分档——op14 夺冠靠这条（+23%）。
- **(b) 带区间收缩**：BLOCK_KV 不必对齐窗口边界，带内逐块扫即可（op23）；BLOCK_Q 对齐无要求但决定收缩粒度。
- **(c) 块稀疏**：tile 边界**必须对齐稀疏块边界**，否则每 tile 内 mask 分支爆炸、UB 翻倍编译失败（FA §5.2）。block_size ≤ 31 时直接 program=单块、BLOCK_M=32 覆盖（op24，M8）。
- **禁止** 朴素稠密 tile 算块稀疏——白算的被掩块吃掉稀疏红利。

### M5 小块 / 小 N 的 launch 开销地板

- 含 softmax 的 kernel 在 triton-ascend 下必编成 `MIX_AIC`，每次发射 ~4.5µs 固定成本（FA §5.3）。
- 本类常是**小 N**（窗口 N=Wh·Ww、轴向 N=H/W、op51 S∈[13,263]）或**多小块**（块稀疏），最弱 case 易被 launch 钉死。op14 N∈[49,361]，最小 case 仅 1.27×；op51 S=13~43 的 case 4~5× 而 S≥128 且 D≥256 的 fp32 case 掉到 1.0~1.7×（计算主导，ieee dot 上限）。**小 case 上不要拆 kernel**。

### M6 ★ 门控掩码 kernel 内实现形态四分（按掩码几何选，Phase 2 草图必须显式二选一以上对比）

| 掩码几何 | 实现形态 | 代表 | 关键 |
|---------|---------|------|------|
| **带/窗口型** | KV 区间收缩：`kv_lo=max(0, q_start−(w−1))`、`kv_hi=min(S, q_start+BLOCK_Q+(w−1))`，带外整块不 load | op23 | 块级跳过的 kernel 内形态，稀疏红利直接 |
| **块对角+小块（≤32）** | program=单掩码块，只 load 本块行，**单 tile 一次性 softmax**（M8） | op24 | KV 循环整个消失 |
| **元素周期型** | dense KV 扫描 + 元素算术掩码（BLOCK≥stride 时每 KV 块含全部残基类，**无法块跳过**） | op25 | 掩码用 `a−(a//b)*b`，禁 `%` |
| **复合型** | 各成分**分别算术生成后 OR 合并**为单个 keep，一次 `tl.where` | op51 | 单 vsel 约束（M10） |

- **禁止** 对元素周期型套带区间收缩（残基类散布全行，收缩后仍须逐元素判定，白忙）。
- Phase 4 再考虑更激进的形态变换（如 op25 残基类 pack），先做微基准——已证伪（§5）。

### M7 ★ 掩码算术的编译器毒性四律（BiShengIR，op51/op25 实测，codegen 层硬约束）

1. **i32 2D LT/GT/LE/GE 比较标量降级 10~100×** → 2D 窗口/带比较**转 fp32 比较**（op51 opt_iter_2，+6.6× 最大单笔）。1D mask 与 EQ 比较保持整型（全量转换反而触发律 2）。
2. **1D mask 的 sitofp+fcmp IR 模式确定性触发 root-alloc 编译失败**（22/50 同指纹）→ pad/区间指示改用 **vmin/vmax 纯算术 clamp** 生成（`pad_f = tl.minimum(tl.maximum(offs−S+1.0, 0.0), 1.0)`），排除动作改为 `s −= pad_f*1e9`。
3. **mask 一旦被用作张量**（`.to(f32)` / 第二次 `tl.where` 消费）→ 物化 i64 标量比较循环（64 次/tile，last_pass.mlir 实锤）；**仅作 load/store mask 时走 subview 边界零开销**。设计时让每 1D mask 只有一种消费方式。
4. **模运算禁 `%`**，用 `a−(a//b)*b`（checklist）；**strided 行索引向量禁参与掩码/寻址算术**（FA L1.17，aicore 507015，op25）——用 dense `qi` + `qi < sr` 等价判定规避。

### M8 单 tile 一次性 softmax 退化（块稀疏/小块算子的最大红利）

- 当每个 program 的 KV 全集能被**单个 tile** 装下（op24 block_size≤31 < BLOCK_M=32；op35 轴向 N=H/W 小），**online softmax 退化为单 `tl.dot` + 一次 max/exp/sum**：无 m/l/acc 滚动状态、无 alpha 重缩放、无 KV 循环。
- **机制**：online softmax 的全部复杂度来自 KV 分块；单 tile 时它纯亏。op24 首版 3.64× 的核心即"块对角 → program=块 → 单 tile softmax"三连。
- **判据**：`max_kv_per_program ≤ BLOCK_KV 上限`（UB 预算内）即触发；否则保持 online softmax。

### M9 对角保底豁免整行被掩 fallback（M-Q4 的推论）

- 窗口带（op23/op51）、strided（op25，`i%stride==i%stride` 恒含自身）、块对角（op24，块内全 attend）**均含对角** → 每行至少一个有限 score → M3 的整行被掩 fallback **不触发**，省掉 fallback 分支。
- **但仍须**：`m_new = tl.maximum(m_new, MCLAMP)`（MCLAMP=-1e30）防 `-inf − (-inf) = NaN`；`l_safe = tl.where(l==0, 1, l)` 防 0/0（padding 行/幽灵行）。
- 纯 causal 无上界、纯 padding 全掩行**不享受**本豁免。

### M10 复合掩码的单 vsel 约束与成分分解

- **s 上只允许一个 `tl.where`**：多 vsel 共存触发 vexp root-alloc 编译错（op51 iter_0/iter_2 两次实测）。复合掩码（win | grow | gcol）**必须**先 OR 成单个布尔 keep，再一次 where。
- keep 的 `& n_mask` 写法在部分参数特化下同样触发 vexp root-alloc——pad 排除改走 M7-2 的算术减（`s −= pad_f*1e9`，有限 −2e9，exp 精确下溢 0），不要在 keep 里加项。
- 每行 self-window 保证至少一个有限 s > −1e9 > −2e9，max 归约不受 pad 污染（M9 推论）。

### M11 fp16 参考链位级复刻三连（条件：参考链本身在 fp16 域，如 NPU `F.softmax(fp16)` 全链）

- **触发判据**：参考是 fp16 sdpa 链（`matmul→*scale→masked_fill→F.softmax→matmul` 全程 fp16）且精度排查 MERE 超阈值时。以下三处必须**逐位复刻**，任一偏差 MERE 1e-3~6e-3（fp16 rel_threshold=9.77e-4）：
  1. **round16 标量乘**：NPU 上 `fp16_tensor × python_float` 实为 round16(scale) 后 **fp16 域**乘；Triton 标量乘会提升 fp32 丢 post-scale 舍入 ⇒ `s = tl.dot(q,k).to(tl.float16); s = s * scale.to(tl.float16)`；
  2. **softmax d16 域减法**：NPU `F.softmax(fp16)` 的 `d=(s−m)` 在 **fp16 域**减法舍入，exp/Σ/除在 fp32（探针 d16/e32/l32 路径 88976 元素 0 mismatch；d32 路径 20% 元素 >1ulp）⇒ `d = s16 - m[:, None].to(tl.float16); e = tl.exp(d.to(tl.float32))`；
  3. **p 归一化后舍入**：参考输出 p = round16(e/Σe)，**归一化后**才舍入 ⇒ online softmax 单边结构不可用（累加时不知最终 l，unnorm-round 与 no-round 仿真均超标），必须**三遍 KV + HBM scratch**：loop1 s16+滚动 max 暂存 scratch；loop2 重读累 `l=Σexp(d16)`；loop3 重读重算 e，`p16=round16(e/l)` 再 PV dot（exp 算两遍换零额外 HBM；max/sum fp32 精确滚动与参考一次性归约差异 ~1e-7，不翻转 fp16 舍入）。
- **探针先行（强约束）**：改 kernel 前先用 ~20 行 torch 探针在 NPU 上逐位比对候选语义路径（dtype 域 × 舍入位置全组合），拿 0-mismatch 证据再动 kernel；禁止"改 kernel→跑 verify→猜"循环。`tl.exp(fp32)` 已实测 ≡ torch NPU fp32 exp 逐位一致，可直接用。
- **Why**：op37 iter_0 9/50 → 三连逐个补齐 → iter_2 50/50。位级锁定后 Phase 4 零优化空间（§5）。M2 是本条的一般形式（注入域对齐），本条是 fp16 特化终态。详见 KKB `exp_debug_013_20260817`。

### M12 确定性掩码的物化分流：几何成分 kernel 算术化 / RNG·查表成分 host 物化 + cache

- **可算术推导成分**（带 / 块对角 / 元素周期 / 全局行列）→ kernel 内算术生成后 OR 合并单 vsel（op51-G1、op23-L1、op25-S1），零额外流量。
- **RNG / 查表成分**（per-row randperm 等）→ **host 物化**为 int8 稠密 `[SQ,SKV]` + 按 shape 参数 `(SQ,SKV,W,NRAND,device)` cache + kernel 单次 load + 单 `tl.where` 注入；随机列**直接 -inf 掩码，不用 gather**（MTE2 代价不划算）。**禁止**把 RNG 搬进 kernel（流不可复刻）；**禁止**每 forward 重建确定性掩码。
- **cache 语义保持依据**：seed 硬编码在参考 forward 内 ⇒ mask 是 shape 参数的确定函数，重建纯浪费。op37 参考每次 forward 重建 mask 占其 94~97.5% 时间（O(SQ) python 循环 + per-row randperm 发射），host cache 后 geomean **67.02×**。
- **RNG 三守**：seed(42) 在任何 randperm 前（注意 seed 在参考 **forward 首行**、不在 mask helper 内，对齐点别搞错）；randperm 必须在 **NPU device**（CPU/NPU RNG 流不同）；零 RNG 分支（全局行、SKV≤NRAND）不调 randperm。

---

## §2 精度闸门（掩码/结构化偏置特有）

- **`-inf` vs 有限极小值**：门控掩码用 `-3.0e38`（fp32）+ `tl.constexpr`，不用 `-inf`（整行被掩 → `exp(-inf-(-inf))=NaN`）。本类 mask 更密，更易整行被掩（FA F2）。参考用 `-1e9` 时（op25）kernel 仍可用 `-3e38`——softmax 后逐位等价，但**必须**确认参考无"读掩码后 score 原文"的路径。
- **整行被掩 fallback**：门控算子必须显式复刻参考的退化路径（如 `softmax→mean(V)`），不可简化为"全零行"。M-Q4 对角保底者豁免（M9）。
- **fp32 锁定判据**：参考全程 `.float()`（如 op14/op23）**且** q/k 量级超 `tl.dot` 安全区 [-5,5]（op14 q/k~290）时，**全 fp32 中间 buffer 是唯一正确解**，不可降 native（op14 iter_1 实测 fp16 QK 21/50 NaN）。继承 FA L1.4；fp16/bf16 输入时 QK 可原生 dtype 操作数+fp32 累加（乘积精确，op25 S3），输出逐步舍入回输入 dtype（FA L1.16）。
- **NaN/Inf 输入 sanitize**（op51 R1）：参考含 `nan_to_num` 时**必须拆独立 elementwise kernel**——nan2num 的 vsel 与 flash kernel 内 mask vsel 共存触发 `Unsupported op for finding the root alloc` 编译错。
- **怪异 scale 逐位对齐**：参考的缩放写法可能反常（op51：除以 `sqrt_d`）——按参考语义逐位复刻，禁止"改正"为 `1/sqrt(d)`。（op35 旧标杆曾有 `scores / (head_dim ** -0.5)` 实为 `× sqrt(head_dim)` 的怪异写法，新标杆 dca573d 已改标准 `× head_dim**-0.5`——**标杆更新会使本条例子失效，逐位复刻前必须重读当前参考源码**。）
- **RNG 复刻**：参考 forward 内动态建 `nn.Linear` 的（op23/op24/op51），用 `manual_seed + uniform_(-1/√D, 1/√D)` 逐位复刻 kaiming_uniform_(a=√5) 的 RNG 时序，host 缓存 + 预转置（exp_debug_010 配方，[[project-localattn-triton-task]]）。（op35 新标杆 dca573d 权重改为外部入参，RNG 复刻整个环节消失——改走权重预拼接/预转置 + data_ptr 缓存，见 §4.1。）

---

## §3 各算子章节

### §3.1 14_WindowAttention（单窗口 MSA + 投影 + 加性相对位置偏置）

**生效层**：(a) host-batched 窗口稀疏 + (b) 加性 rel_bias
**形态**：`[B, N=Wh·Ww, C]`（窗口已切，N∈[49,361]）；2 投影 GEMM（qkv + out）；score 全稠密（无 `-inf`）；加性 rel_bias（`attn+rel_bias`，几何索引 gather）；fp32 锁定。
**结果**：0.7433× → **2.1968×**（50/50，target 2.2× 噪声内），impl 1.115ms → 0.3786ms（−66%），<1.0× case 0/50，≥2.0× 30/50。0817 复跑（不同芯片）：Phase3 3.3972 → opt_iter_0 **4.0054**，同架构复现，数值上移不改结论。

**优化历程（5 轮 Phase4）**：

| 轮 | speedup | impl 延迟 | 关键改动 | 结论 |
|----|--------:|----------:|---------|------|
| 基线(iter_2) | 0.7433 | 1.115ms | 修复投影 `tl.dot` 维度/dtype 双错 | 起跑 |
| opt0 | 1.6625 | 0.501ms | **T1 投影 GEMV→GEMM** | **+124%** 最大单笔 |
| opt1 | 1.7801 | 0.470ms | attn BLOCK_KV 放大(D≥128) | +7% |
| **opt2 ★** | **2.1968** | **0.3786ms** | **T2 attn tile 推 UB(D-档) + T3 Cube 提前 + T4 并行度保护 + 投影 tile 放大** | **+23%** 夺冠 |
| opt3 | 2.0492 | 0.471ms | 投影 BLOCK_M 64→128 | ❌ −6.7% R1 |
| opt4 | 2.0696 | 0.497ms | rel_bias fp32→fp16 | ❌ −5.8% R2 |

#### 制胜技术（通用原理见 FA，此处只给 op14 具体实例）

**T1 投影 GEMV→标准 2D GEMM + 跨 batch M 展平（+124%，最大单笔）**（通用原理：FA §4.5/L1.12）
- **机制**：生成码把投影写成 GEMV（`BLOCK_M=1`，每 program 处理 1 个 token×BLOCK_OC 列）。C=512 时 task 数 = `B·N·cdiv(OC,BLOCK_OC)` ≈ 35k，每个 task 只做 1×K 的 dot → **Cube 几乎空转、launch 开销主导**。改 2D GEMM（`BLOCK_M=64`）后 task 数砍 ~64×，每 task 是真 64×K×128 矩阵乘，Cube 喂饱。
- **跨 batch 展平**（关键）：把 `B·N` 个 token 行**拍平成 M 维**，一个 GEMM kernel 吃完所有 batch×token，靠整数除还原 batch：
  ```python
  m_offs = pid_m*BLOCK_M + tl.arange(0,BLOCK_M)   # 跨 [0, B*N)
  b_idx = m_offs // N                              # batch 索引（gather 指针）
  n_idx = m_offs - b_idx*N                         # token 索引
  ```
- **权重 host 预转置**：`qkv_w_t = qkv_w.t().contiguous()`，使 w load 出即 `[K,OC]` 连续，正是 `tl.dot(x[M,K], w[K,OC])` 所需（FA L1.12：禁把 `nn.Linear` 的 `[out,in]` 原始布局直接喂 GEMM）。
- **判据**：profiling `aic_mac_ratio` 极低 + 代码里投影是逐行循环 → 立即 GEMM 化，不用扫参。

**T2 attention tile 按 D 分层推 fp32 UB 上限（+23%，夺冠）**（通用原理：FA §4.3/L1.13）
- **机制**：online softmax 迭代数 = `B·H·cdiv(N,BQ) × cdiv(N,BKV)`。BLOCK_Q、BLOCK_KV 同时放大 → 迭代数变 1/4 → `pipe_barrier`/同步整套减半（FA §4.3 的核心机理）。
- **fp32 专属 D-档表**（fp32 元素 4B，比 fp16 FA 案例大一倍，同 BLOCK 会溢出 UB，故必须按 D 分档）：
  ```python
  BLOCK_D = _ceil16(D)
  if   BLOCK_D <= 32:  BLOCK_Q, BLOCK_KV = 64, 128   # UB ~122-147KB
  elif BLOCK_D <= 64:  BLOCK_Q, BLOCK_KV = 32, 128   # ~131KB
  elif BLOCK_D <= 128: BLOCK_Q, BLOCK_KV = 32, 64    # ~122KB
  else:                BLOCK_Q, BLOCK_KV = 16, 32    # ~104KB（D=256）
  BLOCK_KV = min(BLOCK_KV, _ceil16(N))               # N 小时不超过 N
  ```
- **UB 账**（D=64, BQ=64/BKV=128）：scores `[64,128]×4`=32KB + acc `[64,64]×4`=16KB + q/k/v ~48KB + multi-buffer → 推到 ~140KB 上限（UB 共 192KB）。
- **放大到编译失败为止**：上限看编译器报的 `requires N bits while 1572864 bits available`（FA §4.3）。⚠️ 不可用阈值预测——占用还受 constexpr 分支组合影响。

**T3 §4.7 Cube 提前发射**（FA §4.7）
- `pv = tl.dot(p, v_tile)` 是 Cube，`p_sum = tl.sum(p)` 是 Vec。把 pv 提到 p_sum 之前，Cube 与 Vec 重叠，掩盖 Vec 延迟。纯重排、数值等价。op14 是 FA §4.7 的一个正数据点（含在 +23% 内）。
```python
pv = tl.dot(p, v_tile)     # Cube 先发射
p_sum = tl.sum(p, axis=1)  # Vec 与上面 Cube 重叠
```

**T4 并行度保护（if 级联，禁 while）**（通用原理：FA §4.5）
- grid 过小（tile 数 < `NUM_CORES//2`）→ 核吃不饱 → 缩 BLOCK_Q 增 task。**必须用 `if` 级联，禁 `while`**（validator 禁 forward 内任何 loop，见 [[triton-validator-forbids-all-loops]]）：
  ```python
  min_tiles = max(1, self.NUM_CORES // 2)
  if B*num_heads*_cdiv(N, BLOCK_Q) < min_tiles:
      BLOCK_Q = max(16, BLOCK_Q // 2)
  if B*num_heads*_cdiv(N, BLOCK_Q) < min_tiles:
      BLOCK_Q = 16
  ```
- op14 把 FA §4.5（原写"仅 fa-mha 投影"）的并行度保护**扩展到 attention 侧 BLOCK_Q**——本类小 N 窗口 attention 极易 tile 数塌陷，attention 侧也必须保护。

#### 两个陷阱（首次生成必避）

**P1 投影 BLOCK_K = `_ceil16(C)` → UB 溢出**
- GEMM 化时若 `BLOCK_K=_ceil16(C)`（误把整个 C 当一个 tile），C=384/512 → x tile `64×512×4=128KB` + w 128KB + acc → **>>192KB UB**，报 `MLIRCompilationError: multi-buffer extra local buffer`（opt0 初版 15/50 挂）。
- **修复**：`BLOCK_K` 固定小值（64/128），`for c0 in range(0, C, BLOCK_K)` 自分块。

**P2 5D permute → `memref.collapse_shape` 编译错**
- qkv 若 `view(B,N,3,H,D).permute(2,0,3,1,4).contiguous()` 再按 5D stride 访问，D=64/128 时报 `collapse_shape: collapsed dim size(256) must equal reassociation group size(128)`。
- **修复**：扁平 `[B,N,3C]` 连续 buffer + stride 算术拆 q/k/v，零 permute：
  ```python
  q_base = qkv_ptr + b_idx*stride_qb + h_idx*D
  k_base = q_base + C          # q 占 [0,C)
  v_base = q_base + 2*C        # v 占 [2C,3C)，head h 在偏移 h*D
  ```

#### 两个回归（证伪，机理必读）

**R1 投影 BLOCK_M 64→128（耦合 BLOCK_K 128→64）→ −6.7%**（opt3）
- **机制**：放大 BLOCK_M 意在复用 w（halve w GM 流量），但为不超 UB 须把 BLOCK_K 减半（128→64）→ **K-loop 翻倍** → 循环/同步开销翻倍，**超过** w 复用收益。大 C case 全变慢（c42 1.438→1.224、c50 1.285→1.046）。
- **教训**：BLOCK_M 与 BLOCK_K 被 UB 预算**耦合**，单维放大必然逼另一维回缩；净效果必须实测，不可假设（印证 FA F5/§4.3「tile 收益不可简单预测」）。

**R2 rel_bias fp32→输入 dtype（fp16/bf16）→ −5.8%**（opt4）
- **机制**：rel_bias 体积小（`[H,N,N]`，如 2×49×49≈5K 元素），halve 其流量省的极少；但 cast 到 fp16 **额外 dtype 转换 + 512B 行对齐劣化**反而拖慢，impl 0.38→0.50ms 反升。
- **教训**：结构化加性偏置体积小，**降精度得不偿失**；保持在 softmax 累加域（M2）。

#### 关键约束清单（首次生成必守）
1. 投影禁 GEMV，标准 2D GEMM，跨 batch M 展平（T1）。
2. 投影 BLOCK_K 固定小值，禁 `_ceil16(C)`（P1）。
3. qkv 拆分用扁平 buffer + stride，禁 5D permute（P2）。
4. attention tile 按 D 分档推 UB 上限（T2）+ attention 侧并行度保护（T4）。
5. fp32 锁定（参考全程 fp32 + q/k 超 dot 安全区，§2）。
6. ghost-column mask：`tl.where(mask_kv, scores, NEG_INF)` + `p = tl.where(mask_kv, p, 0.0)`（KV=N 窗口 token，越界来自 BLOCK_KV 对齐 padding）。

#### 泛化启示（给本类其他算子）
- **窗口/局部 + 投影 + 结构化偏置**算子：先确认稀疏是否 host-batched（§0.2-a）——若是，别在掩码上浪费轮次，直接攻投影 GEMM + attention tile（M1）。
- **加性偏置算子**不触发 ghost 列/整行被掩（M3 例外），但注入精度仍须对齐 softmax 域（M2），且降精度是死路（R2）。
- **小 N 算子**最弱 case 多被 launch 钉死（M5），target 别定太高（op14 的 100× 不可达）。

---

### §3.2 23_LocalAttention（窗口带门控掩码 + 4 投影，带区间收缩）

**生效层**：(b) kernel 内门控掩码 `|i−j| < window`，带区间收缩实现 kernel 内块跳过
**形态**：`[B, S, C]`；4×`nn.Linear(bias=False)`（wq/wk/wv/wo，RNG 复刻+host 缓存+预转置）；参考 `masked_fill(带外, -inf)`；fp32 契约（参考 `q.float()@k.float().T`，q/k/v load 后 cast fp32，输出一步 cast 回）。
**结果**：0815 首版即 **3.2810×**（50/50，NPU10，未做 Phase 4 用户收敛）；0817 复跑 2.7557（芯片间 framework 波动）。

**制胜技术**：

**L1 window 带 KV 区间收缩（稀疏红利主来源）**（M6 带型形态）
- 对每个 query 块，KV 扫描区间收缩为 `|i−j| < w` 的并集，带外整块**不 load 不算**：
  ```python
  kv_lo = q_start - wm1
  if kv_lo < 0: kv_lo = 0
  kv_hi = q_start + BLOCK_Q + wm1
  if kv_hi > S: kv_hi = S
  for kv_start in range(kv_lo, kv_hi, BLOCK_KV):   # 带内逐块 online softmax
  ```
- S=4K、w=128 时 KV 扫描量砍 ~30×；这是 (b) 门控算子可用 (c) 式跳过的唯一形态（带几何可由 q_start 算术判定）。

**L2 keep 掩码 fp32 比较 + 幽灵列显式排除**（M7-1 + M3）
```python
col_f = offs_kv.to(tl.float32)[None, :]
keep = (col_f >= (row_f - wm1f)) & (col_f <= (row_f + wm1f)) & mask_q[:, None] & (offs_kv < S)[None, :]
scores = tl.where(keep, scores, NEG_INF)
...
p = tl.where(keep, p, 0.0)     # 分母清零（op23 未用 S2 下溢技巧，保留 where 亦可过）
```
- ⚠️ masked load `other=0.0` 的幽灵列 score=0 **会通过带判定**（0 落在 [−w, w] 内），必须 `& (offs_kv < S)` 显式排除。

**L3 w≥S 掩码失效特化**：`window_size ≥ S` 时带覆盖全行，keep 恒真——可 constexpr 分支省掉全部掩码算术（0815 首版命中要素之一，多 shape 任务里小 S case 必触发）。

**陷阱/弱项**：
- 大 D fp32 case 0.84~1.03×（ieee dot 吞吐上限，同 [[project-gqa-triton-task]]）——fp32 + D≥256 是结构性天花板，不要在 Phase 4 硬攻。
- 投影部分与 op14 同套：4×标准 2D GEMM + 权重预转置 + 跨 batch M 展平；BLOCK_K 固定小值（P1 同样适用）。

---

### §3.3 24_BlockSparseAttention（块对角门控 + 4 投影，单 tile 一次性 softmax）

**生效层**：(c) 块级结构——块对角掩码（`block_ids[i]==block_ids[j]` 才 attend，`block_ids = positions // block_size`）
**形态**：`block_size∈[2,31]` 极小；4×`nn.Linear(bias=False)`（同 op23 RNG 复刻）；参考 `masked_fill(块外, -inf)`；fp32 契约（QK 原生 dtype 操作数+fp32 累加，PV 中 p fp32、v cast fp32）。
**结果**：0815 首版即 **3.6387×**（50/50，NPU10）；0817 复跑 Phase3 2.6048。opt_iter_0（G 块合组变体）未 bench，记为可选。

**制胜技术**：

**B1 program = 单掩码块（稀疏结构 = 并行粒度）**
- grid 维 `(b, blk, h)`，每 program **只 load 本块的 Q/K/V 行**（`rows = blk*block_size + arange(BLOCK_M)`），跨块计算整块消除——不需要任何"掩码表"或索引 gather，块号即地址。
- 参考语义 `masked_fill(块外, -inf)` 被**结构等价**为"只算块内"，kernel 内 pair_mask 只剩 padding 排除（`row_valid[:,None] & row_valid[None,:]`）。

**B2 单 tile 一次性 softmax（M8 的首个实证）**
- `BLOCK_M=32 ≥ max(block_size)=31` → KV 全集一个 tile 装下 → **取消 KV 循环**，online softmax 退化为：
  ```python
  s = tl.dot(q, tl.trans(k)) * scale          # [32, 32] 单 dot
  s = tl.where(pair_mask, s, NEG)
  m_new = tl.maximum(tl.max(s, 1), MCLAMP)    # 一次性 max，无滚动 m_i
  p = tl.exp(s - m_new[:, None])
  o = tl.dot(p, v.to(tl.float32)) / tl.maximum(tl.sum(p, 1), 1e-30)[:, None]
  ```
- 省掉 m/l/acc 滚动 + alpha 重缩放整套；BLOCK_M=32 同时满足 tl.dot 最小 tile。attention 计算量远小于投影 GEMM，小 tile 浪费可接受。

**B3 G 块合组变体（可选，未验证）**：`G = BLOCK_M // block_size`，一个 tile 装 G 个块，`pair_mask = blk_ids[:,None]==blk_ids[None,:]` 判同块——block_size≪16 时提升 tile 利用率。**未 bench，首版直接用 B1/B2**。

**证伪**：大 D 投影 GEMM tile autotune **−5.2%**（0815）——投影 4 连发已贴近 Cube 上限，扫参无油水。

---

### §3.4 25_StridedAttention（元素级周期掩码，无投影，dense 扫描+算术掩码）

**生效层**：(b) 元素级门控 `j%stride == i%stride`（含对角）——**无法块跳过**：BLOCK≥stride 时每个 KV 块内必含全部残基类
**形态**：纯 attention **无投影**（q/k/v 仅 reshape）；参考 `masked_fill(-1e9)` ≡ softmax 后 `-inf`（§2 等价性）；fp32 契约。
**结果**：0815 Phase3 首版 2.4029（50/50，NPU10）；0817 复跑 Phase3 2.4683 → opt_iter_0 **3.4325**（dtype 白名单化）。

**制胜技术**：

**S1 dense KV 扫描 + 元素算术掩码**（M6 元素周期形态）
```python
mq = offs_q - (offs_q // stride_step) * stride_step        # 禁 %（M7-4）
mk = offs_kv - (offs_kv // stride_step) * stride_step
keep = (mq[:, None] == mk[None, :]) & mask_q[:, None] & mask_kv[None, :]
scores = tl.where(keep, scores, NEG)
```
- EQ 比较保持整型（不触发 M7-1 的 i32 2D 毒性的实测分支）；掩码**不物化稠密 [S,S]、不进 host**。

**S2 p 不二次清零（省每 KV 迭代一个 [BQ,BKV] vsel）**
- 掩码列 `scores=NEG(-3e38)`，`m_new ≥ MCLAMP=-1e30` 时 `exp(NEG − m_new)` **精确下溢为 0** → 不需要 `p = tl.where(keep, p, 0.0)`。M3-① 的 where 在本类负值足够小时可省（op23 保留 where 也过，op25 删掉更快——两者都合规）。

**S3 dtype 白名单（0817 +39% 主来源）**
- fp16/bf16 输入：QK 走原生 dtype 操作数 + fp32 累加（乘积在 fp32 精确，FA L1.1），q/k tile 减半 → `BLOCK_KV=128, bq_budget=8192`；
- fp32 输入：全 fp32 → `BLOCK_KV=64, bq_budget=4096`，另加 UB 估算守卫循环收缩 tile（fp32 v tile 是主要压力）。

**陷阱**：
- **FA L1.17 / aicore 507015**：strided 行索引向量（`r + i*ST`）仅出现在掩码算术即触发 507015，shape 相关、与 task 数无关。规避 = dense `qi` + `qi < sr` 等价判定（[[stridedattn-triton-task]]）。
- **Phase 4 #30 残基类分解两版证伪 −6%**（packed+gather / 自然布局 store+dense K4）：attention 本体 1.7~2.5× 但被 strided KV 行跨步 + pack pass + gather 对冲——元素周期掩码的"稀疏红利"在 Ascend 上**不抵重组开销**，dense 扫描就是终态。

---

### §3.5 51_LongformerAttention（复合掩码 = 窗口带 ∪ 全局行列 + sanitize + out 投影）

**生效层**：(b) 复合门控 `win | grow | gcol`，`-1e9` 有限掩码；对角保底（M9 豁免 fallback）
**形态**：`[B,H,S,D]` BNSD 直入无转置；`nan_to_num` 预处理（参考语义）；out 投影（RNG 复刻 kaiming）；S∈[13,263] 小，w2=16、g1=511。
**结果**：0.4114 → **2.7685×**（50/50，6 轮 Phase4 含 1 IR 轮），speedup_vs_baseline 6.73。小 S case 4~5×，大 S·D fp32 case 1.0~1.7×（M5/ieee dot 双地板）。

**优化历程**：

| 轮 | 改动 | 结果 |
|----|------|------|
| Phase3 | sanitize 拆分 + 复合掩码单 vsel 版 | 0.4114（编译错误三连修） |
| opt_iter_2 | **G2 i32 2D 比较转 fp32** | **2.5366（+6.6× 最大单笔）** |
| opt_iter_5_ir_1 | **G3 pad 列算术化**（IR 轮） | **2.7685（+9.1%）** |
| opt_iter_1/3 | 优化点 6 全量/1D-only fp32 转换 | ❌ 22/50 确定性编译失败，永久耗尽 |

**制胜技术**：

**G1 复合掩码成分分解 + 单 vsel**（M10）
```python
offs_nf = offs_n.to(tl.float32)
win  = (offs_nf[None,:] >= (offs_m-w2).to(tl.float32)[:,None]) & \
       (offs_nf[None,:] <= (offs_m+w2).to(tl.float32)[:,None])   # 仅 2D 比较转 fp32（G2）
grow = ((offs_m == 0) | (offs_m == g1))[:, None]                 # 1D EQ 保持整型
gcol = ((offs_n == 0) | (offs_n == g1))[None, :]
keep = win | grow | gcol
s = tl.where(keep, s, -1e9)      # s 上唯一一个 vsel
```

**G2 i32 2D 窗口比较标量降级 → fp32 比较**（M7-1 的命名案例，+6.6×）：`offs` 整型 2D LT/LE 在 BiShengIR 下降级为逐元素标量循环；`.to(tl.float32)` 后走向量 fcmp。**全量转换（含 1D）反而触发 M7-2 编译失败**——只转 2D 比较项。

**G3 pad 列算术排除**（M7-2/M7-3 的命名案例）
```python
pad_f = tl.minimum(tl.maximum(offs_nf - S + 1.0, 0.0), 1.0)   # vmin/vmax 生成 [0,1] 指示
s = s - pad_f[None, :] * 1e9        # pad 列 s → -2e9（有限），exp 精确下溢 0
```
- 相比 `tl.where(n_mask, p, 0)` 同时消除：(a) n_mask 张量物化的 i64 标量循环；(b) 临时量 s_m；(c) p 上的第二个 vsel。真实列不受影响：self-window 保底 s > −1e9 > −2e9（M9）。

**G4 sanitize 独立 elementwise kernel**（§2）：`nan_to_num` 与 flash kernel 内 mask vsel 共存 → root-alloc 编译错，必须拆出单 kernel（BLOCK=1024 已最优，Block Size Scaling 无收益）。

** flaky 判据**：长 verify 进程内偶发单 case root-alloc/NaN 为 flaky（重跑可过）；**22/50 级别大面积同指纹失败 = 确定性毒性，立即放弃该方向**（优化点 6 据此永久耗尽）。

---

### §3.6 37_BigBirdAttention（复合掩码+随机 gather，host 物化 cache，fp16 位级锁定三遍 KV）

**生效层**：(b) 复合门控 = window 带 ∪ 首/末全局行列 ∪ per-row randperm 随机列；对角保底（M9 豁免 fallback）
**形态**：BNSD 直入纯 attention（无投影）；fp16 参考链 `matmul→*scale→masked_fill(-1e9→NPU 溢出 -inf)→F.softmax→matmul`；可选 pse/sink 加性偏置（M2 注入域对齐）
**结果**：Phase 3 三轮收敛（iter_0 9/50 fp32 中间域 → iter_1 40/50 补 round16(scale) → 47/50 补 softmax d16 → iter_2 50/50 补三遍 KV），iter_2 首过即 **67.0198×**（50/50，NPU12）；Phase 4 数值锁定零优化（0/31 命中，simulator 证实结构极限）。

**制胜技术**：

**BB1 整 mask host 物化 + 参数化 cache（67× 主来源，M12 的命名案例）**
- window/global/random 三成分全部 host 侧构建为 int8 `[SQ,SKV]`，按 `(SQ,SKV,W,NRAND,device)` cache；kernel 内一次 load + 单 `tl.where(keep!=0, s, -inf)` 注入（M10）。
- **随机列直接 -inf 掩码，不用 gather**（骨架时代"待验证"的答案：host randperm 复现过 validator 且全 shape 0 mismatch；gather 的 MTE2 代价不划算）。
- **禁用区间收缩被证实正确**（BB4 首版）：随机成分 per-row 列集不同，无均匀块可跳；dense 扫描已 67×。
- 67× 构成（诚实口径）：framework 每次 forward 重建 mask 占其 94~97.5%（O(SQ) python 循环 + per-row randperm 发射 vs impl 0.036ms）；剔除掩码重建后的纯 attention 对比约 13~33×，仍远超 target。同型"参考带 python mask 构建"算子直接复制本架构，勿在 kernel 内找优化空间。

**BB2 RNG 复刻三守**（M12）：seed(42) 在 forward 首行（mask helper 内部**不** seed，对齐点别搞错）；randperm 在 NPU device（CPU/NPU RNG 流不同）；全局行（i=0/SQ-1）与 SKV≤NRAND 分支零 RNG 消耗。

**BB3 fp16 位级锁定三连 → 三遍 KV + HBM scratch**（M11 的命名案例，KKB `exp_debug_013_20260817`）
- loop1 算 s16（fp16 链）+ 滚动行 max，暂存 HBM scratch `[B,H,SQ,BLOCK_SKV]` fp16；loop2 重读累 `l=Σexp(d16)`；loop3 重读重算 e，`p16=(e/l).to(fp16)` 再 PV dot。
- 探针先行方法论锁定语义后才动 kernel（M11 强约束），单轮 10min+ 的 verify 循环无法归因。

**陷阱/证伪**：
- **Phase 4 零优化**：位级锁定后 31 优化点无一可动；simulator（Ascend910_9391）实测 MMAD 仅 5.7%、Vector 49% 全为精度必需 fp16 链、热点行 L133 `p16=round16(e/l)` 占 24%——**结构极限，勿硬攻**（§5）。
- 探针脚本输入忘 `.to(npu)` 会让参考 `masked_fill(-1e9)` 在 CPU 直接 raise（NPU 上溢出为 -inf 才是被依赖的语义）——先搬设备再探针。

---

### §3.7 105_FlexAttentionBwd（**本卡首个 backward 成员**：lse 重算 p 免 delta kernel，4 kernel 双路径 + WS 物化 fp32 cast）

**生效层**：(b) causal 门控（22/50 case true，运行时属性 → constexpr 特化）；参考实现仅 causal + GQA（2/50），无复合掩码
**路由注记**：按写法本例命中 attention_index 行 7（FA 分块流式反向主链）＋ 行 4（GQA 交叉），实际由 `flash_attention.md`＋`gqa.md` 双卡生成；经验回填本卡是因为 **FlexAttention 家族（mask_mod 通用掩码 API）前向属行 6b**，其 backward 骨架对本卡全族成员的反向写法复用
**形态**：纯 attention backward（无投影；go/q/k/v/o/lse 全外部入参）；**lse 已知 → 免 delta kernel、免 online 递推**（delta=rowsum(do*o) 内联 K1 task 头部，`p=exp(s-lse)` 直接重算）；精度契约 = 全 fp32 ieee（FA L1.18 红线，fp16/bf16 case 同）
**结果**：Phase 3 iter_0 首过 0.7168（50/50，NPU5）→ 点30 两轮 tile 放大 0.9419（+31.4%）/ 0.9772（+3.7%）→ IR 轮点21 WS 物化 **1.0705**（+9.5%，50/50）；target=100 未达，结构性瓶颈 = fp32 ieee cube 吞吐 vs torch 参考走 CANN fp32 优化库（与 op23/op51 大 D fp32 天花板同源）

**制胜技术**：

**FB1 反向 4 kernel 双路径（本卡 backward 骨架基线）**
- K1 dq：q 主序 persistent grid-stride，delta 内联，`WRITE_DS` constexpr 可选把 ds **转置**写 WS `[B,H,Sk,Sq]`（转置靠 store 指针算术，FA L1.20-3）。
- K2 dk：3D grid `(cdiv(Sk,BKV),Hkv,B)` 纯 GEMM（1 dot/iter），直读 WS 转置布局——ds 免重算、免转置物化。
- K3 dv：3D grid 重算 p（sT=k@qᵀ 转置态一步算出，免 trans(p) 物化，2 dot/iter）。
- K4 dkdv 融合：Sk<256 小 S 路径（K/V 常驻，delta 就地）。host 门控 `USE_WS=(Sk>=256)`：True→K1(W)+K2+K3，False→K1(noW)+K4。
- 与行 1/1b 注记的块稀疏反向（双选择表 + delta/dq/dkdv 三 kernel）的区别：**dense 掩码（仅 causal）无需选择表，lse 直给免 delta kernel**；若本卡成员带块稀疏 backward，选择表层叠加在本骨架之上。

**FB2 点30 tile 放大到 UB 上限 + `_protect_bkv` 并行度保护（+31.4% / +3.7%）**
- K2 BKV 64→128、K3 D=128 BQ 32→64 / D≤64 BKV→128（UB 核算留 ≥20KB）；K2 BQ2 64→128 仅 D≤64（D=128 档 192KB 超限保持 64）。
- **保护规则**（点30 §2.7）：BKV 放大后 3D grid 任务数 `cdiv(Sk,BKV)*Hkv*B < ncores//2` 时回退 64——大 tile 挤掉并行度比 tile 效率更伤。
- tile 空间至此耗尽：K1 dq D=128 被 UB 逼在 (32,64)；K4 只服务 2 个小 case（已 3-6x，勿过度投入）。

**FB3 ★ 点21 WS 物化 fp32 cast（IR 证据链驱动，+9.5% 最大单轮增量）**
- **IR 实锤**（last_pass.mlir）：mix kernel 里 `tl.load(fp16).to(tl.float32)` 的 vcast 在 AIV 执行后**经 GM 回退**喂 cube——`store_ubuf_to_gm` + `sync_block_set/wait` 跨核握手 → AIC `nd2nz` GM→L1。每个 cube 操作数的 cast 都是一次 UB→GM→L1 全程往返。
- **冗余结构**：K2/K3 是 kv 主序 3D grid，同一 q tile 被 `cdiv(Sk,BKV)` 个 program 各自重复 load+cast（cast 总量 ×32 @ Sk=4096/BKV=128）。
- **改法**：K1（q 主序，每个 q tile 恰好处理一次）顺手把 `q_t`/`do_t` 的 fp32 cast 结果物化到 `WSQ/WSGO[B,H,Sq,D]` fp32（torch.empty，每元素写一次，masked 区不被读）；K2/K3 改直读 fp32（免 `.to`、免 GM 回退，cube nd2nz 直载，AIV 循环体清零）。**数值 bit 级一致**（同一 vcast 结果搬运，verify 50/50）。
- **边界**：K1 内 k/v 的 per-KV-iteration cast 无法物化（没有任何 kernel 独占处理全量 kv 一次；K3 的 k 是 per-program 一次已 hoist）——反向 WS 物化只覆盖 q 侧张量。

**FB4 warmup-only IR 提取法（debug 二进制崩溃的规避）**
- `run_and_extract.sh` 的 debug 编译标志（`TRITON_DISABLE_FFTS=1 TRITON_ALWAYS_COMPILE=1` 等）下本算子 K2 二进制**运行期崩 507015**（生产二进制正常）→ monkeypatch `JITFunction.run` 强制 `warmup=True`（只编译不发射），4 kernel IR 全提取成功。
- `/tmp` 下他人遗留的 `kernel_N_full_ir_dump.txt`（root 所有、跨任务同名）会让提取脚本读到**陈旧 dump**（last_pass 提到别的 kernel 的 IR）——提取前 `rm -f /tmp/kernel_*_full_ir_dump.txt /tmp/*last_pass*.mlir`。

**陷阱/证伪**：
- **点25 causal WS torch.zeros→empty + K2 掩码 load：证伪回退**（§5 新增行）——"causal zeros memset +23~25ms"源自非紧邻自然实验（#44 vs #49 同 shape 对照），被分段机器漂移污染；时间紧邻复测原代码 causal case 同样快 25ms（memset≈0），掩码 load 反 +1.5%。**判定铁律：A/B/A' 时间交错**（[[feedback-crossattn-honest-benchmark]]，0819 强化）。
- 点30 继续扩 K1 D=128 BQ 32→64：UB 逼死（dq_acc+q/k/v/go fp32 tile 超 192KB），勿再试。

---

## §4 待后续算子填充（骨架占位）

> 预期成员的**生效层 + 预期杠杆**锚点，算子到位后按 §3 体例回填。

### §4.1 35_AxialAttention（轴向分解，无掩码）✅ 0818 已落地（新标杆 dca573d 重跑）
- **生效层**：(a)——2D 轴向稀疏被 host reshape 消费为**两次稠密小 N attention**（`[b·w, h, c]` 沿 H 轴 + `[b·h, w, c]` 沿 W 轴，输出相加）。kernel 内**无任何掩码**。
- **终态**（iter_2 首过 2.4269 → #12 分档 2.4388，50/50，NPU5）：**4 kernel 6 发射**——K0 layout_copy（X[B,C,H,W]→X_PERM[B,H,W,C] 纯搬运）→ K1 qkv_proj_gemm **单发射吃两轴**（两轴投影对同一 token 集算两遍值相同，v22 实证）→ K2 axial_attn ×2（AXIS constexpr，BLOCK_S=32 ≥ max(H,W)=31 单 tile softmax，M8）→ K3 out_proj_gemm ×2（ACCUMULATE 0 覆盖写免清零 / 1 加写，每轴各加一次 bias 逐字复刻两次 F.linear 舍入时序）。
- **权重胶水**：host 预拼接 `w_qkv_t[C,3I]`（空 buffer + 两段 copy_ = cat 逐比特等价）+ `w_out_t` 预转置，按 data_ptr 缓存（FA L1.12）。
- **dtype 契约**（低精度逐步舍入链）：s16 = round(dot) → 低精度域乘 scale → 幽灵列 vsel(-3e4) → fp16 **必须 fp32 减 max + 显式 .to(f16) 舍入**（隐式舍入被编译器消除，M11-2）；bf16 全程 fp32 域减不回舍；p 归一化后舍入再 PV dot。
- **有效杠杆**：仅 #12 大 M（≥1024）BM 64→128（K1/K3 联动，B 权重 tile 重载减半，+0.49%；case49 fp32 大 case +11.2%）。弱 case 1.09-1.23x 全为大 M 大 C GEMM 主导（接近硬件极限），小 case 4.9-6.1x。
- **simulator 终证**（SOC=Ascend910_9372）：K1/K2/K3 MMAD 仅 10-19%，MTE2 装载 43-50% 主导 = 小 M 权重流带宽墙，非 Cube 瓶颈；tile 平衡点 128/128 实证（256/64 扩档目标 case 反而 -12~-24%）。

### §4.2 ~~37_BigBirdAttention~~（✅ 已落地，回填至 §3.6）

### §4.3 causal / padding `-inf` 掩码注意力（TODO）
- **生效层**：(b)；M-Q2 运行时可变 → 大杠杆在 KV 区间收缩（FA §3.2，+21~58%）+ mask 分支 constexpr 特化（FA §5.1-#4，+21%）。
- **必守**：M2（有限极小值）、M3（整行被掩 fallback——causal 首行/纯 padding 全掩行**不享受** M9 豁免）。
- **待验证**：区间收缩在本仓库 triton-ascend 的实际增益（FA 数字来自它算子，须重测）。

### §4.4 topk-selected KV 注意力（TODO）
- **生效层**：(c)；杠杆在索引 gather（只 load 被 topk 选中的 KV 行）。
- **相关**：[[triton-5-lightningindexer-topk-wall]]（topk 本身可能是墙，validator 禁 torch.topk）。

### §4.5 真·shifted-window（带 `attn_mask` 参数的 SW-MSA）（TODO）
- **生效层**：(b)；与 op14 区别——掩码是 kernel 内可见运行时 mask，可用 mask 特化。
- **待验证**：attn_mask 注入精度、整窗被掩退化。

---

## §5 证伪方向（跨成员，持续积累）

> 继承 FA §5.2 全表；以下为本类**新增**。

| 方向 | 结果 | 来源 |
|------|------|------|
| 对 host-batched 稀疏算子套块跳过 / mask 特化 | **no-op**（稀疏已不在 kernel 内） | op14（M1） |
| 加性 rel_bias 降到输入 dtype | **−5.8%**（转换+对齐劣化抵消流量；偏置体积小不值得） | op14 opt4（R2） |
| 投影 BLOCK_M 放大（耦合 BLOCK_K 回缩） | **−6.7%**（K-loop 翻倍主导；M/K 被 UB 耦合，单维放大可净负） | op14 opt3（R1） |
| 投影 BLOCK_K = `_ceil16(C)` | **编译失败**（x tile 128KB+ 溢出 UB） | op14 opt0 初版（P1） |
| qkv 5D permute 拆 q/k/v | **编译失败**（`collapse_shape` for D=64/128） | op14（P2） |
| 把加性偏置当门控掩码"跳块" | 概念错（加性不改分母，无块可跳） | M-Q3 |
| strided 残基类分解（packed+gather / 自然布局 store） | **−6% 两版**（strided KV 跨步 + pack pass + gather 对冲稀疏红利） | op25 Phase4 #30 |
| 大 D 投影 GEMM tile autotune（4 连发已贴 Cube 上限） | **−5.2%** | op24（0815） |
| 优化点 6 全量 fp32 转换 / 1D-mask-only 转换 | **22/50 确定性编译失败**（sitofp+fcmp root-alloc 毒性），优化点永久耗尽 | op51 opt_iter_1/3（M7-2） |
| nan_to_num 与 flash kernel 内 mask vsel 共存 | **root-alloc 编译错** → 拆独立 sanitize kernel | op51 iter_0（G4） |
| s 上多个 tl.where（复合掩码逐项注入） | **vexp root-alloc 编译错** → OR 合并单 vsel + pad 算术减 | op51 iter_0/iter_2（M10） |
| strided 行索引向量参与掩码算术 | **aicore 507015**（shape 相关）→ dense qi + `qi < sr` 等价 | op25（FA L1.17） |
| 大 D fp32 case 硬攻（ieee dot 吞吐上限） | 0.84~1.03× 结构性天花板 | op23/op51 大 D fp32 case |
| 对 per-row 随机掩码强行块跳过 / 带区间收缩 | **不适用**（逐行列集不同，无共享块网格；dense 扫描即终态） | op37（BB4 首版） |
| fp16 位级锁定链上做 Phase 4 kernel 优化 | **0/31 命中**（MMAD 5.7%、Vector 热点全为精度必需——结构极限，勿硬攻） | op37 Phase4 + simulator |
| K^T 转置直载消 vtranspose（小 tile 32×128 场景） | **−1.6%**（跨步转置 load 的 MTE2 代价 > AIV 小 transpose 代价；case2 小 case −11% 最敏感） | op35 IR轮1 |
| softmax p 归一化 divf→倒数乘 | **−1.95% 无收益**（预期 ≤1% 被会话噪声 ~1.5% 淹没，不可确认） | op35 IR轮2 |
| MTE2_high 下 fp16/bf16 big 档 BM=256/BN=64（B 装载减半） | **−3.4%**（BN=64 ⟹ N_tiles 翻倍，A 装载翻倍 + tile 效率下降反噬目标 case −12~−24%；128/128 是实测平衡点） | op35 simulator 轮 |
| msprof op simulator 带 `TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1` | **app signal 11 崩溃**（重编译路径 bug）→ 采集前 unset 这两个 IR 提取遗留变量 | op35 simulator 踩坑 |
| backward causal 分支"WS torch.zeros memset 开销"假设（点25 zeros→empty + 掩码 load） | **证伪回退**（memset≈0；+23~25ms 是非紧邻对照被分段漂移污染的假象，时间紧邻复测原代码同样快 25ms；掩码 load 反 +1.5%） | op105 opt_iter_1（FB3 陷阱） |
| backward K1 dq D=128 档 BQ 32→64 扩 tile | **UB 逼死**（dq_acc+q/k/v/go 全 fp32 tile 超 192KB），tile 空间耗尽标志 | op105 opt_iter_2 |

---

## §6 测量口径与时效

- **几何平均**：每 case 权重相同（FA §0.2-Q4）。小 N / 块稀疏最弱 case 多，**不能只优化大 case**——op14 把 laggard（c42/c50）从 <1× 拉到 >1.2× 是夺冠关键。
- **芯片间波动**：同一最终代码 0815（NPU10）↔ 0817（不同芯片）geomean 可差 ±10~30%（op23 3.28→2.76、op24 3.64→2.60、op25 2.40→3.43），主因 framework 侧噪声。**优化判定一律用 impl_ms 紧邻 A/B**，禁用跨日 speedup 对比（[[feedback-crossattn-honest-benchmark]]）。
- **同芯片同时段漂移**（op105 0819 NPU5 实测强化）：同代码两次全量 benchmark 的 geomean 差 **±2.3%**；分段性漂移可在连续 case 段落叠加 **±20~30%**（fw 与 impl 同步上涨，环境性）。**非紧邻的"自然实验"（同 shape 跨 run 对照）完全不可信**；判定必须改动代码与同代码复跑时间交错（A/B/A'），或靠内置对照组（本轮未改动的代码路径 ratio≈1 佐证机器稳定）。
- **历史结论随编译器/CANN 失效**（FA §7.3）：本文件所有数字标注来源算子与日期，**判据/机理优先于绝对数字**。

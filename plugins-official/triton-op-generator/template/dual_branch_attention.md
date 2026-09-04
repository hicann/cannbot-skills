---
name: dual_branch_attention
description: CV 双分支特征图门控类算子（BAM / Polarized Self-Attention 并行与串行 / TripletAttention）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 三骨架（GEMV 链 / conv 管线 / 三分支管线）、Layer 3 关键技巧与 Phase 4 优化点清单（含证伪全表）
metadata:
  type: reference
---

# CV 双分支特征图门控算子优化经验

本文档是**特征图门控双分支/三分支**这一类 CV 注意力算子的经验合集，覆盖 Phase 2/3/4：

- **§0 适用范围与算子分类**（判别特征 + 形态识别五问）
- **§1 通用经验**（跨形态共有的工程约束）
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（PSA GEMV 链 / BAM conv 管线 / Triplet 三分支管线）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表**

> ⚠️ **本文件覆盖「五 专用类·特征图门控」细分**（attention_index.md 行 13a）：
> 凡满足 §0.1 判别特征的算子（通道支 + 空间支门控特征图、无 softmax(QKᵀ) 主链），**一律使用本文件**。
> BAM / TripletAttention 在 CLAUDE.md 判别表曾落 `convolution` 类（无 softmax(QKᵀ) 判为非注意力）——
> **双分支门控骨架的经验以本卡为准**，其中 k×k conv 的基础经验（G1-G13 通用约束）仍回 `convolution.md` 取。
>
> **证据基础**：四个算子各 50 case 的完整优化轨迹——`36_BAM`（conv 平面化四步 2.1697 → 2.9430）、
> `57_ParallelPolarizedSelfAttention`（任务循环+档位化 1.9255 → 2.3237）、
> `61_SequentialPolarizedSelfAttention`（Split-K 2.3450 → 2.4783）、
> `64_TripletAttention`（IR store 行内仿射化 1.0580 → 2.6433）。
> 四者门控结构、分支内算子、kernel 数各不相同，却独立收敛到同一套瓶颈结论，这是本卡可外推的根据。
>
> ⚠️ **核心优化哲学**：这类算子的头号结构陷阱是 **[B,C2,N] 规模中间量物化**（O(B·C2·N) 流量 × 多遍），
> 第一动作永远是**结合律重排把它消掉**（§1 F1，架构级、Phase 2 就要做）。
> 消掉后主链退化为 GEMV 链 / 小 conv 链，瓶颈二分：
> **小 case = kernel launch 串行主导（~11-13µs/个）→ 融合 kernel / 双路径分派**；
> **大 case = 权重流 / 带宽 → 任务循环档位化 / Split-K / conv 平面化 / store 行内仿射化**。
> 生成时**禁止混用** `flash_attention.md` 的 online softmax / KV 分块经验——本类无 softmax(QKᵀ) 主链，
> FA 的 tile 放大摊薄 CV 同步策略在此完全不适用。

---

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 | 性能基准（geomean） |
|------|---------|---------|---------|------------------|
| ParallelPolarizedSelfAttention | `db-psa-par` | 通道 softmax-over-N 分支与空间 GAP/softmax-over-C2 分支**并行**，`out = x·(S+T)` **加性**融合 | 结合律消物化 → 7-kernel GEMV 链（全 vector 禁 dot）；任务循环 + constexpr 档位化；小 C 双路径融合 5-kernel | **2.3237**（50/50） |
| SequentialPolarizedSelfAttention | `db-psa-seq` | 通道支 T **先行**、空间支吃 channel_out，`out = x·T·S` **乘性**融合 | 同款 7-kernel 链严格串行；xmean 前置消一处串行等待；大 N 走 K2 Split-K；4 处 dtype 舍入点对齐 | **2.4783**（50/50） |
| BAM | `db-bam` | 通道 GAP→MLP(C/r→C) 分支 + 空间 k×k 膨胀 conv 分支，`out = x·(1+sigmoid(ch+sp))` | conv 走 shifted-1×1 dot（Cube 路径）；帧平面化消行循环；k=1 恒等平面；断带/偏移集中到纯搬运环节 | **2.9430**（50/50） |
| TripletAttention | `db-triplet` | CH/CW/HW **三分支**两两交互：zpool(mean/max) → 门控 conv → BN → sigmoid，`out = Σ/3` | permute 消解为索引映射（零 host 布局操作）；全 vector 禁 dot（C2=2/Cout=1）；13-launch 依赖链；store 行内仿射化 | **2.6433**（50/50） |

架构常量（ascend910_93 / Ascend910_9372，实测四个算子共用）：UB 192KB、Cube:Vector = 1:2（20 cube / 40 vector cores）。
全 vector 路径 `NUM_CORES = 40`；含 Cube dot 的路径 `NUM_CORES = 20`。

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. forward 主链是 `out = x ⊙ 门控`，门控量来自**通道支 `T[b,c]` 与空间支 `S[b,n]` 两路**——
   加性 `x·(S+T)`（op57）、乘性 `x·T·S`（op61）、`x·(1+sigmoid(ch+sp))`（op36）、三分支 `(o1+o2+o3)/3`（op64）；
2. softmax 存在但作用在 **N 维（空间 softmax）或 C2 维（通道 softmax）**，**不存在** `softmax(QKᵀ/√d)` 主链；
3. 参考实现含 `[B,C2,N]` / `[B,C2,H,W]` 规模中间张量（V_c/V_s/channel_out 物化）——头号消除对象；
4. BAM 形态：`GAP → MLP(C→C/r→C) → sigmoid` 通道注意力 + `k×k dilated conv → pool` 空间注意力；
5. Triplet 形态：三分支 permute → zpool(mean/max 拼两平面) → 小 conv → BN → sigmoid → 相加平均。

不满足 → attention 家族走 `attention_index.md`；纯卷积无门控走 `convolution.md`；
有 softmax(QKᵀ) 主链（哪怕带 CV 特征图）**不是本类**，回索引重判。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | 分支结构：并行加性 / 串行乘性 / 三分支平均？ | 决定依赖链与 kernel 数：并行两支可交错、串行有硬依赖（T 先行）、三分支互不依赖可乱序 launch（§3） |
| **Q1** | 分支内有没有 k×k conv（含 dilated）？ | 有真实 `[co,ci]×K²` GEMM → Cube dot + 平面化路径（§3.2/§3.3）；纯 1×1/Conv2d(C→C2) GEMV → 全 vector 路径（§3.1）。判据见 L1.2 |
| **Q2** | 归约维最大值 N_max / C2_max / C_max 是多少？ | 决定定宽 softmax/LN lane 档位——**覆盖不足 = 静默精度失败**（L1.4，op57 case48 实锤） |
| **Q3** | 参考实现的中间张量是什么 dtype？ | 决定舍入点数量与位置：op57 仅 load/store 两处降精度，op61 四处（L1.5） |
| **Q4** | 评测口径（geomean）？ | 小 case launch 主导时，省 1 个 kernel ≈ 净赚 12µs，直接进几何平均（§7.0） |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 ★★★ 结合律消除 [B,C2,N] 物化（本类头号杠杆，架构级）

参考实现把两个分支都写成大 matmul，中间量是 `[B,C2,N]` 规模。**必须**在 Phase 2 就用结合律重排消掉：

```
通道支: matmul([B,C2,N], [B,N,1])  ==  W_wv @ (Σ_n a[n]·x[:,n])   ← 先 a 加权池化，再 GEMV
空间支: matmul([B,1,C2], [B,C2,N]) ==  (a_s @ W_sv)·x[:,n] + a_s·b_sv   ← 先权重头重组，再逐 n 点积
```

- 消除后中间量只剩 `[B,N] + [B,C]` 级；`channel_out = x·T` 同理不物化（串行版把 T 折进空间支权重：`w2 = whei ⊙ T`，§3.1）。
- 50 case 中 C2×N 最大 ~1M×B，物化即 O(数十 MB) 流量 ×(写+读) 多遍，消掉是数量级差异。
- **Phase 4 再补做等于重写**——sketch 阶段就要写进架构决策。

### F2 权重复刻必须逐 draw 对齐（含「不使用但仍消耗 RNG」的权重）

**可直接套用的配方（四环节缺一即全错，op61 漏第 ③ 环全量失败）**：

1. **CPU 上 fp32 采样**——禁止直接以 fp16/bf16 采样；
2. 按 `kaiming_uniform_(a=√5)` / `uniform_` 的**同一 RNG 序列顺序**逐个 draw，顺序与次数和参考完全一致；
3. **不参与计算但仍消耗 RNG 的权重也要画**（op61 `ch_wv`：参考实现构造了它，漏画一次 draw 后续全部权重错位）；
4. 采样完再 `.to(device, dtype)`，不要采样时降精度。

- **判别**：失败形态是**全量错**而非部分错 → 先把实现的权重与参考逐个 `allclose` 单独验掉，再查 kernel。

### F3 任务循环 `grid = min(NTASK, NUM_CORES)` + 核内段循环（全部 kernel 统一）

- op57 实测 **+20.7%**：小 C case 的 grid（582/1158）远超 40 vector cores，program 发射开销累积主导
  （同流量 `[6,32,61,101]` 233µs vs `[1,640,21,69]` 89µs，2.6 倍差距）。
- kernel 内 `for t in range(pid, NTASK, NPROG)`，每 program 连续处理若干任务；与 FA 卡 L1.9 同一规则。

### F4 定宽 lane 档位表 + host 侧覆盖断言

- softmax_N / softmax_C2 / LN_C 的 lane 宽度取**定宽档位**（编译次数最少），档位表示例（op61）：
  `NB = 256|2048|16384 by N`、`C2P = 128|1024`、`CP = 128|2048`、`CB = 32|64`，全 2 的幂。
- **必须在 host 断言 `NB ≥ N / C2P ≥ C2 / CP ≥ C`**——op57 iter_0 用 NB=8192 跑到 N=9797 的 case，
  softmax 尾部**静默丢质量**，50 case 只挂 1 个，排查代价极高；换 16384 全过。
- 小 case 按实际尺寸收缩档位（定宽 16384 在 N=64 时纯空转），这是同一条规则的两面。

### F5 小向量的跨 program 同步点用「冗余重算」吸收，不加 kernel

- softmax/LN 这类 `[B,N]`/`[B,C]` 小向量归约：让**每个 program 冗余算全量**（定宽 lane 一次 load），
  而不是为它单开归约 kernel 或 atomic。
- 共享小标量用**单写者直存**：仅 `blk==0` 的 program 写（op61 的 `zb`/`T_d`），其余 program 冗余持有。

### F6 无原子归约三件套（按数据规模选一）

| 形态 | 方案 | 实例 |
|------|------|------|
| 大向量跨 program 归约 | partials 落盘（`[B,T_N,C]` 布局）+ 下游 kernel 归并 | op57 的 xsum_part |
| 每 (b, blk) 一个输出 | 单写者直存，天然无竞争 | op61 K2 的 pool/xmean |
| 全量标量统计 | `grid=(1,)` 单程序双趟确定性归约 | op64 的 bn_stats |

### F7 dtype 舍入点契约：load 升 fp32 → 全程 fp32 → 仅在「参考会舍入的位置」落 dtype

- op61 四处：`u`(K3 出)、`zc`(K4 出)、`T_d`(K5 出)、`out`(K7 出)——匹配参考 conv+matmul 链的中间 dtype 舍入；
- op57 两处：仅 load 后升 fp32 与最终 store 降精度——其容差宽松（fp16 atol=9e-2，sigmoid+LN 阻尼放大）。
- 判别方法：打开参考实现看**中间张量的 dtype**，不是猜；失败集合严格按 dtype 划分 ⇒ 先查舍入点。

### F8 空/退化 case 早退（host 侧逐字匹配参考）

- `B==0 / C==0 / H==0 / W==0` → `return x`；verify 按 0 元素通过，benchmark `PROFILER_COLLECT_FAIL`
  是**预期行为**（框架无法 profile 空 shape），不计入 geomean，不要当代码 bug 修。
- `[1,1,1,1]` 退化 case：`C2 = max(1, C//2) = 1`，softmax 单元 = 1，LN 单元素 var=0 → 除 `sqrt(eps)`；
  各档位取最小档 + mask 全保护即可，无需特判分支。

---

## §2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

> **本节共 12 条（L1.1 ~ L1.12）+ 3 条跨算子冲突判据，全部是 Phase 2 的硬性边界。**
> Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部 12 条**，不得只取前 N 条或按篇幅截断——
> 实测 op57 iter_0 漏掉 L1.4 覆盖断言，NB=8192 撞上 N=9797 的 case，softmax 尾部**静默丢质量**、
> 50 case 只挂 1 个，排查代价极高。
> 带 ★ 的（L1.1 / L1.4 / L1.8 / L1.9 / L1.10）是实测大收益或精度红线，尤其不能漏。
> L1.8 ~ L1.10 三条是 op36/op64 在 Phase 4 花了 6~11 轮才撞出来的结构级解法——
> 落卡之后升级为**首次生成就要写对**的架构项；同类算子再当 Phase 4 优化去试，等于重付一遍学费。
>
> **约束强度分级**（防"约束不到"，也防"约束死成唯一代码"）：
> - **物理红线**——违反必失败，不存在第二种写法：L1.4 覆盖断言、L1.8/L1.9 地址非负仿射、
>   L1.11 `other=-inf`、F2 draw 序列、L1.12 的"逐字核对参考语义"这一动作本身；
> - **架构目标**——约束的是**量级目标**，达成手段不限：L1.1（消 [B,C2,N] 物化；结合律 / 任务循环 /
>   Split-K 都是并列手段）、L1.10（store 行内连续；重构 program 粒度只是已验证的手段之一）；
> - **写法基线**——本类四算子统一验证的写法，非物理强制，有更优写法可替换并回填：
>   L1.3、F5、F6；
> - **门控判据**——数字全部来自本类 4×50 case 的 shape 分布，**跨算子使用前先按新分布重标定**：
>   F4 档位表、§4.1 Split-K 门控、§4.2 `C≤128`、冲突判据 3 的定界方法。
>
> 本卡约束的是**失败模式与量级目标，不是唯一实现路径**——不同架构路线只要 verify+benchmark 通过即有效；
> 新路线按 Phase 7 多样性保护**并列**归档进本卡，不覆盖旧经验。

### L1.1 ★★★ 禁止物化 `[B,C2,N]` / `[B,C2,H,W]` 任何中间张量

公式见 F1。**首次生成就必须做对，不是 Phase 4 的可选优化**——它是 sketch 的架构检查项（3.2b），
回退即架构违约；Phase 4 再补做等于重写。不是性能问题，是**不许写出来**。

**执行检查（3.2b 逐项核对用）**：列出 sketch 中 kernel 间传递的全部中间张量及元素数——
凡单个中间量达到 `B·C2·N` 量级（≈输入本身或更大）即违约；合法中间量为 `[B,N]` / `[B,C]` /
`[B,T,C]` partial 级（T 为分核段数）。

### L1.2 ★ 路径判定：GEMV(M=1) 禁 `tl.dot`；真实 `[co,ci]` GEMM 才用 dot

- PSA 双支 / Triplet 门控 conv（C_in=2、C_out=1）：全是 M=1 或深度 <16 的归约 → **全 vector 路径**，
  禁 `tl.dot`（Triplet 的 conv 用 Cube 16×16 阵列浪费 15/16）。
- BAM 的 k×k conv（cr→cr，真实跨通道 GEMM）→ shifted-1×1 GEMM + `tl.dot`（Cube 路径，详见 `convolution.md`）。
- **判别信号**：simulator 采集 `MMAD = 0%` ⇒ 纯 vector 定局；矩阵深度（min(co,ci)）< 16 ⇒ vector。

### L1.3 1D grid + int32 解码，禁 `%`

- `pid` 解码一律 `a - (a // b) * b`；禁 3D grid、`tl.swizzle2d`、kernel 内嵌套函数定义。

### L1.4 ★ 定宽 lane 覆盖断言（精度红线）

- host 侧断言 `NB ≥ N`、`C2P ≥ C2`、`CP ≥ C`（见 F4）；
- softmax 定宽 load 的 mask `other = -1e38`（参与 max/exp 后归零）；求和类 `other = 0`。
- **判别信号**：失败集合 = 恰好 1 个 case 且其 N（或 C2）为全组最大 ⇒ 直接查 lane 覆盖，
  不用扫参（op57 case48：NB=8192 撞 N=9797，50 case 只挂这 1 个）。

### L1.5 dtype 舍入点写进 sketch 的「精度匹配点」小节

- 逐个标出落 dtype 的位置（参考会舍入的位置才落），并写验证失败时的**回退序**
  （op61 实例：`qs 落盘舍入 → ζ 舍入 → g0 逐元素舍入`，逐级尝试）。

### L1.6 权重 / workspace 必须 host 缓存（shape 键控；zeros 的 halo 环只清一次）

- `_W_CACHE[(C, C2, dev, dtype)]`：miss 时按 draw 序列复刻（F2），一次构造终身复用。
- `_B_CACHE` / zeros workspace 按 shape 键缓存：`torch.empty` 部分全量覆写免清零；
  **zeros 的 halo/pad 环只清一次**，每轮只写 interior（op64 实测 +4.5%）。

### L1.7 host 侧只允许：`torch.empty/zeros` 分配、`.to(device)`、tuple 解包、kernel launch

禁止任何 torch 计算算子（matmul/softmax/conv/sigmoid/mean/max）与 `nn.*` 模块构造——退化检测会拦截。

### L1.8 ★★ conv 平面化：load/store 地址恒非负仿射

- **负地址**（`qi = pn + t_off - off0 < 0` 的 masked lane）→ `aicore exception 507015`。
  masked 正向越界实测安全，**负地址不行**。
- **禁止用 `tl.where` 钳位负地址**：破坏地址仿射性 → 编译器无法跨 tap 复用 UB buffer
  → `216KB > 192KB` UB 溢出 MLIR 编译失败（op36 实锤）。
- **正解（等价改写）**：把 off0 偏移从 load 侧挪到 store 侧——
  `block 平面位置 pn 的值 = Σ w_t·x[pn+t_off]`，写到帧位置 `q = pn + off0`
  （q 的正确 tap 输入恰为 `q + t_off - off0 = pn + t_off`）。两侧地址恒非负仿射。

```python
# ❌ load 侧带负偏移：pn + t_off - off0 < 0 的 lane 触发 507015 / tl.where 钳位触发 UB 溢出
x_t = tl.load(x_ptr + (pn + t_off - OFF0), mask=in_frame, other=0.0)
tl.store(out_ptr + pn, acc)
# ✅ off0 移到 store 侧：load/store 地址都是恒非负仿射
x_t = tl.load(x_ptr + (pn + t_off), mask=in_frame, other=0.0)
tl.store(out_ptr + (pn + OFF0), acc)
```

- **判别信号**：报 `aicore exception 507015` 或 MLIR `requires N bits ... UB` 编译失败
  ⇒ 直接查负地址 / `tl.where` 钳位，不要去查数值路径。
- **与 L1.9 耦合**：k≥2 平面化用本条 off0 改写；k=1 直接走恒等映射（L1.9），不要硬套偏移。

### L1.9 ★ flat 偏移两陷阱（k=1 恒等平面的前置知识）

1. **行宽不等 → 行漂移**：`q = pn + OFF0` 仅在读写行宽相同时保行。
   "carry-free"（OFF0 不跨界）是必要非充分条件——op36 首版读宽 ws1_w ≠ 写帧宽 ws2_w，直接行错位。
2. **OFF0 > 0 → padding lane 射入帧内**：`pn ∈ [FP, padded)` 的 `qn = pn+OFF0` 落在 `< FP_OUT` 区间，
   store mask 不拒绝 → relu(bias) garbage 污染。
- **正解**：恒等映射（OFF0=0，双侧 `pn < FP` mask 天然安全）+ 偏移由独立 `_expand_kernel` 承担。
- **判别信号**：输出整行错位 ⇒ 陷阱 1（读写行宽不等）；帧内特定位置出现 relu(bias) garbage ⇒ 陷阱 2（padding lane 射入）。

### L1.10 ★★ store 目标地址必须行内仿射（IR DiscreteMemAccess 红线）

- **症状**（IR 证据，op64）：`dst = (rows//H + PAD)*D2P + (rows%H + PAD)` 含整除/取模**跨行跳跃非仿射**
  → 编译器判离散地址 → 展开成 `scf.for` 标量循环逐元素 store
  （每迭代 1 float store + 4 次 wait/set_flag），比向量 store 慢一个数量级。
- **正解**：重构 program 粒度让 store 行内连续——`program=(n, c-block)` static_range 逐 c 归约，
  `dst = (c+PAD)*D2P + hv + PAD` 行内仿射 → 1D 向量 store。op64 两处（rowreduce/creduce）合计 **+13.0%**。
- **判别**：dump last_pass.mlir 搜 `DiscreteMemAccess`；或某 kernel 耗时与向量宽度严重不成比例。

### L1.11 max 归约的 mask `other` 必须 `-inf`（mean 归约贡献必须 0）

zpool/空间 max 归约中 `other=0` 会在**负值输入**时取错 max（op64 K1 明确标注）。
**判别信号**：仅含负值输入的 case 失败、非负 case 全过 ⇒ 直接改 `other=-inf`，不用扫参。

### L1.12 BN / LN 语义逐条钉死

- LN：**有偏 var（÷C 非 ÷(C-1)）**、eps=1e-5、γ=1 β=0（ones/zeros 无 RNG 消耗，但 draw 序列要留位）。
- BN（training batch 统计）：单通道每分支标量 μ/σ²、**无偏方差 ×n/(n-1)**、eps=1e-5、γ/β 取自权重元组；
  running stats 更新不影响输出，**不实现**。
- 三分支 z 平面顺序：**以参考实现的 cat 顺序逐字为准**（op64 实例是 `plane0 = mean`、`plane1 = max`，
  与 conv 权重 c2 维一一对应）——约束的是"逐字核对"这个动作，不是 mean/max 的先后；顺序错则分支输出互换。

### ★ 跨算子结论冲突的统一判据（三条，照搬会互相矛盾）

| # | 冲突 | 甲方结论 | 乙方结论 | **统一判据** |
|---|------|---------|---------|------------|
| 1 | 小向量归约：冗余重算 vs 拆 kernel | op57 K2 每 program 冗余算 softmax（不加 kernel） | op61 K2 拆三段 Split-K（+2 launch，+5.68%） | 看**每 program 工作量**：N ≥ 2048 且 B·GC ≤ 16（任务少、单任务长）→ 拆；launch 串行主导的小 case → 冗余。两判据都写进 host 门控 |
| 2 | 门控 conv 走 dot 还是 vector | op36 BAM conv 用 `tl.dot`（Cube） | op64 Triplet conv 禁 dot（vector） | 矩阵深度：真实 `[co,ci]` 且 min(co,ci) ≥ 16 → dot；C_in=2/C_out=1 → 标量权重 × 向量 tile |
| 3 | 档位边界外推 | op57 档位化 +20.7% | op57 档位扩展（C≤256/CP=256）~持平 | 档位表按**实测 shape 分布**扫描定界，命中分布后停止；新增档位前先用单 case 验证，不外推 |

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 PSA 主骨架（7-kernel GEMV 链；`db-psa-par` / `db-psa-seq`）

**并行加性版（op57，out = x·(S+T)）**，数学重排与 kernel 划分：

```
通道支: q_c[b,n] = W_wq·x[:,n]+b_q → a = softmax_N(q_c) → xtil[c] = Σ_n a·x
        → z_c = W_wv·xtil → y_c = W_wz·z_c → T[c] = sigmoid(LN_C(y_c))
空间支: xbar[c] = mean_N(x) → q_s = W_sq·xbar → a_s = softmax_C2(q_s)
        → whead[c] = Σ_i a_s·W_sv[i,c] → S[n] = sigmoid(whead·x[:,n] + ζ)
K1 流过 x（q_c GEMV + xsum partials）→ K2 softmax_N + xbar 归并 → K3 q_s
→ K4 a_s+ζ+whead+xtil（同 grid 三合一）→ K5 z_c → K6 y_c → K7 终融合（LN 冗余 + S + out）
```

**串行乘性版（op61，out = x·T·S）差异**：

1. 空间支吃 channel_out：`g0 = T ⊙ xmean`——**xmean 与通道支无依赖，前置到 K2 并行算**，消一处串行等待；
2. T 折进空间支权重：`w2[c] = whei[c]·T[c]`，K7 的 S 归约只做一遍 GEMV（`S = sigmoid(w2·x + ζb)`）；
3. 依赖链 K1→K2→K3→K4→K5→K6→K7 **严格串行**（6 个全局同步点）；
4. 双路径特化：C≤128 时 K3+K4、K5+K6 各融合 → 5-kernel（§4.2）；
5. Split-K：N≥2048 且 B·GC≤16 时 K2 拆三段（§4.1）。

### §3.2 BAM 管线（8 kernel + 平面化变体；`db-bam`）

```
通道支: K_gap（per-(b,cblk) 行块 GAP）→ K_mlp（双 GEMM，M=16 pad，b≤4 冗余重算 gemm1）
空间支: K_conv1（1×1, c→cr，写 ws1 interior）→ K_conv2a/b（k×k dilated, cr→cr，
        halo workspace 串联 ws1→ws2）→ K_conv3（cr→1 向量归约）→ K_pool（adaptive，仅 Hout≠h 时启动）
融合:   K_combine: out = x·(1+sigmoid(ch[b,c] + sp[b,hw]))（广播逐元素）
```

- **zeros workspace halo**：`ws[b,cr,h+2d,w+2d]` 一次清零，interior 由上一级 conv 写入，
  同时是下一级的 padded 输入——pad 语义由 halo 零承担，**禁 host 侧 F.pad/unfold/切片赋值**。
- **safe-adjust**：`eff = dil·(k-1)+1 > min(h,w)` 时收缩 (k,dil)——host 逐行复刻参考实现。
- 平面化变体（Phase 4 主收益，§4.3）：k=3 帧同尺寸时 `halo_zero → plane → compact` 三件套；
  k=1 恒等平面 + `_expand_kernel` 逆变换（§4.5）。

### §3.3 Triplet 管线（8 kernel / 13 launch；`db-triplet`）

```
三分支 permute 全部消解为索引映射（forward 零 host 布局操作）：
  CH 分支归约 W（stride=1）｜CW 分支归约 H（stride=W）｜HW 分支归约 C（stride=H·W）
K1 zp_outer ×2（cw,hw: 二维 tile 双平面 mean/max）+ K2 zp_rowreduce ×1（ch: 行矩阵 [C·H, W] block_ptr）
→ K3 gate_conv ×3（tap static_range，权重标量加载 2·K²≤98，读 zp padded 区）
→ K4 bn_stats ×3（grid=(1,) 双趟）→ K5 ap_plane / K6 ap_row / K7 ap_scalar（按 attn 形状选：行向量/逐 h 向量/逐 h 标量）
→ K8 combine3（(o_ch+o_cw+o_hw)·(1/3)）
依赖: {K1,K2} → K3 → K4 → {K5,K6,K7} → K8（分支间无依赖，launch 顺序无关）
```

- 每分支 (D1,D2,R,S,S1) 参数化：ch=(C,H,W,1,W)、cw=(C,W,H,W,HW)、hw=(H,W,C,HW,W)。
- K=kernel_size ∈{3,5,7} 由 `tl.constexpr` 特化，单一架构覆盖全部 case（无 host 门控分派）。

---

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构必须重新设计）

### §4.1 ★ Split-K 拆三段（op61 K2，+5.68%）

```python
# host 门控：任务少、单任务长才拆（launch 串行主导的小 case 不拆）
split_k = (N >= 2048) and (B * GC <= 16)
nseg = min(next_pow2(cdiv(N * C, B * GC)), cdiv(N, 256))
# K2a softmax_a  grid=(B,)            a 原地覆写 qc
# K2b pool_part  grid=(B*GC*nseg,)    partial 写 [t*CB] 布局，段间无竞争
# K2c pool_merge grid=(B*GC,)         [NSEGP, CB] tl.sum(axis=0) 树形归并
```

- **门控**：`N ≥ 2048 且 B·GC ≤ 16`（10/50 case 命中；B·GC=15 的 2 个 case 收益 < 3 kernel launch 开销，
  门控应收紧到 ≤12——门控阈值本身要按实测修剪）。
- 跨段树形归并 vs 顺序累加的 fp32 差异由下游 u 的 dtype 舍入吸收（精度无损）。
- 命中 case 8/10 提升（最高 **-51.3%** 时延）。

### §4.2 双路径融合 5-kernel（op57 +2.6%，op61 +0.41%）

```python
# host 双路径分派：小 C 用 5-kernel（省 2 次 launch），大 C 保持 7-kernel
if C <= 128:
    _k14_fused[grid](...)   # K3 并入 K4：任务内冗余算 qs 全量 GEMV
    _k56_fused[grid](...)   # K5/K6 合并：冗余算 zc 全量
else:
    _k3[grid](...); _k4[grid](...); _k5[grid](...); _k6[grid](...)
_k7[grid](...)
```

- **判据**：逐 kernel Event 计时显示小 case 端到端 ≈ launch 串行主导
  （op57 `[1,640]` 79.5µs ≈ 7×11µs）；冗余 GEMV 开销可忽略当 `C2×C ≤ 64×128` MAC。
- 大 C 走原 7-kernel 路径不受影响——双路径 host 分派，不是全局改参数。

### §4.3 ★ conv 帧平面化（op36 主收益，2.17 → 2.94）

- HOUT 行循环并入 dot 的 N 维：tile 数 `b·co_blocks·HOUT → b·co_blocks·⌈FP/BN⌉`
  （idx15 实测 84 → 4，dot 调用 756 → 36）。
- tap 输入列 = `pn + t_off` **纯加法常量偏移，无 div/mod**；输出 halo 列写 garbage，
  下游读侧 `IN_OFF/IN_CORE` constexpr 屏蔽。
- 前置：帧平面含 halo 断带——**断带转换不可消除，只能集中到最便宜的纯搬运环节**（compact/expand）。

### §4.4 halo_zero 三段式纯连续（44x 提升）

```python
# 段选择 = 每 tile 一次标量 3 路分支，段内整块同构 mask（绝不逐元素 tl.where）
seg = (py < d) - (py >= HOUT + d)          # -1: top 带, 0: 核心行, 1: bottom 带
if seg != 0:    tl.store(top_bot_2d_block, 0.0)      # 整带全行连续 2D store
else:           tl.store(core_row_side_steps, 0.0)    # 核心行两侧步进连续段
```

- 行循环版 206-445µs（每行 2 个微型 store 摧毁 MTE 流水，~0.4-1.5µs/行）；
  2D `tl.where` 混合 mask 版 89-143µs（混合 mask 使 store 走逐元素 scatter 慢路 ~0.8GB/s）；
  **三段式** 4.7-6.4µs。
- 通则：**清零/搬运类 kernel 的 mask 必须整段同构**，混合 mask 必然走 scatter 慢路。

### §4.5 k=1 恒等平面 + expand 逆变换（k=1 占 op36 的 10/50 case）

- k=1 conv = 逐像素 GEMM：恒等 flat 无偏移映射（OFF0=0）+ 双侧 `pn < FP` mask，padding lane 天然安全；
- 偏移/断带由独立 `_expand_kernel`（compact 逆变换）承担——把最容易错的偏移逻辑隔离在最便宜的纯搬运里。

### §4.6 tap 分组两阶段（op64，+6.5%）

- `total < 16`（欠并行）时 `TG = min(2k, max(2, cdiv(40, total)))` 自适应分组：
  小 B 小 C 的门控 conv 从 ≤16 program 补足 ~40 program；kernel 体零改动，模块级统一三分支分派。

### §4.7 宽度自适应 `_bw()` 8 档（op64，+2.9%）

- 小维度（W/H ≤ 23）按 8 档宽度函数选 BLOCK，替代定宽 32；配 `BLOCK_H = next_pow2(H)`。

### §4.8 host 缓冲复用（op64，+4.5%）

- zp 的 pad 环零填充一次（workspace 缓存），每轮只写 interior——省掉每轮整张 zeros 重分配重清零。

### §4.9 ★★ store 行内仿射化（op64 IR 轮，+13.0%，全算子最大单点）

```python
# ❌ dst 含整除/取模，跨行跳跃非仿射 → IR 判 DiscreteMemAccess → scf.for 逐元素标量 store
dst = (rows // H + PAD) * D2P + (rows % H + PAD)
tl.store(zp_ptr + dst, val)                       # 每迭代 1 float + 4 次 wait/set_flag
# ✅ 重构 program=(n, c-block) 逐 c 归约（RPB=2 行 static 展开），dst 行内仿射连续
dst = (c + PAD) * D2P + hv + PAD                  # c 为 static_range 循环变量
tl.store(zp_ptr + dst, val)                       # 1D 向量 store
```

- IR 证据：`last_pass.mlir` 中 `scf.for` 标量循环 + `ExtractedLoadOrStore` + `DiscreteMemAccess`，
  每迭代 UB 标量 load + `store_ubuf_to_gm_1d`（1 float）+ 4 次 wait/set_flag——64 值 × 2 通道全部标量 store。
- 修法落点两处（rowreduce / creduce），合计 **+13.0%**。
- **方法论**：Phase 4 常规点耗尽后，dump IR 搜 `DiscreteMemAccess`——凡是 dst 含 `//` `%` 跨行跳跃的
  store 都是嫌疑；这项检查在源码层面看不出来（写法合法，编译后才知道退化）。

### §4.10 分核 BN 收缩（op36，case 级 +5~7%）

- 大帧（FP≥1024）欠并行时 BN 减半提升 grid 占用，仅 BM>16（cap=256）场景启用——
  BM=16/cap=512 时宽 tile 的 MTE 效率优于并行度，**收缩不是万能**。

---

## §5 Phase 4 优化点清单

映射到 `triton-latency-optimizer` 的优化点编号（四算子日志实测命中）：
**#3 分核**（任务循环 +20.7%、K2 Split-K +5.68%、tap 分组、BN 收缩）、
**#12 Grid 形状与多路径特化**（双路径融合 5-kernel）、**#18 Kernel 分裂**（tap 分组两阶段 +6.5%）、
**#22 Latency-Bound 循环维度 Tile 合并**（conv 帧平面化，op36 主收益 2.17→2.94）、
**#31 IR 分析**（store 行内仿射化 +13.0%，两处）。
**#4 离散访存**（block_ptr 替换）、**#7 Pass 消除合并**（双向归约 / softmax_a 并入）、
**#11 Load 重排序**（g-major）在本类**实测证伪**，命中后直接跳过，证据见 §5.2。
本类不命中 **#30（FA 专用）**——无 softmax(QKᵀ) 主链，FA 的 tile 放大 / online softmax 经验整卡不适用。

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|---------|
| 1 | ★★★ 结合律消 [B,C2,N] 物化 | 架构级（Phase 2/3 就要做） | 全部 PSA 形态 |
| 2 | ★ 任务循环 + constexpr 档位化 | **+20.7%**（op57） | grid 超核数、小 C case 多 |
| 3 | ★★ store 行内仿射化（IR 驱动） | **+13.0%**（op64） | dst 含 div/mod 的 store |
| 4 | ★ conv 帧平面化（+ 三件套 + k=1 恒等） | 2.17 → 2.94 累计（op36） | k×k conv 行循环形态 |
| 5 | tap 分组两阶段 | +6.5%（op64） | total<16 欠并行 |
| 6 | Split-K 拆三段 | **+5.68%**（op61） | N≥2048 且 B·GC≤16 |
| 7 | host 缓冲复用（pad 环零一次） | +4.5%（op64） | zeros workspace 形态 |
| 8 | 宽度自适应 8 档 | +2.9%（op64） | 小维度 W/H |
| 9 | 双路径融合 5-kernel | +2.6%（op57） | C≤128 且 launch 主导 |
| 10 | 分核 BN 收缩 | case 级 +5~7%（op36） | FP≥1024 且 BM>16 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

> 证伪结论**绑定当时的架构形态**（tile 尺寸 / kernel 划分 / 任务粒度）。发生结构性改动后，
> 相关条目须用单 case 重测一次再决定是否沿用——"creduce 分档 +4~5% 有效但被行内仿射化整体取代"
> 即属此例：参数级解法在结构级解法落地前后结论相反。

| 方向 | 结果 |
|------|------|
| **pass-merge：K4 xtil 遍顺带产出 zs_part，K7 改读 partials（x 遍数 4→3）** | **-33%**（op57 opt_iter_1）。同一 tile 上 axis=0 + axis=1 **双方向归约**破坏循环流水/寄存器分配，K4 时间翻倍以上；省的只是一遍流式 x 读。**pass-merge 必须保证顺带计算与原归约同方向** |
| block_ptr 全量替换 masked load | **-24~-29%**（op36 opt_iter_0）。小 tile 下 block_ptr 固定开销超过 gather 节省 |
| g-major 任务重排 | -0.25%（op61 opt_iter_2），噪声级放弃 |
| BPR=128（行块翻倍） | -1.4%（op64 opt_iter_11） |
| masked load 形态改造 | -2.1%（op64 opt_iter_14） |
| tap 阈值 40（放宽分组门控） | -1.7%（op64 opt_iter_15） |
| rowreduce BLOCK_H 自适应 | 噪声（op64 opt_iter_19，端到端证伪——后由 IR 行内仿射化整体取代） |
| creduce BHW2 分档 64/32/16 | +4~5% 有效（op64 opt_iter_16-18），但被 §4.9 行内仿射化**整体取代**——同瓶颈的参数级与结构级解法，先试结构级 |
| C≤256/CP=256 档位扩展 | ~持平（op57 opt_iter_3），取 2.3181 < best 放弃 |
| softmax_a 并入 pool_part（#7 融合） | **流量核算证伪**：+61% GM 读 vs 省 1 launch（op61 simulator 轮）——融合前先算流量账 |
| IR/simulator 终局 | multibuffer 编译器自动双缓冲全覆盖；全 kernel 无 MMAD、无 Cube 空等、无元素级标量降级；**剩余开销 = launch 固定成本 ~11-13µs/kernel**，无可落地优化点 |

### §5.3 结构性下限：小 case 的 launch 墙

- 每 kernel launch ≈ **11-13µs** 固定成本（op57/op61 实测 `[1,640]` 79.5µs ≈ 7×11µs）。
- 7-kernel 链的小 case 下限 ≈ 80µs——**省 1 个 kernel 净赚 ~12µs**，这是双路径融合（§4.2）的全部动机。
- 大 C case 的另一堵墙：带宽（op57 `[2,1280]` 达 ~205GB/s，x 三遍流是算法架构下限，
  权重流 MTE2 为必然流量）。**两堵墙的修法互斥**：launch 墙拆/融合 kernel，带宽墙减遍数——先定位在哪堵。

---

## §6 精度闸门（先过闸门，再谈性能）

- **容差**（verify.py，op57/op61 实测口径）：fp16 `atol=9e-2 / rtol=2^-10`、bf16 `1e-1 / 2^-7`、fp32 `1e-3 / 2^-13`。
- **诊断顺序（按此序排查，跳步会陷入长尾 debug）**：
  ① **全量错** ⇒ 权重 draw 序列（F2）——先把实现的权重与参考逐个 `allclose` 单独验掉，再查 kernel；
  ② 失败集合**按 dtype 严格划分** ⇒ 舍入点（L1.5）；
  ③ **恰好 1 个 case 且其 N/C 全组最大** ⇒ 定宽 lane 覆盖（L1.4）。
- **GAP(conv) == conv(GAP) 线性等价**（op57 空间支 `q_s = W_sq·xbar`）——注意只在**线性**层前成立，
  参考 GAP 之后有 ReLU/BN 的不可交换。
- max 归约 `other=-inf`、softmax `other=-1e38`、求和 `other=0`（L1.4/L1.11）。
- 正式结论一律走 verify.py；扫参阶段单 case 脚本秒级淘汰。

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.0 ★ 官方 geomean 每 case 等权

- op57 opt_iter_0 把 `[6,32,61,101]` 从 0.974x 拉到 2.519x（一个 case），几何平均 1.9255 → 2.2650——
  **修最慢的 case 与把快的 case 翻倍贡献相同**。慢 case 常是「grid 超核数的小 C」而非大 shape。
- 空 case（B=0）benchmark `PROFILER_COLLECT_FAIL` 是预期口径，不计入 geomean（F8）。

### §7.1 定位手段

- **逐 kernel Event 计时**：判断 launch 主导（各 kernel 时长 ≈ 常数 ~11µs 且与 shape 弱相关）。
- **kernel_details.csv Duration 占比**：每轮优化后找新瓶颈（op36 由 conv2b 占 76% 触发平面化三件套）。
- **simulator 终局采集**（MMAD/SCALAR/MTE 占比 + 热源码行）：下「硬件极限 / 无可优化」结论前**必须**采集；
  本类四个算子终局画像一致：全 vector、MMAD=0%、SCALAR 为固定开销非降级。
- 双探针噪声闸门、≤3% 改动包夹复测（同 FA 卡 §7.1）。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 直接翻译参考实现，物化 [B,C2,N] | 参考实现用大 matmul 表达 | F1/L1.1 结合律重排 |
| 单个大 N case 精度失败 | 定宽 lane NB < N_max，softmax 尾部静默丢质量 | L1.4 覆盖断言 |
| 权重全量错 | draw 序列漏位（含不用仍耗 RNG 的权重） | F2（op61 ch_wv 实锤） |
| `aicore exception 507015` | masked lane 负地址 | L1.8 off0 移 store 侧 |
| MLIR 编译失败 UB 溢出（216KB>192KB） | `tl.where` 钳位负地址破坏仿射，跨 tap 无法复用 buffer | L1.8 等价改写 |
| flat 输出行错位 | 读写行宽不等，`q=pn+OFF0` 不保行 | L1.9 陷阱 1 |
| relu(bias) garbage 污染输出 | OFF0>0 时 padding lane 射入帧内 | L1.9 陷阱 2（恒等+expand） |
| 某 kernel 比同规模慢 10 倍 | store dst 含 div/mod 非仿射 → DiscreteMemAccess 标量循环 | L1.10/§4.9 |
| max 归约结果错（负值输入） | mask `other=0` 参与 max | L1.11 `other=-inf` |
| 标量 bool 与向量 mask 组合再 507015 | 谓词组合触发编译缺陷（op36 conv2b `kmask & row_ok & col_ok`） | 行判定改标量 `if row_in:` 直接跳过（顺带省 tap load） |
| 平面化后下游读入 garbage | 输出 halo 列未清 | halo_zero 三段式 / IN_CORE constexpr 屏蔽 |
| 清零 kernel 极慢（百 µs 级） | 行循环微型 store / 混合 mask scatter 慢路 | §4.4 三段式纯连续 |
| B=0 case benchmark 假失败 | profiler 无法采集空 shape | F8 预期口径，不计 geomean |
| 三分支输出互换/颠倒 | z 平面 cat 顺序与 conv 权重 c2 维不对应 | L1.12 plane0=mean / plane1=max |
| 融合 kernel 越融越慢 | 双方向归约破坏流水 / 融合引入额外 GM 读 | §5.2 前两条：同方向才 merge，融合前算流量账 |

---

## §9 归档记录

| 算子 | arch / NPU | geomean | 轮次 | 关键路径 |
|------|-----------|---------|------|---------|
| 36_BAM | ascend910_93 / NPU4 | 2.9430（target 2.0 达标） | Phase 3×1 + Phase 4×6 | conv 平面化三件套 + k=1 恒等平面 + BN 收缩 |
| 57_ParallelPSA | ascend910_93 / NPU13 | 2.3237（target 100 经验性高值未达） | Phase 3×2 + Phase 4×6 | 任务循环+档位化 +20.7% → 双路径融合 +2.6% |
| 61_SequentialPSA | ascend910_93 / NPU2 | 2.4783 | Phase 3×2 + Phase 4×5 | K2 Split-K +5.68%；IR/simulator 终局干净 |
| 64_TripletAttention | ascend910_93 / NPU13 | 2.6433 | Phase 3×1 + Phase 4×11（含 IR×2 + sim×1） | tap 两阶段 +6.5% → IR store 行内仿射化 +13.0% |

四个算子 verify 均为 **50/50**；完整轨迹见各自工作目录 `output/opt_iter_*/log.md` 与 `report.md`。

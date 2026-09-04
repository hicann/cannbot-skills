---
name: linear
description: Linear（线性 / 低秩投影注意力）类算子（Linformer 及同类 K/V 低秩投影降维后再 attention 的算子）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧与 Phase 4 优化点清单（含证伪方向全表）
metadata:
  type: reference
---

# Linear（线性 / 低秩投影注意力）类算子优化经验

本文档是 **"先用可学习低秩矩阵把 K/V 的序列维投影降维，再做 attention"** 这一类线性复杂度注意力算子的经验合集，覆盖 Phase 2/3/4：

- **§1 通用经验**：跨形态共有的工程约束
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与姊妹卡的分工**

> ⚠️ **本文件与 `flash_attention.md` 的分工（重要）**：
> 按 `level4写法分类体系.md` 的写法口径，本卡负责 **二 压缩类 · 「低秩 / latent 投影」细分**
> （K/V 取数前插一段可学习低秩投影），与 `gqa.md`（二·1 head 共享）、
> `block_sparse_attention.md`（三·2 块跳过）平行；叠加形态两份都读。
> 凡满足 §0.1 判别特征（K/V 经可学习 `[P,S]` 低秩矩阵投影、序列维 `S→P` 降维后再 attention）的算子，
> **一律使用本文件**。它与 FA 类的本质区别：本类**不对全 S 滚动 online softmax**，而是先把 S 投影成小 P（`P≪S`），
> attention 在 `[B,H,P,D]` 上做——因此 §0.1 的判别、瓶颈分布（LR1）、头号杠杆（§4.1）都与 FA 卡不同。
>
> **证据基础**：一个 Linear 类算子 `15_LinformerAttention` 的完整两阶段优化轨迹（50 case，fp16/bf16/fp32 混合）——
> 阶段一 `round-1~10`（0.089×→1.48×）+ 阶段二多智能体 `v2~v16`（1.494×→**2.0315×，破 2.0×**）。
>
> | 阶段 | 轨迹 | 终值 | 口径 |
> |---|---|---|---|
> | 阶段一（编排器 round1-10） | 0.089×→1.48× | 1.48× | 旧 benchmark |
> | 阶段二（多智能体 v2-v16） | 1.494×→破 2.0× | **2.0119–2.0315×** | 旧 benchmark |
> | 同一 v15b 代码复测 | kernel 未变 | **2.3747×** | 更新后 benchmark（§7.3） |
>
> 硬件 / 工具链：**Ascend 910B3**（Ascend910_9372，**20 AIC**）/
> triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0。
> ⚠️ 与 GQA 家族卡（`gqa.md`）的 910B2C / 24 AIC 是**不同 SKU**——本卡 L1.1 的 fp32 dot 例外、
> L1.3 的 BLOCK 上限等结论与 SKU 绑定，跨芯片外推前必须重验。
>
> **单算子证据**：本卡的高置信结论（cube-bound 硬顶、contraction 维 tile 加深、copy 消除、多输出融合证伪）均经实测；
> 但跨 Linear 子形态（如 Nyström / Performer / kernel attention）尚未验证，凡涉及其它子形态的结论已标注「待跨算子验证」。
>
> ⚠️ **核心优化哲学**：这类算子 86% 的 kernel 时间在 **cube-bound 的投影 GEMM + attention dot** 上——
> triton GEMM 打不过 CANN 原生 `aclnnMatmul`、且 `torch.matmul` 被 AST 门禁用，**是硬顶，不要硬攻**。
> 真正能动的只有 **14% 的低秩归约（`kv_reduce`）和搬运 copy**。一切有效优化等价于：
> **「先用瓶颈分布实测把 cube-bound 与可动项分开 → 消除一切搬运 copy → 把唯一可动的 reduction kernel 的 contraction 维 tile 开到 UB 上限」**。
> 生成时**禁止**套用 `flash_attention.md` 的 KV 分块 / online softmax 策略——本类不滚动全 S。

---

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| Linformer | `linear-linformer` | q/k/v/out 四个 D×D 投影 + **可学习 `w_kr[P,S]`/`w_vr[P,S]` 把 K/V 序列维 `S→P` 降维**（共 6 个 GEMM） → `softmax(Q@Kᵣᵀ/√d)@Vᵣ`，attention 在 `[B,H,P,D]` 上 | 投影/attention 双段 cube-bound（不动）；攻 **`kv_reduce` 低秩归约 + 搬运 copy** |
| Nyström / kernel 线性注意力 | `linear-nystrom`（待验证） | 用 landmark 子集近似 `softmax`，K/V 经采样矩阵降维 | 同上思路待跨算子验证 |
| Performer / 随机特征线性注意力 | `linear-performer`（待验证） | 用随机特征 `φ(·)` 把 softmax 近似为线性核，K/V 先过 `φ` | 待验证 |

### §0.1 判别特征（决定用不用本文件）

满足**全部**即用本文件：

1. 计算图含一个**对 K/V 序列维的显式低秩投影**：`kr[...,p,hd]=Σ_s k[...,s,hd]·w[p,s]`（`w` 是可学习 `[P,S]` 矩阵或采样/特征矩阵），把序列长从 `S` 降到 `P`（典型 `P≪S`，op15 `P∈[2,53]`、`S` 最大 897；⚠️ 1440 量级是 `d_model` 的取值，别与 S 混淆）；
2. attention 在**降维后**的 `[B,H,P,D]` 上做（`Q@Kᵣᵀ` 是 `[P,P]` 级，不是 `[S,S]` 级）；
3. **不**对全 S 做 KV 分块 + online softmax（否则归 `flash_attention.md`）。

边界：若 `P` 退化到 `≈S`（无实质降维）或不含低秩投影 → 归 `flash_attention.md`。

### §0.2 ★ 形态识别四问（Phase 2 第一步必须回答，答案决定后续哪些章节适用）

| # | 问题 | 影响 |
|---|------|------|
| **Q1** | 投影 GEMM（q/k/v）与低秩归约（kv_reduce）**分别**占总 kernel 时间多少？ | 决定主攻是 cube-bound 段（LR1 硬顶、基本不动）还是 reduction/copy 段（§4 头号杠杆） |
| **Q2** | 有没有把降维前的 K/V 先 `.contiguous()`/transpose 再喂归约？ | 有 → §4.2 copy 消除是 +14~16% 的大笔；没有则不适用 |
| **Q3** | 参考实现的 attention/投影在什么精度上算？ | 决定 dot 能否用低精度操作数、以及 bit-exact 契约（§6） |
| **Q4** | 评测口径？ | 几何平均 ⇒ **每 case 一票**，修最慢 case 收益 = 把快 case 再翻倍（§7.0） |

---

## §1 通用经验（跨形态，首次生成必须遵守）

以下约束是本类算子**共有**且**未在 `flash_attention.md` 覆盖**的工程约束。`tensor-transform.md` G1（动态 num_cores）/ G4（grid 不超核数）/ G7（contiguous）、`flash_attention.md` F1（归约 padding 显式排除）/ F4（UB 预算）此处不再重复，引用时标注。

### LR1 cube-bound 硬顶：投影 GEMM 与 attention dot 不要硬攻

- **必须** 优化前先用 `benchmark.py` 的 kernel `Duration(µs)` profiling 把 kernel 按耗时占比 + 性质分类（方法见本条 + §7.1 天花板估算）。
- **禁止** 在未确认瓶颈分布前，凭直觉主攻投影 GEMM 或 attention dot——它们是 cube-bound，triton GEMM 打不过 CANN `aclnnMatmul`，而 `torch.matmul` 又被 AST 门禁用。
- **Why（op15 实测）**：attention 占 47.8% + 投影 GEMM 占 38.2% = **86% 已触 cube 天花板**，10 轮优化里所有攻这 86% 的尝试（多输出融合 v8/v12、fp16 dot v13、num_stages 流水线 v14）**全部回归或中性**（§5.2）。真正破 2.0× 的一笔来自剩下 14% 的 `kv_reduce`。
- **典型应用**：所有 Linear 类算子；阶段二的第一动作永远是「测瓶颈分布」。

### LR2 禁用手写 `time.time` 探针，唯一可信的是 kernel `Duration(µs)`

- **绝对禁止** 用 `time.time`/`time.perf_counter` 包单次 op 调用做性能诊断。
- **必须** 用 `benchmark.py` 的 kernel `Duration(µs)`（来自 `kernel_details.csv` 的片上执行时间）。
- **Why（op15 实测）**：手写探针把每个 op 的 Python/同步开销拉平到 ~0.23ms，**假判为 "launch-overhead-bound"**，把整轮优化方向带偏（阶段一停在 1.48× 误判"投影无解" partly 源于此）。换 kernel `Duration` 后才看清真实瓶颈分布。
- **判别信号**：不同 shape 的"时延"被测成几乎相同的常数（~0.2ms 量级）⇒ 探针被 host 开销主导，立即弃用。

### LR3 几何平均口径 ⇒ 优化投在最慢的 case

- **必须** 从 `perf_result.json` 的 `per_shape_results` 按 `speedup_vs_torch` 升序，专攻 `<1.0×` 那批的共同特征（op15 是 **fp32 / 大 seq**），为它们单开分派路径。
- **禁止** 全局改 tile 参数"让大 case 再快一点"——几何平均下它对总分几乎无贡献，且常同时把快 case 拖慢成净负收益（op15 上 GROUP_M/BK 微调全是噪声，§5.2）。
- **Why（op15 实测）**：round-8 的 Kr/Vr hoist 专门修大 seq 慢 case（case41 0.522→1.001、case40 0.581→0.963、case46 1.202→1.655），sub-1.0× 从 6 个降到 2 个，+5.14%——比任何"全局提速"都有效。

---

## §2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

> **本节 8 条（L1.1 ~ L1.8），全部是 Phase 2 硬性边界。** Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部 8 条**。
> 带 ★★★ 的两条（**L1.3 reduction contraction tile 取上限**、**L1.4 copy 消除**）是收益最大的两笔，尤其不能漏。

### L1.1 ★ `tl.dot` 契约：投影与 attention dot 都用原生 dtype 操作数 + fp32 累加器

- **必须** 投影 `x@Wᵀ`、低秩归约 `w@k`、attention `Q@Kᵣᵀ`/`PV` 一律用 `tl.dot`，操作数**保持原生 dtype**，累加器 fp32（`tl.dot` 默认 `out_dtype=tl.float32`）。
- **禁止** 用逐维标量循环 / rank-1 外积代替矩阵乘（Cube 完全空转，benchmark 超时无数据——op15 round-1 的 0.089× 正是此病）。
- **禁止** 把操作数升 fp32 再 dot（走异构路径，是"fp32 禁 `tl.dot`"误传的真正来源；原生 dtype + fp32 acc 精度 50/50 全过，见 §6）。
- **⚠️ 例外——参考实现显式 `.float()` 的段，fp32 操作数是精度契约**（op15 生产版即如此）：golden 的 attention 是
  `q.float() @ k.float().T`、`(weights @ v.float())`，则 kernel 必须 `.to(tl.float32)` 后 dot 才逐位同构。
  op15 实测：910B3 上 fp32 `tl.dot(allow_tf32=False)` **逐位精确且不慢**；反例 v13 把 attention dot 降到 fp16 操作数
  精度过但**慢 5%**（小 tile `BP≤64,HD≤128` 下 cube 是 latency-bound，fp16 的 2× MAC 速率兑现不出）。
  ⇒ **禁止的是无契约依据的升精度，不是 fp32 dot 本身**（与 `flash_attention.md` L1.1 的表述差异源于 SKU：
  那张卡的证据来自 910B2C，本卡来自 910B3，跨 SKU 先重验）。
- **判别信号**：`aic_mac_ratio < 10%` 且代码里有 `for d in range(...)` ⇒ 立即重写为 `tl.dot`，不用扫参。

### L1.2 投影权重复刻：CPU fp32 采样 + `x @ Wᵀ` 语义

- **必须** 权重 **CPU** 上、**先 fp32 采样再 `.to(dtype)`**、构造顺序/次数与参考一致、rng save/restore 一致。
- **必须** `nn.Linear.weight` 是 `[out,in]`，投影写 `y = x @ Wᵀ`；方阵下 shape 检查会放行错误写法 ⇒ **交换 stride**：kernel 实参传 `w.stride(1), w.stride(0)`。
- **Why**：这两条（权重错 × 投影转置错）各自独立致命且互相掩盖，只修一条 `passed_cases` 仍为 0。详见 `flash_attention.md` L1.11 / L1.12（原 `attention.md` 已退役删除，条目已并入该卡；本类直接承接，不重复）。

### L1.3 ★★★ 低秩归约 kernel 的 contraction 维 BLOCK **必须按面积预算取上限，禁止硬编码小值**

> ★ **这是本类算子最大的单点杠杆**（op15 上单 kernel −22.5%、端到端破 2.0×），且**必须在首次生成就做对**。

- **禁止** 写死 `BLOCK_S = 64` 这类小常量。
- **必须** 按面积预算由 host 算出，取到 UB 允许的最大值：

```python
# 低秩归约 kr[b,h,p,hd] = Σ_s k[b,s,h,hd] · w[p,s]，contraction 维是 S（op15 最大 897）
BLOCK_P  = min(64,  max(16, next_pow2(proj)))      # P 维（投影后维度，小）
BLOCK_HD = min(128, max(16, next_pow2(hd)))        # head_dim
BLOCK_S  = min(256, max(16, next_pow2(s)))         # ★ contraction 维开到 256（不是 64！）
```

- **Why（op15 实测）**：contraction 维从 64 提到 256：
  - **更少的 K 迭代**：`S=897`（op15 最大序列长）时 K-loop 从 15 次降到 4 次，循环/地址开销大减；
  - **更深的 cube reduction**：`tl.dot(w[BP,BS], k[BS,BHD])` 的 contraction 维 64→256，cube MAC 利用率显著提升；
  - **UB 安全**：`acc[64,128]fp32=32KB + w[64,256]fp16=32KB + k[256,128]fp16=64KB = 128KB < 192KB`；
  - **bit-exact**：只改 tile 尺寸不改计算顺序，summation regrouping ~1e-7，远在 MERE 门下（§6）。
- **实测**：`_kv_reduce_strided_kernel` **6.48 → 5.02 µs（−22.5%）**，端到端 1.9921× → **2.0315×（破 2.0×）**。
- **配套**：开到 UB 上限要**放大到编译失败为止**——UB 边界由编译器报数字（`ub overflow, requires N bits while 1572864 bits available`），**不可简单阈值预测**（同 `BLOCK_S=256`，不同 dtype/组合占用不同）。
- ⚠️ **小 shape 要按实际尺寸收缩**（`min(..., next_pow2(s))`）：固定大 tile 在 `S < BLOCK_S` 时纯空转。
- **How to apply:** 见 L3.1 `_kv_reduce_strided`。

### L1.4 ★★★ 消除一切搬运 copy：host 不做 permute/contiguous，输出按目标布局直接写出

- **必须** 直接把原始布局的 stride 传进 kernel，输出按目标布局分配；**用一个 strided-load 的归约 kernel 直接吃原始 `[B*S, H*HD]` 布局**，不在 host 侧先 `.contiguous()`/transpose。
- **禁止** `k.permute(...).contiguous()`、`torch.zeros`+`copy_` 补齐、attention 输出再 permute 回来。
- **Why（op15 实测）**：搬运 copy（`aclnnInplaceCopy_Transpose` / `ViewCopy`）是纯 GM 往返、零计算贡献。两轮消除合计 **+30%**：
  - **v4**：attention 输出按 `[b,S,H,hd]` 布局直接写出（kernel 内 stride 重排为 `out.stride(0),out.stride(2),out.stride(1),out.stride(3)`，forward 做免费 reshape），消 attn2 后的 `.contiguous()` → **1.494× → 1.7393×（+16%）**；
  - **v6**：strided reduce kernel 直接读 `k2d[B*S,H*HD]`（s 维 strided，stride=D），消低精度 k/v 路径上的第二对 Transpose → **1.739× → 1.9921×（+14.5%）**。
- **判别信号**：profiling 时间归属里出现 `Transpose`/`ViewCopy`/`ZerosLike`/`Slice`/`contiguous` ⇒ 直接删，不用扫参。

### L1.5 ★★ reduction / attention kernel 共享 `grid = min(核数, tasks)` + 核内步长循环

- **必须** 算子内**每个** `@triton.jit` kernel（投影、归约、attention、任何辅助 kernel）统一：
  ```python
  total_tasks = <该 kernel 的任务数>            # 归约是 b*H、attention 是 b*H*cdiv(P,BP)、投影按 M 分块
  grid = (min(NUM_CORES, total_tasks),)         # ✅ 永远不越界
  # kernel 内
  for task in range(tl.program_id(0), total_tasks, NUM_PROG): ...
  ```
- **禁止** `grid=(total_tasks,)`（任务被重复计算）或 `grid=(核数,)`（小 shape 空核调度）。op15 上 round-8 把 attention 从"扁平交织 persistent 调度"重构为"(b,H)-major 连续 tile + 满核占用"正是本条，+5.14%。
- ⚠️ `NUM_CORES` 必须**常量 constexpr**（随 case 变化触发编译爆炸）；`grid` 是启动参数可随 case 变。

### L1.6 attention 段：Kr/Vr 跨 S-tile hoist（线性注意力专属，FA 卡无此项）

- **必须** attention kernel 按 (b,H)-block 推进时，**Kr/Vr 每 block 载入一次、hoist 出 S-loop、内层 S-tile 复用**，把 Kr/Vr 的 HBM 流量从 `O(B·H·NUM_S)` 降到 `O(B·H)`。
- **Why（op15 round-8 实测）**：1.4061× → 1.4784×（+5.14%），**靶向修复大 seq 内存 bound case**（case41 0.522→1.001、case40 0.581→0.963），sub-1.0× 从 6 个降到 2 个。
- **与 FA 的区别**：FA 是对全 S 滚动 online softmax，Kr/Vr 必须进循环；本类 attention 在降维后的 `[B,H,P,D]`（P 小）上，Kr/Vr 是**降维结果**、尺寸小、适合 hoist。

### L1.7 dtype-adaptive 分派：fp32 与 lowp 走不同路径

- **必须** 按 dtype 分派：fp32 路径全程 fp32 累加；fp16/bf16 路径 dot 用原生 dtype 操作数 + fp32 acc，中间不无故升精度。
- **禁止** 把 fp16/bf16 输入全程升 fp32 计算——参考实现是逐位舍入的低精度路径，"算得更准"反而不达标（bit-exact 契约见 §6）。
- **例外**：参考显式 `.float()` 的段（op15 的 attention）按契约走 fp32，见 L1.1 例外——分派边界是**段**，不是整个算子。
- **Why（op15 实测）**：dtype-adaptive 分派（v3/round-5）是阶段二基线 1.494× 的来源之一；fp16/bf16 dot 在 Ascend 上用 fp32 acc、乘积精确，**bit-exact 等效**（50/50 全过，§6）。

### L1.8 UB 预算：中间张量元素数有与输入无关的上界

- **绝对禁止** `BLOCK_* = next_power_of_2(<运行期变量>)` 且不加 `min()` 封顶（承接 `flash_attention.md` F4）。
- 本类归约 kernel 的 UB 占用：`acc[BP,BHD]fp32 + w[BP,BS] + k[BS,BHD]`，按 §L1.3 的预算（op15 上 128KB）取上限，UB 共 192KB。

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 主骨架

```
host:
  # 1) 投影权重复刻（CPU fp32 采样 + .to(dtype) + stride 交换，见 L1.2）
  # 2) tile 决策（全部 host 算好 constexpr 传入，见 L1.3/L1.5）
  BLOCK_P, BLOCK_HD, BLOCK_S = 面积预算（L1.3）
  NUM_PROG = NUM_CORES（常量）
  # 3) dtype-adaptive 分派标志（L1.7）

kernel 段 1：投影 GEMM（q/k/v，AST 禁 torch.matmul ⇒ 必须 tl.dot）
  _proj_matmul_kernel: grid=min(核数, M 分块数), 标准 tiling（BM/BN 自适应 32/64/128）

kernel 段 2：★ 低秩归约（Linear 类独有）
  _kv_reduce_strided_kernel: grid=min(核数, b*H), 每 program 一个 (b,h)
    kr[b,h,p,hd] = Σ_s k2d[b,s,h,hd] · w_kr[p,s]
    直接 strided-load k2d[B*S,H*HD]（s 维 stride=D），contraction BLOCK_S 开到 256（L1.3）

kernel 段 3：attention（在降维后的 [B,H,P,D] 上）
  _linformer_attn_kernel: grid=min(核数, b*H*cdiv(P,BP))
    Kr/Vr hoist 出 S-loop（L1.6）；softmax(Q@Krᵀ/√d)@Vr
    输出按 [b,S,H,hd] 直接写出（L1.4）
```

### §3.2 NUM_S-adaptive kernel 选择

按序列长 S 选不同 S-tile 路径（在已有 kernel 内分派，**不分裂新 kernel**——分裂增 launch 开销且本类瓶颈不在 attention，§5.2）。op15 round-9 靠此到阶段一峰值 1.48×。

### §3.3 两段独立优化

投影段（GEMM）与 attention+归约段是**两个独立优化对象**，profiling 里分别看 `Duration` 占比。⚠️ **不要**把多段合并进同一个 `@triton.jit`（用 `MODE: tl.constexpr` 分支）——那是迎合 benchmark 口径、不改变真实性能（`flash_attention.md` §7.2）。

---

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构必须重新设计）

### §4.1 ★★★ 低秩归约 contraction 维 tile 开到 UB 上限（本类头号杠杆）

```python
@triton.jit
def _kv_reduce_strided_kernel(
    k_ptr, w_ptr, out_ptr,
    s, d, H, hd, proj,
    stride_kb, stride_ks,          # k2d 的 b/s stride（strided-load，吃原始布局）
    stride_wp, stride_ws,
    BLOCK_P: tl.constexpr, BLOCK_HD: tl.constexpr, BLOCK_S: tl.constexpr,
    NUM_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    for task in range(pid, /* b*H */, NUM_PROG):    # L1.5 核内步长
        b, h = <由 task 解出>
        offs_p  = tl.arange(0, BLOCK_P)
        offs_hd = tl.arange(0, BLOCK_HD)
        acc = tl.zeros((BLOCK_P, BLOCK_HD), tl.float32)    # fp32 累加（L1.1）
        for s0 in range(0, s, BLOCK_S):                    # contraction 维，BLOCK_S=256（L1.3）
            offs_s = s0 + tl.arange(0, BLOCK_S)
            mask_s = offs_s < s
            k = tl.load(k_ptr + b*stride_kb + offs_s[:,None]*... + h*... ,
                         mask=mask_s[:,None], other=0.0)   # ★ strided-load（L1.4）
            w = tl.load(w_ptr + offs_p[:,None]*stride_wp + offs_s[None,:]*stride_ws,
                         mask=mask_s[None,:], other=0.0)
            acc += tl.dot(w, k.to(w.dtype))                # 原生 dtype 操作数 + fp32 acc
        tl.store(out_ptr + ..., acc.to(out_dtype), mask=(offs_p[:,None]<proj)&(offs_hd[None,:]<hd))
```

- **核心**：`BLOCK_S` 从直觉的 64 提到 256（面积预算允许的上限）。
- **实测**：op15 `_kv_reduce_strided_kernel` 6.48→5.02µs（−22.5%），端到端破 2.0×。
- **适用判据**：归约 contraction 维（序列长 S）大（≳数百）时收益最大；`S` 很小（<BLOCK_S）时按 `min(...,next_pow2(s))` 收缩。
- **待跨算子验证**：非 Linformer 的 Linear 子形态（Nyström/Performer）的归约结构若不同，BLOCK 上限需重扫。

### §4.2 ★★ copy 消除：strided-load + 目标布局直写（两半，缺一不可）

**(a) attention 输出按 `[b,S,H,hd]` 直写**（消 attn2 的 `.contiguous()`）：
```python
# kernel 内输出 stride 实参按目标 [b,S,H,hd] 布局传（最内轴 hd 连续）
tl.store(out_ptr + b*so_b + s*so_s + h*so_h + offs_hd*so_hd, acc, mask=...)
# forward 里做免费的 reshape，不再 .contiguous()
```

**(b) 归约 kernel 直接 strided-load 原始 `k2d[B*S,H*HD]`**（消 k/v 路径的 Transpose）：
- 不在 host 侧把 k 转成 `[b,H,S,hd]` 连续；kernel 内用 `stride=D` 跨 s 维 load（见 §4.1）。
- ⚠️ **输入和输出必须分别传 stride**（承接 `flash_attention.md` L1.15）：真实输入常是非连续转置视图，workspace 是连续的，共用一套 stride 会散射到错误地址。**自测必须用非连续输入**，否则测不出。
- **实测**：(a)+(b) 合计 **+30%**（v4 +16%、v6 +14.5%）。

### §4.3 ★ Kr/Vr hoist + (b,H)-major 连续调度（attention 段）

把 attention kernel 重构为：每个 program 持连续 tile 区间、按 (b,H)-block 推进、Kr/Vr 每 block 载入一次 hoist 出 S-loop、内层 S-tile 复用。
- **Why**：Kr/Vr HBM 流量 `O(B·H·NUM_S)→O(B·H)`；满核占用（合规 L1.5）。
- **实测**：op15 round-8 +5.14%，靶向修大 seq 慢 case。
- **适用判据**：attention 段在大 seq case 上 sub-1.0×、profiling 显示 Kr/Vr load 占比高时做。

### §4.4 NUM_S-adaptive 与 dtype-adaptive 分派

- **NUM_S-adaptive**：按 S 选 S-tile 路径（已有 kernel 内分派，不分裂新 kernel）。
- **dtype-adaptive**：fp32 / lowp 走不同累加路径（L1.7）。
- 两者都是 host 侧 constexpr 分派，零运行时分支开销。

---

## §5 Phase 4 优化点清单

映射到 `triton-latency-optimizer` 的优化点。Linear 类的有效优化集中在 **reduction tile + copy 消除 + 调度**，攻 cube-bound 的方向全部证伪。

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益（op15） | 适用条件 |
|---|------|---------|----------|
| 1 | ★★★ 低秩归约 contraction BLOCK_S 开到 256 | 单 kernel **−22.5%**、端到端**破 2.0×** | 归约 contraction 维（S）大；UB 内放得下（128KB） |
| 2 | ★★ 消除搬运 copy（strided-load + 目标布局直写） | **+30%**（v4 +16%、v6 +14.5%） | profiling 出现 `Transpose`/`ViewCopy`/`contiguous` |
| 3 | ★ Kr/Vr hoist + (b,H)-major 连续调度 | **+5.14%** | attention 大 seq case sub-1.0× |
| 4 | NUM_S-adaptive kernel 选择 | 阶段一峰值 1.48× | S 跨度大 |
| 5 | dtype-adaptive 分派 | 基线 1.494× | 多 dtype 混合 |
| 6 | GEMM tiling（投影按 M/N 自适应 BM/BN） | 1.05×→1.41× | 阶段一基础 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结果（op15） | 根因 |
|------|---------|------|
| **q/k/v 三输出融合**（BM=BN=64，v8） | **−30% 回归**（1.9921→1.388×） | 多 fp32 累加器撑爆 UB，逼 BK 128→64 ⇒ **cube 吞吐 −25~30%** |
| **k/v 两输出融合**（BK=64，v12） | **−24% 回归**（→1.5039×） | 同 v8 失败模式 |
| **attention 改 fp16 操作数 dot**（fp32 acc，v13） | **−5% 回归**（→1.8946×） | 小 tile（BP≤64,HD≤128）下 cube 是 **latency-bound**，fp16 的 2× MAC 速率兑现不出；额外多 `vr.to(fp32)` upcast |
| **projection GEMM 加 num_stages 流水线**（v14） | **中性/负**（→1.9471×，投影 17.59→17.54µs 噪声内） | **直接证明 projection 是 cube-bound（计算受限），内存流水线无用** |
| **GROUP_M / BK 微调扫参**（v9/v10/v11） | **全中性**（≤1.99×，噪声内） | 几何平均下全局改参会同时动到已快的 case，常净负 |
| **按 projection_dim 分裂 attention kernel**（#18，round-5/8 评估） | **未采纳** | profiling 显示瓶颈在投影 GEMM **不在 attention**，分裂收益有限且增 launch 开销 |
| **fp32 dispatch 开关**（v5）、hybrid 调整（v7） | 中性，未采纳 | — |
| **手写 `time.time` 性能探针** | **误导方向** | host 开销拉平到 ~0.23ms，假判 launch-bound（LR2） |

> **统一教训**：所有攻 **cube-bound 的 86%**（投影 + attention dot）的方向都是负反馈或中性。这**反向印证**了瓶颈定位的正确性——Linear 类的有效优化只能落在 reduction（14%）和 copy 上。

### §5.3 结构性下限：小 case 上不要拆 kernel

任何含 softmax/elementwise 的 kernel 在 triton-ascend 下必然编成 `MIX_AIC`，每次发射 ~4.5µs 固定成本（承接 `flash_attention.md` §5.3）。小 case 整个 impl 才 6~8µs，每多拆一个 kernel 多 ~4.5µs。判别：看 `kernel_details.csv` 的 `Accelerator Core` 列（`MIX_AIC` vs `AI_CORE`）。

---

## §6 精度闸门（先过闸门，再谈性能）

- **bit-exact 契约**：fp16/bf16 输入下，参考实现逐位舍入（`torch.matmul` 对 fp16 返回 fp16、`F.softmax` 返回 fp16、下个 matmul 又在 fp16 上）。实现用 `tl.dot`（原生 dtype 操作数 + **fp32 acc**），Ascend fp16 dot 乘积精确、**bit-exact 等效**，op15 50/50 全过。
- **低秩归约的 bit-exact**：只改 contraction tile 尺寸不改计算顺序 ⇒ summation regrouping ~1e-7，远在 MERE 门（fp16 9.77e-4 / bf16 7.8e-3 / fp32 1.22e-4）下。
- **判据推论**：失败集**严格与 dtype 相关**（如"全部且仅仅是 fp16/bf16 case"）⇒ 算术路径不匹配，不用去搜 tiling。
- **正式结论走 `verify.py`，不改阈值、不用 `--verify_not_required`。**

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.0 ★ 几何平均 ⇒ 修最慢 case（承接 `flash_attention.md` §7.0）

`benchmark.py` 的 `speedup_vs_torch` 是**逐 case 几何平均**。把一个 0.5× 修到 1.0×，收益等于把一个 2× 提到 4×。
⇒ 达标路径是"把垫底那批（op15 是 fp32 / 大 seq）修到 ≥1.0×"，不是"整体再快 N%"。操作：从 `per_shape_results` 升序找 `<1.0×` 的共同特征，为它们单开分派路径（L1.7 + §4.3），不要全局改参（§5.2）。

### §7.1 ★ cube-bound 天花板估算

Linear 类 86% 时间在 cube-bound 段，**理论加速上限**由"triton GEMM vs CANN aclnnMatmul 的 cube gap"决定。攻 cube-bound 无解（§5.2），所以**实际可达加速比 ≈ 把可动的 14%（reduction+copy）优化掉的杠杆 ÷ 14% 占比 + cube 段持平**。op15 上这 14% 优化 −22.5% ⇒ 端到端约 +1~2%，正好够把 1.99× 顶过 2.0×。

### §7.2 噪声与包夹协议

- 同一份代码复测可差 ~2%；**≤3% 的改动必须用包夹协议（base→A→base）复测**，否则把噪声当收益。
- 比值型指标（speedup）不能自证有效：分子分母同步变慢时 speedup 反而"看起来变好"。

### §7.3 ⚠️ benchmark / 编译器版本会改变数字——成功与失败都会

op15 内核未变，benchmark 脚本更新后基线度量口径变了（torch_npu 基线测得更慢），加速比从 **2.0315× → 2.37×**。
> **规则：换 benchmark / 编译器 / CANN 版本后，数字作废，必须重标定基线。内核级成就（破 2.0×、bit-exact）独立于具体数字。** 报告里两者分开写。

### §7.4 频率稳定（无 root 时的替代方案）

若 NPU 频率锁不上（`dsmi_set_device_info` 需 root），用 **warmup≥50** 充分升频 + 看 benchmark 的频率漂移计数（应 ≤个位数）替代锁频。op15 上 warmup=50 漂移仅 2 次、满频稳定。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Cube 空转、benchmark 超时无数据（0.0x） | 投影/归约用了逐维外积而非 `tl.dot` | L1.1 |
| **精度 50/50 全过、speedup 却只有 0.0x**（0812 批跑 0.0517×：fp32 路径 `tl.sum(x[:,None,:]*w[None,:,:])` 广播乘 + GEMM `BLOCK_M=1` 硬编码 + 单 query attention + 5 处 `.contiguous()` 四病并发；优化轮只把 BLOCK_M 1→8 就停手） | 每条独立致命且精度全不报错；`aic_mac_ratio` 极低就是指纹 | L1.1/L1.3/L1.4 的坏味道做**生成时静态预检**：`tl.sum(a[:,None,:]*b[None,:,:])`、`BLOCK_M=1`、`for q_idx in range(pid, TOTAL_Q, num_cores)`（单 token 任务）、`.contiguous()` 四个代码指纹，命中即打回 |
| 权重在 NPU 重采样 / 投影算成 `x@W` | CPU≠NPU 两条 RNG 流；`nn.Linear.weight` 是 `[out,in]` | L1.2 |
| **精度全过但卡在 1.99× 破不了 2.0×** | 低秩归约 contraction BLOCK_S 硬编码 64，K 迭代多、cube reduction 浅 | **L1.3**：开到 256（面积预算），单 kernel −22.5% |
| 搬运 copy 占可观时间、speedup 上不去 | host 侧 `.contiguous()`/transpose；归约前先把 k 连续化 | **L1.4**：strided-load + 目标布局直写（+30%） |
| 大 seq case sub-1.0× 拖垮几何平均 | Kr/Vr 每 S-tile 重载，HBM 流量 O(B·H·NUM_S) | **L1.6/§4.3**：Kr/Vr hoist + (b,H)-major |
| **多输出 GEMM 融合后大幅回归（−25~30%）** | 多 fp32 累加器逼 BK 128→64，cube 吞吐崩 | **§5.2 v8/v12**：本芯片 + fp32-acc 约束下多输出融合是死路 |
| fp16 dot 改完精度过但更慢 | 小 tile attention latency-bound，fp16 无效 | §5.2 v13 |
| projection 加流水线零收益 | projection 是 cube-bound，内存流水线无用 | §5.2 v14（反向证实 cube-bound） |
| 全局改 GROUP_M/BK，诊断子集 +10%、正式 −3.7% | 几何平均下动了已快的 case | LR3 + §7.0 |
| **方向被带偏、停在 1.48× 误判"投影无解"** | 用手写 `time.time` 探针，假判 launch-bound | **LR2**：只用 kernel `Duration(µs)` |
| 换 benchmark 版本后加速比跳变 | 基线度量口径变了，内核没变 | §7.3：重标定基线，内核成就与数字分开 |
| 频率漂移导致复测不稳 | 无 root 锁不了频 | §7.4：warmup≥50 + 漂移计数 |
| 归约 kernel 输入用连续张量自测通过、真实输入挂 | 真实输入是非连续转置视图，输入/输出共用一套 stride | §4.2(b)：两套 stride 各传各的，自测用非连续输入 |

---

## §9 与 `flash_attention.md` 的分工

| 判别 | 用哪个卡 | 理由 |
|------|---------|------|
| K/V 经可学习 `[P,S]` 低秩矩阵投影、`S→P` 降维后再 attention、不对全 S 滚动 online softmax | **本文件（Linear.md）** | 瓶颈分布（cube-bound 86% + reduction 14%）、头号杠杆（contraction tile）、证伪方向（多输出融合）都不同于 FA |
| 对全 S 做 KV 分块 + online softmax，或带 causal/window/softcap | `flash_attention.md` | 本类 Kr/Vr 是降维结果、尺寸小、适合 hoist；FA 的 Kr/Vr 必须进循环 |
| 极小 S 的朴素一次性 attention、无 KV 分块、无低秩投影 | `flash_attention.md`（原 `attention.md` 已并入该卡 §0.1 末段） | — |

> **共用条目（本卡直接承接，不重复；`attention.md` 已退役删除，原条目并入 `flash_attention.md`）**：
> `flash_attention.md` F1（归约 padding 显式排除）/ F4（UB 预算）、L1.9（grid min 核数）/ L1.15（输入变换外提 + 两套 stride）、
> L1.11/L1.12（host 侧权重复刻 + 投影布局）、§5.3（小 case 别拆 kernel）、§7.0（几何平均修最慢 case）。

---

## 附录：op15 收益归因（一句话）

> op15 的优化本质——**在 86% cube-bound 硬顶（投影 38% + attention 48%）之外，找到 14% 可动的 `kv_reduce`，用 contraction 维 tile 加深（64→256）这一刀（−22.5%），配合两轮 copy 消除（+30%），把唯一能动处榨干，越过 2.0×。** 期间所有攻 cube-bound 的尝试（多输出融合 / fp16 dot / 流水线）都给负反馈，反向印证了瓶颈定位的正确性。详见 `算子生成经验/15_LinformerAttention_优化全过程复盘.md`。

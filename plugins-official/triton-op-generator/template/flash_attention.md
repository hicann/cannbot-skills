---
name: flash_attention
description: FlashAttention / Attention 主链类算子（MHA / SDPA / flash-attention / GQA-MQA）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧与 Phase 4 优化点清单
metadata:
  type: reference
---

# FlashAttention 类算子优化经验

本文档是 **FA 主链（`Q@Kᵀ → softmax → @V`）+ KV 分块 + online softmax** 这一类算子的经验合集，覆盖 Phase 2/3/4：

- **§1 通用经验**：跨形态共有的工程约束
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表**

> ⚠️ **本文件覆盖「一 标准类」的全部三个细分**（基础三段式 / 分块流式 / 空间 token 版）：
> 凡满足 §0.1 判别特征的算子（KV 分块 + online softmax 状态量，或带 causal / sliding-window / softcap 的 attention 主链），
> **一律使用本文件**。原先另有一份 `attention.md` 承接朴素三段式，已退役——它的 L1.1（禁用 fp32 `tl.dot`）/ L1.3（`next_pow2`）/ L1.6（全程 fp32）
> 三条在 FA 路径上**已被实测推翻**（差异与证据见 §9）。
>
> **证据基础**：三个 FA 类算子各 50 case 的完整优化轨迹——`FlashAttentionV2`（含 4 个投影，3.68 / 换精度契约后 3.30）、
> `FlashAttentionFwd`（无投影、可变 mask，2.17 → 6.24）、`MultiQueryAttention`（含投影 + KV 共享，0.1052 → 2.3724）。
> 三者形态、精度契约、编译器版本各不相同却**独立收敛到同一瓶颈结论**，这是本卡可外推的根据；
> 三者之间的 **3 条冲突结论**及其判别前置条件见 §2 末尾的「跨算子结论冲突的统一判据」。
>
> ⚠️ **核心优化哲学**：这类算子的瓶颈是 **Cube↔Vector 跨核同步**，不是算术量、不是访存、不是 dtype。
> 一切优化等价于**减少「循环体内互有 UB 依赖的向量算子数 × KV 迭代总数」**。
> 生成时**禁止混用**其他类别的经验——尤其不要套用 `transformer-inference.md` 的多 kernel 拆分策略（§5.3 已用访存账否决）。

---

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| MultiHeadAttention | `fa-mha` | 4 个 `nn.Linear(D,D)` 投影 + `softmax(Q@Kᵀ/√d)@V`，`S_Q ≠ S_K`，`head_dim` 普遍非 2 的幂 | host 侧忠实复刻 `nn.Linear` 语义 + kernel 侧 online softmax；**投影侧与 attention 侧分别优化** |
| self/cross-attention、SDPA | `fa-sdpa` | 同上但 q/k/v 由外部传入 | 同上，去掉投影段 |
| flash-attention 前向 | `fa-fwd` | KV 分块，不物化 `[S_Q,S_K]`；常带 causal / window / softcap **运行时属性** | 重点在 **KV 扫描区间收缩 + mask constexpr 特化** |
| GQA / MQA | `fa-gqa` | `H_Q > H_KV`，KV head 共享；**通常自带 4 个投影 GEMM** | 同 `fa-fwd`；⚠️ 若含投影，**最大的一笔在投影侧**（L1.12，实测端到端 5.94x），不要一上来就攻 attention |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. kernel 内对 KV 维度分块循环，且跨迭代滚动 online softmax 状态量 `m` / `l` / `acc`；
2. attention 主链带 `causal` / `window_left` / `window_right` / `softcap` 等 mask 属性；
3. 计算图含 `scaled_dot_product_attention` 或 `softmax(Q@Kᵀ/√d)@V`，且 `S` 大到不宜物化 `[S_Q,S_K]`；
4. ⭐ `forward` 里出现 `torch_npu.npu_xxx(...)`（如 `npu_fusion_attention`）→ **golden = CANN op 的真实行为**，文件内同名的纯 torch fallback 只是 docstring 参考语义、**不是 golden**。满足本条 ⇒ 除本文件外，还**强制**走 §6.1 探针流程再谈性能（否则 §3.2 区间收缩、§3.3 趟数结构等方向会直接算错）。

不满足（如极小 `S` 的朴素一次性 attention、无 KV 分块）→ 仍用本文件，跳过分块流式相关的 Layer 2/3，只取 Layer 1 与通用经验。

### §0.2 ★ 形态识别五问（**Phase 2 第一步必须回答，答案决定后续哪些章节适用**）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | **golden 是 CANN op 还是 torch 数学实现？** | CANN（`forward` 里有 `torch_npu.npu_xxx`）→ **先走 §6.1 探针**，golden 真实行为可能与文档/数学语义相反；torch 数学实现 → 直接看 Q1-Q4 |
| **Q1** | 有没有独立的投影 GEMM？ | 有 → §4.4 / §4.5（权重预重排、分块分档）是最大的一类收益；**没有则这两条完全不适用** |
| **Q2** | mask 是固定的还是可变属性？ | 可变 → §3.2（区间收缩）+ §2.7（constexpr 特化）合计实测可达 **+58%**；固定 causal 拿不到这么多 |
| **Q3** | 参考实现的 attention 在什么精度上算（有没有 `.float()`）？ | 决定 `p` 能否降精度，以及**趟数结构**（见 §6.3） |
| **Q4** | 评测口径？ | 几何平均 ⇒ **每个 case 权重相同**，把一个 case 从 0.5x 拉到 1.0x，与把另一个从 8x 拉到 16x 贡献完全相同。**不能只优化大 case** |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 归约维 padding 必须显式排除，`other=0.0` 不是掩码

- **必须** `S_K` 非 `BLOCK_KV` 整数倍时，`max` 前 `tl.where(kv_valid, scores, NEG)`，`sum` 前 `tl.where(kv_valid, p, 0.0)`。
- **禁止** 只靠 `tl.load(..., other=0.0)` 就认为越界列已被排除。
- **Why**：`other=0.0` 只让越界列 score 变 0，而 `exp(0-m) > 0` —— 这些"幽灵列"会混进 softmax 分母，把 `acc/l` 整体缩小。实测某 MHA 的 50 个 case **全部** `S_K % 32 != 0`，单这一条的 MERE = 8.16e-01（超 fp32 阈值 6700 倍）。
- **验证集要求**：主动构造 `S_K % BLOCK_KV == 0` 与 `!= 0` 两类用例，否则该缺陷会逃逸成"48/50 通过"这种更难诊断的形态。

### F2 掩码用**有限极小值**，不用 `-inf`

- **必须** 用 `-3.0e38`（fp32）/ `-1e30`，且声明为 `tl.constexpr`（否则 `NameError: Cannot access global variable`）。
- **Why**：`-inf` 会在整行被掩时产生 `exp(-inf-(-inf)) = NaN`，并逼着 `tl.maximum` 保留 NaN 处理，与 §4.2 的性能项**直接冲突**。
- **安全性论证（写进注释）**：causal + online softmax 下 KV 循环从 0 开始，第一块必含合法列 ⇒ 每行首次更新时 `m` 即为有限值，后续整块被掩的行 `p = exp(-3e38-m) = 0`，不污染。

### F3 `BLOCK_D` 用 `ceil16(D)`，**不用** `next_pow2(D)`

- **必须** `BLOCK_D = ceil16(head_dim)`，并对 `d_offs < D` 做 mask。
- **Why**：triton-ascend 支持非 2 的幂 `tl.arange`。D=96→96 vs 128、D=160→160 vs 256、D=192→192 vs 256，`next_pow2` 白白浪费 **30~60%** 的 tile 面积，直接转化成迭代数与同步开销。
- **回退**：仅当目标版本 `tl.arange` 不支持非 2 的幂时才退回 `next_pow2`。
- **禁止** 硬编码 `BLOCK_D`（如 16），也**禁止**把 `BLOCK_D` 放进 `@triton.autotune` 搜索空间——`head_dim > BLOCK_D` 时只算前 `BLOCK_D` 维，**静默错**。

### F4 UB 预算：中间张量元素数必须有与输入无关的上界

- **绝对禁止** `BLOCK_* = triton.next_power_of_2(<运行期变量>)` 且不加 `min()` 封顶。实测 `BLOCK_N = next_pow2(S)` 在 `S=577` 时需 792 KB，直接溢出。
- **面积上限有明确来源**：`scores` tile 是 `[BQ, BKV]` fp32 ⇒ `BQ*BKV ≤ 8192`（32KB）是**算出来的**，不是拍脑袋；再叠加 `acc [BQ, BLOCK_D]` fp32。UB 共 192KB（`ub overflow, requires ... while 1572864 bits available`）。
- ⚠️ **UB overflow 会污染设备，导致级联假失败**：一次 `ub overflow` 可触发 vector core exception，使**后续用例连带失败**（实测一次全量从 49/50 掉到 36/50，真正硬失败只有 3 个，其余 11 个是陪葬）。
  → **诊断原则**：看到大批连续失败时，先定位**第一个**失败用例单独复跑；不要按失败总数估计缺陷数量。

### F5 结构性改动之后必须重扫 tile，**不要查历史清单**

实测：一个算子在 KV 区间收缩后，最优 tile 从 `128x64` 变成 `64x128`（与旧结论**相反**）；另一个在权重重排后重扫，结论未变——但那是**验证过**的，不是假设。

### F6 扫参子集必须含大/中/小三档

⚠️ **几何平均口径下最容易犯的错**：拿 5 个大 case 扫参得到的最优配置直接全局套用。
实测一次：诊断子集 +10%，正式测量 **−3.7%**——退化全发生在诊断子集没统计的中等 case 上。
**画像用于定位，判定只能用完整的 verify + benchmark。**

---

## §2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

> **本节共 16 条（L1.1 ~ L1.16），全部是 Phase 2 的硬性边界。**
> Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部 16 条**，不得只取前 N 条或按篇幅截断——
> 实测出现过只载入到 L1.12、把 L1.13（BLOCK 面积预算）漏掉，导致 `BLOCK` 保持硬编码、
> 性能与不带该约束时完全相同的情况。
> 带 ★★★ 的三条（**L1.12 投影权重布局**、**L1.13 BLOCK 面积预算**）是收益最大的两笔，尤其不能漏。


### L1.1 ★ `tl.dot` 契约：原生 dtype 操作数 + fp32 累加器；**fp32 输入同样必须用 `tl.dot`**

- **必须** `tl.dot(q, k_t)`，两操作数**保持原生 dtype**，累加器 fp32（`tl.dot` 默认 `out_dtype=tl.float32`）。
- **禁止** 把操作数升到 fp32 再 dot（`tl.dot(q.to(fp32), k.to(fp32))`）——那会走异构路径，是"fp32 禁用 `tl.dot`"这条流传约束的**真正来源**。
- **禁止** 用逐维标量循环（`for d in range(D): scores += q[:,None]*k[None,:]`）或 `tl.where(d_offs==d, ...)` 逐维散写 `acc` 代替矩阵乘。
- **Why**：逐维外积让 Cube 完全空转（`aic_mac_ratio` 7.5%），实测 benchmark 直接超时无有效数据；换 `tl.dot` 后**不可测量 → 2.3554**，是全链路最大的一笔。fp32 输入下精度实测 **50/50 全过**。
- **判别信号**：`aic_mac_ratio < 10%` 且代码里有 `for d in range(head_dim)` ⇒ 立即重写，不用扫参。

### L1.2 `tl.dot` 两操作数必须显式 cast 到同一 dtype

- **必须** 两侧都 `.to(<统一 dtype>)`——`tl.load(..., other=0.0)` 会把结果推成 fp32。
- **Why**：否则 `semantic.dot` 断言 `Both operands must be same dtype. Got fp32 and fp16`。实测该错误导致 33/50 个非 fp32 case 全部编译失败。

### L1.3 online softmax 必须同步缩放 `acc` 与 `l`

- **必须** 每轮 KV tile：`m_new = max(m, rowmax(scores))`；`alpha = exp(m - m_new)`；`l = l*alpha + sum(p)`；`acc = acc*alpha + tl.dot(p, v)`；`m = m_new`。
- **禁止** 漏掉 `acc *= alpha`（结果偏大）或漏掉 `m = m_new`（下一轮基准错）。
- `m` 用**有限极小值**初始化（`-1e30`），不用 `-inf`（见 F2）。

### L1.4 ★ 精度契约先钉死，再谈性能

- **必须** 在写 kernel 前先确认**参考实现的算术路径**：attention 是写作 `q.float() @ k.float().T`（fp32 归约），还是在**原生 dtype** 上算。
- **两套契约不可混用**：
  - 参考在 fp32 上算 ⇒ `scores`/`p`/`acc`/`l` 全程 fp32，最后 cast 回输出 dtype；
  - 参考在原生 dtype 上算 ⇒ **必须逐位复刻低精度路径**，"算得更准"是错的。四条逐位契约见 `@../../../ops/triton-precision-debug/references/attention-lowprec-contract.md`。
- **Why**：某算子上游一次提交把参考从 fp32 改回原生 dtype，旧契约下 3.6766 的交付版对新参考**只过 20/50**。**参考的算术路径一变，精度类结论全部作废、性能类结论也要重测。**

### L1.5 `p`（softmax 输出）不能直接降到输入 dtype

- **禁止** `tl.dot(p.to(in_dtype), v)`：`p` 是真 fp32 量，降到 fp16 相对误差 ~5e-4，顶穿 fp16 阈值 9.77e-4。实测 27/50 越阈，**失败集合严格等于全部 fp16/bf16 case**。
- **可选替代**（仅低精度输入）：二段拆分 ~22 bit，见 §4.6。**fp32 输入不要开。**

### L1.6 host 端不做 permute / contiguous / pad

- **必须** 直接把原始布局的 stride 传进 kernel（最内轴 D 本来就连续），输出按原布局分配；D 的补齐用 `BLOCK_D=ceil16(D)` 的 masked load（`other=0.0`）。
- **禁止** `q.permute(...).contiguous()`、`torch.zeros`+`copy_` 补齐、输出再 permute 回来。
- **Why**：实测每次调用多出 ~4 个纯搬移 kernel（`Transpose` 占 14.39%），删掉后 **+16%（保守值）**，小 shape 从 2~3x 直接跳到 10~14x。
- **判别**：profiling 时间归属里只要出现 `Transpose`/`ViewCopy`/`ZerosLike`/`Slice`，直接删，不用扫参。

### L1.7 mask 属性作为 `tl.constexpr` 传入，禁止运行时标量分支

- **必须** `HAS_UPPER` / `HAS_LOWER` / `HAS_SOFTCAP` / `IS_CAUSAL` 以 `tl.constexpr` 传入，让编译器把不适用的整段 DCE 掉。
- **禁止** 把 `causal` / `window_left` / `window_right` / `softcap` 作为运行时标量在 kernel 内构造多个 `[BQ,BKV]` 比较再 `or`。
- **Why**：实测 **+21%**。代价是每种组合一个特化版本（~8 种），编译只发生一次且不计入 benchmark。

### L1.8 `kv_lo` **不做** `BLOCK_KV` 对齐

- **禁止** `kv_lo = ... // BLOCK_KV * BLOCK_KV`。
- **Why**：扫描区间常只有 1~3 个 block，对齐平均多出 `BKV/2` 个 key ⇒ **凭空多一整次迭代**，每次迭代要付 8 个跨核握手。实测与 tile 放大**成对**做时 +21%，单独放大 tile 只有 +1.2%。

### L1.9 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

```python
# host
NUM_CORES = 24                                            # 必须是常量 constexpr
total_tasks = B * H * cdiv(S_Q, BLOCK_Q)
grid = (min(NUM_CORES, total_tasks),)                     # ✅ 两头都对
# kernel
for task in range(tl.program_id(0), total_tasks, NUM_PROG): ...
```

三种写法的实测差异：`grid=(total_tasks,)` 在 `total > 核数` 时**任务被重复计算**（实测某 case 的 task 被 3 个 pid 各算一遍）；
`grid=(核数,)` 在 `total < 核数` 时小 shape 白付启动开销；只有 `min()` 两头都对，实测 **+5.8%**。

- ⚠️ `NUM_CORES` 必须保持**常量 `constexpr`**——做成随 case 变化的值会触发编译爆炸（实测 verify 超时、0 个用例启动）。`grid` 是启动参数，随 case 变化无妨。
- ⚠️ 任何改变任务空间的改动（如压平循环维）**必须同步改 grid**，否则表现为**静默算少 → AccuracyError 而非报错**。

#### ★ 本条适用于算子内的**每一个** `@triton.jit` kernel，不只 attention 主 kernel

**禁止**把辅助 kernel（RoPE / 位置编码 / bias 预处理 / layout 变换 / 归一化等）的 grid
按元素数或行数**直接展开**——`grid = (B*N*S,)` 这类写法在大 shape 上会撞硬上限：

```
RuntimeError: grid should be less than 65536!
You can try "export TRITON_ALL_BLOCKS_PARALLEL=1" to avoid this problem.
```

**必须**所有 kernel 统一用同一套写法：

```python
total_tasks = <该 kernel 的任务数>                  # 如 B*N*S 行、或 B*N*cdiv(S,BLOCK)
grid = (min(NUM_CORES, total_tasks),)              # ✅ 永远不会越界
# kernel 内
for task in range(tl.program_id(0), total_tasks, NUM_PROG): ...
```

- **判别信号**：失败集合**严格按 `B*N*S`（或该 kernel 的 grid 维）是否 ≥ 65536 划分**，
  且报的是 `RuntimeError` 而**不是**精度不匹配 ⇒ 直接查 grid，不要去查 dtype 或数值路径。
- **实测**：某带 RoPE 的 FA 算子 50 个 case 里失败 8 个，全部满足 `B*N*S ≥ 65536`
  （65536 / 131072 / 262144），通过的全部 < 65536；attention 主 kernel 遵守了 L1.9、
  而另开的 `rope_kernel` 按 `B*N*S` 展开 grid，是唯一的失败源。
- ⚠️ **不要用 `export TRITON_ALL_BLOCKS_PARALLEL=1` 绕过**：那只是放开上限，
  grid 远超核数仍会带来调度开销，与 L1.9 的初衷（grid 收缩到核数）相反。

- **Why**：原始 grid 大 shape 下可达数百，每个 program 都要付一遍循环外的同步收尾。收益小（+0.5%），但属基础写法。
- **禁止** grid 超订（如 2 倍核数）：MIX kernel 超订不会带来同步延迟的重叠，实测无收益。

### L1.10 张量的运行时索引与元素赋值均不支持

- **禁止** `q_tile[:, d]`（`d` 为循环变量）→ `ValueError: unsupported tensor index: int32[]`；**禁止** `acc[:, d] = x`。
- **禁止** 用运行期变量做 `tl.arange` 的参数 → `ValueError: arange's arguments must be of type tl.constexpr`。
  由运行期 shape 推出的值（如 `BLOCK_D // 2`、`ceil16(D) // 2`）**必须先声明为 `tl.constexpr` 参数从 host 传入**，
  不能在 kernel 内即时计算后拿去做 `tl.arange`。

#### ★ 交错重排（rotate-half / interleaved RoPE）的可执行配方

参考实现里 `torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).flatten(-2)` 这类**偶/奇元素交错**，
在 Triton 里**不要**用「切出偶/奇两半 → reshape 回去」的写法——那条路上必然要用运行期值构造
`tl.arange` 或做动态 `tl.reshape`，直接编译失败。

**正解：用 constexpr 的静态索引 + `tl.where` 就地选择，全程不改形状**：

```python
# host 侧：BLOCK_D 必须是 tl.constexpr
offs_d  = tl.arange(0, BLOCK_D)                 # ✅ 参数是 constexpr
is_odd  = (offs_d % 2) == 1                     # 静态奇偶掩码
partner = tl.where(is_odd, offs_d - 1, offs_d + 1)   # 配对元素的下标
x       = tl.load(x_ptr + row * stride + offs_d, mask=offs_d < D, other=0.0)
x_pair  = tl.load(x_ptr + row * stride + partner, mask=partner < D, other=0.0)
x_rot   = tl.where(is_odd, x_pair, -x_pair)     # 奇位取前一个、偶位取后一个并取负
out     = x * cos + x_rot * sin                 # 形状自始至终是 [.., BLOCK_D]
```

- **判别信号**：`CompilationError` + `arange's arguments must be of type tl.constexpr`，
  且**失败集合严格按某个布尔属性划分**（如"全部且仅仅是 `use_rope=True` 的 case"）
  ⇒ 直接查该分支里的 `tl.arange` 参数与动态 `tl.reshape`，**不要去查 shape/dtype/数值路径**。
- **实测**：某带 RoPE 的 FA 算子 50 个 case 挂 25 个，全部是 `use_rope=True` 的奇数号 case，
  连续 4 轮迭代都没自己定位到，原因就是这条。
- 同族约束：位置编码的 `cos`/`sin` 表若在 kernel 内生成，其索引同样只能用 constexpr `tl.arange`；
  表本身可在 host 侧算好传入（与 L1.6「host 端不做 permute/pad」不冲突——那禁的是**数据布局搬运**，
  不是**常量表预计算**）。
- 需要按维度取向量时直接从 GM 按 `d * stride` load。（若已按 L1.1 用 `tl.dot`，通常不会遇到这条。）

### L1.11 `fa-mha` 专属：先对齐 host 侧语义，再写 kernel

含投影 GEMM 的算子，最致命的两条缺陷**不在 kernel 里**，而在 host 侧那二十行胶水代码，且二者各自独立致命、互相掩盖——修好任意一条 `passed_cases` 仍为 0，迭代循环拿不到梯度信号。

**必须按此顺序**：① 权重与参考逐位一致 → ② `y = x @ Wᵀ` 语义正确 → ③ 再写 attention kernel。

| 位置 | 要求 |
|------|------|
| 权重初始化 | **CPU** 上、**先 fp32 采样再 `.to(dtype)`**、构造顺序与次数一致、rng save/restore 一致。四处缺一即全错 |
| 投影 GEMM | `nn.Linear.weight` 是 `[out, in]`，`y = x @ Wᵀ`。方阵下 shape 检查会放行错误写法 ⇒ 交换 stride：`w.stride(1), w.stride(0)` |
| dtype 时序 | 不要直接以 fp16/bf16 采样权重 |

**★ 可直接套用的权重复刻配方**（四个环节缺一即全错，实测这一条能省掉整轮逐层 debug）：

```python
# 参考实现通常在 forward 里现场建层，必须逐字复刻其 rng 时序：
rng = torch.get_rng_state()          # ① 存 rng
torch.manual_seed(42)                # ② 与参考同一 seed
ws = []
for _ in range(4):                   # ③ 顺序、次数与参考完全一致（q,k,v,o 四个）
    lin = nn.Linear(d_model, d_model, bias=False)   # CPU 上、fp32 采样
    ws.append(lin.weight.detach().clone())
torch.set_rng_state(rng)             # ④ 恢复 rng —— 漏了会影响后续随机量
weights = [w.to(device=dev, dtype=dtype) for w in ws]   # 采样完再 .to(dtype)
```

- `nn.Linear` 的默认初始化是 `kaiming_uniform_(a=√5)`，对 `[out, in]` 权重**等价于** `uniform_(-1/√in, +1/√in)`；
  若参考没有现场建层而是外部传入权重，则跳过本节。
- **诊断顺序**：`passed_cases == 0` 或大面积不匹配时，**先单独验权重**（把实现的 4 个权重与参考的逐个 `allclose`），
  再验投影输出，最后才查 attention kernel。跳过这个顺序会陷入"改 kernel 改不动"的长尾 debug。
- ⚠️ 权重错与投影转置错**互相掩盖**：只修一条，`passed_cases` 仍为 0，迭代循环拿不到梯度信号。

### L1.12 ★★★ `fa-mha` 专属：投影权重**禁止**以 `nn.Linear` 原始 `[out, in]` 布局喂给 GEMM kernel

**这是含投影 GEMM 形态的头号杠杆，首次生成就必须做对，不是 Phase 4 的可选优化。**

- **禁止** 把 `nn.Linear.weight`（`[out_features, in_features]`）原样传进 GEMM kernel 并用 `w.stride(0), w.stride(1)`。
- **必须** 在 `__init__` / 权重准备阶段做 `w.t().contiguous()`（或等价的预重排），kernel 侧 stride 实参相应改为 `w.stride(1), w.stride(0)`。**一次性 host 开销，数值逐位不变。**

```python
# ❌ b tile 的最内轴 stride = in_features ⇒ 离散 gather，无法向量化访存
#    kernel 注释形如 "y = x @ w^T，w 形状 [N, K]（nn.Linear.weight 布局）" 即是此坏味道
proj_kernel[grid](x, w_q, out, ..., w_q.stride(0), w_q.stride(1), ...)

# ✅ 预转置成 [in, out] 连续
self.w_q_t = w_q.t().contiguous()          # host 侧一次性
proj_kernel[grid](x, self.w_q_t, out, ..., self.w_q_t.stride(0), self.w_q_t.stride(1), ...)
```

- **Why（实测）**：同类算子上 `linear_kernel` 6469.7us → 35.6us（**182x**），端到端 0.1052 → 0.6253（**5.94x**），是该算子全程最大的单项收益，且**零精度代价**。
- **判别信号**：profiling 里 `aic_mte2_ratio > 60%`；或扫 tile 时出现"缩小 `BLOCK_N` 反而更快"——那是在绕开离散访存，不是真优化。
- **配套（同样在首次生成就做）**：**布局一改，分块必须重扫**。转置会把算子从"A 不转置 B 转置"变成"A、B 都不转置"，推荐分块随之改变（经验值 `M0=128, K0=256, N0=256` 一档；a tile 沿 K 的字节数要够 512B 对齐，`BLOCK_K=32` 的 fp16 只有 64B，是对齐线的 1/8）。同类算子上这一步再拿 **+13.2%**，MAC 从 17.9% 升到 40.3%（由访存受限转为算力受限）。
- **进阶（可选，+5.0%）**：在转置基础上再按 tile 形状重排成 `[NBN, K_pad, BLOCK_N]`，见 §4.4。**但两级里第一级才是数量级收益，不做第一级直接做第二级没有意义。**

### L1.13 ★★★ `BLOCK_Q` / `BLOCK_KV` 必须按面积预算取到上限，**禁止硬编码小值**

> ★ **面积预算是 16384 元素（64KB fp32），不是 8192。**
> 早期版本写 8192 是因为**输入变换还留在 KV 循环里占着 UB**；按 **L1.15** 把 RoPE 等
> 变换外提之后，UB 腾出来，`128×128` 的 score tile 可以正常编译并跑通。
> op54 实测：仅把这一处 `8192 → 16384`（`BLOCK_Q×BLOCK_KV` 由 64×128 变 128×128），
> 几何平均 **1.7264 → 2.1048**，精度仍 50/50，`<1.0x` 的用例从 16 个降到 6 个。
> ⇒ **L1.13 与 L1.15 是耦合的**：没做外提就放大 BLOCK 会 `MLIRCompilationError`（实测），
> 做了外提就必须把预算提到 16384，否则白白浪费一半 UB。

**这是本类算子最大的一笔收益，且必须在首次生成就做对——它决定 KV 迭代总数，而迭代数就是代价本身。**

- **禁止** 写死 `BLOCK_Q = 32` / `BLOCK_KV = 64` 这类小常量。实测 `32×64` 的 tile 面积只有预算上限的 **1/4**，
  直接意味着约 **4 倍**的 KV 迭代数与 4 倍的跨核握手。
- **必须** 按下式由 host 侧算出，取到面积预算允许的最大值：

```python
BLOCK_D  = ceil16(head_dim)
BLOCK_KV = min(128, ceil16(S_K))                      # 尽量大，但不超过实际 key 数
BLOCK_Q  = 满足 BLOCK_Q * BLOCK_KV <= 16384 的最大 2 的幂  # scores tile fp32 = 64KB（见下方★）
BLOCK_Q  = min(BLOCK_Q, max(32, ceil32(S_Q)))         # 不超过实际 query 数
while BLOCK_Q * BLOCK_D * elem_size > 64 * 1024:      # acc tile 约束
    BLOCK_Q //= 2
```

- **Why（实测）**：迭代总数 = `B*H*ceil(S/BLOCK_Q) × ceil(span/BLOCK_KV)`，两个 BLOCK 同时翻倍 ⇒ 迭代数变 **1/4**，
  整套 `pipe_barrier` 直接少一半以上。同类算子实测：

| SQ/SKV | 64/64 | 128/64 | 64/128 | **128/128** | 256/* |
|---|---|---|---|---|---|
| 子集几何平均 | 0.697 | 1.129 | 1.155 | **1.42** | 编译失败 |

- **面积上限 8192 是算出来的不是拍脑袋**：`scores` tile 是 `[BQ, BKV]` fp32 = 32KB；再叠加 `acc [BQ, BLOCK_D]` fp32。
- **放大到编译失败为止**：UB 上限由编译器直接报数字（`ub overflow, requires N bits while 1572864 bits available`），
  看错误信息就知道差多少。**上限不可用简单阈值预测**——同为 `128×128`，D=96 编得过、D=128 失败、D=160 反而编得过且快 22%。
- ⚠️ **小 shape 要按实际尺寸收缩**（`min(..., ceil16(S_K))` / `min(..., ceil32(S_Q))`）：固定大 tile 在 `S < BLOCK` 时纯空转，
  实测某算子 27/50 个 case 中招、最小 case 空转 85 倍。**"开到上限"与"按 shape 收缩"是同一条规则的两面。**

### L1.14 ★ `tl.maximum` 在多 KV 块路径上必须用 `propagate_nan=ALL`（单块路径必须关）

- **必须** host 侧算出 `MULTI_BLOCK = (max_kvblk > 1)` 作为 `tl.constexpr` 传入，按它分档：

```python
if MULTI_BLOCK:
    m_new = tl.maximum(m, rowmax, propagate_nan=tl.PropagateNan.ALL)
else:
    m_new = tl.maximum(m, rowmax)
```

- **Why**：默认 `propagate_nan=NONE` 在 IR 里被展开成 **7 条向量指令**（2×isnan + vmax + 2×vsel），`ALL` 直接对应硬件单条 `vmax`。
- ⚠️ **禁止全局开启**：实测全局开是 **−9%**，按块数分档才是 +2.7%；另一算子上单块 case 退化 25~45%。
- 语义等价性依赖 F2（掩码用有限极小值、无 NaN），必须在注释里写明论证。

### L1.15 ★★★ 输入变换（RoPE / 位置编码）**必须在 attention 主循环之外**完成

- **禁止** 把 Q/K 的 RoPE、位置编码、bias 变换放进 **KV 循环内**逐块重算。
- **必须** 用**独立的 Triton kernel** 预处理 q/k（写到 workspace），attention kernel 只读取变换后的张量。
  该预处理 kernel 的 grid 同样遵守 **L1.9**（`min(核数, tasks)` + 核内步长循环）。
- **禁止** 改用 host 侧 torch 做这个变换（`torch.einsum` / `stack` / `repeat_interleave` 等）——
  会被退化检测判为主链计算落在 PyTorch 上。**它必须是 Triton kernel，只是不能在 KV 循环里。**

**实测（同一算子、只改这一处）**：

| 配置 | `[8,8,2048,64]` | `[8,8,4096,64]` | `[1,8,2048,64]` | 每次 KV 迭代成本 |
|---|---|---|---|---|
| RoPE 在 KV 循环内（❌） | 0.005x | 0.003x | 0.013x | **630 us** |
| RoPE 外提 + BLOCK 64 | 0.747x | 1.022x | 1.566x | **4.4 us** |
| RoPE 外提 + BLOCK 128（✅） | **1.386x** | **1.985x** | **2.509x** | — |

**三级因果链，缺一环都解释不通**：

1. K 的 RoPE 每个 KV 块重算一次（含 `partner = offs_d±1` 的**元素级 gather**）
   ⇒ 每迭代成本 **630us**，是正常区间（4~10us，见 §5.1）的 **60~100 倍**；
2. 外提后每迭代回落到 4.4us ⇒ 大 shape 直接快 **114~144 倍**；
3. **外提还释放了 UB** ⇒ 原本"编译不过"的 `BLOCK=128` 变得可编译 ⇒ 迭代数减半 ⇒ **再快约 2 倍**。

⚠️ **第 3 点是最容易漏的联动**：kernel 内的额外中间量会把 BLOCK 逼小，
于是同时挨两刀（每迭代变贵 + 迭代数变多）。看到 agent 写下
`BLOCK_Q = min(BLOCK_Q, 64)  # avoid UB overflow with RoPE/ALiBi intermediates`
这类**预防性封顶**时，**不要直接删封顶**（删了确实编译失败），
而要**先把占 UB 的那部分计算移出 kernel**，再重扫 BLOCK。

- ⛔⛔⛔ **交错重排（rotate-half RoPE）禁止用「算出来的伙伴下标」做 gather load。**
  必须用 `tl.split` / `tl.join` 从**一次连续 load** 里拆偶/奇。实测 **70 倍**差距。

  ❌ 错误写法（数据相关寻址 ⇒ 退化成逐元素 gather）：
  ```python
  is_odd  = (offs_d % 2) == 1
  partner = tl.where(is_odd, offs_d - 1, offs_d + 1)     # ← 非规则地址
  x_pair  = tl.load(p + offs_s[:,None]*si_s + partner[None,:]*si_d, ...)
  x_rot   = tl.where(is_odd, x_pair, -x_pair)
  ```

  ✅ 正确写法（一次连续 load + `tl.split`）：
  ```python
  HD: tl.constexpr = BLOCK_D // 2
  x  = tl.load(p + offs_s[:,None]*si_s + offs_d[None,:]*si_d, mask=m, other=0.)  # 一次连续 load
  xe, xo = tl.split(tl.reshape(x, (BLOCK_S, HD, 2)))        # 偶/奇两半
  j   = tl.arange(0, HD)
  inv = 1.0 / tl.exp(((j*2).to(tl.float32) / D) * 9.210340371976184)
  ang = offs_s.to(tl.float32)[:, None] * inv[None, :]
  c, sn = tl.cos(ang), tl.sin(ang)
  y = tl.reshape(tl.join(xe*c - xo*sn, xo*c + xe*sn), (BLOCK_S, BLOCK_D))   # 交错回去
  tl.store(o_ptr + ... + offs_d[None,:]*so_d, y, mask=m)
  ```

  实测（`[32,8,1024,64]`，`BLOCK_S=32`，NPU=910B2C）：

  | 写法 | 耗时 | vs 参考 |
  |---|---|---|
  | `partner` gather | 74.35 ms | — |
  | 同样二次 load、但地址连续（隔离对照） | 1.04 ms | — |
  | **`tl.split`/`tl.join`** | **1.257 ms** | **0.00e+00（位精确）** |

  ⇒ **gather 寻址本身的代价约 70x**（两个 shape 实测 71.3x / 68.7x），
  与「多做一次 load」无关 —— 隔离对照证明二次连续 load 只要 1.04ms。

  ⚠️ 量级参照：该 kernel 用 gather 时单独耗时 **152 ms**，而 torch 的**整个算子**
  只要 **10.5 ms**。也就是说 L1.15 把 RoPE 外提解决了正确性之后，
  **外提出来的 kernel 会自己变成新瓶颈**，必须一并按本条处理，否则
  `use_rope=True` 的用例 speedup 会掉到 **0.03~0.10x**（实测），
  而 `use_rope=False` 的用例是 2.0~6.7x。

  ⛔ 配套证伪：**放大 `BLOCK_S` 不是解法** —— 64/128/256 全部编译失败
  （`ConvertLinalgR...`），只能保持 32，所以必须从寻址模式上解决。

  `tl.split` / `tl.join` / `tl.interleave` 在 triton-ascend 3.2.2 上均可用。

- ⛔⛔ **变换 kernel 必须为「输入」和「输出」分别传 stride，禁止共用一套。**
  这是 L1.15 外提后最容易踩的坑，且**用连续张量自测时完全测不出来**。

  真实输入常是**非连续的转置视图**（典型：`x.view(B,S,N,D).transpose(1,2)`），
  而 workspace 用 `torch.empty_like(x)` 建出来是**连续**的 —— 两者 stride 不同：

  | 张量 | stride |
  |---|---|
  | `query`（转置视图） | `(262144, **64, 512**, 1)` |
  | `q_in = torch.empty_like(query)` | `(262144, **32768, 64**, 1)` |

  若 load / store 共用一套 stride，store 会**散射到错误地址**。实测同一份 kernel：

  | 输入 | max\|diff\| |
  |---|---|
  | 连续张量 | **0.0**（假象） |
  | 非连续转置视图（真实输入） | **6.65** ❌ |

  ⚠️ 该错误是**量级性**的（~1e0），不是精度问题；且**只在需要变换的分支出现**
  （不做变换时 `q_in = query` 无拷贝，不暴露），极易误判为 kernel 数学写错。

  正确写法 —— 两套 stride 各传各的：
  ```python
  rope_kernel[grid](
      query, q_in,
      B, N, S, D,
      query.stride(0), query.stride(1), query.stride(2), query.stride(3),   # 输入
      q_in.stride(0),  q_in.stride(1),  q_in.stride(2),  q_in.stride(3),    # 输出，必须独立
      ...)
  ```
  ```python
  x = tl.load (x_ptr   + b*si_b + n*si_n + offs_s[:,None]*si_s + offs_d[None,:]*si_d, ...)
  tl.store    (out_ptr + b*so_b + n*so_n + offs_s[:,None]*so_s + offs_d[None,:]*so_d, ...)
  ```
  ⇒ **自测变换 kernel 时必须用非连续输入**，否则测不出这个 bug（本条即由此漏测得来）。
  这与 L1.6「host 端不做 permute / contiguous」是配套的：正因为不许调 `.contiguous()`
  把 stride 归一，就必须在 kernel 里显式吃下两套 stride。

- ✅ **长序列 RoPE：`inv_freq` 必须与参考实现「逐位相同」，禁止代数化简。**
  golden 是参考实现的 **fp32** 结果，目标是**对齐 torch**，不是「算得更准」。
  `angle = position × inv_freq`，`position` 最大 `S_K-1`，于是 `inv_freq` 的
  **1 ulp 偏差被线性放大 S 倍**；`S=8192` 时足以把输出顶到 ~2e-3，而 fp32 用例
  容差通常只有 1.0~1.5e-3。

  参考实现（典型写法）：
  ```python
  inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2).float() / d_k))
  ```
  实测各写法与 torch 的逐位一致性（`d_k=64`，共 32 个元素）：

  | Triton 写法 | 逐位相同 | pos=8191 处 sin 差 |
  |---|---|---|
  | `tl.exp(-e * ln1e4)`（代数化简） | 22/32 | 4.11e-04 ❌ |
  | `1.0 / tl.exp2(e * log2_1e4)` | 19/32 | 1.54e-04 ❌ |
  | **`1.0 / tl.exp(e * ln1e4)`** | **32/32** | **0.00e+00** ✅ |

  ⇒ **保留参考实现的运算结构**：它先算正指数再取倒数，你也必须先算正指数再取倒数。
  `exp(-x)` 与 `1/exp(x)` 数学上恒等，但舍入路径不同，差 1 ulp 即失败。
  对齐之后误差回落到 `tl.sin` vs `torch.sin` 的 **1.19e-07**，即 fp32 精度极限。

  ```python
  e   = (tl.arange(0, BLOCK_D) // 2 * 2).to(tl.float32) / D
  inv = 1.0 / tl.exp(e * 9.210340371976184)   # ln(10000)，先正指数、后取倒数
  ang = s_offs.to(tl.float32)[:, None] * inv[None, :]
  ```

- ⛔ **不要指望用 `tl.float64` 提精度**——Ascend 后端不支持，实测直接编译失败：
  `LLVM ERROR: unsupported datatype for arith::TruncFOp`。fp32 是硬上限，
  因此**只能靠对齐参考实现的运算结构**，不能靠提升精度。

- ⛔ **hi+lo 二段拆分对「大角度三角函数」无效**（对 `p` 降精度喂 `tl.dot` 有效，别混淆）。
  把 `inv_freq` 截成 `hi+lo` 后，`a_lo = position × inv_lo` 可达数弧度，
  一阶展开 `sin(a_hi+a_lo) ≈ sin(a_hi) + cos(a_hi)·a_lo` 的前提（`|a_lo| ≪ 1`）不成立。
  实测保留 8~12 位尾数时误差 0.24~14.6，比朴素 fp32 的 2.95e-04 差 3~5 个数量级。

- ⛔ **禁止手工做三角函数的范围规约**。`tl.sin` / `tl.cos` **自带正确的大幅角规约**，
  手工 `a - floor(a / 2π) * 2π` 反而会因 **2π 在 fp32 下不可精确表示**而引入更大误差。

  实测（RoPE 角度 `position × inv_freq`，`S=8192` ⇒ 角度范围 `[0, 8191]`，14.7% 的元素 >1000 弧度）：

  | 写法 | vs `torch.sin` | vs fp64 |
  |---|---|---|
  | `tl.sin(角度)` | **1.19e-07**（fp32 精度极限） | 1.19e-07 |
  | `tl.sin(fmod 2π 之后)` | **4.40e-04**（差 3700 倍） | 4.40e-04 |

  ⇒ 长序列上 RoPE 精度不达标时，**不要往"角度太大、要先规约"这个方向查**——已实测证伪。

- **判别信号**：按 §1.1 反推每次迭代成本，**远超 4~10us 的正常区间**（如数百 us）
  ⇒ 立刻查 KV 循环体内有没有本可外提的输入变换，**不要去查 tiling 或访存**。
  另一个信号是逐 case 分布呈 **小 shape 尚可、大 shape 崩塌**（迭代数放大了固定成本）。

### ★ 跨算子结论冲突的统一判据（三条，照搬会互相矛盾）

同一改动在不同 FA 算子上给出过**相反**结论。下面给出判别前置条件，**不要只看结论**。

| # | 冲突 | 甲方结论 | 乙方结论 | **统一判据** |
|---|---|---|---|---|
| 1 | `scale` 写法 | host 侧 fp32 倒数**乘法**（−36% 时延；低精度下还是逐位契约的硬要求） | 保持 `/ sqrt_dh` **除法**形式（`x/s ≠ x*(1/s)`，数值对齐要求） | **先看参考实现怎么写**：参考用 `aclnnDivs`/除法且判据按 ulp 卡 ⇒ 必须复刻除法语义（见 L1.4 与精度契约卡）；参考对 scale 无逐位要求 ⇒ 看 `aiv_vec_ratio`：**≳0.35 才做倒数乘法**，否则向量核不在关键路径、改了也是打平 |
| 2 | `scf.for` 层数 | 合并 KV 整块/尾块 **+11.7%**、压平循环维 **+3.7%**（每层循环自带一套同步） | 去掉单块路径的 `scf.for` **−10%**（编译器失去 multi-buffer 流水机会），另一算子上三种写法**全部 aicore exception 507015** | **"少一层 `scf.for` 必然更快"不成立**。判据：合并的是**同层级的冗余分支**（整块+尾块两个循环 → 一个）⇒ 值得做；去掉的是**唯一的那层循环**（单块特化）⇒ 大概率负收益或崩编译器，投入前先用单 case 脚本试崩 2~3 个 shape（每次 ~1 分钟） |
| 3 | QKV 三投影融合成一次 GEMM | **无收益**——三次调用读的同一份 x 已命中 L2，"3 次读"实际只有 1 次 GM 流量 | 是**真实收益**（x 只读一遍），但会让 benchmark 口径分数变差 | 判据是 **x 是否真的走 GM**：`aic_mte2_ratio` 高且 x 被重复读时才有收益；x 较小、已驻 L2 时是零收益。⚠️ 无论哪种，**都不要因为口径分数而做或不做**（§7.2） |

> **通例**：跨算子搬运结论前，先确认三件事——**参考实现的算术路径**（决定精度类结论）、
> **profiling 画像**（决定该优化落不落在关键路径）、**编译器/CANN 版本**（决定结构类结论是否仍成立，见 §7.3）。

---

### L1.16 ★★★ fp16 / bf16 路径必须**逐步舍入回输入 dtype**，禁止全程 fp32

低精度用例的 golden 是参考实现在 **fp16 / bf16** 下算出来的：`torch.matmul` 对
fp16 输入返回 fp16、`F.softmax` 返回 fp16、下一个 `matmul` 又在 fp16 上做。
**每一步都有一次舍入。** 实现若全程用 fp32，会「比参考更准」，反而不达标。

判据（`verify.py`）：`matched_ratio >= 0.9`，逐元素相对阈值
`rel_thr = 2^-10`（fp16）/ `2^-7`（bf16），小值域 `sv_thr = 2^-11 / 2^-8`。

实测（全程 fp32 的实现 vs 参考，四组 shape/dtype）：

| shape / dtype | matched_ratio | 判据 |
|---|---|---|
| `d_model=48`, fp16 | 0.7648 | ❌ |
| `d_model=60`, bf16 | 0.7733 | ❌ |
| `d_model=512`, fp16 | 0.7597 | ❌ |
| `d_model=768`, bf16 | 0.7631 | ❌ |

⇒ 全程 fp32 **稳定落在 0.76 左右**，离 0.9 差得很远，与 shape 无关。
op20 实测失败用例的 `matched_ratio` 是 0.705~0.843，同一量级。

正确写法 —— 为低精度单开一条路径，在**参考会舍入的每个位置**都 cast 回 `ELEM_TY`：
```python
# fp32 路径：累加器 fp32，中间不降精度
# 低精度路径（ELEM_TY = tl.float16 / tl.bfloat16）：
s_r = tl.cast(tl.dot(q_tile, k_t), ELEM_TY)     # ← 参考的 matmul 返回低精度
s_r = tl.cast(s_r * inv_sqrt, ELEM_TY)          # ← 参考的除法也在低精度上做
...                                              #   softmax 同理
acc += tl.cast(tl.dot(w, v_tile), tl.float32)   # ← PV 在低精度上做，再累加
```

⚠️ 这与 **L1.1**（`tl.dot` 用 fp32 累加器）**不矛盾**：累加器仍是 fp32，
但**每次 `tl.dot` 的输出**要按参考的行为舍回低精度。区别在于"累加用什么精度"
和"中间结果以什么精度落地"是两件事。

⇒ 与 §2 里 `inv_freq` 那条同一原理：**目标是对齐参考，不是算得更准。**
判断顺序永远是：先问 golden 是用什么精度算的，再决定实现该用什么精度。

#### ⭐ fp32 镜像：cast 分支必须 **fp16 / bf16 / fp32 三档写全，默认兜底 fp32**

L1.16 只讲低精度方向，从零生成容易写出"else = 低精度"的反向 bug：

```python
# ❌ 两趟路径里 E 落盘（fp32 输入被无条件 cast 成 bf16，丢 ~3 个数量级精度）
if is_fp16:
    E = exp_val.to(tl.float16)
else:
    E = exp_val.to(tl.bfloat16)      # ← fp32 输入也走这里，max_abs_diff ~5e-3

# ✅ 三档写全 + fp32 兜底：只在"参考会舍入的位置"才 cast 回输入 dtype
if is_fp16:
    E = exp_val.to(tl.float16)
elif is_bf16:
    E = exp_val.to(tl.bfloat16)
else:
    E = exp_val                       # fp32 路径任何中间量都不许落低精度
```

- **判别信号**：失败集合**严格按 dtype 划分**（全部且仅仅是 fp32 case），且
  `max_abs_diff ~1e-3~5e-3`、`matched_ratio ~0.8` ⇒ 直接查 `else` 分支有没有把
  fp32 中间量 cast 到低精度。实测某 FusionAttention 的 fp32 case 16/17/18 全错即因此。
- **规则一句话**：L1.16 的镜像——低精度"参考会舍入的每个位置都舍"，fp32"参考不
  舍入的每个位置都不许舍"。**cast 的 `else` 兜底永远是 fp32。**

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 主骨架

```
host:
  inv_sqrt = 1.0 / sqrt(head_dim)          # ★ host 侧算好倒数，kernel 内不做除法（§4.1）
  BLOCK_KV = min(128, ceil16(S_K))
  BLOCK_Q  = 满足 BQ*BKV <= 8192 的最大 2 的幂, 再 min(max(32, ceil32(S_Q)))
  while BQ * BLOCK_D * esz > 64KB: BQ //= 2
  BLOCK_D  = ceil16(D)
  MULTI_BLOCK = (max_kvblk > 1)            # ★ 分档开关（§4.2）
  HAS_UPPER/HAS_LOWER/HAS_SOFTCAP          # ★ constexpr 特化（L1.7）
  grid = min(NUM_CORES, B*H*cdiv(S_Q,BQ))

kernel(每个 task = 一个 (b, h, q_block)):
  加载 Q tile 一次，常驻
  m = -1e30; l = 0; acc = 0
  kv_lo, kv_hi = 区间收缩(§3.2)            # ★ 不对齐（L1.8）
  for kv in range(kv_lo, kv_hi, BLOCK_KV):
      scores = tl.dot(Q, Kt) * inv_sqrt    # 原生 dtype 操作数 + fp32 累加
      scores = 掩码(constexpr 分支)         # 有限极小值
      m_new  = tl.maximum(m, rowmax, propagate_nan=ALL if MULTI_BLOCK)
      p      = exp(scores - m_new)
      pv     = tl.dot(p, V)                # ★ Cube 提前发射（§4.7）
      alpha  = exp(m - m_new)
      acc    = acc * alpha + pv
      l      = l * alpha + sum(p)
      m      = m_new
  store(acc / l)                           # 循环外的除法不必改倒数（§4.1 失效边界）
```

### §3.2 KV 扫描区间上下界折叠（`fa-fwd` / `fa-gqa` 核心）

```python
# host 侧：causal 与 window_right 折叠成单一上界，window_left 折叠成下界
upper = min(0 if causal else INF, window_right if window_right >= 0 else INF)
lower = -window_left if window_left >= 0 else -INF
# kernel 内：对 q-block [q0, q0+BQ) 取并集
kv_lo = max(q0 + delta + lower, 0)                       # delta = S_K - S_Q
kv_hi = min(q0 + BQ - 1 + delta + upper + 1, S_K)
```

区间外整块被掩，对 online softmax 无贡献，跳过**数值等价**。实测 **+30%**，冗余倍数最高 70x。
**留一个不可裁剪的 case 当对照组**，改动前后应当不变——立刻确认收益来源正确。

> ⛔ **适用边界：golden 必须真的按 causal/window 收缩。**
> 当 golden 是 CANN op 且它**忽略窗口**（`npu_fusion_attention` 实测
> sparse_mode 0/4 的 pre/next 全被忽略 == 全可见，见 §6.1）时，收缩会把
> **窗口外"本来就有贡献"的列**漏掉，直接算错——此时**禁止**做区间收缩。
> 判别：先探针 golden 对 `pre_tockens / next_tockens` 的敏感性（§6.1），
> 它忽略窗口 ⇒ §3.2 整个不适用。这条对 GQA / SparseFlashAttention 复用同样踩。

### §3.3 趟数结构（由精度契约倒逼，见 §6.3）

| 条件 | 结构 |
|------|------|
| `S ≤ BLOCK_KV`（单块） | **一趟**，scores 常驻片上 |
| `S > BLOCK_KV` + 参考要求"先归一化再舍入" | **延迟归一化**（首选）或 两趟（见下，⚠️在线两趟易踩坑） |
| 参考在 fp32 上算 | 标准一趟 online softmax |

**趟数判据（scale 阈值）**：`scale` 显著 **< 1/√d**（如 0.08838 / 0.125）⇒ softmax
**近均匀** ⇒ online rescaling 的滚动 alpha 累积 fp32 舍入，MERE 顶穿 fp16 阈值。
实测（FusionAttention，fp16，scale=0.08838）：单趟 MERE 1.187e-3 > 阈值 9.77e-4。
**第一次生成时看到小 scale 就直接选多趟/延迟归一化，不要撞一次失败再改。**

#### ★★★ 精度全过首选结构：延迟归一化（0811/3 61/61 实锤）

**softmax 输出「未归一化 E」（不除以 l），最后 `Out = E @ V / softmax_sum`。**
从结构上**绕开 online rescaling 的滚动 alpha**——不存在跨 KV 块的 alpha 累积舍入，
是 FP 精度全过的根本方案（0811/3 FusionAttention 61/61，MERE 对齐到 ~1.8e-4）。

- **必须**：softmax kernel 写 `exp_buf`（未归一化）+ `softmax_max` + `softmax_sum`；
  再单独 `output_kernel` 做 `Out = (E @ V) / softmax_sum`（最后一步才归一化）。
- **为什么比"在线两趟"可靠**：在线两趟的"趟 2 归一化"仍基于趟 1 的滚动 m/l，
  滚动 alpha 的 fp32 舍入并未真正消除；延迟归一化则完全不 rescale，无此累积。
- **性能兼顾**：三段式多 kernel 会带来 GM 往返（scores→GM→softmax→E→GM→PV）。
  用 **score+softmax 融合**（`score_softmax_kernel`，QK dot 后立即算 E/max/sum 写 GM，
  不写完整 scores）把 GM 往返减半；PV 再读 E。这是"精度全过 + 性能好"的平衡点
  （0811/3 用此结构 0.89x，且在 score+softmax 融合后已接近单 kernel 的性能）。

#### ⚠️ 反例（0818/3 实锤）："在线 softmax + two_pass" ≠ 延迟归一化，4 例精度失败

0818/3 按"在线 softmax 加个 `two_pass` 开关"实现，仍 **57/61**（case 20/36/54/55，
全是 fp16 大 S，MERE 9.9e-4~1.59e-3 超阈）。根因：它的 two_pass 只是**在线 rescaling
内多算一次 max**，滚动 alpha 的舍入没消除；而 golden 是**延迟归一化/两趟物化**的语义。
判别：`two_pass` 后 MERE 仍在 1e-3 附近而非 ~1.8e-4 ⇒ 结构没对齐，**改成延迟归一化**，
不要用"golden 噪声"当台阶（0811 用相同 golden 做到 1.8e-4，golden 是可复现的）。

> ⚠️ **D≥256/512：CANN 内部按 K-chunk 累加 QK，triton 复刻只能三段式（拆半核 launch）。**
> golden 是 CANN 时，D=512 的 QK 在 CANN 里是 K=256 分块累加，与整段单 `tl.dot`
> 差 ~5 ULP，exp 放大进 ssum 顶穿 fp32 阈值（实测 1.267e-3）。⚠️ **triton-ascend
> 编译器必然把单 kernel 内多个 `tl.dot` 融合成单个 K 归约**——runtime 循环 /
> debug_barrier / 分离变量 / 显式两半**全部无效**（已实证）⇒ 只能**三段式**：
> 2 次独立 score 半核 launch（各 K=256 dot 写/累加到 GM workspace）+ softmax 核。
> chunk-k=256 后 ssum 违例 1→0。判别信号：bf16/fp32 大 D case 单独失败、
> `ssum` 差 ~1e-3 量级。

⚠️ **不要用"减小 `BLOCK_Q` 换单趟"**：总迭代数 = `ceil(S/BQ)·ceil(S/BKV)·趟数`，BQ 减半使第一项翻倍、趟数 2→1，**正好抵消**。已算账否决。

### §3.4 `fa-mha` 的两段结构

投影段（GEMM）与 attention 段是**两个独立的优化对象**，profiling 里分别看 `Duration` 占比，主攻目标每轮会换（实测从 attention 64%/linear 36% 变到 46%/54%）。
⚠️ **不要**把 4 次投影和 1 次 attention 合并进同一个 `@triton.jit`（用 `MODE: tl.constexpr` 分支）——见 §7.2。

---

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构必须重新设计）

### §4.1 ★ host 侧倒数乘法替代 kernel 内 fp32 张量除法

```python
# ❌ 每个 KV 迭代对 [BQ, BKV] 做一次 fp32 除法
scores = tl.dot(q, tl.trans(k)) / tl.sqrt(tl.cast(Dh, tl.float32))
# ✅ host 侧算好 head_dim ** -0.5
scores = tl.cast(tl.dot(q, tl.trans(k)), tl.float32) * inv_sqrt
```

- **适用判据**：向量核在关键路径（`aiv_vec_ratio ≳ 0.35`）。实测一个算子 **−36%**，另一个算子上（`aiv_vec_ratio` 仅 0.25~0.32、向量核 2/3 时间在等 Cube）**两次证伪**。
- **失效边界**：**不能外推到循环外**。末尾 `acc / l[:,None]`（每 task 一次）改倒数广播乘实测无收益。判据是「该除法是否在内层循环里、作用于二维 tile」。

### §4.2 ★ `tl.maximum(propagate_nan=ALL)` —— **必须按 KV 块数分档**

```python
if MULTI_BLOCK:      # host 侧算出的 max_kvblk > 1
    m_new = tl.maximum(m, rowmax, propagate_nan=tl.PropagateNan.ALL)
else:
    m_new = tl.maximum(m, rowmax)
```

- 默认 `propagate_nan=NONE` 在 IR 里被展开成 **7 条向量指令**（2×isnan + vmax + 2×vsel）；`ALL` 直接对应硬件单条 `vmax`。
- ⚠️ **全局开启是负收益**：实测 6.073 → 5.5302（**−9%**）；按 `max_kvblk>1` 分档 → 6.2362（+2.7%）。另一算子上单块 case 退化 25~45%（两次重复一致）。
- **语义等价性论证必须写进注释**（依赖 F2：无 NaN）。

### §4.3 ★ BLOCK 开到 UB 上限（本类算子最大的一笔）

迭代总数 = `B*H*ceil(S/BQ) × ceil(span/BKV)`，两个 BLOCK 同时翻倍 ⇒ 迭代数变 1/4，**整套 `pipe_barrier` 直接少一半**。

| SQ/SKV | 64/64 | 128/64 | 64/128 | **128/128** | 256/* |
|--------|-------|--------|--------|-------------|-------|
| 子集几何平均 | 0.697 | 1.129 | 1.155 | **1.42** | 编译失败 |

**UB 上限由编译器直接报数字**（`requires N bits while 1572864 bits available`），看错误信息就知道差多少 ⇒ **放大到编译失败为止**。
⚠️ 上限**不可用简单阈值预测**：同为 `128x128`，D=96 编得过、D=128 失败、**D=160 反而编得过且快 22%**——占用还受 constexpr 分支组合影响。

### §4.4 权重按 tile 形状预重排（**仅 `fa-mha`**，判据 `MTE2>60% 且 MAC<50%`）

投影权重是常量，可在 `__init__` 里用 CPU 任意重排，运行期零开销。

```python
# [out,in] → 转置成 [in,out]（数值逐位不变）→ 补零到 [K_pad, NBN*BLOCK_N]
# → 拆成 [NBN, K_pad, BLOCK_N] 连续
t = torch.zeros(kp, nbn * blk_n, dtype=dtype)
t[:d_model, :d_model] = w.t()
packed = t.view(kp, nbn, blk_n).permute(1, 0, 2).contiguous()
```

b tile 基址 `w_ptr + block_n*(K_pad*BLOCK_N)`，行跨度恰为 `BLOCK_N` ⇒ 整块连续，且补零后**不再需要掩码**。
- **两级递进**：先 `[out,in]→[in,out]`（**数量级收益**），再按 tile 重排（+5.0%，**无退化 case**）。
- ⚠️ **同一思路搬到 attention 的 k/v 上完全无效**（§5.2）。

### §4.5 投影分块：dtype 分档 + **并行度保护**（两半缺一不可，**仅 `fa-mha`**）

```python
par_thr = num_cores // 2            # 阈值扫参：24→3.6633、16→3.8283、12→3.8576
while nblocks(blk_m, blk_n) < par_thr and (blk_n > 64 or blk_m > 16):
    交替 blk_n //= 2 / blk_m //= 2  # 先缩 N 再缩 M
```

只放大不保护 = **−3.7%（拒绝）**；加保护 = **+2.7%**。根因是**块数塌陷**（`M=89` 时 `BLOCK_M=128` 只剩 3 个块，24 核只用 3 个）。
**fp32 必须单独一档**（元素大一倍，同 BLOCK 撑爆 UB）：实测 fp32 最优 `(64,128,256)`，16-bit 最优 `(128,256,256)`。

### §4.6 低精度 `p` 二段拆分（**仅 fp16/bf16 输入**）

```python
if PV_SPLIT:                       # = (dtype is not float32)
    p1 = p.to(in_dtype); p2 = (p - p1.to(tl.float32)).to(in_dtype)
    pv = tl.dot(p1, V_tile) + tl.dot(p2, V_tile)     # ~22 bit
else:
    pv = tl.dot(p, V_tile.to(tl.float32))            # fp32：本就是 no-op，别动
```

省掉 UB 里 `[BLOCK_KV, BLOCK_D]` 的 fp32 副本（64x128 时 **64KB，UB 里最大的一块**）。实测 **+2.7%，50/50 全过**。
**fp32 输入不要开**（无 buffer 可省，且阈值最紧、22 位没有余量）。

### §4.7 CV 指令重排（成本极低，试一次，但**不要预设收益**）

把 `tl.dot(p,V)` 提前到 `p_sum`/`alpha` 之前发射（纯重排、数值等价）。
**三个同类算子三个结果：0 / +1.9% / +6.4%。** 幅度不可预期。

---

## §5 Phase 4 优化点清单

映射到 `triton-latency-optimizer` 的**优化点 #30（Attention/FA 类算子专用优化）**。
完整判别流程、profiling 字段对应表、天花板估算模板见
`@../../../ops/triton-latency-optimizer/references/operators/flash-attention-optimization.md`。

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|----------|
| 1 | ★ 全面 `tl.dot` 化 | 不可测量 → 2.36x | `aic_mac_ratio<10%` 且有 `for d in range(head_dim)` |
| 2 | ★ KV 扫描区间按 causal/window 收缩 | **+30%** | 有 causal 或有限 window |
| 3 | ★ `kv_lo` 不对齐 + tile 联合放大 | **+21%** | **两者必须成对做**；单独放大只有 +1.2% |
| 4 | ★ mask 分支 constexpr 特化 + 上下界折叠 | **+21%** | mask 为运行时标量 |
| 5 | ★ BLOCK 开到 UB 上限 | 大 case attention **时延减半** | 放大到编译失败为止 |
| 6 | ★ 删除 host 端 permute/contiguous/pad | **+16%**（保守） | 时间归属出现 `Transpose`/`ViewCopy` |
| 7 | ★ 权重按 tile 预重排 | **+5.0%** | 仅 `fa-mha`，`MTE2>60% 且 MAC<50%` |
| 8 | ★ 分块 dtype 分档 + 并行度保护 | **+2.7%** | 仅 `fa-mha`；缺保护那半 = **−3.7%** |
| 9 | `propagate_nan=ALL` 按块数分档 | 多块 case **−4.4% 时延** | **全局开是 −9%** |
| 10 | CV 指令重排 | 0 / +1.9% / +6.4% | 成本低值得试，不可预期 |
| 11 | 低精度 `p` 二段拆分 | **+2.7%** | 仅 fp16/bf16 |
| 12 | fp32 张量除法 → host 侧倒数 | −36% / 打平 | **仅当 `aiv_vec_ratio ≳ 0.35`** |
| 13 | K 直接读成转置态 | +0.7% | 机理假设已被 IR 证伪，收益来自 load 路径 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结果 |
|------|------|
| 编译开关穷举（14 个逐个单测） | **命中率 0**。`multibuffer` −10%、`enable_mixed_cv` −9%（只在 910_95 生效）、`limit_auto_multi_buffer` 换契约后 −9%。**最多花 20 分钟测完就放弃** |
| 掩码链合并（12 条 → 4 条向量指令） | **0%** |
| `BLOCK_D == D` 时消掉 d 方向掩码（45/50 适用） | −1.3%（噪声内） |
| `libdevice.tanh` 消掉 softcap 的 fp32 除法 | **两次测量均打平** |
| bool 掩码换 fp32 加性 bias | **编译失败**（UB 超限） |
| 运行时 `scf.if` 只在对角/边界块做掩码 | 噪声内 |
| K/V permute 成连续布局 | **15/17 case 变慢**。先用 `.contiguous()` 花 5 分钟证伪，别去实现零拷贝方案 |
| **把 `scale` 折进 `q`**（循环外 `q *= scale`，循环内去掉 `tl.dot(q,k) * scale`） | **一致变慢 14%~39%**（4 个 shape 实测，输出逐位相同）。`tl.dot(...) * scale` 已被融进 cube epilogue，手工折叠反而多出一个向量算子并打断融合。**看起来最"显然"的优化，实际是负收益** |
| ALiBi 的 `q_pos = offs_q.to(fp32)[:,None]` 提出 KV 循环 | **0%**（1.149x → 1.148x）。编译器已自动外提循环不变量，手工提无收益 |
| KV 循环拆「无掩码整块 + 对角/尾块」 | **编译失败**（循环体复制使 UB 翻倍） |
| 单块时去掉 kv 的 `scf.for` | 一个算子 **−10%**（编译器失去 multi-buffer 流水机会）；另一个 **aicore exception 507015**，三种写法全崩 |
| 保留 `for`、把累加改赋值 | `'scf.for' op 0-th result` 编译失败 24/50 |
| grid 超订（24→48） | 无收益 |
| `exp2` + log2e 折进 scale | 无收益（`tl.exp` 本就走 exp2） |
| scale 与 mask 合并成一次 FMA | **编译失败**（UB 超限） |
| 波次量化单目标代价模型 | **−21%**（模型不含访存项，会选出 16×64 这类小 tile） |
| tile 按代价排序 + 编译失败自动回退 | 完全打平（次优候选之间排序不可靠） |
| **纯 Vector 版 attention**（去掉 `tl.dot` 想编成 `AI_VECTOR`） | **小 case 子集 −33%**。根因：triton-ascend 按"有没有向量侧计算"决定 kernel 类型，**去掉 `tl.dot` 仍是 `MIX_AIC`**，只是把矩阵乘搬到 Vector 上还保留全部 MIX 握手开销 |
| `al.scope(core_mode)` 手写 CV 流水 | **`bishengir-compile` LLVM stack dump**（910B2+CANN 8.5.1）。投入前先用最简两段 kernel 做 10 分钟门槛测试 |
| `al.multibuffer` 按张量开双缓冲 | **产生 NaN**，40/50，失败集恰为全部多块 case |
| `al.parallel` 用于 KV 循环 | **40/50 → 33/50 精度崩**。online softmax 有跨迭代依赖（`m`/`l`/`acc` 递推），**语义上不可并行**；用于外层 task 循环语义正确但实测更慢 |
| 手工循环不变量外提 / 手工 buffer 复用 / 手工删常量张量 | **三次实测全部 no-op**。判据：两轮的失败数完全相同（都是 6/50、44 个编译错），而后者比前者多"释放"了一块 `[128,64]` fp32 ⇒ 省下的 UB 是幻觉。**源码层面的内存优化在 triton-ascend 上基本无效，UB 手算模型预测不了真实边界** |
| 批处理 2 个 KV tile 凑重叠 | **−14%**，且串行度 1.69→1.71 纹丝不动 ⇒ 编译器仍逐条串行。**流水要在迭代内部做，不是迭代之间** |
| 手工软件流水（提前发射下一块 QK） | 8/50，多持一个分数块撑爆 UB |
| K 预转置为 `[B,Dh,S]` 消除 `tl.trans` | **回归**。host 侧转置只在 `M*N*K ≥ 1e12` 才划算，本规模下 kernel 内 `tl.trans` 才是正解（注意与 §2.12「K 直接读成转置态 +0.7%」不同：那是 **load 成转置态**，不是 host 预转置） |
| 按 head 分组共享 K/V（`HG=2`） | 不值得做：`HG=2, BLOCK_SQ=64` 与 `HG=1, BLOCK_SQ=128` 的 K/V 扫描遍数**完全相同**、UB 也相同，是同一个杠杆的两种写法 |
| 把 sq tile 拆散做三维并行 `(B,H,S-tile)` | **回归**。K/V 跨 sq 的复用消失、访存随遍数倍增。⚠️ 与 L1.13 是同一个量的两个方向：**每次 K/V 加载服务多少 query 行** |
| 非融合三段式（QK → softmax kernel → PV） | **访存账否决**：物化 `[B*H,S_Q,S_K]` fp32 = 370MB，4 趟 ≈ 1.5GB ≈ 900us，而当前整个 kernel 才 400us |

### §5.3 ⛔ 结构性下限：小 shape 上**绝不要**拆 kernel

任何含 softmax / elementwise 的 kernel 在 triton-ascend 下**必然编成 `MIX_AIC`**，每次发射带 **~4.5us** 固定成本，是工具链的结构性下限，**改写数学表达绕不开**。
⇒ 小 case 的整个 impl 时延也才 6~8us，**每多拆一个 kernel 就多 ~4.5us**。
判别：看 `kernel_details.csv` 的 `Accelerator Core` 列（`MIX_AIC` vs `AI_CORE`），或看 IR 里有没有 `_mix_aiv` 函数。

### §5.4 三个会误导的 profiling 字段

| 字段 | 陷阱 |
|------|------|
| `aic_scalar_ratio` 高 | ❌ **不是标量算术过多**。三个算子上五次改法全落空。真实构成是 113 条同步指令（`set_flag`/`wait_flag`/`pipe_barrier`/`sync_block`）的等待被计入标量管线。**先 dump IR 数同步指令，不要直接改标量代码** |
| `cube_utilization(%)` | ❌ **名字骗人**。同一行 `cube_utilization=88.383%` 与 `aic_mac_ratio=0.084` 并存。判断 Cube 忙不忙**一律用 `aic_mac_ratio`** |
| `aiv_vec_ratio` 高 | ⚠️ **半对**。判据是「这条指令值多少拍」不是「有几条指令」：去掉 1 条 fp32 `vdiv` = −36%，去掉 8 条掩码指令 = 0% |

---

## §6 精度闸门（先过闸门，再谈性能）

- 完整四条逐位契约、1-ulp 匹配率诊断法、误差放大链算法：
  `@../../../ops/triton-precision-debug/references/attention-lowprec-contract.md`
- **判据推论**：若失败集合**严格与 dtype 相关**（如"全部且仅仅是 fp16/bf16 case"），基本可直接判为**算术路径不匹配**，不用去搜 tiling。
- **判据推论（参数版）**：若失败集合**严格按某个运行时参数划分**（如"全部且仅仅 `sparse_mode=4` 的 case"），先探该参数在 golden 里的真实语义（§6.1），**不要去改 kernel 数学**——多半是 golden 行为与文档/直觉相反。
- **正式结论一律走 `verify.py`，不改阈值、不用 `--verify_not_required`。** 扫参阶段可用单 case 脚本秒级淘汰明显错误的变体。

### §6.1 ⭐ CANN 语义探针先行（golden = `npu_*` op 时**必做**，Phase 2 第一步）

`torch_npu.npu_xxx` 的 golden 是**真实 op 行为**，可能与文件内 torch fallback（docstring）
**相反**。每个参数 5 分钟独立探针：固定其他输入，只改该参数，比对三输出 diff。
**判别信号**：失败集合严格按某参数划分 → 探该参数，不要改 kernel。

已实测（FusionAttention `npu_fusion_attention`）语义表：

| 参数 | 文档/直觉语义 | **实测真实语义** |
|---|---|---|
| `sink` | per-head bias，加到 scores 上 | **attention-sink 列**：`m_init = sink[n]`、`l = 1`、`exp(sink-m)` 进 softmax 分母（无对应 V 列） |
| `pse` | — | **先加再乘 scale**：`(QK + pse) * scale`，不是 `QK*scale + pse` |
| `atten_mask` | — | **True = 掩蔽**；掩码值 = `-FLT_MAX`、`m_init=-FLT_MAX`；全掩行 `smax=-FLT_MAX`、`ssum=Skv`、`out=mean(V)` |
| `sparse_mode` + `pre/next_tockens` | 按 band 窗口裁剪 | **0/2/3/4 全被忽略 == 全可见**（§3.2 在此前提下禁止收缩） |
| `inner_precise` 0/1/2 | 精度档位 | **对输出无影响** |
| `keep_prob < 1` | dropout | **golden 不可复现**（NPU 内部 RNG，同输入两次输出 diff~4.13）；任务层强制无 dropout |
| GQA `nkv` | — | `nkv = nq * H_kv // H_q`（按 head dim 比值） |

- ⚠️ 探针要在**直接调 `npu_xxx` 的环境**里做，不要信任静态代码阅读——实测 sink 忽略、
  mode4 忽略窗口、atten_mask 掩蔽方向都是探针实锤的，与文件内 fallback 完全不同。
- 与 [[triton-npu-native-ops-precision-recipe]] 呼应：`torch_npu.*` 命名空间允许用
  CANN 原生算子做 golden 级复刻。

### §6.2 CANN 输出契约（`npu_*` op 的返回值形状必须先钉死）

`npu_fusion_attention` 返回 **7 元组**：
`(attention_out, softmax_max, softmax_sum, reserved, seed, offset, mask_length)`，其中：

| 输出 | 形状 / 值 | 从零生成最容易错 |
|---|---|---|
| `attention_out` | 按输入 layout 的 `[B,S,H]` / `[B,N,S,D]` 等 | 布局错位 → 全错 |
| `softmax_max` | **`[B,N,Sq,8]`（尾维 pad 到 8，8 列同值）** | 按 `[B,N,Sq]` 输出 → shape 不匹配直接 fail |
| `softmax_sum` | **`[B,N,Sq,8]`（同上）** | 同上 |
| `reserved` | `torch.empty(0)` | 非空 → shape 不匹配 |
| `seed` / `offset` / `mask_length` | 恒 `0` | — |

只读 `Model.forward` 签名写 kernel 会按 `[B,N,S]` 输出 softmax_max/sum，verify shape
不匹配 fail 且完全查不到原因。**先按本表钉死输出契约，再写 kernel。**

### §6.3 精度契约会倒逼结构

当参考要求"先用最终的 `l` 归一化再舍入"时，online softmax 只能拿到滚动的 m/l ⇒ 必须多趟（见 §3.3）。
各步实测收益：三趟 → 单块单趟 **+21.9%**；多块三趟 → 两趟 **+10.1%**。

> ⚠️ **首选是延迟归一化而非"在线两趟"**（0811/3 61/61 vs 0818/3 57/61，见 §3.3）：
> 延迟归一化（softmax 写未归一化 E，`Out=E@V/sum`）从结构上绕开滚动 alpha 舍入，
> 是 FP 精度全过的根本方案。"在线 softmax + two_pass"只多算一次 max，滚动 alpha 舍入
> 仍在，fp16 大 S case 仍会 MERE 超阈。**失败后若 MERE 卡在 1e-3 附近而非 ~1.8e-4，
> 结构没对齐，改延迟归一化，不要用"golden 噪声"当台阶。**

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.0 ★★★ 官方 `speedup_vs_torch` 是**几何平均**，优化要盯最慢的用例

`benchmark.py` 的 `_geomean_speedup()` 取的是**逐 case speedup 的几何平均**，
不是总时间比、也不是算术平均。这一条决定了优化的**投入方向**：

几何平均在对数空间 ⇒ **把一个 0.5x 的用例修到 1.0x，收益等于把一个 2x 的用例提到 4x。**
盯着已经很快的大 shape 继续调，对总分几乎没有贡献。

实测杠杆（op54，50 个用例，其中 16 个 <1.0x、均值 0.753）：

| 情形 | 几何平均 |
|---|---|
| 现状 | 1.7413 |
| 算术平均（**不是**官方口径，会严重高估） | 2.3550 |
| **把 16 个 <1.0x 全提到 1.2x** | **2.0422** ✅ |
| 提到 1.4x | 2.1830 |
| 提到 1.6x | 2.3382 |

⇒ **达标路径不是"整体再快 15%"，而是"把垫底的那批修到 1.2x 以上"。**

操作步骤：
1. 从 `perf_result.json` 的 `per_shape_results` 里按 `speedup_vs_torch` 升序排列；
2. 只看 `<1.0x` 的那批，找它们的**共同特征**（典型是中小 shape、`B*N` 大、`S` 小）；
3. 为这批特征单独开一条分派路径，而不是改全局参数
   —— 全局改参数会同时动到已经很快的用例，往往净负收益（实测见 §5.2）。

⚠️ 慢用例常常**不是**大 shape。op54 里最慢的是 `B=16,S=256`（0.535x）和
`B=1,S=1024`（0.641x，仅 0.5M 元素），瓶颈是分块与同步开销占比高，
而 torch 在这些形状上走的是高度优化的批量 matmul。


### §7.1 环境闸门与噪声

- torch 参考代码恒定，其时延是天然环境探针。**必须用双探针**（全量几何平均 + 大 shape 子集）：实测某轮全量探针在带内、大 shape 探针却比基线高 31%，该轮大 shape 数据整体失真。
- **噪声水平**：20 case 单次 ±2%；同一份代码复测两次可差 2.2%；10 case 基线包夹（base→A→base）可到 ±0.3%。
  ⇒ **≤3% 的改动必须用包夹协议复测**，否则会把噪声当收益采纳（实测有 4 个候选栽在这里）。
- ⚠️ **闸门必须双侧——异常变快和异常变慢一样会伪造结论**。实测两次拦截：机器高负载时 torch 时延 +85%（若不拦会把污染数据当成 +3% 提升）；机器临时空闲时 torch 时延 **−40%**（若不拦会把一个**回归**当成 **+52.4% 的最大突破**合入）。只设上界差点收下假突破。
- **A/B 必须对当前 best 做**，不要拿历史数字当参照——实测因此把一个回归误判成大幅提升。
- ⚠️ **profiler 的 `operators` 聚合 ≠ 几何平均**：前者是跨 case 的总耗时（大 case 占大头），后者每 case 一票。实测出现过"attention kernel 聚合耗时降 7%、几何平均反而下降"——改动让大 case 变快、小 case 变慢。**几何平均下"别让小 case 变慢"与"让大 case 变快"同等重要。**
- **"无效"的改动未必中性**：某次手工外提常量张量测下来是噪声（判定无效），但它把该张量提到 kernel 顶层、**全程占 32KB UB**；等 UB 成为瓶颈时这笔账才浮现。
- 比值型指标不能自证有效：分子分母同步变慢时 speedup 反而"看起来变好"。
- ⚠️ **`torch_npu.profiler` 会在部分 device 上提取不到数据**：表现是 `msprof 旧格式回退解析为空` / `SQLite Error: no such table: TASK`，**大面积假失败**（实测某次 57/61 的 benchmark 全因此，代码本身无问题）。**必须 `torch.npu.set_device(<目标 NPU>)` 后再跑 benchmark**（benchmark.py 用默认 `torch.device("npu")`=device 0，不改就会中招）。判别：framework 与 impl 两侧 profiler 都解析为空、且 case 全部失败 → 先查 device/profiler，**不要改代码**。可用单 case smoke（`set_device` 后 measure 一次）先验证 profiler 可用再全量跑。

### §7.2 ⚠️ benchmark 时延口径（写报告时必须注明）

`benchmark.py` 按 **kernel Name 分组**，同名 kernel 一次 forward 内发射 L 次**只被计成 1 次**（低估 L 倍）。

- ✗ **禁止**为迎合口径把多次发射合并成同一个 Name（用 `MODE: tl.constexpr` 分支合并 kernel）——**不改变任何真实性能，纯粹利用口径缺陷**。
- ✗ 也**不要**因为口径而放弃真实优化（如三个投影融合成一次 GEMM 是真实收益，但会让口径分数变差）。
- **做天花板估算前先确认分数口径**：实测某算子 benchmark 口径 6.24 而真实设备时间 **9.24x**，不确认会把接近目标的实现误判为差 40%。

### §7.3 ⚠️ 历史结论会随编译器 / CANN 版本失效——成功与失败都会

| 改动 | 旧编译器 | 新编译器 |
|---|---|---|
| KV 循环整块/尾块拆分 | **+4.5%** | **−11.7%**（撤销它反而更快） |
| `BLOCK_SQ=256` | **0/50 全线编译失败** | **编过且 +3.2%** |
| 原始基线代码（同一份） | 0.1052 | **0.8390**（快 8 倍） |

> **规则：换编译器 / CANN 版本后，全部结论作废，必须重新标定基线并重扫参数。不要查历史失败清单。**
> 这条与 L1.4（参考实现算术路径一变、精度结论全废）是同一类风险的两个来源。

### §7.4 编译器已经做了的事，不要手工做

手工循环不变量外提、手工 buffer 复用、手工删除常量张量——**三次实证全部 no-op**（其中一次还有 UB 副作用）。
**UB 手算模型（把张量尺寸相加）预测不了真实边界，边界只能实测撞出来。**

⚠️ **UB 超限在 MLIR 层的表现要与精度失败区分**：

```
'hivm.hir.vcast' op Unsupported op for finding the root alloc
'hivm.hir.vexp'  op Unsupported op
[ERROR] Failed to run BiShengIR pipeline
```

**这是编译失败，不是精度失败**，处置路径完全不同。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Cube 空转、benchmark 超时无数据 | 用逐维外积代替矩阵乘 | L1.1 |
| 非 fp32 case 全部编译失败 | `tl.dot` 两操作数 dtype 不一致 | L1.2 |
| softmax 分母偏大、结果整体偏小 | KV padding 列未置 `NEG` 就进了 `sum` | F1 |
| 输出大面积 NaN | 掩码用了 `-inf`，整行被掩时 `exp(-inf-(-inf))` | F2 |
| `head_dim` 大时 `ub overflow` | 某 block 维由运行期 shape 决定且无上界 | F4 |
| 大批用例连续失败但逐个复跑只错几个 | `ub overflow` 触发 vector core exception 污染设备 | 定位**第一个**失败用例单独复跑，不按失败总数估计缺陷数 |
| `BLOCK_D` 白白浪费 30~60% | 用了 `next_pow2(D)` | F3 |
| 放大 BLOCK_KV 只有 +1.2%，误判方向无效 | `kv_lo` 仍在对齐，把收益吃掉了 | L1.8，两者成对测 |
| 诊断子集 +10%、正式测量 −3.7% | 扫参子集只有大 case，中等 case 块数塌陷 | F6 + §4.5 并行度保护 |
| 小 shape 性能上不去，想拆 kernel 模块化 | MIX kernel 每次发射 ~4.5us 固定开销 | §5.3，**绝不要拆** |
| **精度全过但 speedup 只有 1.0~1.2 上不去** | **BLOCK 被写死成 32×64 这类小值**，面积只有预算上限的 1/4 ⇒ KV 迭代数与跨核握手多 4 倍。这是本类算子最常见的性能天花板成因 | **L1.13**：按面积预算 `BQ*BKV<=16384` 取到上限，禁止硬编码 |
| 大 case 的 attention kernel 时延居高不下 | `tl.maximum` 默认展开成 7 条向量指令，每次迭代都执行 | **L1.14**：多块路径开 `propagate_nan=ALL`，单块路径必须关 |
| `aic_scalar_ratio` 73%，改了五次标量代码全无效 | 同步等待被计入标量管线 | §5.4，先 dump IR |
| 权重在 NPU 上重采样 / 投影算成 `x @ W` | CPU 与 NPU 是两条独立 RNG 流；`nn.Linear.weight` 是 `[out,in]` | L1.11 |
| **投影 GEMM 很慢、`MTE2>60%`，attention 侧却没问题** | **权重仍是 `[out,in]` 原始布局，b tile 最内轴 stride = `in_features` ⇒ 离散 gather**。坏味道：kernel 注释写着"w 形状 `[N,K]`（`nn.Linear.weight` 布局）"、host 侧传 `w.stride(0), w.stride(1)` | **L1.12**（实测 182x / 端到端 5.94x，首次生成就要做对） |
| 转置了权重但收益不及预期 | 布局改了没重扫分块：转置把算子从"B 转置"变成"都不转置"，推荐分块随之改变；`BLOCK_K=32` 的 fp16 只有 64B，是 512B 对齐线的 1/8 | L1.12 配套项（再 +13.2%） |
| **大 shape 报 `RuntimeError: grid should be less than 65536`** | **辅助 kernel（RoPE/位置编码/bias 预处理）的 grid 按 `B*N*S` 直接展开**。判别：失败集合严格按 `B*N*S ≥ 65536` 划分，且是 RuntimeError 而非精度不匹配 | **L1.9**：算子内**每个** kernel 都用 `grid=min(核数,tasks)` + 核内步长循环；不要用 `TRITON_ALL_BLOCKS_PARALLEL=1` 绕过 |
| **带位置编码的 case 全部 `CompilationError`**，其余全过 | `arange's arguments must be of type tl.constexpr` —— 交错重排里用运行期值构造 `tl.arange` 或动态 `tl.reshape`。判别：失败集合严格按 `use_rope`/`use_alibi` 这类布尔属性划分 | **L1.10 交错重排配方**：constexpr `tl.arange` + `tl.where` 就地选择，全程不改形状 |
| **小 shape 尚可、大 shape 崩塌（<0.01x）**，每次迭代成本达数百 us | **输入变换（RoPE/位置编码）被放进 KV 循环内逐块重算**，且其中间量占 UB 把 BLOCK 逼小——同时挨两刀 | **L1.15**：用独立 Triton kernel 外提，再重扫 BLOCK（实测 0.005x → 1.386x） |
| 看到 `BLOCK = min(..., 64)  # avoid UB overflow` 这类预防性封顶 | 封顶通常是**症状**不是病因：某段本可外提的计算占了 UB | **L1.15**：先移出占 UB 的计算，再重扫 BLOCK。**直接删封顶会编译失败**（已实测） |
| 改了参考实现后老结论突然失效 | 参考的算术路径变了 | L1.4，先钉死契约再优化 |
| **pse case 输出 77~96% 错 / 编译产物错乱** | pse 按实际维数分档：2D `view(1,1,1,-1)` 广播、4D `S=1` 广播、4D dense 三种语义不同；`[B,N,1,SK]` 一维 load + 广播加在 native dot 下编译产物错乱（smax 差 135） | **统一 2D load**（`offs_q*stride_s + offs_kv*stride_k`，`S=1` 时 stride=0 语义相同）；先探 golden 的 pse 广播行为（§6.1） |
| **`npu_*` op 输出 shape 不匹配、查不到原因** | `npu_fusion_attention` 返回 7 元组，`softmax_max/sum` 是 `[B,N,Sq,8]`（8 列同值）、`reserved` 空 | **§6.2 输出契约表**：按 7 元组契约分配输出，再写 kernel |

---

## §9 与 `attention.md` 的差异声明（FA 路径上的实测推翻）

保留原条目以供对照，但**在本类算子上以本文件为准**：

| `attention.md` 条目 | 本文件 | 证据 |
|---|---|---|
| L1.1「FP32 路径**禁用** `tl.dot`，会产生 NaN 或错误结果」 | **L1.1 条件式**：禁止的是"操作数升 fp32 再 dot"；原生 dtype + fp32 累加器**必须用** | fp32 case 精度 50/50 全过，且是最大的一笔（不可测量 → 2.3554） |
| L1.3「`BLOCK_D = next_power_of_2(head_dim)`」 | **F3 用 `ceil16(D)`** | D=96/160/192 上 `next_pow2` 浪费 30~60% tile |
| L1.6「归约与累加**全程** fp32」 | **L1.4 契约驱动**：先钉死参考算术路径，两套契约并列 | 参考改原生 dtype 后，旧契约交付版只过 20/50 |
| A2 路径 (a)「把 `head_dim` 拆出去逐维外积」 | **禁止**（与 L1.1 冲突） | 逐维外积正是 Cube 空转的成因 |
| A1（padding 掩码）、A2（UB 上界）、§2.0（host 语义）、L1.2 / L1.5 / L1.7 | **已承接**为 F1 / F4 / L1.11 / L1.2 / L1.3 / L1.10 | 无冲突，逐条保留 |

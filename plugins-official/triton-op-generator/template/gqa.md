---
name: gqa
description: GQA 家族（二·1 head 共享：GQA/MQA、router 选路多分支、含投影 GQA；以及与三·2 块跳过、二·3 页表寻址叠加的形态）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧、Phase 4 优化点清单与证伪方向全表
metadata:
  type: reference
---

# 二·压缩类 · head 共享（GQA / MQA）算子优化经验

本文件按 `level4写法分类体系.md` 的写法口径，负责 **二 压缩类 · 细分 1「head 共享」**：
取数层的改动只有 `kv_head = q_head // group` 这一处。该细分下的算子是
**21 GroupedQueryAttention · 22 MQA · 33 AdaptiveAttention**（33 是门控选路变体，
按样本 argmax 选一路权重，主链仍是 head 共享）。含投影的 `gqa-proj` 形态也在本文件。

> **与 `block_sparse_attention.md` 的分工**：那份卡片负责 **三 稀疏类 · 细分 2「块跳过/top-k」**
> （`sp-blockmask` / `sp-paged-topk` / `sp-pooled-topk` / `sp-bwd`），即**选择表怎么建、
> 怎么按压缩索引跳块**；`80/81/83/84 GqaSparse 系`、`93 MsaSparse`、`99/100 BlockSparse`、
> `101 SLA` 都归它。
>
> **叠加形态两份都读**：GqaSparse 系是「三·2 块跳过 ＋ 二·3 页表寻址 ＋ head 共享」三者叠加，
> **选择层与跳块看那边，head 分组的任务划分、页内数据布局、tile/UB 预算看这边**。
> 本文件中标注「分页」「页内连续」「多页合并」的条目（L1.1 的 decode 分支、§3.1、§3.4、
> §5.2 的多页合并证伪）就是为这种叠加形态准备的，来自 `80/81` 的实测。
>
> **与 `attention-paged.md` 的分工**：那份卡片负责 **二·3 页表寻址·不跳块** 的纯形态
> （整段 KV 都要算的 paged decode/prefill，op37/38 实证：block_ptr 红线 / helper op 顺序
> 触发流水线 / aux_mask 表 load / split-q persistent）。本文件的分页条目服务于行 1 交叉的
> 块跳过叠加形态，并作为其页内布局（§3.4）与 tile·UB 预算（G3/§5.2）的补充参考。

- **§1 通用经验 G1-G7**（跨形态，首次生成必须遵守）
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**，比 §5.1 更值钱）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与 flash_attention.md 的分工**

> **证据基础**：四个 GQA 类算子各 50 case 的完整优化轨迹，同一台 910B2C
> （24 aicore / 48 vectorcore）、同一套工具链
> （triton 3.2.0 / triton_ascend 3.2.2 / CANN 9.1.0 / bishengir-compile 1.2.0）：
>
> | 算子 | 子类标签 | 形态 | **最终 speedup**（新口径） | 旧口径 |
> |---|---|---|---|---|
> | `21_GroupedQueryAttention` | `gqa-proj` | 4 个投影 GEMM + 稠密 GQA | **2.5119** | 0.0016 → 2.2839 |
> | `33_AdaptiveAttention` | `gqa-router` | router MLP 选 1 路，N∈{2,3} 路完整 GQA | **4.0101** | 精度 0/50 → 50/50，2.9033 |
> | `80_GqaSparseFwd` | `gqa-sparse-fwd` | 块稀疏 GQA 前向，分页 KV | **59.5558** | 2.2904 |
> | `81_GqaSparseDecode` | `gqa-sparse-decode` | 块稀疏 GQA decode，topk 页 + block_table 间接寻址 | **41.9099** | 0.2233 → 1.3791 |
>
> 四者精度均 **50/50**。两栏差异见 §7.2——**口径变了，不是代码变了**；
> 本文件正文里出现的 speedup 除本表外**一律是旧口径**，只用于方案之间的横向比较
> （impl 绝对时间不受口径影响），**不要与新口径的绝对分数混用**。
>
> ⚠️ **核心优化哲学**：与 FA 类一致——瓶颈是 **Cube↔Vector 跨核同步**，不是算术量、不是访存、不是 dtype。
> 但本类算子多了一条 FA 类没有的硬约束：**KV 按页/块间接寻址，页与页之间不连续**，
> 于是「放大 tile 摊薄同步」这条 FA 的万金油**在本类算子上会直接撞 UB 墙**（§5.2 有完整实测）。
> 本类算子的目标函数是 **「循环体内 `tl.dot` 的个数 × KV 页迭代总数」**，
> 而**不是** tile 的大小——这一条与 FA 类的直觉相反，是本文件最重要的一句话。

---

## §0 适用范围与算子分类

| 子类标签 | 判别特征 | 优化重心 |
|---|---|---|
| `gqa-proj` | `H_Q > H_KV` 的稠密 GQA，**自带 4 个 `nn.Linear` 投影** | **最大的一笔在投影 GEMM 的 tile 分档**（实测 +57%），不要一上来就攻 attention |
| `gqa-router` | 前置一个小 MLP，`argmax` 选 1 路，多路完整 attention，最后 one-hot 加权求和 | **只算被选中的那一路**（实测 nopts 倍收益）；难点全在精度与编译，不在性能 |
| `gqa-sparse-fwd` | KV 按块稀疏跳过，分页 cache | 页内连续 + 跳过判据外提 |
| `gqa-sparse-decode` | 每个 query token 由 `topk_idx` 选 K 个页，`block_table` 二级间接寻址；`S_Q` 退化成 1 | **任务单元必须按 kv_head 分组**（实测 6.18x）；之后成本按页线性，杠杆极少 |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件，**不要**落到 `flash_attention.md`：

1. KV 通过 `block_table` / `page_table` / `topk_idx` 等**间接索引**取，页与页在显存中不连续；
2. attention 主链被一个**路由/门控**选择性执行（`argmax` + `one_hot` 或等价形式）；
3. KV 维度上存在**块级跳过**（block-sparse），而不是一段连续区间的 causal / window 收缩。

只有稠密 GQA（KV 是连续区间，靠 causal/window 收缩）→ 用 `flash_attention.md` 的 `fa-gqa`。
两者都命中（如 `gqa-proj` 既有投影又是稠密）→ **两个文件都读**，本文件的 §2/§5 优先。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答，答案决定后续哪些章节适用）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | KV 是连续区间还是按页间接寻址？ | 间接 ⇒ §2 L1.2（任务单元）+ §5.2 的 UB 墙全部适用；连续 ⇒ 回 `flash_attention.md` |
| **Q2** | 有没有独立的投影 GEMM？ | 有 ⇒ §4.1（tile 按 dtype 分档）是最大的一笔；没有则完全不适用 |
| **Q3** | 有没有路由/门控选路？ | 有 ⇒ §4.4（分支门控）理论收益 = 分支数；但**必须先做 §6.4 的 argmax 边距体检** |
| **Q4** | 参考实现的 attention 在什么精度上算（有没有 `.float()`）？ | 决定 `p` 能否降精度、以及要不要二段拆分（§6.2） |
| **Q5** | 评测口径？ | 几何平均 ⇒ **每个 case 等权**。本类算子小 case 极多，把一个 0.5x 拉到 1.0x 与把 8x 拉到 16x 贡献完全相同 |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### G1 ★ 环境闸门必须前置，否则整片结论作废

本类算子踩过两次**整片结论作废**，都不是代码问题：

1. **非交互 shell 读不到 `~/.bashrc`**。`~/.bashrc` 前几行常有 `[ -z "$PS1" ] && return` 守卫，
   于是后面的 `source .../cann/set_env.sh` 在脚本里**永远不执行**，静默跑在容器自带的旧 CANN 上。
   实测：同一份代码在 CANN 8.5.1 + bishengir 0.1.0 上 **0.9438**，在 CANN 9.1.0 + bishengir 1.2.0 上 **2.2904**。
   两套都能跑通、都不报错。
   ⇒ **每轮测量前显式 source，并把 `triton` / `triton_ascend` / CANN / bishengir 四个版本写进产物**，
   版本不符直接非零退出。

2. **卡被自家 kernel 打坏**。kernel 触发 `aicore timeout (507014)` 之后，那块 NPU 进入**跨进程持久**
   损坏状态（重启进程无效），此后卡上一切精度结果都是噪声。
   实测判据只需一行**与被测算子完全无关**的算子对拍 CPU：
   ```python
   c = (a.npu() @ b.npu())    # a, b = randn(1024,1024).half()
   # 坏卡: absmax=5.67e14（另一次 2.67e36 / nan=63）; 好卡: absmax≈162.7
   ```
   代价：某算子的 v2(0/50)、v3(4/50) 全产在坏卡上，其中 20 个「NaN 位置不匹配」+ 11 个
   AccuracyError 是**纯噪声**，为追这批假失败白烧了两轮诊断；换健康卡后同一思路直接 50/50。
   ⇒ **精度异常时的第一件事不是查代码，是查卡。**

   ⚠️ **而且要反复查，不能只在开工时查一次。** 实测：同一台机器上 NPU 2 在 04:36 自检还是
   `absmax=162.8`（正常），两小时后再查已变成 `absmax=nan` —— **中途被别的会话打坏**。
   共享机器上「开工时全绿」不等于「这一轮的结果可信」。
   ⇒ **每次出现新的错误码（507014 / 507015 / 507035 …）或成片精度异常时都重跑一次自检**，
   代价只有几秒，收益是不会白追一整轮不存在的代码 bug。

### G2 tile 预算随编译器版本整片变动，结构性改动后必须重扫

同一份 kernel：bishengir 0.1.0 上预算约 `2^18` 且报误导性的
`hivm.hir.vsel op Unsupported op for finding the root alloc`；
1.2.0 上预算约 `2^20` 且报老实的 `UB overflow`。
⇒ **不要查历史 tile 清单**，也不要把「某一维不能是 128」这种伪判据写进代码。真判据是 UB 字节数。

### G3 UB 是本类算子的第一约束，动手前先记账

910B2C 单核 UB **192KB**。所有「放大 tile 摊薄同步」的想法都要先算这笔账，
且**必须把 `tl.trans(x)` 的物化算进去**——它会额外占一份与 `x` 等大的空间。

一个实测的满预算例子（`gqa-sparse-decode`，BS=128 / D=128 / bf16）：

```
k[128,128]bf16 32KB + tl.trans(k)[128,128] 32KB + v[128,128]bf16 32KB
+ v.to(f32)[128,128] 64KB + q 4KB + scores 8KB + p 8KB + acc 8KB  =  188KB / 192KB
```

**只剩 4KB**。这解释了为什么该算子上一切扩张（多页合并、循环携带量、转置形态）
全部报 MLIR `PlanMemory Failed`——不是写法不对，是预算见底。

> ★ **先读报错里的精确数字，再决定改多大。** `ub overflow` 会给出
> `requires X bits while Y bits available`，**两者之差就是你要省出来的量**：
>
> ```
> ub overflow, requires 1573376 bits while 1572864 bits available   # 实测一例
> ```
> 这里只超 **512 bit = 64 字节（0.03%）** —— 去掉一个中间变量、或把某个 tile 维
> 从 128 调到 120、或把一处 `tl.where` 的临时量复用掉，就够了。
> **不要一看到 UB overflow 就把 tile 砍半**：砍半会让性能白白掉一半，
> 而这类"差一点点"的溢出在实测里相当常见。
>
> 对照：真正需要结构性改动的是**差一个数量级**的情形
> （如 `requires 1859584 while 1572864`，超 18%，那是一整块 W1 tile 放不下）。

### G4 增益不叠加，先算账再动手

每个方向动手前先估：`收益 = 省下的每次迭代成本 × 迭代总数 / 当前总时延`。
本类算子上有两个反复出现的账：

- **访存有没有冗余**：`gqa-sparse-decode` 上曾以为 topk 页在 token 之间可复用，
  实测统计「合计页读 18398 / 去重后 17310 = **冗余仅 1.06x**」⇒ 这条路直接否掉，没写一行代码。
- **理论天花板**：把 DMA 成本全部归零后的时延，若仍达不到目标，说明方向根本不在访存。

### G5 扫参子集必须含大/中/小三档

本类算子 case 分布普遍是「小 shape 极多 + 少数大 shape 吃掉 80% 耗时」，
而评分是**几何平均（等权）**。只在大 case 上调参会得到与最终分数相反的结论。

### G6 ★★★ KV 维 padding 必须掩在 **scores** 上，**清零操作数解决不了、反而正是错因**

**实测：一个算子连续两轮 0/50 栽在这里。** 这条最反直觉，务必读完。

```python
# ❌ 看起来很对，其实是错的
kv_mask = (offs_kv[:, None] < S) & (offs_d[None, :] < D)
k_tile  = tl.load(k_ptrs, mask=kv_mask, other=0.0)     # padding 行填 0
scores  = tl.dot(q_tile, tl.trans(k_tile))             # ⇒ padding 列 score == 0
m_new   = tl.maximum(m, tl.max(scores, axis=1))
p       = tl.exp(scores - m_new[:, None])              # ⇒ exp(0 - m) 是**非零**的
l      += tl.sum(p, axis=1)                            # ⇒ 分母被灌进虚假质量
```

`other=0.0` 把 K 的 padding 行变成 0 向量，于是那些列的 score 恰好是 **0**——
而 0 在 softmax 里**不是**"无贡献"，`exp(0 - m)` 往往还是所有列里最大的那批。
后果是 **`l` 被系统性放大、输出被整体缩小**。

**症状指纹**：impl 与 framework **同号但整体偏小**，比值在 0.3~0.7 之间且随 `S % BLOCK_KV`
变化（`S` 整除 `BLOCK_KV` 的 case 反而正常）。实测一例 `framework=1.495 / impl=0.6006`。
⚠️ 这个误差量级会被误判成"权重复刻错"，见 L1.9 的分诊说明。

```python
# ✅ 掩在 scores 上，用**有限极小值**（不是 -inf，见下）
kv_valid = offs_kv < S
scores = tl.dot(q_tile, tl.trans(k_tile)) * inv_sqrt
scores = tl.where(kv_valid[None, :], scores, -3.0e38)   # 有限极小值
m_new  = tl.maximum(m, tl.max(scores, axis=1))
p      = tl.exp(scores - m_new[:, None])                # padding 列 -> 0
```

三条配套要求：

1. **操作数仍要清零**（`tl.where(mask, val, 0.0)`），这是为了让 `tl.dot` 的归约不吃到脏数据——
   但它**只保证 dot 的正确性，不构成 softmax 的掩码**。两件事都要做，不能互相替代。
2. **用有限极小值而不是 `-inf`**：整块 KV 都无效时 `max` 得到 `-inf`，`exp(-inf - (-inf))` = NaN。
   `-3.0e38` 在 fp32 内、`exp` 后稳定为 0。
   （若确实用了 `-inf`，则必须保证每个 q 行至少有一个有效列。）
3. **`m` 的初值同理**用 `-1e30` 之类的有限值，不用 `-inf`。
4. **★ 出口的除法必须 guard**。上面三条都做对了，仍有一个漏网之鼠：
   若某个 q 行的 KV 列**全部**被掩掉（整块越界、稀疏页全越界、q 行本身是 tile padding），
   则 `p` 全 0 ⇒ `l == 0` ⇒ `acc / l` 是 **0/0 = NaN**。
   这与用 `-inf` 还是有限极小值无关，**两种写法都会栽**。
   ```python
   out = acc / tl.where(l > 0, l, 1.0)      # l==0 的行输出 0，与 golden 对齐
   ```
   症状是 **Implementation 有大量 NaN 而 Framework 一个都没有**
   （verify 报 `NaN 位置不匹配: Framework=0/N, Implementation=M/N`）。
   ⚠️ 出现这个症状时先查这里，**不要**去怀疑掩码值选得不对。

> 这条与 §3.1 骨架里的 `pos_f < kv_len_f` 是同一件事：**KV 轴上任何"不该参与 softmax 的位置"，
> 都必须在 scores 上掩掉**，无论它来自 tile padding、因果上界、还是稀疏页越界。

### G7 forward 里的宿主侧代码受 AST 校验器约束

- `ast.For` 在 forward 里是**无条件违规**（不是只查循环体）⇒ 多分支要**展开成 `if`**；
- `self.xxx(...)` 一律被判「疑似 nn.Module 前向调用」⇒ 辅助函数写成**模块级函数**；
- `torch.cat` **不在**白名单；拼权重要用 `torch.empty` + `.copy_()`（白名单内）；
- 允许的 torch 构造：`empty/zeros/ones/full/tensor/arange/linspace/as_tensor`；
  允许的张量方法：`contiguous/to/view/reshape/permute/transpose/t/copy_/fill_` 等。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ 任务单元：**decode 按 kv_head 分组；prefill 必须按 query 分块**

> ⚠️ **先看适用范围，照搬会亏 4.4 倍。** 本条只对 **`gqa-sparse-decode`**（每个任务天然只有
> 1 个 query 位置，`S_Q == 1`）成立。**把它套到 prefill / 含投影的稠密 GQA 上是重大负优化**：
> 实测某次生成把任务单元写成 `total_tasks = B * KH * S`（每个 query 位置一个 task），
> 精度 50/50 但 **speedup 只有 0.5681**（同算子手工版 2.5119，差 4.4 倍）——
> 因为每个 query 位置都把整条 K/V 重读一遍，**KV 访存被放大 S 倍**（该测试集 S 最大 897）。
>
> | 子类 | 任务单元 | q tile |
> |---|---|---|
> | `gqa-sparse-decode` | `(token, kv_head)` ← 本条 | `[BLOCK_G, D]`，BLOCK_G = 该 kv_head 的 G 个 q head |
> | `gqa-proj` / `gqa-router` / `gqa-sparse-fwd`（prefill） | **`(b, q_head, q_block)`** | `[BLOCK_Q, D]`，**BLOCK_Q 必须按 UB 预算取到上限**（§3.2 骨架） |
>
> **⛔ prefill 下这一行是硬性禁止的**（实测两轮生成都栽在它上面，speedup 0.5681 / 0.872）：
> ```python
> total_tasks = B * KH * S          # ❌ 每个 query 位置一个 task = decode 划分
> offs_g = tl.arange(0, BLOCK_G)    # ❌ 随之而来的 q tile 是 [BLOCK_G, D]
> ```
> ```python
> total_tasks = B * H * tl.cdiv(S, BLOCK_Q)     # ✅ prefill: 按 q_head × q_block 分
> offs_s = q0 + tl.arange(0, BLOCK_Q)           # ✅ q tile 是 [BLOCK_Q, D]，BLOCK_Q 开到 UB 上限
> ```
> **`S_Q > 1` 时看到 `tl.arange(0, BLOCK_G)` 出现在 q tile 上，就是错的**——
> 必须照 §3.2 骨架重写，不是微调。
>
> **判据**：`S_Q > 1` ⇒ 走 prefill 那一行。prefill 下同一个 head 的 `BLOCK_Q` 行**天然共享 K/V**，
> kv_head 分组解决的那个"重复读"问题**根本不存在**；此时真正的杠杆是
> §4.1（投影 tile 分档）与把 `BLOCK_Q` 开大。

**以下仅适用于 decode。这是该子类最大的单笔收益（实测 0.2233 → 1.3791，6.18x）。**

坏味道：kernel 里出现 `kh = h // G`（或 `h * KH // H`），而循环变量是 **q head**：

```python
# ❌ 同一个 kv_head 组内的 G 个 q head 各自把同一份 [BS, 2D] KV 页重新读一遍
for head_idx in range(pid, total_q * H, num_cores):
    kh = head_idx // g_size
    ...  # 读 KV 页
```

```python
# ✅ 一个 program 处理整组 G 个 q head，q tile 由 [1, D] 变成 [BLOCK_G, D]
pid  -> (t, kh)
q_tile = load Q[t, kh*G : (kh+1)*G, :]      # [BLOCK_G, D]
```

三重收益，缺一不可：

1. KV 访存降到 **1/G**（实测 G ∈ {4,6,8,16}，主导耗时的 case 全是 G=16 ⇒ 访存放大 16 倍）；
2. QK 的 `tl.dot` 的 M 从 1 变成 `BLOCK_G`，正好是 Cube 的 fractal 粒度；
3. PV 从 `tl.sum(p[:,None] * v, axis=0)` 的**向量广播归约**变成一次 `tl.dot`，从 Vector 挪到空闲的 Cube。

> **可迁移判据**：任何 GQA/MQA 算子，只要 kernel 里出现 `kh = h // G` 且循环变量是 q head，
> 就一定在按 G 倍重复读 KV。decode 场景最容易漏掉的一处放大。

### L1.2 ★★★ 禁止「运行期标量除 + 向量下标 gather」组合——会让 aicore 挂死

```python
# ❌ 实测在 bishengir 1.2.0 上直接 aicore timeout 507014，并把整块卡打成持久损坏
b_idx = offs_m // S                      # S 是运行期标量
sel   = tl.load(sel_ptr + b_idx, ...)    # 按向量下标离散 gather
```

```python
# ✅ 把 b 提升为 tile 坐标之一，sel 退化成一次标量 load
pid -> (b, s_tile, n_tile)
if tl.load(sel_ptr + b) == BRANCH:  ...
```

同源坏味道还有 `a % b`（运行期标量取模，会标量降级）——用 `a - (a // b) * b`，
或当除数是 2 的幂时用位与。

### L1.3 ★ `tl.dot` 的个数是成本单位，**不是 tile 的大小**

这是本类算子与 FA 类最反直觉的差异，**三轮独立实测互相印证**：

| 轮次 | 改动 | 循环内 dot 数 | 结果 |
|---|---|---|---|
| 基准 | `dot(p_f32, v_f32)` | 2 | 1.3791 |
| A | PV 改 `dot(p_hi,v)+dot(p_lo,v)` 双原生 | **3** | **0.7089（时间正好翻倍）** |
| B | PV 错开一轮，打断 Cube→Vector→Cube 依赖 | 2 | 1.2647，`serial_ratio` 1.738 → 1.707 **几乎没动** |
| C | PV 由 fp32 dot 换 fp16 原生 dot | 2 | 1.3088（**持平**） |

**在 `gqa-sparse-decode` 上的结论**：每个循环内的 `tl.dot` 自带一份约 2 us 的固定 CV 握手，
与 tile 大小、与数据依赖关系无关——B 轮证明它是**结构性插入**的，不是「等前一个向量算子的结果」。

> ⚠️ **但这条不能跨子类外推。上面四轮全部在同一个算子（`81`）上做的，不是四个独立证据。**
> 在 `gqa-sparse-fwd`（`80`）上**实测结论相反**（同一时段、同一卡、同一负载的 A/B）：
>
> | 写法 | 循环内 dot 数 | impl | speedup |
> |---|---|---|---|
> | `dot(p_hi, v) + dot(p_lo, v)` 双**原生 dtype** dot | **3** | 0.3086 ms | **55.33** |
> | `dot(p_f32, v.to(f32))` 单 **fp32** dot | **2** | 0.5491 ms | 27.97 |
>
> **dot 数少的那个慢 1.78 倍。** 原因是 fp32 dot 走的不是 Cube 的原生通路，
> 且每页要多一份 `[BS, BLOCK_D]` 的 f32 转换缓冲（D=128/bf16 即 64KB）——
> 这两项在 `80` 上超过了省下的那一次握手；在 `81` 上则被 CV 握手掩盖。

⇒ **正确的用法是先测再定，不要照搬**：

1. 先做**一次** A/B（同一时段、同一卡）确认本算子属于哪一侧：
   握手主导 ⇒ 压 dot 数；MAC/转换主导 ⇒ 保原生 dtype、宁可多一个 dot。
2. 判别信号：`aic_mac_ratio` 极低（0.02~0.05）且 `serial_ratio` 接近 2 ⇒ 握手主导；
   `aic_mac_ratio` 明显更高 ⇒ 算力/转换主导。
3. **`dot(p_f32, v.to(f32))` 这种「把原生 dtype 操作数升 fp32」的写法要特别小心**：
   它同时付出非原生 Cube 通路与一份大转换缓冲两笔成本。
   若精度允许，优先 `p.to(f16) + v.to(f16)` 的**原生单 dot**（§4.3）；
   精度不够再退回 `p_hi/p_lo` 双原生 dot。

### L1.4 成本模型必须先反推出来，再谈优化

本类算子的时延可以用两点联立解出，误差 <10%：

```
kernel_us ≈ (每页代价 a × 总页数 + 每 program 启动 b × program 数) / 核数
```

实测（`gqa-sparse-decode`，取两个大 case 联立）：**a = 5.81 us/页，b = 0.71 us/program**。
其中 a 的构成：两个 dot 的固定 CV 握手 ≈ 4.1 us + DMA 1.69 us + MAC ≈ 0.05 us。

⇒ **启动开销可忽略，成本全部按页线性**；`aic_mac_ratio` 只有 0.02~0.05，Cube 基本没在算。
⇒ 天花板：即使 DMA 归零也只有 4.1 us/页，对应 speedup 1.95x。**这个账要在动手前算。**

### L1.5 ★ `tl.dot` 契约：原生 dtype 操作数 + fp32 累加

- q/k 从低精度内存加载 ⇒ **保持原生 dtype**，升 fp32 不增加任何有效位；
- ★★★ **两操作数必须显式 cast 到同一 dtype** —— 实测 34 个 case 栽在这里：

  ```python
  # ❌ q 经过 tl.where/补零后被提升成 fp32，k 仍是 fp16 ⇒ 编译期直接挂
  q_tile = tl.where(qd_mask, q_tile, 0.0)          # 0.0 是 python float => 结果变 fp32
  scores = tl.dot(q_tile, tl.trans(k_tile))
  #   AssertionError('Both operands must be same dtype. Got fp32 and fp16')
  ```
  ```python
  # ✅ 补零后显式 cast 回原生 dtype（或两侧一起 cast）
  q_tile = tl.where(qd_mask, q_tile, 0.0).to(k_tile.dtype)
  scores = tl.dot(q_tile, tl.trans(k_tile))
  ```
  **最常见的诱因是 `tl.where(mask, x, 0.0)` / `tl.full(..., 0.0)` 里的 Python 浮点字面量
  把整块 tile 提升成 fp32**，而另一侧还是 fp16/bf16。补零与 cast 要一起做。

- ⚠️ **fp32 `tl.dot` 的适用条件要收窄**：满足以下三条时被 bishengir 1.2.0 **误编译**——
  ① 操作数是 fp32；② 输出宽度 `N == 64`；③ 位于**运行期 trip-count** 的 `scf.for` 内。
  症状是输出**恰好一半、且是偶数列** NaN，奇数列是垃圾正数。
  打破任意一条即规避（`N` 抬到 128 / 改原生 dtype / trip 改常量）。
  > **可迁移方法**：低精度输出的 NaN/垃圾，先看**列的分布**。「恰好一半且是偶数列」
  > 是 tile 交织排布被写坏的指纹，直接指向 dot 的输出宽度，而不是掩码、边界或 `-inf`。

### L1.6 ★ 投影权重禁止以 `nn.Linear` 原始 `[out, in]` 布局喂给 GEMM kernel

> ⛔ **但不要过度套用——只有喂给 `tl.dot` 的权重才需要预转置。**
> 实测一例：`gqa-router` 的六个权重里，**`w2`（router 第二层）是唯一不能转置的**，
> 因为 `router_argmax` 用的是 `tl.sum(w2 * hv[None, :], axis=1)` 的**向量点积**而非 `tl.dot`，
> kernel 按 `[NOPT, DQ]` 原始布局索引。
>
> ```python
> # ❌ 看到其他五个权重都 .t() 了就顺手也转置 —— 布局与 kernel 索引不匹配
> w2.t().contiguous().to(device, dtype)          # 变成 [DQ, NOPT]
> w2 = tl.load(w2_ptr + offs_o[:, None] * DQ + offs_q[None, :])   # 仍按 [NOPT, DQ] 索引
> ```
> **后果**：读到错行 ⇒ `argmax` 选错分支 ⇒ **输出是另一路的结果**，
> 相对误差恰好 **~1.0 量级**（实测 1.08）。
> ⚠️ 这个症状与 L1.9 的「权重复刻错」**完全一样**，但根因不同：
> 权重值是对的，**布局与索引不匹配**。分诊时两者都要查。

**以下是需要预转置的那五个（喂给 `tl.dot` 的）**：

`nn.Linear.weight` 是 `[out, in]`，`y = x @ Wᵀ`。直接喂会让 b tile 的最内轴 stride = `in_features`，
变成离散 gather。**host 侧预转置成 `[in, out]`**（纯数据搬移，数值逐位不变），
kernel 内 b tile 最内轴 stride == 1，且省掉每次迭代的 `tl.trans`。

拼接用 `torch.empty` + `.copy_()`（`torch.cat` 不在 AST 白名单）：

```python
qkv_t = torch.empty((d_model, d_model + 2 * d_kv), dtype=torch.float32)
qkv_t[:, :d_model].copy_(q_w.t())
qkv_t[:, d_model:d_model + d_kv].copy_(k_w.t())
qkv_t[:, d_model + d_kv:].copy_(v_w.t())
```

### L1.7 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

```python
# host
grid = (min(NUM_AICORE, total_tasks),)        # NUM_AICORE 实测 24，不要硬编码 14
kernel[grid](..., NUM_CORES=grid[0])          # 必须把同一个数传进去
# kernel
for task in range(pid, total_tasks, NUM_CORES):
```

### L1.8 ★★★ host 侧禁止 `permute(...).contiguous()`——输入输出都不许

参考实现里的 `transpose(1, 2).contiguous()` 会变成一个独立的 `Transpose` kernel
（实测 19.46 us）。**新口径下这些搬运 kernel 如实计入 impl**，而它们不产生任何计算。

**实测代价：同一个算子，tile 配置完全相同（`128/256/64`、`BLOCK_Q=BLOCK_KV=128`），
只因为多了 4 处 `.contiguous()`，impl 从 0.0715 变成 0.1287 ms（慢 1.8 倍），
speedup 2.5119 → 1.4420，直接从达标掉到不达标。**

```python
# ❌ QKV 融合投影做对了，紧接着又把三份各自物化一遍 —— 前功尽弃
qkv_out = torch.empty((B*S, D + 2*KH*dh), ...)
linear_kernel[grid](x_flat, w_all, qkv_out, ...)
q = qkv_out[:, :D].view(B,S,H,dh).permute(0,2,1,3).contiguous()              # 搬运 1
k = qkv_out[:, D:D+KH*dh].view(B,S,KH,dh).permute(0,2,1,3).contiguous()      # 搬运 2
v = qkv_out[:, D+KH*dh:].view(B,S,KH,dh).permute(0,2,1,3).contiguous()       # 搬运 3
...
attn_flat = attn_out.permute(0,2,1,3).contiguous().view(B*S, D)              # 搬运 4
```

```python
# ✅ 让 kernel 带 stride 直接读原始布局，输出直接写成目标布局，一次搬运都不要
qkv = torch.empty(M, D + 2*d_kv, ...)          # [M, q|k|v] 连续，不再拆
gqa_attention_kernel[grid](
    qkv, out, ...,                              # kernel 内用 k_base=D / v_base=D+d_kv
    qkv.stride(0), out.stride(0), ...)          # 偏移 + stride 定位，不物化
# kernel 末尾: tl.store(o_ptr + row[:,None]*stride_orow + (h*DH + offs_d)[None,:], ...)
#              直接落成 [B, S, H, dh]
```

**自查**：forward 里出现 `.contiguous()`、`.permute(...).contiguous()`、
或对 kernel 输出再做一次 `transpose/view` 重排——**都要问一句「这次搬运能不能用 stride 消掉」**。
本类算子里答案几乎总是「能」。

### L1.9 ★★★ 权重复刻：**先对拍再写 kernel**，否则精度全崩且症状会骗人

**判别**：golden 的 `_layers()` / `__init__` 里出现 `torch.manual_seed(42)` + `nn.Linear(...)`
⇒ 权重不是输入，而是**由 RNG 现场生成**，实现侧必须**逐位复刻**。
`gqa-proj` 与 `gqa-router` 两个子类都命中。

**先按误差量级分两类，但只用来决定"先查哪一边"，不要当成结论**：

| 误差量级 | 含义 | 先查 |
|---|---|---|
| MERE 刚刚超阈（如 1.4e-2 vs 7.8e-3） | 精度余量不够 | §6.2 的 `p` 降精度问题 |
| 相对误差 **0.5~1.5 量级** | 结果整个是错的 | **权重复刻 → 若权重没问题，再查计算本身** |

⚠️ **第二行不能反推成"一定是权重错"。** 实测反例：某次验证跑里相对误差 0.6~1.3，
CPU 对拍却显示权重与参考**逐位相同**（三组不同 shape 全 `torch.equal`），
真正的错在计算侧（GQA head 映射 / 输出布局 / 掩码）。
⇒ **权重对拍是"秒级、零成本、能一次性排除一整条链路"的第一步，不是答案本身。**
先做它，是因为它便宜；做完要接着往下查。

**可直接套用的配方**（已在两个算子上 50/50 验证）：

```python
import math, torch

def _make_weights(d_model, n_heads, n_kv_heads, device, dtype):
    head_dim = d_model // n_heads
    g = torch.Generator()            # 独立 Generator，不碰全局 RNG
    g.manual_seed(42)

    def lin(out_f, in_f, bias):
        # nn.Linear.reset_parameters 的等价展开，顺序不能变
        w = torch.empty((out_f, in_f), dtype=torch.float32)   # ★ CPU + fp32 采样
        torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5), generator=g)
        b = None
        if bias:                       # bias=False 时**不能**抽这一次
            bound = 1.0 / math.sqrt(in_f) if in_f > 0 else 0.0
            b = torch.empty((out_f,), dtype=torch.float32)
            torch.nn.init.uniform_(b, -bound, bound, generator=g)
        return w, b

    # ★ 调用顺序与次数必须与 golden 里 nn.Linear 的**书写顺序**逐一对应
    q_w, _ = lin(d_model, d_model, False)
    k_w, _ = lin(n_kv_heads * head_dim, d_model, False)
    v_w, _ = lin(n_kv_heads * head_dim, d_model, False)
    o_w, _ = lin(d_model, d_model, False)

    # 预转置成 [in, out]（L1.6）+ QKV 三合一；转置/拼接是纯数据搬移，数值逐位不变
    d_kv = n_kv_heads * head_dim
    qkv_t = torch.empty((d_model, d_model + 2 * d_kv), dtype=torch.float32)
    qkv_t[:, :d_model].copy_(q_w.t())
    qkv_t[:, d_model:d_model + d_kv].copy_(k_w.t())
    qkv_t[:, d_model + d_kv:].copy_(v_w.t())
    o_t = torch.empty((d_model, d_model), dtype=torch.float32)
    o_t.copy_(o_w.t())
    return (qkv_t.to(device=device, dtype=dtype),      # ★ 最后才 .to(device, dtype)
            o_t.to(device=device, dtype=dtype))
```

四个必须与 golden 一致的点，**漏一个就全崩**：

| # | 点 | 常见错法 |
|---|---|---|
| 1 | **采样在 CPU、fp32** | 直接在 npu 上、或直接用 `x.dtype` 采样 ⇒ RNG 序列不同 |
| 2 | **调用顺序与次数** | 漏掉/多出一次 `lin(...)`，或顺序与 golden 不同 ⇒ 后面全部错位 |
| 3 | **`bias=False` 时不抽 bias** | 无条件抽 bias ⇒ 每层多消耗一段 RNG |
| 4 | **`.to(device, dtype)` 放在最后** | 提前转换 ⇒ 与 golden 的 `nn.Linear(...).to(...)` 不等价 |

⇒ **写 kernel 之前先花一分钟在 CPU 上对拍**（不占 NPU、秒级出结果）：

```python
torch.manual_seed(42)
ref = nn.Linear(d_model, d_model, bias=False)      # 按 golden 的顺序逐个构造
mine, _ = lin(d_model, d_model, False)
assert torch.equal(ref.weight, mine)               # 必须逐位相等，不是 allclose
```

实测把测试集里出现的 45 个不同 `(D, H, options)` 组合全部对拍通过之后，
**后续任何精度失败都不必再怀疑这条链路**——省下的是好几轮的诊断时间。

### L1.10 ★★★ tile **不能切片、不能下标赋值、不能拼接**——这是首版生成的头号杀手

**实测：两个算子的首版生成 50/50 全挂，根因都是这一条。** 它比看上去更严格——
Triton 的 tile 不是 numpy 数组，**没有任何形式的部分读写**。

```python
# ❌ 切片取值 —— ValueError('unsupported tensor index: slice(None, constexpr[64], None)')
k_tile = kv_tile[:, :BLOCK_D]
v_tile = kv_tile[:, BLOCK_D:]

# ❌ 切片赋值 —— AssertionError()（报错位置指向 `=` 左边，信息几乎没有提示性）
q_tile[g * BLOCK_Q:(g + 1) * BLOCK_Q, :] = tl.load(q_ptrs, mask=mask, other=0.0)

# ❌ 单元素读写、按运行时标量下标取行
x = tile[i]           # i 是运行期值
tile[i, j] = v
```

**正确做法（按意图分三类）**：

| 你想做的事 | 正确写法 |
|---|---|
| 从一块 `[BS, 2D]` 的 tile 里分别拿 K 和 V | **不要切**。发**两次 `tl.load`**，各自用自己的指针偏移（`+0` 和 `+D`）；或走 §3.4 的整页方案——`dot(q补零, kvᵀ)` 与 `dot(p, kv)` 的**列掩码 + 指针偏移 `-D`**，被掩掉的 lane 不写内存 |
| 把若干小块拼成一个大 tile | **拼不了**。`tl.cat` 对 2D 不支持，`tl.join` + 3D `tl.trans` 在本类算子上会 UB 溢出（§5.2 实测）。改成「一次 `tl.load` 用一个大的 offs 向量直接读出整块」 |
| 分段填充一个大 tile（如按 g 分组填 q） | **一次读完**。用 `offs = tl.arange(0, BLOCK_G * ...)` 构造完整下标，一次 `tl.load` 出目标形状；分组语义靠 `offs // G` / `offs % G` 算进指针里 |
| 按条件跳过一段计算 | 循环体内**不支持 `continue`**，用 `if` 把整段包住 |

> **自查**：写完 kernel 后 grep 一遍 `[` 后面跟 `:` 的模式、以及等号左边出现 `]` 的行。
> 命中任何一条都要改掉——这些错误在 Phase 3 才会以极不友好的报错暴露出来
> （切片赋值只报一个裸的 `AssertionError()`，不告诉你原因）。

### L1.11 ★★★ 「按分支/路由选权重」**不等于**「按行选权重」——不许因此把 M 维退化成一行一 program

这是本子类**复现次数最多的一条**（`33_AdaptiveAttention` 上连续 7 轮命中，是唯一卡住达标的原因）。
前 5 轮它表现为「`bm` 被 `_` 丢掉」，第 6、7 轮暴露出真实成因——它不是疏忽，是**一条错误推理**，
生成代码甚至把它当设计写进了 docstring：

```python
# ❌ 第 6/7 轮实测原文（speedup 锁死在 1.13，GEMM 段 75.78us → 1592.84us，慢 21 倍）
def _tile_config(M, N, K, dtype):
    """投影 GEMM tile 按 dtype 分档；linear_sel 每 program 处理一行，BLOCK_M 固定为 1。"""
    return 1, bn, bk                                     # ← 主动写死 1
grid_qkv = min(num_cores, M * ((n_max + bn - 1) // bn))  # ← M * ，一行一个 task
```

**错误推理**：「每个样本选的分支不同 ⇒ 相邻行的权重不同 ⇒ 行之间不能合并成 tile ⇒ 一 program 一行」。

**事实**：`sel` 的形状是 **`[B]`**，索引是 `sel[b]`，**不是 `[M]` 也不是 `[B, S]`**。
一个 batch 内的**全部 `S` 行共享同一个分支**。所以 `M = B*S` 这一维天然是
「`B` 段、每段 `S` 行同权重」的结构：**段内可以自由开到 `BLOCK_M=128`**，只是不能跨 batch 边界拼 tile。

```python
# ✅ 把 b 单独占一级，s 维就完全自由了——这正是 grid 要写成三级而不是两级的原因
bm, bn, bk = (128, 256, 64) if x.dtype == torch.float32 else (128, 256, 128)
g3 = min(num_cores, B * triton.cdiv(S, bm) * triton.cdiv(n_max, bn))   # ★ cdiv(S, bm)，不是 M *
linear_sel_kernel[(g3,)](..., BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)      # ★ 三个都要传
```

**强制要求**：

1. **写之前先看 `sel.shape` 再决定**，不许凭「稀疏 / 路由 / 分支」的直觉推。
   只有当选择**真的是逐行的**（`sel` 为 `[M]` 或 `[B, S]`）才需要考虑退化；本子类不是。
2. **不要把 tile 计算封装成辅助函数**（`_tile_config(...)` 之类）。7 次复现里有 4 次的直接诱因就是它：
   一旦返回三元组，调用处就会写成 `_, bn, bk = ...` 把 `bm` 丢掉。**就地写死一行**，让 `bm` 无处可丢。
3. grep 自查三条，命中任何一条都必须改：
   `return 1, ` ／ `_, bn` ／ `grid = min(核数, M * cdiv(...))`。
   正确的 grid 里**一定出现 `cdiv(S, bm)`**——`grep -c 'cdiv(S'` 为 0 即为不合规。

> ⚠️ 本条属于 **Layer 1（硬性设计约束）**，不是 Layer 2 骨架里的参考写法。
> 教训本身也值得记：同一条规则先后写进 §3.3 骨架 5 次（含 ⛔⛔ 硬性要求、❌/✅ 对照、
> 完整可编译原文）**全部无效**，因为 Layer 2 被框架定义为「参考方向，输出必须是全新草图」、
> Layer 3 被定义为「结构必须重新设计」，而 `precheck.json` 只逐条摘录 **Layer 1**。
> ⇒ **凡是"必须这么写"的规则，一律写进 §2 Layer 1；写进 §3/§4 等于没写。**

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 `gqa-sparse-decode` 主骨架

```
pid -> (t, kh)                         # ★ L1.1：按 kv_head 分组，不是 q_head
req = t // DQL;  q_off = t - req*DQL
kv_len = max(seq_lens[req] - DQL + q_off + 1, 0)
rtopk  = min(ceil(kv_len / BS), TOPK)
kv_len_f = float(kv_len)               # 循环不变量外提；位置比较走 f32（整数 LT 会标量降级）

q_tile = load Q[t, kh*G:(kh+1)*G, :]   # [BLOCK_G, D]
m, l, acc = -inf, 0, 0
for i in range(rtopk):                 # 运行期 trip-count
    blk  = load TOPK_IDX[kh, t, i]     # 标量
    page = load BLOCK_TABLE[req, blk]  # 标量索引 → 标量 load（★ 不要向量 gather）
    k, v = load KV[page, kh]           # 页内连续
    scores = dot(q_tile, trans(k)) * sm_scale
    scores = where(pos_f < kv_len_f, scores, -inf)
    m_new = maximum(m, max(scores, 1))
    p     = exp(scores - m_new)
    pv    = dot(p, v)                  # CV 重排：Cube 的 dot 先发，Vector 的 sum/exp 后算
    delta = exp(m - m_new)
    acc, l, m = acc*delta + pv, l*delta + sum(p, 1), m_new
out = acc / l
```

### §3.2 `gqa-proj` 稠密 GQA 前向骨架（含投影）

```
host:  x[B,S,D] --一次融合 GEMM--> qkv[B*S, D + 2*KH*dh]     # 权重预转置 [in,out]，L1.6
       (小心：QKV 三合一是真优化，旧口径下反而会让分数变差，见 §7.2)

kernel: pid -> (b, h, q_tile)                                 # grid = min(核数, tasks) + 步长循环
        kh = h // (H // KH)                                   # GQA 映射；等价于 h * KH // H
        q_tile = load qkv[b, q0:q0+BQ, h*dh : (h+1)*dh]
        m, l, acc = -1e30, 0, 0                               # ★ 有限极小值，不用 -inf
        for kv0 in range(0, S, BKV):
            kv_valid = (kv0 + arange(BKV)) < S                # ★ G6：先算好有效列
            k_tile = load(..., mask=..., other=0.0)           # 操作数清零（保 dot 正确）
            v_tile = load(..., mask=..., other=0.0)
            scores = dot(q_tile, trans(k_tile)) * inv_sqrt
            scores = where(kv_valid[None, :], scores, -3.0e38)  # ★★★ G6：掩在 scores 上
            m_new  = maximum(m, max(scores, 1))
            p      = exp(scores - m_new)
            pv     = dot(p, v_tile)                           # CV 重排：Cube 先发
            delta  = exp(m - m_new)
            acc, l, m = acc*delta + pv, l*delta + sum(p, 1), m_new
        store acc / l  ->  out[b, s, h, dh]                   # ★ 直接按目标布局落盘，L1.8

host:  out.reshape(B*S, D) --GEMM--> final                    # 与 QKV 投影复用同一个 kernel
```

两个最大的收益点都不在这个骨架里，而在参数上：**投影 GEMM 的 tile 按 dtype 分档**（§4.1，实测 +57%）
与 **grid 收缩 + 核内步长循环**（L1.7）。attention 的 tile 基本没空间（§4.2）。

### §3.3 `gqa-router` 的四段结构

参考实现算满 N 路再乘 one-hot 求和 ⇒ 未被选中的 (N-1) 路贡献恒为 0。

```
kernel 1  router_mean    : xm[B,D] = x.mean(dim=1)          grid = min(核数, B*ceil(D/BD))
kernel 2  router_fc1     : h = relu(xm @ W1ᵀ + b1)          沿 DQ 并行
kernel 3  router_argmax  : sel[b] = argmax(h @ W2ᵀ + b2)    grid = B
然后每一路（forward 里展开成 if，禁 Python for）：
kernel 4  qkv_sel        : 门控 QKV 融合投影
kernel 5  gqa_attention  : 门控 attention
kernel 6  out_sel        : 门控 out_proj，每行在 N 路中恰好被写一次
```

⚠️ router 必须拆成三段：单 kernel `grid = B (≤4)` 意味着最多 4 个核在读 `D²/4` 的 W1，
且 `[BLOCK_DQ, BLOCK_KD]` 的 W1 tile 会直接 UB 溢出
（实测 `ub overflow, requires 1859584 bits while 1572864 bits available`）。

#### ★★★ 完整参考骨架（已实测 50/50 + speedup 5.78，直接照此结构写）

> 本骨架来自一份通过全部 50 case 的实现。**照抄结构与关键行**，变量名/细节自行设计。
> 这个子类是本文档记录里最难的一个（router + 多分支门控 + 权重复刻三者叠加），
> 只给零散规则不足以落地 —— 实测连续四轮生成都栽在不同的点上。

```python
# ============ host: 权重打包(只做一次, 按 key 缓存) ============
def _prepare(d_model, n_heads, options, device, dtype):
    head_dim = d_model // n_heads
    g = torch.Generator(); g.manual_seed(42)          # ★ L1.9 权重复刻
    def lin(out_f, in_f, bias): ...                   # kaiming_uniform_(a=sqrt(5)) + 可选 bias
    w1, b1 = lin(max(1, d_model//4), d_model, True)   # router fc1
    w2, b2 = lin(len(options), max(1,d_model//4), True)
    # ★★★ 各分支权重打包进**一个**连续缓冲, 按最大宽度对齐
    max_dkv = max(options) * head_dim
    n_max   = d_model + 2 * max_dkv
    # ★★★ 必须 zeros 不能 empty: 各分支 d_kv 不同, 窄分支在 d_model+2*d_kv 之后的列
    #     是填充区; kernel 按 BLOCK_DH 补齐读取时会吃到未初始化内存。
    #     实测写成 torch.empty 会让三种 dtype **均匀**失败、相对误差 100 倍量级。
    qkv_all = torch.zeros((len(options), d_model, n_max), dtype=torch.float32)
    o_all   = torch.empty((len(options), d_model, d_model), dtype=torch.float32)
    for i, n_kv in enumerate(options):
        d_kv = n_kv * head_dim
        q_w,_ = lin(d_model, d_model, False); k_w,_ = lin(d_kv, d_model, False)
        v_w,_ = lin(d_kv, d_model, False);    o_w,_ = lin(d_model, d_model, False)
        qkv_all[i, :, :d_model].copy_(q_w.t())                       # ★ L1.6 预转置 [in,out]
        qkv_all[i, :, d_model:d_model+d_kv].copy_(k_w.t())
        qkv_all[i, :, d_model+d_kv:d_model+2*d_kv].copy_(v_w.t())
        o_all[i].copy_(o_w.t())
    dkv_t  = torch.tensor([k*head_dim for k in options], dtype=torch.int32)   # 每分支 kv 宽度
    khs_t  = torch.tensor(list(options), dtype=torch.int32)                   # 每分支 n_kv_heads
    nsel_t = torch.tensor([d_model+2*k*head_dim for k in options], dtype=torch.int32)
    return (..., n_max)      # ★ 全部 .to(device, dtype) 放在最后

# ============ forward: 6 次发射(router 3 + 分支 3), 分支侧各只发一次 ============
xm = torch.empty(B, D, ...)          # ★ [B, D] 不是 [B, DQ]! kernel 按 b*D+d 索引
router_mean_kernel[(g1,)](x, xm, B, S, D, ...)                       # xm = x.mean(dim=1)
router_fc1_kernel[(g2,)](xm, w1_t, b1, h, B, DQ, D, ...)             # relu(xm @ W1ᵀ + b1)
router_argmax_kernel[(B,)](h, w2, b2, sel, DQ, ..., NOPT=n_opt,
                           BLOCK_NOPT=triton.next_power_of_2(n_opt)) # ★ 非 2 幂必须补齐
qkv = torch.empty(M, n_max, ...)                                     # ★ 按最大宽度分配
# ⛔⛔⛔ 先破除一个**会让你主动写出 BLOCK_M=1 的错误推理**(实测第 6 次复现时,
#   生成代码在 `_tile_config` 的 docstring 里把它当成设计写了出来:
#   「linear_sel 每 program 处理一行, BLOCK_M 固定为 1」, 并 `return 1, bn, bk`)。
#
#   ❌ 错误推理: 「每个样本选的分支不同 ⇒ 相邻行用的权重不同 ⇒ 行之间不能合并成 tile
#                ⇒ 只能一 program 一行」。
#   ✅ 事实: **`sel` 是按 batch `b` 选的, 不是按行 `m` 选的**。
#      `sel` 的形状是 `[B]`, 索引是 `sel[b]`; 一个 batch 内的 **全部 S 行共享同一个分支**。
#      所以 M = B*S 这一维天然是「B 段, 每段 S 行同权重」的结构 ——
#      **段内可以自由开到 BLOCK_M=128**, 只是不能跨 batch 边界拼 tile。
#   ⇒ 这正是骨架里 grid 写成 `B * cdiv(S, bm) * cdiv(N, bn)` 三级、而不是
#      `cdiv(M, bm) * cdiv(N, bn)` 两级的原因: **b 单独占一级就是为了不跨边界**,
#      拆出 b 之后 s 维就完全自由了。
#   ⇒ 只有当选择真的是**逐行**的(`sel` 形状为 `[B, S]` 或 `[M]`)才需要考虑退化,
#      本算子不是。写之前先看一眼 `sel` 的 shape 再决定, 不要凭「稀疏/路由」直觉推。
#
# ⛔⛔ 硬性要求: **不要把 tile 计算封装成辅助函数**(如 `_tile_config(...)`)。
#   实测 5 次复现同一个 bug, 其中 3 次的直接诱因就是这个抽象 ——
#   一旦它返回三元组, 调用处就会写成 `_, bn, bk = _tile_config(...)` 把 bm 丢掉。
#   **就地写死三行**, 让 bm 无处可丢:
#       bm, bn, bk = (128, 256, 64) if x.dtype == torch.float32 else (128, 256, 128)
#   然后三个都必须出现在 kernel 调用里。
#
# ⛔ 下面这三行是实测出现过 5 次的错法, 写完请逐行 grep 自查:
#     _, bn, bk = _tile_config(...)                      # ❌ 用 `_` 把 BLOCK_M 丢掉
#     grid = min(核数, M * cdiv(N, bn))                  # ❌ M * 而非 cdiv(M, BLOCK_M) *
#     kernel[(grid,)](..., BLOCK_N=bn, BLOCK_K=bk)       # ❌ 不传 BLOCK_M
#   后果: 一行一个 task, GEMM 段从 75.78us 变 1592.84us(慢 21 倍), speedup 5.78 -> 1.15。
#   「M 维塌掉」在本子类上出现过 4 次(router_fc1 退化 1D / prefill 任务划分 /
#    grid 写成 M* / 用 `_` 丢弃 bm), 是最顽固的一类错。
# ★★★ grid 必须按 (b, s_tile, n_tile) 三级算, 且 BLOCK_M 一定要传进去!
#     实测漏传 BLOCK_M、grid 写成 min(核数, M * cdiv(N, BN)) => 一行一个 task,
#     linear_sel_kernel 从 75.78us 变成 1592.84us(慢 21 倍), speedup 5.78 -> 1.15。
bm, bn, bk = (128, 256, 64) if x.dtype == torch.float32 else (128, 256, 128)
g3 = min(num_cores, B * triton.cdiv(S, bm) * triton.cdiv(n_max, bn))      # ★ cdiv(S, bm) 不是 M
linear_sel_kernel[(g3,)](x2, qkv_all, qkv, sel, nsel_t, B, n_max, D, S, ...,
                         BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk)              # ★ 三个都要传
gqa_attention_kernel[(g4,)](qkv, attn, sel, khs_t, dkv_t, B, S, ...)
linear_sel_kernel[(g5,)](attn, o_all, out, sel, nout_t, B, D, D, S, ...)   # ★ 复用同一 jit

# ============ kernel: 分支在内部选, 宽度全部来自同一处 ============
br    = tl.load(sel_ptr + b)          # 标量
n_b   = tl.load(nsel_ptr + br)        # 该分支真实输出宽度
kh_v  = tl.load(khs_ptr + br)         # 该分支 n_kv_heads(运行期标量)
dkv   = tl.load(dkv_ptr + br)
kh    = h * kh_v // H                 # 每 task 一次标量除, 不在 KV 循环内
k_base, v_base = D, D + dkv           # ★ 偏移由 dkv 决定, 不要硬编码
```

#### ★★★★ router 三段 kernel 的**完整可编译参考实现**（直接照此写）

> 连续六轮生成**全部栽在 router 段**（UB 超 6.7 倍 → tile 宽度 64-vs-128 不一致 →
> `tl.dot` 操作数退化 1D），共同的误区是「`M = B ≤ 4` 很小，所以那一维可以省掉」。**不能省。**
> 下面是通过全部 50 case 的原文；`router_fc1` 的二维 tile + 掩码写法是重点。

```python
@triton.jit
def router_mean_kernel(
    x_ptr, xm_ptr,
    B, S, D,
    stride_xb, stride_xs,
    BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr, NUM_CORES: tl.constexpr,
):
    """xm[B, D] = x.mean(dim=1)，沿 (b, d_tile) 并行。torch 对低精度输入是 fp32 累加后回铸。"""
    pid = tl.program_id(0)
    d_tiles = tl.cdiv(D, BLOCK_D)
    total = B * d_tiles
    offs_s = tl.arange(0, BLOCK_S)
    offs_d0 = tl.arange(0, BLOCK_D)

    for task in range(pid, total, NUM_CORES):
        b = task // d_tiles
        dt = task - b * d_tiles
        d_idx = dt * BLOCK_D + offs_d0
        d_mask = d_idx < D

        acc = tl.zeros([BLOCK_D], tl.float32)
        for s0 in range(0, S, BLOCK_S):
            s_idx = s0 + offs_s
            m = (s_idx < S)[:, None] & d_mask[None, :]
            v = tl.load(
                x_ptr + b * stride_xb + s_idx[:, None] * stride_xs + d_idx[None, :],
                mask=m)
            acc += tl.sum(tl.where(m, v, 0.0).to(tl.float32), axis=0)

        tl.store(xm_ptr + b * D + d_idx,
                 (acc / S).to(xm_ptr.dtype.element_ty), mask=d_mask)

@triton.jit
def router_fc1_kernel(
    xm_ptr, w1_ptr, b1_ptr, h_ptr,
    B, DQ, D,
    stride_w1,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """h[B, DQ] = relu(xm[B,D] @ W1ᵀ[D,DQ] + b1)。relu 保号, 落盘顺序不影响数值。"""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(DQ, BLOCK_N)
    num_tiles = tl.cdiv(B, BLOCK_M) * num_pid_n
    num_programs = tl.num_programs(0)

    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(pid, num_tiles, num_programs):
        pid_m = tile // num_pid_n
        pid_n = tile - pid_m * num_pid_n
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        m_mask = offs_m < B
        n_mask = offs_n < DQ

        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k0 in range(0, D, BLOCK_K):
            k_idx = k0 + offs_k
            k_mask = k_idx < D
            am = m_mask[:, None] & k_mask[None, :]
            bm = k_mask[:, None] & n_mask[None, :]
            a = tl.load(xm_ptr + offs_m[:, None] * D + k_idx[None, :], mask=am)
            a = tl.where(am, a, 0.0)
            b = tl.load(w1_ptr + k_idx[:, None] * stride_w1 + offs_n[None, :], mask=bm)
            b = tl.where(bm, b, 0.0)
            acc += tl.dot(a, b)

        acc += tl.where(n_mask, tl.load(b1_ptr + offs_n, mask=n_mask), 0.0).to(tl.float32)[None, :]
        acc = tl.maximum(acc, 0.0)
        tl.store(h_ptr + offs_m[:, None] * DQ + offs_n[None, :],
                 acc.to(h_ptr.dtype.element_ty),
                 mask=m_mask[:, None] & n_mask[None, :])

@triton.jit
def router_argmax_kernel(
    h_ptr, w2_ptr, b2_ptr, sel_ptr,
    DQ, stride_w2,
    NOPT: tl.constexpr, BLOCK_NOPT: tl.constexpr, BLOCK_DQ: tl.constexpr,
):
    """sel[b] = argmax(h[b] @ W2ᵀ + b2)。回铸 x.dtype 后再比较, 让 argmax 落在同一量化格点。"""
    b = tl.program_id(0)
    offs_q = tl.arange(0, BLOCK_DQ)
    q_mask = offs_q < DQ
    hv = tl.where(q_mask, tl.load(h_ptr + b * DQ + offs_q, mask=q_mask), 0.0).to(tl.float32)

    offs_o = tl.arange(0, BLOCK_NOPT)
    o_mask = offs_o < NOPT
    w2m = o_mask[:, None] & q_mask[None, :]
    w2 = tl.load(w2_ptr + offs_o[:, None] * stride_w2 + offs_q[None, :], mask=w2m)
    w2 = tl.where(w2m, w2, 0.0).to(tl.float32)

    logits = tl.sum(w2 * hv[None, :], axis=1)
    logits += tl.where(o_mask, tl.load(b2_ptr + offs_o, mask=o_mask), 0.0).to(tl.float32)
    logits = logits.to(h_ptr.dtype.element_ty).to(tl.float32)
    logits = tl.where(o_mask, logits, float("-inf"))

    best = tl.max(logits, axis=0)
    idx = tl.where(logits == best, offs_o, BLOCK_NOPT)   # tie-break 取最小下标
    tl.store(sel_ptr + b, tl.min(idx, axis=0))
```

#### ★★★★ `gqa_attention_kernel` 的**完整可编译参考实现**

> 实测九轮首验 28/50，剩余 22 个失败**全部**是同一条：
> `MERE=1.264e-02 > rel_thr=7.813e-03`（只超 **1.6 倍**）——根因是
> ```python
> acc = ... + tl.dot(p.to(qkv_ptr.dtype.element_ty), v_tile)   # ❌ p 降到输入 dtype
> ```
> `p` 舍入到 bf16 的半 ulp `2^-9` 在 `Σp·v / Σp` 上被放大后越阈（§6.2）。
> **正确写法是 fp32 单 dot**：`tl.dot(p, v_tile.to(tl.float32))` —— 见下面原文。

```python
@triton.jit
def gqa_attention_kernel(
    qkv_ptr, o_ptr, sel_ptr, khs_ptr, dkv_ptr,
    B, S,
    stride_row, stride_orow,
    sm_scale,
    H: tl.constexpr, DH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DH: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    """GQA(无因果掩码) online softmax —— **只发一次**, 分支的 KH / kv 偏移在 kernel 内取。

    qkv_ptr 指向融合投影缓冲 [B*S, D + 2*max_dkv]（按最大分支宽度分配, 只用前 D+2*dkv_b 列）;
    输出直接写成 [B, S, H, DH], 省掉参考实现的 transpose(1,2).contiguous()。
    `kh_v` 是运行期标量 ⇒ `kh = h * kh_v // H` 有一次标量除, 但每个 task 只算一次、
    不在 KV 循环内, 与 L1.2 禁止的"运行期标量除 + 向量下标 gather"组合无关。
    """
    pid = tl.program_id(0)
    m_tiles = tl.cdiv(S, BLOCK_M)
    tiles_per_b = H * m_tiles
    total = B * tiles_per_b
    D = H * DH

    offs_d = tl.arange(0, BLOCK_DH)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < DH

    for task in range(pid, total, NUM_CORES):
        b = task // tiles_per_b
        rem = task - b * tiles_per_b
        h = rem // m_tiles
        mt = rem - h * m_tiles

        br = tl.load(sel_ptr + b)
        kh_v = tl.load(khs_ptr + br)
        dkv = tl.load(dkv_ptr + br)
        kh = h * kh_v // H                    # GQA: repeat_interleave 的等价索引
        k_base = D
        v_base = D + dkv

        offs_m = mt * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = offs_m < S
        row = b * S + offs_m

        qm = m_mask[:, None] & d_mask[None, :]
        q_tile = tl.load(
            qkv_ptr + row[:, None] * stride_row + (h * DH + offs_d)[None, :], mask=qm)
        q_tile = tl.where(qm, q_tile, 0.0)

        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DH], tl.float32)

        for n0 in range(0, S, BLOCK_N):
            n_idx = n0 + offs_n
            n_mask = n_idx < S
            krow = b * S + n_idx
            kvm = n_mask[:, None] & d_mask[None, :]

            k_tile = tl.load(
                qkv_ptr + k_base + krow[:, None] * stride_row
                + (kh * DH + offs_d)[None, :], mask=kvm)
            k_tile = tl.where(kvm, k_tile, 0.0)

            scores = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale
            scores = tl.where(n_mask[None, :], scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            p = tl.where(n_mask[None, :], p, 0.0)

            v_tile = tl.load(
                qkv_ptr + v_base + krow[:, None] * stride_row
                + (kh * DH + offs_d)[None, :], mask=kvm)
            v_tile = tl.where(kvm, v_tile, 0.0)

            # CV 重排: Cube 的 dot(p,v) 先发, Vector 的 sum/exp 后算
            pv = tl.dot(p, v_tile.to(tl.float32))
            p_sum = tl.sum(p, axis=1)
            alpha = tl.exp(m_i - m_new)

            acc = acc * alpha[:, None] + pv
            l_i = l_i * alpha + p_sum
            m_i = m_new

        out = acc / l_i[:, None]
        o_ptrs = o_ptr + row[:, None] * stride_orow + (h * DH + offs_d)[None, :]
        tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=qm)
```

#### ★★★ 六个 kernel 的 tile 取值（照抄，不要自己拍）

实测 UB 超 **6.7 倍**（`requires 10551296 bits while 1572864 bits available`）就是拍错 tile 的后果。
下表来自 50/50 + 5.78x 的实现：

| kernel | tile | 说明 |
|---|---|---|
| `router_mean` | `BLOCK_D = min(256, next_pow2(D))`<br>`BLOCK_S = max(1, min(next_pow2(S), 8192 // BLOCK_D))` | ★ `BLOCK_S` 由 `8192 // BLOCK_D` 反推，**这是防 UB 溢出的关键**；grid = `min(核数, B * cdiv(D, BLOCK_D))` |
| `router_fc1` | `BLOCK_M=16, BLOCK_N=32, BLOCK_K=128` | ★★★ M 只有 `B ≤ 4`。**既不能开 128（撑爆 UB），也不能因为 M 小就退化成 1D 向量** —— 必须保持 `[BLOCK_M, BLOCK_K]` 二维 tile + `m_mask = offs_m < B` 掩码。实测把 M 维消掉会让全部 50 个 case 报 `AssertionError('Both inputs must be either 2D or 3D; lhs: [constexpr[128]]')`。沿 DQ 并行，grid = `min(核数, cdiv(DQ, 32))` |
| `router_argmax` | `BLOCK_NOPT = next_pow2(NOPT)`<br>`BLOCK_DQ = next_pow2(DQ)` | grid = `B`；只读 `[B,DQ]` 与 `[NOPT,DQ]` 两块小张量 |
| `linear_sel`（QKV 投影 / out_proj 复用） | fp32 `128/256/64`<br>低精度 `128/256/128` | 见 §4.1，**不要写成 `128/64/128` 或 `64/128/64`** |
| `gqa_attention` | `BLOCK_DH = next_pow2(head_dim)`<br>`BDH≤64 → 128/128`<br>`BDH=128` 且 fp32 → `64/64`，低精度 → `128/64` | 再按 `next_pow2(S)` 收窄：`block_m = max(16, min(side_m, cap))` |

⚠️ **`BLOCK_DH` 只在 host 算一次**并作为 constexpr 传入；kernel 里**不要出现字面量 `128`**
（实测 44 个 case 报 `Cannot make_shape_compatible: incompatible dimensions at index 1: 64 and 128`
就是宽度来自多处不一致）。

**四条最容易翻车的地方**（多轮生成各栽过一次）：

| # | 坑 | 症状 | 对策 |
|---|---|---|---|
| 1 | tile 宽度来自多处、彼此不一致 | `Cannot make_shape_compatible: incompatible dimensions at index 1: 64 and 128` | **`BLOCK_DH` 只在 host 算一次**（`next_power_of_2(head_dim)`）作为 constexpr 传入；所有 `k_base`/`v_base` 由 `dkv` 推导，**不要出现字面量 128** |
| 2 | 门控写出没覆盖全部行 | `impl` 恰好为 0、相对误差恰好 **1.0** | 见 §4.4 三个自查点 |
| 3 | `tl.arange` 用运行期长度 | `arange's arguments must be of type tl.constexpr` | host 侧算好 constexpr + 掩码 |
| 4 | 用下标做 argmax | `unsupported tensor index: constexpr[0]` | 见下方三行惯用法 |

#### ★★★★ `linear_sel_kernel` 的**完整可编译参考实现**（QKV 投影与 out_proj 复用同一份）

> `make_shape_compatible` 类错误在这一段**出现三次**（`64 vs 128`、`1D 退化`、`16 vs 256`），
> 共同点都是**掩码与 tile 的维度对不上**：`a` 的掩码要 `[BLOCK_M, BLOCK_K]`、
> `b` 的掩码要 `[BLOCK_K, BLOCK_N]`，两者**不能共用**。
> 下面是通过全部 50 case 的原文。

```python
@triton.jit
def linear_sel_kernel(
    x_ptr, w_ptr, y_ptr, sel_ptr, nsel_ptr,
    B, N_MAX, K, S,
    stride_xm, stride_wbr, stride_wk, stride_ym,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Y[b,s,:N_b] = X[b,s,:] @ W[sel[b]][K, :N_b]  —— **只发一次**, 分支在 kernel 内选。

    v5 是每分支发一次、用 `if sel[b]==BRANCH` 跳过; 新的时延口径会如实计入每一次发射,
    而未选中的那几次仍要把整个 task 空间走完。这里改成一次发射:
      - `br = sel[b]` 一次标量 load, 定位到 W 的第 br 片;
      - `N_b = nsel[br]` 是该分支真实输出宽度, task 空间按 N_MAX 铺, 超出的 n tile 提前退出。
    W 已在 host 侧预转置为 [K, N] ⇒ b tile 最内轴 stride == 1, kernel 内无需 tl.trans。
    tile 空间按 (b, s_tile, n_tile) 展开 —— b 是坐标之一, sel 因此是标量 load
    (v2 曾用 M=B*S 平铺 + `b = m // S` 反推 + 向量下标 gather, 实测直接 aicore timeout)。
    """
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N_MAX, BLOCK_N)
    s_tiles = tl.cdiv(S, BLOCK_M)
    num_tiles = B * s_tiles * num_pid_n
    num_programs = tl.num_programs(0)

    offs_k = tl.arange(0, BLOCK_K)
    for tile in range(pid, num_tiles, num_programs):
        t = tile // num_pid_n
        pid_n = tile - t * num_pid_n
        b = t // s_tiles
        st = t - b * s_tiles

        br = tl.load(sel_ptr + b)
        n_b = tl.load(nsel_ptr + br)
        if pid_n * BLOCK_N < n_b:                     # 该分支用不到的 n tile, 一次标量比较跳过
            offs_s = st * BLOCK_M + tl.arange(0, BLOCK_M)
            s_mask = offs_s < S
            offs_m = b * S + offs_s
            offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
            n_mask = offs_n < n_b

            wb = w_ptr + br * stride_wbr
            acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
            for k0 in range(0, K, BLOCK_K):
                k_idx = k0 + offs_k
                k_mask = k_idx < K
                am = s_mask[:, None] & k_mask[None, :]
                bm = k_mask[:, None] & n_mask[None, :]
                a = tl.load(x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :], mask=am)
                a = tl.where(am, a, 0.0)
                b_t = tl.load(wb + k_idx[:, None] * stride_wk + offs_n[None, :], mask=bm)
                b_t = tl.where(bm, b_t, 0.0)
                acc += tl.dot(a, b_t)

            tl.store(
                y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :],
                acc.to(y_ptr.dtype.element_ty),
                mask=s_mask[:, None] & n_mask[None, :],
            )
```

#### ★ kernel 内 argmax + 最小下标 tie-break 的唯一写法

`sel[b] = argmax(logits)` 是 router 的核心，但 **tile 不能用下标访问**（L1.10），
`logits[0]` 会报 `ValueError('unsupported tensor index: constexpr[0]')`（实测整批全灭）。

```python
# ❌ Python 式扫描 —— tile 不支持任何下标
logits = acc.reshape((NOPTS,))
sel_idx = 0; max_val = logits[0]
for i in range(1, NOPTS): ...
```

```python
# ✅ 三行纯向量惯用法，且天然满足 torch.argmax 的「取最小下标」语义
offs_o = tl.arange(0, BLOCK_NOPT)                      # BLOCK_NOPT = next_pow2(NOPT)
o_mask = offs_o < NOPT
logits = tl.where(o_mask, logits, float("-inf"))       # 无效槽不参与
best = tl.max(logits, axis=0)
idx  = tl.where(logits == best, offs_o, BLOCK_NOPT)    # 命中处放下标, 否则放哨兵
tl.store(sel_ptr + b, tl.min(idx, axis=0))             # 取最小 => tie-break 与 torch 一致
```

⚠️ `NOPT ∈ {2,3}` 不是 2 的幂，`tl.arange` 要求长度是 2 的幂 ⇒ 必须用
`BLOCK_NOPT = triton.next_power_of_2(NOPT)` + 掩码，否则 35/50 的 3 路 case 直接编不过。

#### ★★★ 分支 kernel：**整体只发一次**，不要每分支发一次

```python
# ❌ 每分支发一次。nopts=3 时这三个 kernel 各发 3 次共 9 次发射；
#    其中 2/3 不干活，但**每次发射仍要把整个 task 空间走一遍**
qkv_sel_kernel[grid](..., BRANCH=0); gqa_attention_kernel[grid](..., BRANCH=0); out_sel_kernel[grid](..., BRANCH=0)
qkv_sel_kernel[grid](..., BRANCH=1); ...          # BRANCH=1
qkv_sel_kernel[grid](..., BRANCH=2); ...          # BRANCH=2
#    kernel 内: if tl.load(sel_ptr + b) == BRANCH:  ...
```

```python
# ✅ 每个 kernel 只发一次，分支在 kernel 内按 sel[b] 取偏移
#    host: 把各分支权重打包成 [nopts, K, N_MAX] 连续缓冲 + 两张小表
linear_sel_kernel[grid](x2, qkv_all, qkv, sel, nsel_t, B, N_MAX, D, S, ...)
gqa_attention_kernel[grid](qkv, attn, sel, khs_t, dkv_t, B, S, ...)
linear_sel_kernel[grid](attn, o_all, out, sel, nout_t, B, D, D, S, ...)   # 复用同一个 @triton.jit
#    kernel 内:
#        br   = tl.load(sel_ptr + b)          # 标量 load
#        n_b  = tl.load(nsel_ptr + br)        # 该分支真实输出宽度
#        wb   = w_ptr + br * stride_wbr       # 定位到第 br 片权重
#        if pid_n * BLOCK_N < n_b:  ...       # 各分支宽度不等: 按 N_MAX 铺 task 空间,
#                                             # 用一次标量比较跳过空 tile
```

**实测收益（新口径，如实计入每次发射）：impl 0.1386 → 0.0969 ms（1.43x），
speedup 4.0101 → 5.7479，精度 50/50。**

三个配套要点：

1. **`KH` / `d_kv` 变成运行期标量**（各分支不同）。`kh = h * kh_v // H` 有一次标量除，
   但**每个 task 只算一次、不在 KV 循环内**，与 L1.2 禁止的「运行期标量除 + 向量下标 gather」
   不是一回事；
2. 各分支 qkv 宽度不等（实测同 case 内最大/最小之比可达 2.77），共用缓冲按**最大宽度**分配。
   实测空转 n-tile 占 35.3%，但空转 tile 只做几次标量 load + 比较、不做 GEMM，代价可接受；
3. **QKV 投影与 out_proj 用同一个 `@triton.jit`**（形状语义相同，只是 W 与 N 不同），
   少一份代码也少一次编译。

### §3.4 页内连续 vs 跨步半行

KV cache 常见布局 `[num_blocks, KH, BS, 2*D]`，一行是 `[K(D) | V(D)]`。
分两次 load（K 取前半、V 取后半，行距 `2D`）与一次连续 load 整行，实测差 **2.6 倍**：

| load 形态 | us/页 | 有效带宽 |
|---|---|---|
| 只读 K（每行取一半，行距整行） | 0.575 | 1367 GB/s |
| 读 K + V（两次跨步半行） | **1.692** | 929 GB/s |
| 一次连续读整页 `[BS, 2D]` | **0.642** | 2452 GB/s |

**怎么用一条 load 还能分别拿到 K 和 V**（不需要 `tl.split` / 三维转置）：

- QK：把 q 补零成 `[BLOCK_G, 2D]`（后 D 列恒 0），则 `dot(q2, kvᵀ) == q·Kᵀ`；
- PV：`dot(p, kv)` 得 `[BLOCK_G, 2D]`，前 D 列是 `p·K`（丢弃），后 D 列正是 `p·V`。
  `acc` 保持 `[BLOCK_G, 2D]`，store 时用**列掩码 + 指针偏移 `-D`** 取后 D 列，被掩掉的 lane 不写内存。
- 代价：两个 dot 的 K 维 / N 维都翻倍。算术强度 16 flop/byte ≪ 机器平衡点 ~235 ⇒ Cube 本来就闲，划算。
- ⚠️ **前置条件**：这条会把 UB 需求推高一倍，**必须先把 PV 的 fp32 缓冲换成 fp16**（§4.3），否则必然 `PlanMemory Failed`。

---

## §4 Layer 3: 关键技巧

### §4.1 ★★★ 投影 GEMM 的 tile 按 dtype 分档（`gqa-proj` / `gqa-router` 最大的一笔）

实测 `linear_kernel` **128.8 us → 35.9 us（3.6x）**，端到端 **1.4518 → 2.2839（+57%）**。

⛔ **不要自己拍一组**。实测生成侧多次写成 `128/64/128` 或 `64/128/64` —— 前者 `BLOCK_N`
只有 256 的一半，后者更差。**照抄下面这段**：

```python
if x.dtype == torch.float32:  tgt_m, tgt_n, tgt_k = 128, 256, 64
else:                         tgt_m, tgt_n, tgt_k = 128, 256, 128
bm = max(16, min(tgt_m, triton.next_power_of_2(M)))      # 小形状按 next_pow2 收窄
bk = max(16, min(tgt_k, triton.next_power_of_2(K)))
bn = max(16, min(tgt_n, triton.next_power_of_2(N)))
```

机理：`BLOCK_M=64` 时 M-tile 数 = `ceil(M/64)`，**权重被整份重读这么多遍**——
实测 `M=1149, N=4416, K=1472, fp32` 下是 18 遍 × 26MB ≈ **468MB** GM 流量。

UB 可行域（bishengir 1.2.0 实测，大致 `BM*BN*BK ≤ 2^20`）：

```
fp16/bf16:  128x128x 64 OK    128x128x128 UB overflow    128x256x128 OK
fp32:       128x128x 64 OK    128x 64x128 UB overflow  <== 同乘积也过不了
```

⇒ **fp32 下 BK=128 比同乘积的 BK=64 更吃 UB**（操作数字节翻倍），不能只看乘积。

### §4.2 attention tile 基本没有空间，不要在这里花时间

全档扫描实测：`128/128` 只比各形状的最优档慢 **3~8%**。
把时间花在投影侧（§4.1）或任务划分（L1.1）上。

### §4.3 ★ PV 段用 fp16 原生 dot 换 UB（不是换速度）

golden 通常是 `softmax(fp32) @ V.float()`。三种写法实测：

| 方案 | 循环内 dot 数 | 精度 | 速度 | UB（V 侧缓冲） |
|---|---|---|---|---|
| `dot(p_f32, v.to(f32))` | 2 | 通过 | 1.3791 | **64KB** |
| `dot(p.to(bf16), v)` 单次原生 | 2 | **FAIL** MERE 9.004e-3 > 7.812e-3 | — | 0 |
| **`dot(p.to(f16), v.to(f16))`** | 2 | **通过** | 1.3088（持平） | **32KB** |
| `dot(p_hi,v) + dot(p_lo,v)` | **3** | 极好 1.03e-5 | **0.7089** | 0 |

关键：**bf16 → fp16 在本类算子的取值域内逐位无损**（bf16 的 8 bit 有效位完全装进 fp16 的 11 bit，
`|v|` 远小于 fp16 上限 65504；300 组随机样本实测偏差 **0.0**），
而 `p` 的尾数从 bf16 的 8 bit 涨到 fp16 的 11 bit——单次原生 dot 之前只超阈 **1.15 倍**，
补上这 3 bit 就有充足余量。

⇒ ⛔ **fp16 路线是负优化，不要用它来提速。** 干净同时段 A/B（`81`）：

| 版本 | PV dot | impl | speedup |
|---|---|---|---|
| `dot(p_f32, v.to(f32))` | 2 | **0.3200 ms** | **53.97** |
| `dot(p.to(f16), v.to(f16))` | 2 | 0.4546 ms | 30.99 |

**fp16 慢 1.42 倍**，尽管 UB 缓冲更小（32KB vs 64KB）、dot 个数相同 ——
多出来的是**两次 Vector 转换**（`p` 与 `v` 各一次）而不是一次。
⚠️ 早期在旧口径下测得的 1.3088 vs 1.3791 方向是**反的**，属噪声量级：
**这类 5% 以内的差异必须同时段背靠背复测才能下结论**（见 §7.2b）。

它唯一的价值是**腾出 32KB UB**——只有当你确实需要那 32KB 去装别的东西（如 §3.4 整页
连续 load）、且能证明那个东西的收益大于这 1.42 倍损失时才值得。**本文档记录的两次尝试
都没能满足这个条件。**

> ⛔ **这条只在 bf16 主导的算子上成立，不要当成通用首选。** 三个算子实测：
>
> | 算子 | dtype 构成 | 最严阈值 | fp16 单 dot 结果 |
> |---|---|---|---|
> | `81` | bf16 38 / fp16 12 | bf16 `2^-7` | **50/50 通过** |
> | `80` | bf16 38 / fp16 12 | — | **48/50**（2 个 MERE 超阈 1.8 倍） |
> | `21` | fp16 17 / bf16 17 / fp32 16 | fp16 `2^-10` | **24/50**（7 个 fp16 精度失败 + 19 个 MLIRCompilationError） |
>
> **判据是精度阈值，不是「能不能无损转 fp16」**：V 侧 bf16→fp16 确实逐位无损，
> 但 `p` 从 fp32 降到 fp16 引入的 `2^-11` 相对误差，在 `2^-7`(bf16) 阈值下有 16 倍余量、
> 在 `2^-10`(fp16) 阈值下只剩 2 倍——**后者不够**。
> ⇒ **算子里只要有 fp16 case，就不要走这条**；先用 §6.3 的 CPU 模拟筛一遍再动手。

### §4.4 ★ 路由门控：只算被选中的分支

理论收益 = 分支数的几何平均（实测 35 个 3 路 + 15 个 2 路 ⇒ `3^0.7 × 2^0.3 ≈ 2.656x`）。
三个 per-branch kernel 的 tile 空间里 `b` 都已经是一个坐标，门控只是一次标量 load：

```python
if tl.load(sel_ptr + b) == BRANCH:     # 整块跳过；循环体内不支持 continue
    ...
```

⛔ **写出必须覆盖全部行——这是门控最容易漏的一处**（实测：14 个 case 的 `impl` 整片为
`0.000000e+00`，相对误差恰好 1.0）：

```python
out = torch.empty(M, D, ...)     # ❌ 未被任何分支写到的行 = 未初始化内存
```

参考实现是 `(stack(cands,1) * one_hot).sum(1)`，**每一行都有值**。门控实现必须保证
「每行在 N 路中**恰好**被写一次」。三个自查点：

1. `sel[b] ∈ [0, nopts)` 对所有 `b` 成立（argmax 的 tie-break、越界分支都要看）；
2. forward 里展开的 `if` 段**覆盖了全部 nopts**（`nopts=2` 时不要漏掉第 3 段的守卫，
   `nopts=3` 时不要只写 2 段）；
3. kernel 内的 n-tile / m-tile 提前退出条件不能把该写的行也跳掉。

**症状指纹**：`impl` 恰好为 0、相对误差恰好 1.0（而不是"误差偏大"）⇒ 一定是没写，
不是算错。先查覆盖，不要去查数值。

⚠️ **语义前提（必须实测，不能假设）**：跳过 ≠ 乘 0。仅当未选中分支会产出 NaN/Inf 时两者不等价
（参考实现里 `0 * NaN = NaN` 会传播出来）。
⇒ 跑一遍参考实现，**同时统计最终输出与每一路候选值**的 NaN/Inf；
实测 `cases with non-finite: 0` 之后，这条才从「风险项」变成「已证事实」（且仅对该测试集成立）。

### §4.5 位置比较走 f32，不要用整数 LT

整数比较在本类算子族上会标量降级（实测 +23%）。把 `kv_len` 提前转 f32 作为循环不变量：

```python
kv_len_f = kv_len.to(tl.float32)           # 循环外
pos_f = (blk * BS + n_offs).to(tl.float32) # 循环内
valid = pos_f < kv_len_f
```

### §4.6 CV 指令重排（成本极低，试一次，但不要预设收益）

把 Cube 的 `dot(p, v)` 先发、Vector 的 `sum`/`exp` 后算。实测收益在噪声量级（+1.9%），
但改动成本接近 0，可以顺手做。**不要指望它解决 CV 串行**（见 L1.3 B 轮）。

### §4.7 冗余边界运算消除

当 `BLOCK_D == D` 时 `d_mask` 恒全 true，掩码 + `tl.where` 是纯冗余：

```python
if BLOCK_D == D:                  # constexpr 分支
    k_tile = tl.load(k_ptrs)      # 无掩码 load 让 AIC 可以 GM→L1 直载，
    v_tile = tl.load(v_ptrs)      # 省掉 AIV 的 vsel 清零与 GM 暂存往返
else:
    ...
```

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测收益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 环境闸门（正确工具链 + 健康卡） | 0.9438 → **2.2904** | 无条件，第一件事 |
| 2 | ★★★ attention 用 `tl.dot` + online softmax 重写 | 0.0016 → 1.4518（**~900x**） | 基线里没有 `tl.dot`（先看有没有 `for d in range(head_dim)` 这类标量循环） |
| 3 | ★★★ 任务单元改按 kv_head 分组（L1.1） | 0.2233 → **1.3791（6.18x）** | kernel 里有 `kh = h // G` 且循环变量是 q head |
| 4 | ★★★ 投影 GEMM tile 按 dtype 分档（§4.1） | 1.4518 → **2.2839（+57%）** | 有独立投影 GEMM |
| 5 | ★★ 路由门控只算选中分支（§4.4） | 理论 = 分支数（实测端到端 **2.9033**） | 有 argmax/one-hot 选路 |
| 6 | ★ QKV 三合一 + 权重预转置 `[in,out]`（L1.6） | 含在 #2 内 | 有多个共享输入的 `nn.Linear` |
| 7 | ★ 输出直接按目标布局落盘（L1.8） | 消掉一个 19.46 us 的 Transpose kernel | golden 里有 `transpose(1,2).contiguous()` |
| 8 | 整页连续 load（§3.4） | DMA 1.692 → 0.642 us/页 | 页布局是 `[K|V]` 同行，且已做 §4.3 |
| 9 | 冗余边界运算消除（§4.7） | 数个百分点 | `BLOCK_D == D` |
| 10 | CV 指令重排（§4.6） | +1.9%（噪声量级） | 成本≈0，顺手做 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结论 | 证据 |
|---|---|---|
| **多页合并成一个大 tile 摊薄 CV 握手** | **UB 墙，编不过** | NP=2 的三种写法（标量构造行基址 / `tl.cat` / `tl.join`+3D `tl.trans`）**全部** MLIR `PlanMemory Failed`。记账见 G3：基准已占 188KB/192KB |
| **用行基址向量 gather 拼多页** | **NP=1 就只有 0.52x** | `aiv_mte2_ratio` 从 0.081 暴涨到 0.539——gather 形态走不了 Cube 的 GM→L1 直载，被迫由 AIV 搬进 UB 再转交 Cube |
| **PV 错开一轮，打断 Cube→Vector→Cube 依赖** | **1.3791 → 1.2647** | `serial_ratio` 1.738 → 1.707 几乎没动 ⇒ **握手是按 dot 结构性插入的，与数据依赖无关**；多出的 64KB 循环携带量反而挤爆 UB |
| **转置 QK 形态**（`dot(k, qᵀ)` 省掉 `trans(k)`） | **慢一倍** | 11.9 vs 5.78 us/页：转置布局的 q load 是跨步的，且 `axis=0` 归约更贵 |
| **`p_hi/p_lo` 二段拆分提精度** | **性能 -49%** | 循环内 dot 2→3，5087 → 10500 us。精度确实极好（1.03e-5）但本类算子瓶颈是 dot 个数 |
| **PV 段换低精度 dot 提速** | **不提速** | dot 个数没变 ⇒ 时间持平（1.3088 vs 1.3791）。它的价值只在腾 UB（§4.3） |
| **编译参数调优** | **无收益，其中一个有害** | 逐个试 9 个全在 ±1.6% 内；`ops_reorder` **x0.759**；`enable_flatten=False` 编译失败；`enable_mixed_cv` 确认仅 Ascend950 生效 |
| **索引链软件流水**（把 `blk`/`page` 做成循环携带变量，提前一轮预取） | **编译器拒绝** | `'scf.for' op Failed to collect vector loop tiling info` + `'hivm.hir.store' op only support store ub to gm currently!`。⇒ **循环携带的、值来自 GM 标量载入的变量，会让该 `scf.for` 无法做向量 tiling** |
| **两级 gather 拆成独立建表 kernel**（依赖链深度 2→1） | **impl +12%，主 kernel 反而变慢** | 5087 → 5765 us（另加建表 kernel 182 us），`aic_scalar_ratio` 只从 0.5705 降到 0.5440。⇒ 标量占比**不是**索引载入链造成的；原表小、局部性好，换成新开的平铺缓冲反成冷 GM 访问 |
| **把两个小 kernel 融成一个省发射/同步**（前一个的输出是后一个的规约维，可流式累加） | **该段慢 2.6 倍** | 1182.8 → 3035.7 us。融合要求一个 program 独占一行，操作数退化成 `[1,K]`，`tl.dot` 用不了（要求 2D 且 M≥16），只能改写成 `tl.sum(a[:,None]*w, axis=0)`——**等于把 GEMM 从 Cube 搬到 Vector**。⇒ 小 kernel 的耗时先确认是不是发射/同步撑起来的，再谈融合 |
| **`p` 二段拆分（hi+lo）在稠密 attention 段** | **数值成立，UB 不可行** | 通过的 case 无一精度失败，但其余报 `ub overflow, requires 1779712 bits while 1572864 available`（超 13%）：`p`(fp32)/`p_hi`/`p_lo` 三份 `[BLOCK_M,BLOCK_N]` 同时活着 |
| **靠调整表达式顺序 / 就地覆写来省 UB** | **完全无效** | 把 `p_sum` 提前算完再就地覆写 `p` 为残差，源码级活跃 tile 从 3 份降到 2 份，**需求位数逐 bit 不变**（仍是 1779712）。⇒ bishengir 按 tiling basic block 统计 UB，不看源码级活跃期；省 UB 只能缩 tile |
| **`tl.load(mask=m, other=0.0)` 替代 `load(mask=m)` + `tl.where(m,x,0)`** | **不等价，精度崩** | 34/50（13 `AccuracyError` + 3 `AssertionError`，横跨 fp16/bf16/fp32）。⇒ 在 triton_ascend 3.2.2 上被掩掉的 lane **不保证真的写成 0** |
| **掩码恒全真时改走无掩码 load 省掉 `where`** | **9/50** | 报 `NaN 位置不匹配: Framework=42480/63720, Implementation=0/63720`——**torch 标杆自己就产生 NaN，验证要求逐位置复现**；那几行 `where` 连同掩码一起塑造了与标杆一致的 NaN 传播路径，是**承重的**，不是冗余。详见 §8 |
| **K 预转置消除每迭代的 `tl.trans`** | **更慢或编不过** | fp32 档 x0.969；低精度档 `[BDH,BN]` 布局直接 UB overflow |
| **靠 tile 继续压 attention** | **没空间** | 全档扫描后最优档只快 3~8%（§4.2） |
| **堆同组 q head 复用 KV** | **账上不成立** | tile 行数被 UB 钉死：堆 G 个 head 就得把每 head 行数除以 G，任务数等比例上升，**总 KV 访存量完全不变** |
| **利用 token 之间的 topk 页复用** | **账上不成立** | 实测合计页读 18398 / 去重 17310 = **冗余仅 1.06x** |
| **split-K / flash-decoding** | **净亏** | 收益只在 `programs < 核数` 的少数 case，而多一个 merge kernel 会给**每个** case 加一次它的时延 |
| **`tl.where` 数量/位置导致编译崩溃** | **证伪** | 删掉 3 处冗余 `where` 后失败集合**逐个相同** |
| **「tile 某一维不能是 128」** | **证伪** | 旧编译器上 `32x32x128` 就是好的；真判据是 UB 字节数，且上限随编译器版本变（G2） |

### §5.3 三个会误导的 profiling 字段

| 字段 | 直觉解读 | 实际含义 |
|---|---|---|
| `aic_scalar_ratio` / `aiv_scalar_ratio` 高（0.5~0.73） | 标量运算太多 | **同步等待被计入标量管线**。IR 里对应 100+ 个 `set_flag`/`wait_flag`/`pipe_barrier`/`sync_block_*` |
| `cube_utilization` 高（70~93） | Cube 很忙 | 与 `aic_mac_ratio`（0.02~0.15）一起看才有意义——Cube 是**被占用**，不是在算 |
| `scf.for` 计数 | 循环嵌套层数 | Mix kernel 的 IR 里**每层循环有 Cube/Vector 两份**（`hivm.func_core_type` / `hivm.part_of_mix`）。4 个 `scf.for` = 2 层循环 × 2 份 |

---

## §6 精度闸门（先过闸门，再谈性能）

### §6.1 判据

三项 AND：`allclose(atol, rtol)` / `matched_ratio ≥ 0.9` / **`MERE < rel_threshold`**。
`rel_threshold` = fp16 `2^-10`、bf16 `2^-7`、fp32 `2^-13`（fp32 那档比想象的松）。

### §6.2 `p`（softmax 输出）不能直接降到输入 dtype

`p` 舍入到 bf16 的半 ulp `2^-9` 在 `Σp·v / Σp` 上会被放大越阈。两条出路：

- **fp16 中转**（§4.3）：若输入是 bf16 且值域在 fp16 内，`p.to(f16)` + `v.to(f16)` 既保精度又省 UB。**首选**；
- `p_hi/p_lo` 二段拆分：精度最好但**多一个 dot**，在本类算子上是性能负项（§5.2）。

> ⚠️ 「`.to(dtype).to(fp32)` round-trip 会被编译器消除、必须 bitcast+RNE」这条在 bishengir 1.2.0 上
> **没有复现**——最朴素的 `p_hi = p.to(dt); p_lo = (p - p_hi.to(f32)).to(dt)` 即可。

### §6.3 先用 CPU 模拟筛精度方案

不占 NPU、秒级出 50 个 case 的三项判定。多个候选精度方案可以在不写 kernel 的情况下先淘汰掉。

### §6.4 ★★ `gqa-router` 专属：argmax 边距体检（动手前必做）

选路靠 `argmax(routing_logits)`，**一旦重算的 logits 把 top1/top2 翻转，输出直接是另一路的结果**，
不是「误差变大」而是整个 case 崩掉。

⇒ 在 CPU 上把所有 case 的 `top1 - top2` 边距扫一遍。实测（50 case）：

| case | dtype | 绝对边距 | 相对边距 |
|---|---|---|---|
| 最小 | fp32 | **5.77e-4** | 0.0076 |
| 次小 | fp32 | 7.30e-4 | 0.0487 |
| **最险** | **bf16** | 2.26e-3 | 0.0489 |

fp32 那几个安全（fp32 求和顺序差异约 1e-6，还有 ~400x 余量）；
**吃紧的是 bf16**：logits 尺度 0.046、bf16 量子 ≈1.8e-4，边距只有 **12 个量子**。

⇒ **router 必须逐层复刻 golden 的精度语义**：`mean` fp32 累加后回铸 `x.dtype`；
两层 Linear 都 fp32 累加后**回铸 `x.dtype`**，让 argmax 的比较发生在同一量化格点上；
tie-break 取最小下标。
**不要「为了稳一点全程 fp32 不回铸」——那反而会与 torch 分道扬镳。**

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

### §7.1 环境闸门

见 G1。每轮至少三道闸门：**benchmark 全局互斥锁**（本机常是共享的）、
**必须测满全部 case**、**工具链指纹前后对照**。

实测被拦下的两次无效测量：
1. 别的会话在测量中途重装 triton，我方进程从第 14 个 case 起每次新编译都
   `OSError('could not get source code')`，**只测到 13/50 却仍算出一个漂亮的 2.83x**；
2. 跑错工具链（G1 第 1 条）。

### §7.2 ⚠️ benchmark 时延口径已变更（决定要不要拆 kernel）

- **旧版**：`parse_operator_latency` 按 **kernel 名** groupby，取末尾 `repeats` 行除以 `repeats`
  ⇒ 同一个 kernel 在一次 forward 内发射 N 次**只按一次计**。
  在这个口径下「把不同名 kernel 合并成同名」会让测得的时延凭空变好看，**但那不是真加速**。
- **新版**（`fix(triton-op-verifier): benchmark 时延改为按单次调用计算`）：
  由整组行数反推发射次数 `L = len(group) / active_count`，取末尾 `L × active_count` 行求和再除以
  `active_count` ⇒ **一次 forward 内的多次发射如实全部计入**。

⇒ **在新口径下，多发射不再有任何度量红利，按真实 wall-clock 设计即可**；
但**跨版本比较历史数字时必须核对用的是哪一版脚本**。

**实测影响（同一份代码，只换脚本）**：

| 算子 | 旧 fw / impl | 旧 speedup | 新 fw / impl | **新 speedup** |
|---|---|---|---|---|
| `21` (投影 kernel 发 2 次) | 0.1206 / 0.0528 | 2.2839 | 0.1797 / 0.0715 | **2.5119** |
| `33` (3 个 per-branch kernel 各发 N 次) | 0.1877 / 0.0646 | 2.9033 | 0.5559 / 0.1386 | **4.0101** |
| `80` (单 kernel) | — / 0.0476 | 2.2904 | 2.9083 / 0.0488 | **59.5558** |
| `81` (单 kernel) | 0.1044 / 0.0757 | 1.3791 | 3.1449 / 0.0750 | **41.9099** |

两条要点：

1. **impl 侧变化很小**（`81` 0.0757→0.0750，`80` 0.0476→0.0488）；`21`/`33` 涨了，
   正是因为它们**确实**在一次 forward 里多次发射同名 kernel，旧口径漏计了。
2. **变化最大的是 torch 那一侧**。这类算子的 golden 常是
   `for pid_b in range(total_q): for kh in range(KH):` 的双层 Python 循环，
   一次 forward 发射上千个小 kernel，旧口径按名字只计一次，**把 torch 的真实代价低估了约 30 倍**
   （`81`：0.104 → 3.14 ms）。

⇒ **在旧口径下推出的"天花板只有 1.95x"这类结论必须重算**——那套成本模型
（每页 5.81us、dot 个数是成本单位、UB 188/192KB）**约束的是 impl 的绝对时间，与口径无关，仍然成立**；
失效的只是把它换算成 speedup 的那一步。

### §7.2b ★ 共享机器上单点测量一律不可信，且「比值抗污染」不是普遍成立的

共享机器上被别的会话占满时，**绝对时延会整体放大 6~10 倍**（实测 framework 2.91 → 17.08 ms、
impl 0.0488 → 0.3086 ms）。`--lock-frequency` 只挡频率漂移，**挡不住核/带宽争抢**。

很自然会想「取比值就好了」，但比值的抗污染程度**取决于 torch 侧本身有多慢**：

| 算子 | torch framework | 干净 speedup | 重负载 speedup | 变化 |
|---|---|---|---|---|
| `80`（golden 是逐 (b,h,block) 的 Python 循环，发上千个小 kernel） | 2.91 ms | 59.56 | 55.33 | **−7%** |
| `21`（golden 只有几个大 GEMM） | 0.18 ms | 2.5119 | 1.7681 | **−30%** |

⇒ **两条纪律**：
1. **任何版本对比都必须同时段、同一块卡、背靠背连测**，不要拿今天的数和昨天的数比；
2. **开测前 `npu-smi info` 看一眼 AICore 占用**，>50% 就换卡或等。

实测代价：一次把「更慢的版本」误判成更快（v9 看起来 1.6007 > 1.3791，干净重测后
impl 0.0956 vs 0.0750，实为负优化），另一次把绝对时延虚高 10 倍的数据当成回归。

### §7.3 `speedup_vs_torch` 是各 case speedup 的几何平均

小 case 与大 case 等权（G5）。本类算子有一个额外陷阱：
**torch 侧 framework 时延与 case 规模基本无关**（实测恒在 60~130 us），
于是 `speedup ≈ 常数 / impl_us` ——**优化目标退化成「压 kernel 绝对时间」**。

---

## §8 陷阱表

| 陷阱 | 症状 | 处理 |
|---|---|---|
| 非交互 shell 读不到 CANN `set_env` | 无报错，分数系统性偏低一半 | 显式 source + 版本指纹校验（G1） |
| 卡被 aicore timeout 打坏 | 精度结果随机、跨进程持续 | 无关算子对拍 CPU（G1）；换卡，共享机器上不要复位 |
| 运行期标量除 + 向量下标 gather | aicore timeout 507014 | 把该维提升为 tile 坐标（L1.2） |
| fp32 dot + N=64 + 运行期 trip-count | **偶数列** NaN，奇数列垃圾 | N 抬到 128 / 改原生 dtype（L1.5） |
| `tl.trans(x)` 的隐藏 UB 开销 | `PlanMemory Failed` | 记账时把它算成一份与 x 等大的空间（G3） |
| forward 里写 Python `for` | AST 校验器无条件判违规 | 展开成 `if`（G7） |
| `torch.cat` 拼权重 | AST 校验器拒绝 | `torch.empty` + `.copy_()`（L1.6） |
| 相对误差 **0.5~1.5 量级**（不是 1e-3 量级） | 结果整个是错的，不是精度不足 | **不要**去调 dtype/加二段拆分。先用 L1.9 的 CPU 对拍秒级排除权重复刻；权重没问题再查计算侧（head 映射 / 输出布局 / 掩码） |
| **tile 切片 `x[:, :N]` / 切片赋值 `x[a:b, :] = ...`** | `ValueError: unsupported tensor index: slice(...)` 或裸 `AssertionError()` | **首版生成的头号杀手**。发多次 `tl.load` 或一次读完整块，见 L1.10 |
| 循环体内 `continue` | Triton 不支持 | 用 `if` 包住整段（L1.10） |
| **decode 的任务划分套到 prefill 上** | 精度全过但 speedup **< 1**（实测 0.5681 vs 手工 2.5119） | `S_Q > 1` 必须按 `(b, q_head, q_block)` 分块并把 `BLOCK_Q` 开到 UB 上限，见 L1.1 适用范围表 |
| **grid 把 M 维塌掉 / 漏传 `BLOCK_M`** | 精度全过但 speedup **< 2**；逐 kernel 看 GEMM 那段慢 **一个数量级**（实测 75.78us → 1592.84us） | `grid = min(核数, M * cdiv(N,BN))` 是错的，应为 `min(核数, B * cdiv(S,BLOCK_M) * cdiv(N,BLOCK_N))`，且 `BLOCK_M` 必须传进 kernel。**M 维塌掉是本类算子重复出现三次的错误**（`router_fc1` 1D 退化 / prefill 任务划分 / 这里） |
| **host 侧缓冲形状与 kernel 索引不匹配** | 先 `aicore exception`(507015 越界)，修掉崩溃后仍有个别 case **选错分支**（误差 ~1.0） | 实测一例：`xm = torch.empty(B, dq)` 但 kernel 写 `xm_ptr + b*D + d_idx` —— `x.mean(dim=1)` 的形状必须是 **`[B, D]`** 不是 `[B, DQ]`。**每个缓冲都要对着 kernel 里的索引表达式核一遍维度** |
| **对不喂 `tl.dot` 的权重也做预转置** | 相对误差 **~1.0 量级**（与「权重复刻错」症状相同，但权重值是对的） | `router_argmax` 的 `w2` 用向量点积、按 `[NOPT, DQ]` 索引，**不能 `.t()`**。见 L1.6 |
| **打包缓冲用 `torch.empty` 而非 `zeros`** | 三种 dtype **均匀**失败、相对误差 **100 倍**量级（不是某一档偏大） | 各分支宽度不等 ⇒ 窄分支的填充列是未初始化内存，被 `BLOCK_DH` 补齐读取时吃进 dot。打包缓冲一律 `torch.zeros` |
| **`tl.dot` 两侧 dtype 不一致** | `AssertionError('Both operands must be same dtype. Got fp32 and fp16')` | 多半是 `tl.where(mask, x, 0.0)` 的 Python 浮点把一侧提升成 fp32。补零后 `.to(other.dtype)`，见 L1.5 |
| **`tl.dot` 操作数退化成 1D** | `AssertionError('Both inputs must be either 2D or 3D; (lhs: [constexpr[128]] vs rhs: ...)')`，**整批全灭** | 某一维很小（如 router 的 `M = B ≤ 4`）时容易被写成向量。**`tl.dot` 两侧必须都是 2D**：保持 `[BLOCK_M, BLOCK_K]` + `offs_m < M` 掩码，`BLOCK_M` 取 16 |
| **`tl.arange(0, dh)`——用运行期标量当长度** | `ValueError("arange's arguments must be of type tl.constexpr")`，**整批 50 个 case 全灭** | `head_dim` / `d_kv` / `n_kv_heads` 都是运行期值。必须 host 侧算好 `BLOCK_DH = triton.next_power_of_2(head_dim)` 作为 `tl.constexpr` 传入，kernel 内 `tl.arange(0, BLOCK_DH)` + `d_mask = offs_d < DH` 掩码。**两个不同算子独立踩过同一个坑** |
| **`ptr.dtype` 当成元素类型用** | `AssertionError('cannot cast fp32[constexpr[128]] to <[128], pointer<fp16>>')`，**一处写错 50 个 case 全灭** | `ptr.dtype` 是**指针类型**。回铸一律写 `ptr.dtype.element_ty`：`acc.to(out_ptr.dtype.element_ty)` |
| **KV padding 只清零操作数、没掩 scores** | impl 与 framework 同号但整体偏小(比值 0.3~0.7)，`S` 整除 `BLOCK_KV` 的 case 正常 | **首版生成第二杀手**。掩在 scores 上用有限极小值，见 G6 |
| 只在大 case 上扫参 | 结论与最终分数相反 | 大/中/小三档（G5） |
| **把某个算子的结论跨子类外推** | 照搬后性能不升反降（实测 1.78 倍） | 「dot 个数是成本单位」只在握手主导的形态成立，见 L1.3 的反例表。**先 A/B 再定** |
| 拿 ratio 当结论 | framework 漂移会让 impl 更快的版本分数更低 | 同时看 `impl.avg_latency_ms` 绝对值 |
| `topk_idx` 选出超过 `kv_len` 的整页 | Implementation NaN 位置不匹配（softmax 对全 `-inf` 行产出 NaN） | 用 `if blk * BLOCK_BS < kv_len:` 包裹页处理体；并 guard `out = acc / l` 的除零 |
| 读 Triton driver 的 `multiprocessor_count` | `KeyError: 'multiprocessor_count'` | 用 `torch.npu.get_device_properties(dev).cube_core_num` 取核数（本机实测 `cube_core_num=24` / `vector_core_num=48`，与规格一致） |

> 表中最后两行来自 **Kimi 验证跑（`81_GqaSparseDecode`）自己补写的踩坑记录**，非本文档作者原始实测；核数那条已由作者复核（见括注），`topk_idx` 整页越界那条与 §3.1 骨架里的 `pos_f < kv_len_f` 掩码是同一件事的两种写法。

---

## §9 与 `flash_attention.md` 的分工

| 条目 | `flash_attention.md` | 本文件 |
|---|---|---|
| KV 形态 | 连续区间，靠 causal/window 收缩扫描区间 | **按页间接寻址**，页间不连续 ⇒ 区间收缩这条完全不适用 |
| 放大 tile 摊薄同步 | 主力手段（`BLOCK_Q × BLOCK_KV` 开到 UB 上限） | **撞 UB 墙**：基准已占 188/192KB，多页合并三种写法全部编不过（§5.2） |
| 目标函数 | 循环体内**互有 UB 依赖的向量算子数** × KV 迭代数 | 循环体内 **`tl.dot` 的个数** × KV 页迭代数（L1.3，与数据依赖无关） |
| `p` 二段拆分 | 正收益（+2.7%） | **负收益 -49%**（多一个 dot） |
| 任务单元 | 按 (batch, head, q_tile) | **必须按 (token, kv_head)**（L1.1，6.18x） |

两者共通、不要重复踩的：`tl.dot` 契约、UB 预算随编译器变动、`aic_scalar_ratio` 的真实含义、
grid 收缩 + 核内步长循环、投影权重预转置。

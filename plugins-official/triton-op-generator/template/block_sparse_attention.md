---
name: block_sparse_attention
description: 块稀疏 / 掩码跳块类 attention 算子（block-sparse、streaming、paged topK、pooled topK 选块，含前向与反向）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧与 Phase 4 优化点清单
metadata:
  type: reference
---

# 块稀疏 / 掩码跳块类 attention 算子优化经验

本文档是 **"attention 主链 + 一张块级选择表（哪些 KV 块参与本次 softmax）"** 这一类算子的经验合集，覆盖 Phase 2/3/4。

- **§1 通用经验 S3–S7**：跨形态共有的工程建议（S1/S2 已上升为 L1.1/L1.2）
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与 flash_attention.md / attention.md 的分工**

> ⚠️ **本文件与 `flash_attention.md` 的分工**：
> `flash_attention.md` 覆盖"KV 全扫 + online softmax"的稠密 FA 主链，**它的所有结论在本类算子上依然成立**（尤其
> L1.1 `tl.dot` dtype 契约、L1.9 grid 收缩、L1.13 BLOCK 开到 UB 上限）。
> 本文件**只补充"块选择表"带来的那部分**：选择表怎么建、放哪算、kernel 里怎么跳块，以及由此引出的
> 一批稠密 FA 上不会遇到的坑（前缀和、原子归约、越界地址）。
> **两份都要读**；冲突时以本文件为准（本文件的结论都在稀疏形态上实测过）。
>
> **证据基础**：四个块稀疏算子各 50 case 的完整轨迹——
> `BlockSparseAttnFwd`（varlen 打包 + dense/blocksparse/streaming 三类头，**2.73 → 71.63**）、
> `BlockSparseAttnBwd`（同掩码语义的反向，dq/dk/dv 三 kernel，**5.32**）、
> `MsaSparseAttnFwd`（paged KV + CSR 逆推 topK 选块）、
> `SlaFwd`（pooled 打分 + topk 选块 + 并行的线性 attention 分支）、
> `SparseFlashAttentionFwd`（`sp-index-given` 形态；**真实场景 = lightning_indexer topk 随机索引（无重复、升序）**：16 case 随机访存核时比 **0.070 → 0.4110（实际分布）/ 0.4095（压力对照）**；arange 顺序索引只是历史 bench 特例、非常规，见 L1.10~L1.15 / §3.4 / §4.6-4.7 / §6.4）。
>
> ⚠️ **核心优化哲学**：这类算子的头号问题**不是稀疏本身，而是"稀疏被写成了逐 token 的 program"**。
> 掩码是块粒度的（通常 128×128），program 却按 token 发 —— 同一个掩码块里的 128 个 query
> 每人把整块 K/V 搬一遍，访存放大 128 倍，同时 M=1 让 Cube 完全闲置。
> **第一优化动作永远是：把 program 粒度提到掩码块粒度。** 实测 26.25×（几何平均）。

---

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。

| 子类标签 | 入参里出现什么（可模式匹配） | 优化重心 |
|---|---|---|
| `sp-blockmask` | 一个块级 bool/int 掩码张量 + 每头一个类型码（dense / blocksparse / streaming 三选一）+ `causal` 开关；掩码是**块粒度**（如 128×128），不是 token 粒度 | 表用**无 dot 的 Triton kernel** 在 device 侧建；kernel 按压缩索引跳块 |
| `sp-paged-topk` | KV 经 `block_table` 分页寻址，且给的是 **k→q 方向**的 CSR（`indptr`/`indices`），需反推每个 query 的选中块 | 表用**无 dot 的 Triton kernel** 在 device 侧建；⚠️ 不要做跨 program 原子归约（§5.2） |
| `sp-pooled-topk` | 无现成掩码，参考实现先把 Q/K 沿序列维池化成打分矩阵再 `torch.topk(...)` + `scatter_` | top-k 自己实现；⚠️ **紧致化不能用 `tl.cumsum`**（§4.3） |
| `sp-bwd` | 入参含 `softmax_max` / `softmax_sum` / 前向输出，出 `dq`/`dk`/`dv`；掩码语义与前向同一套 | m/l/O 预给 → 不需要 online softmax；需要**转置的选择表** |
| `sp-index-given` | 选择表 **sparse_indices 直接是算子输入**（上游 lightning_indexer 的 top-k 产物，行内下标或 `-1` 填充），KV_N=1 全部 Q 头共享同一稀疏集合；常见 token-wise（sparse_block_size=1） | 无 device 侧建表（bsa-L1.2 不适用）；**随机访存是默认场景，主路径 = gather 连续化暂存**（L1.10/L1.14/§3.4）；gather 路径的 UB 瓶颈是地址张量（L1.11） |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. 入参里有块级掩码 / 块索引 / CSR / topK 之类"哪些 KV 块参与"的结构；
2. 参考实现里出现 `nonzero()` + `torch.cat([k[c*BLK:(c+1)*BLK] for c in cols])` 这种"按选中块拼 K/V"的写法；
3. 参考实现里出现 `torch.topk(...)` 选块 + `scatter_` 成 bool 掩码；
4. 参考实现只写了 `S = S.masked_fill(mask, -inf)`、GEMM 规模没变，**但 `mask` 是块规则的**
   （window / local / strided / 块对角 / sink+local 等能按块判定的规律）。
   ⚠️ 这一条对应索引表**行 8**：`masked_fill` 是 golden 的表达方式，不是生成目标——
   必须改写成本文件的块跳过骨架（先建选择表，再 `for blk in sel`），否则等于白做稀疏。
   判据：能不能只看"块号"就断定整块被丢弃。能，就属于本文件；
   只能逐 token 判定（如按内容动态生成的不规则 mask）才留在 `flash_attention.md`。

### §0.2 ★ 形态识别五问（**Phase 2 第一步必须回答**）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | 选择表怎么建、多大？ | **必须在 device 侧用无 dot 的 Triton kernel 建**（L1.2）。表的形状决定 kernel 的任务划分，不决定放哪 |
| **Q2** | 掩码的块粒度是多少？ | 决定 `BLOCK_M` 的上限。**tile 取到块粒度**掩码行才不用拆（§2.3） |
| **Q3** | 每个 query 的选中块数 topK 与因果可见块数 n_vis 的关系？ | 决定用**逐 tile 扫描**还是**逐 token 压缩表**（§3.2 判据） |
| **Q4** | 参考实现的 attention 在什么精度上算？ | 决定 `p` 是否需要二段拆分（§6.2），以及 fp16 下拆分是否安全 |
| **Q5** | 参考实现里有没有"顺手写出来的布局重解释"？ | paged KV 的 `k[pages].reshape(nb*blk, Hkv, D)` 会把页内两轴换序，**必须逐位复刻**（§8） |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 → 已上升为 **L1.1**（见 §2）

program 粒度必须提到掩码块粒度。这条是"必须这么写"的硬约束，
放在 §1 时 `precheck.json` 不会逐条摘录、Step 3 也没有门禁，因此移入 Layer 1。

### S2 → 已上升为 **L1.2**（见 §2）

索引 / 掩码表必须在 device 侧构建。同上，这是合规红线，必须进 Layer 1 才有门禁。

### S3 掩码合并成一个 `[BM]` 向量，只留一次 2D 比较

```python
# ❌ 三个条件各自广播，UB 里物化 3 个 [BM,BN] int32 临时量
keep = n_valid[None, :] & (offs_n[None, :] <= offs_m[:, None] + (sk - sq))
# ✅ causal 上界与序列边界折叠成同一个 [BM] 向量
lim = tl.minimum(offs_m + (sk - sq), sk - 1)
s = tl.where(offs_n[None, :] <= lim[:, None], s, NEG)
```

这是 `ub overflow` 时第一个要查的地方。

### S4 整行全掩码：用"填 −3e38 + m 下钳 −1e30"，不要第二次 `tl.where`

```python
s = tl.where(keep, s, NEG)                                   # NEG = -3e38
m_new = tl.maximum(tl.maximum(m_i, tl.max(s, 1)), MCLAMP)    # MCLAMP = -1e30
p = tl.exp(s - m_new[:, None])   # 全掩码行: exp(-3e38 + 1e30) 精确为 0
...
o = acc / tl.maximum(l_i, 1.0e-30)[:, None]                  # l==0 的行 acc 也恒 0
```

比"`p = tl.where(keep, p, 0)` + 末尾 `tl.where(l>0, acc/l, 0)`"少两个大张量。

### S5 结构性改动之后必须重扫 tile，**不要查历史清单**

tile 可行域由 UB 预算决定，而 UB 预算**随编译器版本变**。同一份代码实测：

| 编译器 | `--enable-ubuf-saving` | D=128 可编 tile | D=64 可编 tile |
|---|---|---|---|
| bishengir 0.1.0（CANN 8.5.1 自带） | **不支持**，传了直接 `Unknown command line argument` | 只有 32×32 | 到 64×32 |
| bishengir 1.2.0（triton_ascend 3.2.2 **pip 包自带**） | 支持 | **128×128** | **128×128** |

差 16 倍 tile 面积。扫法：跑单 case 编译探针，看
`ub overflow, requires X bits while 1572864 bits available`，按 X 反推砍哪个中间张量。

⚠️ **编译器来自哪里要先确认**：`triton_ascend` 的 `_get_npucompiler_path()` **优先用 pip 包自带的**
`backends/ascend/bishengir/bin/bishengir-compile`，找不到才回落到 `PATH`。
也就是说 **CANN 的 `set_env.sh` 决定 runtime，pip 包决定编译器**，两者可以不同源。

### S6 ★ 不同 tile 之间"每次 K/V 加载服务多少 query 行"是同一个杠杆

`(#tiles) × (每 tile 扫的块数)` 才是访存量。加大 `BLOCK_M` 同时减少 tile 数，
在"每 tile 扫的块数 ≈ 常数"（见 §3.2 判据 B）时是净赚；
在"每 tile 扫的块数 ∝ BLOCK_M"（判据 A）时是白搭。**先判是哪种，再决定要不要加大。**

### S7 任务序按 (kv_head, batch, ...) 排，让同一时刻所有核共享同一份 KV 工作集

`for task in range(pid, total, NUM_CORES)` 的惯用法下，相邻 task 落到不同核**同时**执行。
把 kv_head 放在任务编号的最外层，24 个核就会同时工作在同一个 (batch, kv_head) 上，
该组合的 KV 工作集（`nblocks × 块字节数`，最大 case 16MB）可整体驻留 L2。
反例：`grid=(total_q, Hq)` 相邻 program 落在不同 kv_head，是最差局部性。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ program 粒度必须提到掩码块粒度，**禁止逐 token 发 program**

坏味道：`grid = (total_q, H)` / `grid = (B*S, H)`，kernel 里只处理一行 query。

```python
# ❌ 每个 program 一行 query：K/V 被重复搬 BLOCK_M 倍，且 M=1 时 Cube 闲置
grid = (total_q, H)
s = tl.sum(q[None, :] * k, axis=1)          # 手写点积，Cube 完全没用上

# ✅ 任务粒度 = (batch, q_tile=掩码块粒度, head)
grid = (min(NUM_CORES, total_tasks),)
for task in range(pid, total_tasks, NUM_CORES):
    ...
    s = tl.dot(q, kt) * scale               # [BM, BN]，Cube 正常工作形状
```

**实测：impl 平均时延 2.2833 ms → 0.0868 ms（26.3×），几何平均 speedup 2.73 → 71.63，50 case 无一回退。**
判据：只要 grid 里出现 `total_q` / `seq_len` 这一维且掩码是块粒度的，一定能合并；
合并上限就是掩码块粒度，再大要跨掩码行、得拆表。

### L1.2 ★★★ 索引 / 掩码表**必须在 device 侧构建**，禁止挪到 host

这是本类算子最容易出现的**不合规**写法，务必先看清楚。

```python
# ⛔ 禁止：把选择表搬到 CPU 张量上算完再传回来
keep = torch.zeros(B, NROW, H, NCOL, dtype=torch.bool)      # CPU
... torch.clamp / torch.argsort / Python 循环 ...
blk_idx = keep_compacted.to(device)

# ⛔ 更严重：把参考实现的 mask helper 原样复制过来，物化 token 级稠密掩码
keep_mask = torch.zeros(B, H, max_q, max_k, dtype=torch.bool, device='cpu')
blocked = torch.logical_or(col > torch.minimum(row + sk - sq, ...), ...)   # 抄自 golden

# ✅ 用一个无 tl.dot、无 atomic 的 Triton kernel 在 device 上建表
build_block_table_kernel[(min(NUM_CORES, tasks),)](cu_q, cu_k, head_type, ..., cnt, idx, ...)
```

**为什么不能放 host（三条，缺一不可）**：

1. **索引/掩码准备是算子的一部分。** 参考实现是在 **NPU 上**展开这套掩码的
   （`base_blockmask.repeat_interleave(128,0).repeat_interleave(128,1)`、streaming 的循环），
   framework 侧付了这笔时间；把等价工作挪到 CPU 后 impl 侧一分不付，
   benchmark 的 `operators` 统计里根本看不到它。**这是口径优势，不是真实收益。**
2. **AST 退化预检拦不住它。** `argsort` / `clamp` / `sum` 都不在主链算子名单里，
   放在 helper 里能拿到 `valid=True`。**预检通过 ≠ 合规**，不要拿预检当免责依据。
3. **复制参考实现的 helper 更是直接出局。** 生成代码的职责是"用 Triton 实现算子"，
   不是"把 golden 的 `_head_keep_mask` 抄一份、让 kernel 去读它算好的表"——
   那样 kernel 退化成查表器，算子语义有一大块根本没被实现。

**device 侧建表的标准写法**（99 实测：表 kernel 只占 impl 侧很小一部分）：

- 任务编号**与 attention kernel 完全同构**（同样是 `(batch, q_tile, head)` + 同样的
  `for task in range(pid, total, NUM_CORES)`）。附带好处：表的行粒度天然等于 `BLOCK_M`，
  L1.3 那个"行粒度不一致"的 bug 从结构上不可能再发生。
- 形状（NROW / NCOL / NBLK）**从入参张量的 `.shape` 拿**，不要从 device 读数值回来：
  `base_blockmask` 的 shape 按定义就是 `(B, nblk, ceil(max_q/128), ceil(max_k/128))`。
- 需要"h 之前有几个 blocksparse 头"这类前缀计数时，在 kernel 里
  `tl.sum(((ht_all == 1) & (hidx < h)).to(tl.int32))` 数出来，别在 host 上循环。
- 紧致化用比较矩阵，不要用 `tl.cumsum`（§4.3）：`NCOL` 通常 ≤32，
  `pos = tl.sum(tl.where(cols[:,None] > cols[None,:], keep[None,:], 0), 1)` 代价可忽略。

**host 侧只允许做两件事**：`torch.empty/zeros` 分配输出与中间 buffer；
`.view/.reshape/.transpose/.contiguous` 这类纯布局操作。
出现 `.cpu()` / `.tolist()` / `.item()` / Python 循环建表，一律判不合规。

### L1.3 `tl.dot` 契约：原生 dtype 操作数 + fp32 累加器

```python
# ❌ q/k 升 fp32 —— 输入本就是 bf16/fp16 存储，低精度操作数的乘积在 fp32 累加器里
#    本来就是精确的，升 fp32 一位有效数字都不增加，却让 UB 流量翻倍、Cube 走 fp32 通路
s = tl.dot(q.to(tl.float32), k.to(tl.float32))
# ✅
s = tl.dot(q, kt)
```
反向里 `q/k/v/dout` 同理（实测某反向实现四个量全部 `.to(tl.float32)` 后再 dot，是明确的浪费）。

### L1.4 ★ `p`（softmax 输出）不能直接降到输入 dtype，用 hi+lo 二段拆分

`p` 是真 fp32 量，`p.to(bf16)` 的 2^-9 相对误差在 PV 里被 `Σp|v| / |Σpv|` **放大约 20 倍**
（实测 rel 2~4%，bf16 阈值才 7.8e-3）。

```python
ph = p.to(DTYPE)
pl = (p - ph.to(tl.float32)).to(DTYPE)
pv = tl.dot(ph, v) + tl.dot(pl, v)      # 代价：PV 的 Cube 工作量 ×2
```
修完 MERE 从 0.43 → 1e-6、`max_abs_diff` 3.9e-3（正好是最终 bf16 舍入本身）。

⚠️ **fp16 上二段拆分不是无条件安全的**：实测某反向算子的 `P` 取值可低到 **4e-21**，
远低于 fp16 最小非规格化数（约 6e-8），`ph = P.to(fp16)` 直接变 0、`pl` 随之失真。
**拆分前必须先看该量的动态范围**；bf16 指数位多（最小非规格化数约 9e-41）不受影响。

### L1.5 ★ 选择表的行粒度必须与 kernel 的 `BLOCK_M` 严格一致

最容易犯、且**错误形态会伪装成精度问题**的一个 bug：
表按掩码块粒度（128）建行，kernel 里 `m0 = qb * BLOCK_M`（BLOCK_M=64），
于是每个 128 行组只有第一个 tile 被算，其余行留在 `torch.empty` 的**未初始化内存**里。

实测表现：`matched_ratio=0.558 / max_abs_diff=5.0 / MERE=0.43` + NaN 位置断言同时触发——
看起来完全像"精度不够"，实际是覆盖不全。
**判据：只要输出用 `torch.empty` 且不是被 kernel 全覆盖，先怀疑覆盖率，不要先怀疑精度。**

### L1.6 ★ masked load/store 仍然会算出越界地址

```python
# ❌ GP = max(16, next_power_of_2(G)) 时 (hk*G + offs_g) 会越过 Hq，
#    即使 mask=gvalid 也会先算出越界地址；实测直接产生 NaN（GP=16 崩、GP=4 正常）
q = tl.load(q_ptr + t * (Hq*D) + (hk*G + offs_g)[:, None]*D + offs_d[None, :], mask=gvalid[:, None])
# ✅ 行号先钳在界内，再靠 mask 丢弃无效行
hqs = tl.minimum(hk * G + offs_g, Hq - 1)
q = tl.load(q_ptr + t * (Hq*D) + hqs[:, None]*D + offs_d[None, :], mask=gvalid[:, None], other=0.0)
```

### L1.7 ⛔ 禁止把参考实现的 helper 原样复制进生成代码

判据：生成代码里出现与 golden 逐行同构的函数（典型是 `_head_keep_mask` /
`_streaming_keep_mask` 这类掩码构造），无论它跑在 CPU 还是 device 上，都判不合规。
掩码语义要用 Triton 重新实现成**块级选择表**，而不是物化 token 级稠密 mask 再喂给 kernel。

### L1.8 Grid 收缩到 `min(核数, tasks)` + 核内步长循环，且 grid 与步长必须一致

`NUM_CORES` 作为 constexpr 传进 kernel 时**必须与 grid 完全一致**，
不一致会**静默算少或算重**，不会报错。

### L1.9 kernel 内不能读非 constexpr 的模块级全局变量

`NEG = -3.0e38` 要写成 `NEG: tl.constexpr = -3.0e38`，否则报
`Cannot access global variable ... from within @jit'ed function`。

### L1.10 ★★★ 稀疏索引的主路径必须按「随机访存」设计——顺序（arange）索引不是常规场景

- **必须** 把 gather（慢）路径当作**主路径**设计：真实选择表来自上游 topk 算子
  （下标无重复、升序、分布随机），逐行随机 gather 是常态，不是边角 case。
- **禁止** 用 arange 连续索引的 bench 结果代表算子性能——"索引恰好连续"只是特例；
  专项 bench 必须用随机索引（真实分布生成法见 §7 的 `randtopk_sorted`）。
- **主路径结构 = L1.14 的「gather 连续化暂存」**（骨架见 §3.4）：
  unmasked gather→store（满带宽）→ per-core scratch → 全仿射 flash 消费。
- **历史注记**：早期版本曾按「检测连续段 → 仿射快路径」做双路径分派
  （arange bench 核时比 0.585→0.836）；该模式对真实随机索引零收益，
  属于机会性优化而非常规主线，已从 skill 主路径删除（生产 kernel 中保留该分支
  仅作兼容兜底，新生成的同类算子**不必**复制此模式）。
- **Why（实测）**：仿射/连续路径的 UB 预算与 gather 路径是两套——路径一变，
  **BLOCK 必须重扫**，旧 tile 结论作废（见 L1.11 的阶梯表）。

### L1.11 ★★ gather 的隐藏 UB 大项是 `[BN,D]` int32 地址张量 → D 拆半或仿射化

- **必须** 对 gather 类 kernel 的 ub overflow，先用消融定位主因
  （删掉一个 gather 负载再编译，读 overflow 数字的变化量），再决定砍什么。
- **Why（实测）**：朴素 tile 估算只到 ~122KB，实际 overflow 需要 225~282KB——
  差额就是按行间接寻址物化的 int32 地址张量（与 BN×D 成正比）。把 D 拆半（2×256）
  加载 k_v/q、QK 变 3 个 dot（lo+hi+rope）、acc 拆 lo/hi、末尾两段 store 后，
  BLOCK_KV 32 → 64，收益 = KV 迭代数减半 ≈ **+59%**（31.57 → 49.25）。
  D 拆半纯 tiling、算术顺序不变，数值与单段一致。
- **实测阶梯**（单段 D、multibuffer=False）：gather BN=32 ✓ / 40/48/64 ✗ overflow；
  gather+D 拆半 BN=64 ✓、96/128 ✗；**仿射 BN=128 ✓、256 ✓**——地址张量一消失，UB 墙整体平移。
  ⚠️ 仿射 BN=256 + multibuffer=True 会顶到 **L1 上限**（requires 5013504/4194304 bits = 512KB L1，
  与 UB 192KB 是两套容量），multibuffer 保持关闭。

### L1.12 ★ fp32 化的 load 掩码 + 第二处 masked load 组合禁止（bishengir 非确定性缺陷）

- **禁止** 同时出现：① `tl.arange` 派生的 fp32 化掩码（如 `valid_h = h.to(tl.float32) < QN`）
  用于一个 masked load；② 同 kernel 里另一个带 mask 的 load（如 gather 的 `mask=kv_valid`）。
- **Why（复核实测，最小复现逐级消融）**：
  - 单要素均不复现：纯 load/store（A 级）、单任务 dot 循环（B 级）、
    fp32 掩码 + 无 mask 的 gather（C3 级）——全部确定性；
  - **组合才复现**：fp32 掩码 + 带 mask 的 gather load（C3+maskedKV / C4 级）→
    同输入 4 次运行 self-diff = 5.12e+02、偶发 NaN；
  - int32 掩码在任意组合下 self-diff = 0（A~C4 级全部确定性）；
  - `tl.load` 读入值（sel）的 fp32 化比较安全（v18 实证）。
- **How to apply**：与 latency-optimizer checklist「比较转 fp32」冲突时以正确性优先，
  掩码比较保持 int32 并在注释标注；判别信号 = verify 同 case 时过时挂 / self-diff 非零。
- **验证方法**：判据 = 目标 kernel 同输入连续运行 4 次、两两 self-diff 必须为 0；
  非零时逐级消融（去掉第二处 masked load / 改回 int32 掩码）确认诱因。核心复现骨架：

```python
@triton.jit
def mask_probe(q_ptr, kv_ptr, idx_ptr, out_ptr, BQS, K, KVS, QN,
               USE_FP32: tl.constexpr, NUM_CORES: tl.constexpr,
               BLOCK_KV: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    for t in range(pid, BQS, NUM_CORES):            # 多任务步长循环（复现需要该结构）
        h = tl.arange(0, 16)
        d = tl.arange(0, D)
        if USE_FP32:
            valid_h = h.to(tl.float32) < QN        # 变体 A: fp32 化 arange 掩码（复现缺陷）
        else:
            valid_h = h < QN                        # 变体 B: int32 掩码（确定性）
        hc = tl.minimum(h, QN - 1)
        q = tl.load(q_ptr + t * QN * D + hc[:, None] * D + d[None, :],
                    mask=valid_h[:, None], other=0.0).to(tl.float16)
        q_t = tl.trans(q)
        acc = tl.zeros((16, D), tl.float32)
        for k0 in range(0, K, BLOCK_KV):
            k_offs = k0 + tl.arange(0, BLOCK_KV)
            sel = tl.load(idx_ptr + t * K + k0 + tl.arange(0, BLOCK_KV))
            kv_ok = (sel >= 0) & (sel < KVS) & (k_offs < K)
            selc = tl.minimum(tl.maximum(sel, 0), KVS - 1)
            k_v = tl.load(kv_ptr + selc[:, None] * D + d[None, :],
                          mask=kv_ok[:, None], other=0.0).to(tl.float16)  # 第二处 masked load
            s = tl.trans(tl.dot(k_v, q_t))          # [16, BN] fp32
            p = tl.exp(s - tl.max(s, axis=1)[:, None]).to(tl.float16)
            acc = acc + tl.dot(p, k_v)
        tl.store(out_ptr + t * QN * D + h[:, None] * D + d[None, :],
                 acc.to(tl.float16), mask=valid_h[:, None])
```

每变体跑 4 次取 pairwise max self-diff：变体 A 非零（实测 2.04e+01~2.21e+01，单任务 grid=1 时不复现）、
变体 B 为 0 即复现；逐级消融（去掉 `k_v` 的 mask / 去掉 rope dot / 改回 int32）确认诱因。

### L1.13 双路径分派下，两分支的局部变量禁止同名

- **必须** FAST/SLOW 两分支的局部变量分别命名（如 `f_`/`s_` 前缀），只共享分支前定义的值。
- **Why（实测）**：同名不同形的变量（`k_offs` [128] vs [32]）触发
  `AssertionError('Mismatched type for k_offs between then block and else block')`，
  一眼像语义错误实则命名冲突。
- **How to apply**：双路径写完先做一次空 shape 编译冒烟，再进 verify。


### L1.14 ★★★ unmasked 2D gather 直接喂 `tl.dot` 产出错值 → 慢路径用「连续化暂存」结构

- **禁止** 把无 mask 的 2D 按行间接寻址 `tl.load`（gather）的结果直接喂 `tl.dot`：
  本编译器版本（bishengir 0.1.0 系）该 lowering **编译能过但数值是垃圾**
  （输出出现 ±512 级原始值；最小复现：gather→store 数值正确、gather→dot 报错/错值，
  极简 gather+dot 还会编译失败 `Unknown core type: llvm.func @malloc`）。
- **masked gather 喂 dot 正确**（bsa-L1.6 clamp + score 层 where 屏蔽语义不变），
  但每迭代有 ~12us 固定开销（与 BN 无关，BN 32→64 总耗时严格减半）。
- **为什么可以绕（实测）**：纯 gather→store 拷贝 kernel（无 dot）unmasked 数值**完全正确**
  且跑满带宽 ~1.4TB/s（BN=128, 含写）；把 gather 地址换成仿射地址耗时**不变**——
  即离散访存本身不是瓶颈，瓶颈是「masked gather 喂 dot」这条 lowering 路径。
- **How to apply（SLOW 路径骨架，实测随机访存 vs CANN 0.138→0.4075）**：
  ```python
  # phase1: 每核一块全局 scratch (NUM_CORES*K*(D+DR) fp16, 可驻留 L2)
  for k0 in range(0, K, STAGE_BN):            # STAGE_BN=64
      selc = clamp(load(idx_row + k0 + s_offs, mask=tail), 0, KVS-1)
      store(scr_v + (k0+s_offs)[:,None]*D + d, load(kv + (base+selc)[:,None]*D + d))  # unmasked
      store(scr_r + (k0+s_offs)[:,None]*DR + dr, load(kr + (base+selc)[:,None]*DR + dr))
  tl.debug_barrier()                            # store->load 可见性
  # phase2: 与 FAST 完全同构的 affine flash (CONSUME_BN=256, 含 pl 二段拆分)
  for k0 in range(0, K, CONSUME_BN):            # 从 scratch 仿射载入 + 重读 idx 重建 kv_valid
      ...
  ```
- **代价**：多读+多写各一遍（约 +2/3 流量），换来 dot 路径全仿射化；scratch 用
  `torch.empty` per-call 分配（caching allocator 摊销）。
- **simulator 证据（case16 瓶颈定位）**：stage 相 gather 载入占 vector 核 MTE2 的 41%cycles；逐行小事务（1KB/行）的 MTE2 发起开销主导，cube MMAD≈0 空等 40%——小 K case 的 stage:consume 迭代比越大越吃亏；不要在此类瓶颈上试 multibuffer/ping-pong（均实测证伪），那是流水问题而本瓶颈是事务指令流。

### L1.15 ★★ 快慢双路径必须留在**同一 kernel** 的 if/else 分支内，禁止拆成两个独立 kernel

- **Why（实测证伪）**：把 FAST/SLOW 拆成两个 kernel（各自按 flag 跳过对方任务、慢 kernel
  获得独立 UB 预算可开 BN=64）——随机路径如期 2×，但**同一文本的 FAST 代码独立成 kernel
  后从 6.9ms 退化到 10.1ms**（arange 16 case 核时比 0.8362→0.6044）；逐 kernel profile
  确认退化全部发生在 fast kernel 本体（与 flag 检查、early-exit、launch 开销均无关，
  已逐项消融）。合并回单 kernel if/else 后 FAST 恢复 6.87ms（0.8390）。
  编译器对「组合 kernel 内分支」与「独立 kernel」生成不同调度，方向无法预测。
- **How to apply**：双路径分派一律单 kernel `if flag==0: FAST else: SLOW`；
  需要改 UB 预算时只能在分支内调 BLOCK/结构（如 SLOW 的 STAGE_BN/CONSUME_BN），
  不得以"独立预算"为由拆 kernel。

### L1.16 ★★ 随机访存为主场景时：**删掉 FAST 分支本身**比保留双路径更快（-4.4%）

- **场景前提**：评测/生产输入是 lightning_indexer topk 随机索引（`sp-index-given` 形态），
  arange 连续索引不在基线内。
- **实测（910B3, case16 B=64/QS=64/K=512, msprof 核时）**：
  - 双路径组合 kernel（`if flag==0: FAST else: SLOW` + contig_check 建 flag）：
    7.6404ms = check 0.157 + sfa 7.483（前序会话归档产物, 与本次单路径同脚本同设备口径）；
  - **同 SLOW 代码文本、仅删 FAST 分支与 contig_check（单路径 slow-only）**：
    **7.1966ms（-5.8%）**，随机 16 case 几何平均 0.4110 → **0.4228**（vs CANN）。
  - 反事实对照（同会话内最可靠对照）：把 flag 强制置 1 走双路径 kernel 的 SLOW，7.53ms；单路径 7.20ms。
- **Why**：同 L1.15 —— 本编译器（bishengir 0.1.0 系）对分支组合的调度方向不可预测；
  分支存在本身即改变 SLOW 路径的指令调度（不是 flag 检查的运行时开销）。
- **How to apply**：当评测/部署输入确定为随机 topk 索引时，**不要**为机会性的 arange
  快路径保留 `contig_check` + 双分支；直接交付单路径通用 gather kernel（对任意索引
  语义正确，代价仅是 arange 场景失去仿射快路径优化）。若必须兼顾 arange 基线，
  则按 L1.15 保留同 kernel 双分支，不要拆 kernel。
- **顺带实测（同场景否决清单）**：PV 单段（去 pl）MERE 1.4e-3 超 2^-10 阈值（max_abs
  仅 6e-5，是 MERE 卡阈值）；softmax 改在 [BN,16] 上做 axis=0 归约 + 转置 f16
  4.66→9.87ms；p 存 scratch 转置往返 6.72ms；kr 独立循环 / 更大 BLOCK 7.69~7.89ms；
  任务级双缓冲流水 7.44ms（均不优于现行 stage/consume 交错结构）。

---

## §3 Layer 2: 算法骨架

### §3.1 主骨架（前向）

```
kernel 0（无 dot、无 atomic，grid 与主 kernel 同构）：
    建选择表 -> 压缩成 (cnt[task], idx[task, j])   # 必须在 device 侧
kernel（grid = min(核数, tasks)）：
    for task in range(pid, tasks, NUM_CORES):
        解出 (batch, q_tile, head)；越界 tile 用 if 守卫跳过
        载 q tile [BM, D]
        lim = min(offs_m + (sk - sq), sk - 1)          # causal 与边界折叠
        for j in range(cnt[task]):                     # ★ 循环体内无数据依赖分支
            c  = idx[task, j]
            kt = load [D, BN];  s = tl.dot(q, kt) * scale
            s  = tl.where(offs_n[None,:] <= lim[:,None], s, NEG)
            m_new = max(max(m_i, max(s,1)), MCLAMP)
            p  = exp(s - m_new[:,None])
            v  = load [BN, D]                          # ★ dot(p,v) 提前到 alpha/p_sum 之前
            pv = dot(ph, v) + dot(pl, v)
            alpha = exp(m_i - m_new)
            acc = acc*alpha[:,None] + pv;  l_i = l_i*alpha + sum(p,1);  m_i = m_new
        o = acc / max(l_i, 1e-30)[:,None]
```

### §3.2 ★★★ 逐 tile 扫描 vs 逐 token 压缩表 —— 判据

设 `n_vis` = 该 q-tile 的因果可见块数，`topK` = 单 token 选中块数，`BM` = tile 行数。
一个 tile 要扫的块数 ≈ **`min(topK × BM, n_vis)`**（并集大小）。

- **判据 A：`topK × BM ≥ n_vis`** ⇒ 并集≈全部可见块，**扫描与压缩表等价**，
  此时**只能靠加大 BM 摊薄**（每 token 摊到 `n_vis / BM` 块）。
- **判据 B：`topK × BM < n_vis`** ⇒ 压缩表严格更省，用逐 token 的 topK 列表。

⚠️ **两个方向都实测过，都不是万灵药**：
- 逐 tile 扫描（BM=64）在某算子最大 case 上 1386 ms；
- 换成逐 token 压缩表（每 token 只扫自己的 topK 块）后 **954 ms → 1327 ms 全面变差**，
  因为任务数从 3072 暴涨到 98304，tile 缩到 `[G, D]`（G=2），
  **每任务的标量前导开销直接占满**。
⇒ **访存量不是唯一目标函数；tile 太小的代价会吃掉全部访存收益。**

### §3.3 反向的三 kernel 结构

m / l / O 都是**给定输入**，所以不需要 online softmax：

```
S = (q*scale) @ kᵀ  (掩码位置填 -3e38)
P = exp(S - softmax_max) / softmax_sum          # 全掩码行 max=0/sum=1 ⇒ P 自然为 0，不必特判 NaN
delta = rowsum(dout ⊙ attention_in)
dP = dout @ vᵀ ;  dS = P * (dP - delta)
dV = Pᵀ @ dout ;  dQ = dS @ k * scale ;  dK = dSᵀ @ q * scale
```
dq / dk / dv 三个 kernel 会**各自重算一遍 S 和 P**（实测占 impl 侧 ~95%）：
dq 与 dk 可以融合成一个 kernel（dS 只算一次），dv 只需要 P。
dk/dv 的 kernel 需要**转置的选择表**（每个 (b, k_block, h) 对应哪些 q_block）。
⚠️ 它同样受 L1.2 约束：**必须在 device 侧再建一个 kernel 把它算出来**，不能在 host 上
把 q 主序表转置一遍。做法与 q 主序表完全对称——同一份块级判据 `_blk_keep`，
只是把 `(b, q_tile, h)` 的任务分解换成 `(b, k_tile, h)`，判据本身一行都不用改；
两张表各发一个 kernel，合计几十 µs 量级。
### §3.4 随机访存主路径的「gather 连续化暂存」骨架（L1.14 的落地形态）

动机（L1.14 实测）：unmasked gather 直喂 `tl.dot` 产出错值、masked gather 直喂 dot
每迭代 ~12us 固定开销；但 unmasked gather→store 拷贝路径**正确且满带宽 ~1.4TB/s**。
故把"选中行"先连续化到 per-core 全局 scratch，再以全仿射结构消费——用 +2/3 流量
换 dot 路径全仿射化：

```python
# 每 program 处理一个 (b, s1) 任务 (bsa-L1.1); scratch = torch.empty(NUM_CORES*K*(D+DR), fp16)
scr_v = scratch_ptr + pid * K * D
scr_r = scratch_ptr + NUM_CORES * K * D + pid * K * DR
for k0 in range(0, K, STAGE_BN):                 # 阶段1: gather→store 连续化, STAGE_BN=64
    sel = tl.load(idx_row + k0 + s_offs, mask=(k0 + s_offs) < K, other=0)
    selc = tl.minimum(tl.maximum(sel, 0), KVS - 1)          # L1.6: clamp 保证地址在界
    tl.store(scr_v + (k0+s_offs)[:, None]*D + d[None, :],
             tl.load(kv_ptr + (b*KVS + selc)[:, None]*D + d[None, :]))        # 无 mask 拷贝路径
    tl.store(scr_r + (k0+s_offs)[:, None]*DR + dr[None, :],
             tl.load(kr_ptr + (b*KVS + selc)[:, None]*DR + dr[None, :]))
tl.debug_barrier()                               # 同核 store→load 可见性(实测零成本)
for k0 in range(0, K, CONSUME_BN):               # 阶段2: 仿射 flash 消费, CONSUME_BN=256
    k_v = tl.load(scr_v + (k0+c_offs)[:, None]*D + d[None, :])      # 仿射, 大描述符 DMA
    k_r = tl.load(scr_r + (k0+c_offs)[:, None]*DR + dr[None, :])
    sel2 = tl.load(idx_row + k0 + c_offs, ...)   # 重读 idx 重建 kv_valid, score 层屏蔽无效行
    ... online softmax + PV(hi+lo 二段拆分, fa-L1.5 不可省)  # 与任意 flash 骨架同构
```

实测：随机访存 16 case 核时比 vs CANN **0.070 → 0.4110（实际分布）/ 0.4095（压力对照）**
（5.9× 于首版 gather 直喂）；瓶颈残留 = gather 逐行小事务的 MTE2 发起开销
（simulator 实测 stage 相占 vector 核 68~86%，小 K case 最敏感，结构性下限）。

---

## §4 Layer 3: 关键技巧

### §4.1 CV 指令重排：`tl.dot(p, v)` 提前到 `alpha` / `p_sum` 之前发射

纯重排、数值等价。成本极低值得做；但**不要期望它打开 CV 重叠**——
实测重排后串行度 `(aic+aiv)/Duration` 仍是 1.66（>1.5 即串行）。

### §4.2 ★ paged KV 的"布局重解释"必须逐位复刻，复刻完再预转置

参考实现里 `k[pages].reshape(nblocks*blk_kv, Hkv, D)` 会把页内的 `(Hkv, blk)` 两轴
**当成 `(blk, Hkv)` 重新解释**：页内第 i 个 token、第 h 个 kv-head 的元素落在偏移 `(i*Hkv + h)*D`。
这不是"打乱"，是行距 `Hkv*D` 的规则 2D 访问，**无需 gather**，但每行只有 `D*2` 字节连续。

```python
# 复刻完之后，Hkv > 1 时预转置一次，让同一块的 blk 行变成完整连续
kk = k.view(P, BLK, Hkv, D).transpose(1, 2).contiguous()   # 只是布局操作，不是主链计算
```

### §4.3 ★★★ `tl.cumsum` 在 Ascend 上极慢，前缀和用 `tl.dot × 下三角矩阵` 替代

**实测（同一 kernel 形状，[11392, 2048]）：带一次 `tl.cumsum` 14.33 ms，去掉后 0.254 ms —— 56×。**
慢到会**触发 Vector core 看门狗**（`507035 / The vector core execution is abnormal`，
`aivec error ... subErrType:4`），在正式 benchmark 里表现为跑到某个 case 直接崩。

⚠️ **分层 cumsum 无效**：把 `[BR, NP2]` 拆成 `[BR, NCH, CH]` 做两级（CH=64/128/256）
实测 15.8 / 15.1 / 14.7 ms，**全部不比扁平的 14.3 ms 快** ⇒
代价与"参与前缀和的元素总数"成正比，不是与轴长成正比，**拆轴没有意义**。

正解——把前缀和写成矩阵乘，丢给 Cube：

```python
# inclusive_prefix(x)[i] = Σ_{j<=i} x[j] = (x @ U)[i],  U[j,i] = 1 if j <= i
c = tl.arange(0, CH)
U = (c[:, None] <= c[None, :]).to(tl.float16)     # 下三角(含对角)全 1，CH<=128
# 行按 [NCH, CH] 二维摆开：块内前缀一次 [NCH,CH]x[CH,CH] 的 Cube 乘法
inner = tl.dot(x, U)
# 块间前缀用 [NCH,NCH] 的比较+归约（NCH<=16，可忽略）
tot   = tl.sum(x, 1)
pref  = tl.sum(tl.where(a[:, None] > a[None, :], tot[None, :], 0.0), 1)
pos   = inner + pref[:, None]
```
计数值 ≤ CH ≤ 128，**fp16 精确可表示**，累加器是 fp32 ⇒ 结果逐位正确。

### §4.4 ★★★ top-k 选块：必须逐位复刻参考实现的选择集合（照抄下面的配方）

> 注意与 L1.7 的区别：这里要求复刻的是**选择语义**（哪些块被选中），配方写在本文档里，
> 照着自己实现即可；L1.7 禁止的是把参考实现的 helper **源码**原样搬进生成代码。

**这一节是 `sp-pooled-topk` 形态最容易翻车的地方。** 实测：一个只差了"打分张量有没有
先 round 回输入 dtype"的实现，`verify` 直接 0/50，形态是 `matched_ratio` 从 0.9+
断崖到 **0.76 / 0.31**，而 `max_error_cap` 反而是 `True` —— 看起来像"精度略差"，
实际是**选错了块**。选块是离散的：差一个块 = 整行结果完全不同。

`torch.topk(pooled.float(), rtk, -1, sorted=False)` 的下游是 `scatter_` 成 bool 掩码，
所以**只要选中集合一致即可，顺序无关**。要让集合一致，四件事缺一不可：

**(1) 打分张量必须先 round 回输入 dtype，再喂给 top-k。**
参考实现写的是 `pooled = (qm.float() @ km.float().T).to(dtype)`，然后
`torch.topk(pooled.float(), ...)` —— 也就是说进 top-k 的值是
**fp32(dtype(fp32 结果))**，不是 fp32 结果本身。
```python
# ❌ 直接拿 fp32 的矩阵乘结果选块 —— 排序会和参考实现不同
# ✅ 先按参考实现 round 到 dtype 存下来，kernel 里再 .to(tl.float32) 读回
```
bf16 只有 8 bit 尾数、fp16 只有 11 bit，round 之后**大量并列值会被制造出来**，
这正是 (3) 变得关键的原因。

**(2) `real_topk` 的取法照抄**：`real_topk = min(NK, int(topk * NK))`（`int()` 是截断）。

**(3) float → 单调 int32 key，然后对 key 二分阈值。**
```python
i   = v.to(tl.int32, bitcast=True)
key = (i ^ ((i >> 31) & 0x7FFFFFFF)) >> SHIFT   # 正数保序，负数翻转低 31 位
```
`SHIFT` 按 dtype 取（fp16 尾数少，可右移更多），二分 `NITER = 33 - SHIFT` 轮。

**(4) 并列规则：`> thr` 全取，`== thr` 的按列号从小到大补足到 `real_topk`。**
`aclnnTopk` 并列时**取小索引** —— 17 个 case 逐个比对确认：
「取小索引」0 处不一致，「取大索引」8 个 case 有 1~12 行不一致。

**(5) ⚠️ `-0.0` 必须归一到 `+0.0`，且要放在 bitcast 之前。**
位序 key 把 `-0.0` 排在 `+0.0` 之前，而 `torch.topk` 认为两者相等。
打分张量里 fp32 的极小负值 round 到低精度就会变成 `-0.0`，是真实可达的分歧点。
CPU 穷举 3120 组随机行（651 组含 `-0.0`）：不修 **179 组不一致**，修完 **0 组**。
```python
v = tl.load(pooled_ptr + ..., mask=m, other=NEG).to(tl.float32)
v = tl.where(m, v, NEG)
v = tl.where(v == 0.0, 0.0, v)      # ← 必须在 bitcast 之前
i = v.to(tl.int32, bitcast=True)
```

**(6) 写完先单独自检选块集合，不要直接跑 50 case 的 verify。**
用几组随机 `pooled` 把 kernel 选出的列集合和 `torch.topk(...).indices` 的集合逐行比：
```python
got = set(lut[r][lut[r] >= 0].tolist())
exp = set(torch.topk(pooled[r].float(), rtk).indices.tolist())
assert got == exp
```
集合对不上就先修集合，**不要去调 tiling 或精度** —— 那是完全不同的病。

### §4.5 二分循环里禁止向量↔标量往返

```python
# ❌ 每轮把归约结果落成 0-d 标量再 if/where：每行 NITER 次同步往返，被延迟支配
c = tl.sum(...)            # 0-d
if c >= RTK: lo = mid
# ✅ 一次处理 BR 行，lo/hi/c/take 全是 [BR] 向量
c = tl.sum((key > mid[:, None]).to(tl.int32), 1)
take = c >= RTK
lo = tl.where(take, mid, lo);  hi = tl.where(take, hi, mid)
```

---

### §4.6 Q 转置 dot 形式省每迭代转置缓冲

```python
# ✅ 每迭代只转 [BN,16] 的小块; k_v 行主序加载同时复用为 PV 的 B 操作数
s_t = tl.dot(k_v, q_nope_t)      # [BN,512] @ [512,16] -> [BN,16]
s   = tl.trans(s_t)              # [16,BN]
# ❌ 每迭代物化 [512,BN] 的转置缓冲（BN=64 时 64KB, 是 UB overflow 主因之一）
s = tl.dot(q_nope, tl.trans(k_v))
```
q 的转置在 KV 循环外 hoist（每 program 仅 2 次），代价几乎为零。

### §4.7 「转置直接加载」比「自然布局加载 + 寄存器 trans」更费 UB

q 按 `[D,16]` 布局直接加载（列=头、行=d 连续）看似省一次 trans，实测单段 D BN=32
时反而 +24KB 溢出——两者 UB 行为与直觉相反，**只有实测能定**；默认用
「自然布局加载 + `tl.trans`」，UB 溢出时再试另一侧。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）


| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ program 粒度提到掩码块粒度 + 全面 `tl.dot` 化 | **26.25×**（几何平均） | grid 里有 `total_q` 这一维 |
| 2 | ★★ 索引/掩码表改用**无 dot 的 Triton kernel** 在 device 侧建 | 去掉 impl 侧全部 `aclnnSort`/`aclnnRepeatInterleave`（**不是**把它们挪到 CPU） | 参考实现用 torch 反推索引时 |
| 3 | ★★ `tl.cumsum` → `tl.dot × 下三角` | **56×**（前缀和这一步） | 有紧致化 / rank 计算 |
| 4 | ★★ tile 取到掩码块粒度（128×128） | 掩码行不用拆，K/V 复用最大 | 受 UB 约束，**必须实测扫** |
| 5 | ★ 掩码合并成 `[BM]` 向量 | UB 省 3 个 `[BM,BN]` int32 临时量 | 编不过报 `ub overflow` 时先查 |
| 6 | ★ 整行全掩码用 −3e38 + m 下钳 | 省一个 `[BM,BN]` 与一个 `[BM,D]` 张量 | 总是 |
| 7 | ★ paged KV 预转置成块内连续 | MTE2 从 256B 粒度变整块搬运 | `Hkv > 1` |
| 8 | ★ 任务序 kv_head 最外层 | KV 工作集驻 L2 | 有 GQA / paged KV |
| 9 | `p` 二段拆分 | 精度必需（不是性能） | bf16 输入；fp16 先查动态范围 |
| 10 | CV 指令重排（dot 提前） | 成本极低，收益不可预期 | 总是可以试 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱**）

| 方向 | 结果 |
|---|---|
| **k-centric（沿 CSR 行遍历）+ `tl.atomic_*` 跨 program 归约** | ⛔ **工具链不支持**。kernel 里只要出现 `tl.atomic_*`，凡是操作数由本 kernel 内 Vector 运算（`exp`/`.to()`）产生的 `tl.dot` **就静默返回全 0**；两个操作数都来自 `tl.load` 时正常。表征是编译期 WARNING `AutoBlockify disabled ... Unsafe ops: atomic operations`。⇒ **split-K / flash-decoding 那种把一行 softmax 拆到多 program 再原子归约的设计在 Ascend Triton 上暂时不可用**，必须设计成单 program 内闭合 |
| ↑ 的所有绕法 | 全部失败：`tl.debug_barrier()` 无效；`has_auto_blockify_blacklist_op=False`（编译器 WARNING 自己给的建议）无效；关 `multibuffer`/`enable_ubuf_saving` **更糟**（连 `atomic_max` 求出的 m 都错）；转置形式 `tl.dot(vT, tl.trans(p))` **运行期 AI Core aivec timeout/trap**；把 p 经 per-program 小 scratch 走一遍 GM 再 load **编译报错**；把 atomic 换成普通 `tl.store` **bishengir SIGABRT** |
| 逐 token 压缩表（每 token 只扫自己的 topK 块） | ⛔ **全面变差**：某算子 case 20 从 2.78 → 21.3 ms、case 40 从 54.5 → 954 ms。任务数从 3072 暴涨到 98304，tile 缩到 `[2, 128]` ⇒ 每任务标量前导占满。**访存量不是唯一目标函数** |
| 去掉逐块的数据依赖分支 `if tl.max(sel) > 0:` | ⛔ **无收益**（1472 ms vs 1386 ms，略慢）。曾假设"Vector 算 tl.max → 同步给标量单元"打断流水，**证伪**；分支不是主要开销 |
| 分层 cumsum（`[BR,NP2]` → `[BR,NCH,CH]` 两级） | ⛔ 15.8/15.1/14.7 ms vs 扁平 14.3 ms，**无效**。cumsum 代价 ∝ 元素总数，拆轴没意义 |
| `tl.atomic_add` 抢占式紧致化（替代 cumsum） | 语义**正确**（cnt 与集合都对），但同址 lane 高度竞争，**触发 Vector core 看门狗**，不可用 |
| `enable_ubuf_saving` 编译开关 | 在 bishengir **0.1.0 上根本不可用**（`Unknown command line argument`，编译直接失败）；1.2.0 上可用，且是 128×128 能编过的前提。**开关的可用性先于收益** |
| `BM=BN=128` 用于 paged topK 形态 | ⛔ `ub overflow`（编译器已自动关掉 code-motion 和 multi-buffer 重试仍失败）。同一个 128×128 在 `sp-blockmask` 形态上却能编过 ⇒ **可编域是"形态 × 编译器"的函数，不能跨算子照抄** |
| SLOW gather 路径开 `multibuffer=True` | ⛔ **无收益且易顶 L1**（BN=16~64 全扫过，±0.3%；组合 kernel 内直接 L1 overflow 5013504/4194304）。gather 慢路径瓶颈不在流水 |
| `tl.max_contiguous` / `tl.multiple_of` 列连续提示 | ⛔ **不改变 gather lowering**，地址张量照样物化、耗时不动（BN=32/64/128 实测逐一相同） |
| 快慢路径拆成两个独立 kernel（各自 flag 门控） | ⛔ **FAST 路径莫名回退 44%**（6.9→10.1ms，同代码文本；arange 0.8362→0.6044）。见 L1.15 |
| SLOW 路径去掉 PV 的 hi+lo 二段拆分 | ⛔ **精度闸门翻车**：MERE 1.3e-6 → 1.3e-3（阈值 9.77e-4）。随机索引下 softmax 近均匀、输出近零，相对误差极敏感，pl 不可省 |
| ping-pong 双缓冲重叠 stage/consume（无 barrier） | ⛔ **无收益**（14.5ms vs 串行 13.3ms），编译器对同 base 指针保守序化，叠加子步 staging 的循环开销反而更慢 |
| 暂存慢路径的微修剪（NO_TAIL_S 去尾 mask / QN_FULL 去 Q mask / CHK 自适应） | ⛔ **更差**（case16 7.49→7.62ms，case1 13.9→14.5ms）。编译器对这些谓词已最优，多出的分支反伤调度 |
| 小 K case 去掉 per-task `tl.debug_barrier` | ⛔ **零成本**（7.611 vs 7.605ms），barrier 不是开销项；store→load 同址依赖编译器本已插同步 |
| 小 K 退化为 dense 全表注意力（免 gather） | ⛔ **计算浪费 8×**（case16 估算 27ms vs 7.5ms），稀疏度 12.5% 时 dense 永远亏 |
| stage 循环手工 unroll×2（增多 MTE2 在飞事务） | ⛔ **更差**（case16 7.09→7.26ms、case1 13.27→13.59ms）。simulator 证据：MTE2 队列已饱和（283 cyc/条×14336 条=57% vector cycles），瓶颈是事务**数量**而非在飞度；事务数=任务数×K 行×2 张量，结构性固定 |
| 由 simulator 诊断驱动的小 K 优化 | ⛔ case16（K=512）地板 = stage 3.45ms(满带宽拷贝) + consume 3.44ms ≈ 6.9ms vs 0.4 目标线 7.43ms；实测 7.49ms。**源码层无解**，需编译器修 gather→dot lowering（L1.14） |

### §5.3 天花板估算：先算账再动手

块稀疏算子最容易高估收益。三步：

1. **块加载次数** = `Σ_tiles (每 tile 扫的块数) × heads × batches`，乘以块字节数（K+V）；
2. **工作集** = `nblocks × 块字节数`，若 ≤ L2 则上面那个数是 **L2 流量**不是 HBM 流量；
3. **每次块迭代的理论代价** = Cube MAC / 200 TFLOPS + Vector 元素数 / 1e13 + 访存 / 带宽。

实测一次对照：某算子 case 40 有 147456 次子迭代、实测 54.5 ms ⇒ **每次 8.9 µs**，
而理论只需 ~12 ns（Cube 8 ns + Vector 4 ns），**差 700 倍且与访存量无关**
⇒ 判定为"指令发射/同步开销主导"，此时**继续减少访存量是没有用的**，
只能"把每次迭代做的事变大"（加大 tile、去掉子块拆分、砍每次迭代的 Vector 趟数）。

---

## §6 精度闸门

### §6.1 判定顺序（错一步会把 bug 归错类）

1. **先看覆盖率**：输出是否被 kernel 全覆盖？`torch.empty` + 部分覆盖 = 未初始化内存，
   错误形态**同时**触发 NaN 位置断言和精度判定，极像"精度不够"（L1.5）。
2. **再看是不是选块集合不一致**：选块是离散的，差一个块 = 整行结果不同，
   表现是 `matched_ratio` 断崖而不是缓慢劣化。
3. **最后才是数值精度**：此时才看 `p` 的拆分、累加器 dtype。

### §6.2 `tl.dot` dtype 契约在本类算子上的落点

| 量 | 是否需要提精度 | 理由 |
|---|---|---|
| q / k / v / dout | **否** | 本来就是 bf16/fp16 存储，升 fp32 一位有效数字都不增加 |
| `p`（softmax 输出） | **是**，hi+lo 二段拆分 | 真 fp32 量，误差被 `Σp|v|/|Σpv|` 放大 ~20× |
| `dS`（反向） | 同 `p` | 同理 |
| 累加器 | 全程 fp32 | — |

### §6.3 定位手法：dump kernel 的中间量，不要只看最终输出

某算子最终输出全错，但把 `m` / `l` / `acc` 三个中间量单独 dump 出来跟 torch 复算比，
立刻看到 **`m` 完全正确、`l` 完全正确、只有 `acc` 恒 0** —— 一眼锁定是 PV 那一步，
省掉在 tiling / 精度上的全部无效搜索。
### §6.4 双路径实现的分支验证盲区

benchmark 若全为连续索引，SLOW 路径一次都不会被执行——它上面的 bug 会静默逃逸。
交付前必须用最小复现脚本（非 arange 索引）单独跑一遍 SLOW 路径的 golden 对比与
确定性检查（同输入 3~4 次运行 self-diff 必须为 0）。

---

## §7 测量口径

- `benchmark.py` 的 `parse_operator_latency` **按 kernel `Name` 分组**。
  新版（`triton_dev_v2`，commit "benchmark 时延改为按单次调用计算"）会**反推发射次数 L**
  （`整组行数 = L × active_count`）再算单次调用耗时；旧版直接除以 `active_count`，
  **同名 kernel 一次 forward 内发射多次会被低估为 1/L**。
  ⇒ **报 speedup 前先确认用的是哪个版本**，两个版本的数字不可比。
- 报告里必须带 **`launch/call`**：impl 侧应为 1.0（每个 kernel 每次 forward 只发射一次）；
  framework 侧若是 Python 循环参考实现，会是几百，这时旧口径下的 speedup 分子被严重低估。
- `implementation.avg_latency_ms` / `framework.avg_latency_ms` 报的是**几何平均**，不是算术平均
  （实测：报告值 3.0149，逐 case 算术平均 94.1179，几何平均 3.0149）。
- ⚠️ **把工作挪到 host 会让 speedup 虚高**：CPU 上的 torch op 不产生 NPU kernel，
  不进 `operators` 统计，而参考实现同样的工作是在 NPU 上做的。
  这不是优化，是口径不对等 —— 见 L1.2，索引/掩码准备必须在 device 侧。
- 环境探针：torch 参考实现代码恒定，其 `framework.avg_latency_ms` 就是天然的环境探针，
  设一个基准带，超出即判该次测量无效、自动重测。**比值型指标不能自证有效**。
- ⚠️ **随机索引的分布必须模拟真实场景，禁止用均匀 `randint` 偷懒**：真实 lightning_indexer
  topk 产物**无重复且升序**；均匀 `randint` 有重复（K=2048/KVS=32768 时每行约 64 个重复下标）
  且无序，只是压力下限。正确生成（分块控内存）：`sc=rand(rows,KVS); sc.topk(K).indices.sort()`
  ——实现记为 `randtopk_sorted`。两个分布都应测量并同时报告（实测 SFA 16 case 两分布
  核时比 0.4110 vs 0.4095，几乎一致 = 鲁棒性证据）。精度 golden 用 fp32 torch（CANN 自身
  MERE(CANN,fp32)=1.6e-3 不宜作精度标杆），性能对照用 CANN 实测延迟经
  `benchmark.py --skip_framework --framework_latency_ms` 注入（逐 case 跑再合并）。

---

## §8 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| 44% 元素错 + NaN 位置断言同时触发 | 选择表行粒度 ≠ kernel `BLOCK_M`，输出未被全覆盖 | L1.5 |
| `tl.dot` 结果恒 0，但同 kernel 里另一个 dot 正常 | kernel 内有 `tl.atomic_*` | §5.2 第一行 |
| 跑到某个 case 直接 `507035 vector core abnormal` | `tl.cumsum` 慢到触发看门狗 | §4.3 |
| `NaN` 只在某些 (Hq, Hkv) 组合出现 | masked load 仍算出越界地址 | L1.6 |
| 编译报 `Unknown command line argument '--enable-...'` | 编译器版本不支持该开关 | S5，先确认编译器来自 pip 包还是 PATH |
| 精度全过但 speedup 反而变好 | 12 个 case 编译失败被排除在几何平均之外 | 看 speedup 前先看 `passed_cases == total_cases` |
| `aic_scalar_ratio` 很高 | **不一定**是标量运算多，常常是同步等待被计进标量管线 | 下结论前先 dump IR 确认 |
| 同输入多次运行结果漂移（self-diff 非零）/ 偶发 NaN | load 掩码比较被 fp32 化（`h.to(fp32)<QN`）触发 bishengir 非确定性缺陷 | L1.12，掩码比较保持 int32 |
| 双路径 kernel 报 `Mismatched type ... between then block and else block` | 两分支定义了同名的不同形状局部变量 | L1.13，分支局部变量分别命名 |
| gather 类 kernel 的 UB overflow 数字远超朴素 tile 估算 | 按行间接寻址物化的 int32 地址张量（∝ BN×D） | L1.11，D 拆半或仿射化 |
| gather 后接 `tl.dot` 输出 ±512 级垃圾值 / 极简复现编译报 `Unknown core type: malloc` | unmasked 2D gather 喂 dot 的 lowering 缺陷 | L1.14，改 masked gather 或「连续化暂存」 |
| 双路径算子拆 kernel 后主路径变慢 40%+ | 编译器对组合分支 vs 独立 kernel 调度不同 | L1.15，保持单 kernel if/else |
| 随机索引 case 精度 MERE 卡在阈值 1.3 倍处 | 省略了 PV hi+lo 拆分；近零输出对 fp16 舍入敏感 | L1.14 骨架保留 pl |

---

## §9 与其它模板的分工

| 文件 | 何时用 |
|---|---|
| `flash_attention.md` | 稠密 FA 主链（KV 全扫 + online softmax）。**本类算子也要读**，其 L1.1/L1.9/L1.13 依然成立 |
| `gqa.md` | 「二 压缩类」：head 共享 / 页表寻址 / 路由门控。**当本类算子的主链还有 `kv_head = q_head // group` 或分页寻址时（索引表行 1a）追加读它的 §2/§5**；纯 MHA / varlen 组织的块跳过（行 1）不需要 |
| `attention_index.md` | attention 家族定 `category` 的唯一入口。本文件对应其**行 1a / 1b**（三·2 块跳过 / top-k） |
| **本文件** | 入参里有块级选择结构（块掩码 / CSR / topK / pooled 打分） |
| `flash_attention.md` | 一 标准类全部三个细分（基础三段式 / 分块流式 / 空间 token 版），含极小 `S` 的朴素一次性 attention |

冲突时以本文件为准（本文件结论均在稀疏形态上实测）。

---

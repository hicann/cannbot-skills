---
name: attention-paged
description: 分页 KV Cache Attention（Paged Decode / 批量 GQA Decode / Paged Prefill，含 SWA / GQA / 因果 / golden 舍入链精度契约）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧、Phase 4 优化点清单与证伪方向全表
metadata:
  type: reference
---

# 分页 KV Cache Attention 类算子优化经验（Decode / Prefill）

本文档是 **"KV 经 `block_table` 页表间接寻址、页间不连续、整段 KV 都参与 softmax（不跳块）"**
这一类算子的经验合集，覆盖 Phase 2/3/4。

- **§1 通用经验 S1–S4**：跨形态共有的工程建议
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（三个子形态的骨架与选择判据）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表**、天花板估算）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与其它模板的分工**

> ⚠️ **本文件与 `gqa.md` 的分工**：`gqa.md` 给出 GQA 任务划分、页表标量 load 红线（L1.2）、
> scores 掩码（G6）、tile·UB 预算记账（G3/§5.2）的完整论述——**那些结论在本类算子上依然成立**。
> 本文件只管"页表寻址形态"特有的部分：三个子形态的目标函数、block_ptr/流水线/掩码表、
> 以及 golden 舍入链精度契约。**两份都要读**；冲突时以本文件为准
> （本文件的结论均在分页形态上验证过）。
>
> **证据基础**：三个分页形态算子的完整探索轨迹——
> `paged_prefill_kernel`（因果 GQA，**计算受限**，重写持平 baseline 即可达上限）、
> `_swa_paged_decode_kernel`（SWA/GQA 单序列大页，**内存受限**，geomean 硬上限 ~0.89×）、
> `paged_attention_v2_decode_kernel` 系列（批量 GQA decode，vLLM 风格小页 + golden 舍入链，
> **每页固定开销受限**）。
> 性能数字取自 Ascend950PR（prefill / 大页 decode）与 Ascend910B2（批量小页 decode）实测。
>
> ⚠️ **核心优化哲学：先分形态再动手。** decode 大页是内存受限（优化方向是减 HBM 流量、提并行），
> 批量小页是每页固定开销受限（优化方向是**减少页迭代数**，加大 tile 内计算无效），
> prefill 是计算受限（核心是"让流水线重叠生效"而非"减流量"）。**三者的目标函数不同，
> 禁止跨形态套用**（典型反例：把 decode 的 GQA-grouped 减 KV load 套到 prefill，实测不可行，见 §5.2）。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. KV 经 `block_table[seq][logical]` 间接寻址（`phys = block_table[...][...]` → `K = cache[phys]`），
   页与页在显存中不连续，但**不跳块**——整段 KV 都要算；
2. 参考实现里有 `cache[block_indices].view(-1, KH, D)` / 按页表 gather 页再 reshape 的取数写法；
3. SWA/窗口死区的**整页跳过属算术区间收缩**（只收缩循环范围，不做块选择），不算跳块，仍用本文件。

有块级选择结构（块掩码 / CSR / topK 选块）→ 走 `block_sparse_attention.md`；
主链为矩阵吸收（`o = softmax(s) @ ckv`）→ 走 `mla.md`；
KV 是连续区间、仅 head 共享 → 走 `gqa.md` §3.2 ＋ `flash_attention.md`。

### §0.2 ★ 形态识别四问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | decode 还是 prefill？query_len 是否为 1？ | 定子形态（批量小页 decode / 大页 decode / prefill），三套目标函数（见上） |
| **Q2** | golden 的 cast 链长什么样——有没有 `einsum(...).float()`、`softmax(...).to(dtype)`、`x * scale` 这类显式 dtype 边界？ | 决定精度契约（§6）：**有显式 cast ⇒ 复刻舍入链、两轮 KV 循环**（§3.1）；无 ⇒ 单轮 online（§3.2） |
| **Q3** | G = H/KH 是 1 还是 ≥2？ | G==1 走向量通路 kernel（§4.5），G≥2 走 Cube dot 通路（host 一个 if 分派） |
| **Q4** | 页大小 BS 是多少、是否 2 幂？ | `BLOCK_BS ≤ BS 且整除`是精度红线（L1.2）；BS 全 2 幂时直接取 `BLOCK_BS = BS`，页内行掩码整体消除 |

### §0.3 子形态分类表

| 子形态 | 场景特征 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| 大页 decode（§3.2） | SWA/GQA，页大、序列少 | M=1（GQA 组 pad BLOCK_M≥16），**内存受限** | GQA-grouped grid + block_ptr DMA + SWA 死区跳页 |
| 批量小页 decode（§3.1） | 批量 NS 条序列，GQA+滑窗，页小（BS=16/32/64），golden 带 cast 链 | M=G≤8，**每页固定开销（SCALAR pipe）受限**，MMAD 个位数占比 | persistent 任务循环 + **两轮 KV 循环（精度契约）** + G==1 向量通路分派 |
| prefill（§3.3） | 因果 GQA，M=BLOCK_M 多 query 行 | **cube 计算受限** | split-q persistent + helper 内 block_ptr 触发 auto-multi-buffer 流水线 + aux_mask 表 load |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 ★ 先用 Q1 定形态，禁止跨形态套用优化方向

三个子形态的瓶颈画像不同（内存 / 每页固定开销 / 计算），一个形态的头号杠杆在另一个形态上
可能是证伪项（§5.2 全表）。Phase 2 草图必须先写明子形态与目标函数，再选骨架。

### S2 性能度量必须 CANN profiler kernel-only

op-level `npu.Event` 紧凑 Python 循环含 ~3× host dispatch 开销，对 ~10-54µs 级 kernel 严重失真
（实测 prefill BLK=256 下 Event 报 186µs vs kernel-only 54µs）。完整口径见 §7。

### S3 num_stages 在本工具链被忽略，勿浪费时间

decode/prefill 侧 2/3/4 均无变化。流水线的正确触发方式是 helper 内 op 顺序（L1.7），不是 num_stages。

### S4 杂项工程习惯（实测踩过）

- masked load 的 `other` 不可靠：掩码 lane 用 `tl.where(mask, x, 0.0)` 显式清零后再参与运算。
- `kv_len_cap = max_blocks_per_seq * block_size` 在 host 算好传标量，kernel 里 `min(kv_len, cap)` 防页表越界。
- q/out 直接按目标布局落盘，host 侧禁止 `permute(...).contiguous()`（`gqa.md` L1.8）。
- 核数获取：cube = `torch.npu.get_device_properties(0).cube_core_num`；
  vector = `triton.runtime.driver.active.utils.get_device_properties(dev)["num_vectorcore"]`。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ K/V 加载：block_ptr（或标量页基址的 affine tile load），禁止向量基址跨页 gather

- **必须**: 分页 K/V cache 的加载用 `tl.make_block_ptr` + `boundary_check=(0,1), padding_option="zero"`
  （页内连续布局 → DMA），再喂给 `tl.dot`。prefill 侧还要求 K/V/Q/O 全部 block_ptr 且 KV load 写在
  helper 内部（L1.7）。
- **禁止**: `tl.load(ptr + 向量页基址[:,None]*stride + ...)` —— 页基址是向量的跨页 gather，
  bishengir 把 dot 逐出 Cube、访存不走 DMA。
- **实测（decode）**: offset-load `mac 1.7% / vec 91% / mte2 0.1%` → block_ptr `mac 17-27% / mte2 35-69% / vec 51-73%`；geomean `251.8µs → 17.6µs`。
- **实测（prefill）**: 同工作量下 explicit-ptr 版 90.7µs（重叠效率 1.12×）→ block_ptr+helper 版 54.5µs（重叠效率 1.72×）；mac/mte2 工作量相同，差异全在软件流水线。
- **边界**: 红线的本质是「**页基址是向量**」。当 tile 严格不跨页（BLOCK_BS ≤ PAGE_SIZE）且页基址是
  **标量**时，`tl.load(k_ptr + 标量页基址 + 行距[:,None] + 列距[None,:])` 的 affine 2D offset load
  也能保住 dot 上 Cube（MMAD 正常发射）。block_ptr 版 MTE2 占比更优，两种写法都合规，按页大小与形态选。

### L1.2 ★★ BLOCK_N ≤ PAGE_SIZE 且整除（精度红线）

- **规则**: `BLOCK_N = min(128, next_pow2(PAGE_SIZE))`，保证 `PAGE_SIZE % BLOCK_N == 0` → tile 完全落在单页内，每个 KV tile 只查一次 block_table。
- **禁止**: BLOCK_N > PAGE_SIZE —— block_ptr 按 `shape=(PAGE_SIZE,…)` 对跨页行补零，但 keep mask 按 `j < kv_seq_len` 判定，会把补零产生的 `qk=0` 当作真实 key（应为 -inf）→ 精度炸。
- **UB**: block_ptr 连续 BN=128 ≈92KB（OK）；gather 跨页 BN=128 触发 `ub overflow`（需 3.0M bits > 1.77M）。
- paged 寻址: `logical_page = kv_start // PAGE_SIZE`；`in_page = kv_start % PAGE_SIZE`；
  `phys = tl.load(block_tables + b*stride_bt_batch + logical_page*stride_bt_block)`。
- 批量小页 decode 直接取 `BLOCK_BS = BS`（BS 全 2 幂时恒整除）：granule 恰好一页，页内行掩码
  整体消除，唯一掩码点是 scores 位置掩码。

### L1.3 GQA head 映射（强制，注意两方向互逆）

- decode（program 按 kv_head **反推**组内 q-head）: ABAB(gqa_interleave) `head_idx = kv + r*nkv`；AABB `head_idx = kv*ratio + r`（r=arange(BLOCK_M)，mask r<ratio）
- prefill（task 按 q_head **正推** kv_head）: ABAB `kv_head = q_head % NUM_KV_HEADS`；AABB `kv_head = q_head // (NUM_Q_HEADS//NUM_KV_HEADS)`
- 批量 decode（`k[page, row, kv_head, d]` 布局，q 头按 kv 头分组连续排布）: `head = kh * G + g`（AABB 语义，g∈[0,G)）

### L1.4 ★★★ 任务划分：decode 按 (kv_head, batch) 分组；prefill 按 split-q persistent

- **decode（G≥2）**: `grid = (num_kv_heads, batch_size)` 或 persistent 任务循环
  （task=(seq, kv_head, gb)，`grid = min(核数, tasks)` + 核内步长循环）。每个 program 加载分页 KV
  **一次**，用 `tl.dot` 一次性算出同组 G 个 q-head（pad BLOCK_M≥16）。
- **禁止**: `grid = (num_q_heads, batch)` —— 同组 q-head 各自重复加载同一 kv-head 的 KV（G× 冗余流量）。
- **边界（G==1）**: MHA 时分组无 KV 复用收益（每 kv_head 只服务 1 个 q head），dot 的 M=1 pad 到
  16 行浪费 15/16 Cube MAC——必须走**向量通路分派**（§4.5），不走本条。
- **prefill**: `grid = (cube_num,)`（按芯片取，如 950PR cube_num=28），kernel 内轮询领取 `(batch, q_block, q_head)` 任务。禁止把任务全展开成 grid（粒度太细、launch/sync 开销高）。
- **prefill 禁 GQA KV 复用**：GROUP 头共享 K/V 减 load 不可行——顺序处理多组头寄存器压力过大，融合进 M 维需慢 gather load；且 prefill 是计算受限，减 load 无效（§5.2）。

### L1.5 ★★ 因果掩码用 operator 下发的 aux_mask 表 load（prefill）

- **必须**: 因果掩码用下发的下三角 `aux_mask` bool 表，按 `offset = clip(kv_start - q_abs_start, …)` 索引 `tl.load`。该 load 走 **mte2（不占向量核）**。
- **禁止**: 算术掩码 `mask = (col_idx[None,:] <= q_abs[:,None]) & col_valid & row_valid` —— 比较与按位与占**向量核**，与 softmax 的 exp/max/sum/where 争夺向量单元。
- **实测**: 算术 mask 0.84× → aux_mask load 1.00×。掩码成本从向量核挪到 mte2，被流水线重叠掩盖。

### L1.6 ★ SWA：大页 decode 死区整页跳过；query_len=1 折叠成区间收缩

- **死区跳页（大页）**: 页粒度遍历 KV；整页落在死区 `[global_window, kv_seq_len-1-local_window)` 则跳过（不加载 K/V）。判据 `page_in_global = page_start < global_window`；`page_in_local = page_end > kv_seq_len-1-local_window`；二者皆假则 skip。
- **收益**: 长序列（kv=8192, local=1023）只加载 ~1024 keys 而非全量。
- **区间收缩（批量小页，query_len=1）**: `kv_start = max(kv_len - sw, 0)`，KV 循环从 `kb0 = kv_start // BLOCK_BS` 起——golden 的 triu 掩码在 query_len=1 时等价于这个区间，直接**减少页迭代数**，无逐页分支。
- 位置比较走 fp32（`(pos + offs).to(f32)` 与 `kv_len.to(f32)` 比）：int 比较会触发标量降级（`gqa.md` Layer 3 实测 +23%）。

### L1.7 ★★ helper 内 op 顺序决定流水线（prefill 核心）

- **必须**: 把单 KV block 的 K/V load + softmax + dot 封进 helper，且 op 顺序为：**早 V-load → `tl.math.exp` → `propagate_nan` → 分离的 scale / sub → 3-arg 累加 dot `tl.dot(p, v, acc)`**。只有这个顺序触发 bishengir auto-multi-buffer。
- **禁止**: inline 展开、或把 V-load 放在 softmax 之后（晚 V-load）—— 流水线不触发。
- **Why:** bishengir 的 auto-multi-buffer pass 对 op 调度顺序敏感；早 V-load 让 V 的 DMA 与 QK→softmax 的标量/向量计算重叠，形成 K/V 双缓冲。

### L1.8 ★★ 禁止破坏单循环流水线的"优化"（prefill）

- **禁止①**: 单循环内运行时 `if` 跳过 full-block 的 mask —— BLOCK_N=128 触发 bishengir 编译失败。
- **禁止②**: 两阶段 skip-mask / skip-load（把循环拆成"整页跳掩码"和"边界页算掩码"两段）—— 破坏单循环流水线，实测省下的 mask load 抵消不了流水线损失。

### L1.9 ★★★ golden 舍入链契约：p 必须在归一化后舍入 ⇒ 两轮 KV 循环（批量 decode 精度硬门槛）

- **契约**: golden 的 p 是 `softmax(attn).to(v.dtype)`——softmax（全局 max + 归一化）**之后**才舍入到
  v 的 dtype。kernel 必须复刻这个顺序：先拿到终态 `m_g/l_g`，再算
  `p = exp(s - m_g)/l_g`（fp32）→ `.to(native)` → PV dot。
- **为什么单轮 online 不行**: online softmax 的 p_eff 在**运行 max、未归一化尺度**上舍入，与 golden 的
  舍入尺度不同 ⇒ diff ≈ golden 自身 p 舍入噪声，bf16 下足以超 MERE 阈值（7.81e-3）。
- **结构（两轮，单 kernel 内）**:

```
轮 1（stats）: online 求 m_g/l_g —— QK dot + round-trip + scores 掩码 + max/exp-sum；
               无 V load、无 PV dot（max 精确 ⇒ 终态 m 逐位 ≡ golden 的 m_g）
轮 2（apply）: 重算 QK scores（与轮 1 同一舍入链，逐位一致），
               p = exp(s - m_g) / l_g → .to(native) → 单次 PV dot 累加；
               m 固定 ⇒ 无 rescale；出口不再除 l（p 已归一化）
```

- **代价账**: 每 granule 3 个 dot（2×QK + 1×PV）vs online 2 个 dot；UB 反而更省（无 v.to(f32) 大缓冲、
  无 rescale 的 acc 重写）。**精度是硬门槛，性能账在 Phase 4 再算**——不要为省一个 dot 回退单轮。
- **禁止项**: ① 回退 online 单轮；② fp32 PV dot（"更精确"= 不可复刻）；③ p_hi/p_lo 二段拆分
  （前提反转，见 §6.2）。

### L1.10 ★ einsum round-trip + 标量乘法同序舍入

- golden 的 `torch.einsum('qhd,khd->hqk', q, k).float()` 输出**先落 q/k 的原生 dtype 再升 fp32**。
  kernel 复刻为 `scores = tl.dot(q, tl.trans(k)).to(native).to(tl.float32)`——少一个 round-trip 都会在
  MERE 上显形。**两轮循环里的 QK 必须走完全相同的舍入链**（重算而非缓存，保证逐位一致）。
- `scale` 乘在 q_tile 上（每 task 一次、KV 循环外）：`(q_tile * scale).to(native)`，复刻 golden 的
  `query * scale` 同序舍入（标量本身按 L1.11 舍入）。

### L1.11 ★★★ golden 的标量乘法在 NPU 上有 dtype 舍入——host 侧必须 dtype-aware 复刻

- **现象指纹**: verify 失败集合**严格按 dtype 划分**（全部 fp16 case 失败、bf16 全过）⇒ 不是精度不够，
  是**算术路径不匹配**，别去调 tile / 累加器精度。
- **根因**: golden 的 `tensor * python_float` 在 NPU 上走 `aclnnMuls`，**fp16 下把标量先舍入到 fp16（RNE）
  再算**；Triton kernel 的标量参数走 fp32 路径。两条路径的差在 fp16 判据阈值（9.766e-04）量级——
  其余舍入链没对齐时被掩盖，**对齐之后才显形**。bf16 的 aclnnMuls **不**预舍入标量
  （fp32 路径本就 bitexact）。
- **修复**（host 侧按 dtype 舍入标量后再传 kernel）:

```python
def _scale_for_dtype(scale, dtype):
    """复刻 aclnnMuls 的标量处理：fp16 先舍入到 fp16（RNE），bf16/fp32 保持 fp32。"""
    if dtype == torch.float16:
        return struct.unpack("e", struct.pack("e", scale))[0]
    return scale
```

- **泛化**: 适用于任何 golden 含 `tensor * float` 的算子（不限于 attention）。排查顺序排在舍入链
  复刻（L1.9/L1.10）之后。

### L1.12 ★ scores 上的位置掩码用有限极小值，禁 -inf

KV 维 padding/越界必须掩在 **scores** 上（`tl.where(valid, s, -3.0e38)`），m 初值用有限极小值
（-1.0e30）而非 -inf，出口除法 guard（`l>0`）。完整论证见 `gqa.md` G6——清零操作数会把 padding
列的 score 变成 0，`exp(0 - m)` 是非零的，l 被灌进虚假质量。

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 批量小页 decode：两轮 KV 循环 + 双通路分派（golden 含 cast 链时）

```
host（forward，AST 白名单内）:
  G = H // KH；sc = _scale_for_dtype(scale, Q.dtype)          # L1.11
  out = torch.empty((NS, H, D), dtype=Q.dtype)                # 唯一 host 分配
  if G == 1:   # MHA 向量通路（§4.5）
      task = (seq, q_head)；grid = min(num_vec_cores, NS*H)；kernel 无 tl.dot
  else:        # GQA Cube dot 通路
      BLOCK_G = 16（tl.dot M 下界；G 个 q head 一组并成一个 q tile，行掩码 g<G）
      task = (seq, kh, gb)；grid = min(num_cube_cores, NS*KH*GB)
  NUM_CORES 与 grid[0] 同值传入（核内步长循环 for task in range(pid, total, NUM_CORES)）

kernel（每 task，两轮 KV 循环）:
  kv_len  = min(load(SL + seq), MB*BS)                        # 标量 load + cap
  kv_start= max(kv_len - sw, 0) if HAS_SW else 0              # L1.6 区间收缩
  q_tile  = load Q[seq, kh*G + g, :]（行掩码 g<G）
  q_tile  = (q_tile * sc).to(native)                          # L1.10：与 golden 同序舍入
  m = -1.0e30（有限极小值）; l = 0
  轮 1 for kb in [kv_start//BS, cdiv(kv_len,BS)):
      phys = load(BT + seq*MB + kb)                           # 标量（L1.1/L1.2）
      k = load [BS, D]（标量页基址 affine / block_ptr）
      s = dot(q, kᵀ).to(native).to(f32)                       # L1.10 round-trip
      s = where(valid[None,:], s, -3.0e38)                    # L1.12：掩在 scores 上
      m_new = max(m, rowmax(s)); l = l*exp(m-m_new) + rowsum(exp(s-m_new)); m = m_new
  inv_l = 1.0 / where(l>0, l, 1.0)                            # §4.6：倒数预计算
  轮 2 for kb in [kb0, kb1):
      k = load; v = load（v 提前到 dot 前，§4.7）
      s = dot(q, kᵀ).to(native).to(f32)                       # 与轮 1 逐位一致
      p = exp(s - m) * inv_l                                  # ★ L1.9：先归一化再舍入
      acc += dot(p.to(native), v)                             # m 固定 ⇒ 无 rescale
  store out[seq, head, :] = acc.to(native)，行掩码 g<G        # p 已归一化，出口不除 l
```

**UB 预算（192KB；原生 PV 无 v.to(f32) 大缓冲）**: 最坏 D=256,BS=64 时
k 32KB + trans(k) 32KB + v 32KB + q 8KB + scores/p 8KB + acc 16KB ≈ **130KB** ✓；
D=128,BS=64 ≈ 70KB。`tl.trans(k)` 按一份等大空间记账（`gqa.md` G3）。

### §3.2 大页 decode：单轮 online GQA-grouped FlashDecoding（golden 无 cast 链时）

1. `grid=(num_kv_heads, batch)`；`pid_kv=program_id(0)`, `pid_b=program_id(1)`
2. `kv_seq_len = load(seqlens[pid_b])`；按 L1.3 算 `head_idx[r]`；`qg = load(Q[b, head_idx, :BLOCK_D])`（pad BLOCK_M≥16）
3. `m=-inf, l=0, acc[BLOCK_M,BLOCK_D]=0`
4. **页循环** `for nb in range(max_num_blocks)`: 若 `page_start>=kv_seq_len` 停；L1.6 死区判据 skip；否则 `phys=load(block_table[b,nb])`
   - **页内 tile 循环** `for tn in range(0,PAGE_SIZE,BLOCK_N)`:
     - `k = load_block_ptr(K[phys, kv, tn:tn+BN, :BD])`（L1.1）
     - `qk = dot(qg, trans(k)).fp32 * scale`；apply SWA keep mask（L1.6）；在线 softmax 合并（m_new/alpha/p/l）
     - `v = load_block_ptr(V[…])`；`acc = acc*alpha + dot(p.bf16, v)`
5. `out = where(l>0, acc/l, 0)`；store `O[b, head_idx, :HEAD_DIM]`

> BLOCK_M 用 `tl.dot` 最小形（≥16）；decode ratio 通常 ≤8，pad 到 16。空 seq（l==0）输出 0。
> ⚠️ 前提是 golden 的 p 是 fp32 干净值（无显式 cast）。golden 带 `softmax(...).to(v.dtype)`
> 时单轮 online 不满足舍入链契约，须改 §3.1 两轮骨架——**两轮结构与 G==1 分派对大页形态同样适用**。

### §3.3 prefill：split-q persistent FlashPrefill

1. `grid=(cube_num,)`；`pid=program_id(0)`, `n_progs=num_programs(0)`
2. 遍历 batch：`q_start/end = load(cu_q_lens[b]/[b+1])`；`kv_seq_len = load(seqlens_kv[b])`；`cur_q_tasks = cdiv(q_seq_len,BLOCK_M)*NUM_Q_HEADS`
3. **轮询领取任务** `for q_task_id in range((prev_q_tasks+pid)%n_progs, cur_q_tasks, n_progs)`:
   - `q_block_id = q_task_id // NUM_Q_HEADS`, `q_head_id = q_task_id % NUM_Q_HEADS`
   - GQA head 映射见 L1.3
   - `Q_bp/O_bp = make_block_ptr(...)`；`q = load(Q_bp)`；`m=-inf, l=0, acc=0`
   - **KV block 循环** `for kv_block_id in range(num_kv_blocks)`: paged 查 phys（L1.2）→ `K_T_bp/V_bp = make_block_ptr(...)` → `mask = causal_mask_fn(aux_mask,…)`（L1.5）→ `acc,l,m = _prefill_flash_block(...)`（L1.1/L1.7 helper，见 §4.3）
4. `m_i += log(l_i)`；`out = acc/l_i[:,None]`；`store(O_bp, …)`

> BLOCK_M=128, BLOCK_D=HEAD_DIM, BLOCK_N=min(128,next_pow2(page_size))。online softmax（fp32）。
> 同 §3.2 前提：带 cast 链的 prefill 变体须按 L1.9 契约改造。

### §3.4 骨架选择判据

| 判据 | 选择 |
|---|---|
| query_len==1 且 NS 大、页小（BS≤64）、golden 带 cast 链 | §3.1 两轮 + persistent 任务循环 |
| 序列少、页大、golden 无 cast 链 | §3.2 单轮 online + 2D grid |
| query_len>1（prefill） | §3.3 split-q persistent |
| G==1（任意 decode） | 向量通路 kernel（§4.5）替换 dot 通路 |
| 任务数 < 核数（如 NS=1） | 属结构约束（§5.3），不要靠改骨架硬撑 |

---

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构须重设计）

### §4.1 block_ptr 加载 + dot（L1.1 杠杆）

```python
k_bp = tl.make_block_ptr(base=k_ptr + phys*stride_kb + kvh*stride_kh,
                         shape=(PAGE_SIZE, HEAD_DIM), strides=(stride_ks, stride_kd),
                         offsets=(tn, 0), block_shape=(BLOCK_N, BLOCK_D), order=(1, 0))
k = tl.load(k_bp, boundary_check=(0, 1), padding_option="zero")
qk = tl.dot(qg, tl.trans(k)).to(tl.float32) * scale
```

### §4.2 SWA keep mask（大页 decode；批量小页用 L1.6 区间收缩替代）

```python
j = page_start + tn + tl.arange(0, BLOCK_N)
local_thr = kv_seq_len - 1 - local_window
keep = (j < kv_seq_len) & ((j >= local_thr) | (j < global_window))
qk = tl.where(keep[None, :], qk, -float("inf"))
```

### §4.3 helper：早 V-load + 分离 scale/sub 触发 auto-multi-buffer（L1.7 核心）

```python
@triton.jit
def _prefill_flash_block(acc, l_i, m_i, q, K_T_bp, V_bp, softmax_scale, mask,
                         HEAD_DIM, BLOCK_M, BLOCK_N, BLOCK_D, is_fp8):
    # 早 V-load（与下面的 QK→softmax 标量链重叠 → 触发 bishengir 双缓冲）
    v = tl.load(V_bp, boundary_check=(0, 1), padding_option="zero")
    qk = tl.dot(q, K_T_bp_load, ...)               # K load 也在 helper 内
    qk = qk * softmax_scale
    qk = tl.where(mask, qk, -float("inf"))
    # 分离的 softmax 标量链（不要 inline 进 dot）
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.math.exp(m_i - m_new)
    p = tl.math.exp(qk - m_new[:, None])
    p = propagate_nan(m_new, p)                     # 处理 -inf 行
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    # 3-arg 累加 dot
    tl.dot(p.to(v.dtype), v, acc)
    return acc, l_i, m_new
```

### §4.4 aux_mask 表 load（L1.5，替代算术 mask）

```python
@triton.jit
def causal_mask_fn(mask_ptr, mask_size, stride_m, stride_n,
                   q_abs_start, kv_start, Q_BLOCK, KV_BLOCK):
    offs_m = tl.arange(0, Q_BLOCK)
    offs_n = tl.arange(0, KV_BLOCK)
    # operator 下发的下三角表，按 offset 索引；走 mte2，不占向量核
    mask_offs = q_abs_start * stride_m + (kv_start + offs_n)[None, :] * stride_n \
                + offs_m[:, None] * 0  # (示意：实际按表布局计算线性偏移)
    return tl.load(mask_ptr + mask_offs).to(tl.int1)
```

### §4.5 ★★★ G==1（MHA）向量通路分派（批量 decode 最大杠杆）

原理：G==1 时 ① KV 分组**无复用收益**（每 kv_head 只服务 1 个 q head，访存无 1/G 折扣）；
② `tl.dot` 的 M=1 pad 到 16 行 ⇒ **15/16 Cube MAC 浪费**；③ 任务天然可细分为 (seq, q_head)。
⇒ MHA 形态走**无 dot 的纯向量 kernel**（QK/PV 用「升 fp32 逐元素乘 + 树形归约」），可调度到
全部 vector 核（vector 核数 = 2× cube 核数），且不占用基本闲置的 Cube。

```python
# 向量通路的 QK（无 dot）：树形 fp32 归约后过同一 round-trip，与 dot 通路舍入语义一致
s = tl.sum(q_f[None, :] * k_tile.to(tl.float32), axis=1)   # [BS]
s = s.to(native).to(tl.float32)                            # L1.10 round-trip
# PV 同理：acc += tl.sum(p.to(f32)[:, None] * v.to(f32), axis=0)
```

- **精度对齐**: 两通路必须共享同一舍入链契约（L1.9/L1.10）——树形归约与 Cube 分组累加都过
  `.to(native).to(f32)` round-trip 后语义一致。
- **泛化判据**: 「M 维天然 ≤ dot pad 下界（16）且无跨行复用」的 decode 形态都该考虑向量通路；
  M=G≥2 时恢复 dot 通路（KV 访存 1/G 折扣 + 行数填满 dot tile）。
- 收益见 §5.1；分派开销 = host 侧一个 if。

### §4.6 ★ 循环不变量除法外提（inv_l 预计算）

```python
# ❌ 轮 2 每页一次 [BLOCK_G, BLOCK_BS] 全 tile 向量除法（l_safe 循环不变）
p = tl.exp(scores - m_i[:, None]) / l_safe[:, None]
# ✅ 循环外一次倒数（每 task 一次 [BLOCK_G] 除法），循环内乘法
inv_l = 1.0 / l_safe
p = tl.exp(scores - m_i[:, None]) * inv_l[:, None]
```

- 除法运算强度远高于乘法，SCALAR pipe 是瓶颈时整 tile 除法显形（simulator 定位）。精度：x/l 与
  x*(1/l) 差 ~1ulp fp32，p 舍入 native 翻转概率 ~2^-13——MERE 余量充足时安全，落地后走 verify 确认。
- **泛化**: 任何「循环不变量的除法/倒数」都该外提；标量分支（向量通路的 l_i）同理。

### §4.7 v_tile 提前 load（MTE2/Cube 重叠）

v 的地址只依赖 phys/row，与 scores 无数据依赖。**编译器不重排用户 load 顺序**——手动把
`v_tile = tl.load(...)` 提到 QK dot 之前发射，MTE2 load 与 Cube 计算重叠。收益小但零风险。

### §4.8 grid-split FlashDecoding（仅 grid≤4 的大页 decode）

- **只对 grid≤4 的 case 有效**（kv*batch 远小于核数）：BIGPAGE（grid=2）SPLITS=4 → 端到端 **1.25×**；GROUP1（grid=4）→ 1.11×。
- **grid≥8 无效甚至有害**：tile-work 守恒，grid 已近满时 split 双 kernel 开销 > 并行收益（LONG1 grid=8 → 0.97×，MBF16-512 grid=16 → 0.78×）。
- **工程权衡**：hybrid（双路径 + workspace + reduce kernel）只多 ~0.06× 且仅 2 case 受益，复杂度不值得；保持 single block_ptr kernel。
- split kernel 要点：tile-strided（`for tid in range(pid_split, MAX_TILES, NUM_SPLITS)`），`page=tid//TILES_PER_PAGE`（BN 整除 PS 保证 tile 不跨页），partial `m/l/acc[BLOCK_M,BLOCK_D]` 写 workspace，reduce kernel online-merge（注意 `acc*a[:,None]` broadcast）。
- ⚠️ 本条的 split 是**双 kernel 显式 split+reduce**；kernel 内多页合并/拼页是另一回事，已证伪（§5.2）。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 收益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ K/V 加载改 block_ptr（去 offset-load 向量 gather） | decode：mac 1.7%→17-27%，geomean 251.8µs→17.6µs；prefill：重叠效率 1.12×→1.72× | 所有形态（L1.1） |
| 2 | ★★★ G==1 向量通路分派 | **+24.3%**（geomean，批量 decode） | G==1 的 decode（§4.5） |
| 3 | ★★ helper 内 op 顺序触发 auto-multi-buffer（早 V-load + 分离 scale/sub） | 82µs → 64.7µs | prefill（L1.7） |
| 4 | ★★ aux_mask 表 load 替代算术 mask | 0.84× → 1.00× | prefill（L1.5） |
| 5 | ★ 循环不变量除法外提 | **+7.1%**（geomean，批量 decode） | 两轮结构的轮 2 / 任何循环不变除法（§4.6） |
| 6 | ★ SWA 区间收缩 / 死区跳页 | 页迭代数最多减半（sw=kv_len/2 时） | 带滑窗的 decode（L1.6） |
| 7 | v_tile 提前 load | 个别百分点 | 总是可试（§4.7） |
| 8 | grid-split FlashDecoding | 最高 1.25× | 仅大页 decode 且 grid≤4（§4.8） |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱，不要重跑这些死路**）

| 方向 | 结果 |
|---|---|
| **prefill 的 GQA KV 复用**（GROUP 头共享 K/V 减 load） | ⛔ 不可行——顺序处理多组头寄存器压力过大，融合进 M 维需慢 gather load；且 prefill 计算受限，减 load 无效 |
| **kernel 内多页合并 / 行基址向量拼页 / in-kernel split-KV** | ⛔ MLIR `PlanMemory Failed`（`gqa.md` §5.2 三种写法全灭）。扩展并行只允许 §4.8 的双 kernel split+reduce 形态 |
| **两阶段 skip-mask / skip-load**（拆"整页跳掩码"+"边界页算掩码"两段） | ⛔ 破坏单循环流水线，实测 62.2µs > 单循环 always-load 54.5µs |
| **单循环内运行时 `if` 跳过 full-block mask** | ⛔ BLOCK_N=128 触发 bishengir 编译失败（L1.8） |
| **num_stages 2/3/4** | ⛔ 被忽略，无变化（S3） |
| **对 grid≥8 的 case 用 grid-split** | ⛔ tile-work 守恒，split 开销 > 收益（§4.8） |
| **p 的 hi+lo 二段拆分（golden 自带 cast 链时）** | ⛔ 前提反转（§6.2）：golden 自己舍入 p 时，二段拆分制造一条 golden 里不存在的"更干净"路径，语义不可复刻；且 `gqa.md` §5.2 实测 -49% |
| **fp32 PV dot（golden 自带 cast 链时）** | ⛔ 同上，"更精确"= 不可复刻（L1.9） |
| **回退单轮 online（golden 自带 cast 链时）** | ⛔ p 舍入尺度不同，diff ≈ golden 自身舍入噪声量级，超 MERE 阈值（L1.9） |
| **放大 BLOCK_BS 超过页大小** | ⛔ 页界上限（L1.2），且跨页补零是精度红线 |
| **三参数 `tl.dot(p,v,acc)` / `num_warps` 调参 / `tl.compile_hint("tile_cube_loop")`** | ⛔ 无收益 / 无变化 / 本版本 `tl` 命名空间无此 API |
| **BLOCK_M 扫描（prefill）** | 128 最优（64→68.5µs 变慢、256 寄存器压力）；PROGS∈{28,56,84} 无影响 |

### §5.3 天花板估算：先分形态算账，再动手

**prefill（计算受限）——计算下限诊断法**：读 aic_mac/mte2/scalar/mte1/fixpipe 各核时间，算
`mac_ratio = mac/aicore` 与 `重叠效率 = sum(各核)/aicore`。若 **mac_ratio ≥ 0.4 且重叠效率 ≥ 1.6×**
→ 近计算下限，减 load / 调流水线都难撼动 → 目标加速比大概率不可达，应**诚实报告 infeasible**
而非编造。典型分核画像：mac 0.464 / mte2 0.466 / 重叠 1.72×——cube_mac（如 25µs）是硬下限、
总 54.5µs，要 1.2×（45µs）需 mac_ratio ≥ 0.56（即不减 MAC 下重叠再提 ~20%），本工具链不可行；
**从零重写持平 baseline（geomean 1.0× 量级）即可达上限**。

**大页 decode（内存受限）**：baseline 已近 HBM 峰值（实测 ~18µs/case ≈ 16MB @ ~1.3TB/s，
device_perf_npu kernel-only 复测一致）时，geomean 硬上限 ~0.89×。真正卡目标的是 vec 核
softmax（exp×2/max/sum/where/除法；分核画像 mac ~7% / vec 50-62% / mte2 ~35%）：每 program
~8 tiles 串行 softmax（attended keys / BN128），BN=128 已是 UB 上限，无法靠增大 tile 减轮数；
softmax 是数学必需，无 codegen 辅助时无法与 Cube/DMA overlap。
**可能的下一步（未验证，风险高）**：bishengir auto-pipeline——参考 prefill 经验（L1.7 的
helper + 早 V-load op 顺序可触发软件流水线），让 softmax 与 Cube/DMA overlap；finicky，列为后续。

**批量小页 decode（每页固定开销受限）**——排除法定位（simulator per-instruction pipe）：

| 候选信号 | 画像 | 判定 |
|---|---|---|
| MMAD > 50%（计算 bound） | 个位数占比 | ✗ |
| Cube WAIT_FLAG（空等 Vector） | 非主导 | ✗ |
| 标量降级（SCALAR 高且 calls=元素数） | SCALAR 主导但 calls 为任务/页级 | ✗（非降级） |
| MTE2/MTE3 dominant | 次要 | ✗ |

**确诊：SCALAR pipe 上的每页固定开销主导**——页表标量访存（每 task 每页 1 次 × 两轮）+ 循环控制
+ CV flag 交互；小页下每页有效工作量小，固定开销占比高。压缩手段全部被结构约束封死
（页表标量 load 禁向量 gather / 两轮结构是精度契约 / BLOCK_BS=BS 页界），此时**继续减访存或调
tile 无用**，可动的只有减少页迭代数（滑窗收缩）与循环不变量外提（§4.6）。

**弱 case 结构约束**：NS=1 时 GQA tasks = KH < 核数，核没用满且无合法并行扩展（§5.2）——这类
case 的 speedup 上限受任务数约束，属形态结构约束，不要在 tile/流水线上反复消耗轮次。

---

## §6 精度闸门

### §6.1 判定顺序（错一步会把 bug 归错类）

1. **失败集合怎么划分**：**严格按 dtype 划分**（fp16 全挂 / bf16 全过）⇒ 算术路径不匹配，
   查 aclnnMuls 标量舍入（L1.11），不要调精度。
2. **golden 的 cast 链**：有 `softmax(...).to(dtype)` / `einsum(...).float()` 显式边界 ⇒ 走舍入链
   复刻（L1.9/L1.10），两轮 KV 循环；"更精确的全程 fp32"是与 golden 的系统性偏差。
3. **最后才是数值精度**：此时才看 tile / 累加器 dtype / p 的处理方式。

### §6.2 `tl.dot` 与 softmax 的 dtype 契约落点

| 量 | 处理 | 理由 |
|---|---|---|
| q / k / v 操作数 | **原生 dtype** + fp32 累加器（不升 fp32 再 dot） | 输入本就是低精度存储，升 fp32 不加有效数字却翻倍 UB 流量（`gqa.md` L1.5） |
| scores（golden 有 `einsum().float()`） | `dot(...).to(native).to(f32)` round-trip | 复刻 einsum 先落原生再升 fp32（L1.10） |
| p（golden 有 `softmax().to(dtype)`） | **归一化后 `.to(native)`**，两轮 KV 结构 | 复刻 golden 舍入顺序（L1.9） |
| p（golden 的 p 是 fp32 干净值） | 不降 dtype / hi+lo 二段拆分 | `gqa.md`/`flash_attention.md` 的规则，**与本文件 L1.9 按 golden cast 链二选一** |
| scale（fp16） | host 侧 RNE 舍入到 fp16 再传 | aclnnMuls 契约（L1.11） |
| 位置/边界掩码 | 有限极小值 -3e38 掩在 scores 上 | `gqa.md` G6（L1.12） |

### §6.3 定位手法

- **CPU 仿真对拍**：把 kernel 的算术路径（online vs two-pass、标量舍入前后）在 CPU 上按位仿真，
  与 golden 复算比 MERE——能在不占 NPU 的情况下确认"diff 就是 golden 自身舍入噪声"。
- **dump 中间量**：m / l / acc 分别落盘对拍，m 对 l 对而 acc 错 ⇒ 锁定 PV 段，省掉 tiling 上的无效搜索。
- **fp64 对照**：impl 与 golden 各自对 fp64 参考的偏差同量级 ⇒ 两者是"互相不一致的各自正确"，
  问题在路径复刻而非精度不足。

---

## §7 测量口径

- **必须 kernel-only**: `device_perf_npu`（torch_npu.profiler active=5）→ `kernel_details.csv` 的本
  kernel `Duration(us)` 均值（`Name` 过滤掉 `aclnn` 开头行），或 `op_statistic.csv` 的
  `Avg Time(us)`（AI_CORE 行）。
- **禁止 op-level `npu.Event`**: 含 ~3× host dispatch 开销，对 ~10-54µs 级 kernel 严重失真（实测 prefill BLK=256 下 Event 报 186µs vs kernel-only 54µs，差 3.4×）。
- **framework 侧参照系**：paged decode 的 golden 通常是逐 seq Python 循环 + `.item()` 同步 +
  `repeat_interleave` 物化 ⇒ 小 case 恒 60~130µs 起步，speedup 数字天然偏大。评估优化收益看
  impl 绝对时间 / geomean 变化，不要被单 case 大 speedup 带偏。
- **simulator 占比 ≠ 端到端收益**：结构不变的等价变换（如除法外提）落地后 pipe 占比可能不动而
  benchmark 明显提升——占比定位瓶颈类型，终判以 benchmark 为准。
- **环境探针**：torch 参考实现代码恒定，其 `framework.avg_latency_ms` 是天然的环境探针，设基准带，
  超出即判该次测量无效、自动重测。

---

## §8 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| 只看总 Duration 就下"Cube 墙 / HBM 峰值 / 无法优化"结论 | 未看 mac/vec/mte2 分核占比，向量核/标量开销瓶颈被误诊为硬件极限（曾有形态因此错过 19×，0.0436×→0.8281×） | §5.3 分核诊断 + [profiler-core-breakdown](../.claude/skills/triton-latency-optimizer/references/profiler-core-breakdown.md) 铁律 |
| BLOCK_N > PAGE_SIZE | 跨页补零被 keep mask 当真实 key | L1.2 |
| `grid = (num_q_heads, batch)` | 同组 q-head 重复加载 KV（G× 流量） | L1.4 |
| verify 失败集合**严格按 dtype 划分** | aclnnMuls 在 fp16 下把标量先舍入到 fp16，Triton 标量走 fp32 路径 | L1.11 `_scale_for_dtype`（RNE） |
| online softmax"看起来更精确"却超 MERE 阈值 | golden 的 p 在归一化后才 `.to(v.dtype)`，online 的 p 在运行 max/未归一化尺度舍入 | L1.9 两轮 KV 循环 + §6.1 判定顺序 |
| G=1 case 全面偏慢 | dot M=1 pad 到 16 行，15/16 Cube MAC 浪费且无 KV 复用收益 | §4.5 向量通路分派 |
| 两轮结构轮 2 每页整 tile 除法拖慢 | 循环不变量 `l_safe` 未外提 | §4.6 inv_l 预计算 |
| NS=1 的 case speedup 上不去还在反复调 tile | GQA tasks = KH < 核数，无合法并行扩展（拼页 PlanMemory Failed） | §5.3 弱 case 结构约束 |
| npu.Event 计时严重偏大 | 含 ~3× host dispatch 开销 | §7 kernel-only |
| 算术 mask 占向量核与 softmax 竞争 | 比较与按位与走向量核 | L1.5 aux_mask 表 load（走 mte2） |
| 从零重写首版大幅劣化 | explicit-ptr masked load + 算术 mask，流水线未触发 | L1.1 block_ptr-in-helper + L1.7 op 顺序 |
| 拆两段循环"省掩码"反而变慢 | 破坏单循环 auto-multi-buffer | L1.8 保持单循环 always-load |
| 以为调 num_stages 能加速 | 本工具链 bishengir 忽略 num_stages | S3；改 op 顺序触发（L1.7） |
| 误判目标加速比可达而反复尝试 | 没看分核，未识别计算下限/带宽峰值/结构约束 | §5.3 天花板估算 |

---

## §9 与其它模板的分工

| 文件 | 何时用 |
|---|---|
| `attention_index.md` | attention 家族定 `category` 的唯一入口。本文件对应其**行 2**（二·3 页表寻址，不跳块） |
| `gqa.md` | GQA 任务划分、页表标量 load 红线（L1.2）、scores 掩码（G6）、tile·UB 预算（G3/§5.2）的完整论述；行 1 交叉（页表＋块跳过）时继续主责 |
| `flash_attention.md` | KV 是**连续区间**（无页表）的 FA 主链；golden 无 cast 链时的 online softmax 约束 |
| `block_sparse_attention.md` | 页表寻址 **＋ 块级选择**（topk/CSR/块掩码）叠加形态 |
| `mla.md` | 矩阵吸收（`o = softmax(s) @ ckv`）＋页表叠加形态（索引行 0） |
| **本文件** | KV 经 `block_table` 间接寻址、不跳块的 decode / prefill 全部形态 |

冲突时以本文件为准（本文件结论均在分页形态上实测）。

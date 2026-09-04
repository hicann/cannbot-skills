# Attention 家族子类索引（按**写法**分类）

> **本文件是 attention 家族定 `category` 的唯一入口。**
> Phase 2 Step 1 判定为 attention 家族后，**必须以显式路径读取本文件**（`loaded_via: explicit_path`，
> 禁止自动发现），据此定出子类与 `template_path`，再去读该 template 并逐条摘录其 Layer 1。
> `precheck.json` 必须同时记录 `index_path`（本文件）、`writing_class`（下表的大类·细分编号）
> 与最终的 `category` / `template_path`。
> 跳过本文件就拿不到 `category` 与 `template_path`，Step 1 门禁不通过，**无法进入 Step 2**。

## 分类口径：看写法，不看降本手段

判据是**打开 kernel 看代码骨架**——写一个新算子时，相对标准注意力要多写什么、改写什么、还是整个换掉。
**写法相同的算子共用同一套模板与优化知识**，这正是模板复用要的口径。

同一个算法按"降本手段"和按"写法"可能落在不同大类。典型：**一个只在 scores 上加
`masked_fill` 的算子，按手段是稀疏，按写法只是标准三段式**——GEMM 规模一点没变，
与最朴素的三段式共用同一套模板。

基准写法（所有变体的参照系）：

```python
S = Q @ K.transpose(-2, -1) * scale     # GEMM-1
P = softmax(S, dim=-1)                  # 行归约
O = P @ V                               # GEMM-2
```

| 大类 | 相对基准的改动 | 模板可复用性 |
|---|---|---|
| **一 标准类** | 不变，或只加 mask / 改 head 索引 / 改成分块流式 | 同一套模板，参数化即可 |
| **二 压缩类** | K/V **取数前**插一段：head 映射、latent 投影、页表寻址、量化（MLA：latent ＋ 页表叠加**矩阵吸收**，走行 0 `mla`） | 标准模板 + 取数层替换 |
| **三 稀疏类** | KV 循环**之前**插「选哪些块」，循环范围变成子集 | 标准模板 + 选择层 + 循环改写 |
| **四 线性类** | **整个换掉**：无 softmax，改状态递推或结合律重排 | 独立模板，与标准无共用 |
| **五 专用类** | 换目标函数（去归一化）或换数据组织（特征图门控） | 各自独立模板 |
| **六 非注意力** | 不是注意力 | 不走本索引 |

## 判别表

按**从上到下**匹配，**命中即停**。同时命中多行时以先命中者为准（见下方"叠加与优先级"）。

| # | 写法特征（打开 kernel 看骨架） | 写法类 | `category` | `template_path` |
|---|---|---|---|---|
| 0 | 主链是**矩阵吸收**：`ckv` 同时作 K 与 V（`o = softmax(sm_scale·(q_nope·ckv + q_pe·kpe)) @ ckv`，无独立 V 张量），叠加 q 拆 `nope`/`pe` 双 dot（或融合维 kernel 内切分）＋分页/块表间接寻址 | 二·2 latent 投影 ＋ 二·3 页表寻址（叠加）＋ 矩阵吸收 | **`mla`** | `.claude/template/mla.md` |
| 1 | 有块级选择结构（块掩码 / 块索引 / CSR / topK），**且主链有 `kv_head = q_head // group`**（交叉写法） | 三·2 块跳过 ＋ 二·1 ＋ 二·3（叠加） | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md`（选择表构建与跳块）**＋** `.claude/template/gqa.md`（§3.1 分页 decode 骨架 / §3.4 页内连续 / tile·UB 预算）——⚠️ **交叉行：两份的 Layer 1 都必须摘录**，见下 |
| 1b | 有块级选择结构（块掩码 / CSR / pooled-topk / 行图），**主链不含 GQA 分组**；或参考实现里有 `nonzero()` + 按选中块拼 K/V、`torch.topk` + `scatter_` 成 bool 掩码 | 三·2 块跳过/top-k | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md` |
| 2 | `phys = block_table[seq][logical]` → `K = cache[phys]`，页与页在显存中不连续，但**不跳块**（整段 KV 都要算；SWA/窗口死区的整页跳过属算术区间收缩，不算跳块） | 二·3 页表寻址 | **`attention-paged`** | `.claude/template/attention-paged.md`（block_ptr 红线 / decode·prefill 全套 Layer 1，op37/38 实证；批量 GQA decode 的 **golden 舍入链精度契约**（两轮 KV 循环，其 L1.9-L1.11＋§6 精度闸门）与 **G==1 向量通路分派**（其 §4.5）见其 §0.2 形态识别四问；KV 页内 `[K\|V]` 同行布局技巧另见 `gqa.md` §3.4，tile·UB 预算另见 `gqa.md` G3/§5.2） |
| 3 | attention 主链被**路由/门控**选择性执行（`argmax` + `one_hot` 或等价），多路权重按样本选 | 二·1 head 共享（门控变体） | **`gqa`** | `.claude/template/gqa.md`（§3.3 gqa-router） |
| 4 | 取数层只有 `kv_head = q_head // group` 这一处改动，KV 是连续区间 | 二·1 head 共享 | **`gqa`** | `.claude/template/gqa.md`（§3.2 稠密骨架）**＋** `flash_attention.md` |
| 5 | `kv_c = W_dkv @ h` 压缩存储、用时 `K = W_uk @ kv_c` 展开，rope 维单独一路 | 二·2 latent 投影 | `flash_attention` | `.claude/template/flash_attention.md`（**该细分暂无专属 template**） |
| 6 | `q8, sq = quant(Q)`；`s = int8_gemm(q8,k8) * sq * sk` | 二·4 量化 | `flash_attention` | `.claude/template/flash_attention.md`（**暂无量化专属**） |
| 6a | 参考实现含 **4 个 `nn.Linear` 投影**（Q/K/V/O），主链是三段式或分块流式 `softmax(Q@Kᵀ/√d)@V`（self/cross/MHA 同构）——⚠️ **K/V 生产链含空间缩减下采样（depthwise conv / AvgPool / stride-conv）或 scores 链含 transform conv＋instance_norm 的 CV 空间形态走行 10a，不在本行** | 一·1 标准类·含投影段 | **`mha`** | `.claude/template/mha.md`（投影段终态架构 + 实证）**＋** `.claude/template/flash_attention.md`（attention 主链）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 6b | 掩码/偏置**结构化但非均匀**：多成分复合（window 带 ∪ 全局行列 ∪ 随机列，op51/op37）、逐行/逐元素非均匀（per-row randperm、元素周期 `j%stride==i%stride`，op25）、结构化加性偏置（rel_bias / ALiBi，无 `-inf` 门控，op14）、host 侧结构稀疏（Swin 窗口预切分 / 轴向分解为两次 1D 稠密 attention，op35） | 三·1 掩码填充·非均匀变体 / 加性偏置 / 结构局部 | **`sparse_mask`** | `.claude/template/sparse_mask.md`（继承 FA 只写掩码 delta；先按其 §0.2 定生效层 a/b/c） |
| 6c | **反传链（标准/因果类 SDPA backward）**：入参给 dO/O/LSE，出参 dQ/dK/dV，主链含 softmax 重算 `p` 与 `ds = p∘(gq−delta)` 反传结构，掩码语义仅 causal 或无——⚠️ FlexAttention（mask_mod 非均匀掩码）家族 backward 不走本行，见行 6b backward 注记 | 一·1 标准类·反传变体 | **`mha`** | `.claude/template/mha.md`（§7 S_K 二分混合分派 + fp32 ieee 红线 + bishengir 三禁）**＋** `.claude/template/flash_attention.md`（前向主链 Layer 1）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 6d | **无 head 维单头 SDPA**：`softmax((q·C^-0.5)@kᵀ)@v`，q/k/v 缺省同源 x，shape `[B,N,C]`，C 可达 1024 | 一·1 标准类·单头大 C | **`mha`** | `.claude/template/mha.md`（§6 PV1 / SPLIT_D / tile 桶治理）**＋** `.claude/template/flash_attention.md`（attention 主链）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 7 | kernel 内对 KV 分块循环、跨迭代滚动 `m`/`l`/`acc`；或带 `causal`/`window`/`softcap` mask；或 `S` 大到不宜物化 `[S_Q,S_K]` | 一·2 分块流式（FA 系） | **`flash_attention`** | `.claude/template/flash_attention.md` |
| 8 | `S = S.masked_fill(mask, -inf)` 是唯一改动，**GEMM 规模不变** | 三·1 掩码填充 | **`block_sparse_attention`** | `.claude/template/block_sparse_attention.md`（⚠️ 见下节：标杆是掩码填充，生成目标必须是块跳过） |
| 9 | KV 由**固定几何邻域展开**构造：`kv = F.unfold(x, kernel=blk+2*halo, stride=blk, padding=halo)` / im2col，或手写窗口 gather 邻域块（非 unfold 同骨架）；token 来自 NCHW 分块展开 `reshape(b,c,h//blk,blk,w//blk,blk).permute(0,2,4,3,5,1)`，输出 scatter 回 NCHW；常带 `nn.Linear` 投影段（op50 落地） | 三·3 邻域展开（Sparse-展开型） | **`sparse_unfold`** | `.claude/template/sparse_unfold.md`（unfold 物化消除 + 每像素单投影 + 恒单块 softmax + aclnn 舍入契约 L1.1~L1.5，op50 实测 2.0973/2.1243）**＋** 含投影段时加读 `.claude/template/mha.md`（M1~M4 投影段硬约束 + §4.1 投影 GEMM 终态架构）——⚠️ **交叉行：两份的 Layer 1 都必须摘录**，见下 |
| 10 | `x.flatten(2).transpose(1,2)` 把 `(B,C,H,W)` 拉成 token 再走三段式，**无任何插段** | 一·3 空间 token 版 | `flash_attention` | `.claude/template/flash_attention.md` |
| 10a | **CV 空间 token · 带插段**：特征图 token 化（`Lq=H·W`）且 K/V 生产链含**空间缩减下采样层**＋LN——depthwise conv（`nn.Conv2d(D,D,k=ratio+1,stride=ratio,padding=ratio//2,groups=D)`）、`AvgPool2d(ratio)`（PVT 系常见）、stride-conv 等写法皆算，共同特征是 `Lk=ho·wo < Lq` 且由 token 网格几何推导；或 scores 链含跨 head 1×1 transform conv（`nn.Conv2d(H,H,1)`，`apply_transform` 门控）＋ softmax 后 `F.instance_norm`（逐 `(b,h)` 平面）；常带 4 投影且 `d_k≠d_v` | 一·3 空间 token 版·带插段 | **`spatial_attention`** | `.claude/template/spatial_attention.md`（fp32 全链路契约 / 下采样权重预转置 / S 流量模型 / 融合分派）**＋** `.claude/template/mha.md`（§2.5 M1~M4 + §4.1 投影段）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 10b | **CV-Attn-空间自注意力 · 卷积-块化窗口 transformer 型**：输入 NCHW 先过 k×k conv＋1×1 conv 局部表示，再按 patch 展开成 token（`reshape(b,nh,ph,nw,pw,D).permute(0,2,4,1,3,5)` 类 NCHW→(b,p,n,D) 块化布局变换＋对称 fold），窗口 n=nh·nw 常在数百内做**稠密**多头 attention（无跨窗口交互、无 KV 下采样），transformer 块 ×depth（LN+qkv/out 投影+残差+LN+FFN），尾部 `conv(cat([x,y]))` 双输入 k×k 融合卷积（MobileViT 系） | 一·3 空间 token 版·块化窗口细分 | **`patch_window_attention`** | `.claude/template/patch_window_attention.md`（复合点积双 B 指针禁令 / 拼接缓冲紧凑布局 / im2col NHWC 一次转置 / attn 按 n 单块特化 / GEMM 变体同档档位 / GEMM 搬运瓶颈诊断）**＋** `.claude/template/mha.md`（§2.5 M1~M4 + §4.1 投影段）**＋** `.claude/template/flash_attention.md`（attention 主链 L1.1/L1.12/L1.13）——⚠️ **交叉行：三份的 Layer 1 都必须摘录** |
| 10c | **CV 空间 token · 无投影双分支**：特征图 flatten token 化的自注意力，**无 `nn.Linear` Q/K/V 投影**——`S=Y·Yᵀ`、`O=P·Y`，Q=K=V 为同一 Y（conv 输出直接充当）；每分支前置**分辨率保持的 k×k conv**（`nn.Conv2d(C,C,k,padding=k//2)`）代替投影段；**position/channel 对偶双分支**并行——position 分支 token=空间位置（T=N=H·W, D=C, scale=C^-0.5, 布局 (B,N,C)）、channel 分支 token=通道（T=C, D=N, scale=N^-0.5, 布局 (B,C,N)），`out=out_pos+out_ch` 残差相加；golden 分支内 `.float()` 全 fp32、分支末尾 `.to(dtype)` 后 dtype 相加；conv 权重常在 forward 内 lazy 创建（`manual_seed(hash(...))`+`_cache`，DAModule/DANet 系写法） | 一·3 空间 token 版·无投影双分支变体 | **`position_channel_attention`** | `.claude/template/position_channel_attention.md`（双分支统一物理布局 L1.1 / PV 二段拆分 L1.2 / conv implicit GEMM+RNG 复刻 L1.3-L1.4 / 双路径分派 L1.6，实测 geomean 2.4313、50 case 全过）**＋** `.claude/template/flash_attention.md`（attention 主链 Layer 1）——⚠️ **交叉行：两份的 Layer 1 都必须摘录** |
| 11 | 纯基准三段式，不含 KV 分块 | 一·1 基础三段式 | `flash_attention` | `.claude/template/flash_attention.md` |
| 12 | **无 softmax**，改为状态递推（`state = a*state + k⊗v`）或结合律重排 | 四 线性类 | **`linear-recurrent`**（状态递推细分；结合律重排细分仍 `new_category`） | `.claude/template/linear-recurrent.md`（状态递推细分，op92 实证 5.75×；结合律重排细分暂无专属，见下） |
| 13a | 特征图门控**双分支加性融合**（通道 softmax-over-N 分支 + 空间 GAP/softmax-over-C2 分支，`out = x·(S+T)`，无 softmax(QKᵀ) 主链；op57 落地） | 五 专用类 | **`polarized_attention`** | `.claude/template/polarized_attention.md`（结合律消除 [B,C2,N] 物化 + 任务循环/档位化/双路径融合，2.3237） |
| 13b | 去归一化（无 softmax 但保留 QK/PV 结构）、或其他特征图门控（乘性融合/单分支） | 五 专用类 | `new_category` | **无可复用 template**，见下 |
| 14 | **CV 特征聚合-分发**：输入经**变换聚合**（`Linear` / `conv1x1` / 通道扩展 / spatial shift / mean 归约）→ **轻量 attention 加权**（KV 维极小或特殊：`k=3` 分支、逐通道权重、softmax-over-hw 空间维——**不是**序列 attention）→ **聚合回空间**（再经 `Linear`/卷积/加权求和还原）；算子操作 `[B,C,H,W]`/`[B,N,C]` 特征图而非 token 序列，attention 用于**特征门控/通道注意力**而非 token 间关系——§0.2 三形态：形态 (a) 通道扩展-分支分发 = S2Attention(op59)、形态 (b) 空间-通道双 attention = CoTAttention(op42)、形态 (c) conv 特征-双 softmax 汇聚 = DoubleAttention(op47) | 五·专用类·特征门控聚合-分发 | **`cv_attn_agg`** | `.claude/template/cv_attn_agg.md`（聚合-分发结构轴 §0.2 / C1 RNE 逐 conv 舍入 / C2 Kahan / C5 主成本在投影 GEMM / C6 路由含 cube tile 深度 / C7 结合律重排配精度门控） |

## ⚠️ 最重要的一条警示：掩码填充是 golden，不是生成目标

**三·1 掩码填充型的 torch 标杆用 `masked_fill` 表达，最直白、便于校验——
但生成的 kernel 必须改写成三·2 的块跳过写法，否则等于白做稀疏。**

两种写法**数学等价、性能天差地别**。这类算子的头号问题不是稀疏本身，而是
**「稀疏被写成了逐 token 的 program」**——掩码是块粒度的（通常 128×128），program 却按 token 发，
同一掩码块里的 128 个 query 每人把整块 K/V 搬一遍，访存放大 128 倍，同时 `M=1` 让 Cube 完全闲置。
**第一优化动作永远是把 program 粒度提到掩码块粒度**，实测 **26.25×**（几何平均，
见 `block_sparse_attention.md`）。

## 叠加与优先级

- **行 1 与行 1b 的分界是"主链有没有 `kv_head = q_head // group`"**。两者都属三·2，
  但行 1 是交叉写法：选择层与取数层替换同时存在，**主 `category` 取块跳过**（循环范围改写会
  重塑整个 kernel 骨架，取数层替换只是几行索引），GQA 侧的任务划分、页内布局与 tile/UB 预算
  由 `gqa.md` 补齐。
- ⚠️ **交叉行（行 1）的 `layer1_constraints_loaded` 必须同时摘录两份卡片的 Layer 1**，
  `template_path` 记两条。理由是实测出来的机制：**只有被摘录进 `precheck.json` 的 Layer 1
  才受 Phase 2 Step 3 的合规检查门管辖**——同一条规则写在 Layer 2 连续 5 轮无效、
  原样移进 Layer 1 后一轮就生效。只摘一份的话，另一份的硬约束等于没写。
  两份约束若有冲突，以 `category` 对应那份（`block_sparse_attention.md`）为准，
  并把冲突点连同实测差异补进本索引。
- **压缩类的四种插法可叠加**：latent 投影 + 页表寻址是常见组合（先压成 latent 再分页存）；
  原则上还能再叠加量化。唯一的组合约束是 **head 共享与 latent 投影二选一**——
  两者都在改 KV 的组织方式，冲突。
- **行 0（MLA）必须排在最前**：矩阵吸收是**硬判别子**——一个 MLA 算子常同时满足行 1/1b
  （DSV4 稀疏选择）、行 2（页表寻址）、行 5（latent 投影）或行 7（FA 分块）的特征，
  被任一行抢先后都会误路由到 `block_sparse_attention`/`gqa`/`flash_attention`，拿错的 L1
  校验 MLA 草图。MLA 的目标函数与这些都不同：头号约束是**可证仿射**（`BLOCK_QO=1` 固定，
  head 交错 cache 的展平索引按 `mla.md` L1.2 的公式逐字复刻参考）；KV 组内共享 ＋ split-KV ＋
  16-bit 输入 `P_SPLIT` 双点积；掩码用有限常量、softcap 用 native `tl.math.tanh`、lse 底数
  逐字对照参考（`mla.md` L1.11 / F8）。FA 的主力手段「放大 tile 摊薄 CV 同步」在分页/吸收
  形态上会撞 UB 墙，不适用。边界：`mla-vllm` 非吸收变体与 `mla-preprocess` 上游生成阶段按
  `mla.md` §0.1 单独核对，仍由本行路由到 `mla.md`。
- **行 1/1b/2 排在行 7 之前**是刻意的。分页/块稀疏同时满足两行条件，但
  `flash_attention.md` 的主力手段「放大 tile 摊薄 CV 同步」在分页形态上**会直接撞 UB 墙**
  （实测多页合并三种写法全部 `PlanMemory Failed`），且「`p` 二段拆分」在 FA 上是 **+2.7%**、
  在分页 GQA 上是 **−49%**。两类的目标函数不同：FA 是「互有 UB 依赖的向量算子数 × KV 迭代数」，
  分页 GQA 是「`tl.dot` 的**个数** × KV 页迭代数」。
- **行 1 / 1b 是「三·2 块跳过」小类，走 `block_sparse_attention.md`；`gqa.md` 负责的是
  「二 压缩类·head 共享 / 路由」小类；「二·3 页表寻址（不跳块）」小类走 `attention-paged.md`（行 2）。**
  行 2 原指向 `gqa.md` §3.1/§3.4，已改指：`gqa.md` §3.1 骨架自带 topk 跳块选择，与行 2
  「不跳块」判别特征错位；且 op37/38（纯页表 decode/prefill）的核心 Layer 1——block_ptr
  分核 19× 杠杆、helper op 顺序触发流水线、aux_mask 表 load、split-q persistent——在
  `gqa.md` 中不存在。批量 GQA decode 形态的落地轨迹进一步佐证：
  该形态的两个头号 Layer 1——**golden 舍入链精度契约**（golden 自带 `softmax(...).to(dtype)` /
  `einsum(...).float()` 显式 cast 链时必须复刻舍入而非提精度，两轮 KV 循环落地）与
  **aclnnMuls 标量 fp16 预舍入**（失败集合严格按 dtype 划分的指纹）——同样只在
  `attention-paged.md`（L1.9-L1.11、§6），`gqa.md` 不覆盖。`gqa.md` 的页内 `[K|V]` 布局
  （§3.4）与 tile·UB 预算（G3/§5.2）仍作为行 2 的补充参考，并在行 1 交叉时继续主责。
  分工依据见 `level4写法分类体系.md`：
  块跳过与 head 共享是两个并列小类；**同时**具备选择层与取数层替换的交叉写法归**稀疏·块跳过**
  （循环范围改写会重塑整个 kernel 骨架，取数层替换只是几行索引），所以行 1 以前者为主。
  按本文件"新增子类必须给出实测差异"的规矩，块跳过**不能**直接并进 `gqa.md` 的 Layer 1，两条依据：
  1. **本小类的选择集合是从「掩码语义」推导的，不是从数据打分 top-k 得来**——
     入参是块级掩码 / 每头类型码 / streaming 参数 / `cu_seqlens` + `causal`，
     参考实现用 CPU 上可展开的稠密掩码表达它，于是最容易犯的错是**把建表搬到 host**。
     实测同一算子两种写法的口径差 **2.09×**（host 建表 7842 vs device 建表 3758），
     前者是不合规的虚高。这条红线（`L1.2`）与配套禁令（`L1.7` 禁止复制参考实现的 mask helper）
     `gqa.md` 不覆盖。
  2. **本小类含反向写法**（入参给 `softmax_max` / `softmax_sum` / 前向输出，出 dq/dk/dv）：
     需要 q 主序与 k 主序**两张**选择表、三 kernel（delta / dq / dkdv）结构，两张表都必须 device 侧建。
     `gqa.md` 的骨架是前向 / decode 形态，没有对应 Layer 2。
- **行 4（稠密 GQA）两个文件都读**，`gqa.md` 的 §2/§5 优先——它的 §3.2 是稠密骨架，
  而 FA 的分块流式经验同样适用。行 1 同理，两份都读：`gqa.md` 给 GQA 主链与 tile/UB 预算，`block_sparse_attention.md` 给选择表构建。
- **行 6a（含投影段）排在行 7 之前**是刻意的：含 4 个投影的 MHA 形态同时满足行 7 的
  分块流式特征，但实测**最大的一笔收益在投影侧**（CrossAttention 权重预转置 -15%，
  见 `mha.md` §4.2）——若先命中行 7，首次生成只拿 FA 卡的 attention 侧约束，
  投影段头号杠杆（`flash_attention.md` L1.12）要靠 Phase 4 十几轮 rediscover
  （误分类事故见 `mha.md` §4.5）。行 6a 为交叉行：投影段看 `mha.md`，
  attention 主链 Layer 1 仍由 `flash_attention.md` 提供，两份都摘录。
- **邻域展开形态（行 9）命中即以 `sparse_unfold` 为主 category，不再落到行 6a**：
  HaloAttention 同时满足行 6a 的"投影段 + 三段式"特征，但 `F.unfold` 邻域展开是比重投影段
  更强的写法特征——它重塑整个 kernel 骨架（unfold 物化消除、恒单块 softmax、NCHW↔token
  布局变换），且 op50 实测**头号失败源全在展开侧的舍入契约**（`sparse_unfold.md` L1.1
  aclnn 边界行布局 / L1.2 softmax 分 dtype 配方，缺一即 96-99% 元素 ulp 级全错），投影段
  只是几行 L1.11/L1.12 复刻约束。主 category 取 `sparse_unfold`，投影段约束由交叉行加读
  `mha.md` 补齐（与行 1"循环范围改写重塑骨架优先于取数层替换"同一论证方式）。
- **空间 token 带插段形态（行 10a）命中即以 `spatial_attention` 为主 category，不再落到行 6a**：
  这类算子同样满足行 6a 的"4 投影 + 三段式"特征，但两段空间插段各有 6a/FA 卡不覆盖的
  头号失败源与收益杠杆——① 空间缩减插段（本例 depthwise conv，AvgPool/stride-conv
  同构）**重塑 K/V 生产链**：`Lk` 由网格几何推导
  （`ho=(H+2·pad−K)//ratio+1`），depthwise conv 权重不预转置 `[K²,D]` 时列主序碎段
  load 单项拖慢 **8.5×**（1345→157us，load-only 对照 138us 实测），邻域循环持大 tile
  会被静态物化 K² 份（UB 溢出 17.2Mb）；② transform conv＋instance_norm 链**重塑
  softmax 后处理**：IN 是 softmax 值的全平面 `(b,h)` 归约 ⇒ S 必须物化，目标函数变成
  S 显存流量（基线 8 遍 → 终态 5 遍，靠平面统计恒等式 `mean≡1/Lk`、`Σp²=Σq2ᵢ/Zᵢ²`），
  pad 行 `0×inf=NaN` 统计污染是 0/50 全 NaN 的头号失败源；③ golden 全 `.float()` 的
  **fp32 全链路契约**推翻 FA 卡 L1.16 三档 cast 在本类的适用。实测 7 轮归因中最大两笔
  收益均在插段侧（conv 权重预转置 +18% 轮、S 8→5 遍），若先命中行 6a，这些插段
  失败源无 Layer 1 管辖（同一条规则写在 Layer 2 连续 5 轮无效的机制，见上）。
  投影段约束由交叉行加读 `mha.md` §2.5/§4.1 补齐，attention 主链仍参照
  `flash_attention.md`（经 `spatial_attention.md` §9 分工表索引）。
- **行 10c（无投影双分支）排在行 10/11 之前是刻意的**：本类算子的 position 分支单独看
  满足行 10 的「flatten token + 三段式」特征，但按行 10 路由只拿 FA 卡会撞上三个实测
  首轮全挂的失败源——① **PV 升精度异构 dot**（p 真 fp32 × v 升 fp32，AccuracyError 与
  NaN 断言混合出现；FA 卡 L1.1 禁的是"操作数升 fp32"，本类是 p 本就 fp32 撞 v 强升 fp32
  的异构路径，hi+lo 二段拆分配方 FA 卡未以这种形态落过）；② **双分支物理布局错位**
  （position 按 (B,N,C) 物理写 vs channel 按 (B,C,N)，残差融合读错地址 + 最终 view 错位——
  FA 卡单分支无此问题）；③ **[128,128]+kernel 内残差融合的 hivm-plan-memory 失败**
  要求面积预算收紧到 8192/UB 96K（FA 卡 16384/150K 是无残差融合的预算）。与相邻行的
  分界——**vs 行 10/11**：本类每分支有 conv 前置（非「无任何插段」）且双分支对偶；
  **vs 行 10a**：10a 的插段是空间缩减下采样+LN 或 transform conv+IN 且带 4 个
  `nn.Linear` 投影，本类无投影、conv 前置为分辨率保持型（`padding=k//2` 不改分辨率，
  Lk=Lq）；**vs 行 10b**：10b 是卷积-块化窗口 transformer（patch 展开 + qkv/out 投影 +
  transformer 块 ×depth 堆叠 + 尾部 `conv(cat)` 融合），本类无 patch 块化布局变换、
  无任何投影、单块双分支残差相加而非 depth 堆叠；**vs 行 13a**：13a 双分支是特征图门控、
  无 softmax(QKᵀ) 主链，本类两分支都是完整三段式；**vs 行 14**：14 是聚合-分发
  （KV 维极小/逐通道权重），本类 position 分支是 N×N 全量 token 自注意力。实测差异：
  本类首个落地（DAModule/DANet 系写法，50 case）按本卡 Phase 3 一轮 geomean 2.4313、
  精度全过、target 2.0 达标，Phase 4 三轮（tile 分档/conv tile 收缩/constexpr 化）
  全部证伪、IR 无新建议——**本类 Phase 3 架构即终态**（证伪全表见
  `position_channel_attention.md` §5.2）。
- **行 6b 排在行 7 之前是刻意的**：复合掩码（BigBird/Longformer）自带 window 成分，若先命中
  行 7 的「window mask」子句，整套掩码经验会被降级为 FA 区间收缩。行 6b 与行 8/1b 的分界是
  「掩码能否表达为**共享块网格**」：块跳过（行 8/1b 的生成目标）要求掩码块粒度一致可跳，
  而行 6b 的形态逐行列集不同（per-row randperm）、残基类散布全行（元素周期）、或根本没有
  `-inf` 门控（加性偏置）、或稀疏已在 host 侧消费（窗口预切分 / 轴向分解）。
  实测：op37 随机列 **host 物化 + cache 67.02×**（对随机掩码强行区间收缩被 BB4 首版证伪）；
  op25 残基类分解两版 **−6%**（dense 扫描即终态）；op51 复合带 kernel 内算术化 **2.77×**；
  op14 rel_bias **4.01×**；op35 轴向分解（生效层 (a) 首落地）**2.4388**——双轴投影
  token 集恒同 ⟹ 投影单发射吃两轴是头号结构杠杆，弱 case 是 MTE2 权重流带宽墙
  （MMAD 仅 10-19%，tile 平衡点 128/128，扩档三连证伪）——制胜手段（生效层路由 /
  掩码算术毒性 M7 / 单 vsel M10 / 对角保底豁免 M9 / fp16 位级三连 M11 /
  掩码物化分流 M12 / 双轴投影单发射）全部在 `sparse_mask.md`，
  `block_sparse_attention.md` 不覆盖。行 7 保留给单一带型/因果类 FA（算术可推导、区间收缩即可）。
  **backward 注记（op105 FlexAttentionBwd，0819）**：FlexAttention 家族（mask_mod 通用掩码）的
  backward 形态——参考仅 causal+GQA 时按写法命中行 7＋行 4 交叉（`flash_attention.md`＋`gqa.md`
  生成，geomean 1.0705）；其反向骨架（lse 已知免 delta kernel、4 kernel 双路径 `USE_WS=(Sk>=256)`、
  点21 WS 物化 fp32 cast 消 mix-kernel GM 回退 +9.5%）已回填 `sparse_mask.md` §3.7，
  本行（三·1 非均匀掩码）任何成员的 backward 变体直接复用该骨架，块稀疏 backward 再叠加双选择表层。
- **行 6c/6d 排在行 7 之前**同理：反传链（6c）与无 head 维单头大 C SDPA（6d）的
  参考实现都会命中行 7 的「KV 分块循环」特征，但两者的头号硬约束都不在 FA 卡经验域——
  6c 是 **fp16 cube 内部累加固定 fp16（MERE 48x，三次证伪，唯一合规路径全 fp32 ieee dot）**
  与 **bishengir 动态循环三禁**（`mha.md` §7.2/§7.3），FA 卡没有反传链 Layer 1；
  6d 的 C=64~1024 **超出 FA 卡 head_dim ≤ 256 经验域**，头号杠杆是 PV1/SPLIT_D 的 UB
  解锁与 tile 桶治理（`mha.md` §6），误入行 7 会拿一套只覆盖一半目标的约束。
  **backward 分流（6b 注记 vs 行 6c 的分界）**：非均匀掩码（FlexAttention/mask_mod）家族的
  backward 先命中行 6b，复用 `sparse_mask.md` §3.7 反向骨架（lse 已知免 delta kernel）；
  纯 SDPA / 仅 causal 的 backward 才落行 6c（三件套以 delta 重算为前提，`mha.md` §7.1）。
  两者 S_K ≥ 256 都走 workspace 物化，但骨架前提不同，不互替。
  6c/6d 均为交叉行：`mha.md` 给本细分硬约束，attention 主链 Layer 1 仍由
  `flash_attention.md` 提供，两份都摘录。
- **行 14（CV-Attn 聚合分发）命中即用 `cv_attn_agg.md`，不落入相邻行**（op59/op42/op47 均落地，
  `cv_attn_agg.md` §0.2 三形态）。与既有行的分界——① **vs 行 10a（spatial_attention）**：10a 仍是
  token 间序列 attention（`Lq=H·W`、K/V 生产链含空间缩减下采样层 + LN、有 `QKᵀ` 主链），CV-Attn 的
  attention 是特征门控，KV 维极小（`k=3` 分支 / 逐通道 / softmax-over-hw），无 token 网格几何推导、
  无下采样层重塑 K/V 生产链；② **vs 行 13a（polarized_attention）**：13a 是双分支加性融合
  `out=x·(S+T)`，无「聚合→分发→聚合回空间」三拍链；③ **vs 行 6a（mha）**：CV-Attn 即使有 4 个投影
  （op59），主链也**无** `softmax(Q@Kᵀ/√d)@V` 三段式——op59 是 `softmax(mean(q,dim=1))@v` 通道注意、
  op42 是 grouped conv + 空间/通道双 attention、op47 是 softmax-over-hw + bmm 汇聚，均非标准序列
  attention。实测差异（`cv_attn_agg.md`）：本类主成本在投影 GEMM 而非 attention（C5）；路由决策漏
  cube tile 深度项会让大 shape 集体回退、几何平均崩到 0.4961x（C6）；fp16/bf16 中间量逐 conv 舍入须
  RNE 位运算复刻、fp16 域加法对齐 torch（C1，3.7e-2 vs 阈 9.8e-4）；结合律重排须配 shape 签名精度
  门控（C7，不门控 14/50 超阈）——这些硬约束行 10a/13a/6a 均不覆盖。
- **一 标准类的三个细分（基础三段式 / 分块流式 / 空间 token 版）统一走 `flash_attention.md`；**
  **含 4 个 `nn.Linear` 投影的变体（MHA / SelfAttention / CrossAttention）走行 6a、
  反传链变体走行 6c、无 head 维单头大 C SDPA 走行 6d——三者均为
  `mha.md` ＋ `flash_attention.md` 交叉行。**
  依据是分类体系里这一大类的判断：「同一套模板，参数化即可」。原先另有一份 `attention.md`
  承接朴素三段式，已退役——它的 L1.1（禁用 fp32 `tl.dot`）/ L1.3（`next_pow2`）/
  L1.6（全程 fp32）三条在 FA 路径上已被实测推翻，两份并存时最常见的故障就是
  朴素卡片的约束被套到 FA 算子上，让 Step 3 门禁把正确草图判为 A 类错误。

## 无 template 的子类怎么办

行 5/6（latent 投影、量化）、行 12 的**结合律重排**细分、行 13（专用类，除 13a 外）尚无专属 template
（行 12 的**状态递推**细分已归档 `linear-recurrent.md`）。
按 Step 1 的既有机制处理：
标 `new_category`，先回落到表中给出的 `template_path`（没有则不加载 Layer 1 约束），
**草图通过后新建 `.claude/template/{category}.md` 并回填 Layer 1**。

⚠️ 行 5 的 latent 投影若同时是**矩阵吸收**（`o = softmax(s) @ ckv`，无独立 V）→ 不是
"暂无专属 template"，走**行 0** `mla.md`（已有专属卡片）；只有纯展开式 latent 投影
（`K = W_uk @ kv_c` 后仍按标准三段式 `softmax(QKᵀ)V`）才回落 `flash_attention.md`。

⚠️ 线性类（四）与专用类（五）**不要**回落到 `flash_attention.md`：它们没有 softmax，
FA 的 online softmax 约束整套不适用，误用会让 Step 3 门禁把正确草图判为 A 类错误。

## 新增子类的规矩

再出现新的 attention 变体时，**在本表加一行，不要回头去改 `AGENTS.md`**——
`AGENTS.md` 只保留"attention 家族 → 读本文件"这一跳。

新增行必须同时给出：可模式匹配的**写法**特征（打开 kernel 能对上的骨架，不是降本手段的名字）、
所属大类·细分、`category`、`template_path`，以及**它为什么不能落到已有的某一行**（附实测差异，
像上面行 1/1b vs 行 7 那样）。拿不出实测差异就不要新开子类——加进已有 template 的 Layer 1 更可靠。

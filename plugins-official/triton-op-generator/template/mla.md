---
name: mla
description: MLA 类算子（MultiHeadLatentAttention 族：BatchMLAPagedAttention / BatchDecodeMlaPaged / TRT-LLM MLA / DSV4 稀疏 MLA / vLLM MLA）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束、Layer 2 算法骨架、Layer 3 关键技巧与 Phase 4 优化点清单
metadata:
  type: reference
---

# MLA 类算子优化经验

本文档是 **MLA（Multi-head Latent Attention）矩阵吸收主链**这一族算子的经验合集，覆盖 Phase 2/3/4：

- **§1 通用经验**：跨形态共有的工程约束
- **§2 Layer 1 设计约束**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表**

> ⚠️ **本文件与 `flash_attention.md` 的分工（重要）**：
> FA 卡处理"连续寻址的 `Q@Kᵀ → softmax → @V`"，本卡处理 **矩阵吸收（`ckv` 同时作 K 与 V）＋ q 拆 `nope`/`pe` 双 dot（或融合维 kernel 内切分）＋ 分页/块表间接寻址**。
> 满足 §0.1 判别特征的算子**一律使用本文件**，不要读 `flash_attention.md`——后者的双 dot 打分 / 页表间接寻址 / `BLOCK_QO` / `P_SPLIT` / lse 底数（逐字对照参考） / split-KV 六项与 MLA 不同（差异与证据见 §9）。
>
> **证据基础**：`BatchMLAPagedAttention` 全量 case 的完整优化轨迹——基线（reduce kernel 平均耗时病态）→ 结构重写（片上 acc + KV 共享 + split-KV + UB 自适应 + P_SPLIT）→ **`+ BLOCK_QO=1` 仿射化修复（决定性一步，见 §4.1）**。Ascend910B2C，triton-ascend 3.2.x / CANN 8.5.x。
>
> ⚠️ **核心优化哲学**：本类算子的头号约束是 **Ascend 后端的"可证仿射（provable affine）"**——任何由运行时 `div`/`mod` 推导的行号都会触发逐元素标量 scatter（单 `[16, Dc]` store 毫秒级、甚至间歇死循环）。**宁可缩小 `BLOCK_QO` 换仿射，也不要多 query 混排。**
> 生成时**禁止混用**其他类别的经验——尤其不要套用 `flash_attention.md` 的 `BLOCK_Q` 大 tile / `kv_lo` 对齐策略。

---

## §0 适用范围与算子分类

| 算子 | 子类标签 | 计算特征 | 优化哲学 |
|------|---------|---------|---------|
| BatchMLAPagedAttention | `mla-paged` | FlashInfer 家族，`q_len>1`（MTP / spec-decode）+ 增量因果掩码；`q_nope`+`q_pe` 拆分，`ckv_cache`/`kpe_cache` 分页 | **主样例形态**：结构重写 + `BLOCK_QO=1` 仿射化是端到端最大收益（§4.1）；`q_len>1` 时主循环变二维 |
| BatchDecodeMlaPaged | `mla-paged` | FlashInfer BatchDecodeMlaWithPagedKVCache（已废弃的 decode 形态），单 query，`window_left` + `logits_soft_cap` | decode 子集：主循环单 query、掩码 constexpr 特化；结构**几乎直接复用 paged 主骨架** |
| TrtllmBatchDecodeWithKvCacheMla | `mla-paged` | TRT-LLM：query/kv_cache 均为**融合维度**（`kv_lora_rank`+`qk_rope_head_dim`=576），`bmm1_scale`/`bmm2_scale`，block_tables 稠密/稀疏 | 拆分 nope/pe 与融合维只是**布局差异**，核心吸收模式相同；kernel 内按 constexpr 切分（L1.10） |
| TrtllmBatchDecodeSparseMlaDsv4 | `mla-sparse` | DSV4：SWA 池 + 压缩池双分页 KV，sparse_indices 扁平槽索引（前 128 列 SWA），sinks | paged 主骨架 + **稀疏 gather + 双池**；`-1` 索引掩码为 0（§3.3） |
| MultiHeadLatentAttention | `mla-vllm` | vLLM 风格：`kv_cache [pages, block_size, H, fused]` 融合 `head_dim=64`，block_table + cache_seqlens + causal；**非矩阵吸收**（K 用全 64 维、V 取前 `headdim_v` 维切片，`scores = q@kᵀ/√d`） | 共享分页间接寻址，但主链数学不同（§0.1 Q0 / §9 边界）；生成时单独核对参考语义 |
| MlaPreprocessOperation | `mla-preprocess` | **非 attention**：低秩投影（`q_lora_rank`/`kv_lora_rank`）+ RMSNorm + GPT-J RoPE，从 hidden_states 生成 q_nope/q_pe/kv | 上游生成阶段，GEMM 为主；走 §3.4 GEMM 骨架 |

**共性核心模式**（吸收主链各形态一致；`mla-preprocess` 为上游生成、`mla-vllm` 为非吸收变体）：
- 矩阵吸收：`ckv` 同时作 K 和 V；`o = softmax(sm_scale·(q_nope·ckv + q_pe·kpe)) @ ckv`；
- 分页/块表 KV 映射（页乱序可冗余）；decode 语义（单 query token 注意全部或子集 KV）；
- `lse` 底数逐字对照参考 kernel（F8），`return_lse_base_on_e` 可选；
- 布局差异三种：**拆分双张量**（`q_nope`/`q_pe` + `ckv_cache`/`kpe_cache`）、
  **融合单张量**（query/kv_cache 内沿 head_dim 拼接 `kv_lora_rank+qk_rope_head_dim=576`，kernel 内切分 nope/pe；非吸收变体的 kv_cache 也是融合 64 维但 **V 取前缀切片**）、
  **双池稀疏**（SWA + 压缩池，`-1` 槽掩码）。

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. **矩阵吸收**：`ckv` 同时作 K 与 V，`o = softmax(scores) @ ckv`（不是 `@V` 独立张量）；
2. q 拆 `nope`/`pe` 双 dot（或融合维 kernel 内按 constexpr 切分）+ 分页/块表间接寻址（虚拟位置 → 物理 cache 行）；
3. `lse` 输出（底数逐字对照参考 kernel：带 `return_lse_base_on_e`/`log2` 语义 → base-2；参考用 `tl.log` → base-e，见 F8/L1.8）。

**Q0（先用本文件前的第一问）是否为吸收主链**：`mla-vllm`（vLLM 风格非吸收变体）的 K 用全 64 维、V 取前 `headdim_v` 维**前缀切片**，为**边界算子**——共享分页寻址与 UB/仿射经验，但主链数学不同，生成时需单独核对参考语义；`mla-preprocess`（MlaPreprocessOperation）是上游 GEMM 生成阶段，不在本卡主链。
不满足上述任意一条（如无分页的连续寻址 `Q@Kᵀ→softmax→@V`）→ 退回 `flash_attention.md`。

### §0.2 ★ 形态识别四问（**Phase 2 第一步必须回答，答案决定后续哪些章节适用**）

| # | 问题 | 影响 |
|---|------|------|
| **Q0** | 是否为吸收主链（`ckv` 作 V）？ | 否 → `mla-vllm` 类边界算子，单独核对参考语义；`mla-preprocess` 走 GEMM 骨架 |
| **Q1** | 双张量拆分 or 融合维（kernel 内切分）？ | 融合维 → §3 布局特化；`query [..,576]` 按 constexpr 切 q_nope/q_pe |
| **Q2** | 页表布局（indices 乱序 / block_table 稠密 / sparse `-1` 掩码）？ | 决定 §4.1 仿射约束与 §3.3 稀疏 gather 骨架 |
| **Q3** | 参考精度（fp32 全程 or 原生 dtype）？ | 决定 `P_SPLIT` 是否必开（16-bit 输入），见 §6 |
| **Q4** | `q_len` 是否 >1（增量因果掩码）？ | 是 → 主循环变二维 `[q_len, kv]`，掩码按 constexpr 特化 |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### F1 仿射寻址 > 一切：`BLOCK_QO=1` + GH head 共享

MLA decode 的正确姿势是 **一个 program 固定处理同一 request 的 GH 个 head，`BLOCK_QO=1`**：
`rows = tl.arange(0, BLOCK_M)` 直接作为 head 索引，`h = h_group * GH + rows` 为**仿射**、`q_row` 为**标量**，
于是所有 Q/O/LSE 的 load/store 对 Ascend 后端**可证仿射**，回到向量化 store。
任何由 `rows//GH`、`rows%GH` 推导的行号即使恒为 0 也会触发逐元素标量写（见 §4.1 决定性证据）。

### F2 页表映射必须 kernel 内做，`BLOCK_KV` 与 `page_size` 解耦

虚拟 KV 位置 → 物理 cache 行在 kernel 内映射：`(pages[j // (page_size*H)] * page_size + (j // H) % page_size) * H + j % H`
（C 序 head 交错展平）。`BLOCK_KV` 按 UB 预算自适应，**不得依赖** `page_size` 对齐。

### F3 KV 复用：一个 program 服务 GH 个 head，KV tile 只 load 一次

`BLOCK_QO=1` 的收益来源——GH 个 head 共享同一 KV tile，KV 只加载一次（生成版是每个 head 各自扫一遍 KV）。

### F4 16-bit dot + fp32 累加；fp32 输入保持 fp32 dot

对齐参考算术路径：fp32 输入走 fp32 dot，不要强行降到 16-bit。

### F5 16-bit PV 精度判据式：先单 f16 dot，超差再 P_SPLIT 双点积（非 blanket）

16-bit 输入**先试单次 f16 PV dot**（`p.to(f16)`，尾数 11bit，多数场景够用）直接过验证闸门；
只有单 dot 超差（紧闸门 / 大数值范围）才用第二次 cube dot 补回（`p_hi + p_lo` 双点积，
对齐标杆 fp32 全程）。**不要一上来就双点积**——省一次 cube dot。

### F6 split-KV 长 KV 才开 + reduce 标量循环

`ns = max(1, min(8, max(ns_par, ns_kv)))`；长 KV 切到多个 program，小 reduce kernel 归并 partial `(acc, m, l)`。
reduce kernel **禁止** 3D masked load（Ascend 后端触发 vector core timeout），改用 `for s in range(num_splits)` 标量循环。

### F7 UB 预算静态估算自适应（含编译器额外缓冲余量）

`esz*(bm*Dc + bm*Dp + 2*bkv*(Dc+Dp)) + 4*(bm*Dc + 2*bm*bkv) + (2*bm*bkv*esz if P_SPLIT else 0)` < 140KB；
bm 从 32/64（Dc≥512 取 32）往下减、bkv 从 128 往下取，取首组满足预算的配置。

> ⚠️ **预算必须含编译器额外缓冲**：triton-ascend 编译会再分配 ~10+KB 临时缓冲，静态估算
> 贴满 140KB 的配置会超 UB 编译失败；预算上限要留出该余量（实测 ~12KB 是成败分界）。

### F8 lse 底数契约 = 逐字对照参考 kernel（禁止默认 base-2）

- 参考用 `tl.exp`/`tl.log`（base-e，与 torch 参考的 `torch.log` 一致）→ **全程 base-e**：
  内部 `p = tl.exp(s - m_new)`、`lse = m + tl.log(l)`，**禁止 ×LOG2E**
  （曾有实现因 ×LOG2E 把 base-e 输出变 base-2，MERE 指纹 ≈ log2(e)−1 = 0.442695，见 §8）。
- 参考用 `exp2`/`log2` 或带 `return_lse_base_on_e` flag（FlashInfer 约定）→
  内部 base-2 + LOG2E，输出按 flag 归一。
- kernel 内只能用 `tl.log`/`tl.exp`（triton 自带自然底），**禁止 `torch.*` 接口**。

### F9 掩码区间收缩：KV 循环 trip 从容量全量缩到每 query 有效长度（paged/sparse 通用）

带掩码的 KV 扫描不要傻扫 `capacity` 全量：按形态收缩循环上界——SWA 段
`range(0, n_swa, BLOCK_N)`、压缩池段 `range(prefix_len, topk_len, BLOCK_N)`（前缀固定偏移）、
decode 段直接 `range(0, kv_len, BLOCK_N)`。每 query 实际有效长度做 trip，省掉大量空迭代。
**收缩后尾迭代的掩码仍不可删**（精度红线，见 §6）。

---

## §2 Layer 1: 设计约束（Agent 必须遵守，首次生成就要全部满足）

### L1.1 ★★★ `BLOCK_QO=1` 仿射寻址（决定性）

固定 `BLOCK_QO = 1`，`h = h_group * GH + rows`、`q_row` 标量，全部 Q/O/LSE load/store 对 Ascend **可证仿射**。
**禁止**多 query 混排 / `rows//GH` 取行号。判别：profiling 单 store 毫秒级或间歇 aicore timeout。

### L1.2 页表映射与 `BLOCK_KV` 解耦；越界页显式掩码；索引公式逐字复刻参考

页可乱序、可冗余；`other=0` 对越界页必须显式掩码（padding 陷阱，见 §8）。页表索引用 int32。

- **展平索引必须逐字复刻参考**（torch 参考的 `reshape(-1, Dc)[:kv_len]`，或参考 kernel 的
  `PAGE_SIZE*H` 展平循环）：虚拟位置 j → `(page=j//(page_size*H), offset=(j//H)%page_size,
  head=j%H)`（F2 公式）；**head 交错 cache（`H_cache == H` 且逐 head 独立数据）下
  每个 head-slot 都是序列中的独立 KV 位置**。
- **GH 分组"共享 KV / load head-0 only"仅当 cache head 维 == 1（跨 head 共享）才允许**；
  head 交错 cache 下禁止 head-0 替代整组（曾有实现只 load head-0 → H>1 全量失败）。
- 首次生成必须读参考的 reshape/切片/permute 推导索引，不得凭假设。

### L1.3 精度契约先钉死：fp32 输入 fp32 dot；16-bit 输入按 F5 判据

对齐参考算术路径；16-bit 输入先跑单 f16 PV dot，闸门超差才补 `p_hi+p_lo` 双点积（F5）。

### L1.4 KV 复用（GH 共享）

一个 program 处理同一 request 的 GH 个 head，共享每个 KV tile。

### L1.5 split-KV + reduce 标量循环

长 KV 必开 split-KV；reduce kernel 用 `for s in range(num_splits)` 标量循环规避 3D masked load 死循环。

### L1.6 mask constexpr 特化

`causal` / 增量因果（`j > kv_len - qo_len + i`）/ `window_left` / `logits_soft_cap` / `softcap` 全部作为 `tl.constexpr` 传入，
禁止运行时标量分支。

### L1.7 UB 预算自适应（<140KB 静态估算）

F7 公式；bm/bkv 双层循环取首组可编译配置。

### L1.8 lse 底数逐字对照参考（F8）

参考 base-e（`tl.log`，与 torch 参考一致）→ kernel 内 `tl.exp`/`tl.log`，禁止 ×LOG2E；
参考 base-2 / 带 `return_lse_base_on_e`（FlashInfer 约定）→ 内部 LOG2E，输出按 flag 归一。

### L1.9 host 不做 permute / contiguous / pad

分页语义下 host 侧任何重排都会破坏虚拟位置 → 物理行的映射，禁止。

### L1.10 融合维切分：kernel 内按 constexpr 切 `nope`/`pe`

`query [.., 576]` / `kv_cache [.., 64]` 在 kernel 内按 constexpr 切 q_nope/q_pe（融合维布局），host 不拆张量。

### L1.11 掩码用有限常量；softcap 用 native `tl.math.tanh`

- 掩码一律用有限常量 `NEG_INF = -1.0e9`（或 -3.0e38），**禁止 `float("-inf")`**：
  全掩码 tile/行（window 前置区、GH 组内被 mask 的 head 行）在 online softmax 中
  `exp(-inf - (-inf))` 产生 NaN。
- `logits_soft_cap > 0` 时打分后/softmax 前的 tanh **一律用 `tl.math.tanh`（native 饱和）**，
  **禁止 exp 公式手写 `_tanh(x) = (e^x−e^{-x})/(e^x+e^{-x})`**：NEG_INF 大负值
  （x = −1e9/cap ≈ −2e7）使 `e^{|x|}=inf` → `-inf/inf = NaN` → 经 `tl.max` 传播整 tile 全 NaN。

### L1.12 ★★★ cube dot 的 M 维分界：M=1 禁区（vector 归约），M≥8 的 head 组可用 cube dot

Ascend cube `tl.dot` 的 M 维分界是 **M=1 才是禁区**——整组只有 1 行的 dot
（单头 `H=1`，或 `GH` 被 UB/语义钳到 1）在 aicore 上**挂死**
（间歇 `ACL stream synchronize failed 507014`，全量 timeout）或**严重退化**。

**M≥8 的 head 组可用 cube dot**：`BLOCK_QO=1` + GH head 共享（L1.1）时 dot 的 M 维 = GH，
GH=8 实测是稳定通过且最优的配置。**不要把"M=1 禁区"泛化成"所有 M<16 都不能用 dot"**
（head 组 M≥8 的 cube dot 完全可用，且比 vector 归约快）。

**二选一，禁止不设防的 M=1 dot**：

1. **保证 head-group 落点 ≥ 8**：head 数足时 `BLOCK_QO=1` + GH head 共享（L1.1）天然满足；
   **单头（`H=1`）或 head-group 上限被 UB/语义钳到 1 时这条路根本走不通**；
2. **整体改走 vector 归约**（§3.5 / §4.6）：`grid = (total_q * H,)` 每 program 单 head，
   QK/PV 全部用 `tl.sum(逐元素乘, axis=…)`，**零 `tl.dot`**。

> ⚠️ "dot 不是瓶颈"只在 M 不落在 M=1 禁区时成立（L1.12）；M=1 时 dot 从"工具"变"地雷"。
> 判定：`H` 单头，或 `GH` 被钳到 1 → 直接走 §3.5。

---

## §3 Layer 2: 算法骨架（参考方向，输出必须是全新草图）

### §3.1 主骨架（`mla-paged`）

```
grid = (num_requests, num_splits)
program:
  load q_nope/q_pe [BLOCK_QO=1, Dc/Dp]（逐 request）
  m/l/acc 初始化为 -inf / 0 / 0（fp32 片上）
  for j in range(0, kv_len, BLOCK_KV):
    虚拟 KV 位置 → 页表映射 → 物理 cache 行（kernel 内，int32）
    load ckv tile / kpe tile
    s = dot(q_nope, ckvᵀ) + dot(q_pe, kpeᵀ)；s *= sm_scale
    mask 特化（causal / window / softcap）
    m_new = max(m, rowmax(s))；p = exp((s - m_new))（base-e，参考用 tl.exp 时；参考 base-2/exp2 时走 F8 base-2 写法）
    l = l*exp((m - m_new)) + rowsum(p)；acc = acc*缩放 + dot(p, ckv)
  split-KV：写 partial (acc, m, l) → reduce kernel 归并
  输出 o [BLOCK_QO=1, Dc]；lse 底数逐字对照参考（F8）
```

### §3.2 `mla-paged` decode 特化

无因果、可 window 收缩（`window_left` 区间折叠）、`logits_soft_cap` 用 `tl.math.tanh`
（native 饱和，softcap 在打分后、softmax 前；**禁止 exp 公式手写 tanh**——NEG_INF 大负值
会溢出 NaN，L1.11）。

### §3.3 `mla-sparse` 稀疏 gather 骨架

SWA 128 前缀 + 压缩池双池；`sparse_indices` 扁平槽索引，`-1` 索引掩码为 0（不参与 softmax）。

### §3.4 `mla-preprocess` GEMM 骨架

低秩投影（`q_lora_rank`/`kv_lora_rank`）+ RMSNorm + GPT-J RoPE，GEMM 为主，不在吸收主链。

> ⚠️ **GEMM 链的性能注意（有成功先例，大 B 仍是重灾区）**：
> 多 kernel 拆解（GEMM→RMSNorm→GEMM→RMSNorm→RoPE，中间 buffer 落盘 GM）在小/中 B
> 有明确收益；但大 B（4096–8192）仍因**小 tile + 大 grid** 剧烈劣化——固定小 BLOCK（如 64）时
> `grid=(cdiv(B,64), cdiv(N,128))` 可达数千 program，每个 program 计算量小，调度/发射
> 开销主导；且 K 循环内反复重载逐行归一化系数放大约束。
> 成功配置（泛化方向）：
> - **grid 封顶 + 持久化交织循环**：`grid=(min(tiles_m*tiles_n, cube),)` +
>   `for tile in range(pid, total, nprog)`；**禁止无界 2D grid**——program 数随 M×N 二次
>   膨胀时 tile 数与核数不匹配（每核串行链过长），且超量并发写 GM 有 race 风险；
> - **BLOCK_M 按 T 自适应**（小 T 32 / 中 T 64 / 大 T 128）、BLOCK_N=128、BLOCK_K 取大（256）；
> - RoPE 用**奇偶列分段 2D load + `tl.interleave`**（避开 3D load + `tl.split`/`tl.join` 的
>   UB 临时缓冲与后端 miscompile 面）；RMSNorm **两遍扫描**（先整段 sum_sq 求 rstd 再乘，
>   与 ln 权重融合）；q-rope 用 `[BLOCK_T, BLOCK_H, ROPE]` 3D 块一次处理多 head 组
>   （避免"单 head 一 program"的 grid 爆炸）；
> - 数学函数统一自然底 `tl.exp`/`tl.log`，位置与频率计算走 fp32。
> 复测对比 torch 批量 `aclnnMatmul`。不要在吸收主链卡上找答案。

### §3.5 `mla-paged` 小 M vector 归约骨架（单头 / head-group 落点 =1 时，L1.12 路径 2）

```
grid = (total_q * H,)                      # 每 program 一个 (q_token, head)，零 tl.dot
program:
  q_nope_vec = load(q_nope[pid//H, pid%H, :Dc]).to(fp32)   # 1D 向量
  q_pe_vec   = load(q_pe[pid//H, pid%H, :Dp]).to(fp32)
  m/l/acc = -inf / 0 / zeros([BLOCK_DC])                    # acc 1D，PV 无 dot
  for sk in range(0, kv_len, BLOCK_SK):                     # BLOCK_SK = page_size (≤32)
    sk_offs = sk + arange(BLOCK_SK)
    页表映射 → ckv_tile [BLOCK_SK, Dc]、kpe_tile [BLOCK_SK, Dp]（kernel 内，int64 地址）
    nope_score = tl.sum(q_nope_vec[None,:] * ckv_tile, axis=1)   # QK = vector 归约
    pe_score   = tl.sum(q_pe_vec[None,:] * kpe_tile, axis=1)
    score = sm_scale * (nope_score + pe_score); 掩码有限 NEG_INF（L1.11）
    s2 = score * LOG2E; m_new = max(m, max(s2)); p = exp2(s2 - m_new)
    acc = acc * exp2(m - m_new) + tl.sum(p[:,None] * ckv_tile, axis=0)   # PV = vector 归约
  输出 o = acc / l_safe；lse = m + log2(l_safe)（底数对照参考，F8）
```

`BLOCK_SK` 取 `page_size`（16/32）保仿射；UB 内放 `[BLOCK_SK, Dc]` tile，不需 cube。

---

## §4 Layer 3: 关键技巧（技巧可参考，变量名/结构必须重新设计）

### §4.1 ★★★ 仿射 store 修复（**决定性**，SINGLE 开关 A/B）

Ascend 后端将**无法证明仿射**的 store（行号由 `rows//GH` 等除法/取模计算，即使恒为 0 也无法折叠）
lowering 为逐元素标量写（且该路径间歇死循环）。
**修复**：`BLOCK_QO=1` 固定，`h = h_group*GH + rows` 为仿射、`q_row` 标量 → 回到向量化 store。

### §4.2 P_SPLIT 双点积（16-bit 输入）

`p` 转 16-bit 的精度损失用第二次 cube dot 补回：`p_hi`（高段）+ `p_lo`（残差）双点积，对齐 fp32 全程。

### §4.3 页对齐大块加载（剩余空间方向）

`BLOCK_KV` 受 UB 限制（Dc≥512 fp32/bf16 大 case，KV 扫描本身 dominate，bkv≈32）。
可考虑页对齐大块加载（一次拉整页）进一步降低迭代开销（§5.1 末位方向，未验证）。

### §4.4 host 侧倒数乘法（sm_scale）

`sm_scale` 用 host 侧倒数乘法传入，kernel 内不做 fp32 张量除法。

### §4.5 BLOCK 开到 UB 上限

F7 预算内把 `BLOCK_KV` 取到最大，压 KV 迭代数；**但 `BLOCK_QO` 恒为 1**（L1.1）。

### §4.6 vector 归约替代 cube dot（小 M 专用，L1.12 路径 2）

`BLOCK_QO=1` 单 query 时 QK/PV 可**整体改 vector FMA 归约**，`acc` 降为 1D：

```python
nope_score = tl.sum(q_nope_vec[None, :] * ckv_tile, axis=1)      # [SK]（QK）
pe_score   = tl.sum(q_pe_vec[None, :] * kpe_tile, axis=1)        # [SK]
acc = acc * delta + tl.sum(p_masked[:, None] * ckv_tile, axis=0) # [Dc]（PV）
```

- 与 GH≥8 的 cube 版相比是**降级**（vector 归约比 cube 慢），**只在 M=1 时用**——
  此时 cube 是地雷（L1.12），vector 归约反而更稳。
- **Ascend 前端陷阱**：`tl.dot` 返回的 2D tensor **不支持标量索引 `[0]`**
  （`ValueError('unsupported tensor index: constexpr[0]')`）——曾有实现想用
  `USE_DOT` 开关把 vector 归约改回 `tl.dot(q_2d, ckvᵀ)` 提速，编译即失败，回退 vector 版。
  优化时不要走这条路；需要 2D dot 时直接用归约结果，不做 dot 2D 标量索引。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| 优先级 | 方向 | 作用/适用 |
|---|---|---|
| 1 ★★★ | 仿射 store 修复（BLOCK_QO=1） | 消除非仿射 store 病态（核心约束） |
| 2 ★★ | KV 复用（GH head 共享） | 结构重写阶段的组成项 |
| 3 ★★ | 16-bit PV 单 f16 dot，超差才 P_SPLIT（F5） | 精度过闸门的前提（判据式） |
| 4 ★ | split-KV + reduce 标量循环 | 长 KV 病态消除 |
| 5 ★ | UB 预算自适应 | 让 bkv 取到上限，压迭代数 |
| 6 ★ | M=1 用 vector 归约替代 cube dot（L1.12） | 规避 M=1 cube dot 禁区 |
| 7 | 页对齐大块加载 | 剩余空间方向，未验证 |

### §5.2 ⛔ 结构性下限：小 shape 上**绝不要**拆 kernel

MIX 每次发射 ~4.5us 地板；decode 单 request 数据量小时拆多 kernel 只会被启动开销吃光（同 FA 卡 §5.3 访存账结论）。

### §5.3 会误导的 profiling 字段

- **profiler 伪影实为真慢**：偶发 aicore timeout / 设备 hang 常被归为"伪影"，实为逐元素标量 store 的真实耗时。
- 病态耗时集中在 **ns=1 的 fwd 和多数 case 的 reduce**——先按 Name 拆开看，不要整 kernel 调。
- **aicore timeout（507014）归因顺序（先 kernel 后环境）**：`ACL stream synchronize failed` 是
  异步 kernel 在同步点暴露的真慢/挂起——先查 **M=1 的 cube dot**（L1.12）、再查非仿射 store 逐元素化
  （L1.1）、再查 **`kv_len/BLOCK_KV` 串行迭代爆炸**（长 KV + 小 BLOCK_KV，split-KV 只除 NS 倍），
  全部排除后才考虑环境竞争 / 参考侧问题；**禁止**判为"瞬态设备竞争"而跳过修 kernel。

---

## §6 精度闸门（先过闸门，再谈性能）

- **fp32 全程 vs 原生 dtype**：fp32 输入保持 fp32 dot（L1.3）；16-bit 输入按 F5 判据（先单 f16 PV dot 过闸门，超差才 `P_SPLIT` 双点积）。
- **lse 底数逐字对照参考（F8）**：参考 base-e（`tl.log`）就 base-e 逐位核对，禁止 ×LOG2E；
  参考 base-2（带 `return_lse_base_on_e`）才走 LOG2E 归一。

- **精度失败先分四类再动手，不要一上来怀疑 dot 精度**：
  1. matched_ratio≈0 且 H>1 全崩 → 查页表/head 展平索引（最高频，L1.2）；
  2. AssertionError/NaN 且含 window_left≥0 或 H%GH≠0 → 查掩码常量是否有限（L1.11）；
  3. NaN 且 `logits_soft_cap>0`（存在 NEG_INF 大负值）→ 查 softcap 是否用 native
     `tl.math.tanh`（exp 公式手写会溢出，L1.11）；
  4. 失败 metrics 用 fp32 阈值且 MERE ≈ log2(e)−1 = 0.442695（`return_lse=True`）→
     查 lse 底数与参考是否一致（kernel 内用 `tl.log` 去 LOG2E，不引入 torch 接口，F8）；
  5. 仅 dtype 相关小幅度超差 → 才查 dot 精度/累加路径（既有 §8"fp32 误用 16-bit dot"）。
- **失败集合与 dtype 相关**：精度失败通常集中在特定 dtype（fp32 误用 16-bit dot 的 case 集合）。
- **掩码区间收缩后尾迭代掩码不可删（纯精度红线）**：循环 trip 收缩到有效长度（F9）后，
  尾迭代的越界列 / `-1` 槽掩码仍必须显式 mask——删除会越界读/写、污染 softmax 统计，属通用红线。

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

- **几何平均口径**：官方 score 用几何平均，逐 case 记录 min/max/分布。
- **按 Name 分组**：`_mla_paged_fwd_kernel` / `_mla_splitkv_reduce_kernel` 分开统计，reduce 病态才能暴露。
- **launch_count per-shape**：一次 forward 内多次启动同一 kernel 时按**单次调用**耗时算（per-call），不要按发射次数平均（会把耗时低估为 1/L）。
- **环境闸门**：Ascend910B2C + triton-ascend 3.2.x / CANN 8.5.x；同代码同 env 字符串下结果会漂移（详见 FA 卡 §7.3）。

---

## §8 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|---|---|---|
| 仿射 store 逐元素化 | 行号由 `rows//GH` 等推导，无法证明仿射 | `BLOCK_QO=1` 固定（L1.1） |
| 页表越界幽灵列 | 页可乱序可冗余，越界页未掩码 | `other=0` 显式掩码（L1.2） |
| 3D masked load 死循环 | Ascend 后端 vector core timeout | reduce 用标量循环（L1.5） |
| reduce kernel 病态 | ns=1 case 归并路径慢 | 按 Name 分组定位 + split-KV 参数自适应 |
| `BLOCK_QO>1` 多 query 混排 | 想省启动，触发非仿射 store | 宁可缩小 BLOCK_QO 换仿射 |
| fp32 输入误用 16-bit dot | 复制 FA 卡经验 | fp32 保持 fp32 dot（L1.3） |
| head 交错展平索引错位 | GH 分组只 load head-0 的 KV，忽略 head 维参与序列 | 位置公式逐字复刻 F2；用 H=1 的 case 做快速 sanity |
| float("-inf") 掩码 NaN | 全掩码 tile/行（window 前置区、GH 被 mask 的 head 行）exp(-inf-(-inf)) | 掩码统一用有限 NEG_INF（-1.0e9 / -3.0e38） |
| lse 底数照搬 base-2 | 卡片默认 base-2，但参考 kernel 用 `tl.log`（base-e）且无 `return_lse_base_on_e` | 用 `tl.log`（triton 自带自然对数）直接输出、**去掉 ×LOG2E**；仅参考带 base-2/`base_on_e` 语义时才 ×LOG2E。kernel 内禁止 `torch.*` 接口 |
| softcap 用自定义 exp 公式 tanh 溢出 NaN | `logits_soft_cap>0` 且 NEG_INF 大负值（H%GH≠0 mask head 行 / window 前置区 / 越界列）进入 `_tanh(x)=(e^x−e^-x)/(e^x+e^-x)`，`e^大正数=inf` → `-inf/inf=NaN` | softcap 一律用 `tl.math.tanh`（native 饱和，tanh(大负)→-1 不溢出）；禁止 exp 公式手写 tanh 用于可能含大负掩码值的路径 |
| M=1 的 cube dot 挂死 | 单头 GH=1 或 head-group 上限被钳到 1，dot M 维只有 1 行 | 保证 GH≥8 或整体改 vector 归约（L1.12 / §3.5 / §4.6） |
| 掩码区间收缩后删尾迭代掩码 | 想省空迭代，越界列 / `-1` 槽未 mask | 收缩 trip 后尾迭代掩码保留（F9 / §6） |
| UB 预算贴满 140KB 编译失败 | 未计编译器额外缓冲（~10+KB） | 静态估算留余量（F7） |
| `tl.dot` 2D 结果做标量索引 `[0]` | Ascend 前端不支持 tensor 索引 constexpr[0]（编译失败） | 小 M 直接用 vector 归约 1D 结果，不做 dot 2D 索引（§4.6） |

---

## §9 与 `flash_attention.md` 的差异声明（MLA 路径上的实测差异）

| 维度 | FA 卡（flash_attention.md） | 本卡（mla.md） |
|---|---|---|
| 打分结构 | `scores = tl.dot(Q, Kᵀ) * inv_sqrt` 单次 dot | **两次 dot 相加**：`s = dot(q_nope, ckvᵀ) + dot(q_pe, kpeᵀ)`，再 `* sm_scale`；或融合维单 dot |
| 访存地址 | 连续 `[S, D]` | **页表间接寻址**：虚拟位置 → 物理 cache 行（head 交错展平） |
| 决定性约束 | `BLOCK_Q`/`kv_lo` 对齐、`BLOCK_KV` 面积 | **`BLOCK_QO=1` 仿射** + 页表 kernel 内映射 + `BLOCK_KV` 与 `page_size` 解耦 |
| KV 复用 | 无（每 (b,h) 独立扫 KV） | **GH head 共享同一 KV tile**（`BLOCK_QO=1` 的收益来源） |
| 精度补偿 | §4.6 p 二段拆分（fp16/bf16） | `P_SPLIT` `p_hi+p_lo` 双点积（16-bit 输入） |
| lse | 不强调 | 底数逐字对照参考：base-2（带 `return_lse_base_on_e`）或 base-e（`tl.log`，禁止 ×LOG2E） |
| split-KV | 提及 | **核心路径**（长 KV 必开）+ reduce 标量循环避 3D masked load |

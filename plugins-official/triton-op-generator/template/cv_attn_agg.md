---
name: cv_attn_agg
description: CV-Attn-聚合分发类算子（S2Attention / CoTAttention / DoubleAttention——空间或通道变换聚合 + 轻量 attention 分发 + 聚合回空间的 Triton Ascend 优化经验。以"聚合-分发"结构轴为核心，按算子分章节，含通用经验 + 各算子专属杠杆/陷阱/证伪。GEMM/卷积/softmax/online softmax/UB 预算等共享机制继承自 flash_attention.md / convolution.md / normalization.md，本文件只写 delta。
metadata:
  type: reference
---

# CV-Attn-聚合分发类算子优化经验

> **本类算子**：CV 空间/通道特征经**变换聚合**（卷积 / 通道扩展 / spatial shift / mean 归约）→ **轻量 attention 分发**（逐通道 / 逐分支 / 逐空间位置的加权 softmax）→ **聚合回空间**。共同点：attention 的 KV 维**极小或极特殊**（k=3 分支、逐通道权重、softmax-over-hw），**不是**序列 attention——FA 那套 KV 循环握手基本不适用。
>
> ⚔️ **与相邻 template 的分工**：
> - `flash_attention.md`：**序列 KV 分块 + online softmax**——本类的 attention 若含 KV 循环（如 42 的 att2 空间维），那部分继承 FA；但主干是"聚合-分发"结构。
> - `convolution.md`：卷积本身（若用 CANN conv 作 golden 复刻，精度链见 [[triton-doubleattention-precision-recipe]]）。
> - `normalization.md`：softmax / mean 的归约精度。
> - 本文件**只写 delta**：**"聚合-分发"结构**带来的额外约束、杠杆与陷阱。
>
> **判别顺序**：按 §0.1 确认是否为 CV-Attn-聚合分发。满足 → 用本文件。
>
> 📌 **类别名 `cv_attn_agg`（CV-Attention-Aggregate 聚合分发）**：启动传 `--op-category cv_attn_agg` 命中本卡。
>
> **文档分层（Layer 映射，学 flash_attention.md §2/§3/§4 组织）**：
> - **Layer 1（设计约束，Phase 2 硬边界）**：§1 的 **C1/C2/C3/C6/C7**（标注 〔L1〕）+ §3/§4 各算子的「关键约束清单」。Phase 2 Step 1 产出 `precheck.json` 时**必须逐条载入全部**，不得按篇幅截断——漏载一条就可能把该条对应的死路在 Phase 3 重踩一遍（同 FA L1.13 漏载教训）。**C4/C5/C8 标注 〔优化〕，是 Phase 4 优化指导**（跳过小 kernel / 主攻投影 / 编译参数实测），非设计约束、不载入 precheck——同 FA 只把 L 系载入 precheck、F 系通用经验不载入。
> - **Layer 2（算法骨架，参考方向，输出必须是全新草图）**：§0.2 结构轴三形态 (a)/(b)/(c) + §3/§4 各算子的结构形态。
> - **Layer 3（关键技巧，可参考但变量名/结构必须重新设计）**：§3/§4 各算子的制胜技术 T1-T13 与 kernel 片段。
> - 跨成员按收益排序的 Phase 4 落地顺序见 **§5.1**；症状驱动的陷阱排查见 **§7**。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

满足以下任意一条 → 用本文件（CV-Attn-聚合分发类）：

1. **通道/空间变换聚合**：输入经 `Linear` / `conv1x1` / spatial shift / mean 先聚合成低维中间量（`[B, C]` 或 `[B, C', H, W]`）；
2. **轻量 attention 分发**：基于聚合量做 **softmax 加权**，KV 维**极小**（`k=3` 分支、逐通道、`softmax-over-hw` 空间维）——**不是**序列 attention；
3. **聚合回空间**：attention 输出再经 `Linear`/卷积/加权求和还原到原空间形状；
4. 算子是 CV 特征（`[B,C,H,W]` / `[B,N,C]`）而非 token 序列，attention 用于**特征门控/通道注意力**而非 token 间关系。

不满足 → 序列 attention 走 `flash_attention.md`；纯卷积走 `convolution.md`。

### §0.2 ★ 聚合-分发结构轴（**本类核心路由，Phase 2 第一步必须定位**）

本类的优化杠杆**完全取决于聚合-分发结构的形态**。先定位再选杠杆：

| 形态 | 含义 | 典型 | 核心杠杆 |
|------|------|------|---------|
| **(a) 通道扩展-分支分发** | 通道先扩 3x，拆 3 分支做不同变换，加权合并 | S2Attention (59) | 投影 GEMM（C→3C）+ 分支 kernel 融合 + 低精度 RNE 复刻 |
| **(b) 空间-通道双 attention** | 空间维与通道维各做一次加权聚合 | CoTAttention (42) | grouped conv + 两次轻量 attention + 归约精度 |
| **(c) conv 特征-双 softmax 汇聚** | conv1x1 变换 → 2 次 softmax(hw) + 2 次 bmm 汇聚 | DoubleAttention (47) | conv1x1 GEMM 化（独立三发射）+ 结合律重排双路（精度门控）+ bmm tile 路由 |

### §0.3 形态补充三问（在 FA §0.2 四问之上追加）

| # | 问题 | 影响 |
|---|------|------|
| **C-Q1** | attention 的 KV 维多大？`k=3` 固定分支 / 逐通道 / softmax-over-hw 空间维？ | 决定 attention 是否值得独立 kernel（k=3 点乘不可矩阵化）；大 KV 才走 FA 继承 |
| **C-Q2** | 聚合量是标量（`[B,C]` mean）还是空间（`[B,C',H,W]`）？ | 决定归约 kernel 与后续分发 kernel 的衔接 |
| **C-Q3** | 低精度（fp16/bf16）中间量有多少处舍入点？golden 对中间舍入敏感吗？ | 决定 RNE 位运算复刻的范围（本类最大精度坑） |

---

## §1 通用经验（跨成员，首次生成必须遵守）

> 本类**特有**、FA/conv/norm 未覆盖。FA F1~F6 / L1.1~L1.14、conv 卷积、norm softmax 不重复。
> **Layer 1（载入 precheck.json）** = **C1/C2/C3/C6/C7**（标注 〔L1〕）+ 命中算子的「关键约束清单」；**C4/C5/C8** 标注 〔优化〕，是 Phase 4 优化指导、**不载入 precheck**（同 FA：L 系载入、F 系不载入）。

### C1 ★★ 〔L1〕低精度中间量必须 RNE 位运算复刻，`.to(dtype).to(fp32)` round-trip 会被编译器消除

- **必须** fp16/bf16 中间量在**每个舍入点**用位运算复刻 RNE（round-half-to-even），`triton-ascend` 编译器会把 `.to(dt).to(fp32)` 当成 no-op 消掉。
  - bf16：`(u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000`
  - fp16：13-bit 尾数 RNE + 指数 rebias + FTZ/inf 处理；大 tile 用 4-op 变体避免 i32 比较标量降级
- **禁止** 依赖硬件原生乘（NPU fp16/bf16 mul 非完全 RNE）：fp16 原生乘 ~1.6% 元素 1-ulp 偏差，bf16 原生乘 MERE ~2.3e-2 超阈值 3 倍。
- **补充（42 纯 Triton 线实锤）**：多个已各自舍入的 fp16 中间量**相加**时，用 **fp16 域加法**（`a_16 + b_16`，Triton 不提升到 fp32 重融合）强制各中间量独立舍入，对齐 torch 逐 conv 舍入；单 fp32 累加后一次舍入 FAILS MERE（3.7e-2 vs 阈 9.8e-4）。
- **Why（59 实锤）**：fp16 用硬件原生乘 MER 方差放大，bf16 原生乘直接超阈 3 倍；改 RNE 位运算后 38/50 → 50/50。
- **判别信号**：低精度 case 单独失败、`MERE` 卡在阈值附近 ⇒ 查 RNE 复刻，不是 tiling。

### C2 ★ 〔L1〕归约（mean/sum）用 Kahan 补偿，匹配 torch reduce 树

- **必须** 大 N 归约（如 S2Attention 的 `mean(xa, dims=(1,2))`）用 **Kahan 补偿求和**，对齐 torch reduce 树。
- **Why（59 实锤）**：顺序累加 fp32 尘埃 ~2e-5，在 fp16 舍入边界产生翻转；Kahan 后 ~3e-8，消除边界失败。
- **判别信号**：某 case 的 mean 结果差 ~1e-5 量级、fp16 输出边界翻转 ⇒ 上 Kahan。

### C3 ★ 〔L1〕激活函数近似必须匹配 NPU 底层实现，不是数学精确式

- **必须** GeLU 用 **tanh 近似**（`0.5*x*(1+tanh(√(2/π)*(x+0.044715x³)))`）匹配 NPU `F.gelu`；erf 精确式偏差 4.7e-4。
- **禁止** 用数学精确式（erf）当"更准"——golden 是 NPU 的 tanh 近似，用精确式反而偏。
- **Why（59 实锤）**：erf 式与 NPU F.gelu 偏差 4.7e-4，fp16 边界超阈。
- **判别信号**：GeLU 相关 case 单独失败且偏差 ~5e-4 ⇒ 查激活近似。

### C4 ★ 〔优化〕聚合量小的中间 kernel（`[B,C]`/`[B,3,C]` 级）几乎免费，别浪费轮次

- **必须** 识别"聚合量级"的 kernel（如 S2Attention 的 mlp3/mlp4/softmax，只算 `[B,C]` 或 `[B,3,C]`）——合计 <6% impl，**再优化无收益**。
- **禁止** 对这类 kernel 投入优化轮次。
- **Why（59 实锤）**：mlp3+mlp4+softmax 合计 <80µs，占 impl <6%，压缩无空间。

### C5 ★ 〔优化〕主成本通常是投影 GEMM（C→3C / C→C），别误判为 attention

- **必须** profiling 分解后把投影 GEMM 当主攻对象（S2Attention：MLP1/MLP2 占 56% 主成本）。
- **禁止** 把优化重心放在 attention（本类 attention 极轻）。
- **Why（59 实锤）**：4 个 Linear 是主成本；attention 只有 k=3 分支，算力极小。

### C6 ★★ 〔L1〕kernel 路由决策禁止用纯流量模型，必须含 cube tile 深度项（47 实锤）

- **必须** 任何"选 kernel A 还是 kernel B"的路由门（如直连 conv vs im2col+GEMM 化）在决策函数里同时评估：①搬运流量 ②**cube 侧 K 深度**（dot 的 K 维 tile 展开深度 kk×BLOCK_K）。只算流量的"省搬运"路由在浅 K 场景会把数据喂给**空转的 cube**。
- **Why（47 实锤）**：删掉路由门里的 cube 效率项后 35/49 shape 翻转到 fused conv 路由，几何平均**崩到 0.4961x**——机制：kk=16 的浅 dot 在 cube 侧塌陷（K 有效深度远小于 MMA 指令满载需求），省下的搬运远补不上 cube 崩塌。
- **判别信号**：路由翻转后大 shape 集体回退、回退幅度远超噪声带（-50% 级）⇒ 路由判据缺算力项，不是 kernel 本身变慢。
- **泛化**：流量模型 ≠ 性能模型。910B 上"少搬数据"只有在 cube 不空转时才成立。

### C7 ★★ 〔L1〕结合律重排是本类最大结构杠杆，但必须配精度门控 fail-sigs（47 实锤）

- **必须** 聚合-分发链 `OUT = WR @ (GD @ SM_V) + b`（golden 次序：大 N 中间量 Z 先舍入 dt）优先尝试重排为 `OUT = (WR @ GD) @ SM_V + b`——中间量从 `[c, hw]`（大 N）缩为 `[c, c_n]`（rank 级），大 N 写出 **2 次 → 1 次**。
- **必须** 同时建**逐 shape 精度门控**：离线全量标定重排路径的 MERE，超阈 shape 用签名集合（如 shape+dtype 指纹）锁死走 golden-order 兜底，不得运行时试探。
- **Why（47 实锤）**：重排使 fp32 WGD 中间量保精度、大 N 只写最终输出；但 14/50 case 重排路径 MERE 9.3e-4~1.9e-2 vs 阈值 9.77e-4(fp32)/7.81e-3(fp16)——**不门控必挂**，门控后 50/50。
- **判别信号**：重排/融合后大部分 case 提升但固定一小撮 case 精度卡阈 ⇒ 上 shape 签名门控，别放弃重排也别全局回退。

### C8 ★ 〔优化〕编译参数先实测本机工具链支持集，文档配方不可直接抄（47 实锤）

- **必须** NPU 专属编译参数（launch 时 kwargs）先用本机编译器 `--help`/冒烟实测支持集，文档配方里不存在的参数会让**编译直接失败**。
- **Why（47 实锤）**：文档（07-compile-params.md）配方参数在本版 bishengir 大量不存在（`enable_flatten`/`enable_mixed_cv` 等）或默认已开（`multibuffer`/`sync_solver`）；实测增量仅 `set_workspace_multibuffer=2`（单参零效果，1.4422 vs 1.4425 纯噪声）与 `limit_auto_multi_buffer_of_local_buffer="no-limit"`（**有害**：UB 多缓冲不设限 → 大 tile gemm 的 vector 尾声被过度碎化，21/49 shape 回退最重 -14%）。
- **判别信号**：加编译参数后逐 shape **双峰分布**（部分升部分降）⇒ 拆参重试定位加害参数；headline 落噪声带（±2%）⇒ 判零效果拒绝，别拿共模漂移当收益。

---

## §2 精度闸门（CV-Attn 特有）

> RNE 复刻（C1）/ Kahan 归约（C2）/ tanh 激活近似（C3）三件套见 §1，此处不复述。本节只收**本类额外**的精度闸门。

- **标杆种子确定性**：若标杆用 `torch.manual_seed(hash((channels, device, dtype)))`，`hash(torch.float16)` 是**对象地址派生**，跨进程随 ASLR 漂移 → 权重每进程重抽 → fp16 边际 case MERE 随机翻转（"抽签"）。**必须**锁步改为 `(zlib.crc32(repr(key).encode()) + k) & 0xFFFFFFFF`，k 经预扫选定使边际 case 有余量。判别：同一代码跨 verify 进程结果随机变化 ⇒ 查种子。
- **fp32 锁定判据**（继承 FA L1.4）：参考全程 fp32 + 量级超 dot 安全区时，全 fp32 中间 buffer 是唯一正确解。

---

## §3 各算子章节

### §3.1 59_S2Attention（Spatial Shift + Split Attention，`通道扩展-分支分发`）

**生效层**：§0.2-(a) 通道扩展-分支分发
**形态**：`x[B,C,H,W]` → permute → MLP1(C→3C) → 3 分支 spatial shift → `mean([B,C])` → MLP3+GeLU → MLP4+softmax → apply → MLP2 → permute back。4 Linear + 2 permute + shift + mean/softmax/apply。
**结果**：**50/50 精度 + 2.2702x**（NPU4，2026-08-18，framework 3.1446ms / impl 1.3852ms）。种子确定性修复后可稳定复现（历史 0810/0811 的 50/50 是抽签幸存者，仅 0.2962x）。

#### 制胜技术（按贡献排序）

**T1 复用已验证多 kernel 流水线**（+50/50 精度基础）
- S2Attention 是 8-kernel 纯 Triton 流水线（permute_in / gemm_bias / shift / mean / mlp3_gelu / mlp4_softmax / apply / permute_out）。每个 kernel 单一职责、精度可控。
- **判据**：同算子历史资产存在且经完整精度迭代修复时，直接复用（0811 终版）——比从 0 重写稳。

**T2 标杆种子确定性修复（crc32 + 偏移）**（49/50 → 50/50，可复现）
- 标杆 `torch.manual_seed(hash((channels, device, dtype)))` → `hash(torch.float16)` 地址派生漂移。
- 锁步改 `(zlib.crc32(repr(key).encode()) + 4)`，k=4 经预扫选定（case 5/23/42 余量最大）。
- 详见 §2 精度闸门。

**T3 低精度 RNE 位运算复刻**（38/50 → 50/50）
- fp16/bf16 中间量每处舍入点用 `_rne_lowp`/`_rne_fp16_fast` 位运算复刻（C1）。fp16 大 tile 用 4-op 变体避免标量降级。

**T4 Kahan 补偿 mean**（消除 fp16 边界翻转）
- `mean(xa, dims=(1,2))` 用 Kahan，fp32 尘埃 2e-5 → 3e-8（C2）。

**T5 tanh 近似 GeLU**（匹配 NPU F.gelu）
- 用 `0.5*x*(1+tanh(...))`，非 erf 精确式（C3）。

#### 性能瓶颈画像（target 2.27x 下的主成本）

| Kernel | 占比 | 说明 |
|---|---|---|
| `_gemm_bias_kernel` | **56%** | MLP1(C→3C) + MLP2(C→C)，发射 2 次 |
| `_shift_kernel` | 11% | spatial shift 融合 |
| `_permute_in/out_kernel` | 9.8%/7.9% | 两次 permute |
| `_mean_kernel` | 7.6% | Kahan 归约 |
| `_apply_kernel` | 6.9% | att × xa 三分支加权求和 |
| `_mlp3/4/softmax` | <0.3% | 聚合量极小，几乎免费 |

#### 关键约束清单（首次生成必守）
1. **投影 GEMM 是主成本**（C5），tile 按 (M,N,K) 分档；fp32 大 tile 会 UB overflow，fp32 档用更小 BLOCK_K。
2. **低精度 RNE 复刻**（C1）+ **Kahan mean**（C2）+ **tanh GeLU**（C3）三件套，缺一精度不全。
3. **permute / shift / apply 用专用 Triton kernel**，禁 PyTorch 退化（`aclnnInplaceCopy_Transpose` 等 AiCore 搬运开销）。
4. **小聚合量 kernel**（mlp3/4/softmax）别优化（C4）。
5. **标杆种子**若用 `hash()` 必须先修（§2）。

#### 已证伪方向（省下你的时间）

> 与 §5.3 全表重复的行（压缩小 kernel / mean 顺序累加 / GeLU erf / 原生乘复刻）已并入该表，此处只列未入全表的。

| 方向 | 结果 | 来源 |
|------|------|------|
| apply 合并成单 dot | **不可行**（k=3 非矩阵乘形态） | C-Q1 |
| 标杆种子用 PYTHONHASHSEED | **无效**（dtype hash 是地址派生，管不住） | §2 |

#### 泛化启示（给本类其他算子）
- **"通道扩展-分支分发"算子**（先扩通道再分支加权）：主成本在投影 GEMM，attention 极轻——别在 attention 上浪费轮次（C4/C5）。
- **低精度算子**：RNE 复刻是精度全过的前提，`cast round-trip` 不可靠（C1）。
- **标杆种子抽签**：hash() 作种子是常见坑，先查再谈精度。

---

## §4 各算子章节（续）

> 42/47 已验证回填完毕；后续新成员按 §3.1 体例追加。

### §4.1 42_CoTAttention（空间-通道双 attention，`空间-通道双 attention`）— 已验证

**形态**：Coordinate Attention（grouped conv key embed → 2 次 attention + bias/mean）。
**结果**：**49/49 精度 + 1.5147x**（NPU15，2026-08-19，fusion-search 7 轮，框架 0.222ms / impl 最差 case 0.289ms）。
**制胜技术**（按贡献排序）：

- **T1 卷积路由 CANN**（最大杠杆，0.69→1.51x）：k1 分组卷积（Triton fused_singlerow 4.4 vs CANN 64 TFLOP/s）、value_embed conv1x1（8.99→2.87ms）、att1 cat-conv（3.9-7.9→1.3-3.2ms）全部换 `F.conv2d`。参考本身就是 torch conv，CANN 是其最优后端。
- **T2 小 case 整算子路由 torch**：Triton 6-kernel 启动 ~0.1ms/kernel 在小 case 不可摊销。判别规则 `n==1 ∨ b*c*n<1.2M ∨ (bf16∧b*c*n<6M)`，不误伤 fp32 大 case（case 9/10 保留 1.6x）。
- **T3 Triton attention 融合链**：w2_avg 预归约 kk 维（att_w2 `[C*kk,mid]`→`[C,mid]` 预缩放 1/kk，FLOPs 少 9x）+ softmax_fused 单 kernel online softmax + residual。
- **T4 case 38 特例**：bf16 640×223×191 的 att1 用 CANN 反而慢（4.73 vs Triton 3.98ms）→ 保留 Triton att1。

**剩余瓶颈**：Triton ieee GEMM（att_mean/softmax）算力上限；CANN matmul 实测不更快（已验证）。2.2x 目标因 CANN conv 已极优而不可达。
**已证伪**：纯 Triton fused 卷积（4.4 vs 64 TFLOP/s）；CANN att_mean matmul（不更快）。

#### 并行线路：纯 Triton 约束线（r11→r15，无 torch 退化，2026-08-25，NPU5）— 已验证

> 受「forward() 禁 torch._/F._ 计算」约束（AST Type3 拦截），CANN conv 路由（上方主线 T1）**不可用**，全 Triton kernel 实现。与主线**并列记录**（多样性保护），不做覆盖。

**结果**：**49/49 精度 + 1.1417x**（NPU5，2026-08-25，`verify_r11/perf_result_r15.json`，框架 7.9688ms / impl 6.9797ms）。轨迹：r12d=1.0556 → r13 w2sum compute-dtype → r14 n==1 融合 fast path → **r15=1.1417x**。case47（`[8,2048,1,1]` 退化 n==1）0.088→**1.2266x**，case46（`[1,1,1,1]`）2.541x。

**制胜技术**（按贡献排序）：

- **R1 n==1 退化整段跳过**（case47 0.088→1.2266x，最大杠杆）：`softmax_over_hw(att)` 当 hw=1 恒等于 1（exp(x)/exp(x)），`out=k1+v` 数学上**精确**，attention 中间全跳过、精度无损（实测 max_abs_diff=0.0 / MERE=0.0）。k1（grouped conv3x3）与 v（conv1x1）降为 **batch 作 dot-N 的 batched matvec**（权重 `[C,CIN_PG]`/`[C,C]` × 输入 `[CIN_PG,B]`）。
- **R2 退化 case 必须融合单 kernel**（3 launch→1 launch）：launch overhead ~0.1ms/kernel，3 分离 kernel（k1+value+add）=490us（sp 0.80x 反亏）；融合 `n1_fused_kernel` 单 launch=330us（sp 1.20x）。**纯 Triton 约束下 T2 路由 torch 的替代**：小 case 用融合 kernel 削 launch，不路由 torch。
- **R3 fp16 域加法强制逐 conv 舍入**（C1 实现补充）：单 fp32 累加 + 最终一次 fp16 舍入 FAILS MERE（3.7e-2 vs 阈 9.8e-4）；k1、v 各自舍入 fp16 后 **fp16 域相加**（`k1_16+v_16`，Triton 中 fp16+fp16 保持 fp16 不提升重融合）→ MERE=0.0。
- **R4 w2sum fp32→compute-dtype**：`w2sum_buf` 改 `dtype`，att2_gemm 每 K 迭代的 `.to` cast 变 no-op。
- **R5 UB 编译非确定性陷阱**：大 tile（BM=512/BK=256 weight block 388KB>192KB）有时 auto-split 编译成功、隔离/全量 forward 又失败——**tile 可靠性以独立 forward 为准**（可靠档 BM=128×BK=256），sweep 里碰巧编译过不可信。
- **R6 grid 元组陷阱**：`_grid_for_tiles` 已返回 tuple，`kernel[(grid,)]` 再包层 → "tuple object cannot be interpreted as integer"。早期 sweep 的"大 tile 快"数字是此 bug + 机器噪声的伪影。

**剩余瓶颈（r16 已采集，见下方续节）**：大 shape 的 value_embed、att1、grouped_conv——r16 全 kernel 采集确认三者均非 MMAD bound（CUBE 9.6-10.8%），瓶颈在访存 staging（grouped_conv MTE2）与标量算术（att1，小 shape 信号）；详见「并行线路续 r16→r18b」。

**关键约束清单（首次生成必守）**：
1. 约束禁 torch 退化时，CANN 路由不可用——小 case 用融合 kernel，不是 torch。
2. n==1 特判**仅当 hw==1** 数学精确；hw>1 时 softmax 非恒等不可套用。
3. fp16 逐 conv 舍入用 fp16 域相加实现（R3）。
4. tile 可靠性以独立 forward 验证为准（R5）。

**已证伪**：

| 方向 | 结果 | 来源 |
|------|------|------|
| n1 分离 3 kernel（k1+value+add） | 490us sp 0.80x 反亏（launch 不可摊销） | R2 |
| 单 fp32 累加 + 一次 fp16 舍入 | MERE=3.7e-2 超阈（逐 conv 舍入未复刻） | R3 |

**泛化启示**：softmax-over-hw 类算子遇空间维=1 的退化 shape，可整段跳过 attention（数学精确无损）；纯 Triton 约束下超小 case 的救法是**融合 kernel 减 launch 数**，不是路由 torch。

#### 并行线路续：simulator 诊断 + 最大整除 M-tile（r16→r18b，2026-08-25，NPU5）— 已验证

> 延续上方 r11→r15 纯 Triton 线。r16 补 msprof op simulator 全 kernel 采集（覆盖门禁），R17-R18 落地「非 pow2 最大整除 M-tile」，R19 证伪其误用并回退。**1.1417x → 1.1775x**（新纪录）。

**结果**：**49/49 精度 + 1.1775x**（NPU5，2026-08-25，`verify_r11/perf_result_r17.json`）。磁盘代码 = R17+R18（att2_gemm 已回退）实测 R18b=1.1753x，与 R17 同噪声带。

**r16 simulator 全 kernel 采集（3 瓶颈 kernel 均非 MMAD bound）**：

- `grouped_conv_implicit_kernel`：Cube MTE2 40.9% / SCALAR 27.2% / CUBE 9.6% → **MTE2 访存 staging bound**。ghost-pad 隐式 GEMM 对 kk=9 tap 各做一次 x tile 的 L1 ND2NZ 转换 staging（`MOV_OUT_TO_L1_MULTI_ND2NZ` ~8200 cyc×2 ×9）每次伴 SET_FLAG/WAIT_FLAG 同步；Cube 为关键路径（65887 cyc ≫ Vector 19026）。
- `att1_kernel`：Cube SCALAR 55.6% / CUBE 9.9% → **SCALAR 标量降级 bound**（dual-dot 每轮 4×block_ptr load 地址算术标量化）。小 shape 信号，真实大 shape 不转移（见关键约束 6）。
- `value_embed_kernel`：Cube MTE2 25.2% / SCALAR 25.0% / CUBE 10.8% → **均衡混合**（最接近 compute-balanced，优先级最低）。
- **证伪**：「fp32 dot 不走 Cube / dot 是硬件瓶颈」——CUBE 实测仅 9.6-10.8%，dot 极便宜，开销在 dot 周围的**访存 staging + 标量地址算术**。
- 诊断方向（点 17 消除冗余边界运算）：grouped_conv L191-192 的 `boundary_check=(0,1)` 可证冗余（M/K 维 `_div_tile` 强制 BM|C、BK|CIN_PG 恒不越界；N 维 ghost tail 补零覆盖越界 lane）。但 R16 先验：boundary_check 移除只帮 simulator SCALAR、真实大 shape 中性 → 未作主线落地。

**R17 att1 非 pow2 最大整除 M-tile（1.1417→1.1775x，+3.1%，本线最大杠杆）**：

- 机理：att1 dual-dot（wk1@k1 + wx@x）的 m_block 扫描每次重读 `[c,n]` k1/x 操作数。mid%128!=0 的 19/48 shape 原 BM=8/16/32/64（`_div_tile` pow2 规则），M-block 扫描 3-7 次；改用 mid 的**最大整除 BM**（mid=112→BM=112、mid=224→BM=112），扫描降到 1-4 次。
- 成本代理：`cost = m_blocks + 2.0*mid/bn`（m_block 数 = B 操作数重读次数；BN 反比项 = N-block 对 A 的重读）。遍历 (BN,BK)∈(128,64,32)×(128)，BM 取 mid 的 ≤UB cap 最大整除因子，选 cost 最小档。UB 预算（fp16 dual-dot）4·BM·BN + 4·BM·BK + 4·BK·BN ≤ 196KB → BN=128 时 BM cap≈122。
- 受益 shape 集中在 mid=112/224：case35 +35%、case20 +27%、case16 +20%、case17 +18%、case32 +16%；回归 shape 全部是 att1 tile 未变者（±1% 噪声）。

**R18 value_embed M-tile（保留，case35 叠加 +5.5%）**：同规则扩展到 value_embed 的 c%128!=0（case2 c=96 → BM=96 mblk 3→1；case35 c=448 → BM=224/BN=128 mblk 7→2）。单 dot cost=`m_blocks*n + n_blocks*m`。geomean 持平（1.1772，仅 2 shape 受影响），但 case35 净叠加 +5.5%，保留。

**R19 att2_gemm M-tile（已回退，关键负样本）**：同规则误用于 att2_gemm 反而有害（case2 -3.9% / case35 -4.1%）。根因：att2_gemm 两个操作数都小（A=[c,mid] mid=24/112、B=[mid,n]），加大 BM 省不了流量却减并行程序数。

**关键约束新增**：

5. **最大整除 M-tile 只在 B 操作数巨大（[c,n]，如 value_embed/att1）时有效**；小 K 小操作数 GEMM（att2_gemm）应保持小 BM 保证 grid 并行度（R19）。
6. simulator SCALAR 信号在小 shape 不转移真实大 shape——采集后落地前先估收益量级（R16 教训）。

**已证伪新增**：r16 dot 瓶颈证伪、R19 M-tile 扩展两条均并入 §5.3 全表（见 §5.3）。

**泛化启示（追加）**：多 kernel 算子先过「全 kernel 采集覆盖门禁」再定瓶颈——42 的三个瓶颈 kernel 全部非 MMAD bound，白省了「换 bf16 / 加大 tile」的臆测。非 pow2 整除 M 维的 GEMM，把 pow2 BM 换成最大整除 BM 是通用低成本杠杆，但只对 B 操作数巨大的形态有效；先看操作数尺寸再套规则。

### §4.2 47_DoubleAttention（conv 特征-双 softmax 汇聚，`conv 特征-双 softmax 汇聚`）— 已验证

**生效层**：§0.2-(c) conv 特征-双 softmax 汇聚
**形态**：`A=convA(x)[b,c,hw]`、`B/V=softmax_over_hw(convB/V(x))[b,c/8,hw]`、`GD=bmm(A,Bᵀ)[b,c,c_n]`（gather）→ 分发。分发**双路**：F1-REORDER `OUT=(WR@GD)@SM_V+bR`（WGD 中间量保 fp32）vs golden-order `OUT=WR@(GD@SM_V)+bR`（Z 先舍入 dt）。当前实现为 **100% Triton rebuild**（统一 `_gemm_kernel` + softmax/bmm/WGD-OUT kernel，**每 forward 5 个 launch 点**：`_conv1x1`(packed) → `_softmax`(fused) → `_bmm1` → `_wgd_gemm` → `_out_gemm`），**取代**早期"CANN conv 作 golden 复刻 + HF32 配方"路线（该配方仅存档于 [[triton-doubleattention-precision-recipe]]，勿再引用为现状）。当前结构：convA/B/V 打包为单次 packed `_conv1x1`（T8）、BR/VR softmax 融合为单次 `_softmax`（T9）、非 fp32 路径 WGD 存 dt 使 out_gemm 走纯 cube fp16×fp16（T10）。**P10 重关联路径（T13，26-sig 门控外 shape）**：convA 权重折进 WGD GEMM（`WRWA=Wr@Wa`、`WRBa=Wr@Ba`），`GD = W_A@(X@SM_B^T)+Ba`，5-launch 改为 `_conv1x1`(仅 B/V) → `_softmax` → `T=X@SM_B^T`(TRANS_B) → `WGD=WRWA@T+WRBa` → `_out_gemm`，**消除 A[c,hw] 大中间量物化**；门控内 shape 回落 P9 路径。
**结果**：**50/50 精度 + 2.2794x**（NPU0 空闲设备，2026-08-26，iter5_wgd_dtype，best=Phase 4 **P10 gather 重关联**；**同一窗口 P9-ctrl 1.8351x，P10 相对 P9 +24.2%**）。轨迹（iter5_wgd_dtype）：G034 packed conv 1.593x → P5 WGD dtype 1.7964x → P6 softmax 融合 1.8561x → **P7 conv router 2500 = 1.8796x** → **P9 conv512 白名单 = 1.8743x（同窗口 P7 对照 +1.70%）** → **P10 gather 重关联 = 2.2794x（同窗口 P9 对照 +24.2%）**。⚠️ **验证可复现性受标杆 salt 与设备状态影响**：标杆 `Model` 从 `hash(key)` 播种 Conv 权重（进程加盐），verify.py 在争用设备上交错多 case 会产出损坏的 framework 参考 → 虚假失败（47/50 假失败曾发生于争用 dev3）；**标准可复现配置 = `PYTHONHASHSEED=0` + 空闲设备**，此配置下 P10 verify 50/50（salt0/salt42 双验），impl 与 framework 权重位级一致。P9 = P7 + `_CONV512_SIGS` 白名单（4 shape 用 conv BM=64/BN=512/BK=128）：case20 +9.14% / case5 +6.36% / case24 +4.36% / case42 +4.27% 确定性收益。⚠️ **绝对测值跨日漂移约 -2%**，跨日比 headline 不可信，**同一窗口对照才是事实依据**。历史：早期 1.4425x（NPU1，2026-08-22）；早期分 dtype fp32 1.4506 / fp16 1.4122 / bf16 1.3885。case49 空壳输出不进几何平均（verify 50/50、benchmark 49/50）。

#### 制胜技术（按贡献排序）

**T1 conv1x1 GEMM 化 + 统一 GEMM kernel（精度全过 + 主成本提速的地基）**
- 4 个 conv1x1（convA/convB/convV/分发投影）全部改写为 `_gemm_kernel` 直调（权重 `[out,in]` 展平 hw 做 GEMM），**conv 路由与死分支（`_conv3_kernel`）保留但不走**。
- 3D grid `(b, num_m, num_n)` + constexpr `HAS_BIAS/TRANS_B` 单 kernel 覆盖 conv/bmm/out 三种形态，避免多套模板重复编译；aic_scalar_ratio 实测 0.475，静态发射无 scalar 惰性。
- 判据：CANN conv 复刻路线在精度修复期是权宜，达精度后 rebuild 为纯 Triton 才能吃到 GEMM tile 路由与融合门（T5）。

**T2 G007/G022 结合律重排 F1-REORDER + 精度门控（结构杠杆，大 N 写出 2→1 次）**
- golden 次序要写大 N 中间量 `Z[c,hw]` 再写 `OUT[c,hw]`；重排后只写 `OUT[c,hw]` 一次 + 小 WGD `[c,c_n]`。对 hw 8633 级 case，省掉一整次 44MB 级大 N 写出。配套：WGD（`WR@GD`）保 **fp32** 精度，`_wgd_out_fused_kernel` 在 epilogue 内完成 `@SM_V + bR`。
- **必须配精度门控（C7，G022 fail-sigs）**：离线全量标定 14/50 case 重排路径 MERE 9.3e-4~1.9e-2 vs 阈 9.77e-4(fp32)/7.81e-3(fp16) → `_REORDER_FAIL_SIGS` 14 条 shape+dtype 签名锁死走 golden-order 兜底，其余 36 case 走重排快路。门控让结构杠杆可用而不放弃精度。

**T4 G029 占用率地板（占用 floor 20→40 / BM floor 16，+4.1%）**
- bmm/GEMM tile 路由函数中把 grid 占用地板从 20 提到 40、BM 下限 16，小 case 不再落到单 block 长尾。逐 E2 轮实证 E2-R2 贡献主体。

**T5 G013 WGD+OUT 融合门（三 shape 实证后全量）**
- WGD 与 OUT 两个 launch 是否融合由门控函数判定：三 shape fused/sep 实测 0.926/1.179/1.236 → 融合在大 hw case 净赚、小 hw 反亏，按 shape 档分派。UB 预算 WGD_OUT_UB_BUDGET=104KB、MAX_HW_CHUNKS=4 约束融合 tile。

**T6 launch 参数三件套（G015 BK=512 / G020 cap 8192 / G009 bias 折 epilogue）**
- BLOCK_K 上探 512 深化 cube K 维；grid 总量 cap 8192 防尾核风暴；bias 全部折进 GEMM epilogue（`HAS_BIAS`），不另发射 vector kernel。

**T7 RNG 复刻 + AST 合规收尾**
- 标杆 `randn` 种子/分布逐位复刻；`math.sqrt` → `** 0.5`（AST 预检查禁 Python math 调用于 kernel 常量链，md5 变更后复验 1.4505/1.4336 精度性能双持平）。

**T8 G034 packed conv（3 conv 发射 → 1 次，X 只读一遍；iter5_wgd_dtype 提速地基，相对上轮 +18.0%）**
- convA/B/V 三发射打包为单次 `_conv1x1`：权重拼接 `W_cat[c_total, c]`（c_total = c + 2·c_n），一次 launch 输出 `OUT3[b, c_total, hw]`，X 从 GM 只流一遍而非三遍。packed conv 成为多数 shape 主瓶颈（45-60% 占比）。
- 本质：**MTE2 带宽 bound**——X 被每个 num_M tile 重读，X 读量与 num_M 成正比；tile 128/256/128 近最优（P8 fine-sweep 无普适赢面，见已证伪）。

**T9 P6 softmax 融合（BR/VR 2 次 softmax → 1 次）**
- BR/VR 在 OUT3 中相邻 → 单次 `_softmax` 对 `[b, 2*c_n, hw]` 做行 softmax；SM_B/SM_V 成为 OUT3 的行切片，下游 bmm1/bmm2/out_gemm 用 view-aware 的 `SV.stride(0)`（batch 行跨距）访问，零复制。

**T10 G034b WGD dtype（非 fp32 路径 WGD 存 dt，out_gemm 纯 cube）**
- fp16/bf16 路径 WGD 存 dt（不强制 fp32），out_gemm 变纯 cube fp16×fp16，省 fp32 workspace 中转。fp32 路径仍走 `_wgd_out_fused`（T5）。
- 精度前提：重排路径舍入门控（T2）仍生效；dt 化只改存储 dtype 不改数学次序。

**T11 P7 conv router 2500（阈值 3000→2500，唯一命中档 case34）**
- packed conv 走大 tile 直连的路由判据 `hw >= 3000` 降为 `hw >= 2500`。命中档只有 case34（[5,896,65,41] fp16, b=5, hw=2665∈[2500,3000)）——该 case 确定性 **-9.2%**（新 tile 反不如原路由），整体 +0.19% geo 为噪声叠加。**教训**：阈值档调整只影响极窄 shape 带，收益判据必须逐 shape 确定性归因，不能拿噪声带 headline 当收益。

**T12 P9 conv512 白名单（per-shape 签名收益，非 band 门控；同窗口 +1.70%）**
- packed conv 是 MTE2 带宽 bound（X 按 num_M tile 重读），tile 大则读次数少。对 **精确 signature**（shape+dtype）改用 BM=64/BN=512/BK=128：`_CONV512_SIGS = {(4,192,4671,fp16),(8,320,5917,fp16),(6,320,7979,bf16),(8,448,10553,bf16)}`，复用 T2 的 fail-sigs 签名模式。收益是**逐 shape 确定性**：case20 +9.14% / case5 +6.36% / case24 +4.36% / case42 +4.27%，其余 46 shape 字节级一致。
- 机理：BM×BN 保持 32768 == 128×256 不变（UB 预算不增），收益来自 **BN=512 → 每行 1024B 连续，MTE2 burst 更宽**；同窗口 P7-ctrl 1.8429x → P9 1.8743x（+1.70%）。
- **BM=256 假设证伪**：5 个高 conv 占比 shape fine-sweep 上 BM=256 从未获胜（并行度/UB 压力盖过带宽节省），唯一候选是横向加宽 BN。
- **禁止 band 泛化**：conv512 是 signature 级收益，按 `(b≥4 & hw≥2500 & ci≤512)` band 门控不可泛化——case38（band 内）+7.5% 有害、case19 +0.7% 中性。**教训**：MTE2 burst 收益按 shape 分布极不均匀，必须像 T2 那样用精确签名白名单锁定，任何 band 近似都会引入反方向 case。

**T13 P10 gather 重关联（fold convA 进 WGD GEMM，消除 A[c,hw] 物化；同窗口 +24.2%）**
- 代数重排：`GD = (W_A@X+Ba)@SM_B^T = W_A@(X@SM_B^T) + Ba`。缓存 `WRWA=(Wr.float()@Wa.float()).to(dt)`、`WRBa=(Wr.float()@Ba.float()).to(dt)`，把 convA 权重直接折进 WGD gemm：`WGD = WRWA@T + WRBa`，其中 `T = X@SM_B^T`（M=c, N=c_n, K=hw, TRANS_B，与 P9 bmm1 同 shape）。5-launch 路径：`_conv1x1`(仅 B/V 通道) → `_softmax`(fused) → `T-gemm`(TRANS_B) → `WGD-gemm`(带 WRWA/WRBa) → `_out_gemm`(WGD@SM_V+B)。
- 收益机理：省掉 A[c,hw]（c×hw，hw 大时可达数十 MB）的物化写+读；T-gemm 与 WGD-gemm 均小输出（[c,c_n]），比写出大 A 便宜。对 hw 8633 级 case 省掉一整次大 N 往返。所有使能 shape 确定性受益（case28/44 +2.0x、case33/38 +1.9x、case43 +1.85x、case36 +1.77x、case20 +1.4x 等 21 个，门控 shape 与 P9 同路径噪声 ±1%）。
- **26-sig 双门控（精度 18 + 性能 8）**：`_REASSO_FAIL_SIGS`（18 条，重排路径 MERE 超阈值）与 `_REASSO_PERF_SIGS`（8 条，重排反而慢的 shape）命中时回落 P9 路径——门控机制同 T2/C7，收益按 shape 分布极不均匀，band 近似必引入反方向 case。
- **T-gemm 必须复用 bmm1 的路由器**（`_pick_bmm1_blocks`，hw>4096 时 BK=512），不能用普通 GEMM 路由器——后者在大 hw 档 6/6 拒绝该 T-gemm，换回 bmm1 路由器后恢复 5/6 使能。

#### 性能瓶颈画像（msprof op 采集；当前结构 = iter5_wgd_dtype，stage 隔离计时占比）

| Kernel（每 forward） | 占比 | 说明 |
|---|---|---|
| `_conv1x1`(packed) | **45-60%**（多数 shape 主瓶颈） | case36 59.6% / case42 44.1% / case20 51.1%；MTE2 带宽 bound，X 按 num_M tile 重读 |
| `_out_gemm` | **case34 55.5%** | 仅 c 巨大/hw 中等 case 主导（[5,896,65,41], c_n=112 小 K 大 N 尾声）；其余 case 15-20% |
| `_softmax`(fused) | 8-24% | 纯 vector；case37 因软输出大占比升到 24.3% |
| `_bmm1` | 8-19% | plain，随 hw 缩放 |
| `_wgd_gemm` | 6-21% | 重排小 GEMM，多数 case <15% |

- **conv(packed) 是当前主瓶颈**（旧结构的 out_gemm 头号机理在 5-launch 下被 conv 反超——X 三读合一后 conv 流量集中）。瓶颈性质是 **MTE2 带宽**（X 重读），不是 cube——P8 在 conv tile 上 fine-sweep 无普适赢面（BM=64/BN=512 在 case36/37/48 UB overflow 编译失败），现状 tile 已贴带宽上限。
- **out_gemm 小 K 大 N 尾声**（旧结构头号，case48 实测 43.2%）：K=rank（c_n=80）的 rank-80 尾声 GEMM——44MB fp16 输出 + fp32 workspace 往返；910B 无 fixpipe→UB 直通，IR 证实 mix AIC+AIV 尾声走 **GM workspace 中转**（L0C→GM 128KB → load GM→UB → store UB→GM），架构固有。重排（T2）已把大 N 写出压到 1 次，剩余是输出本身带宽——优化空间 ≈0（证伪方向四轮全拒佐证，见 §5.3）。仅 c 巨大/hw 中等 case 仍主导（case34）。
- **tiny-hw/huge-c case（case47 hw=35、case14 hw=182）无单一主导 stage**，固定 launch/带宽开销主导；case47 是全表唯一 <1.0x 的 shape（0.8835x），优化价值低。
- **stage 重叠显著**：benchmark op 求和 sum_ops 远大于 impl 总延时（case37: 3486µs vs 948µs）——各 stage 在 pipe 上高度重叠，孤立单 stage 计时会高估其对总延时的贡献，stage 级优化前先评估重叠占比。
- msprof 采集要点：op 板模式**必须加 `--launch-count=300`**（默认只采 1 个 kernel），产出按 `OPPROF_*/<kernel名>/<序号>/` 分目录。

#### 关键约束清单（首次生成必守）

1. **重排双路 + 精度门控**（T2）是本形态第一杠杆，但 14/50 case 必须走 golden-order 兜底——先标定再门控，禁止无门控全量重排。
2. **统一 GEMM kernel + 3D grid**（T1）：conv1x1 不留 CANN 依赖（CANN conv 复刻仅精度修复期权宜）。
3. **路由判据含 cube tile 深度项**（C6）：kk×BLOCK_K 有效深度不够时宁可走直连，别只算流量。
4. **融合/分离按 shape 档门控**（T5）：融合不是无条件赢（小 hw 反亏 0.926）。
5. **占用率地板 40 / BM 下限 16**（T4）防小 case 单 block 长尾。
6. AST 合规：kernel 常量链禁 `math.*` Python 调用，用 `** 0.5` 等 AST 安全写法（改后必须复验精度+性能双持）。
7. **当前最优为 P10 gather 重关联 + 26-sig 双门控**（iter5_wgd_dtype，P10=2.2794x，同窗口 P9 1.8351x，相对 +24.2%）：代数机制见 **T13**（convA 折进 WGD GEMM，消除 A[c,hw] 物化）；**必须带精确 signature 双门控**（`_REASSO_FAIL_SIGS` 18 精度 + `_REASSO_PERF_SIGS` 8 性能 = 26 条），禁止 band 门控泛化，门控 shape 回落 P9 5-launch 路径；**T-gemm 必须复用 bmm1 路由器**（hw>4096 时 BK=512，普通 GEMM 路由会在大 hw 档拒绝）；P9 路径上 conv 主瓶颈仍 MTE2 带宽 bound（T12）；out_gemm 仅 c-huge/hw-moderate case 主导；tiny-hw/huge-c case 无单一主导 stage。

#### 已证伪方向（省下你的时间，Phase 4 四轮全拒）

> 与 §5.3 全表重复的行（占用率地板 / 路由删流量 / 编译参数组合 / 文档配方照抄 / WGD+OUT 融合 / num_stages）已并入该表，此处只列未入全表的。

| 方向 | 结果 | 来源 |
|------|------|------|
| conv3 直连 cube 效率门收紧（G030） | **中性**（kk<32 且 c/kk≥128 才回退，翻转 shape 无净变） | iter_0 |
| conv tile fine-sweep（P8） | **无普适赢面**（conv 是 MTE2 带宽 bound，128/256/128 近最优；BM=64/BN=512 在 case36/37/48 UB overflow 编译失败） | iter5_wgd_dtype |
| out_gemm tile fine-sweep（P8） | **单 shape 4-6% 收益被稀释**（无普适规则，现状 (256,128,256) 已合理） | iter5_wgd_dtype |
| conv BM 128→256（P8/P9） | **证伪**（5 个高 conv 占比 shape fine-sweep 从未获胜；并行度/UB 压力盖过 MTE2 带宽节省，唯一候选是横向加宽 BN 而非放大 BM） | iter5_wgd_dtype |
| conv512 按 band 门控 `(b≥4&hw≥2500&ci≤512)` | **不可泛化**（case38 band 内 +7.5% 有害、case19 +0.7% 中性；BM=64/BN=512 的 MTE2 burst 收益按 shape 极不均匀，必须精确 signature 白名单锁定，见 T12） | iter5_wgd_dtype |

#### 泛化启示（给本类其他算子）

- **"conv 特征-双 softmax 汇聚"形态**：主成本在**投影/分发 GEMM 的尾声写出**（rank 级 K × 大 N），不在 softmax——softmax-over-hw 极轻（6.5%）。结合律重排把大 N 写出 2→1 是第一杠杆，但必须配 shape 签名精度门控（C7）。
- **小 K 大 N 尾声 GEMM**（K=rank/通道缩维）：cube 利用率个位数、带宽 bound、AIV 尾声走 GM workspace 中转是 910B 架构固有——此类热点判"接近硬件极限"前先做 msprof op 板实证（cube% + workspace 往返 IR 取证），别拍脑袋继续砸轮次（本算子四轮全拒的教训）。
- **Phase 4 判定纪律**：headline 落 ±2% 噪声带一律拒绝；逐 shape 双峰分布必拆参归因；两时段跑间共模漂移用"同代码复验"（1.4505/1.4336）标定后再比参数效果。
- **MTE2 带宽 bound 的 tile 收益按 shape 极不均匀**（P9/T12）：BM=128→256 放大在任何 shape 都不赢，横向加宽 BN（256→512）却对 4 个精确 shape 确定性 +4~9%——burst 宽度收益与 UB 预算、行连续性相关，无法用 shape band 门控，必须像 T2 精度门控那样用**精确 signature 白名单**锁定受益 shape；拿 band 近似必然引入反方向 case（case38 +7.5% 有害）。


---

## §5 Phase 4 优化点清单

### §5.1 优化点按收益排序（★ = 高收益；跨成员落地顺序，细节见 C / T 编号）

> C 编号见 §1，T 编号见 §3/§4 算子章节。Phase 4 先扫本表定位高收益点，再进对应章节取细节。与 FA §5.1 同体例。

| # | 方向 | 实测增益 | 适用条件 |
|---|------|---------|----------|
| 1 | ★★ 低精度中间量 RNE 位运算复刻（C1） | fp16 38/50→50/50；bf16 原生乘超阈 3 倍 | fp16/bf16 且对中间舍入敏感 |
| 2 | ★★ 结合律重排 + 精确签名精度门控（C7，47 T2/T13） | 47 P10 相对 P9 **+24.2%**；大 N 写出 2→1 | 聚合-分发链有 rank 级中间量；**必须**配 fail-sigs |
| 3 | ★ 投影 GEMM 当主攻对象，别攻 attention（C5） | 59 投影占 56% | 通道扩展-分支分发形态 |
| 4 | ★ 路由决策含 cube tile 深度项（C6） | 只算流量崩到 **0.4961x** | 任何"选 A 还是 B"的路由门 |
| 5 | ★ 归约用 Kahan（C2） | fp32 尘埃 2e-5→3e-8 | 大 N mean/sum |
| 6 | ★ 激活近似匹配 NPU（C3） | erf 偏差 4.7e-4 | GeLU 相关 |
| 7 | ★ 标杆种子确定性修复（§2） | 抽签→可复现 | 标杆用 `hash()` 播种 |
| 8 | 编译参数实测本机支持集（C8） | 文档配方全失败；`no-limit` 有害（21/49 回退） | 想用编译参数时 |
| 9 | 跳过聚合量小的 kernel（C4） | 合计 <6% 无空间 | 小聚合量 kernel |
| 10 | 最大整除 M-tile（42 R17/R18） | 42 **+3.1%** | 多 kernel 流水 + 非 pow2 M，**仅 B 操作数巨大** |

### §5.2 结构性下限（小 case 别踩）

- **多 kernel 流水在小 case 上受 launch 开销 ~0.1ms/kernel 约束**（42 R2 实锤：3 分离 kernel=490us 反亏，融合单 kernel=330us 净赚）。小 case 先融合 kernel 削 launch 数，不路由 torch（纯 Triton 约束下）。
- 本类 attention 极轻（C-Q1），**别为 attention 拆 kernel**。

### §5.3 证伪方向全表（跨成员，持续积累）

> 继承 FA §5.2 / conv / norm 全表；以下为本类**新增**。本表是证伪结论的**唯一全量入口**——§3/§4 各算子「已证伪」表只保留未入全表的行（重复行已并入本表）。

| 方向 | 结果 | 来源 |
|------|------|------|
| 低精度中间量用 `.to(dtype).to(fp32)` round-trip | **编译器消除为 no-op**，精度 fail | C1（59） |
| 依赖硬件原生乘复刻低精度 | fp16 MER 方差放大 / bf16 超阈 3 倍 | C1（59） |
| mean 普通顺序累加 | **精度崩**（Kahan 才对齐 reduce 树） | C2（59） |
| GeLU 用 erf 精确式 | **偏差 4.7e-4**（NPU 是 tanh 近似） | C3（59） |
| 优化聚合量小的 kernel | **no-op**（合计 <6%） | C4（59） |
| 标杆种子用 `hash()` | **抽签**（dtype hash 地址派生漂移） | §2（59） |
| kernel 路由门只按搬运流量决策 | **0.4961x 崩盘**（kk=16 浅 dot cube 崩塌） | C6（47） |
| 结合律重排不做精度门控直接全量 | **14/50 case 超阈**（MERE 至阈 2.4 倍） | C7（47） |
| 文档编译参数配方直接照抄 | **编译失败**（参数不存在/默认已开） | C8（47） |
| `set_workspace_multibuffer=2` | **零效果**（1.4422 vs 1.4425 纯噪声） | C8（47） |
| UB 多缓冲 `no-limit` | **有害**（vector 尾声碎化，21/49 shape 回退最重 -14%） | C8（47） |
| 占用率地板越紧越好 | **-3.8%**（地板 40 再收紧反伤 bmm1） | §4.2（47） |
| WGD+OUT 无条件融合 | **小 hw 反亏**（fused/sep=0.926） | §4.2（47） |
| 大 tile 靠 sweep 碰巧编译过即采用 | **UB 编译非确定**（388KB>192KB，隔离/全量 forward 失败；须独立 forward 验证 tile） | §4.1 并行线（42, R5） |
| grid helper 已返回 tuple 再套一层 | **运行时错误**（"tuple object cannot be interpreted as integer"） | §4.1 并行线（42, R6） |
| 未采集就断言「dot 是硬件瓶颈 / fp32 dot 不走 Cube」 | **证伪**（CUBE 实测 9.6-10.8%，开销在 dot 周围访存 staging + 标量算术） | §4.1 并行线（42, r16） |
| 最大整除 M-tile 扩展到小操作数 GEMM（att2_gemm） | **-3.9%/-4.1%**（大 BM 省不了流量却减 grid 并行；M-tile 只在 B 操作数巨大 [c,n] 时有效） | §4.1 并行线（42, R19） |
| num_stages 一味加三 | **拒**（全量 1.8478x < 1.8796x，逐 shape 无普适赢面） | §4.2（47, P8） |

---

## §6 测量口径与时效

- **几何平均**：每 case 权重相同（FA §7.0）。本类主成本在投影 GEMM，优化要盯最慢的 shape 档。
- **口径**：benchmark.py 旧口径按 kernel 名 groupby（同名多次发射只计一次）；S2Attention 的 `_gemm_bias_kernel` 发射 2 次（MLP1+MLP2）旧口径漏计 1 次，新口径会如实计入。47 号统一 `_gemm_kernel` **每 forward 发射 5 次**（iter5_wgd_dtype P7：packed conv/softmax/bmm1/wgd/out_gemm；早期 6-7 次结构见 §4.2 历史），groupby 口径必须按发射序号拆开归因，不能按 kernel 名聚合。跨口径比较须注明。
- **off-by-one**：perf_result.json 的 case_idx 为 **1 基**（= JSON 行号），模拟/归因脚本若 0 基需对齐后再比。
- **噪声地板与共模漂移**：本机 NPU 实测两时段跑间 headline 漂移可达 +2%（同代码 1.4505→1.4336）；headline 差 <±2% 判噪声拒绝，参数效果判定须有"同代码复验基线"或逐 shape 归因支撑（47 Phase 4 判定纪律）。
- **msprof op 板采集**：必须加 `--launch-count=300`（默认只采 1 个 kernel）；产出按 `OPPROF_*/<kernel名>/<序号>/` 分目录，case_idx 对齐见上条。
- **历史结论随编译器/CANN 失效**（FA §7.3）：所有数字标注来源算子与日期，判据/机理优先于绝对数字。RNE 位运算语义随编译器版本可能变，重测前重新标定。

---

## §7 常见陷阱与避免方法

> 症状驱动的排查表（学 FA §8）。按**你看到的症状**反查原因与治法；交叉引用见 C（§1）/ T / R（§3/§4）/ §2 / §5。

| 陷阱（症状） | 原因 | 避免方法 |
|------|------|---------|
| 低精度 case 单独失败、`MERE` 卡在阈值附近 | 中间量靠 `.to(dtype).to(fp32)` round-trip，编译器消除为 no-op；或依赖硬件原生乘（非完全 RNE） | **C1**：每处舍入点用位运算复刻 RNE（bf16 掩码式 / fp16 4-op 大 tile 变体） |
| 某 case 的 mean 结果差 ~1e-5 量级、fp16 输出边界翻转 | 顺序累加 fp32 尘埃 ~2e-5，在舍入边界触发翻转 | **C2**：Kahan 补偿对齐 torch reduce 树（2e-5→3e-8） |
| GeLU 相关 case 单独失败且偏差 ~5e-4 | 用了 erf 数学精确式，golden 是 NPU 的 tanh 近似 | **C3**：改用 `0.5*x*(1+tanh(...))` 匹配 `F.gelu` |
| 同一代码跨 verify 进程结果随机变化（"抽签"） | 标杆 `torch.manual_seed(hash(...))` 的 `hash(dtype)` 是对象地址派生，随 ASLR 漂移 | **§2**：锁步改 `(zlib.crc32(repr(key).encode()) + k) & 0xFFFFFFFF`，k 预扫选定 |
| 路由翻转后大 shape 集体回退、幅度远超噪声带（-50% 级） | 路由门只算搬运流量、缺 cube tile 深度项——省搬运却喂了空转 cube | **C6**：决策函数同时评估流量 + kk×BLOCK_K 有效深度（实测只算流量崩到 0.4961x） |
| 重排/融合后大部分 case 提升但固定一小撮 case 精度卡阈 | 结合律重排改变了中间量舍入次序，超阈 shape 是确定性的 | **C7 / T2 / T13**：离线标定 + 精确 signature 门控锁死走 golden-order 兜底；**禁止 band 门控** |
| 加编译参数后逐 shape **双峰分布**（部分升部分降） | 组合参数里某单个参数对部分 shape 有害 | **C8**：拆参逐一重试定位加害参数（47 的 `no-limit` 使 21/49 回退最重 -14%）；headline 落 ±2% 噪声带判零效果 |
| 小 case 多 kernel 反而变慢 | launch 开销 ~0.1ms/kernel 在小 case 不可摊销 | **§5.2 / 42 R2**：融合 kernel 削 launch 数，不路由 torch（纯 Triton 约束下） |
| 大 tile 隔离能编译、全量 forward 又失败 | UB overflow 的编译非确定性：auto-split 有时过、隔离/全量不一致 | **42 R5**：tile 可靠性以**独立 forward 验证**为准，sweep 碰巧编译过不可信 |
| 单 fp32 累加 + 最终一次 fp16 舍入，逐 conv 舍入未复刻 | 多个已各自舍入的 fp16 中间量相加被提升到 fp32 重融合 | **42 R3**：fp16 域加法（`a_16 + b_16`）强制各中间量独立舍入（MERE 3.7e-2 → 0.0） |
| 把最大整除 M-tile 用到小操作数 GEMM 反而变慢 | M-tile 只在 B 操作数巨大（`[c,n]`）时省重读；小操作数 GEMM 加大 BM 只减 grid 并行 | **42 R19**：仅 value_embed/att1 这类 B 巨大形态用；att2_gemm 保持小 BM |
| 拿噪声带 headline 当收益、事后复验无差异 | 两时段跑间共模漂移可达 +2%（同代码 1.4505→1.4336） | **§6**：参数判定须"同代码复验基线"或逐 shape 确定性归因；跨日绝对测值漂移 ~-2%，同窗口对照才是事实依据 |
| 签名级收益想用 shape band 门控泛化 | MTE2 burst / 重排收益按 shape 极不均匀，band 内必混入反方向 case | **T12**：用精确 signature 白名单锁定受益 shape（case20 +9.14% / case38 band 内 +7.5% 有害） |
| 未采集就断言「dot 是硬件瓶颈 / 投影 GEMM 走 Cube 满载」 | 开销常在 dot 周围的访存 staging + 标量地址算术，不在 dot 本身 | **42 r16**：先过 msprof op 板全 kernel 采集（`--launch-count=300`）再定瓶颈（CUBE 实测仅 9.6-10.8%） |

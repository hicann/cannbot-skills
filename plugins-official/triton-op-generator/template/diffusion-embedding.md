---
name: diffusion-embedding
description: 扩散模型条件嵌入类算子（timestep 正弦嵌入 + 小 MLP + 并行投影相加，如 FluxTimestepGuidanceProjectionEmbedding）的 Triton Ascend 优化经验合集，按算子分章节组织，含类别级通用经验 + 各算子专属约束/骨架/kernel
metadata:
  type: reference
---

# 扩散模型条件嵌入类算子优化经验

本文档合并该类别算子的优化经验。按以下结构组织：
- **§1 通用经验**：**类别级**工程约束（以结构特征和形态参数表述，不绑定具体算子的形状常量；与其他类别共用的通用约束见 `linear.md` L1.4-L1.8 / `tensor-transform.md` G1-G8 等，此处只写本类别特有的）
- **§2 10_FluxTimestepGuidanceProjectionEmbedding**（本类首个实例，§1 全部规则的实测证据来源）
- **§3 常见陷阱**（通用 + 实例）

> ⚠️ **关键区分**：本类别的核心优化哲学是 **"batch 维极小的 embedding/投影链：launch 开销主导 → 按计算阶段深度融合；GEMM 走向量外积不走 tl.dot；tile 分档必须是覆盖全 batch 域的 law"**。生成时**禁止混用**其他类别的经验：不要套用 `linear.md` 的低秩归约/attention hoist（本类无 attention 段），不要在 K-loop 内做 gather/加权 prologue（同 `vision-mlp.md` V1），不要按大 M GEMM 的直觉选 tl.dot 或拆 kernel。

---

## §0 适用范围与算子分类

**类别定义（结构特征，符号化）**——命中任意两条即入本类：

| # | 结构特征 | 符号化描述 |
|---|---------|-----------|
| F1 | 正弦时间/频率嵌入 | 标量或 `[B]` 信号 `×scale` → broadcast 外积 `freqs[D/2]` → `sin/cos` → `concat` → `[B, D_emb]` |
| F2 | 嵌入经小 MLP | `Linear(D_emb→D_inner) → 激活(SiLU/GELU) → Linear(D_inner→D_out)`，宽度为模型超参常量 |
| F3 | 并行投影路 | 另一输入（pooled text / label / 条件向量）经 `Linear(D_in→D_out)` 投到同一输出宽度 |
| F4 | 合并输出 | 两路**相加**（或 concat）→ cast 回输入 dtype |
| F5 | 形态参数 | GEMM 的 M 由 batch 维决定且 ∈ [1, 数百]（**skinny-M**）；N/K 为常量（数百~数千）；权重 MB 级、从 forward 入参传入（无 `__init__` 随机权重） |
| F6 | 精度契约 | 参考实现常显式全程 fp32（每个 `F.linear` 前 `.to(torch.float32)`），bias dtype 与权重 dtype 分离，层后带显式 dtype 回转 |

**家族来源**：Flux / DiT / Stable Diffusion 3 / PixArt 等扩散模型的条件编码器（timestep embedder + pooled text projection embedder），每个采样步执行一次，B 通常很小。

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| 10_FluxTimestepGuidanceProjectionEmbedding | `diffusion-embedding` | F1-F6 全命中；B∈[1,384]，宽度 {768, 3072} | 3-kernel 全融合 + 向量外积替代 tl.dot + BM 分档 law + 共享 K 双 GEMM pass-merge |

**入类流程**：新算子命中本类后，§1 全部约束直接适用；§2 中的具体数值（B 分布、宽度、档位边界）是**本例的实测标定值**，新算子必须按自己的形状重新标定，规则形式不变。

---

## §1 通用经验（类别级，首次生成必须遵守）

以下约束按**结构特征 + 形态参数**表述。标注"实测证据"的数据来自 §2 实例；规则本身与具体形状无关。其他文件已提取的通用约束（`tensor-transform.md` G1 动态 num_cores / G2 pow2 BLOCK / G4 grid 不超核数）此处不再重复。

### DE1 ★★★ skinny-M GEMM 默认禁用 tl.dot，走向量外积-累加；入类先 probe 定路径
- **触发条件**：GEMM 的 M ≤ 数百（batch 维决定）且本类别 shape 族
- **默认** 用向量外积-累加替代 `tl.dot`：
  ```python
  acc += tl.sum(a[:, :, None] * w[None, :, :], axis=1)   # a:[BM,BK], w:[BK,BN] → acc:[BM,BN]
  ```
- **Why:** `tl.dot` 在该形态降级非 Cube 路径。实测证据（§2 实例，910B1）：fp32 路径 790µs–269ms、fp16 路径 13.5–67.9ms，比 torch 31–42µs 慢 **300–1600 倍**。TTIR 显示向量外积会被 pattern-match 成 `tt.dot(inputPrecision=tf32)`，该 lowering 在 skinny-M 性能塌陷
- **决策程序**：新算子首次生成前，用最大 M 的 case 各跑一次 tl.dot 与向量外积单点 probe；tl.dot 慢 10x 以上即全族禁用并记录，不需逐 tile 重试
- 与 `linear.md` L1.1 的差异：L1.1 要求"投影 GEMM 一律 tl.dot"，**本类是例外**——例外由 M 维形态触发，不是算子偏好

### DE2 ★★★ 按计算阶段深度融合：kernel 数 = 阶段数（典型 2-4），段内禁止再拆
- **必须** 把参考实现的 ~10 个独立 torch op 按计算阶段聚合：`嵌入生成段` / `MLP 段` / `尾段合并（多路投影+add+cast）`，每段一个 kernel
- **禁止** 把 add / cast / 激活等轻操作拆成独立 kernel；禁止进一步按 shape 段拆分（重复 launch）
- **禁止** 跨段做生产者-消费者融合（把上游段的输出改为在下游 kernel 内即时重算），除非下游对上游输出**只读一次**。判据：若下游 kernel 按 N-tile 切分且每个 N-tile 都需要上游输出，融合会使上游计算量放大 N-tile 数倍。实测证据（§2 实例）：嵌入段（sin/cos）折进 MLP 段后每个 N-tile 重算三角函数（放大 24x），MLP 段 79µs→302µs，60/60 case 全劣化后回退
- **Why:** 小 B 段 launch 与中间量物化开销主导，融合是本类第一收益来源。实测证据（§2 实例）：B≤12 段融合后 **1.95x**（60 case 中最高段）
- **嵌入生成段单 kernel 化**（F1 命中时）：broadcast 外积 + 三角函数 + concat 一次完成，中间不落 HBM

### DE3 ★★ 参考实现的每个显式 dtype 转换点都是位级契约，必须 constexpr 逐点复现
- **必须** 全程 fp32 累加；参考实现在线性层后显式 `.to(bias.dtype)`（或其他）回转的，kernel 内用 constexpr 分支逐点复现该舍入后再做后续运算
- **必须** dtype 组合（权重 dtype × bias dtype × 输出 dtype）经 constexpr 分派到特化路径；跳过任何中间舍入会破坏位级对齐（数值近似对但 verify 以 HiddenStateMismatch 失败）
- **Why:** 实测证据（§2 实例）：加回中间舍入复现后 60/60 通过
- 与 `linear.md` L1.7 的差异：L1.7 讲段级分派边界，本条讲**逐舍入点复现**

### DE4 ★★★ 共享 contraction 维的多路 GEMM 必须 pass-merge：单 K-loop 多累加器
- **触发条件**：多条 GEMM 的 K 完全相同且共享激活面板（F3+F4 的标配尾巴：MLP 尾层与并行投影路同 K 同输出宽度）
- **必须** 融合为单 kernel：一个 K-loop 同时推进各路累加器，输出 tile 几何相同；各路先 round 到自己的中间 dtype（DE3），**之后**才 add/合并，最后统一 cast
- **Why:** 消除 (路数−1) 次 kernel 发射 + 一整趟 tile-index 标量算术 + 中间量的 GM round-trip。实测证据（§2 实例双路）：geomean **+20.9%**
- **推广**：N 路同 K 投影 → N 个累加器共用一个 K-loop；K 不同则不适用（勿强行合并）

### DE5 ★★ tile 分档必须是覆盖全 batch 域的一条 law，禁止只碰少数 case 的定制 gate
- **必须** BM（及 BN 特化）分档写成整域 law（如三段式 tier），档位边界由"weight 面板重读次数 vs masked-lane 浪费"的权衡实测标定
- **档位边界的标定尺**：向量外积路径下 GEMM kernel 的时间与 **M-tile 数（NUM_M）近似成正比**（weight 面板 GM 重读是工作单元，masked lane 免费）——档位边界应画在"减少一档 NUM_M"的 B 值处，而非等间距取整。实测证据（§2 实例）：边界 32→17 使 B∈[17,31] 从 NUM_M=2 降为 1，5 个 case 各 +33~38%
- **Why:** tier 边界移动影响全部 case，在 geomean 层**可认证**；只影响 6/60 case 的单段 gate 即使单段 +57%，也会被未触碰 case 的 ±1-3% profiler 噪声淹没而**不可认证**（§2 实例 round-3 实测关门）
- **禁止** 把小 BM 档扩入中大 B 域：weight 面板重读随 M-tile 数放大，实测证据 -20~-43%

### DE6 ★ no-op cast 编译期剪除
- **必须** 输入已是 fp32 时 `.to(tl.float32)` 仍生成指令；用 constexpr（如 `A_IS_F32`）在编译期剪除
- **Why:** 实测证据 +1.5%；多 dtype 任务（fp16/bf16/fp32 混合输入）普遍适用

### DE7 ★ tile 参数硬顶前置检查 + 起点值
- **必须** 生成前检查 `[BM,BK,BN]` 3D 乘积中间量 ≤ triton 1M numel 上限；`multibuffer=True` 默认开启
- **起点值**：BM ∈ {16,32,64}（按 DE5 分档）、BK=128、BN=128（可条件加宽到 256，见下）；BM=4/128 及低精度 3D 中间量直接编译失败（`ConvertLinalgRToBi`），不要试——低精度 3D 路线封死即封死，勿绕
- **BN 加宽条件**（N 向程序数减半、削减主导的标量地址算术）：仅在"加宽后总程序数仍 ≥ 向量核数 × 安全系数"时启用，且必须整域 law 认证（DE5）；BK 缩小（sub-split / BK=8）全线回归，默认不做
- **Why:** 实测证据见 §2 实例 attempts（31 个优化点中 tile 族全线证伪，唯 BN 条件加宽 + BM tier 有效）

### DE8 ★ 多 kernel 级联禁止共享 grid 常量
- **必须** 每个 launch 独立计算 grid；不同 kernel 的 N 维 tile 数不同时，误共享 → verify 全挂且无报错指向
- **Why**: §2 实例 K2 需 `N_inner//BN` 个 tile、K3+4 需 `N_out//BN` 个，误用同常量 verify 0/60

### DE9 ★ 环境：hostname 含特殊字符时 msprof 全挂，必须 sitecustomize shim
- **必须** benchmark 前 `PYTHONPATH=<shim_dir>`（sitecustomize patches `socket.gethostname`）；残余 flaky parse 用 `--max_retries 2`
- **Why:** hostname 含 `=` → msprof "Invalid input path"，全部 case PROFILER_COLLECT_FAIL；shim 是强制项不是可选项

### DE10 ★ 收益认证必须连续两次全量 run（同一代码），单次 geomean 不可作为采纳依据
- **必须** 每个"拟采纳"的改动跑两次完整 benchmark（代码不变），两次 geomean 及目标段收益方向一致才可 KEEP；单 flaky case（如 1/60 PROFILER_COLLECT_FAIL）重跑即过的可豁免
- **Why:** 同代码 run-to-run 噪声带宽实测 ~±0.2 个 geomean 点（§2 实例：同代码两次 1.3956/1.4083，case 级漂移可达 0.82→2.01）；两次 run 认证能把"真实段收益"从噪声中分离（实例 round-6 的边界移动两次 run 均 +33~38% 才确认）
- **配套**：跨轮对比代码时，先看两次 run 的自方差再下结论；profiler pipe-share 数值只作类型证据不作对比依据（见 §3.1）

---

## §2 10_FluxTimestepGuidanceProjectionEmbedding 算子（本类首个实例）

**算子类别**: `diffusion-embedding`（F1-F6 全命中）
**典型特征**: timestep[B] ×1000 → 外积 freqs[384] → sin/cos concat 得 [B,768]；Linear(768→3072)+SiLU+Linear(3072→768) 全程 fp32；pooled[B,3072]@Wt^T；两路相加 cast 回输入 dtype。B∈[1,384]，60 case，dtype ∈ {fp16, bf16, fp32} × bias {fp32, lowp}
**性能基准**: 60/60 pass，几何平均 **1.4083x** vs torch（target 5.0 未达；7 轮迭代后确认平台期，~±0.2 点为同代码 run-to-run 噪声带）

> 本节的常量（HALF=384 / TD=768 / INNER=3072 / 档位边界 128、17、96）是**本例的标定值**。新算子入类时：宽度替换为该算子的模型超参；档位边界按其 B 分布重新 probe 标定（DE5 的 law 形式不变）。

### §2.0 首次生成必读：为什么必须把段级融合框架写对

首次生成最大的坑是 **GEMM 实现路径**：按大 M 直觉写 `tl.dot` 得到 300-1600x 灾难退化（DE1）。其次是**中间舍入**：漏掉参考实现的 `.to(bias.dtype)` 回转，verify 位级失败且难定位（DE3）。骨架必须在 Phase 2 就定为：**段级融合 + 向量外积 + 舍入 constexpr 复现**；结构错了后续 round 在 tile 上救不回。

### §2.1 Layer 1: 设计约束（引用 §1 + 本例具体化）

#### L1.1 tl.dot 禁用 → 向量外积（同 DE1，本例 probe 结论：全族禁用）
#### L1.2 `F.linear` 语义 `y = x @ Wᵀ`：kernel 传 `w.stride(1), w.stride(0)`（交换 stride），行主 `[N,K]` 直接 load，无 Host 侧 transpose
- 权重从 forward 入参传入（无 RNG 复刻）；方阵下算成 `x @ W` shape 检查不报错、只有数值错

#### L1.3 3-kernel 结构（同 DE2，本例阶段划分）
- K1 `_timestep_embed_kernel`（纯 Vector：scale/outer/sin/cos/cat 单 kernel，中间不落 HBM）
- K2 `_vec_linear_silu_kernel`（linear1 + bias + SiLU）
- K3+4 `_vec_dual_linear_add_cast_kernel`（DE4 双路 pass-merge + per-branch round + add + cast）

#### L1.4 tile law（同 DE5/DE7，本例标定值；bm32 档下界 32→17 为 round-6 两次 run 认证）
```python
bm  = 64 if B >= 128 else 32 if B >= 17 else 16
# 边界演进：初始 32 → round-4 调至 128/32 → round-6 把 bm32 下界 32→17（B∈[17,31]
# 从 bm16/NUM_M=2 升为 bm32/NUM_M=1，少付一整趟 W 面板重读，5 case 各 +33~38%）
bn2 = 256 if (bm == 32 and B >= 96) else 128   # 仅 K2（N=3072 段），NUM_M>=3 保核饱和
BK = 128                                        # 缩小全线回归，不试
```

#### L1.5 精度契约（同 DE3）：linear2 / textproj 输出各自 round 到自己 bias dtype 后再相加，最终 cast 到 `result_type` 推导的 out_dtype
#### L1.6 动态核数 + 1D 扁平 grid（同 `tensor-transform.md` G1/G4）：`get_device_properties` 取核数禁硬编码；GEMM kernel grid = `N_tiles * NUM_M` 一维展开，`pid_n = pid // NUM_M`；K1 用 `grid = min(NVEC, B)` + 核内步长循环
#### L1.7 硬顶清单（同 DE7）：BM≤64（1M numel cap）；禁 BM=4/128、fp16 3D 中间量（`ConvertLinalgRToBi`）；`multibuffer=True`
#### L1.8 grid 禁共享（同 DE8）：K2 用 `INNER//bn2`、K3+4 用 `TDIM//BN`，独立计算

### §2.2 Layer 2: 算法骨架（首次生成就要写对）

#### L2.1 Host 侧分派决策树

```python
# 宽度常量来自算子定义（本例 384/768/3072），泛化时替换
bm = 64 if B >= 128 else 32 if B >= 32 else 16        # 档位边界按 B 分布实测标定
bn2 = 256 if (bm == 32 and B >= 96) else 128
NUM_M = (B + bm - 1) // bm
grid2  = (D_inner // bn2) * NUM_M                     # K2: N = D_inner
grid34 = (D_out // BN) * NUM_M                        # K3+4: N = D_out，禁与 grid2 共享
# out_dtype 按 torch.result_type 语义在 host 推导
# _MID = {fp16:0, bf16:1, fp32:2} → 每路 MID*_DTYPE constexpr（DE3）
```

#### L2.2 kernel 启动模式（1D 扁平 grid + constexpr 特化）

```python
_vec_linear_silu_kernel[(grid2,)](temb, w1, b1, h, B, NUM_M,
    temb.stride(0), w1.stride(0), w1.stride(1), h.stride(0),
    K_C=D_emb, BN_C=bn2, BK_C=128, BM_C=bm, SILU=True,
    MID_ROUND=False, MID_DTYPE=2, multibuffer=True, A_IS_F32=True)
```

### §2.3 Layer 3: 关键 kernel 实现（优化重点与易错代码）

#### L3.1 `_timestep_embed_kernel`（F1 嵌入段：broadcast 外积 + 三角函数 + concat 单 kernel）

```python
@triton.jit
def _timestep_embed_kernel(ts_ptr, f_ptr, out_ptr, B, s_ts, s_f,
                           HALF_C: tl.constexpr, BLK: tl.constexpr, NUM_PROG: tl.constexpr):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLK)          # BLK = next_pow2(HALF_C)
    fmask = offs < HALF_C
    f = tl.load(f_ptr + offs * s_f, mask=fmask, other=0.0).to(tl.float32)
    for b in range(pid, B, NUM_PROG):  # 核内步长循环，每 task 一行
        t = tl.load(ts_ptr + b * s_ts).to(tl.float32) * 1000.0   # scale 来自算子定义
        arg = t * f                    # fp32 契约
        tl.store(out_ptr + b * (2*HALF_C) + offs, tl.sin(arg), mask=fmask)
        tl.store(out_ptr + b * (2*HALF_C) + HALF_C + offs, tl.cos(arg), mask=fmask)
```

#### L3.2 `_vec_linear_silu_kernel`（DE1 向量外积 GEMM + bias + 激活 + 可选中间舍入）

```python
@triton.jit
def _vec_linear_silu_kernel(a_ptr, w_ptr, bias_ptr, o_ptr, M, NUM_M,
                            s_a, s_wo, s_wi, s_o,
                            K_C: tl.constexpr, BN_C: tl.constexpr, BK_C: tl.constexpr,
                            BM_C: tl.constexpr, SILU: tl.constexpr,
                            MID_ROUND: tl.constexpr, MID_DTYPE: tl.constexpr,
                            A_IS_F32: tl.constexpr):
    pid = tl.program_id(0)
    pid_n = pid // NUM_M
    pid_m = pid - pid_n * NUM_M
    offs_n = pid_n * BN_C + tl.arange(0, BN_C)
    offs_k = tl.arange(0, BK_C)
    offs_m = pid_m * BM_C + tl.arange(0, BM_C)
    m_mask = offs_m[:, None] < M
    acc = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    wbase = w_ptr + offs_k[:, None] * s_wi + offs_n[None, :] * s_wo   # Wᵀ: 交换 stride
    for k in range(0, K_C, BK_C):
        a_raw = tl.load(a_ptr + offs_m[:, None]*s_a + (k+offs_k)[None, :], mask=m_mask, other=0.0)
        a = a_raw if A_IS_F32 else a_raw.to(tl.float32)               # DE6 剪除 no-op cast
        w = tl.load(wbase + k * s_wi).to(tl.float32)
        acc += tl.sum(a[:, :, None] * w[None, :, :], axis=1)          # DE1 向量外积
    acc += tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
    if SILU:
        acc = acc * tl.sigmoid(acc)
    if MID_ROUND:                                                     # DE3 中间舍入复现
        if MID_DTYPE == 0:   acc = acc.to(tl.float16).to(tl.float32)
        elif MID_DTYPE == 1: acc = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(o_ptr + offs_m[:, None]*s_o + offs_n[None, :], acc, mask=m_mask)
```

#### L3.3 `_vec_dual_linear_add_cast_kernel`（DE4 pass-merge：单 K-loop 双累加器）

```python
@triton.jit
def _vec_dual_linear_add_cast_kernel(a1_ptr, w1_ptr, bias1_ptr,   # 路 A（MLP 尾层）
                                     a2_ptr, w2_ptr, bias2_ptr,   # 路 B（并行投影）
                                     o_ptr, M, NUM_M,
                                     s_a1, s_a2, s_w1o, s_w1i, s_w2o, s_w2i, s_o,
                                     K_C: tl.constexpr, BN_C: tl.constexpr, BK_C: tl.constexpr,
                                     BM_C: tl.constexpr, OUT_F32: tl.constexpr,
                                     MID2: tl.constexpr, MIDT: tl.constexpr,
                                     MID2_DTYPE: tl.constexpr, MIDT_DTYPE: tl.constexpr):
    # 同 L3.2 的 pid 分解与 offs 计算…
    acc2 = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    acct = tl.zeros((BM_C, BN_C), dtype=tl.float32)
    for k in range(0, K_C, BK_C):            # 一个 K-loop 同时推进两个累加器
        a1 = tl.load(...).to(tl.float32); w1 = tl.load(...).to(tl.float32)
        acc2 += tl.sum(a1[:, :, None] * w1[None, :, :], axis=1)
        a2 = tl.load(...).to(tl.float32); w2 = tl.load(...).to(tl.float32)
        acct += tl.sum(a2[:, :, None] * w2[None, :, :], axis=1)
    acc2 += bias1[None, :]; acct += bias2[None, :]
    # 各自 round 到自己中间 dtype 之后才 add（DE3 位级契约）
    if MID2:  acc2 = round_through(acc2, MID2_DTYPE)
    if MIDT:  acct = round_through(acct, MIDT_DTYPE)
    acc = acc2 + acct
    tl.store(..., acc if OUT_F32 else acc.to(o_ptr.dtype.element_ty), mask=m_mask)
```

### §2.4 性能基准（本例实测，round-4 champion 按 B 分段）

| B 段 | cases | 加速比 geomean | 区间 | 备注 |
|------|-------|---------------|------|------|
| 1-12 | 26 | **1.953x** | [1.86, 2.10] | 融合压 launch 的主收益段 |
| 13-31 | 9 | 1.284x | [1.08, 1.49] | bm16 消 masked-lane |
| 32-95 | 10 | 1.139x | [0.99, 1.44] | bm32 段 |
| 96-127 | 3 | 1.067x | [1.00, 1.18] | BN256 gate 生效段 |
| 128-255 | 5 | 1.217x | [1.04, 1.57] | bm64 gate 生效段 |
| 256+ | 7 | **0.730x** | [0.62, 0.86] | 回归段：vector 外积 < aclnn Cube |
| 全量 | 60 | **1.3939x** | — | target 5.0 未达，平台期 |

**关键结论**（对本类的启示，非本例专属）:
1. 收益全部来自**结构**（段级融合 + pass-merge + tier law）；tile 族参数微调全线证伪
2. 最大 B 段可能结构性回归（向量外积吞吐 < aclnn Cube 路径 + tf32 降级）；若新算子的 B 上界更大，突破需另寻非 tl.dot 的 Cube 路径，不要在 tile 上消耗 round
3. 迭代史（7 轮）：R1 融合框架 1.2063 → R3 pass-merge 1.3579 → R4 tier law 1.3939 → R6 边界 32→17 **1.4083**（新 champion，两次 run 认证）；R5/R7 为 recipe 复推 + 结构 probe（嵌入段融合进 MLP 段被证伪回退，见 DE2）——**最后一个有效 lever 往往是 tier 边界而非新结构**
4. 同代码不同轮 geomean 差 ~±0.2 点属 profiler 方差（DE10），平台期判定前先排除
5. 1.39-1.41x 是本例 recipe 在 910B1 的实际天花板；新算子的天花板由其权重流量下限决定，勿直接沿用该数值作预期

---

## §3 常见陷阱与避免方法

### §3.1 通用陷阱（类别级）

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 按大 M 直觉用 tl.dot | skinny-M 降级非 Cube，慢 300-1600x | DE1 probe 决策程序 |
| 漏掉参考实现的显式 dtype 回转 | 位级不对齐，HiddenStateMismatch 难定位 | DE3 逐舍入点 constexpr 复现 |
| 共享 K 的多路 GEMM 各自开 kernel | 多一趟 tile-index 标量算术 + 中间量 GM round-trip | DE4 pass-merge |
| 把 add/cast/激活拆独立 kernel | 小 B 段 launch 开销主导 | DE2 段级融合 |
| 只碰少数 case 的定制 gate | 未触碰 case 的 ±1-3% 噪声淹没收益，不可认证 | DE5 整域 law |
| 单 kernel probe 结果直接外推全量 | probe 环境与全量 run 不同（程序数/噪声） | probe 建议必须全量 run 认证 |
| 跨段生产者-消费者融合（上游输出在下游 kernel 内重算） | 下游按 N-tile 切分时上游计算放大 N-tile 数倍 | DE2 判据：仅当下游只读一次才可融合 |
| tier 边界按等间距/2 的幂取整 | 边界不在 NUM_M 降档点上，白付一趟 W 面板重读 | DE5 标定尺：边界画在 NUM_M 减 1 的 B 值处 |
| 把 kernel 内三角函数重算当"免融合收益" | UB live-set 翻倍（3D 中间量 ×2 份）超限编译失败，`multibuffer=False` 救不回 | DE7 硬顶前置检查；重算需换 `tl.static_range` 半拆等降 UB 手段 |
| 重算路径中索引解耦错误（cos 半区读 freqs 对错权重列） | sin/cos 两半共享 freqs 但落不同输出列，索引必须分开维护 | 数值对不上先查索引映射（实测 rel err 3.6e-3 的来源） |
| 单次全量 run 即下 KEEP/REJECT 结论 | 同代码 run-to-run 噪声 ~±0.2 geomean 点，case 级可达 0.8→2.0 | DE10 两次 run 认证 |
| K-loop 重排族（A-resident / N-span / dual-N-tile） | 小 B 下程序数变化引发向量核饥饿，收益被淹没 | 默认证伪；NJP 累加链仅支持 {1,2,4}，组合不当 cc overflow / aicore 507014 挂起 |
| 改代码后 verify 失败值"逐位不变" | stale `__pycache__` 缓存旧代码 | 清缓存再 verify |
| 不同 kernel 误共享 grid 常量 | N 维 tile 数不同，verify 全挂无指向 | DE8 独立计算 |
| benchmark 进程未退出就读 perf 结果文件 | 文件按 case 逐个重写 | 先 poll 进程退出 |
| 用 profiler pipe-share 数值做轮间对比 | 同代码不同轮 share 漂移 >20% | share 只作类型证据，不作可比数值 |
| msprof 全 case PROFILER_COLLECT_FAIL | hostname 特殊字符 → "Invalid input path" | DE9 sitecustomize shim（强制） |

### §3.2 方法论（可迁移）

- **TTIR 检查法**：`~/.triton/cache/*/...ttir` 确认外积被 pattern-match 成 `tt.dot(tf32)` 属 lowering 结构性产物，源码层剪不掉（实测 EVEN_M 剪除中性）
- **认证粒度律**：改动覆盖整个 B 段的 law → geomean 可认证；只碰 <10% case → noise-dominated 关门
- **控制组方法**：改动只 gate 在某 B 段时，用未触碰段做对照组；真回归（如 2x）显著高于 ±0.05 噪声
- **平台期判定**：tile 族优化点全部已试/证伪 + profiler 确认非硬件极限（如 MMAD<50%）但修复点无落地点 → 停止 tile 尝试，结构上限已到

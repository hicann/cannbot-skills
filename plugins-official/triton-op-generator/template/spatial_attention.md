---
name: spatial_attention
description: CV-Attn-空间自注意力算子（特征图 token 化的空间 self-attention 带插段：4 个 nn.Linear 投影 + depthwise 空间缩减 conv+LN（ratio>1 压缩 KV）+ softmax(QKᵀ/√d)V + 跨 head 1×1 transform conv + instance_norm，PVT/SR-Attention 系）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束（fp32 全链路契约/conv 权重预转置/邻域 tile 物化禁令/tile 档位化）、Layer 2 算法骨架（八 kernel + S 矩阵流量模型 + 融合分派）、Layer 3 关键技巧（平面统计数学分解/转置 GEMM/两遍法 softmax）、精度闸门与证伪方向
metadata:
  type: reference
---

# CV-Attn-空间自注意力算子优化经验

本文档是 **"特征图 token 化的空间 self-attention，且 K/V 生产链或 scores 链带插段"**
这一类算子（`attention_index.md` 行 10a，一·3 空间 token 版 · 带插段）的经验合集，
覆盖 Phase 2/3/4。

- **§0 适用范围与算子分类**（子类标签 + 判别特征 + 形态识别五问）
- **§1 通用经验**（S1~S3：精度契约探测 / tile 档位化 / 核数动态读取）
- **§2 Layer 1 设计约束 L1.1~L1.8**（Phase 2 硬性边界，precheck 整节摘录）
- **§3 Layer 2 算法骨架**（八 kernel + S 流量模型 + 双路径分派判据）
- **§4 Layer 3 关键技巧**（平面统计数学分解 / 跨 head conv 融合 / 转置 GEMM / 诊断方法论）
- **§5 Phase 4 优化点清单**（收益排序 + **证伪方向全表** + 天花板估算）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与其它模板的分工**

> ⚠️ **本文件与 `mha.md` / `flash_attention.md` 的分工**：
> 本类算子**同时含 4 个 `nn.Linear` 投影段**（与行 6a 形态相同），投影段硬约束
> （`mha.md` §2.5 M1~M4、FA 卡 L1.11/L1.12 权重预转置、§4.5 分档）**依然全部适用**；
> attention 主链的 online softmax / 面积预算（FA 卡 L1.1/L1.3/L1.13）同样适用。
> 本文件**只补充"空间插段"带来的那部分**：depthwise 空间缩减 conv 的权重布局与
> 邻域 gather、K/V 序列长度的几何推导、transform conv + instance_norm 链要求的
> S 矩阵物化与流量模型，以及由此引出的 fp32 全链路契约、pad 行 NaN 污染、
> 3D tile 广播崩等 MHA 上不会遇到的坑。**三份都要读**；冲突时以本文件为准。
>
> **证据基础**：`EMSA`（Efficient Multi-head Attention，PVT 系 Spatial Reduction
> Attention）50 case 完整轨迹——
> B∈[1,4]、Lq∈[6,1295]、D∈[48,1536]、H∈[2,24]、**d_k≠d_v**、dtype fp32/fp16/bf16 混合、
> ratio∈[1,4]、apply_transform 半数 case。910B2C（20 AI cores）/ CANN 9.1.0 /
> triton-ascend 3.2.2。Phase 3 首版 1.1088，Phase 4 七轮 → **2.0973**（target 2.0 达标），
> 精度 **50/50**。
>
> ⚠️ **核心优化哲学**：这类算子的目标函数是**两段的加权和**——投影段是 fp32 GEMM
> 算力（占比常 >70%），transform 段是 **S 矩阵 `[B,H,Lq,Lk]` 的显存流量**（基线 8 遍
> 全量读写，最大 case 253MB/遍）。第一优化动作永远是：① 投影/conv 权重全部 host
> 预转置到"读侧连续"布局；② 把 S 的遍数从 8 压到 5（数学恒等式消统计遍）。
> attention 侧的 tile 微调在最后做。

---

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。

| 子类标签 | 入参/写法特征（可模式匹配） | 优化重心 |
|---|---|---|
| `sa-emsa` | 全插段：ratio>1 的空间缩减下采样＋LN 产 K/V（本类实测形态为 depthwise conv），**且** scores 链有跨 head 1×1 conv + softmax 后 instance_norm（apply_transform 门控） | 本卡全章节；S 流量模型 §3.2 是主目标函数 |
| `sa-sr` | 仅空间缩减插段：K/V = LN(下采样(queries))，下采样写法不限——dwconv / `AvgPool2d(ratio)` / stride-conv；无 transform/IN | §3 阶段 A（有权重时邻域 gather＋权重预转置 L1.2；AvgPool 无权重，退化为纯均值下采样）；attention 侧回落 FA 卡 |
| `sa-t` | 仅 transform 插段：ratio=1（K/V 直接投影），但 scores 过跨 head conv + IN | §3 阶段 C 物化链（S 物化 + 两遍法 + 平面统计恒等式）；无阶段 A |
| `sa-plain` | 纯特征图 token 三段式（`x.flatten(2).transpose(1,2)` 后无任何插段） | **不走本卡**，走行 10 `flash_attention.md` |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. K/V 生产链含**空间缩减下采样层**＋`nn.LayerNorm(D)`。下采样写法不限——depthwise
   conv（`nn.Conv2d(D, D, kernel_size=ratio+1, stride=ratio, padding=ratio//2,
   groups=D)`，本类实测形态）、`AvgPool2d(ratio)`（PVT 系常见）、stride-conv 等皆
   属本类；共同特征是 K/V 的 token 数 `Lk < Lq` 且由 `H×W` 网格几何推导（§0.2 Q2，
   不同下采样公式不同，逐字复刻参考）；
2. scores 链含跨 head 1×1 transform conv：`nn.Conv2d(num_heads, num_heads, 1)`
   作用在 `[B,H,Lq,Lk]` 上（等价逐平面 `S @ W[H,H]` + bias），常由 `apply_transform`
   布尔门控；
3. softmax 后接 `F.instance_norm`（对每个 `(b,h)` 平面 over `Lq×Lk` 归约，
   `affine=False`）；
4. 输入 queries 形如 `[B, Lq, D]` 且 Lq = H·W（特征图 token 化），投影段 d_k/d_v 可不等。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 影响 |
|---|------|------|
| **Q1** | golden 是否把全部输入/权重 `.float()` 后计算、仅末尾 cast 回 orig dtype？ | 决定 L1.1 **fp32 全链路契约**是否生效——本类普遍成立（探针法见 S1）。成立 ⇒ `input_precision="ieee"` 全程 + 禁一切中间降精度；不成立 ⇒ 回落 FA 卡 L1.4/L1.16 分 dtype 配方 |
| **Q2** | ratio>1？K/V 是否经空间缩减下采样？何种写法？ | 决定阶段 A 存在与 **Lk 几何推导**——下采样公式随写法不同：dwconv `K=ratio+1, pad=ratio//2, ho=(H+2·pad−K)//ratio+1`；`AvgPool2d(ratio)` 是 `ho=H//ratio`；ratio=1 时 `Lk=keys.shape[1]`（`H·W=Lq`）。**Lk 必须逐字复刻参考的推导**，错一格整盘 shape 崩 |
| **Q3** | `apply_transform` 且 `num_heads>1`？ | 决定双路径分派（§3.4）：False → flash 单 kernel（online softmax，S 不物化）；True → **S 必须完整物化一次**（instance_norm 是 softmax 值的全平面归约，结构上无法 online 融合），走阶段 C 物化链（§3.1） |
| **Q4** | d_k ≠ d_v / head_dim 非 2 的幂？ | dot 操作数 2 幂封顶 + 分维 mask（QK 用 dk 维、PV 用 dv 维，两套 BLOCK_D 独立分档）；本类实测 D∈[48,1536]、H∈[2,24]，head_dim 普遍非 2 幂 |
| **Q5** | S 规模多大（B·H·Lq·Lk 元素数）？ | 决定 qk+transform 融合分派阈值（§3.3，实测 **S≤500K 融合**）与 S 物化的显存预算（最大 case 253MB fp32，HBM 可容纳）；UB 预算 BQ·BLK≤8192（flash 路径） |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 精度契约探测：先钉死计算域再写任何 kernel

host 侧逐行 replay 参考 forward，找两处证据：① 输入/权重是否被 `.float()`；
② 输出是否仅末尾 `.to(orig_dtype)`。两者都成立 ⇒ fp32 全链路（L1.1）。
**探测结论写进 sketch 头部注释**，Phase 3/4 全程不可推翻（实测 rel_thr 1.2e-4 下
任何一处中间降精度都直接越阈）。

### S2 tile 参数档位化（bucket 量化）是本类生死线

50 case 的 `(Lq, Lk, D, H, dk, dv, ratio, dtype)` 唯一组合数 → 每个组合编译一个
kernel 版本。**tile 必须量化到固定档位集合**（如 BLK∈{128,256,512}、TOK∈{4,8,16}、
投影 tile∈{16/32 小档, 64×128 中档, 128×128×128 大档}），且**档位再按实际维度
封顶**（`tile = min(档位, bucket(ceil16(维)))`，防小 case 拿大 tile 整块 mask 浪费），
编译版本数收敛到 ~30 个且磁盘缓存跨 case 复用。不档位化 ⇒ verify 900s 超时、
0 用例启动（§8 陷阱表）。

### S3 核数动态读取 + 双核数分流

`NUM_CORES` 模块导入时 `get_device_properties()["num_aicore"]` 读一次（910B2C=20），
进程内保持常量（`mha.md` M1 同源）。depthwise conv / LN / gather 类纯 Vector 段
可用 vector 核数上限（40），GEMM/attention 用 cube 核数（20），grid 分别取 min。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

> 共 8 条（L1.1~L1.8），全部在本类算子上实测。L1.1/L1.2 决定成败量级，
> L1.3~L1.5 是编译崩溃/超时类（极难从输出反推），L1.6~L1.8 是静默错值类。
> precheck 必须整节摘录。

### L1.1 ★★★ fp32 全链路精度契约：golden `.float()` ⇒ 实现 `input_precision="ieee"` 全程，唯一舍入点在最终输出 store

```python
# ✅ 全链路：load 后 .to(tl.float32)，所有 dot 带 input_precision="ieee"
acc = tl.dot(a, b, input_precision="ieee")        # 投影 / QK / PV / transform 全部如此
# ❌ 任何中间降精度（输入 dtype 是 fp16/bf16 也不行——golden 在 fp32 上算）
acc = tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
```

- FA 卡 L1.16（三档 cast / half 路径 hi-lo 双 dot）在本类**不适用**——golden 计算域
  与输入 dtype 无关，hi/lo 拆 3 dot 同时违反 `mha.md` M4（dot 链 ≤2）。
- **"fp32 cube 慢、必须降精度"是误传**：fp32 ieee cube 的实测吞吐可达当前硬件
  标称峰值的 1/4 量级，单 kernel 投影微基准能拿到 fp32 实测上限的 ~1/3
  （*910B2C 标定示例：27~31 / 85 TFLOPS；标定与判据见 §5.3-2*）。
  1.9 TFLOPS 级读数全部来自分段 sync 剖析毛刺（§5.2），不是 cube 真实吞吐。

### L1.2 ★★★ depthwise conv 权重必须 host 预转置成 `[K*K, D]` 行主序

```python
# ❌ kernel 内按 [D, K, K] 原始布局加载：列主序地址 d*KKW+kk 碎成 16 元素宽离散段
w = tl.load(w_ptr + d * KK + kk)                    # 该 load 单项拖慢 8.5×
# ✅ host 侧 w.reshape(D, K*K).t().contiguous() → [K*K, D]，kernel 行连续 load
w_tile = tl.load(wt_ptr + kk_offs[:, None] * D + d_offs[None, :])   # [K_PAD, DBLK]
```

实测：完整 kernel 1345us → 157us（**8.5×**），而 load-only 对照（同寻址、只 gather +
平凡累加）138us——证明瓶颈就是权重 load 的寻址模式，不是 conv 算本身。
与投影权重预转置（FA 卡 L1.12）同一原理在 conv 权重上的镜像。
（AvgPool 型下采样无权重，天然无此瓶颈，本条跳过。）

### L1.3 ★ 邻域循环禁止持大 tile：循环体内的 `[TOK, BLOCK_D]` 会被静态物化 K² 份

```python
# ❌ for kk in range(K*K): acc += load(x[b, ih*W+iw, :]) * w   # tile 编译期展开 K² 份
#    实测 UB 需求 17.2Mb vs 预算 1.57Mb，直接溢出
# ✅ 任务切到 (token × d_block)，邻域一次 [K_PAD, DBLK] 2D gather（K_PAD=ceil16(K²)），
#    循环体只持一份 tile
```

### L1.4 ★ LN / 逐 token 类 kernel 的 `TOK × BLOCK_D` 乘积封顶 ~4K 元素（fp32 多临时）

TOK 按 BLOCK_D **反比**取档：BLOCK_D≤256→TOK=16，≤512→8，≥1024→4。
16×1024 组合必溢出（实测 231KB > 192KB UB）。

### L1.5 ★ tile 档位化（见 S2）——违反表现是 verify 全量超时而非报错

### L1.6 ★ 3D tile 的 head 维必须放最后一维 `[BQ, BLK, HP]`

```python
# ❌ [HP, BQ, BLK] 布局：offs_hp[:,None,None] 与 rows[:,None,None] 在 index 0 广播，
#    HP==BQ 时静默融合成错误形状（报 make_shape_compatible ... 32 and 16，且报错行
#    距真因 200+ 行）
# ✅ acc3 = tl.zeros((BQ, BLK, HP), tl.float32); acc3 += s[:, :, None] * wcol[None, None, :]
```

### L1.7 ★ pad 行统计污染：`s_sum=0 → inv=inf → e=0×inf=NaN` 传染全平面

tile 行 `TI > Lq` 的 pad 行在 softmax 统计中产生 `inf/NaN`：S 本体有 store mask 幸免，
**行统计 buffer（Z/q²/part）会全 NaN** 并经 stats 归约传染整个输出（实测 0/50 全 NaN
形态）。任何行级统计落盘前必须 `tl.where(r_ok, x, 0.0)` 逐处包住。

### L1.8 全局命名常量（`_NEG` 等）必须 `tl.constexpr` 注解，否则
`NameError: Cannot access global variable`（kernel 内引用 host 全局时）。

---

## §3 Layer 2: 算法骨架

### §3.1 主骨架（数据流按"空间缩减 → 投影 → attention → 输出投影"四阶段组织）

阶段职责与任务分解是**参考方向**——子类不同（`sa-sr` 无 transform 段、`sa-t` 无
缩减段）、shape 不同，最优 kernel 数在 4~8 之间浮动。照抄**阶段职责与合并判据**才有
意义；kernel 命名与数量是具体算子的终态产物，重新设计（四层隔离：Layer 2 只给骨架
方向，输出必须是全新草图）。

```
阶段 A（ratio>1） 空间缩减   queries[b,Lq,D] --depthwise conv＋LN--> reduced[b,Lk,D]
    · Lk = ho·wo 由 H×W 网格几何推导（K=ratio+1、pad=ratio//2），逐字复刻参考（§0.2 Q2）
    · conv 任务 = (out_token × d_block)：邻域一次 [K_PAD, DBLK] 2D gather，
      权重 host 预转置 [K²,D] 行主序（L1.2），向量 FMA 无 dot（depthwise 无 Cube 需求）
    · 邻域循环禁止持大 tile（L1.3）；LN 与 conv 默认分离两个 kernel——LN 需整行 D
      归约，TOK×BLOCK_D 预算（L1.4）与 conv 的 DBLK 档位互相牵制（§3.3）

阶段 B          投影          Q/K/V = src @ W{q,k,v}，src = ratio>1 ? reduced : keys/values
    · 投影权重全部 host 预转置 [in,out]（FA 卡 L1.12）；tile 按 shape 分档
      （大 128×128×128 / 中 64×128×128 / 微 16~32，S2 档位化）
    · K 投影产出转置布局 [B,H·dk,Lk]：直接算转置 GEMM Wk@xᵀ（权重保持原始
      [out,in] 免转置），读侧连续、免散射 store（§4.3）
    · d_k≠d_v ⇒ QK/PV 两套 head_dim 档独立分（§0.2 Q4）

阶段 C          attention 主链   host 条件 launch 分派（禁 kernel 内 constexpr 分支，§3.4）
    · 面积预算管**两处** tile：S tile（BQ·BLK）与累加器 tile（BQ·BLOCK_D）都 ≤ 8192，
      预算内 **BQ 优先放大**（q 端复用率高）
    · transform=False → flash 单 kernel：online softmax＋PV（FA 卡骨架），
      dot 链=2（M4 合规），S 永不物化；`Lk ≤ BLK` 一块装下时按 constexpr 特化
      **单趟精确 softmax**（免 online rescale 分支——ratio 缩减后 Lk 普遍小，
      命中率高）
    · transform=True → S 物化一次（IN 是 softmax 值的全平面 (b,h) 归约，结构上
      无法 online 融合），子链按 S 规模分派（§3.3）：
      ① QK·scale 与跨 head 1×1 conv：小 S 融合单 kernel（3D 累加器
         acc[BQ,BLK,HP] += s[:,:,None]·wcol，h 循环每迭代 1 dot；**HP 取 head
         的 2 幂档 16/32**，权重 host pad 补零）；大 S 分离两个 kernel
         （QK GEMM 与 conv 各一，QK 同 8192 预算 + BQ 优先）
      ② softmax 两遍法：pass1 行 max；pass2 存未归一 e ＋ 行统计 Z_i=Σe、q2_i=Σe²
         （pad 行 tl.where 包住，L1.7）
      ③ IN 平面统计：**两级确定性归约**（部分和 [B·H·n_it, 2] → 归约 kernel →
         [B·H, 2]；**禁 fp32 atomic 累加**——非确定）；mean ≡ 1/Lk 恒等免统计遍；
         sumsq = Σ_i q2_i/Z_i²（§4.1）
      ④ PV：读取时归一 p = e·(rstd/Z_row) − mean·rstd，dot(p,V)（归一遍消失）

阶段 D          输出投影      O @ Wo ＋ bo → cast orig dtype（全链路唯一舍入点，L1.1）

host
  · ⚠️ 权重经 **forward 传参**（非 __init__ 持有）⇒ 首次用时预转置 ＋ 按
    (data_ptr, shape, dtype) **跨调用缓存**（第二次起零开销，缓存超限整清防泄漏）；
    小权重（如 conv 的 D×K²）现转开销 µs 级，缓存与否皆可
  · apply_transform and num_heads>1 决定阶段 C 分派；S workspace 仅物化路径分配
```

### §3.2 ★ S 矩阵流量模型（transform 路径的目标函数）

S = `[B,H,Lq,Lk]` fp32，基线（逐算子分离实现）**8 遍全量流量**：
qk 写 1 + tconv 读 1 写 1 + softmax 读 1 写 1（另需统计读）+ IN 归一读 1 写 1 + PV 读 1。
终态 **5 遍**：

| 削减 | 手段 | 省几遍 |
|---|---|---|
| softmax 两遍法（存未归一 e + 行统计 Z/q²） | 统计并入 pass2，免第三遍读 | −2 |
| 平面统计数学分解（§4.1） | mean 恒等免统计、sumsq 由行统计组装 | （含上行） |
| IN 归一并入 PV 读取（`p=e·rstd/Z − mean·rstd`） | 免独立归一 kernel 的读写 | −1 |

最大 case（B=2,H=20,Lq=Lk=1258）S≈253MB → 8 遍 ≈ 2.0GB → 5 遍 ≈ 1.27GB，
直接决定该 case 档的延迟下限（§7 MFU：3.12% 属结构性预期，非实现问题）。

### §3.3 阶段合并判据（Phase 2 草图期逐条回答）

| 判据 | 融合 | 分离 |
|---|---|---|
| **QK 与 transform conv** | S ≤ 500K（**全局 B·H·Lq·Lk**，§0.2 Q5）：融合单 kernel（QK 与 conv 同体），3D 累加器 `acc[BQ,BLK,HP] += s[:,:,None]·wcol`，h 循环每迭代 1 dot | S > 500K：h 循环内重复 load q/kt + `[16,dk]×[dk,32]` 小 dot，大 T case 实测退化 30-40%（§5.2） |
| **缩减 conv + LN** | D 小（≤256）时可试融合 | 本类默认分离：LN 需整行 D 归约，TOK×BLOCK_D 预算（L1.4）与 conv 的 DBLK 档位互相牵制 |
| **k 投影与 Kt 布局** | 转置 GEMM 直接产 `[B,H·dk,Lk]`（§4.3） | 自然 GEMM + 转置 store = 多一个 scatter 通道，实测恒劣 |

### §3.4 双路径分派（Q3 的落点）

`transform=False` → flash 单 kernel（S 永不物化，Lk 缩减收益直接体现）；
`transform=True` → 物化链（S 物化一次，流量按 §3.2 账本优化）。
**分派在 host forward 做条件 launch，不在 kernel 内 constexpr 分支**
（4 段 + attention 合并进同一 `@triton.jit` 是 `mha.md` §2 明令禁止的形态）。

---

## §4 Layer 3: 关键技巧（可参考，变量名/结构必须重新设计）

### §4.1 ★★★ 平面统计数学分解（instance_norm 的两遍扫描终态）

softmax 后值的平面统计有恒等式可用：

```
mean(p) ≡ Σ_i (Σ_j p_ij) / (Lq·Lk) = Lq·1 / (Lq·Lk) = 1/Lk    # 每行和为 1，免统计
Σp²    = Σ_i Σ_j (e_ij/Z_i)² = Σ_i q2_i / Z_i²                  # 行局部量组装
```

归一挪到 PV 读取时一次完成：`p = e·(rstd/Z_row) − mean·rstd`（仿射两系数先在
stats 归约 kernel 里算好成行向量）。**效果：IN 的统计+归一遍全部消失**，是 S 8→5 遍
的核心（单轮 +2.5% 几何平均，且大 S case 受益最大）。

### §4.2 ★ 跨 head 1×1 conv 融进 QK（小 S 专属）

transform conv 是逐平面 `S[b,:,i,j] = Σ_h W[h,h']·S0[b,h,i,j] + bias[h']`——
h 循环内 `acc3[BQ,BLK,HP] += s[:,:,None] * wcol[None,None,:]`，每迭代 1 个 QK dot
（M4 合规）。**仅小 S 用**：大 S 的 h 循环重复 load q/kt 且 dot 形状退化（§3.3）。

### §4.3 ★ 转置 GEMM 免散射

需要 `[B, H·dk, Lk]` 转置布局（KV 循环读侧连续）时，**不要**自然 GEMM 后转置 store——
直接算 `Wk @ xᵀ`（M=H·dk, N=Lk），权重保持原始 `[out,in]` 行主序**免预转置**，
store 自然落转置布局。读写两侧同时连续。

### §4.4 softmax 两遍法（存未归一 e + 行统计）

pass1 行 max → pass2 `e = exp(s − m)` **不除 Z 直接 store**，同时累行统计
`Z_i = Σe`、`q2_i = Σe²`（供 §4.1 组装）。pad 行统计 `tl.where(r_ok, q2·invz, 0)`
（L1.7）。下游 PV 才做归一——省掉独立的归一遍。

### §4.5 ★ 诊断方法论：load-only 对照 + 单 kernel 微基准 A/B

- **load-only 对照**：怀疑访存瓶颈时写一个同寻址、只 load + 平凡累加的对照 kernel。
  实测纯 gather 138us vs 完整 1345us → 一行代码把矛头指向权重 load（L1.2 的发现路径）。
- **微基准 A/B**：单 kernel 20 次平均，**不要用分段 sync 剖析**——权重缓存冷启动
  时它给出 96ms 级毛刺（比真实差 70×，§5.2）。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益；七轮实测轨迹，50-case 几何平均判定）

| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 投影 shape 分档 tile（微 tile → 128×128×128）+ attention 面积预算 4096→8192 + softmax 多行/program | 1.1088 → 1.5302（**+38%**，单轮最大） | 总是 |
| 2 | ★★★ 缩减 conv 权重预转置（该 kernel 8.5×）+ softmax/stats 融合 | → 1.8073（+18%） | ratio>1 |
| 3 | ★★ softmax 两遍法 + 平面统计数学分解（S 8→5 遍） | → 1.9425（+2.5%，大 S case 更多） | transform 路径 |
| 4 | ★★ tconv tile 放大 + softmax BLK 512 档 + flash BQ 优先 | → 1.8944（+5%） | 总是 |
| 5 | ★ 小 S 的 qk+tconv 融合分派（S≤500K）+ BLK 256 档 | → 1.9784（+1.8%） | transform 且 S 小 |
| 6 | ★ 融合阈值 100K→500K + tconv 加宽 + TI 放大 + pv BLK 256 | → **2.0973**（+6%） | 同上 |
| 7 | tile 档位化 bucket（S2/L1.5） | 编译爆炸 → 可运行 | 首版必须（非收益项） |

最终分布：≥2x 共 29 / 1.5~2x 共 7 / 1~1.5x 共 8 / <1x 共 6；最快 case 7.19x
（小 Lq + 大 D 投影主导），最慢 5 个 0.82~0.86x（超大 Lq/D 的 transform 或大 flash
形态，§5.3 结构域上限）。

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱**）

| 方向 | 结果 |
|---|---|
| **邻域 `tl.static_range` 展开 + 逐 tap 1D load** | ⛔ 11.8ms，比 2D gather 版差 **8×**（串行小 load 无 tile 化；且踩 L1.3 物化） |
| **缩减 conv 的 DBLK 放大到 512/1024** | ⛔ 无收益（1451 vs 1345us）——行段长度不是瓶颈，**权重寻址模式才是**（L1.2） |
| **qk+tconv 无条件全量融合** | ⛔ 大 T case 退化 30-40%（h 循环小 dot + 重复 load）；仅 S≤500K 融合（§3.3） |
| **分段 sync 剖析定位瓶颈** | ⛔ 冷启动毛刺 70×（96ms vs 真实 1.3ms），结论全错；用微基准 A/B（§4.5） |
| **dot 降精度（bf16/fp16 输入 case）** | ⛔ 精度契约禁止（L1.1，rel_thr 1.2e-4）；hi/lo 拆 3 dot 违反 M4 |
| **`a % b` 取模寻址** | ⛔ checklist 违规 + 本后端代码生成劣化——改 `a - a//b*b`（实测 11 处全改后通过） |

### §5.3 天花板估算：先标定、再算账（⚠️ 数字随 NPU 型号/集群状态变化，禁止直接搬用）

每条**先标定、后估算**——标定给出方法，判据给的是**相对量**（跨硬件有效），
括号里的 910B2C 数字只是量级示例：

1. **S 流量下限 = 遍数 × S 字节数 ÷ 实测 HBM 带宽**
   遍数按 §3.2 账本（终态 5 遍）；`S 字节数 = B·H·Lq·Lk·4B`；带宽用当前硬件
   **实测值**（大块 copy 微基准，或 `triton-mfu-analyzer` 的 peak 标定），不要用
   规格书标称。得到的毫秒数是 transform 路径的结构下限——**任何 tile 微调都越不过**，
   优化只剩"减遍数"（§3.2/§4.1）。
   *量级示例（910B2C，实测带宽 ~100GB/s 级）：S=253MB ⇒ 5 遍 ≈ 1.27GB ⇒ ~13ms。*
2. **cube 吞吐上限现场标定，判据用百分比**
   fp32 ieee 的实测上限 ≠ 标称（标称是 fp16/bf16 牌面，且不同型号差距大）。先测
   当前硬件的 fp32 实测上限（大 GEMM 微基准 / mfu peak_calib），再算**投影段微基准
   吞吐 = fp32 实测上限的百分比**。**判据（相对，跨硬件有效）：投影段 < 40% ⇒
   剩余空间在 tile/流水调度；≥ 50% ⇒ 已近结构域上限，转攻流量侧，不要再磨 GEMM。**
   *量级示例（910B2C）：fp32 实测上限 85 TFLOPS（标称 354 的 24%），投影段微基准
   27~31 TFLOPS = 35% ⇒ 判"还有 tile 空间"。*
3. **噪声带自测，双跑判定**
   同代码两次 benchmark，per-case 均值差 > 5% 判优化无效；geomean 达标判定必须
   双跑。*量级示例（910B2C 集群）：跨会话 ±2~5%，单 case 尾部可达 ±30%。*

---

## §6 精度闸门

### §6.1 判定顺序（错一步会把 bug 归错类）

1. **先看是不是全 NaN**：pad 行统计污染（L1.7）的典型形态是 0/50 全 NaN 或
   大面积 NaN——先查行统计的 `tl.where`，不要调精度；
2. **再看覆盖率**：`torch.empty` workspace + 部分路径未写 = 未初始化内存
   （NaN 位置断言 + 精度判定同时触发）；
3. **最后才是精度契约**：按 S1 探测结论核对每一处 dot 的 `input_precision` 与
   唯一 cast 点（L1.1）。

### §6.2 舍入点契约（fp32 全链路 = 单舍入点）

| # | 舍入点 | 落点 |
|---|--------|------|
| ① | 全部中间量 | **无舍入**——load 后 .to(f32)，ieee dot，统计/softmax/归一全 fp32 |
| ② | 最终输出 store | `.to(orig_dtype)` 唯一一次，对齐 golden 末尾 cast |

与 FA 卡 L1.16（三档 cast）、`sparse_unfold.md` L1.2（分 dtype softmax 域）**口径相反**
——那两类的 golden 在输入 dtype 上逐步落盘；本类 golden 在 fp32 上算。
**判据永远是 S1 探测结果，不是卡片习惯**。

### §6.3 与 FA / mha 卡既有条目的冲突判据（3 条，搬运结论前先核对）

| # | 冲突 | 本类实测 | **判据** |
|---|------|----------|----------|
| ① | FA 卡 L1.16 三档 cast vs 本卡 L1.1 单舍入点 | 本类 golden `.float()` 全链路，任何中间 cast 越阈 | golden 计算域在哪，cast 点就在哪（§6.2） |
| ② | FA 卡 L1.5「p 不能降 dtype」 | 本类 p（未归一 e）恒 fp32 存储 | 同①，本类无低精度存储域 |
| ③ | `mha.md` M4 dot 链 ≤2 | 本类物化链各 kernel 天然 1 dot/循环；QK+conv 融合 kernel h 循环每迭代 1 dot 合规 | 融合时按 h 循环拆 dot，不堆 3 dot |

### §6.4 定位手法（"结构对但 NaN/越阈"时的四步）

1. host 逐行 replay 参考 forward（含 Lk 几何推导，Q2）确认语义 = 0 差；
2. 逐层中间量隔离（reduced → Q/K/V → S → softmax 统计 → stats → O → out），
   找第一个 maxd 跳变层；
3. **load-only 对照**（§4.5）隔离访存与计算；
4. pad 行探针：把 `TI > Lq` 的行单独 dump，确认统计 buffer 无 inf/NaN（L1.7）。

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

- **几何平均 = 唯一汇总口径**：50 case 的 speedup 按几何平均，异常 shape（s_i 非有限
  正数）不计入但保留明细；`passed_cases` 以 verify_result.json 为准（perf 的 pass 只代表
  进程未崩溃）。
- **MFU 三口径**（triton-mfu-analyzer，top-3 理论 FLOPs case，msprof 采集）：

| case | 形态 | 实际吞吐 | MFU@标称 353.89 | **MFU@fp32 实测 85.01** |
|---|---|---|---|---|
| 36 | fp32, r=1, 无 T, (2,667,1440) | 13.66 TFLOPS | 3.86% | **16.07%** |
| 45 | bf16, r=2, T, (2,1015,1152) | 17.74 TFLOPS | 5.01% | **20.87%** |
| 48 | bf16, r=1, T, (2,1258,1280) | 2.65 TFLOPS | 0.75% | **3.12%** |

  ⚠️ **修正口径必须按 fp32 实测上限归一**：输入 dtype 是 bf16/fp16 也不行——计算全
  fp32（L1.1），按输入 dtype 峰值归一会高估（高估倍数 = 实测 fp16 上限/fp32 上限，
  *910B2C ≈ 3.8×*，换硬件须重标定，方法见 §5.3-2）。case 48 类大 transform case 是
  S 流量受限（§5.3-1），MFU 3% 属结构性预期。
- **kernel 级 MFU 拆解**可能因 host 预转置引入的 ACL kernel 行不整除被工具 skip——
  属工具限制，case 级结论不受影响。

---

## §8 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| verify 900s 超时、0 用例启动 | tile 唯一组合编译爆炸 | S2/L1.5 档位化 |
| `PlanMemory Failed` / UB 溢出 17.2Mb | 邻域循环持大 tile 静态物化 K² 份 | L1.3 |
| LN kernel UB 溢出 231KB | TOK×BLOCK_D 超 4K | L1.4 反比档 |
| 输出大面积/全部 NaN（0/50） | pad 行 `0×inf` 统计污染 | L1.7 `tl.where(r_ok,…)` |
| `make_shape_compatible ... 32 and 16`（报错行距真因很远） | 3D tile head 维在 index 0 广播 | L1.6 `[BQ,BLK,HP]` |
| `NameError: Cannot access global variable` | host 全局常量未 constexpr | L1.8 |
| checklist `%` 违规 / 代码生成劣化 | `a % b` 寻址 | 改 `a - a//b*b` |
| 融合版大 case 退化 30-40% | h 循环小 dot + 重复 load | §3.3 S≤500K 才融合 |
| 权重 load 单项 8.5× 慢 | conv 权重列主序碎段 load | L1.2 预转置 `[K²,D]` |
| 剖析读数与实际差 70× | 分段 sync 冷启动毛刺 | §4.5 微基准 A/B |
| Lk 推导错一格，全盘 shape 崩 | 未逐字复刻几何推导 | §0.2 Q2 |

---

## §9 与其它模板的分工

| 文件 | 何时用 |
|---|---|
| `flash_attention.md` | attention 主链（无 transform 的 flash 路径、online softmax、面积预算）。**本类算子也要读**，其 L1.1/L1.3/L1.9/L1.11~L1.13 依然成立 |
| `mha.md` | **投影段交叉读**（本类含 4 投影）：§2.5 M1~M4 核数/grid/UB 剪枝/dot 链、§4.1 投影 GEMM 终态架构与 tile 档 |
| `sparse_unfold.md` | 同为 CV 空间类但机制不同：它是 `F.unfold` **邻域展开构造 KV**（块局部 softmax），本类是**全局空间 token + 空间缩减 conv**（S 全平面）。Lk 减少的手段不同，骨架不互替 |
| `attention_index.md` | attention 家族定 `category` 的唯一入口。本文件对应其**行 10a**（一·3 空间 token 版 · 带插段）；纯三段式无插段是行 10（`flash_attention.md`） |
| **本文件** | §0.1 判别特征命中即用（sr-conv / transform conv / instance_norm 任一插段） |

冲突时以本文件为准（本文件结论均在空间插段形态上实测）。

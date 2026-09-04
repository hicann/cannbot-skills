---
name: sparse_unfold
description: 邻域展开（Sparse-展开型）attention 算子（F.unfold/im2col 构造 halo 邻域 KV + 块局部 softmax(QKᵀ/√d)V，常带 nn.Linear 投影段）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束（aclnn 舍入契约/解码安全/dot 2 幂/BN 档位）、Layer 2 算法骨架、Layer 3 关键技巧、精度闸门与证伪方向
metadata:
  type: reference
---

# 邻域展开（Sparse-展开型）attention 算子优化经验

本文档是 **"attention 主链的 KV 由 `F.unfold` / im2col 邻域展开构造"** 这一类算子
（`attention_index.md` 行 9，三·3 邻域展开）的经验合集，覆盖 Phase 2/3/4。

- **§1 通用经验**：跨形态共有的工程建议（权重复刻 / 空输入分支 / 常量表边界）
- **§2 Layer 1 设计约束 L1.1~L1.5**（Phase 2 硬性边界）
- **§3 Layer 2 算法骨架**（Phase 2 参考方向 + 收益判据）
- **§4 Layer 3 关键技巧**（Phase 3 编码 + Phase 4 优化）
- **§5 Phase 4 优化点清单**（含**证伪方向全表** + 天花板估算）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与其它模板的分工**

> ⚠️ **本文件与 `flash_attention.md` 的分工**：
> `flash_attention.md` 覆盖"KV 全扫 + online softmax"的稠密 FA 主链，**它的所有结论
> 在本类算子上依然成立**（尤其 L1.1 `tl.dot` dtype 契约、L1.9 grid 收缩、L1.11/L1.12
> 权重复刻与预转置、L1.16 三档 cast）。本文件**只补充"邻域展开"带来的那部分**：
> unfold 物化的消除、恒单块的块局部 softmax、NCHW↔token 布局变换，以及由此引出的
> 一批稠密 FA 上不会遇到的坑（aclnn 行布局舍入、softmax 分 dtype 域、解码 miscompile）。
> **两份都要读**；冲突时以本文件为准（本文件的结论都在展开形态上实测过）。
>
> **证据基础**：`50_HaloAttention` 50 case 的完整定位轨迹——
> fp16 24 / bf16 15 / fp32 10（block∈{2..8}，halo∈{0..4}，c∈{8..2048}，含 b=0 空输入
> case），910B2C / 24 AI 核。iter_0 **1/50** → 16 个对照实验逐项定位 → **精度 50/50，
> 两次独立 benchmark 几何平均 2.0973 / 2.1243**（target 2.0 达标；分 dtype
> fp16 2.0240 / bf16 2.3424 / fp32 1.9353）。
>
> ⚠️ **核心优化哲学**：这类算子的参考实现里，`F.unfold` 物化 `[b, c·S_kv, L]` 巨量拷贝
> 是**显性浪费**——halo 重叠使同一像素被读进多个窗口，读放大 `(1+2·halo/blk)²` 倍
> （block=8/halo=4 时 **4 倍**），外加 NCHW→块主序的 permute 链。
> **第一优化动作永远是：把展开从"物化"改成"每像素单次 dense 投影 + 按块窗口 gather"。**
> 同时认清：投影 GEMM 占总算力 **>90%**（c=2048 case 估算：投影 ~866 GFLOP vs
> attention ~68 GFLOP），**attention 侧优化的天花板有限，先攻投影侧**。

---

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。

| 子类标签 | 入参/写法特征（可模式匹配） | 优化重心 |
|---|---|---|
| `su-halo` | `kv = F.unfold(x, kernel=blk+2*halo, stride=blk, padding=halo)` 构造 KV 邻域 + `nn.Linear` 投影段（to_q/to_kv/to_out，forward 内 `hash(key)` seed 化创建）+ 块局部 `softmax(QKᵀ/√d)V` | 每像素单次 dense 投影 + kv 按块 gather；**§6 舍入点契约**（本类头号失败源）；NCHW↔token 布局 kernel 化 |
| `su-plain` | 同上但**无投影段**（q/k/v 直接来自展开的通道） | 同上去投影段；主攻 gather 与 attention 融合 |
| `su-win` | 非 unfold 但同为"固定几何窗口选 KV"写法（手写邻域块 gather） | 复用窗口行号推导与舍入契约；无 unfold 物化可消时主攻访存 |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. 参考实现里出现 `F.unfold(...)` / `tensor.unfold(...)` 且其输出喂进 attention 的 K/V；
2. 输入是 NCHW 特征图，attention 的 token 来自
   `x.reshape(b,c,h//blk,blk,w//blk,blk).permute(0,2,4,3,5,1)` 分块展开；
3. 存在 `kernel_size=block+2*halo` 这种"块 + halo 邻域"参数对；
4. 输出需要从块主序 token scatter 回 NCHW（`.permute(0,5,1,3,2,4).reshape(b,c,h,w)` 形态）。

### §0.2 ★ 形态识别五问（**Phase 2 第一步必须回答**）

| # | 问题 | 影响 |
|---|------|------|
| **Q1** | `S_kv = (blk+2·halo)²` 是否 ≤ 256？ | 本类普遍成立 ⇒ **恒单块、单趟精确 softmax**，online rescale 分支零执行（FA 卡 L1.3 保留但退化）；>256 时回落 FA 卡分块流式（单块/分块取舍见 §3.2 判据行） |
| **Q2** | halo padding 位置参考是否掩码？ | 展开型参考**普遍不掩码**（`# Mask out padding (not implemented here)` 注释为证）——padding 列 k=v=0 → score=0 **真实参与 softmax 分母**，与 FA 卡 F1 语义**相反**（§6.2 判据③） |
| **Q3** | 投影段第三个 `nn.Linear` 是否 `bias=True`（默认）？ | 逐字核对参考源码；**漏 bias 是本类隐蔽失败源**（op50 首版实测踩坑，见 §8） |
| **Q4** | 权重是否 forward 内 seed 化创建（`hash(key)`）？ | 复刻配方见 §1 S1；`manual_seed(42)` 无条件调用 + cache-miss 分支 rng 时序逐字复刻 |
| **Q5** | 评测 dtype 是否三档混合（fp16/bf16/fp32）？ | 决定 L1.2 的 softmax 域配方是否分档——fp16 专属一步 cast，bf16/fp32 **不能**做 |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 权重复刻：`hash(key)` seed 配方（FA 卡 L1.11 在本类的落点）

forward 内 `torch.manual_seed(42)` **无条件调用**（逐字复刻，勿省）；cache-miss 分支
rng save → `manual_seed(hash(key) & 0xFFFFFFFF)` → **CPU fp32** 逐字复刻
`nn.Linear` 初始化 → rng restore → `.to(dev,dt)` → 预转置 `[in,out]`（FA 卡 L1.12）。

```python
# ❌ NPU 上低精度采样 / 公式手推 bound —— 构造 device、采样 dtype、顺序次数、rng 四处缺一即全错
wq = torch.empty(INNER, c, device='npu', dtype=torch.float16).uniform_(-b, b)
# ✅ CPU fp32 字面量常数（host 侧禁 math.sqrt，repr 级字面量 round-trip 逐位一致）
_GAIN  = 0.5773502691896257      # sqrt(2/(1+sqrt(5)**2)) 的双精度 repr
_SQRT3 = 1.7320508075688772
bound_in  = _SQRT3 * (_GAIN / (c ** 0.5))      # kaiming_uniform_(a=√5) ≡ uniform(±1/√fan_in)
bound_out = _SQRT3 * (_GAIN / (INNER ** 0.5))
wq  = torch.empty(INNER, c).uniform_(-bound_in, bound_in)     # CPU fp32，顺序/次数与参考一致
wkv = torch.empty(2*INNER, c).uniform_(-bound_in, bound_in)
wo, bo = ..., ...               # ⚠️ to_out 默认 bias=True：weight 与 bias 都要建（Q3）
```

同进程内 `hash(key)` 稳定（tuple 含 int/dtype/device，无 str）；**权重单验先行**——
先逐个 `torch.equal` 再验投影最后验 attention（FA 卡 L1.11 诊断顺序，
权重错与布局错**互相掩盖**，只修一处 passed_cases 仍为 0）。

### S2 空输入分支逐字复刻

参考含 `if b == 0 or c == 0 or h == 0 or w == 0: return x` 时，实现同样直接返回 x，
**不进任何 kernel launch**（b=0 case 在 benchmark 里 `PROFILER_COLLECT_FAIL` 属预期，
勿当成失败去修——见 §7）。

### S3 常量表 host 预计算是被允许的模式，但有边界

窗口行表 / tile 表 / 像素表等**纯索引常量**在 host Python 预计算（shape 级 cache）
不违反 FA 卡 L1.6（那禁的是**数据布局搬运**）。但注意两点：
① 表的生成不能依赖"无 dot kernel 自己算行号"——大规模下同样踩 L1.3 的解码
miscompile（op50 实测 rowmap kernel 自身产出越界行号 qrow=4108 > 4095）；
② 终态更优解是 **constexpr 除数直接在 kernel 内解码**（零额外表与 launch，L1.3），
host 表只作兜底。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

> 本节共 5 条（L1.1 ~ L1.5），全部在本类算子上实测。违反前两条的表现是
> **"结构正确但 ~96-99% 元素错 1-2 ulp"**，违反后三条是 miscompile/编译崩溃——
> 都极难从最终输出反推，precheck 必须整节摘录。

### L1.1 ★★★ aclnn 低精度算子的舍入依赖输入行布局——凡与参考共用 aclnn 的边界必须同布局；自研 Triton GEMM 顺序 K 累加对 M 行布局不敏感

实测（op50，case `[6,192,38,10]` fp16）：**同一份输入数据**，aclnn `F.linear`
像素主序 vs 块主序（参考 `q_inp/kv_inp/out` 的 `(b, nblk, s)` 序）——

| 对比 | 结果 |
|---|---|
| 块主序 `F.linear` vs 参考输出 | **maxdiff = 0.0（逐位一致）** |
| 像素主序 `F.linear` vs 参考输出 | **98.7% 元素不等，maxd 0.0449，mismatch(2⁻¹⁰) 96.3%** |

aclnn 的分块/累加序随 M 布局变化，1 ulp 的舍入差经 softmax→PV→out 投影逐层放大后
**整盘越阈**。

```python
# ❌ attention 输出按像素主序落盘，再喂 out 投影 —— 即使数值路径全对也全错
tl.store(o_ptr + pixel_row[:, None] * INNER + ...)          # 像素主序
y = aclnn_linear_or_next_stage(o)                            # 下游还是 aclnn 语义
# ✅ attention 输出按参考块主序落盘（bh*SQ + s），下游边界与参考同布局
tl.store(o_ptr + (bh * SQ + ms)[:, None] * INNER + ...)     # 块主序 = 参考 to_out 输入
```

**判据**：中间量若**还回 torch/aclnn 计算**（或对齐 aclnn 的 golden 链），行序必须逐字
复刻参考；换成**自研 Triton GEMM（K 循环顺序累加）**后 M 布局不敏感——终态代码的
q/kv 投影即按**像素主序**每像素投影一次（实测 50/50 通过）。attention 输出 → out 投影
边界按参考**块主序**落盘是实测通过的稳妥路径。

### L1.2 ★ aclnn softmax 逐位配方（分 dtype）：sim 物化在输入 dtype；fp16 把 `(sim − m)` 舍回 fp16 再 exp；bf16/fp32 减法直接在 fp32；exp/sum/div 恒 fp32

fp16 case 穷举 8 种实现变体（内部全 fp32 / 全 fp16 / sum fp16 / exp 移位 / …），
**只有下述配方逐位一致**，其余全部差 4.88e-4（fp16 ulp 级）：

```python
sim = tl.dot(qs, kt, out_dtype=tl.float32)
sim = sim.to(tl.float16).to(tl.float32)          # 舍入点③：bmm 输出物化在输入 dtype
row_max = tl.max(sim, axis=1)[:, None]
d = sim - row_max
if IN_DT == 1:                                    # ⚠️ 仅 fp16 需要，bf16/fp32 禁做
    d = d.to(tl.float16).to(tl.float32)           # (x−max) 减法结果落 fp16 再升回
e = tl.exp(d)                                      # exp/sum/div 恒 fp32
p = (e / row_sum).to(elem_ty)                      # 舍入点④：softmax 输出落输入 dtype
```

⚠️ `(x−m)` 舍回输入 dtype 这步 **bf16/fp32 做了反而引入新偏差**（终态代码 15 个 bf16
case 实测通过为证）。`q * scale` 同理先升 fp32 乘再落回输入 dtype（scale=2⁻³ 为 2 的幂，
精确可折进投影 epilogue，FA 卡 X3 同源）。

### L1.3 ★ 运行时（非 constexpr）除数的标量 div/mod 多层解码大规模 miscompile；constexpr 除数的解码实测可用

```python
# ❌ 除数是运行时标量（host 传参）：Nblk=512 规模产生越界行号（qrow=4108 > 4095，非确定）
def _attn_kernel(..., nblk_per_b, wb, ...):        # 运行时参数
    hb = nblk // wb                                # miscompile；Nblk=16 规模却正确
# ✅ 除数 constexpr（编译期强度削减成乘法），终态代码 50/50 验证
def _attn_kernel(..., NH: tl.constexpr, nhbnwb: tl.constexpr, nwb: tl.constexpr, ...):
    bh = work // NH; head = work - bh * NH
    b = bh // nhbnwb; brem = bh - b * nhbnwb
    hb = brem // nwb;  wb_ = brem - hb * nwb
```

触发特征（op50 mini 系列实测）：**纯 store kernel 中同一解码正确，dot kernel 中才
miscompile**；错误率 ~75% 均匀分布、跨运行非确定；全 constexpr 化除数以外的部分
不解决（多层派生才是触发面）。**优先序**：constexpr 除数解码（终态验证）>
host 常量表（S3）> 运行时除数（禁用）。行号能用纯乘法 `nbk * SQ + offs_s` 最稳。

### L1.4 ★ dot 操作数必须 2 的幂——ceil16 仅适用向量 tile

```python
# ❌ ceil16(81) = 96 做 BSKV：bishengir 编译崩溃（fp32 ieee dot 同样崩）
BSKV = (SKV + 15) // 16 * 16
# ✅ dot 操作数形状 next_pow2 封顶 + mask；向量 tile（load/store/where/arange）仍可 ceil16
BSKV = _pow2(SKV, lo=16, hi=128)                   # 16/32/64/128/256
```

崩溃签名：`ConvertLinalgRToBinary encounters error: LLVM ERROR: Failed to obtain op
buffer shape size which should be static`。FA 卡 F3 的 ceil16 红利只能在**向量 tile**
上拿；同一 kernel 内"向量 ceil16 + dot 2 幂"两套规则并存是常态。

### L1.5 ★ 合并投影 kernel 的 weight-tile 转置路径在 BN>64 时静默错值

本后端 merged-projection（q|kv 拼单 GEMM，`[BN, BK] weight → tl.trans → tl.dot`）
在 **BN>64 时静默产出错值**；out-projection BN=128 不受影响。16 位 dense 取
`(BM, BN=64, BK=64)` 最快且可靠；BN=256 / BK=128 的 masked-B-tile padding 缺陷
同样要排除。**判据：扫参扫出"更快"的配置必须先过精度再采信——静默错值不会报错，
只会在 verify 里表现为 matched_ratio 断崖。**

---

## §3 Layer 2: 算法骨架

### §3.1 主骨架（前向五阶段，kernel 划分是参考不是教条）

数据流按"**连续化 → 投影 → 窗口取数 → attention → 回写**"五阶段组织；每个阶段是一个
独立 kernel，任务分解与关键约束如下。**阶段可合并/省略的判据见 §3.2**——同类算子的
shape/c/halo 不同，最优 kernel 数量在 4~7 之间浮动，照抄六个 kernel 没有意义，
照抄**阶段职责与任务分解**才有意义。

```
阶段 A（可选） 布局连续化    X[b,C,HW] --纯搬运--> XT[b,HW,C']
    · NCHW 的 channel 大步离散，token 化后 A 操作数行才连续（FA 卡 L1.6：禁 host permute）
    · C' = C 向上对齐（与权重同步 pad）→ 下游 GEMM 的 K 循环免 masked 尾块
    · 纯 Vector kernel：grid = min(VEC 核数, tiles) + 核内步长循环（FA 卡 L1.9）
    · tile 双向连续：load 沿像素维 stride 1、store 沿通道维 stride 1，tile 内 tl.trans

阶段 B          融合投影      P = XT @ [Wq | Wkv]ᵀ
    · q|kv 拼一张权重（预转置 [in,out]，FA 卡 L1.12），**每个像素只投影一次**，
      列段约定自定（如 [0,inner)=q / [inner,3inner)=kv）
    · 行序任意（自研 Triton GEMM 顺序 K 累加对 M 布局不敏感，L1.1）——按最方便的行序写
    · 任务 = (m_tile, n_tile)，解码用 constexpr 除数或查表（L1.3）
    · scale 为 2 的幂时可折进 epilogue（数值等价）；BN 档位受 L1.5 约束

阶段 C          窗口取数      KV[nblk, (blk+2h)², 2*inner]  <- 从 P 按固定几何窗口读
    · unfold 物化在此消除：窗口行号由 (b, hb, wb, ki, kj) 纯乘加推导
    · halo 越界行取 0 → k=v=0 → score=0 保留在 softmax 分母（Q2，勿当 padding 排除）
    · 纯 Vector kernel：grid = min(VEC 核数, NBLK_total)

阶段 D          块局部 attention   O = softmax(Q Kᵀ · scale) V
    · task = (b, nblk, head)；q/kv 行号 = 块基址 × 常数 + 块内偏移（纯乘法，L1.3）
    · S_kv ≤ 单块上限 → **单趟精确 softmax**（m/l 全列归一，免 online rescale）
    · S_kv 超上限 → 分块 online（FLASH 式）或两趟法（趟1 全局 m/l，趟2 归一 @V）
      ——两者数学等价但舍入路径不同，见 §6.2 注意项
    · softmax 按 L1.2 分 dtype 配方；QK/PV 两处 dot 输出均落输入 dtype（FA 卡 L1.16）

阶段 E          out 投影 + 回写    y = O @ Woutᵀ (+ bias) --> NCHW
    · A = O 按**参考 to_out 输入的行序**落盘（本类参考为块主序，L1.1 边界约束）
    · bias 加在 fp32 累加器再 cast（Q3：nn.Linear 默认 bias=True，逐字核对）
    · 块主序 → NCHW 的 fold 是独立 Vector kernel：2D grid (NBLK_total, c_tiles) 零解码

host
  · 权重复刻按 S1（含 bias 核对）；空输入直接 return（S2）
  · 核数 __init__ 时读 npu_config（mha 卡 M1 同源）；Vector 类阶段与 Cube 类阶段
    的核数上限不同（910B2C 参考：vector=40 / cube=20），grid 分别取各自的 min
  · 全部 dot 操作数 2 幂（L1.4）；16 位融合投影的 BN 档受 L1.5 限制
```

### §3.2 阶段合并/省略判据（Phase 2 草图期逐条回答）

| 判据 | 取 A | 取 B |
|---|---|---|
| **A 阶段是否需要** | x 的 C 维大步离散且 C 不对齐 → 做（投影是算力大头，A 操作数连续 + K 免尾块值得一次全量搬移） | C 已对齐/极小（c=8 级）→ 省，投影 A 直接 strided 读 |
| **B/C 合并还是分离** | 窗口 gather 直接进 B 的 A 操作数（省 P 物化，但 halo 区像素**重复投影 R 倍**，§3.3） | 分离：先每像素投影一次再 gather（消 R，多一份 P 中间量与一次 launch）——**本类默认** |
| **D 的单块/分块** | S_kv ≤ 256（恒成立域）→ 单趟精确 softmax | S_kv 超单块上限 → 分块 online 或两趟法 |
| **E 的 fold 融合** | 投影 store 直接按 NCHW 地址 scatter（省一次全量搬移，但 store 沿通道维大步、短连续段） | 独立 fold kernel（store 连续性好）——**本类默认**，何者更快随 shape 变，Phase 4 实测定 |

### §3.3 ★★★ 投影侧 vs attention 侧——收益判据（先算账再动手）

设读放大 `R = (1+2·halo/blk)²`，投影 FLOPs 占比 `φ = F_proj / (F_proj + F_attn)`：

- **判据 A（本类常态）**：`φ > 80%`（c 大、head_dim=64 固定时几乎恒成立；
  op50 的 c=2048 case：投影 ~866 GFLOP vs attention ~68 GFLOP，φ≈93%）⇒
  **主攻投影侧**（每像素单次 dense 投影消读放大 R、权重预转置、tile 档位），
  attention 侧只在 SKV 贴 UB 上限时扫一档。
- **判据 B**：`φ < 50%`（c 极小如 c=8 的 case）⇒ attention/gather 侧主导，
  主攻 K3 gather 与 K4 的 tile。

**每像素单次投影 vs 按窗口重复投影**：窗口直接进 GEMM A 操作数会把 halo 重叠区的
像素**重复投影 R 倍**（block=8/halo=4 时 4 倍投影量）；先每像素投影一次再 gather（K2+K3）
把重复消到 1——这是本类与"窗口 gather GEMM"写法的分水岭（§5.2 证伪表末行）。

---

## §4 Layer 3: 关键技巧

### §4.1 softmax 舍入配方（照抄 L1.2 的代码，逐位对齐 aclnn）

配方已在 L1.2 给出。补充两点：① `RECIP` 开关——终态代码保留 `p = e / row_sum` 与
`p = e * (1.0/row_sum)` 双路径（后者是 IR 级 vdiv→乘法优化点，数值等价性依赖
参考对 scale 的写法，先按除法落地）；② 全 fp32 内部或全 fp16 内部都**差 4.88e-4**
（8 变体穷举），不要试图"更准"——目标是**对齐 aclnn**，不是算得更准。

### §4.2 constexpr 除数解码 + 纯乘法行号（L1.3 配方的完整形态）

见 L1.2/L1.3 代码块。终态的解码链全部落在 constexpr 除数上
（`NH / nhbnwb / nwb / BLK`），q 行基址 `q_row0 + i1*W + j1` 纯乘加——
既过解码安全，又天然给出 L1.1 要的块主序。

### §4.3 C-pad 64 倍数 + 权重同步 pad

XT 通道维补零到 64 倍数、投影权重同步 pad → dense matmul 的 K 循环**无 masked
尾块**：累加精确（pad 列乘 0 恒 0）且省掉 mask 向量开销。归因第 5 位，零风险顺手做。

### §4.4 K 转置直读（继承 mha 卡 §4.1）

`kt` 直接按 `[HD, BSKV]` 转置态 load（d 方向 stride 1、列 stride=行宽），免 `tl.trans`。
op50 终态 K4 走的是"正常 load [BSKV, HD] + `tl.trans(k)`"（同样验证可行）——
两种写法都过，转置直载省一次 UB 内重排。

### §4.5 VEC/CUBE 双核数分流

转置（K1/K6）与 gather（K3）是纯 Vector kernel，grid 上限用 `vector_core_num`（40）；
GEMM（K2/K5）与 attention（K4）用 `cube_core_num`（20）。混用同一上限会让
Vector kernel 少拿一倍核。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 舍入点逐位复刻（L1.1 边界布局 + L1.2 softmax 域 + bias 修复） | 96-99% mismatch → **50/50**，缺一即全错 | 精度闸门，先于一切性能优化 |
| 2 | ★★★ 每像素单次 dense 投影 + kv gather（消读放大 R） | 消 R 倍投影量（block8/halo4 时 4×） | 参考按窗口重复投影时 |
| 3 | ★★ 解码安全化（constexpr 除数 / host 表） | 非确定错值/NaN → 确定正确 | 首版必须 |
| 4 | ★★ dot tile 2 幂化（L1.4）+ 16 位 BN=64 档（L1.5） | fp32 编译崩溃/静默错值 → 通过 | 首版必须 |
| 5 | ★ C-pad 64 倍数 + 权重同步 pad | K 循环免 mask 精确 | dense 投影（K2/K5） |
| 6 | ★ K 转置直读 / RECIP 倒数乘 | 零风险小量 | 总是 |
| 7 | 布局变换全部 kernel 化（K1/K6），host 零 permute | 消 host 搬运 kernel | FA 卡 L1.6 落点 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱**）

| 方向 | 结果 |
|---|---|
| **attention 输出像素主序落盘 + 下游对齐 aclnn golden** | ⛔ 98.7% 元素 1 ulp 不等、mismatch 96.3%（L1.1）。自研 Triton GEMM 内部换布局**不在此列**（终态 q/kv 投影像素主序通过） |
| **softmax 内部全 fp32 或全 fp16** | ⛔ 8 变体穷举全差 4.88e-4，仅 L1.2 配方逐位；`(x−m)` cast 到 bf16/fp32 同样引入新偏差 |
| **运行时（非 constexpr）除数的 kernel 内 div/mod 解码链** | ⛔ Nblk=512 规模越界行号（qrow=4108>4095）、~75% 均匀错、非确定；Nblk=16 却正确（规模相关，最险）。constexpr 除数不在死路之列（终态在用） |
| **用"无 dot kernel"生成行号表再查表** | ⛔ 生成 kernel 自身在大规模踩解码 miscompile，表即污染（op50 mini16 实锤）；且多一张表一次 launch。不如 constexpr 除数 |
| **非 2 幂 dot tile（ceil16 进 dot 操作数）** | ⛔ `ConvertLinalgRToBinary ... should be static` 编译崩溃，fp32 ieee dot 同样（L1.4） |
| **merged 投影 weight-tile 转置 BN>64 / BN=256 / BK=128** | ⛔ 静默错值或 masked-B-tile padding 缺陷（L1.5）；扫参"更快"的档必须先过精度 |
| **按窗口重复投影（窗口直接进 GEMM A 操作数）** | ⛔ halo 重叠区读放大 R=(1+2h/blk)² 倍投影量；先每像素投影再 gather 严格更省 |
| **clamp+裸 load 替代 masked load（K2/K5）** | ⛔ 本后端**非确定变慢**（终态代码注释实测）；masked load 保留 |

### §5.3 天花板估算：先算账再动手

1. **读放大** `R = (1+2·halo/blk)²`——参考 unfold 物化的额外流量倍数；
2. **投影占比** `φ = F_proj/(F_proj+F_attn)`——c≥512 时普遍 >85%，attention 侧优化
   天花板 `(1-φ)` 封顶（op50 全类 geomean 2.1x 中 attention 侧贡献 <15%）；
3. **单 case 噪声带 ±30%、跨会话 ±2~5%**（mha 卡 §3.3/§7.4）——geomean 2.0x 的
   达标判定必须**双跑**（op50 两跑 2.0973/2.1243）。

---

## §6 精度闸门

### §6.1 判定顺序（错一步会把 bug 归错类）

1. **先看覆盖率**：输出是否被 kernel 全覆盖？`torch.empty` + 部分覆盖 = 未初始化内存
   （NaN 位置断言 + 精度判定同时触发，像"精度不够"实为覆盖不全——block_sparse 卡 L1.5
   同款陷阱，本类 K6 scatter 漏 tile 时同样表现）；
2. **再看是不是解码 miscompile**：~75% 均匀错 + 非确定 + 纯 store kernel 正常
   ⇒ L1.3，不要去调精度；
3. **最后才是舍入契约**：此时按 §6.2 的六舍入点逐处核对（本类 96-99% 1-2 ulp
   全错的三大来源：边界布局 / softmax 域 / 漏 bias）。

### §6.2 舍入点契约（golden 在输入 dtype 上逐步落盘，六处一个不能少）

| # | 舍入点 | 对应参考算子 | 落点 |
|---|--------|--------------|------|
| ① | q 投影输出（含 ×scale=2⁻³，精确可折 epilogue） | `to_q(q_inp) * scale` | 投影后 cast 输入 dtype |
| ② | kv 投影输出 | `to_kv(kv_inp)` | 同上 |
| ③ | QK bmm 输出 | `torch.bmm` | fp32 acc → cast 输入 dtype → 升回 fp32 计算 |
| ④ | softmax 输出 | `F.softmax` | L1.2 分 dtype 配方 |
| ⑤ | PV bmm 输出 | `torch.bmm` | fp32 acc → cast 输入 dtype |
| ⑥ | out 投影输出（**含 bias**，Q3） | `to_out` | bias 加 fp32 累加器再 cast |

fp32 case（本类 10/50）：全程 fp32 ieee dot，任何中间量不落低精度；cast 分支
fp16/bf16/fp32 三档写全、默认兜底 fp32（FA 卡 L1.16 镜像条款）。

**dot vs aclnn 的固有累加差是可容忍的**：sim 端 aclnn vs fp32 累加仅 0.04% 元素不等
（maxd 0.0078），传播到输出 maxd 0.0039 / mismatch 0.035%——远低于阈值。
**真正的杀手是舍入点错位**，不是累加顺序。

### §6.3 与 FA 卡既有条目的冲突判据（3 条，搬运结论前先核对）

| # | 冲突 | 本类实测 | **判据** |
|---|------|----------|----------|
| ① | FA 卡 F3 `BLOCK_D=ceil16` vs 本卡 L1.4 dot 2 幂 | ceil16(81)=96 做 dot 操作数直接编译崩溃 | **ceil16 只在向量 tile 上拿红利；进 `tl.dot` 操作数形状必须 2 的幂**。同一 kernel 两套规则并存 |
| ② | FA 卡 L1.5「p 不能降到输入 dtype」vs 本卡 L1.2（p 必须落输入 dtype） | 本类参考 `F.softmax` 在原生 dtype 上 IO，p 落 fp16 是 golden 的一部分，50/50 通过 | 看 FA 卡 L1.4 先钉死契约：**参考的 softmax 在什么 dtype 上 IO 决定 p cast 方向**——原生 dtype IO ⇒ 必须 cast（本类）；fp32 上算 ⇒ 禁止 cast（L1.5 适用域） |
| ③ | FA 卡 F1「归约维 padding 显式排除」vs 本类 halo padding **保留**在分母 | 展开型参考普遍未实现 padding 掩码：padding 列 k=v=0 → score=0 真实参与 softmax 分母 | **以参考语义为准**：逐字核对参考有没有掩 padding。两种"padding"语义相反——halo 像素越界列保留；tile 补齐列（s ≥ S_kv）仍须 F1 排除 |

### §6.4 定位手法（"结构对但 1-2 ulp 全错"时的四步）

1. **逐字 replay 参考 forward**（host torch 逐行复制）确认语义理解 = 0 差；
   host 重算若与参考有差，**先修 debug 脚本再怀疑 kernel**（op50 两次 host 脚本自身
   bug 各浪费一轮：bmm reshape 头序弄反、SKV 少写平方）。
2. **逐层中间量隔离**（xt → q_buf/kv_buf → attn_out → out_tok），找第一个 maxd
   跳变层；权重单验先行（S1）。
3. **aclnn 行为探针**：把可疑算子（softmax/bmm）单独拿出来穷举实现变体，找逐位一致
   的那条路径（op50 softmax 8 变体命中 L1.2）。
4. **mini 二分系列**：单 task 固定解码 → 独立地址并发 → 加 div/mod → 加查表，
   每步只加一个变量；"纯 store kernel 正确 + dot kernel 错"即锁定 L1.3。

---

## §7 测量口径（不做这一步，上面所有数字都是噪声）

- **两次独立 benchmark 才能判达标**：op50 两跑 2.0973 / 2.1243（repeats=20）；
  target 2.0 达标但余量 ~5% < 跨会话噪声地板（~4.6%，mha 卡 §7.4）。
- **b=0 空输入 case**：benchmark `PROFILER_COLLECT_FAIL` 属预期（S2——forward 直接
  返回无 kernel 可采集），计入 total 不计入 geomean；该 case 精度验证正常通过。
  perf 的 `passed_cases`（49/50）是"采集成功数"，与精度语义不同——
  **精度只看 verify_result.json 的 passed/total（50/50）**。
- **隐藏状态探针警告不阻断**：`HIDDEN_STATE_VALUE_MISMATCH`（权重以预转置布局存放，
  探针按原始 `[out,in]` 比对找不到逐位相等项）——最终精度判定为准，exit 0 即通过。
- 分 dtype 几何平均：fp16 2.02 / bf16 2.34 / **fp32 1.94**——fp32 是本类薄弱档
  （ieee dot 吞吐天花板，mha 卡 §7.2 同结论），报告需分档呈现。

---

## §8 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| NaN 237929/2297856（~10%）+ 后续 case 级联 `507015 aicore exception` | 运行时除数解码 miscompile 越界行号；级联是设备污染（先单跑第一个失败 case 定位） | L1.3 |
| ~96-99% 元素差 1-2 ulp、值域正常、maxd ~0.045 | 舍入契约错位三连：边界布局 / softmax 域 / 漏 bias | §6.1 步 3 + §6.2 |
| fp32 case `ConvertLinalgRToBinary ... should be static` 编译崩溃 | 非 2 幂 dot tile | L1.4 |
| 单 case 输出恒 0 或半张图未写 | 空输入分支未复刻 / K6 scatter 漏 tile | S2 + 覆盖率检查（§6.1 步 1） |
| 扫参后更快但 matched_ratio 断崖 | BN>64 weight-tile 转置静默错值 | L1.5 |
| verify 全过但 speedup 分子异常好 | 编译失败 case 被排除在 geomean 外 / host 侧搬运 | 先看 `passed_cases == total_cases`（block_sparse 卡 §8 同款） |
| host 重算链与参考输出不等 | debug 脚本自身 reshape 序错 | §6.4 步 1，先修脚本 |
| `q*scale` 折进投影 epilogue 后 fp16 边界 case 失败 | scale 非 2 的幂（本条仅当 dim_head≠64²ⁿ 时） | 非 2 幂 scale 改回独立乘法并落输入 dtype |

---

## §9 与其它模板的分工

| 文件 | 何时用 |
|---|---|
| `flash_attention.md` | 稠密 FA 主链（KV 全扫 + online softmax）。**本类算子也要读**，其 L1.1/L1.9/L1.11/L1.12/L1.13/L1.16 依然成立 |
| `mha.md` | **含 `nn.Linear` 投影段时追加读**（交叉行）：M1~M4 核数/grid/UB 剪枝/dot 链 ≤2、§4.1 投影 GEMM 终态架构与 tile 档 |
| `block_sparse_attention.md` | 掩码/选择表类（行 1/1b/8）。与本文档同属"三 稀疏类"但机制不同：它是**块级选择**，本类是**固定几何邻域展开**（无选择表） |
| `attention_index.md` | attention 家族定 `category` 的唯一入口。本文件对应其**行 9**（三·3 邻域展开 / Sparse-展开型） |
| **本文件** | §0.1 判别特征命中即用（`F.unfold`/im2col/halo 邻域展开构造 KV） |

冲突时以本文件为准（本文件结论均在展开形态上实测）。

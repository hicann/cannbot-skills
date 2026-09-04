---
name: position_channel_attention
description: CV-Attn-空间自注意力·无投影位置/通道双分支类算子（特征图 flatten token 化的自注意力，无 nn.Linear Q/K/V 投影——Q=K=V 同一 Y，每分支前置分辨率保持的 k×k conv，position/channel 对偶双分支残差相加，DAModule/DANet 系）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束（双分支统一物理布局/PV 二段拆分/conv implicit GEMM/lazy 权重 RNG 复刻/双路径分派）、Layer 2 算法骨架、Layer 3 关键技巧与 Phase 4 优化点清单（含证伪方向全表）
metadata:
  type: reference
---

# CV-Attn-空间自注意力（无投影位置/通道双分支）算子优化经验

本文档是 **"特征图 token 化的空间 self-attention：无 Q/K/V 投影（Q=K=V 同一 Y）、
每分支前置分辨率保持的 k×k conv、position/channel 对偶双分支残差融合"** 这一类算子
（`attention_index.md` 行 10c，一·3 空间 token 版 · 无投影双分支变体）的经验合集，
覆盖 Phase 2/3/4。

- **§0 适用范围与算子分类**（子类标签 + 判别特征 + 形态识别五问）
- **§1 通用经验**（S1~S3：单 kernel 参数化双分支 / 架构即终态判定 / 极端形状画像）
- **§2 Layer 1 设计约束 L1.1~L1.8**（Phase 2 硬性边界，precheck 整节摘录）
- **§3 Layer 2 算法骨架**（4 kernel + host 门控双路径分派）
- **§4 Layer 3 关键技巧**（ADD_OUT 融合 / p 二段拆分 / conv implicit GEMM 配方 / Q=K=V 共享 load）
- **§5 Phase 4 优化点清单**（收益排序 + **证伪方向全表** + 遗留瓶颈画像）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与其它模板的分工**

> ⚠️ **本文件与 `flash_attention.md` 的分工**：
> 本类两个分支都是完整的 `softmax(Y·Yᵀ)·Y` 三段式，FA 卡 attention 主链的 Layer 1
> **依然全部成立**（L1.1 dot 契约、L1.3 rolling rescale、L1.5 p 禁降、L1.6 host 禁布局搬运、
> L1.8/L1.9 grid、L1.13 面积预算、L1.14 propagate_nan 分档、F1/F2 通用条目，清单见 §2 末尾）。
> 本文件**只补充"无投影 + 双分支对偶 + conv 前置"带来的那部分**：双分支统一物理布局、
> PV 升精度异构 dot 禁令与二段拆分、conv implicit GEMM + lazy 权重 RNG 复刻、
> 按 (T,D) 的双路径分派。**两份都要读**；冲突时以本文件为准（本文件结论均在
> 无投影双分支形态上实测过）。
>
> **证据基础**：DAModule（DANet 双重注意力模块：position attention + channel attention
> 双分支并行）50 case 完整轨迹——B∈[1,4]、C∈[8,1024]、H·W∈[16,5041]、fp16 为主 +
> 少量 fp32/bf16，910B2（24 CubeCore）。Phase 3 一轮通过（轮内修 3 个 A 类：PV 异构 dot /
> UB 面积 / 双分支布局），首版即 geomean **2.4313**、精度 **50/50**、target 2.0 达标；
> Phase 4 三轮优化（tile 分档 / conv tile 收缩 / launch 常量 constexpr 化）**全部实测无提升**，
> IR 无新建议，终局以 Phase 3 基线收尾；同码跨日复测 50/50，偏差 −1.1% 在波动带内。
>
> ⚠️ **核心优化哲学**：这类算子**没有投影段 GEMM 可吃**（mha/spatial_attention 的主战场
> 不存在），收益全部在**结构**上：① 双分支共用一套参数化 kernel（stride/constexpr 区分
> 布局与语义），channel 分支 kernel 内 ADD_OUT 融合残差相加——省一路 kernel + 一遍 out
> 显存流量；② 按 (T,D) 双路径分派——D≤256 走 online softmax 单 kernel，D>256（此时
> T≤128）走 P 物化两阶段，绕开 acc[BQ, D>256] 的 UB 墙；③ conv 前置外提为 implicit-GEMM
> kernel + 权重预重排。**tile/launch 级微调在本类已全部实测证伪（§5.2），
> Phase 3 架构即终态。**

---

## §0 适用范围与算子分类

形态由**入参与写法**识别，不按算子名匹配。

| 子类标签 | 入参/写法特征（可模式匹配） | 优化重心 |
|---|---|---|
| `pc-dual` | 全形态：conv 前置 + position/channel 对偶双分支 + `out = out_pos + out_ch` 残差相加（DAModule/DANet 系写法） | 本卡全章节 |
| `pc-single` | 单分支无投影自注意力 + conv 前置（position 或 channel 之一） | 双分支相关条目（L1.1/§4.1）退化为单路直 store，其余照用 |

### §0.1 判别特征（决定用不用本文件）

满足**任意一条**即用本文件：

1. 参考实现里 `S = torch.bmm(y, y.transpose(1, 2))` 且 `y` 直接来自 conv/特征图，
   **无 `nn.Linear` Q/K/V 投影**——Q=K=V 是同一张量；
2. 同一份特征图按 `(B,N,C)` 与 `(B,C,N)` **两种 token 化各做一次三段式**（position/
   channel 对偶双分支，N=H·W），输出相加；
3. 分支前置**分辨率保持**的 k×k conv（`nn.Conv2d(C, C, k, padding=k//2)`）代替投影段；
4. 分支内 `.float()` 全 fp32 bmm+softmax，分支末尾 `.to(x.dtype)`，两分支在输入 dtype 上相加；
5. conv 权重在 forward 内 lazy 创建（`torch.manual_seed(hash(...))` + `self._cache` dict，
   cache key 由形状/kernel_size/device/dtype 构成）。

### §0.2 ★ 形态识别五问（**Phase 2 第一步必须回答**）

| # | 问题 | 影响 |
|---|---|---|
| **Q1** | 无投影（Q=K=V）还是带 4 个 `nn.Linear` 投影？ | 无投影走本卡；带投影走行 6a（`mha.md`）或行 10a（`spatial_attention.md`）——**最大收益杠杆完全不同**（投影段 vs 结构融合） |
| **Q2** | 单分支还是 position/channel 双分支？ | 决定 L1.1 布局统一与 §4.1 ADD_OUT 融合是否生效；单分支退化为 `pc-single` |
| **Q3** | 每分支 (T, D) 的量级与组合？ | 决定双路径分派（§3.2）：`D≤256 or T>128` → 路径 A online softmax；否则 → 路径 B P 物化。本类实测两个方向都会出现：position 分支 T=N≤5041、D=C≤1024；channel 分支 T=C≤1024、D=N≤5041——**同一 case 两分支可走不同路径，判据按分支独立** |
| **Q4** | golden 的精度契约分段？ | 本类 golden 分支内 `.float()` 全 fp32 ⇒ attention 段全程 fp32；conv 在原生 dtype ⇒ conv 段原生 dtype + 输出舍回；两分支 `.to(dtype)` 后相加 ⇒ ADD_OUT 在 dtype_in 上做（L1.7）。逐段契约，禁止一刀切 |
| **Q5** | conv 权重怎么来的？ | forward 内 lazy 创建（seed+hash+cache）⇒ L1.4 RNG 逐字复刻；init 固定权重 ⇒ 常规常量处理 |

---

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 双分支是同一骨架的两次参数化调用，不是两个 kernel

position 分支 (T=N, D=C) 与 channel 分支 (T=C, D=N) 只差 token/特征的含义。conv kernel
用 `TRANS_OUT`、attention kernel 用 `TOK_STRIDE/D_STRIDE + scale + ADD_OUT` 参数化，
**一套 kernel 吃两个分支**。好处不止少写代码：tile 扫描、精度配方、布局修复只维护一份，
Phase 4 调参不会改了一边忘另一边。

### S2 "Phase 3 架构即终态"的判定信号

本类 Phase 4 三种常规微调（tile 面积分档 / conv tile 收缩 / launch 常量 constexpr 化）
全部实测无提升（§5.2），IR 分析判定 conv kernel 的同步开销为 MIX kernel 结构性固有
（`ir_has_more_suggestions=false`）。若 Phase 3 已达 target 且优化器报告耗尽，
**不要硬凑优化轮次**——按退出前置门 (b) 终局即可，报告里写清证伪记录。

### S3 极端形状画像先于调参

最慢 case 集中在两端：**大 N 小 C**（position 分支 KV 迭代数多、D=16 时 dot K 维
计算强度低，tile 已达面积上限）与**大 C 小 N**（conv BN=128 块数塌陷 < 核数，
收缩已双向证伪）。这两端属结构固有画像，写报告时直接标注（§5.3），**不要反复扫参**。

---

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ 双分支输出物理布局必须统一为 `(b,c,n)`，position 分支转置 store——本类头号 bug

坏味道：position 分支按 `(B,N,C)` 物理写、channel 分支按 `(B,C,N)` 物理写，再靠 host
侧 view/permute 对齐。实测后果：残差相加读错位 + 最终 `view(b,c,h,w)` 错位，
错误形态伪装成精度问题（实测首轮全挂中的布局类失败）。

修法：out 统一按 `(b, c, n)` 分配，stride 参数化吃两种分支布局——

```python
# position 分支（T=n tokens, D=c features）: TOK_STRIDE=1, D_STRIDE=T   # 转置 store
# channel  分支（T=c tokens, D=n features）: TOK_STRIDE=D, D_STRIDE=1   # 直 store
base = bi * CN + offs_q[:, None] * TOK_STRIDE + offs_d[None, :] * D_STRIDE
tl.store(out_ptr + base, ov, mask=msk)
# 残差融合（ADD_OUT）时两分支读到的就是同一批地址
```

**禁止** host 侧 `permute/contiguous` 补救（FA L1.6：布局全部由 kernel store 方向承担）。

### L1.2 ★★★ PV 的 dot 禁止走升精度异构路径；非 fp32 输入必须 p_hi+p_lo 二段拆分

p 是真 fp32 量。`tl.dot(p, v.to(tl.float32))`（p fp32 × v 升 fp32）在融合 kernel 内
实测触发**设备异常/NaN**（首轮全挂，AccuracyError 与 NaN 断言混合出现）。

```python
# ❌ 升精度异构 dot：设备异常 / NaN（实测大面积中招）
acc = acc * alpha[:, None] + tl.dot(p, kt.to(tl.float32))
# ✅ 非 fp32 输入：p 拆成原生 dtype 的 hi+lo 两项，两次原生 dtype dot（fp32 累加器）
p_hi = p.to(kt.dtype)
p_lo = (p - p_hi.to(tl.float32)).to(kt.dtype)
acc = acc * alpha[:, None] + tl.dot(p_hi, kt).to(tl.float32) + tl.dot(p_lo, kt).to(tl.float32)
# ✅ fp32 输入：参考本身就是 fp32 算，原生 fp32 dot 保持，不拆
```

p∈[0,1]，hi+lo 恢复 ~22（fp16）/ ~16（bf16）位尾数。拆分同时消掉 fp32 v 副本的 UB 占用
（L1.5 的 ub overflow 有一半是它贡献的）。用 `ELEM_IS_F32: tl.constexpr` host 门控分档。

### L1.3 ★★ conv 前置必须外提为独立 implicit-GEMM kernel，权重预重排 w9[k²,ci,co]

FA L1.15 同款（输入变换必须在 attention 主循环之外）：禁止把 conv 融进 attention 的
KV 循环逐块重算；禁止 host 侧 `F.conv2d`（PyTorch 退化）。

- kernel 内 k² 次移位 `tl.dot` + halo masked load（§4.3 配方），全部 Cube，无 im2col 物化；
- 权重 `[co,ci,k,k]` 在 host 预重排为 `w9[k*k, ci, co]` 连续
  （`w.permute(2,3,1,0).reshape(kk,c,c).contiguous()`，数值逐位不变），bias 原样——
  FA L1.12 的 conv 版：kernel 侧 w_tile 按 `[BC,BN]` 连续 load；
- 输出逐分支布局：position 用 `TRANS_OUT=true` 转置 store 成 `(B,N,C)`，channel 直 store
  `(B,C,N)`。

### L1.4 ★★ lazy 创建的 conv 权重必须 RNG 逐字复刻

参考在 forward 里现场建层（manual_seed + cache dict），复刻链**缺一即全错**
（FA L1.11 的 conv 适配）：

```python
torch.manual_seed(42)                        # 逐字复刻参考 forward 开头的 seed 时序
key = <参考的 cache key 构造>                  # 通常由 (channel, kernel_size, device, dtype) 构成
if key not in cache:
    rng_state = torch.get_rng_state()
    torch.manual_seed(hash(key) & 0xFFFFFFFF)
    bound = 1.0 / math.sqrt(channel * kk)    # kaiming_uniform_(a=sqrt(5)) ≡ uniform_(±1/√fan_in)
    w_pos = torch.empty((c,c,k,k), dtype=torch.float32).uniform_(-bound, bound)  # CPU fp32
    b_pos = torch.empty((c,),   dtype=torch.float32).uniform_(-bound, bound)
    w_ch  = torch.empty((c,c,k,k), dtype=torch.float32).uniform_(-bound, bound)  # 顺序与参考一致
    b_ch  = torch.empty((c,),   dtype=torch.float32).uniform_(-bound, bound)
    torch.set_rng_state(rng_state)
    # 预重排 + .to(device, dtype) 之后入 cache
```

四个对齁点：seed 时序、CPU fp32 采样、构造顺序与次数（w/b 交替、分支顺序与参考一致）、
rng save/restore。**诊断顺序**：`passed_cases == 0` 或大面积不匹配时，先单独验权重，
再验 conv 输出，最后才查 attention kernel。

### L1.5 ★★ tile 预算按本类实测收紧：面积 ≤ 8192 + UB ≤ 96KB（esz-aware 双向收缩）

FA 卡的 16384 面积 / 150K UB 是**无残差融合**的预算，本类（kernel 内 ADD_OUT）直接照搬
会编译失败——实测首轮近半数 case `MLIRCompilationError`（`ub overflow` /
`hivm-plan-memory` 失败）：

```python
_S_AREA = 8192          # BQ*BKV 上限：s tile ≤32KB fp32（[128,128]+ADD_OUT 组合实测 PlanMemory 失败）
_UB_BUDGET = 96 * 1024  # esz-aware 公式双向收缩：
# BQ*BD*esz + BKV*BD*esz + BQ*BKV*4 + BQ*BD*4 ≤ 96K（先收 BQ 到 16，再收 BKV 到 16）
BQ = max(16, min(BQ, _ceil16(T)))   # 小 shape 按实际收缩（FA L1.13 的另一面）
```

结构性改动之后必须重扫 tile（FA F5）；UB overflow 会污染设备导致级联假失败，
大批连续失败时先定位第一个失败用例单独复跑（FA F4）。

### L1.6 ★★ attention 双路径分派（host 门控，两路径都必须实现）

```
if D <= 256 or T > 128:   # 路径 A：online softmax 单 kernel，S 不物化
    attn_online[grid](...)
else:                     # 路径 B：P 物化两阶段（T×T ≤ 128² fp32 ≤ 64KB 可物化）
    attn_smax[grid](...)  # 阶段 a：S + 单趟 softmax → p_buf[B,T,T] fp32
    attn_pv[grid](...)    # 阶段 b：PV GEMM 沿 D 分块补并行度 + 残差融合
```

判据出处见 §3.2。⚠️ 按分支独立判定：同一 case 的 position/channel 分支可走不同路径。
3.2b 架构符合性核对：4 个 kernel（conv / attn_online / attn_smax / attn_pv）齐全 +
门控两分支都接通，缺一即 A-SketchDeviation。

### L1.7 ★ 精度契约分段执行，禁止一刀切

| 段 | 契约 | 依据 |
|---|---|---|
| conv 段 | 原生 dtype 计算（原生操作数 + fp32 累加），输出**舍回 dtype_in** | golden 的 conv 在输入 dtype 上算 |
| attention 段 | **全程 fp32**（scores/p/acc/l），禁止中间降精度 | golden 分支内 `.float()` 后 bmm+softmax |
| 输出相加 | 各分支 `.to(dtype_in)` 后**在 dtype_in 上加**（kernel 内 ADD_OUT 融合） | golden `out_pos + out_ch` 是 dtype add |

两个方向都禁：attention 段降精度（p 直降 fp16 顶穿 9.77e-4 阈值，FA L1.5）；
conv 段/输出相加全程 fp32（"比参考更准"反而不达标，matched_ratio 稳定 0.76，FA L1.16）。

### L1.8 grid 与索引通用约束（继承 FA 卡，本类实测同样命中）

- `grid = (min(NUM_CORES, tasks),)` + 核内步长循环，NUM_CORES 常量 constexpr（910B2=24），
  适用于每个 kernel（FA L1.9）；
- `kv_lo` 从 0 开始不做 BLOCK_KV 对齐（FA L1.8）；
- mask 属性 constexpr（本类无 mask，天然满足，FA L1.7）；
- host 预计算 `scale`（position 分支 `C**-0.5` / channel 分支 `N**-0.5`）fp32 标量传入，
  kernel 内零开销。

**依然成立、直接引用 FA 卡不重复展开的约束**：L1.1 dot 契约（原生 dtype 操作数 +
fp32 累加器）、L1.2 两操作数同 dtype、L1.3 online softmax 滚动 rescale 三件套
（m 用 -1e30 有限极小值初始化）、L1.5 p 禁降输入 dtype、L1.6 host 禁 permute/contiguous/pad、
L1.13 面积预算取上限 + 按 shape 收缩、L1.14 `propagate_nan=ALL` 按 MULTI_BLK 分档
（多块开、单块关）、F1 KV padding 列显式 `where` 排除（ghost 列不进 softmax）、
F2 掩码用有限极小值。

---

## §3 Layer 2: 算法骨架

### §3.1 主骨架（4 kernel + host 门控）

```
host_prepare():                              # cache miss 时一次，warmup 吸收
    RNG 逐字复刻两分支 conv 权重（L1.4）
    预重排 w9[k²,ci,co] 连续（L1.3）；cache[key] = (w9_pos, b_pos, w9_ch, b_ch)

kernel 1  conv3x3（implicit GEMM，TRANS_OUT: constexpr）:
    acc[BM(n), BN(co)] fp32
    for ci_block in range(0, C, BC):
        for k in range(k²):                  # k² 次移位 dot，全部 Cube
            halo = (0≤p+dh<H) & (0≤q+dw<W)   # padding=k//2 零填充语义
            xt = load(x[ci_block, n+dh·W+dw], mask=ci_ok & halo, other=0)   # [BC,BM]
            wt = load(w9[k, ci_block, co_block])                            # [BC,BN] 连续
            acc += dot(trans(xt), wt)
    y = cast(acc + bias, dtype_in)           # conv 段舍回（L1.7）
    TRANS_OUT=true  → store (B,N,C)；false → store (B,C,N)
    两次 launch：position 分支(true) → y_pos；channel 分支(false) → y_ch

kernel 2  路径 A online softmax（D≤256 or T>128）:
    q_tile[BQ,BD] 常驻；m=-1e30, l=0, acc=0
    for kv in range(0, T, BKV):              # kv_lo=0 不对齐（L1.8）
        s = dot(q, trans(k)) * scale         # 原生操作数 + fp32 累加
        s = where(kv_ok, s, -1e30)           # F1：ghost 列显式排除
        m_new = maximum(m, rowmax(s), propagate_nan=ALL if MULTI_BLK)
        p = exp(s - m_new)；alpha = exp(m - m_new)
        acc = acc*alpha + （p 二段拆分 dot | fp32 原生 dot）（L1.2）
        l = l*alpha + sum(p)；m = m_new
    o = cast(acc / l, dtype_in)
    if ADD_OUT: o += load(out 同位置)        # dtype_in add（L1.7/§4.1）
    统一 (b,c,n) stride store（L1.1）

kernel 3+4  路径 B P 物化两阶段（D>256 且 T≤128）:
    3a attn_smax:  s = Σ_d dot(a[:,d], trans(a[:,d]))   # Q=K=V 共享同一 load（§4.4）
                   单趟 softmax（rowmax→exp→sum→div，无 online 状态）→ p_buf[B,T,T] fp32
    3b attn_pv:    o = dot(p, v)（p 二段拆分，L1.2）沿 D 分块并行补并行度
                   → cast dtype_in → ADD_OUT → store

host 门控（每分支独立判定，L1.6）:
    for (y, T, D, scale, trans, add) in [(y_pos, N, C, C**-0.5, true,  false),
                                          (y_ch,  C, N, N**-0.5, false, true )]:
        D<=256 or T>128 → 路径 A 一次 launch；否则 → 路径 B 两次 launch
    return out.view(b, c, h, w)
```

### §3.2 双路径判据的出处

路径 A 的 UB 瓶颈在 `acc[BQ, BD]`：D>256 时 96KB 预算装不下有意义的 BQ。但实测
shape 空间里 **T·D 此消彼长**——D>256 只在 T≤128 时出现（position 分支 D=C 大则 N 小；
channel 分支 D=N 大则 C 小），此时 `T×T ≤ 128²` 的 P 可物化（fp32 ≤64KB），把 online
softmax 换成"S 全量 + 单趟 softmax + PV GEMM"：S 内积维分块累加（D 可达数千），
阶段 b 再沿 D 分块补并行度。**先查 (T,D) 组合再定路径，禁止对 D>256 硬走 online。**

### §3.3 launch 计数与小 case 画像

conv 每分支 1 次 + 路径 A 每分支 1 次（路径 B 2 次）⇒ 全算子 **4~6 次 launch**。
小 case（B=1, N 仅十几）launch 串行占比高，是本类小 case speedup<1 的结构原因之一，
与 FA 小 case 画像一致；**不接受** host 侧合并两分支 conv（权重不同），
小 case 回退属结构固有，写入报告画像即可。

---

## §4 Layer 3: 关键技巧

### §4.1 ★★★ ADD_OUT kernel 内融合残差相加（省一路 kernel + 一遍 out 流量）

```python
if ADD_OUT:
    prev = tl.load(out_ptr + base, mask=msk, other=0.0)
    ov = ov + prev          # dtype_in add，对齐参考 out_pos + out_ch 的语义
tl.store(out_ptr + base, ov, mask=msk)
```

position 分支先写（ADD_OUT=false），channel 分支 kernel 内加（ADD_OUT=true）。
**前提是 L1.1 布局统一**——两分支 stride 算出同一批地址，融合才成立。
顺带把"两分支在 dtype_in 上相加"的契约（L1.7）天然落实。

### §4.2 ★★★ p hi+lo 二段拆分（见 L1.2 代码块）

p∈[0,1] 先 `p_hi = p.to(dtype)`，再 `p_lo = (p − p_hi.float()).to(dtype)`，两次原生
dtype dot 共享 fp32 累加器。仅非 fp32 输入需要（fp32 case 参考本来就 fp32 算）。
⚠️ 与 FA 卡 L1.5"禁降"不冲突：拆分不是降精度，是用两次原生 dot 还原 fp32 乘积。
**不要复制变量名/结构**——按语义重新实现。

### §4.3 ★★ k×k conv implicit GEMM 配方（k² 移位 dot + halo mask）

```python
p_r = offs_n // W;  q_c = offs_n % W            # token n 的 (行,列)
for k in range(KK):                             # k×k = k² 次移位 dot
    dh = k // KS - KS // 2;  dw = k % KS - KS // 2
    halo = (p_r+dh >= 0) & (p_r+dh < H) & (q_c+dw >= 0) & (q_c+dw < W)
    xt = tl.load(x_ptr + bi*C*N + offs_ci[:,None]*N + (offs_n+dh*W+dw)[None,:],
                 mask=ci_ok[:,None] & (n_ok & halo)[None,:], other=0.0)   # [BC,BM]
    wt = tl.load(w9_ptr + k*C*C + offs_ci[:,None]*C + offs_co[None,:],
                 mask=ci_ok[:,None] & co_ok[None,:], other=0.0)           # [BC,BN] 连续
    acc += tl.dot(tl.trans(xt), wt)
```

3×3 conv 的 tile 实测最优 **BM=64(n) / BN=128(co) / BC=64(ci)**——两个方向的收缩都
已证伪（§5.2）。halo 用静态索引 + 边界 where（不动态 gather），other=0 即 padding
零填充语义。

### §4.4 ★ 路径 B 的 Q=K=V 共享 load

无投影自注意力的访存红利：阶段 a 同一 `a` tile 自乘 `tl.dot(a, tl.trans(a))`，
K/V 零额外访存；内积维分块循环累加（D 可达数千，BDC=128/64 分档）。

### §4.5 scale host 预计算

`C**-0.5` / `N**-0.5` 在 host 算成 fp32 标量传入，kernel 内只做一次 s*scale。

---

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 双分支统一 kernel + ADD_OUT 融合 + 双路径分派（Phase 3 架构本身，L1.1/§4.1/L1.6） | 首版即 geomean **2.4313**（target 2.0，50/50） | 总是 |
| 2 | ★★ conv 外提 implicit GEMM + w9 预重排（L1.3） | conv 段不进 profiling 热点 | 总是 |
| 3 | ★★ p 二段拆分（L1.2/§4.2） | **精度必需**（不是性能）；顺带消 fp32 v 副本解 UB | 非 fp32 输入 |
| 4 | ★ tile 预算收紧到 8192/96K（L1.5） | 编译通过率 0→全过（是解锁不是提速） | kernel 内残差融合形态 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱**）

| 方向 | 结果 |
|---|---|
| tile 面积按分支分档（无 ADD_OUT 的分支恢复 16384） | ⛔ geomean −0.6%，无提升回退 |
| conv BN 收缩（块数不足核数时 BN→16，补并行度假设） | ⛔ 大 C case conv 430→1271µs（**3x 恶化**），小 dot 效率崩塌 |
| conv BM 收缩（BN=128 保持，BM→更小） | ⛔ 大 C case conv 430→961µs（**2.2x 恶化**），halo 窄条 MTE 效率下降 |
| launch 常量全 constexpr 化（T/D/CN/stride/TOTAL/NUM_PROG） | ⛔ 预检即持平（一个分支 conv 改善 25%、另一分支变差 8%、attn 持平），无一致收益方向，未进正式 4.2/4.3 |
| IR 级挖掘（conv kernel last_pass.mlir） | ⛔ pipe_barrier×33 + sync_block×14 + sync_block_wait×6 = MIX kernel **结构性开销**（同步为 conv+GEMM 混合固有），`ir_has_more_suggestions=false`；大量 i64 为地址计算常态，非标量降级 |

### §5.3 遗留瓶颈画像（写报告用，不要再扫参）

- **大 N 小 C**（position 分支）：KV 迭代数 × D=16 的 dot K 维计算强度低，tile 已达
  面积上限——实测最慢 case（speedup 0.44~0.57x）全部落在此族；
- **大 C 小 N**（conv 段）：BN=128 时块数塌陷（C=1024 → 8 块 < 24 核），收缩双向证伪（§5.2）；
- **小 case**（B=1, N 仅十几）：launch 串行占比高（§3.3）。

---

## §6 精度闸门

### §6.1 判定顺序（错一步会把 bug 归错类）

1. **先查双分支布局统一**（L1.1）：错位伪装成精度问题（残差融合读错地址 + view 错位）；
2. **再查权重 RNG 复刻**（L1.4）：`passed_cases == 0` 或大面积不匹配时，先单独验权重
   数值，再验 conv 输出，最后才查 attention kernel；
3. **再查编译失败集合**（L1.5）：ub overflow / PlanMemory 失败按 shape 划分集合，
   会污染设备造成级联假失败——先定位第一个失败用例单独复跑；
4. **最后才是数值精度**：p 拆分（L1.2）、分段 dtype 契约（L1.7）。

### §6.2 隐藏状态探针 VALUE_MISMATCH 为预期

cache 存的是预重排 `w9[k²,ci,co]`，与参考的 `[co,ci,k,k]` 布局不同但**数值逐位相同**。
验证手段是单独验权重数值 + conv 输出；**不要拿探针 mismatch 当失败信号**，
更不要为它放弃权重预重排（那是 L1.3 的性能前提）。

### §6.3 dtype 契约矩阵

| case 类型 | conv 段 | attention 段 | PV dot | 输出相加 |
|---|---|---|---|---|
| fp16（本类主流） | 原生 fp16 + fp32 acc，舍回 | 全程 fp32 | p_hi+p_lo 两次 fp16 dot | fp16 add（ADD_OUT） |
| fp32 | 原生 fp32 | 全程 fp32 | 原生 fp32 dot（不拆） | fp32 add |
| bf16 | 原生 bf16 + fp32 acc，舍回 | 全程 fp32 | p_hi+p_lo 两次 bf16 dot（~16 位尾数） | bf16 add |

---

## §7 测量口径

- 50 case **几何平均**；torch 参考实现代码恒定，其 `framework.avg_latency_ms` 是天然的
  环境探针——以首次全过时的均值为基准带（本类实测约 0.21 ms 量级），漂移超带即判该次
  测量无效、自动重测。**比值型指标不能自证有效**。
- 慢 case 集中在极端形状（§5.3），几何平均口径下小 shape 权重相同：扫参子集必须含
  大/中/小三档（FA F6），画像用于定位，**判定只能用完整的 verify + benchmark**。
- 同代码跨次测量存在约 **±1.5% 波动带**（实测 2.4313 → 2.4036，−1.1%）——Phase 4
  判定"无提升"时，|Δ| 在波动带内一律按无提升处理。

---

## §8 陷阱表

| 现象 | 根因 | 处理 |
|---|---|---|
| 大面积 AccuracyError 与 NaN 断言混合出现 | PV 升精度异构 dot（p fp32 × v.to(fp32)） | L1.2 / §4.2 二段拆分 |
| 近半数 case MLIRCompilationError（ub overflow / hivm-plan-memory） | s tile 64KB + ADD_OUT 组合超 PlanMemory；fp32 v 副本占 UB | L1.5（面积 8192 + UB 96K）；p 拆分顺带消 fp32 v 副本 |
| 输出错位、像精度差但误差无规律 | 双分支物理写序不一致（position (B,N,C) vs channel (B,C,N)） | L1.1 统一 (b,c,n) + stride 参数化 |
| passed=0 且权重对不上 | RNG 复刻缺一环（seed 时序/采样顺序/CPU fp32/bound） | L1.4 诊断顺序：先权重→conv→attention |
| 探针报 VALUE_MISMATCH | cache 存 w9 预重排，布局不同数值同 | §6.2 预期，非失败 |
| 大 C case conv 突然变慢 2-3 倍 | BN/BM 收缩"优化" | §5.2 证伪，回退 BM64/BN128/BC64 |
| fp16 case matched_ratio 稳定 ≈0.76 | conv 段或输出相加全程 fp32（比参考更准） | L1.7 分段契约，逐步舍回 dtype_in |
| D>256 case 编译崩 / UB 爆 | online 路径 acc[BQ,BD] 超预算 | L1.6 路径 B P 物化 |

---

## §9 与其它模板的分工

| 文件 | 何时用 |
|---|---|
| `flash_attention.md` | attention 主链 Layer 1 **全部依然成立**（清单见 §2 末尾），本卡只写 delta；两份都读 |
| `spatial_attention.md`（行 10a） | 空间自注意力**带插段**：K/V 链有空间缩减下采样+LN，或 transform conv+IN，且有 4 个 `nn.Linear` 投影。本类无投影、无下采样、无 IN，其 S 流量模型/平面统计恒等式不适用 |
| `patch_window_attention.md`（行 10b） | 卷积-块化窗口 transformer：patch 展开布局变换 + qkv/out 投影 + transformer 块 ×depth 堆叠 + 尾部 `conv(cat)` 融合。本类无 patch 块化、无投影、单块双分支残差相加 |
| `mha.md`（行 6a） | 4 个 `nn.Linear` 投影段的 MHA 形态。本类无投影段，conv 前置代替，投影 GEMM 经验不适用 |
| `dual_branch_attention.md`（行 13a） | 双分支**特征图门控**（BAM/PSA/Triplet 系），无 softmax(QKᵀ) 主链。本类两分支都是完整三段式 |
| `cv_attn_agg.md`（行 14） | CV 特征聚合-分发，KV 维极小或逐通道权重。本类 position 分支是 N×N 全量 token 自注意力 |
| `attention_index.md` | attention 家族定 `category` 的唯一入口。本文件对应其**行 10c**（一·3 空间 token 版 · 无投影双分支变体） |

冲突时以本文件为准（本文件结论均在无投影双分支形态上实测）。

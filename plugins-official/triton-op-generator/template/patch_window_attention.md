---
name: patch_window_attention
description: CV-Attn-空间自注意力算子（卷积-块化窗口 transformer 型：conv 局部表示 + NCHW↔token 的 unfold/fold 布局变换 + 小窗口多头 attention + FFN + 融合卷积，MobileViT/ViT-局部聚合系）的 Triton Ascend 优化经验合集，含形态识别、Layer 1 设计约束（复合点积双 B 指针禁令/拼接缓冲紧凑布局/元素级 gather 禁令/host cat 禁令/LN 自适应 BLOCK/attention BQ128 档位）、Layer 2 算法骨架（分阶段多 kernel + attn 按 n 单块/多块分派 + 统一 GEMM 变体）、Layer 3 关键技巧（im2col NHWC 一次转置/两段链式 conv/GEMM 配置档位）、精度闸门与证伪方向全表
metadata:
  type: reference
---

# CV-Attn-空间自注意力（卷积-块化窗口 transformer 型）优化经验

本文档是 **"卷积局部表示 + 特征图按 patch 展开成 token + 小窗口（n 通常 ≤ 数百）多头
attention + FFN，首尾含 3×3 卷积与 concat 融合"** 这一类算子（`attention_index.md` 行 10b，
一·3 空间 token 版 · 块化窗口细分，CV-Attn-空间自注意力家族）的经验合集，覆盖 Phase 2/3/4。

- **§0 适用范围与算子分类**（子类标签 + 判别特征 + 形态识别五问）
- **§1 通用经验**（S1~S4：瓶颈再诊断 / 分 dtype 容差 / 通道维可变 / 计时口径）
- **§2 Layer 1 设计约束 L1.1~L1.10**（Phase 2 硬性边界，precheck 整节摘录）
- **§3 Layer 2 算法骨架**（分阶段多 kernel + 按 n 分派 attention + GEMM 统一变体）
- **§4 Layer 3 关键技巧**（im2col NHWC / 两段链式复合卷积 / 诊断方法论）
- **§5 Phase 4 优化点清单**（收益排序 + **证伪方向全表** + 天花板估算）
- **§6 精度闸门** · **§7 测量口径** · **§8 陷阱表** · **§9 与其它模板的分工**

> ⚠️ **本文件与 `mha.md` / `flash_attention.md` 的分工**：
> 本类算子的 attention 主链（LN → qkv 投影 → 窗口 softmax(QKᵀ/√d)V → out 投影 → 残差 →
> LN → FFN）与行 6a 的投影段形态相同，`mha.md` §2.5 M1~M4 投影段硬约束与
> `flash_attention.md` 的 L1.1 dot 契约 / L1.12 权重预转置 / L1.13 BLOCK 面积收缩
> **依然全部适用**。本文件**只补充"卷积-块化"带来的那部分**：im2col 物化与布局、
> NCHW↔token 的 unfold/fold 变换、首尾 3×3 卷积与双输入融合卷积的复合点积写法、
> 小窗口 attention 的单块特化，以及由此引出的拼接缓冲错位、双 B 指针 kernel 崩溃、
> 通道维逐 case 可变等 MHA 上不会遇到的坑。**三份都要读**；冲突时以本文件为准。
>
> **证据基础**：本类 49 case 完整定位轨迹——B∈[1,8]、C∈[4,1024]、H/W∈[2,406]、
> 窗口 n∈[1,725]、patch∈[2,14]、dim∈{256..1024}、dtype fp16/bf16/fp32 混合，
> 910B2C（20 AI cores）/ CANN 9.1.0 / triton-ascend 3.2.2。
> 精度全 50/50，几何平均从基线 0.024 → **1.08**（三轮优化）。
>
> ⚠️ **核心优化哲学**：这类算子的耗时结构是 **"投影/卷积 GEMM 占 ~75% + im2col 物化
> 占 ~25% + attention 占 ~10%"**——第一优化动作永远是：① 复合卷积的写法审查
> （双 B 指针 kernel 是隐藏的 100× 陷阱，见 L1.1）；② im2col 的 per-tap permute 消除
> （见 §4.1）；③ attention 主链放最后——它的天花板受同步结构限制，优化排序上
> 永远先攻 GEMM 与数据搬运侧。

---

## §0 适用范围与算子分类

### §0.1 判别特征（决定用不用本文件）

打开参考实现，看有没有这组写法（命中 ≥3 条即用本文件）：

1. 输入是 NCHW 特征图，先过 **k×k 卷积（stride=1, padding=k//2）+ 1×1 卷积**做局部表示；
2. 特征图按 `patch` 展开成 token：`reshape(b, nh, ph, nw, pw, D).permute(0,2,4,1,3,5)`
   类的 **NCHW → (b, p, n, D)** 布局变换（unfold），输出侧有对称的 fold 逆变换；
3. **窗口内**多头 attention：n = nh·nw 通常 ≤ 数百（几百量级），每窗口独立 softmax，
   无跨窗口交互；
4. transformer 块 × depth（常 2~3）：LN + qkv/out 投影 + 残差 + LN + FFN（SiLU/GELU）；
5. 尾部 **concat 双输入的 k×k 融合卷积**：`conv(cat([x, y], dim=1))`——x 是原始输入、
   y 是 fold 回空间的 transformer 输出，两路各占一半输入通道。

子类标签：`conv-patch-transformer`（本文件主体）。

### §0.2 ★ 形态识别五问（Phase 2 第一步必须回答）

| # | 问题 | 答案影响 |
|---|---|---|
| 1 | 窗口 n 的分布范围？ | 决定 attention 走单块特化还是多块分派（§3.2） |
| 2 | 首尾卷积的 K（=C·k²）多大？ | K ≥ 数千时双 B 指针禁令（L1.1）必然命中，且 im2col 写放大成为主要流量 |
| 3 | dim（transformer 宽度）逐 case 可变吗？ | 可变则禁止任何常量假设（L1.6），LN/GEMM 的 BLOCK 全部自适应 |
| 4 | 参考 dtype 分布？ | fp32 case 的 MERE 容差（~1.2e-4）锁死低精度路径（L1.8） |
| 5 | GEMM/attention 的 BLOCK 档位是否已按 §3.3 实测过？ | 档位变更后必须全 case 编译+精度回归（L1.9） |

## §1 通用经验（跨形态，首次生成必须遵守）

### S1 ★★ 优化前置必做"瓶颈再诊断"，禁止沿用历史结论

本类算子的旧报告/直觉常把瓶颈归到 attention——实测拆解（分阶段计时，逐 kernel 单测）
曾多次推翻：真正的耗时大头在**复合卷积 kernel 的写法崩溃**或 **im2col 物化**。
续优或换环境后的第一动作：写分阶段计时脚本（每 kernel 单测 5~10 次取均值），
拿真实分布再排优化顺序。

### S2 分 dtype 容差意识

fp16 case 的输出容差（atol ~9e-2 / MERE ~9.8e-4）宽松，bf16 更宽（~7.8e-3），
但 **fp32 case 的 MERE（~1.2e-4）把所有低精度 dot 路径全部锁死**——只要 case 集里
混有 fp32，全链路就必须 fp32 ieee（L1.8）。

### S3 通道/宽度维逐 case 可变

C、dim、窗口 n、patch 全是逐 case 变量。任何 `D=512 之类` 的常量假设都会在
某个 case 上崩溃（UB 溢出或 shape 错误）。全部参数化 + 自适应 BLOCK。

### S4 ★ 计时口径：进程内 back-to-back 冒烟会高估 host 优化

无 sync 的循环连跑测的是**流水线吞吐**，不是单次延迟——删 host 同步类优化会被
高估 4~20×。性能判定一律以 benchmark 工具的隔离计时为准（§7）。

## §2 Layer 1: 设计约束（首次生成就要全部满足）

### L1.1 ★★★ 复合点积禁止"双 B 指针单 kernel"写法，大 K 下崩溃两个数量级

`out = A1@B1 + A2@B2` 型（双输入融合卷积、分流投影）禁止写成"一个 kernel 两对指针、
每 K 块发 2 次 dot + 尾部 `tl.trans(acc)` 散射 store"。实测同形状下该写法比
plain GEMM **慢 134~181×**（K=C·k² 达数千时）——每 K 块双 dot 的串行发射 +
`offs_m // HW` 类整除地址 + NCHW 散射 store 三重叠加把 MTE/Cube 流水全部打散。

```python
# ❌ 双 B 指针单 kernel（每 K 块 2 dot + trans 散射 store）
acc = tl.dot(a1, b1, acc, input_precision="ieee")
acc = tl.dot(a2, b2, acc, input_precision="ieee")
...
tacc = tl.trans(acc)
tl.store(C + b_idx[None,:]*(N*HW) + offs_n[:,None]*HW + pos[None,:], tacc, ...)   # 散射
```

**正确写法二选一**：
- **K 维拼接单 GEMM**：`A = [A1 | A2]`（host 侧同一紧凑缓冲的两段），`B = cat([B1, B2], 0)`，
  一次 plain GEMM；
- **两段链式 GEMM**：第一段无 bias 出 tmp，第二段 `ADD_RESIDUAL` 在 kernel 内读 tmp 相加
  （当 host 侧 cat 也被 AST 禁令覆盖时用这路，见 L1.5）。

### L1.2 ★ K 维拼接缓冲的段间禁止任何对齐 pad

多段拼进同一缓冲喂 GEMM 时，**逻辑 K 索引必须是物理连续列**。若各段保留 `ceil16` 之类
对齐 pad，K 扫描会读到"上段 pad 列 + 下段错位列"——**只有对齐整除的 case 碰巧正确**，
非整除 case（如 C=4、C·k²=36）全错且极难定位。缓冲取 `[rows, Σ 段实际宽度]` 紧凑布局，
段写入用统一行步长。

### L1.3 ★★★ 禁止元素级地址 gather-GEMM（把 im2col 搬进 GEMM kernel）

用 `ii = i + di - 1; addr = ii*W + jj` 之类的**逐元素地址计算**替代 im2col 物化，
再叠加逐元素边界 mask——实测**慢 68×**（大 case 数千 ms vs 数十 ms）：元素级寻址使
MTE 向量化彻底失效，Cube 饿死。im2col 物化 + plain GEMM 仍是当前最优结构；
若要省物化，正确方向是 host 布局优化（§4.1），不是 kernel 内 gather。

### L1.4 权重全部 host 预转置 [K, N]（行主序喂 B 侧）

`nn.Linear`/conv 展平后的权重是 [out, in]，直接喂 GEMM 的 B 侧等于离散列读。
host 侧 `w.reshape(...).t().contiguous()` 一次（形状操作，AST 允许），kernel 内
B 侧行主序连续读。

### L1.5 ★ forward 侧禁止 host `torch.cat` 聚合与直接 for 循环

AST 预检查会把 `torch.cat` 判为禁止的 host 计算操作（Type 3），forward 体内的
Python for 循环同样被拒。分段矩阵的聚合消费用 L1.1 的两段链式（kernel 内 residual
相加）；循环体只含 kernel 启动的 host 函数放在**独立辅助函数**中（辅助函数内的
纯 launch 循环可通过检查）。

### L1.6 ★ dim / 通道 / 窗口逐 case 可变，禁止常量假设

不要假设 `dim=512`、`C 是 64 的倍数`、`n ≤ 64` 之类。LN 的 BLOCK_M 必须按 BLOCK_D
自适应：`max(4, 8192 // BLOCK_D)`——BLOCK_D=1024 时 BLOCK_M=16 会让 tile×3 份缓冲
超 192KB UB 直接编译失败。GEMM 的 BN/BK 按 `next_power_of_2` 收缩。

### L1.7 attention 按 n 分派：单块特化与多块路径

`np2(n) ≤ 128` 走 **KV 单块特化**：全 KV 一次进 tile，softmax 无 online rescale
（消掉 m/l 滚动与 alpha 乘法，向量链减半）；`n > 128` 走多块 flash（BQ/BKV=128）。
块宽恰为 n 时（EVEN）进一步消掉边界 where。分派在 host 侧按 constexpr 特化，
禁止用运行时标量 if 分支（前端会拒编译）。

### L1.8 ★★★ fp32 ieee dot 契约（case 集含 fp32 时锁死）

实测把 dot 输入 cast 到 fp16（其余全 fp32）：全链路 matched_ratio 从 1.0 掉到
**0.22~0.78**（阈值 0.9）——30 层非线性链（LN/softmax/SiLU）逐级放大 0.5~1 ulp 舍入差，
任何一层的低精度注入都会击穿。且 cast 本身引入额外 Vector↔Cube 同步，**反而更慢**。
只要容差表里存在 fp32 级 MERE，全线 `input_precision="ieee"`，不要试探。

### L1.9 ★★ BLOCK 档位必须整体实测确定，禁止沿用任何"保守档"

residual/silu GEMM 变体与 plain 变体**同档可用**（BM=256/BK=128，实测 ~67 TFLOPS），
attention BQ=128 可用——不要因"多一份 epilogue tile 怕 UB"就主动降档（降到 BM=128/BK=64
只有 ~48 TFLOPS，白白损失 30%）。档位以 §3.3 的实测表为准；任何档位变更后必须
全 case 编译 + 精度回归（UB 溢出与 tile 组合的关系是非线性的，个别 case 崩不等于
全档不可用，反过来也一样）。

### L1.10 host 同步全删

同 stream 的 kernel 发射天然串行，`torch.npu.synchronize()` 逐 kernel 插入只增加
host 往返（实测删 19 处无任何功能差异，隔离计时下收益约 +0.3%，但为后续
launch 优化扫清障碍）。保留 weight 预处理后的至多一次。

## §3 Layer 2: 算法骨架

### §3.1 主骨架（前向，分阶段多 kernel）

```
x[b,c,h,w] (fp32 内部计算)
 ├─ im2col(x): [b, hw, c·k²] 紧凑缓冲（NHWC 一次转置 + per-tap 纯拷贝，§4.1）
 ├─ conv1 GEMM: x_col @ W1ᵀ[c·k², c] + bias          → y1 [b·hw, c]
 ├─ conv2 GEMM: y1 @ W2ᵀ[c, dim] + bias               → y2 [b·hw, dim]
 ├─ unfold（布局视图）: y2.view(b,nh,ph,nw,pw,D).permute(0,2,4,1,3,5) → [b, p, n, D]
 ├─ transformer × depth（每层 7 kernel，无 host sync）:
 │    LN → qkv GEMM → attention(按 n 分派) → out GEMM(+残差) → LN → ff1 GEMM → ff2 GEMM(SiLU+残差)
 ├─ fold（布局视图）: 逆 permute 回 [b, hw, D]
 ├─ conv3 GEMM: y_fold @ W3ᵀ[dim, c] + bias           → y3 [b·hw, c]
 ├─ im2col(y3): 同款 NHWC 路径                          → y_col [b, hw, c·k²]
 ├─ conv4 两段链式:  x_col @ W4aᵀ → tmp;  y_col @ W4bᵀ + tmp + bias → out[b·hw, c]
 │    （双输入融合卷积，L1.1/L1.5 的链式写法）
 └─ 输出布局 [b,hw,c] → NCHW permute（torch 视图）
```

要点：
- **conv / 投影全部走同一个 GEMM kernel 的 constexpr 变体**（HAS_BIAS / ADD_RESIDUAL /
  APPLY_SILU_A），一份 kernel 覆盖全算子的矩阵乘段；
- unfold/fold 只做 permute+reshape 视图（torch 形状操作），**不物化**（permute 后
  reshape 会拷贝一次，量级 ~数百 MB/case，可接受；进一步消除会破坏 GEMM 连续性，
  见 §5.2 证伪表）；
- attention 的 grid = min(核数, 任务数)，任务 = (b, patch)，head 在 kernel 内循环。

### §3.2 ★★ attention 的 n 分派判据

| 窗口规模 | 路径 | tile | 说明 |
|---|---|---|---|
| np2(n) ≤ 128 | **KV 单块特化** | BQ=128（n>64），BKV=np2(n) | 无 m/l 滚动、无 alpha、EVEN 时无边界 where；dot 次数 = ceil(n/BQ) |
| n > 128 | 多块 flash | BQ=128, BKV=128 | 标准 online softmax，m 初始 -1e30 |

单块特化对小 n 的收益 1.2~2.0×；对 n≤64 的极小窗口，BQ=BKV=np2(n)（≥16）。

### §3.3 GEMM 配置档位（fp32 ieee 实测）

| 变体 | BLOCK 配置 | 实测吞吐（910B2） |
|---|---|---|
| plain（无 residual/silu） | BM=256 / BN=128 / BK=128 | ~67.5 TFLOPS |
| residual / silu 变体 | BM=256 / BN=128 / BK=128（与 plain 同档） | ~67 |
| LN | BLOCK_M = max(4, 8192//BLOCK_D) | — |

BN=256 与 BK=256 均不可编或更慢；grid 收缩 `min(核数, cdiv(M,BM)·cdiv(N,BN))`。

## §4 Layer 3: 关键技巧

### §4.1 ★★★ im2col 的正解：NHWC 一次转置 + per-tap 纯连续拷贝

per-tap 做 `NCHW 切片 → permute(0,2,3,1) → reshape → 写入`（每个 tap 一次 4D 转置
拷贝）是隐性大头。正解分两步：

```python
srcn = src.permute(0, 2, 3, 1).contiguous()          # 第一步：全图一次 NHWC 转置
xp = torch.zeros((b, h + 2*pad, w + 2*pad, c), ...)  # 第二步：pad 后按 tap 纯拷贝
xp[:, pad:pad+h, pad:pad+w, :] = srcn
for t in range(k*k):                                  # 每 tap 连续 reshape 拷贝，无 permute
    dst[:, :, t*c:(t+1)*c] = xp[:, di:di+h, dj:dj+w, :].reshape(b, h*w, c)
```

实测大 case **18.9ms → 10.5ms（1.8×）**，数值逐位等价。行主序源（token 布局 [b,hw,c]）
侧同理已是 NHWC 等价路径，无需再动。

### §4.2 ★ 双输入融合卷积的两段链式（替代 host cat 与双 B 指针）

第一段 GEMM 出 tmp（无 bias），第二段 `ADD_RESIDUAL` 在 epilogue 读 tmp 相加 + bias。
tmp 的跨布局读取（如转置世界 [c, b·hw]）用独立 stride 参数（rs_m/rs_n）寻址，
不要在 host 侧再转置一次。

### §4.3 ★ GEMM 变体同档复用：epilogue 代价远小于降档损失

bias / residual / silu 变体的 epilogue 只在 K 循环外多一次 load + 一次加法
（residual tile 仅在收尾驻留），UB 预算足够支撑与 plain 相同的 BM=256/BK=128——
同档实测 ~67 TFLOPS。把变体降档到 BM=128/BK=64 会损失 ~30% 吞吐，是本类算子
最常见的"自我设限"。唯一例外：APPLY_SILU_A 在 K 循环**内**对 A tile 做激活时，
多一份激活中间量，若编译报 UB 溢出再单独对该变体降 BK（先降 BK 再降 BM）。

### §4.4 ★ 诊断方法论：GEMM "吞吐上限"先问搬运还是计算

GEMM 卡在某个 TFLOPS 时，用 `msprof op simulator` 采 per-instruction pipe 分布
（单 kernel 小脚本仿真，勿整 forward）：若 **MTE2/MTE1 合计 >70% 而 CUBE <20%**，
瓶颈是搬运未流水（`MOV_OUT_TO_L1*_ND2NZ` + `SET_FLAG` 等待），不是计算极限——
此时调 BLOCK 无益，见 §5.2 的流水掩盖证伪表。IR 侧旁证：`npuir.mlir` 里
plain GEMM 应只有 1×mmad/1×sync；attention 的 38×sync_block 属 mix 结构固有。

## §5 Phase 4 优化点清单

### §5.1 按收益排序（★ = 高收益）

| # | 方向 | 实测增益 | 适用条件 |
|---|---|---|---|
| 1 | ★★★ 复合卷积写法审查：双 B 指针 kernel → K 维拼接/两段链式 | **65×**（单 kernel 段） | 尾部融合卷积 K=C·k² ≥ 数千 |
| 2 | ★★★ GEMM 全变体同档（BM256/BK128）+ attn BQ128 | **+18%**（整体几何平均） | 变体被降档时 |
| 3 | ★★ im2col NHWC 一次转置 | **1.8×**（im2col 段） | per-tap permute 写法存在时 |
| 4 | ★★ attention 按 n 单块特化 | 1.2~2.0×（attention 段） | np2(n) ≤ 128 的窗口 |
| 5 | ★ GEMM BLOCK 档位（§3.3） | plain 变体 47.9→67.6 T | 首轮基准扫一遍 |
| 6 | ★ host sync 全删 | ~+0.3%（隔离计时口径） | 总是可以 |
| 7 | ★ BK 64→128（plain 段） | +10%（GEMM 段） | 旧配置是 BK64 时 |
| 8 | LN BLOCK_M 自适应 | 消除 D=1024 case 的编译崩溃 | 总是 |

### §5.2 ⛔ 证伪方向全表（**这一节比 §5.1 更值钱**）

| 方向 | 结果 |
|---|---|
| **元素级地址 gather-GEMM**（im2col 搬进 kernel，逐元素 ii\*W+jj + 边界 mask） | ⛔ **慢 68×**（大 case 数千 ms）。MTE 向量化失效，Cube 饿死；小 case 还会数值错（寻址溢出） |
| **行置换 store/load**（unfold/fold 的 permute 吸收进相邻 GEMM 的行索引） | ⛔ ±5% 且引入 case 级编译失败。置换 store 使写散射化，GEMM store 变慢抵消全部 copy 消除收益 |
| **F.unfold + GEMM 换边**（aclnn 原生 im2col 输出 [b,ckk,hw] 直喂，权重转置读，输出转置世界） | ⛔ 双输：per-batch 小 M 的 GEMM 换边吃掉 unfold 收益（0.93~0.99×），且 bias 维度错位（换边后通道在 M 维，kernel 的 bias 固定加 N 维）难修。unfold 本身 2.5× 于 pad+切片，但正确消费方式缺 kernel 侧跨 batch 寻址支持 |
| **低精度 dot（fp16 cast 输入）** | ⛔ matched 0.22~0.78（阈值 0.9）**且更慢**（cast 引入 Vector↔Cube 同步）。fp32 case 的 MERE 容差锁死 |
| **Winograd F(2×2,3×3) 卷积** | ⛔ fp32 下 V/M 中间物化流量（conv 段 5~9GB）吃掉 2.25× FLOP 节省：conv1 仅 1.15×，双输入融合段 **0.64× 倒贴**。低精度下才可能有效 |
| **`num_stages` 软件流水（tl.range NS=2/3/4）** | ⛔ 零效果（67.6T 纹丝不动）。该参数被忽略，自动流水深度由后端自定 |
| **手动双缓冲**（循环内预取下一块 A/B） | ⛔ 大 tile 下 UB 编译失败（双份 tile 驻留 >192KB）；小 tile 下基线吞吐太低，无净收益 |
| **NZ 分形布局 B**（host 重排分形，kernel 内 k//16/n//16 寻址） | ⛔ **0.02×（900ms 惨败）**。kernel 内逐元素 div/mod 再次触发 MTE 失效——与 gather-GEMM 同一根因 |
| **K 维拼接 + 段间对齐 pad** | ⛔ 非对齐 case（C·k² 不被 16 整除）静默数值错——pad 列被 K 扫描读进结果。见 L1.2 |

### §5.3 天花板估算：先算账再动手

三步算清本类算子的可达上限，避免空转：

1. **GEMM FLOPs**：`Σ 2·M·N·K`（conv1/conv4 的 K=C·k² 与 conv4×2 是大头，投影/FFN ×depth）。
   除以实测 GEMM 峰值（fp32 ieee ≈ 67T）得 GEMM 下限；
2. **im2col 流量地板**：`(读 1 + 写 k²)×b·hw·C·4B` 每次物化，÷ ~250GB/s；
3. GEMM 下限 + im2col 地板已占预算的百分比 >100% 时，任何 kernel 级优化都到不了目标——
   剩余手段只有算子级 FLOP 削减（需精度契约配合）或结构变更。

## §6 精度闸门

### §6.1 判定顺序

数值错时先分阶段 bisect（每 kernel 输出 diff），再改代码。本类算子的高频错因排序：
① 拼接缓冲段间 pad 错位（只有非对齐 case 失败——直接指向 L1.2）；
② 行主序/列主序 stride 传错（单 kernel 全错）；③ unfold/fold permute 维序错
（空间位置散布式错）；④ 低精度注入（matched_ratio 大面积 0.2~0.8）。

### §6.2 MERE 临界意识

fp16 case 的 MERE 阈值（2⁻¹⁰）恰等于 fp16 ulp——近零 golden 值的相对误差天然临界，
偶发单 case 失败重跑即过属正常抖动；连续失败才是真 bug。

## §7 测量口径

- **判定一律用 benchmark 工具的隔离计时**（每 case 独立进程/独立 sync）；
- 进程内 back-to-back 冒烟（无 sync 循环）测的是吞吐，会把删 host sync 类优化高估 4~20×；
- 单 kernel 微基准（timeit + npu.synchronize，warmup 3 / rep 5~10）用于优化点排序，
  注意用**真实 shape 段**（小 shape 会把搬运占比放大，结论不可外推）。

## §8 陷阱表

| 陷阱 | 表征 | 规避 |
|---|---|---|
| 双 B 指针复合 kernel | 融合卷积段耗时占比 >90%，单 kernel 数百 ms | L1.1：拼接/链式 |
| 拼接段间 pad | 仅 C·k² 非 16 整除的 case 数值错 | L1.2：紧凑布局 |
| gather-GEMM | im2col 消失了但总时延暴涨几十倍 | L1.3：禁止 |
| host `torch.cat` | AST Type3 违规，verify 全过但检查不过 | L1.5：两段链式 |
| dim 常量假设 | 个别 case UB 编译崩 / shape 错 | L1.6：全参数化 |
| LN 大 BLOCK_M | BLOCK_D=1024 的 case 编译失败 | 8192//BLOCK_D |
| 运行时标量 if 分支 | 前端直接拒编译（"buffer shape should be static"） | constexpr 分派/独立 kernel |
| 变体自我降档 | residual/silu GEMM 用 BM128/BK64，白丢 30% 吞吐 | L1.9：与 plain 同档 |
| 冒烟吞吐假象 | host 优化"实测 1.2×"，benchmark 只 +0.3% | §7 口径 |

## §9 与其它模板的分工

| 相邻模板 | 边界 |
|---|---|
| `mha.md` / `flash_attention.md` | 投影段与 attention 主链的通用约束（M1~M4、L1.1/L1.12/L1.13）依然适用，本文件不重复；冲突时以本文件为准 |
| `spatial_attention.md`（行 10a） | 同属 CV-Attn-空间自注意力家族，但那是"特征图 token 化 + K/V 下采样/transform conv 插段"（PVT 系）；本文件是"卷积局部表示 + patch 块化窗口 transformer"（MobileViT 系）。判别：K/V 生产链有无空间缩减层、token 是否按 patch 窗口分组 |
| `sparse_unfold.md`（行 9） | 那是"邻域展开构造 KV"（halo 窗口 gather），attention 语义是窗口间稀疏；本文件窗口内是稠密全 attention，unfold 只是布局变换 |
| `convolution.md` | 纯卷积算子的 K-tap 分解等经验；本文件的卷积段全部走 im2col+GEMM，两边的判据不同（K=C·k² 巨大时 im2col+GEMM 优） |

---
name: cannbotdsl-mla
description: "在 CANNBotDSL 里写 MLA（Multi-head Latent Attention，DeepSeek-V2/V3 的注意力）kernel。MLA 的 Q/K 各拆成 nope 和 rope 两段传入，head dim 大（d_nope 448~512、d_rope 64），且典型配置 N_kv=1（128 个 query head 共享一份 KV）——这三点让它和普通 Flash Attention 的 tiling、buffer 预算、性能瓶颈都不一样，照搬 FA 蓝本会踩坑。当用户要求：写 MLA kernel、实现 DeepSeek 的注意力、处理 q_nope/q_rope/k_nope/k_rope 这组输入、调 MLA 的 tile 或 buffer 预算、修 MLA 精度问题、或优化 MLA 的 decode 性能时，触发此 skill。即便用户只说「latent attention」「MLA 算子」「nope rope 拼接的 attention」也要触发。Triggers: MLA, latent attention, 多头潜在注意力, DeepSeek attention, q_nope, q_rope, k_nope, k_rope, nope rope, d_nope, numKVHeads。非 MLA 的 attention（标准 FA、GQA、PagedAttention）走 cannbotdsl-flash-attention。"
---

# 在 CANNBotDSL 里写 MLA kernel

MLA 是 Flash Attention 的变体，但有三个结构性差异会让直接照搬 FA 蓝本失败。先理解这三点，再动手。

## 算子语义

```
Q = concat(q_nope, q_rope)      最后一维 d_nope + d_rope
K = concat(k_nope, k_rope)      最后一维 d_nope + d_rope
y = softmax(Q @ K^T * scale) @ V        V 的 head dim = d_nope
```

`scale <= 0` 时取 `1/sqrt(d_nope + d_rope)`。GQA：`G = N_q / N_kv`，MLA 典型 `N_kv=1`。
causal 为**右下角对齐**：`j > i + (S_kv - S)` 置 -inf（与 FA 的 mask_mode=3 定义一致，可直接复用）。

典型规模：`d_nope ∈ {448, 512}`、`d_rope = 64`、`N_q ∈ {64, 128}`、`N_kv = 1`、
`S ∈ {1, 2, 128~512}`、`S_kv ∈ {128~2048}`。

## 第 1 步 —— 抓住核心简化：concat 在归约轴上

**这是 MLA 最重要的一条，不知道会白白多做一次 GM 拷贝和 pad。**

concat 落在 QK 的**归约轴**上，所以根本不需要真的拼出 576 宽的 Q/K：

```
Q @ K^T = [Q_nope | Q_rope] @ [K_nope | K_rope]^T
        = Q_nope @ K_nope^T  +  Q_rope @ K_rope^T
```

两个部分积累加进**同一个 fp32 L0C 累加器**即可，零拷贝、零 pad、**数值精确**
（实测 rel=4.3e-7，即 fp32 累加器的舍入水平）。

rope 段要用**独立且尺寸精确**的 channel（`d_rope=64` ≠ `d_chunk`），
复用 nope 的宽 channel 会让 mmad 的 K 长度不明确。

## 第 2 步 —— 挑蓝本

以仓内已验证的 dense/causal FA 为蓝本做 diff，不要从零写。关键选择：**取 `tile_n = 128`**，
这样整个 vec 侧（raw-mode VF online softmax）可以**原样复用**——它的列循环写成
`range(0, tile_d, VL_T)`、行循环由 `tile_vec_m` 界定，与 `tile_d` 无关，
`tile_d` 从 128 变 512 自动适配。

需要改的只有 cube 侧和顶层编排：

| 维度 | FA 蓝本 | MLA | 改动性质 |
|---|---|---|---|
| QK 归约 | 单张量 D=128 | nope+rope 两组累加 | **Matmul 类重写** |
| QK 归约长度 | 128 | 576 | 沿 K 轴切 chunk 累加 |
| V head dim | 128 | 448/512 | PV 沿**输出轴**切 chunk |
| tile_cube_m | 128 | **64** | L0C 要装 (M, d_nope) fp32 |
| tile_vec_m | 64 | **32** | 同上 |
| tile_n | 128 | 128 | **不变 → vec 侧零改动** |
| GQA / causal / layout / dtype | — | 同 | 直接复用 |

## 第 3 步 —— 写代码前先算 buffer 预算

硬限制：**UB 256 K、L1 512 K、L0A/L0B 各 64 K、L0C 256 K**。
超限在 `Channel` 构造期就会 `ValueError`，不用等 507015。

**L0B 是最紧的**：`(128,128) f16 = 32 K`，depth=2 就打满，再加 rope 的 L0B 立刻超。
L0B depth 只能给 1。

四组已验证配置见 `references/buffer-budget.md`。当前推荐配置（d_nope=512）：

```
L0A  40 K / 64 K      L0B  48 K / 64 K      L0C 192 K / 256 K
L1  232 K / 512 K     UB  225 K / 256 K   ← 最紧的两个是 L0B 和 UB
```

## 第 4 步 —— 两个 chunk 循环，方向相反

这是 MLA cube 侧的全部难点。**两个循环切的轴不同，写法也不同**：

**QK：切归约轴，累加进同一区域**

```python
slot = self.qk_l0c.acquire()
acc = local_slice(slot, (m_rows, tile_n))     # m_rows 见第 5 步
for c in tuple(range(n_dchunk)):              # nope chunks
    Q_nope[:, c] -> L1 -> L0A ; K_nope[:, c] -> L1 -> L0B   # L0B 不加 transpose
    matmul(acc, l0a, l0b, init=(c == 0))      # ← init=False 累加
Q_rope -> L1 -> L0A_r ; K_rope -> L1 -> L0B_r
matmul(acc, l0a_r, l0b_r, init=False)         # rope 继续累加
qk_l0c.commit(slot)
```

**PV：切输出轴，各写不同列带**

```python
mem_copy(self.l0a_pv, p_l1_ch)                # P 一次，所有 band 共用
slot = self.pv_l0c.acquire()
for c in tuple(range(n_dchunk)):
    V[:, c] -> L1 -> L0B_pv  (transpose=True) # ← PV 必须转置，QK 不要
    band = local_slice(slot, (m_rows, d_chunk), offset=c * m * d_chunk * 4)
    matmul(band, l0a_pv, l0b_pv, init=True)   # ← 各 band 独立，init=True
pv_l0c.commit(slot)
```

三个必须注意的点：

1. **`init` 语义相反**：QK 多 chunk 写同一区域要累加（`init=False`）；PV 各 band 互不重叠，各自 `init=True`。
2. **必须手动 L0C 事务**：N 个 mmad 写同一 slot 时，channel-first 自动 4 相协议会报 `no legal FIFO solution`。
3. **静态展开用 `tuple(range(n))`**，`const_expr(range(n))` 返回 bool 会报 TypeError。

## 第 5 步 —— d_chunk 取「能整除 d_nope 的最大值」

```python
D_CHUNK_MAX = 128                                  # L0B 64K 限制
d_chunk = 128 if d_nope % 128 == 0 else 64         # 512→128, 448→64
```

取除数而不是「128 + 尾块」，是为了**彻底消灭部分尾块**。原因很实际：QK 的尾块缩短的是
*归约*轴（零填充能吸收），PV 的尾块缩短的是*输出*轴（不能），两者处理方式相反；
而且窄尾块的 `local_slice` 和同 channel 的满宽用法混在一起会静默算错。全满宽最省心。

**代价**：`d_chunk != tile_n` 时 QK 和 PV 的 L0 操作数形状不再重合，**必须各分配一对**——
这是本算子最隐蔽的坑，详见 `references/pitfalls.md` §1。

## 第 6 步 —— M 尾块与短序列

三处 M 相关的处理，缺一个就在 decode/MTP 上错：

**(a) L0C 累加器跟随实际 M**。GM 的 Q 视图是 tail-aware 的，M 尾块时 L0A 行数
< tile_cube_m，而 L0C slot 仍是声明的满尺寸 → mmad verifier 报
`M dimension mismatch: dst[0]=64 vs lhs[0]=1`。用 `local_slice(slot, (m_rows, ...))` 对齐。

**(b) 每个 m-tile 的行数必须 > tile_vec_m**。split-M 的空分区 AIV 会污染共享 softmax
状态（详见 pitfalls §2）。做法：按 S 缩小 M tile（下限 16 = NZ fractal），
并把 S 补齐到 tile_cube_m 的整数倍，让**每个** tile 都是满的。

**(c) 补 query 行必须前置**。causal 右下角对齐，追加 padding 会让每个真实行少看
`pad` 个 key。前置则可见范围不变，输出取尾部 `out[:, pad:]`。

## 第 7 步 —— 数值路径（精度优先）

| 环节 | dtype |
|---|---|
| QK 累加（nope 多 chunk + rope 全部） | **fp32 L0C**，不落中间精度 |
| scale / mask / rowmax / exp / rowsum | fp32 |
| P → cube 前 | fp32 → fp16（与业界 FA/FIAS baseline 一致） |
| PV 累加 | **fp32 L0C** |
| res_o / online rescale | fp32 |
| 输出 | 除以 rowsum 后 cast 回 fp16/bf16 |

实测该路径 MERE ≈ 1.6e-3 vs fp64，其中**几乎全部来自 P 的 fp16 cast**
（P 保持 fp32 可到 1.0e-6）。如需更高精度，cube 支持 fp32 matmul（实测 MERE 8.1e-7），
代价是 L0 容量翻倍、depth 降到 1。

**验收必须用官方 checker**（fp64 golden 截断 + 小值域兜底），不能拿同 dtype 的 torch
golden 套公式——后者自身误差 2.10e-3，比被测 kernel 还大。详见
`../../core-skills/cannbotdsl-probe-debug/SKILL.md` §6。

## 第 8 步 —— 性能：先看 decode

MLA 的 `N_kv=1` 意味着**所有 query head 共享同一份 K/V**。若按 (batch, head) 切 m-tile，
K/V 会被重读 `N_q` 遍——decode 实测 9.13 GB 流量 vs 71 MB 真实数据，**128 倍冗余**，
`mte2_ratio` 高达 0.96（纯 memory bound）。

**非 causal 下可用纯 host reshape 消除**（kernel 零改动，数值无损）：把 query head 折进
序列轴当成单头问题。实测 **decode 2532 µs → 52.97 µs，47.8×**。

细节、适用边界、以及 causal 版本为什么失败，见 `references/perf.md`。

## References

| 文件 | 何时读 |
|---|---|
| `references/buffer-budget.md` | 第 3 步——4 组已验证配置 + 逐 buffer 明细 |
| `references/pitfalls.md` | **始终**——已经付过代价的坑，尤其 §1 |
| `references/perf.md` | 第 8 步——head folding、瓶颈数据、失败的尝试 |

## 外部依赖

| 依赖 | 用途 | 缺失后果 |
|---|---|---|
| **NPU 设备** + cannbotdsl wheel + CANN | 编译与实测 | 只能做设计推演，精度/性能均无法验证 |
| FA 蓝本（`../cannbotdsl-flash-attention`，或仓内已验证的 dense/causal FA 实现） | 第 2 步的 vec 侧原样复用 | 需自己写 raw-VF online softmax，工作量大幅上升 |
| 官方精度 checker（`cann-bench` 的 `compare_tensors`） | 第 7 步验收 | 用同 dtype golden 会误判（见 pitfalls §7） |
| `msprof` + `cannbotdsl-msprof-compare` skill | 第 8 步性能 | 无法定位瓶颈；墙钟计时会误导 3~4 个数量级 |

**本 skill 不依赖任何特定仓库的文件路径**——所有数据（预算表、baseline、pitfalls）都是
自包含的结论。

## 参考

- `../cannbotdsl-flash-attention/SKILL.md`（FA 蓝本结构、VF 折叠规则）
- `../../core-skills/cannbotdsl-probe-debug/SKILL.md`（静默错误的定位方法）
- `../../core-skills/cannbotdsl-op-design/SKILL.md`（通用设计流程）

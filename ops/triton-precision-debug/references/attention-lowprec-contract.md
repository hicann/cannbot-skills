---
schema_version: okf.v1
kind: field_note
type: precision_card
source_family: curated
category: precision
title: "Attention 低精度逐位契约 — 参考实现算术路径复刻"
description: "FA 类算子在 fp16/bf16 下必须逐位复刻参考实现的算术路径，四条契约与 1-ulp 诊断法。"
phenomenon: bit_exact_drift
signal:
  - 失败集合严格等于「全部 fp16/bf16 case」，fp32 全过
  - matched_ratio 0.60~0.88，max_diff 很小但越阈
  - 实现「算得更准」反而不过
solution_layer: precision_strategy
severity: high
confidence: verified
reproduce_count: 2
tags: [precision, numerical_stability]
keywords: [1-ulp, matched_ratio, aclnnDivs, ELEM_TY, round_e, online_softmax, propagate_nan]
created_at: '2026-08-11T00:00:00Z'
updated_at: '2026-08-11T00:00:00Z'
---

# Attention 低精度逐位契约 — 参考实现算术路径复刻

> FA / MHA / SDPA 类算子在 fp16 / bf16 下的精度闸门。**性能优化前必须先过这一关**——
> 精度不过时做的所有性能结论都会在契约变化后作废。
> 与 [`divide-scale-calibration.md`](divide-scale-calibration.md) 同根因（NPU 设备除法非正确舍入），
> 但那篇解决的是"把 NPU 除法结果取回 host 当查找表"，本篇解决的是"整条算术路径的逐位复刻"。

---

## 1. 现象

- 失败集合**严格与 dtype 相关**：全部且仅仅是 fp16/bf16 case 越阈，fp32 case 一个不错；
- `max_diff` 数值很小，但 `matched_ratio`（1-ulp 匹配率）只有 0.60~0.88，判据要求 ≥0.90；
- 最刺眼的一条：**实现"算得更准"反而不过**——把中间量都升到 fp32 计算，匹配率反而下降。

> **判据推论**：只要失败集合严格按 dtype 划分，基本可直接判为**算术路径不匹配**，
> 不用去搜 tiling、不用改 BLOCK。这一条能省掉整整一轮试错。

---

## 2. 根因

判据是**按 ulp 定的**：`rel_threshold` 恰为 1 个 dtype ulp、要求 `matched_ratio ≥ 0.9`，
即"≥90% 元素与参考相差 1 ulp 以内"。

- 当参考实现的 attention 写作 `q.float() @ k.float().T`（在 fp32 上算）时，误差预算宽松，实现只要"算得准"就行；
- 当参考实现在**原生 dtype** 上算时，**"算得更准"是错误的**——必须逐位复刻参考的低精度算术路径。

**误差放大链要算清楚**：`scores` 的 1 ulp 绝对误差经 `exp` 变成同等**相对**误差。
fp16 在 `|s|~10` 处的 ulp 是 0.0078 → 权重 0.78% 误差，而判据只允许 0.05%。
⇒ **`scores` 必须逐位一致，1 ulp 都不能差。** 这条决定了值不值得继续深挖。

---

## 3. 解决配方：四条契约（逐条实测钉死）

| # | 规则 | 怎么发现的 | 不遵守的后果 |
|---|------|-----------|-------------|
| 1 | `matmul(q, kᵀ)` 输出**舍回 `ELEM_TY`** | `aclnnMatmul` = 原生操作数 + fp32 累加 + **输出原生 dtype** | 30/50 越阈 |
| 2 | scale 用 **host 侧 fp32 倒数做乘法**，不要在设备上做除法 | `aclnnDivs = round_e(fp32(t)/fp32(round_e(scalar)))`；Ascend 设备 fp32 除法是 Newton 迭代、**非正确舍入** | `scores` 差 1 ulp → 经 `exp` 放大成 ~0.8% 权重误差，判据只允许 0.05% |
| 3 | fp16 下 `s - max` 在 **`ELEM_TY`** 上做，且**显式 cast 回 `ELEM_TY`** | 1-ulp 匹配率：fp32 减法 0.7683 → `ELEM_TY` 减法 **1.0000** | ⚠️ Triton 把 `fp16 - fp16` **隐式提升到 fp32**，不显式 cast 等于没做 |
| 4 | 权重**先归一化再舍入**：`w = round_e(exp(·) / l)` | 参考是 `w = softmax(s)`（dtype）→ `matmul(w, v)` | 顺序反了 `matched_ratio` 掉到 0.60~0.88（两次独立验证） |

### after（kernel 骨架）

```python
s_r    = tl.cast(tl.dot(q, tl.trans(k)), ELEM_TY)              # 规则 1
scores = tl.cast(tl.cast(tl.cast(s_r, tl.float32) * inv_sqrt,  # 规则 2（inv_sqrt 是 host 侧倒数）
                         ELEM_TY), tl.float32)
scores = tl.where(valid, scores, -3.0e38)                      # 掩码用有限极小值，见 §5
m      = <行最大>
d_e    = tl.cast(tl.cast(scores, ELEM_TY) - tl.cast(m, ELEM_TY)[:, None], ELEM_TY)   # 规则 3
w      = tl.cast(tl.exp(tl.cast(d_e, tl.float32)) * inv_l[:, None], ELEM_TY)         # 规则 4
o     += tl.dot(w, v)
```

host 侧的 `inv_sqrt`：

```python
sq = math.sqrt(head_dim)
if dtype == torch.float16:      # fp16 下 aclnnDivs 会把标量先舍入到 fp16；bf16 不会
    sq = float(torch.tensor(sq, dtype=torch.float16))
inv_sqrt = 1.0 / sq
```

### before（典型错法）

```python
scores = tl.dot(q, tl.trans(k)).to(tl.float32) / tl.sqrt(tl.cast(Dh, tl.float32))  # 设备除法
d_e    = scores - m[:, None]                                                       # 隐式提升 fp32
w      = tl.exp(d_e)                                                               # 先舍入后归一化
```

---

## 4. `tl.dot` 的 dtype 契约（与性能强相关，不要凭直觉改）

| 量 | 处理 | 理由 |
|----|------|------|
| q / k（从低精度内存加载） | ✅ **保持原生 dtype + fp32 累加器**（`tl.dot` 默认 `out_dtype=fp32`） | 本就只有 11(fp16)/8(bf16) 位有效位，升 fp32 一位不增，反而走异构路径 |
| **p（softmax 输出）** | ❌ **不能直接降到输入 dtype**（~11 bit） | 是真 fp32 量，降精度后相对误差 ~5e-4，直接顶穿 fp16 的阈值 9.77e-4。实测 50 个 case 里 27 个越阈，**失败集合严格等于全部 fp16/bf16 case** |
| p（**二段拆分**，~22 bit） | ✅ 低精度输入下可行，且带来 +2.7% 性能 | 见 §4.1 |
| V | fp16/bf16 输入下**不要升 fp32** | 升 fp32 不增信息，只占 UB（64x128 时 64KB，是 UB 里最大的一块） |

> ⚠️ **fp32 输入下不要禁用 `tl.dot`**。流传的"FP32 路径禁用 `tl.dot`"指的是**把操作数升 fp32 后再 dot**的写法；
> 原生 dtype 操作数 + fp32 累加器在 fp32 输入下实测精度 50/50 全过，且是最大的一笔性能收益。两者是不同的事。

### 4.0 ★ 哪些 fp32 是"过度对齐"（隔离实验结论）

参考实现常写作 `torch.matmul(q.float(), k.float().T)`，直译成 `tl.cast(q, tl.float32)` 是**源码层面最忠实、数值上最无用**的做法。
第三个算子（MQA）用隔离实验把两处分开测：

| 位置 | 判决 | 证据 |
|---|---|---|
| **Q@Kᵀ 升 fp32** | **过度对齐，纯浪费** | 去掉后精度**仍 50/50**，性能 **+22%** |
| **P@V 用 fp32** | **真需要** | 换 fp16 二段拆分 → 50/50 掉到 **26/50** |

**分界线：操作数是否携带真实的 fp32 有效位。** q/k 来自低精度内存，升 fp32 一位不增；`p` 是 softmax 输出，是真 fp32 量。

⚠️ 同一算子上，`P@V` 改原生 dtype 的完整代价/收益是**精度 22/50、性能仅 +5.3%**——
**精度余量在这类算子上只值约 5% 的性能，不是性能约束所在**，不要为这 5% 冒险。

⚠️ **必须单变量测**：某次把 Q@Kᵀ 与 P@V 两处改动捆绑提交，结果 26/50，无法归因；
拆开才发现前者精度不掉且 +22%、后者才是元凶。

### 4.0.1 其它数值对齐要点

- `scale` 是否保持除法形式，取决于参考实现——见主卡「跨算子结论冲突的统一判据」第 1 条；
- `sqrt` 在 **host 侧算好传入**，不要 kernel 内 `tl.sqrt`；
- `exp2` 替换 `tl.exp`：与 torch 的 `exp` **不 bit-equivalent**（文档零提示），且在 Ascend 上**连性能都不赚**（1.2527 vs 1.2591，反而略慢）。**双重不成立，用 `tl.exp`。**

### 4.1 `p` 二段拆分的余量论证（可用来预估会不会掉精度）

| dtype | 判据阈值 | 正确实现本身已占 | p 拆分后 | 余量 |
|-------|----------|------------------|----------|------|
| fp16 | 2⁻¹⁰ | ~2⁻¹² | 2⁻²² | 足够 |
| bf16 | 2⁻⁸ | ~2⁻¹² | 2⁻²² | 足够 |
| fp32 | 最紧 | — | 2⁻²² vs fp32 的 2⁻²⁴ | **不够，不要开** |

```python
p1 = p.to(in_dtype); p2 = (p - p1.to(tl.float32)).to(in_dtype)
pv = tl.dot(p1, V_tile) + tl.dot(p2, V_tile)     # ~22 bit，两次同构 dot
```

⚠️ **该结论在不同算子上冲突过**（一个算子的记载是"二段 fp16 拆分也不够"，另一个实测 50/50 全过）。**必须重测，不能照抄。**

---

## 5. 掩码用有限极小值，不用 `-inf`

`-inf` 会在「整行都被掩掉」时产生 `exp(-inf - (-inf)) = NaN`，并且会逼着 `tl.maximum` 保留 NaN 处理
（与性能项 `propagate_nan=ALL` 直接冲突）。用 `-3.0e38` / `-1e30`：`exp(-3e38 - m_finite)` 直接下溢成 0，语义相同且无 NaN。

**安全性论证模板**（必须写进注释）：

- causal + online softmax：KV 循环从 0 开始，第一块必含 `kv ≤ row` 的合法列 ⇒ 每行第一次更新时 `m` 就是有限值，后续整块被掩的行 `p = exp(-3e38 - m) = 0`，不污染；
- 可变 window：对 causal/window 的每种组合逐一验证区间非空；越界行（`offs_q >= S_Q`）可能整行被掩，其 NaN 在 `tl.store(mask=...)` 时被丢弃且不跨行传播。

---

## 6. 定位手法（比试错快一个数量级）

1. **用 1-ulp 匹配率，不要用 MERE。** 朴素 MERE 的分母对接近零的 golden 会爆掉，得到 `1e+01` 这种无意义的数；判据本身也是按 1 ulp 定的，直接对齐它：
   ```python
   matched = ((a - b).abs() <= (b.abs() * 2**-mant).clamp(min=tiny)).float().mean()
   ```
2. **在 torch 里同时搭「参考路径」和「我的等价路径」，逐阶段替换。** 靠它把范围从"整个 kernel"缩到"softmax 的减法"和"scale 的除法"两处，**全程不用重编 kernel**。
3. **算子级探针**：写 10 行的 Triton probe kernel，把 `tl.dot` / `tl.exp` / 除法逐个与 aclnn 对拍逐位一致性。由此可排除 dot（含 padding，逐位同）和 exp（逐位同），把嫌疑锁死在除法上。
4. **快速自查不替代正式判定**：扫参阶段用单 case 脚本算 `max|diff|` 秒级淘汰明显错误的变体；**正式结论一律走 `verify.py`，不改阈值、不用 `--verify_not_required`。**

---

## 7. 结构代价：低精度契约会倒逼「多趟」

契约 4 要求 `w` 用**最终的 `l`** 归一化后再舍入，而 online softmax 只能拿到滚动的 m/l ⇒ 必须多趟：

- `S ≤ BLOCK_KV`（单块）：**一趟**，scores 常驻片上；
- `S > BLOCK_KV`：**两趟**（趟 1 在线求 m 与 l，趟 2 归一化 + PV），共 2 遍 QK dot。

各步收益：三趟 → 单块单趟 **+21.9%**；多块三趟 → 两趟 **+10.1%**。

⚠️ **不要试图用「减小 `BLOCK_Q` 换单趟」**：总迭代数 = `ceil(S/BQ)·ceil(S/BKV)·趟数`，
BQ 减半使第一项翻倍、趟数 2→1，**正好抵消**。已算账否决。

---

## 8. 失效边界

- 参考实现的 attention 显式写作 `.float()` 时，本篇四条契约**不适用**（那时应保持 fp32 归约）。
- **参考实现的算术路径一旦变化，精度类结论全部作废，性能类结论也要重测。** 实证：某算子上游一次提交把参考从 fp32 改回原生 dtype，旧契约下 3.6766 的交付版对新参考**只过 20/50**，整轮返工。
- ⇒ 生成新算子时**先用 §6 的手法把参考的算术路径钉死，再开始优化**。顺序反了代价是整整一轮。

---

## 9. 坏实践

- 遇到低精度越阈就去搜 tiling / 调 BLOCK：失败集合按 dtype 划分时，这是纯浪费。
- 把中间量都升到 fp32"提高精度"：参考在低精度上算时，这会让匹配率下降。
- 为了过判据而放宽阈值或加 `--verify_not_required`：性能数据随即失去意义。

<!-- okf:related:start -->

## 相关

- 同根因·不同配方: [NPU 浮点除法 scale 校正](divide-scale-calibration.md) — 都是"设备除法非正确舍入"，那篇解 host 查找表，本篇解整条算术路径复刻
- 性能侧对应: [Attention / FlashAttention 类算子优化点库](../../triton-latency-optimizer/references/operators/flash-attention-optimization.md) — 本篇是其 §4 精度闸门的展开

<!-- okf:related:end -->

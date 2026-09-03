---
name: sampling-top-p-filter
description: Sampling Top-P Filter（核采样过滤）算子的 Triton Ascend 优化经验
metadata:
  type: reference
---

# Sampling Top-P Filter 算子类别经验（核采样过滤）

> 类别: `sampling-top-p-filter` — Top-P / 核采样过滤算子
> 锚定算子: `_top_p_filter_kernel` (op25)
> 性能参考: Ascend950PR_9579
> - 几何平均加速比: **1.297×**（目标 1.2×，**已达** ✅）
> - 精度: 全 case 通过（`sorted_topk_indices` 逐元素与 torch 参考比对）

---

## Layer 1: 设计约束（硬性规则）

### L1.1 索引必须与 `torch.topk` 精确一致（强制）
- **规则**: wrapper 中保留 `torch.topk(logits, top_k, dim=-1)`，不要在 kernel 内实现自定义 top-k，除非能证明对测试分布输出顺序完全一致。
- **原因**: mojo_opset 精度测试将 `sorted_topk_indices`（int64/int32）与 torch 参考逐元素比对。

### L1.2 数值归约必须 fp32（强制）
- **规则**: logits 在进入 top-k 前由 wrapper cast 到 fp32；kernel 内所有 softmax 归约（max, exp, sum）必须保持 fp32。
- **原因**: fp16/bf16 中间精度导致 cumsum 累积误差使 remove mask 错位。

### L1.3 移除规则必须与 torch 参考一致（强制）
- **规则**: 对已排序的 top-k 概率 `p` 及 inclusive cumsum `c`，keep/remove mask 为：
  ```
  remove_i = ((c_i - p_i) > top_p) & (i >= min_tokens_to_keep)
  ```
  其中 `c - p` 是 exclusive cumsum。这一行闭式写法替代 torch 参考的三步 mask 操作；任何偏离都会改变参考语义。
- **禁止**: 自行推导等价的 mask 公式（看似等价实则浮点舍入路径不同）。

### L1.4 forward 内禁止 PyTorch 回退（强制）
- **规则**: Triton kernel 必须完成 filter 和两次 softmax；`F.softmax` / `masked_fill` 不得出现在热路径。
- **原因**: kernel 调用往返开销会吃掉所有性能收益。

### L1.5 BLOCK_K 补零为可选（非强制）
- **规则**: 提交版 wrapper 直接传 `TOP_K=top_k`，分配 `output_probs[B, top_k]`，kernel 内用 `tl.arange(0, TOP_K)` 无需额外 mask。
- **若选择** pad 到 next_pow2：加 `BLOCK_K: tl.constexpr` 和 `mask_k`，保持 wrapper/output tensor 一致。

---

## Layer 2: 算法骨架（逐行双 pass softmax）

```
grid = (batch_size,)

每个 program（一行）:
    offsets = tl.arange(0, TOP_K)
    load sorted_logits[TOP_K]
    p = softmax(sorted_logits)            # pass 1: logits → 概率
    c = inclusive_cumsum(p)
    remove = ((c - p) > top_p) & (offsets >= min_keep)
    filtered = where(remove, filter_value, sorted_logits)
    out = softmax(filtered)               # pass 2: 过滤后重新归一化
    store out[TOP_K]
```

- **按 batch 行分 program 是自然粒度**: cumsum 沿 K 轴向串行（已知 workload 中 K ≤ 1024），拆分 K 跨 program 会引入同步开销，收益为零。
- **不要**把 K 维拆到多个 program。

---

## Layer 3: 关键代码片段（技巧可参考，变量名/结构须重设计）

### L3.1 Kernel 签名（HEAD 版本，无 padding）

```python
@libentry()
@triton.jit
def _top_p_filter_kernel(
        sorted_logits_ptr,
        output_ptr,
        top_p,
        filter_value,
        min_tokens_to_keep,
        stride_logits_b,
        stride_logits_k,
        stride_out0_b,
        stride_out0_k,
        TOP_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_logits_ptr = sorted_logits_ptr + pid * stride_logits_b
    row_out_ptr = output_ptr + pid * stride_out0_b
    offsets = tl.arange(0, TOP_K)
    logits = tl.load(row_logits_ptr + offsets * stride_logits_k)

    x_max = tl.max(logits, 0)
    e = tl.exp(logits - x_max)
    p = e / tl.sum(e, 0)

    c = tl.cumsum(p, 0)
    remove = ((c - p) > top_p) & (offsets >= min_tokens_to_keep)

    filtered = tl.where(remove, filter_value, logits)
    f_max = tl.max(filtered, 0)
    fe = tl.exp(filtered - f_max)
    f_p = fe / tl.sum(fe, 0)

    tl.store(row_out_ptr + offsets * stride_out0_k, f_p)
```

### L3.2 `@libentry()` 减少 per-call launch 开销

小 B case（B=15/18）有微弱正向收益，大 B case 影响可忽略。

### L3.3 自适应 topk 分派（1.2× 达标的关键杠杆）

**最大性能杠杆不在 kernel 内部，而在 wrapper 如何获取排序后的 top-k**。Ascend 上 `torch.topk(..., sorted=True)` 内部做全排序开销大，`sorted=False` 跳过排序。按 `top_k` 大小分支：

```python
if top_k >= 512:
    # 大 k：额外排序 k 个元素的开销 > 省下的排序 → 直接用 sorted topk
    sorted_logits, sorted_topk_indices = torch.topk(logits_f, top_k, dim=-1)
else:
    # 小 k：unsorted topk + 对小 k 个元素排序
    vals, idx = torch.topk(logits_f, top_k, dim=-1, sorted=False)
    sorted_logits, order = torch.sort(vals, dim=-1, descending=True)
    sorted_topk_indices = torch.gather(idx, dim=-1, index=order)
```

- 两条路径产生**完全一致**的降序 (values, indices) —— topk 无论 `sorted` 取值都返回相同 k 个值；`sort` + `gather` 仅恢复顺序，bit-for-bit 等价。
- 分界点 512 是当前 shape（k ∈ {100, 1000}）的经验值；workload 的 k 分布变化时需重新测定。
- 配合 `sorted_logits` 原地写输出，几何平均从 ~1.14×（仅 sorted=True）提升到 **1.297×**。

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| 索引精度失败 | `sorted_topk_indices` 与 torch 参考逐元素比对 | L1.1 wrapper 保留 `torch.topk`；sorted=False 时 `sort`+`gather` 恢复顺序（L3.3） |
| 概率精度失败 | remove mask 偏移公式错误 | L1.3 闭式公式，勿自行推导 |
| 1.2× 目标未达 | `torch.topk` 占总延迟大头 | 先用自适应 topk 分派（L3.3），再考虑 fused kernel |
| `sorted=False` 反而更慢 | 大 k 下额外排序 k 个元素代价高 | 分支: `top_k >= 512` 时用 `sorted=True`（L3.3） |

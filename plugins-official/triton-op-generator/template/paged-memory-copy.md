---
name: paged-memory-copy
description: Paged Memory Copy 类算子（StorePagedKVCache / Scatter-Copy）的 Triton Ascend 优化经验
metadata:
  type: reference
---

# Paged Memory Copy 类算子优化经验

> Anchored by `_store_paged_kv_cache_chunk_kernel` exploration (op19, 2026-07-29; revisited 2026-07-30).
> Performance reference on Ascend950PR / `ascend910_9382`:
> - 12 accuracy cases, all passed
> - **2026-07-29 (scalar-loop variant)**: geomean `base_us / gen_us` = **1.0×** vs stored `基线性能/` (target 1.2×, not reached)。
> - **2026-07-30 (vectorized 2D-tile CS32 variant)**: geomean = **1.80×** vs the ORIGINAL kernel **re-measured on the same box** (CANN profiler, kernel-only, active=5). Target 1.2× **reached** on the fair on-HW comparison.
> - ⚠️ The stored `基线性能/` numbers are **NOT reproducible on this Ascend950PR box** (collected on faster/different HW): re-measuring the original kernel fresh is 4–8× SLOWER than stored for large-prefill cases (e.g. val3 fresh 29.5µs vs stored 3.6µs; geomean stored/fresh = 0.536×). **Always compare gen vs an on-box re-measurement of the original, not the stored numbers.**

---

## §0 适用范围与算子分类

| 算子 | 类别 | 计算特征 | 优化哲学 |
|------|------|---------|---------|
| StorePagedKVCache | `paged-memory-copy` | 按预计算元数据 scatter-copy K/V token 到 paged cache | 2D tile 向量化 + 连续 head_dim 访存 |

> ⚠️ **关键区分**：Paged-memory-copy 是 memory-bound scatter 类算子。与连续搬运（Pad/Split）不同，它使用预计算索引进行离散写入，优化方向应是减少 launch 开销和改善访存模式，而非简单的循环展开。

---

## §1 通用经验

### Layer 1: 设计约束（硬性规则）

1. **禁止 constexpr-unrolling 标量 scatter 循环**：
   - 对于含离散地址计算的 scatter-copy 循环，`tl.static_range(N)` 展开会产生大量代码膨胀和寄存器压力
   - 实测 8-way unrolling 导致 **6-10×（约 10×）性能退化**（20.9µs vs 2.4µs）
   - 标量循环（`off += 1`）是 Triton Ascend 编译器最擅长的模式（但见 L3 结论：950PR 上应改用 2D tile）

2. **禁止在 scatter-copy kernel 上使用 compiler hints**：
   - `multibuffer=True` 引发额外 buffer 分配/copy（TensorMove），**实测增加 ~30% 开销**（0.9×）
   - `unit_flag=True` 无实际收益
   - 这些 hint 为计算密集/连续搬运设计，不适用于离散 scatter-copy

3. **Precomputed base pointers 无实际收益**：
   - 编译器已自动执行循环不变量外提，手动预计算 base pointer 与编译器生成的代码等效（零可测差异）

4. **Grid 配置需保守**：
   - `grid = max(1, min(num_cores, num_chunks))` 对小 workload（num_chunks < 10）可能过度并行
   - 对于 num_chunks ≤ 8 的场景，考虑 `grid = (1,)` 减少 launch 开销

### Layer 2: 算法骨架

```python
@triton.jit
def scatter_copy_kernel(
    src_ptr, dst_ptr, meta_ptr, num_items,
    stride_src_tok, stride_src_head, stride_src_dim,
    stride_dst_blk, stride_dst_head, stride_dst_tok, stride_dst_dim,
    head_dim: tl.constexpr,
    num_heads,
):
    """1D grid, strided program loop over (item, head) pairs.
    Each item processed with scalar token loop + vectorized head_dim access."""
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    idx = pid
    while idx < num_items:
        item_idx = idx // num_heads
        head_idx = idx % num_heads

        # Load metadata
        row = meta_ptr + item_idx * stride_meta_row
        src_start = tl.load(row + 0)
        dst_block = tl.load(row + 1)
        dst_offset = tl.load(row + 2)
        item_len = tl.load(row + 3)

        # Scalar loop: one token per iteration
        off = 0
        while off < item_len:
            src_tok = src_start + off
            dst_tok = dst_offset + off

            # Vectorized head_dim access (coalesced)
            dim_idx = tl.arange(0, head_dim)
            src_addr = src_ptr + src_tok * stride_src_tok + head_idx * stride_src_head + dim_idx * stride_src_dim
            dst_addr = dst_ptr + dst_block * stride_dst_blk + head_idx * stride_dst_head + dst_tok * stride_dst_tok + dim_idx * stride_dst_dim

            val = tl.load(src_addr)
            tl.store(dst_addr, val)

            off += 1

        idx += num_programs
```

### Layer 3: 关键技巧与已验证变体

#### 3.1 已验证变体对比（孰优孰劣一图定）

| 变体 | 实测结果 | 结论 |
|------|---------|------|
| 8-way constexpr-unrolled（`tl.static_range(8)` + `if off+t < len`） | 0.15×（20.9µs vs 2.4µs） | **禁止**（L1.1） |
| compiler hints（`multibuffer/unit_flag`） | 0.9×（+30% TensorMove） | **禁止**（L1.2） |
| 标量循环 + head_dim 向量化（L2 骨架） | vs 存档基线 1.0×；但 **on-box 实测灾难性慢**（163-token case 14.7µs vs tile 版 2.3µs，0.16×） | 950PR 上被 2D tile 取代（2026-07-29 的"1.0×"是对不可复现存档基线测的，见头部 ⚠️） |
| **2D tile `[CHUNK_SIZE, head_dim]`，CS32**（3.2） | **1.80×**（vs on-box 重测 original） | **当前最优** |

> 经验法则：never unroll scatter-copy loops with per-token address computation；on Ascend950PR 优先 2D tile 而非 per-token 标量循环。

#### 3.2 Vectorized 2D tile over [CHUNK_SIZE, head_dim]（Ascend950PR 最优）

- Work item = (logical_chunk, head)；1D grid `min(num_vectorcore, num_items)`，strided `while`。
- 每 work item：scalar-load 4 个 chunk_metadata ints，然后 token 维 tiled `[CHUNK_SIZE, head_dim]` 2D masked load/store，动态 `while off < chunk_len: off += CHUNK_SIZE`；K 和 V 在同一 program 内共享同一 tile/索引算术。
- **Tile 宽度关键**：在 Ascend950PR 上 sweep `{8,16,32,64,128}`，**CHUNK_SIZE=32 是最佳固定折中**（`chunk_size = min(block_size, 32)`，须保持 power-of-two 以配合 `tl.arange`）：
  - 128/64：长 prefill case 尚可（≈1.0–1.1×），但**短/decode chunk（chunk_len≈1）浪费宽向量 → 0.80–0.90×**。
  - 8/16：短 chunk 表现好，但**长 chunk 崩溃（0.48–0.64×）**。
  - 32：长 case 1.09–1.10×，短 case ≈0.90×，**geomean 最佳**。
- 大 prefill case 驱动 geomean：tiled kernel 在这些 case 上比 on-box original 快 **4.3–8.5×**；small/decode case 上 original 略快（0.79–1.08×，绝对差 ~0.3µs）。

---

## §4 常见陷阱与避免方法

| 陷阱 | 原因 | 避免方法 |
|------|------|---------|
| Constexpr-unrolling scatter 循环 | 地址计算复制导致寄存器压力 | L1.1（10× 退化） |
| scatter-copy 上用 compiler hints | multibuffer 额外 buffer copy | L1.2（+30% 开销） |
| 手动预计算 base pointers | 编译器已自动外提 | L1.3（零收益） |
| 小 workload 过度并行 | num_items < 10 时 launch 开销主导 | L1.4 考虑 `grid=(1,)` |
| 保留未使用的 constexpr 参数（CHUNK_SIZE/num_subchunks） | 设计残留 | 不针对它优化 |
| 950PR 上用 per-token 标量循环 | 标量 `off += 1` 比向量化 tile 慢 6× | L3.1/3.2 改用 2D tile |
| Tile 宽度选错 | 128/64 短块浪费宽向量；8/16 长块崩溃 | L3.2 sweep 后取 `min(block_size, 32)` |
| 直接采信存档 `基线性能/` 数字 | 存档采集自更快/不同硬件 | 头部 ⚠️：与 on-box 重测的 original 对比 |

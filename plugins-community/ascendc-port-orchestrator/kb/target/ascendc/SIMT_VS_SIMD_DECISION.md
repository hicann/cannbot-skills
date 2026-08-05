# SIMT vs SIMD Decision Framework

> When to use SIMT, when to use SIMD, and why. Based on verified production experience
> with Sparse-Gather (E8-E14) and MXFP4 quantization (2026-04-07).
>
> **This is the authoritative reference for P-P9 (SIMT vs SIMD choice).**

## Quick Decision Table

| 算子特征 | SIMT | SIMD | 原因 |
|----------|:----:|:----:|------|
| atomicAdd / scatter-write | ✅ | ❌ | SIMD 无 atomicAdd |
| 间接寻址 (arr[index[i]]) | ✅ | ❌ | SIMD DataCopy 不支持间接地址 |
| 连续读 + 向量计算 (Add/Muls/Cast) | ❌ | ✅ | MTE2/VEC pipeline 重叠 1.6-2.3x |
| **Group-local 计算 (group_size < tile_size)** | ✅ | ❌ | **SIMD per-group 循环慢于 SIMT 并行** |
| Per-element 异构操作 (不同 shift/branch) | ✅ | ⚠️ | SIMD 可用 Compare+Select，但开销大 |
| 纯位操作 (reinterpret float↔int) | ⚠️ | ⚠️ | 两者都可，看并行度 |
| 批量数据搬运 + 简单计算 | ❌ | ✅ | MTE2 DMA 带宽 >> dcache |

## Decision Tree (完整版)

```
Step 1: 是否有 atomicAdd / scatter-write?
  YES → SIMT (SIMD 无法做原子操作)
  NO  → Step 2

Step 2: 是否有间接寻址 (arr[index[i]])?
  YES → SIMT (SIMD DataCopy 需要连续地址)
        例外: 如果 index 有序且可 batch，考虑 sort-to-reuse (P-P24)
  NO  → Step 3

Step 3: 计算是否有 group-local 依赖?
  (即: 每 N 个元素共享一个参数，N < tile_size)
  YES → Step 3a
  NO  → Step 4 (所有元素独立，纯 element-wise)

Step 3a: group_size >= 256?
  YES → SIMD (group 足够大，per-group 循环开销可接受)
         使用 TQue<VECIN,4> pipeline + per-group VEC 操作
  NO  → Step 3b

Step 3b: per-group 计算是否全部可向量化?
  (即: group 内每个元素执行完全相同的指令序列)
  YES → SIMD 可能有优势，但需验证:
         - 如果 group=32, tile=1024: 需要 32 次 per-group 循环
         - SIMT 128 threads 并行 128 groups = 4096 elements/dispatch
         - 经验: MXFP4(group=32) SIMD V3 比 SIMT 慢 6x
         → **默认 SIMT，除非 profiling 证明 SIMD 更快**
  NO  → **SIMT** (per-element 异构计算，SIMT 线程并行最优)
         例: MXFP4 per-element x_exp + variable shift → SIMT 快 6-20x

Step 4: 计算是否可用 SIMD 向量指令表达?
  (Abs, Add, Muls, Cast, Compare+Select — 同一操作应用到所有元素)
  YES → **SIMD** (MTE2/VEC pipeline 重叠)
         例: SG forward (DataCopy + Muls + Add) → SIMD TQue 1.6-2.3x
  NO  → **SIMT** (需要 per-element 分支/位操作)
```

## ⚠️ 精度约束 (OL-30)

**SIMD 优化不能以精度降级为代价。**

常见陷阱: 为了 SIMD 性能，将 per-group 操作改为 tile-wide 操作。
- ❌ MXFP4: 1024 元素共享一个 exponent（spec 要求每 32 个）→ **精度 bug**
- ❌ A3 手写 SIMD: BATCH=512 共享 exponent → **精度 bug（已确认）**
- ✅ SG forward: per-token 计算，每个 token 独立 → 无精度问题

**规则**: 如果优化改变了算法的 group/block precision 语义，
则不能作为 production kernel。必须标注 "approximate" 并提供精确版本。

## Verified Case Studies

### Case 1: Sparse-Gather Forward (SIMD wins)

| 特征 | 值 |
|------|---|
| 计算 | DataCopy + Muls + Add (全向量化) |
| Group 依赖 | 无 (per-token 独立) |
| 数据访问 | 连续 (expert embedding) |
| **结果** | SIMD TQue<4> **1.6-2.3x** faster than PipeBarrier |

为什么 SIMD 赢: 每个元素执行完全相同的 Muls+Add，无 group-local 依赖，MTE2 prefetch 有效。

### Case 2: Sparse-Gather Backward Sorted (SIMD wins)

| 特征 | 值 |
|------|---|
| 计算 | DataCopy + Muls + Add (累加到 TQue<VECOUT>) |
| Group 依赖 | per-expert 累加，但 expert run 够长 |
| 数据访问 | 连续 |
| **结果** | SIMD TQue **1-10%** faster, 7 PipeBarrier→0 |

为什么 SIMD 赢（但幅度小）: scalar pipe 是真瓶颈 (0.648)，MTE2/VEC 重叠收益有限。

### Case 3: MXFP4 Quantization (SIMT wins)

| 特征 | 值 |
|------|---|
| 计算 | per-element log2 + floor + pow2 + round (异构) |
| Group 依赖 | **per-32-element shared exponent** |
| 数据访问 | 连续但 group-local |
| **结果** | SIMT **6x** faster than SIMD V3, **20x** faster than SIMD V1 |

为什么 SIMT 赢:
1. group_size=32 太小 → SIMD 需要 per-group 循环 → 串行
2. SIMT 128 threads 并行处理 128 groups (4096 elements) → 天然并行
3. 消除 per-group 循环 (SIMD V4 fast) 性能提升但 **精度不符 spec**
4. Per-element x_exp 导致 Compare+Select 开销（SIMD V3），SIMT 线程各自独立计算无开销

### Case 4: MXFP4 Tile-Wide Approximate (SIMD wins, but wrong precision)

| 特征 | 值 |
|------|---|
| 计算 | tile-wide Abs + Muls + Cast(FLOOR) + Select |
| Group 依赖 | **无** (放弃了 per-group，改为 tile-wide) |
| 精度 | ⚠️ **不符合 MXFP4 spec** |
| **结果** | SIMD **1.08x** faster than SIMT on 4K |

为什么 SIMD 赢: 消除 per-group 循环后，全部是 tile-wide SIMD 向量操作，MTE2 pipeline 生效。
**但不可用于 production** — 精度降级不可接受。

## Pattern P-P9 Summary

**默认选择**:
- 有 atomicAdd / 间接寻址 → **SIMT**
- 有 group-local 依赖 (group < 256) → **SIMT**
- 连续读 + 纯向量计算 + 无 group 依赖 → **SIMD**

**验证必须**:
- 选择后必须 A/B benchmark 确认（OL-27: 同 NPU 同 session）
- SIMD 优化不得降低精度（OL-30）
- 如果 SIMD 比 SIMT 慢 → 直接用 SIMT，不需要继续优化 SIMD

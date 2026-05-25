---
name: catlass-op-perf-tune
description: "Tune CATLASS kernel performance by adjusting TileShape, DispatchPolicy, Swizzle, and Kernel type parameters. Change one variable at a time for attribution. Use when optimizing CATLASS kernel performance, analyzing profiler bottlenecks, or exploring tiling configurations."
---

# Catlass Kernel Tuning

## Source Code Locations

```
catlass/
├── docs/zh/1_Practice/11_matmul_optimization.md    # ★ 调优主文档
├── docs/zh/2_Design/01_kernel_design/04_matmul_summary.md  # 模板清单
├── docs/zh/2_Design/01_kernel_design/03_dispatch_policies.md  # DispatchPolicy 详解
├── docs/zh/2_Design/01_kernel_design/02_swizzle.md  # Swizzle 策略
├── tools/tuner/                                     # Tiling 自动寻优工具
└── examples/                                        # 可参考的优化配置
```

## Tunable Parameters

| 参数 | 位置 | 影响 |
|------|------|------|
| `L1TileShape` / `L0TileShape` | `using L1TileShape = GemmShape<M,N,K>;` | Buffer 利用率、K-tile 循环次数 |
| `DispatchPolicy` | `using DispatchPolicy = ...;` | 流水调度方式 |
| `BlockScheduler` (Swizzle) | `using BlockScheduler = ...;` | 数据访问顺序 |
| Kernel 类型 | `using Kernel = ...;` | 分核策略（SplitK / SingleCore / Small） |

## Tuning Principles

- **以 catlass 官方优化指南为准**（`11_matmul_optimization.md`）
- 每次**只动一个变量**，便于归因
- 性能下降 → 立即回滚，换方向
- 性能提升 → 记录配置，继续按指南尝试下一项

## Bottleneck Diagnosis

| Profiler 现象 | 瓶颈 | 优先尝试 |
|-------------|------|---------|
| MTE2 占比高、Cube 利用率低 | GM→L1 带宽 | Preload (DispatchPolicy 换 Preload)、ShuffleK |
| Cube 利用率高、Vector 空闲 | 搬运瓶颈 | 调大 L1TileShape K、ShuffleK |
| 任务块 < AIC 核数 | 核利用率不足 | SplitK (Kernel 换 SplitkMatmul) |
| 小 Shape | 标量开销 | SmallMatmul Kernel |
| A 矩阵反复重读 | L1 重复加载 | FullLoadA |
| AIC/AIV 协同空泡 | 同步开销 | 调 workspaceStages 或 DispatchPolicy |

## Code Modification Pattern

```diff
// Pingpong → Preload
- using DispatchPolicy = Gemm::MmadAtlasA2Pingpong<true>;
+ using DispatchPolicy = Gemm::MmadAtlasA2Preload<true>;

// Swizzle offset 0 → 3
- using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<0, 0>;
+ using BlockScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;

// L1TileShape 调大 K
- using L1TileShape = GemmShape<128, 256, 128>;
+ using L1TileShape = GemmShape<128, 256, 256>;
```

改完后重新构建并采集 profiler 数据。

## Never / Always

**NEVER**:
- 未读优化指南就改参数
- 一次修改多个变量
- 忽略硬件资源限制
- 把探测用极简 kernel 留作交付

**ALWAYS**:
- 先读 `11_matmul_optimization.md` 再动手
- 每次只动一个 `using`
- 性能下降立即回滚
- 改动仅限 catlass 拼装类的 `using`

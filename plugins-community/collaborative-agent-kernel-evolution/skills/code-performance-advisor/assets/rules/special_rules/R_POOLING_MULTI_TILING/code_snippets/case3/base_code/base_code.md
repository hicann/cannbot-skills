# Base Code: 简单均分任务,负载不均

来源: lingxi-code (adaptive_max_pool3d_grad, 推断)

```cpp
// Tiling 函数: 简单均分
ge::graphStatus AdaptiveMaxPool3DGradTiling::Tiling(gert::TilingContext* context)
{
    const uint32_t BLOCK_DIM = 16;  // 固定 16 核

    // 问题1: 简单计算总任务数（NC 维度合并,但未优化分配）
    uint32_t totalTasks = nDim * cDim * doDim * hoDim * woDim;

    // 问题2: 简单向下取整,导致负载不均
    uint32_t tasksPerCore = totalTasks / BLOCK_DIM;

    // 问题3: 未处理余数任务
    // 例如: totalTasks = 100, BLOCK_DIM = 16
    // → tasksPerCore = 6, 余数 4 个任务未分配
    // → 前 4 核需要额外处理 1 个任务,但代码未体现

    TilingData.blockDim = BLOCK_DIM;
    TilingData.tasksPerCore = tasksPerCore;

    return ge::GRAPH_SUCCESS;
}

// Kernel 实现: 简单任务分配
extern "C" __global__ __aicore__ void adaptive_max_pool3d_grad(
    GM_ADDR grad_output, GM_ADDR argmax, GM_ADDR grad_input, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t blockIdx = GetBlockIdx();

    // 问题4: 简单任务起止计算
    uint32_t startTask = blockIdx * tiling_data.tasksPerCore;
    uint32_t endTask = startTask + tiling_data.tasksPerCore;

    // 问题5: 最后一个核需要特殊处理,但未考虑所有余数情况
    if (blockIdx == (tiling_data.blockDim - 1)) {
        // 最后一个核处理所有剩余任务
        endTask = tiling_data.totalTasks;
        // 这可能导致最后一个核负担过重
    }

    // 处理任务
    for (uint32_t task = startTask; task < endTask; task++) {
        // 分解任务 ID 到 (n, c, do, ho, wo)
        // ... 处理逻辑
    }
}
```

**问题分析**:

1. **简单除法导致负载不均**
   - `tasksPerCore = totalTasks / BLOCK_DIM` 向下取整
   - 余数任务 `remainder = totalTasks % BLOCK_DIM` 未合理分配
   - 例如: 100 个任务 / 16 核 = 6 余 4
     - 如果余数任务全分给最后一个核: 前 15 核处理 6 个,最后 1 核处理 10 个
     - 最后一核负担是其他核的 1.67×
   - 极端情况: 17 个任务 / 16 核 = 1 余 1
     - 前 15 核处理 1 个,最后 1 核处理 2 个 (2× 负载)

2. **未考虑 NC 合并优化**
   - N 和 C 维度独立处理,未利用它们的数据独立性
   - 合并 NC 维度可以:
     - 简化 Tiling 逻辑（从 5D 降为 4D）
     - 提高并行粒度（NC 组合更多）
     - 更灵活的负载分配

3. **核数不匹配硬件**
   - 硬编码 BLOCK_DIM = 16
   - 在 8 核硬件上浪费资源
   - 在 32 核硬件上利用不足

4. **最后一核负担不确定**
   - 最后一核处理"所有剩余任务"
   - 剩余任务数 = remainder,可能是 0 到 BLOCK_DIM-1
   - 负载方差大,影响整体性能

5. **无法处理任务数少于核数的情况**
   - 如果 totalTasks < BLOCK_DIM (例如 10 个任务 / 16 核)
   - `tasksPerCore = 0`,导致前 6 核空闲
   - 只有最后一核处理所有任务（完全没有并行）

**性能影响**:

- **负载不均衡率**: 最大可达 **2×**（极端情况）
- **平均负载不均衡**: ~**15-30%**
- **整体性能损失**: 由最慢核决定,损失 **10-20%**
- **核利用率**: 可能低至 **50-70%**（任务数少于核数）

**典型问题场景**:

| 场景 | totalTasks | BLOCK_DIM | tasksPerCore | 最后核任务数 | 不均衡率 |
|-----|-----------|-----------|--------------|-------------|---------|
| 场景1 | 100 | 16 | 6 | 10 | **1.67×** |
| 场景2 | 65 | 16 | 4 | 17 | **4.25×** |
| 场景3 | 17 | 16 | 1 | 2 | **2×** |
| 场景4 | 10 | 16 | 0 | 10 | **∞** |

**根本原因**:

传统思维: 简单除法均分,余数交给最后一核处理。这在任务数远大于核数时问题不大,但在:
1. 任务数接近核数
2. 任务数不是核数的整数倍
3. 任务执行时间不均匀

时会导致显著的负载不均衡。

**需要**: 余数任务均匀分配到各核的算法。

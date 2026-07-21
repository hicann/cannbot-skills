# Good Code: 负载均衡 Tiling + NC 合并策略

来源: expert code (adaptive_max_pool3d_grad)

```cpp
// Tiling 类: 负载均衡算法
class AdaptiveMaxPool3DGradScatterTiling {
private:
    MaxPoolGradParams params_;
    uint32_t coreNum_;  // 实际核数

public:
    ge::graphStatus Tiling(gert::TilingContext* context) {
        // 步骤1: 获取实际核数
        coreNum_ = context->GetHardwareInfo().aiCoreNum;

        // 步骤2: 合并 NC 维度（关键优化）
        // 原因: N 和 C 维度的数据完全独立,合并可以提高并行粒度
        int64_t ncDim = params_.nDim * params_.cDim;

        // 步骤3: 计算 NC 维度的分配（核心负载均衡算法）
        int64_t ncRound = 0;        // 向上取整: 前 preCoreNum 个核的任务数
        int64_t ncRoundTail = 0;    // 向下取整: 剩余核的任务数
        int64_t preCoreNum = 0;     // 需要多处理一个 NC 的核数

        CalculateNCDistribution(ncDim, coreNum_, ncRound, ncRoundTail, preCoreNum);

        // 步骤4: 填充 TilingData
        TilingData.ncDim = ncDim;
        TilingData.ncRound = ncRound;
        TilingData.ncRoundTail = ncRoundTail;
        TilingData.preCoreNum = preCoreNum;
        TilingData.blockDim = std::min(static_cast<int64_t>(coreNum_), ncDim);  // 动态核数

        // 步骤5: 计算其他维度的 Tiling
        TilingData.doDim = params_.doDim;
        TilingData.hoDim = params_.hoDim;
        TilingData.woDim = params_.woDim;

        return ge::GRAPH_SUCCESS;
    }

private:
    // 核心算法: 计算 NC 维度的负载均衡分配
    void CalculateNCDistribution(
        int64_t ncDim,          // NC 总数
        uint32_t coreNum,       // 核数
        int64_t& ncRound,       // 输出: 向上取整任务数
        int64_t& ncRoundTail,   // 输出: 向下取整任务数
        int64_t& preCoreNum)    // 输出: 处理 ncRound 的核数
    {
        if (ncDim <= coreNum) {
            // 情况1: NC 数量少于或等于核数
            // 每个核最多处理 1 个 NC,部分核空闲
            ncRound = 1;
            ncRoundTail = 0;
            preCoreNum = ncDim;  // 前 ncDim 个核各处理 1 个,其余核空闲
        } else {
            // 情况2: NC 数量多于核数
            // 需要均匀分配余数任务

            // 向上取整: ceil(ncDim / coreNum)
            ncRound = (ncDim + coreNum - 1) / coreNum;

            // 向下取整: floor(ncDim / coreNum)
            ncRoundTail = ncDim / coreNum;

            // 计算余数
            int64_t remainder = ncDim % coreNum;

            // 关键公式: 前 remainder 个核处理 ncRound 个 NC
            //          后 (coreNum - remainder) 个核处理 ncRoundTail 个 NC
            preCoreNum = remainder;

            // 数学验证:
            // preCoreNum * ncRound + (coreNum - preCoreNum) * ncRoundTail
            // = remainder * ceil(ncDim/coreNum) + (coreNum - remainder) * floor(ncDim/coreNum)
            // = remainder * (floor(ncDim/coreNum) + 1) + (coreNum - remainder) * floor(ncDim/coreNum)
            // = remainder + coreNum * floor(ncDim/coreNum)
            // = remainder + ncDim - remainder
            // = ncDim ✓
        }
    }
};

// Kernel 实现: 负载均衡任务分配
extern "C" __global__ __aicore__ void adaptive_max_pool3d_grad(
    GM_ADDR grad_output, GM_ADDR argmax, GM_ADDR grad_input, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t blockIdx = GetBlockIdx();

    // 关键优化: 根据 blockIdx 和 preCoreNum 计算当前核的任务范围
    int64_t ncIndex = 0;      // 当前核处理的第一个 NC 索引
    int64_t ncRealRound = 0;  // 当前核实际处理的 NC 数量

    if (tiling_data.preCoreNum == 0 || blockIdx < tiling_data.preCoreNum) {
        // 前 preCoreNum 个核: 处理 ncRound 个 NC
        ncIndex = blockIdx * tiling_data.ncRound;
        ncRealRound = tiling_data.ncRound;
    } else {
        // 剩余核: 处理 ncRoundTail 个 NC
        ncIndex = tiling_data.preCoreNum * tiling_data.ncRound +
                 (blockIdx - tiling_data.preCoreNum) * tiling_data.ncRoundTail;
        ncRealRound = tiling_data.ncRoundTail;
    }

    // 边界检查: 确保不超出范围
    if (ncIndex >= tiling_data.ncDim) {
        return;  // 当前核空闲（NC 数量 < 核数的情况）
    }

    // 调整最后一个核的任务数（防止越界）
    if (ncIndex + ncRealRound > tiling_data.ncDim) {
        ncRealRound = tiling_data.ncDim - ncIndex;
    }

    // 处理分配的 NC 任务
    for (int64_t ncOffset = 0; ncOffset < ncRealRound; ncOffset++) {
        int64_t currentNC = ncIndex + ncOffset;

        // 分解 NC 索引到 (n, c)
        int64_t n = currentNC / tiling_data.cDim;
        int64_t c = currentNC % tiling_data.cDim;

        // 处理该 NC 的所有输出点
        for (int64_t do_idx = 0; do_idx < tiling_data.doDim; do_idx++) {
            for (int64_t ho_idx = 0; ho_idx < tiling_data.hoDim; ho_idx++) {
                for (int64_t wo_idx = 0; wo_idx < tiling_data.woDim; wo_idx++) {
                    // 处理 (n, c, do_idx, ho_idx, wo_idx) 的梯度反向传播
                    ProcessGradientScatter(n, c, do_idx, ho_idx, wo_idx);
                }
            }
        }
    }
}

// TilingData 定义: 包含负载均衡参数
BEGIN_TILING_DATA_DEF(AdaptiveMaxPool3DGradTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, blockDim);      // 实际使用的核数
    TILING_DATA_FIELD_DEF(int64_t, ncDim);          // NC 总数
    TILING_DATA_FIELD_DEF(int64_t, ncRound);        // 向上取整任务数
    TILING_DATA_FIELD_DEF(int64_t, ncRoundTail);    // 向下取整任务数
    TILING_DATA_FIELD_DEF(int64_t, preCoreNum);     // 处理 ncRound 的核数
    TILING_DATA_FIELD_DEF(int64_t, cDim);           // C 维度大小（用于分解 NC）
    TILING_DATA_FIELD_DEF(int64_t, doDim);          // 输出 D 维度
    TILING_DATA_FIELD_DEF(int64_t, hoDim);          // 输出 H 维度
    TILING_DATA_FIELD_DEF(int64_t, woDim);          // 输出 W 维度
    // ... 其他参数
END_TILING_DATA_DEF;
```

**改进点分析**:

1. **负载均衡算法（核心优化）**
   - 数学公式:
     ```
     ncRound = ceil(ncDim / coreNum)      // 向上取整
     ncRoundTail = floor(ncDim / coreNum)  // 向下取整
     remainder = ncDim % coreNum
     preCoreNum = remainder
     ```
   - 分配策略:
     - 前 `preCoreNum` 个核: 各处理 `ncRound` 个任务
     - 后 `coreNum - preCoreNum` 个核: 各处理 `ncRoundTail` 个任务
   - 负载差异: **最多 1 个任务** (ncRound - ncRoundTail = 1)

2. **NC 维度合并（关键优化）**
   - 将 N 和 C 维度合并为单一维度: `ncDim = nDim * cDim`
   - 优势:
     - 简化 Tiling 逻辑（5D → 4D）
     - 提高并行粒度（N=4, C=64 → NC=256,任务数更多）
     - 更灵活的任务分配（256 个任务分 16 核 vs 4 个任务分 16 核）
   - 适用性: N 和 C 维度的数据完全独立,无数据依赖

3. **动态核数适配**
   - 通过 `context->GetHardwareInfo().aiCoreNum` 获取实际核数
   - `blockDim = min(coreNum, ncDim)`: 避免核空闲
   - 适配不同硬件: 8核/16核/32核

4. **边界检查与空闲核处理**
   - `if (ncIndex >= ncDim) return;`: 当 NC 数量 < 核数时,多余核直接返回
   - `if (ncIndex + ncRealRound > ncDim)`: 调整最后一核的任务数,防止越界
   - 确保正确性,避免非法内存访问

5. **任务索引计算优化**
   - 使用 if-else 分支根据 `blockIdx` 和 `preCoreNum` 确定任务起点
   - 分支预测友好（blockIdx 通常顺序增长）
   - 计算简单（仅乘法和加法,无除法）

**性能提升数据** (理论分析):

| 场景 | Base Code (最后核) | Good Code (最大差异) | 改善比例 |
|-----|-------------------|---------------------|---------|
| 100 tasks / 16 cores | 10 tasks | 7 vs 6 (差1) | **负载均衡 1.67× → 1.17×** |
| 65 tasks / 16 cores | 17 tasks | 5 vs 4 (差1) | **负载均衡 4.25× → 1.25×** |
| 17 tasks / 16 cores | 2 tasks | 2 vs 1 (差1) | **负载均衡 2× → 2×** (最坏情况) |
| 10 tasks / 16 cores | 10 tasks (单核) | 10 核各 1 task | **并行度 1× → 10×** |

**负载均衡理论保证**:

- **最大负载差异**: 1 个任务（ceil 和 floor 的差）
- **负载不均衡率**: `ncRound / ncRoundTail = (floor + 1) / floor ≤ 1 + 1/floor`
  - 当 ncDim >> coreNum 时, 不均衡率 → 1 (完美均衡)
  - 最坏情况: ncDim = coreNum + 1, 不均衡率 = 2/1 = 2×
- **平均负载方差**: O(1 / ncDim) → 0 (ncDim 增大时)

**典型场景性能提升**:

- **[32, 64, 8, 8, 8]**: NC = 2048, 16 cores
  - Base: tasksPerCore = 128, 最后核 128 (均衡)
  - Good: ncRound = 128, ncRoundTail = 128 (完美均衡)
  - 性能提升: **持平** (任务数远大于核数)

- **[4, 128, 8, 8, 8]**: NC = 512, 16 cores
  - Base: tasksPerCore = 32, 最后核 32 (均衡)
  - Good: ncRound = 32, ncRoundTail = 32 (完美均衡)
  - 性能提升: **持平**

- **[1, 64, 8, 8, 8]**: NC = 64, 16 cores
  - Base: tasksPerCore = 4, 最后核 4 (均衡)
  - Good: ncRound = 4, ncRoundTail = 4 (完美均衡)
  - 性能提升: **持平**

- **[2, 17, 8, 8, 8]**: NC = 34, 16 cores
  - Base: tasksPerCore = 2, 最后核 **4** (2× 负载)
  - Good: ncRound = 3, ncRoundTail = 2, preCoreNum = 2
    - 前 2 核: 各 3 个任务
    - 后 14 核: 各 2 个任务
  - 性能提升: **~15-20%** (负载均衡改善)

- **[1, 10, 8, 8, 8]**: NC = 10, 16 cores
  - Base: tasksPerCore = 0, 最后核 **10** (∞ 不均衡)
  - Good: 前 10 核各 1 个任务, 后 6 核空闲
  - 性能提升: **~10×** (并行度提升)

**内存开销分析**:

- TilingData 增加: ~16 Bytes (4 个负载均衡参数)
- 运行时计算开销: 每核 1-2 次条件判断 + 几次算术运算（可忽略）

**适用场景**:

- 任何需要多核并行的算子
- 任务数接近或小于核数的场景
- 任务数不是核数整数倍的场景
- 需要精确负载均衡的高性能场景

**不适用场景**:

- 任务数远大于核数 且 任务执行时间完全均匀: 简单均分已足够
- 单核执行场景: 无需负载均衡

**关键设计原则**:

1. **余数均匀分配**: 余数任务分配到前 `remainder` 个核,而非最后一核
2. **维度合并**: 合并独立维度提高并行粒度
3. **动态核数**: 根据实际硬件和任务数动态确定使用核数
4. **最大差异最小化**: 确保任意两核的任务数差异 ≤ 1

**技术洞察**:

这是一个**通用负载均衡算法**,适用于任何需要将 N 个任务分配到 M 个核的场景。

**数学本质**:

给定 N 个任务和 M 个核,如何分配使得负载最均衡?

**朴素方法**:
- 每核 `floor(N/M)` 个任务
- 最后一核处理剩余 `N - (M-1)*floor(N/M)` 个任务
- 最大负载差异: `N % M` (可能很大)

**最优方法**:
- 计算 `q = floor(N/M)` 和 `r = N % M`
- 前 `r` 个核: 各 `q+1` 个任务
- 后 `M-r` 个核: 各 `q` 个任务
- 最大负载差异: **1** (理论最优)

**证明最优性**:

假设存在更好的分配方案,使得所有核任务数相等。
则 N = M × k (k 为整数),即 N % M = 0。
但当 N % M ≠ 0 时,必然存在至少一对核任务数差异 ≥ 1。
因此,差异为 1 是理论最优。

**推广**:

这个算法可推广到:
1. 多级负载均衡（NUMA 节点 → 核 → 线程）
2. 异构计算（不同核性能不同,按比例分配）
3. 动态负载均衡（运行时根据任务完成情况重新分配）

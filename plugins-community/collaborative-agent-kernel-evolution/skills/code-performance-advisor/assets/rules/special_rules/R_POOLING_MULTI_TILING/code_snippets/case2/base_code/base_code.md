# Base Code: 固定单维度 Tiling

来源: lingxi-code (max_pool_with_argmax_v3, 推断)

```cpp
// OpInfo 文件: 简单的 Tiling 策略
ge::graphStatus MaxPoolWithArgmaxV3Tiling::Tiling(gert::TilingContext* context)
{
    const uint32_t BLOCK_DIM = 16;  // 固定 16 核
    const uint32_t MAX_TILE_C = 512;  // 固定 C 维度分块大小

    // 问题1: 仅在 C 维度分块,其他维度不分块
    uint32_t tileC = (MAX_TILE_C < channels) ? MAX_TILE_C : channels;
    uint32_t cTiles = (channels + tileC - 1) / tileC;

    // 问题2: 简单计算总任务数,未考虑 UB 容量限制
    uint32_t totalTasks = batchSize * outH * outW * cTiles;

    // 问题3: 简单均分任务,未考虑负载均衡
    uint32_t tasksPerCore = (totalTasks + BLOCK_DIM - 1) / BLOCK_DIM;

    // 单一 Tiling Key
    TilingData.tilingKey = 1;
    TilingData.blockDim = BLOCK_DIM;
    TilingData.tileC = tileC;
    TilingData.tasksPerCore = tasksPerCore;

    return ge::GRAPH_SUCCESS;
}

// TilingData 定义: 简单参数
BEGIN_TILING_DATA_DEF(MaxPoolWithArgmaxV3CustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, channels);
    TILING_DATA_FIELD_DEF(uint32_t, height);
    TILING_DATA_FIELD_DEF(uint32_t, width);
    TILING_DATA_FIELD_DEF(uint32_t, outH);
    TILING_DATA_FIELD_DEF(uint32_t, outW);
    TILING_DATA_FIELD_DEF(uint32_t, tileC);  // 仅 C 维度分块
    TILING_DATA_FIELD_DEF(uint32_t, tasksPerCore);
    // ... 其他基础参数
END_TILING_DATA_DEF;

// Kernel 实现: 按固定 Tiling 处理
extern "C" __global__ __aicore__ void max_pool_with_argmax_v3(
    GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    uint32_t blockIdx = GetBlockIdx();
    uint32_t startTask = blockIdx * tiling_data.tasksPerCore;
    uint32_t endTask = startTask + tiling_data.tasksPerCore;

    // 问题4: 固定的任务分配,未考虑不同场景
    for (uint32_t task = startTask; task < endTask; task++) {
        // 分解任务 ID 到 (n, h, w, c_tile)
        uint32_t n = task / (tiling_data.outH * tiling_data.outW * cTiles);
        uint32_t tmp = task % (tiling_data.outH * tiling_data.outW * cTiles);
        uint32_t h = tmp / (tiling_data.outW * cTiles);
        tmp = tmp % (tiling_data.outW * cTiles);
        uint32_t w = tmp / cTiles;
        uint32_t cTile = tmp % cTiles;

        uint32_t cStart = cTile * tiling_data.tileC;
        uint32_t cEnd = (cStart + tiling_data.tileC < tiling_data.channels) ?
                        (cStart + tiling_data.tileC) : tiling_data.channels;

        // 处理 Pooling (可能超出 UB 容量)
        ProcessPooling(n, h, w, cStart, cEnd);
    }
}
```

**问题分析**:

1. **单维度 Tiling 限制**
   - 仅在 C 维度进行分块（固定 MAX_TILE_C = 512）
   - N/H/W 维度不分块,直接合并为"任务数"
   - 无法适应不同的输入 shape 特征
   - 对于 large H×W (如 224×224) 或 small C (如 64) 的场景,Tiling 策略不优

2. **UB 容量检查缺失**
   - 未检查 tileC 大小是否满足 UB 容量限制
   - Pooling 需要同时存储: 输入数据、输出数据、argmax 索引
   - 内存需求 ≈ tileC × (kernel_h × kernel_w × sizeof(input) + sizeof(output) + sizeof(argmax))
   - 对于大 tileC 或大 kernel,可能超出 UB 容量导致编译失败或运行错误

3. **负载不均衡**
   - 简单的 `totalTasks / BLOCK_DIM` 可能导致某些核空闲
   - 例如: totalTasks = 100, BLOCK_DIM = 16 → tasksPerCore = 7
   - 前 4 核处理 7 个任务,后 12 核处理 6 个任务（实际是 100 / 16 余数问题）
   - 尾部任务分配不均,最后几个核可能提前完成

4. **未考虑多维分块优先级**
   - 不同维度的分块对性能影响不同
   - N 维度分块: 适合大 batch,各核独立处理不同 batch
   - H/W 维度分块: 适合大特征图,减少单次 UB 占用
   - C 维度分块: 适合大 channel,但会增加重复加载输入数据的开销
   - 应优先尝试外层维度分块（N → H → W → C）

5. **固定核数假设**
   - 硬编码 BLOCK_DIM = 16,未根据实际可用核数动态调整
   - 在不同硬件平台上（如 8 核 / 32 核）,固定核数可能导致资源浪费或利用率不足
   - 应根据 context->GetHardwareInfo() 获取实际核数

6. **单一 Tiling Key**
   - 所有场景使用同一实现路径
   - 无法针对不同 shape 特征选择最优算法
   - 例如: small C 场景适合向量化,large C 场景适合分块处理

**性能瓶颈**:

- **UB 容量溢出风险**: 大 tileC 可能导致 UB 不足,降级为多次循环
- **负载不均衡**: 某些核提前完成,整体执行时间受最慢核限制
- **内存带宽浪费**: C 维度分块导致输入数据重复加载
- **单一策略低效**: 不同 shape 使用同一算法,无法充分优化

**典型问题场景**:

- **Large Feature Map**: [1, 64, 224, 224] → tileC=64 不分块,H×W 太大导致单核负担重
- **Large Channel**: [32, 2048, 7, 7] → tileC=512 分为 4 块,C 维度分块导致输入重复加载
- **Small Batch Large HW**: [1, 128, 112, 112] → 单 batch 无法并行,16 核大部分空闲
- **核数不匹配**: 32 核硬件使用 BLOCK_DIM=16,浪费 50% 计算资源

**根本原因**:

传统简单 Tiling 思维:选择一个固定维度（通常是 C）分块,其他维度展平为线性任务。这种方法在早期简单场景下有效,但无法适应现代 AI 模型的多样化 shape 特征。

需要:
1. 多维度动态 Tiling
2. UB 容量感知
3. 负载均衡算法
4. 多策略自适应选择

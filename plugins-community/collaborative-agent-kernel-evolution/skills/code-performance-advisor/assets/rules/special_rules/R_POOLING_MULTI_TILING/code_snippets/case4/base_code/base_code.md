# Base Code: 固定 Tiling 策略,无法适应不同 Shape

来源: lingxi-code (max_pool_grad_with_argmax_common, 推断)

```cpp
// Tiling 函数: 固定策略
ge::graphStatus MaxPoolGradWithArgmaxTiling::Tiling(gert::TilingContext* context)
{
    const uint32_t BLOCK_DIM = 16;
    const uint32_t TILE_C = 256;  // 固定 C 维度分块

    // 问题1: 无论输入 shape 如何,都使用相同策略
    // 未考虑 H×W×C 的不同组合特征

    uint32_t cTiles = (cDim + TILE_C - 1) / TILE_C;
    uint32_t totalTasks = nDim * hArgmax * wArgmax * cTiles;

    TilingData.blockDim = BLOCK_DIM;
    TilingData.tileC = TILE_C;
    TilingData.tilingKey = 1;  // 单一 Tiling Key

    return ge::GRAPH_SUCCESS;
}

// Kernel: 单一实现
extern "C" __global__ __aicore__ void max_pool_grad_with_argmax(...)
{
    // 问题2: 固定的循环结构
    for (uint32_t n = 0; n < nDim; n++) {
        for (uint32_t h = 0; h < hArgmax; h++) {
            for (uint32_t w = 0; w < wArgmax; w++) {
                for (uint32_t c = 0; c < cDim; c += TILE_C) {
                    // 问题3: 未利用不同 shape 的特性优化
                    // - Small C: 可以向量化处理整个 C 维度
                    // - Large H×W: 可以合并 H×W 提高并行度
                    // - Large C: 需要特殊的分块策略
                    ProcessGradient(n, h, w, c, c + TILE_C);
                }
            }
        }
    }
}
```

**问题分析**:

1. **无法识别 Shape 特征**
   - **Small C 场景**: [32, 56, 56, **64**] - C 小,但 H×W 大
     - 固定 TILE_C=256 导致 C 不分块,但未利用向量化机会
   - **Large C 场景**: [8, 7, 7, **2048**] - C 大,H×W 小
     - TILE_C=256 分为 8 块,但未考虑 UB 容量优化
   - **Balanced 场景**: [16, 28, 28, **512**] - H×W×C 都适中
     - 固定策略可能不是最优

2. **未利用维度合并优化**
   - **H×W×C 合并** (当 H×W×C 较小时): 可以一次性处理多个输出点
   - **W×C 合并** (当 H 较大时): 减少循环层数,提高并行度
   - 固定循环结构无法灵活调整

3. **单一 Tiling Key 限制**
   - 所有场景使用同一 kernel 实现
   - 无法针对特定 shape 特征选择最优算法
   - 性能无法达到各场景的最优

4. **循环开销未优化**
   - 4 层嵌套循环 (N → H → W → C)
   - 循环控制开销随维度数线性增长
   - 可以通过合并维度减少循环层数

**性能影响**:

- **Small C 场景**: 向量化机会浪费,**损失 30-50%**
- **Large C 场景**: 分块策略不优,**损失 20-40%**
- **Large H×W 场景**: 并行度不足,**损失 15-30%**
- **平均性能损失**: **20-35%**

**典型问题场景**:

| Shape | 特征 | 问题 |
|-------|------|------|
| [32, 56, 56, 64] | Small C | C 不分块但未向量化 |
| [8, 7, 7, 2048] | Large C, Small H×W | 分块策略不优 |
| [16, 112, 112, 128] | Large H×W | 未合并 H×W 提高并行度 |

**根本原因**: One-size-fits-all 思维,忽略了不同 shape 的特性差异。

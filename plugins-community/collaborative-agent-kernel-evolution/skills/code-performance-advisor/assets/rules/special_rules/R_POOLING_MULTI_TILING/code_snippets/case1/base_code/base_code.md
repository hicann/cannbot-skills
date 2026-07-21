# Base Code: 单一 Tiling 策略

来源：lingxi-code (adaptive_avg_pool3d)

```cpp
// OpInfo 文件
ge::graphStatus AdaptiveAvgPool3dTiling::Tiling(gert::TilingContext* context)
{
    // 问题：固定的 Tiling 策略，未考虑不同 shape 的特点
    const uint32_t BLOCK_DIM = 16;  // 固定 16 核

    // 简单的输出点数量计算
    uint32_t total_output_elems = N * D_out * H_out * W_out;

    // 简单均分到各 Core
    uint32_t elems_per_core = (total_output_elems + BLOCK_DIM - 1) / BLOCK_DIM;

    // 单一 Tiling Key
    context->SetTilingKey(1);
    context->SetBlockDim(BLOCK_DIM);

    return ge::GRAPH_SUCCESS;
}

// TilingData 定义
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, D_in);
  TILING_DATA_FIELD_DEF(uint32_t, H_in);
  TILING_DATA_FIELD_DEF(uint32_t, W_in);
  TILING_DATA_FIELD_DEF(uint32_t, D_out);
  TILING_DATA_FIELD_DEF(uint32_t, H_out);
  TILING_DATA_FIELD_DEF(uint32_t, W_out);
  TILING_DATA_FIELD_DEF(uint32_t, elems_per_core);
  // ... 简单参数
END_TILING_DATA_DEF;

// Kernel 实现：单一实现
extern "C" __global__ __aicore__ void adaptive_avg_pool3d_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 问题：单一实现无法针对不同场景优化
    KernelAdaptiveAvgPool3d op;
    op.Init(x, y, tiling_data.C, ...);
    op.Process();
}
```

**问题**：

1. **未考虑 UB 空间限制**
   - Channel 很大时（如 C = 2048），UB 无法容纳完整 Channel
   - 未进行 Channel 维度切分，导致 UB 溢出或性能下降
   - 一刀切的策略无法适应不同硬件（UB 大小不同）

2. **未考虑数据复用机会**
   - Pooling 窗口重叠时（Adaptive Pooling 常见），输入数据有复用潜力
   - 单一策略无法优化数据复用
   - 内存带宽浪费

3. **未考虑多核负载均衡**
   - 固定 16 核，输出点数量 < 16 时浪费核心
   - 输出点数量不能被 16 整除时，负载不均衡
   - 最后一个 Core 处理的数据量可能过少或过多

4. **未考虑不同数据类型的影响**
   - FP32 和 FP16 的内存占用不同（2 倍差异）
   - 单一 Tiling 策略无法针对数据类型优化
   - 低精度类型的内存优势未充分利用

5. **缺乏 Tiling Key 分发机制**
   - 所有场景都使用同一个 Kernel 实现
   - 无法针对不同场景生成最优代码
   - 编译器优化受限

**典型问题场景**：
- C = 2048, UB 无法容纳（需要 Split-C 模式）
- C = 64, UB 空间富余（可以使用 Multi-W 模式提升复用）
- 输出点数量 = 5, 16 核浪费严重
- FP16 数据类型，内存占用减半但未优化 Tiling

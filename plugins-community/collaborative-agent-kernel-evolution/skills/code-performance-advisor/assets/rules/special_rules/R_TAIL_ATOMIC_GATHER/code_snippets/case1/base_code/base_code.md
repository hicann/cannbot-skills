# Base Code: 简单的数据搬出，未处理尾块

来源：lingxi-code (adaptive_avg_pool3d)

```cpp
class KernelAdaptiveAvgPool3d {
    __aicore__ inline void CopyOut(uint32_t out_offset)
    {
        AscendC::LocalTensor<float> outputLocal = outQueue.DeQue<float>();

        // 问题：简单的 DataCopyPad，未考虑尾块对齐问题
        AscendC::DataCopyPad(outputGm[out_offset], outputLocal,
                             {1, static_cast<uint16_t>(C * sizeof(float)), 0, 0, 0});

        outQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void Process()
    {
        uint32_t start_elem = GetBlockIdx() * elems_per_core;
        uint32_t end_elem = start_elem + elems_per_core;
        if (end_elem > total_output_elems) {
            end_elem = total_output_elems;  // 简单截断
        }

        for (uint32_t i = start_elem; i < end_elem; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
};
```

**问题**：

1. **数据对齐问题未处理**
   - Channel 维度不是 32B 对齐时，搬出操作效率低
   - 最后一个 Core 处理的元素数量可能不对齐
   - 边界数据可能跨越 Cache Line，导致性能下降

2. **跨 Core 边界数据竞争**
   - 多核并行时，尾块数据可能跨越 Core 边界
   - Core N 的最后几个元素与 Core N+1 的起始元素共享一个对齐块
   - 可能导致部分数据被覆盖或丢失

3. **内存访问效率低**
   - 非对齐访问触发多次内存事务
   - 部分有效数据周围的 Padding 也被搬运
   - 浪费带宽和延迟

4. **边界情况未覆盖**
   - Channel 数量 < 对齐要求时的处理
   - 最后一个输出点的 Channel 数量不足时的处理
   - 可能导致越界访问或数据错误

**典型问题场景**：
- Channel = 17 时（非 32B 对齐，FP32 需要 8 个元素对齐）
- 总输出点数不能被 Core 数整除
- 最后一个 Core 处理的元素较少
- 多核写入相邻内存区域

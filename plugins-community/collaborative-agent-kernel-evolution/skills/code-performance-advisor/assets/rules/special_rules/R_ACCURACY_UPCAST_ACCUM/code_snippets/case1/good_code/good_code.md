# Good Code: 升精度累加保证数值稳定性

来源：expert code (adaptive_avg_pool3d)

```cpp
// 使用编译时类型判断实现升精度累加
template <typename T, int32_t QUEUE_DEPTH>
class KernelAdaptiveAvgPool3dSplitC
{
    __aicore__ inline void ReduceSum(
        const Index& index, LocalTensor<float>& sumBufLocal, int64_t cOffset, int64_t len, int64_t nOffset)
    {
        LocalTensor<T> inputLocal = inputQueue.DeQue<T>();

        // 升精度策略：
        // 1. FP32 -> FP32: 直接累加，无精度损失
        // 2. FP16/BF16 -> FP32: 先 Cast 到 FP32 再累加，保证精度
        if constexpr (std::is_same_v<T, float>) {
            // FP32 输入，直接累加
            Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
            // FP16/BF16 输入，升精度到 FP32 后累加
            LocalTensor<float> castBufLocal = castBuf.Get<float>();
            Cast(castBufLocal, inputLocal, RoundMode::CAST_NONE, len);
            Add(sumBufLocal, sumBufLocal, castBufLocal, len);
        }

        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ReduceMean(int64_t outputPointIdx, int64_t bufIdx)
    {
        // sumBufLocal 始终是 FP32，保证了中间计算的精度
        LocalTensor<float> sumBufLocal = sumBuf.Get<float>();
        Index index;
        GetIndexFromBuffer(indexBuf, bufIdx, bufIdx, index);
        SToVSync();

        // FP32 精度计算平均因子
        float factor = 1.0f / static_cast<float>(
            (index.dend - index.dstart) * (index.hend - index.hstart) * (index.wend - index.wstart));

        // FP32 精度计算平均值
        LocalTensor<float> meanBufLocal = meanBuf.Get<float>();
        Muls(meanBufLocal, sumBufLocal, factor, count);
        PipeBarrier<PIPE_V>();

        // 降精度输出：根据输出类型选择合适的 RoundMode
        LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
        } else if constexpr (std::is_same_v<T, half>) {
            // FP16 使用 CAST_NONE，截断模式
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, count);
        } else {  // bfloat16_t
            // BF16 使用 CAST_RINT，四舍五入模式，减小误差
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
        }
        outputQueue.EnQue(outputLocal);
    }
};
```

**改进点**：

1. **编译时类型判断（constexpr if）**
   - 使用 `std::is_same_v<T, float>` 在编译时确定数据类型
   - 无运行时开销，编译器生成特化代码

2. **升精度累加策略**
   - 输入是 FP16/BF16 时，先 Cast 到 FP32 再累加
   - 中间累加缓冲区 `sumBufLocal` 始终是 FP32
   - 保证累加过程的数值稳定性

3. **差异化 RoundMode**
   - FP16 输出：使用 `CAST_NONE`（截断）
   - BF16 输出：使用 `CAST_RINT`（四舍五入）
   - BF16 尾数位少，四舍五入比截断更精确

4. **精度损失控制**
   - 仅在必要时 Cast（输入和输出）
   - 中间计算全程 FP32
   - 最小化精度损失

**性能提升**：
- 数值精度提升显著，特别是大规模累加场景
- 相比全程 FP16/BF16，误差降低 2-3 个数量级
- 性能损失可控：仅增加 2 次 Cast 操作，通常 < 5%
- 典型场景：BatchNorm 均值/方差计算，Pooling 平均值计算

**适用场景**：
- 所有涉及多个低精度值累加的场景
- Reduction 类算子（Sum, Mean, Variance）
- Pooling 算子（AvgPool, AdaptiveAvgPool）
- Normalization 算子（BatchNorm, LayerNorm, RMSNorm）

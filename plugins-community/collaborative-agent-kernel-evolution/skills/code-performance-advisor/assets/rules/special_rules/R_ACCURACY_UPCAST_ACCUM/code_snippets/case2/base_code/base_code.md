# Base Code: 直接低精度累加计算

来源：lingxi-code (batch_norm_v3)

```cpp
// 全程 FP16 计算（无精度提升）
class BatchNormV3Kernel {
private:
    AscendC::GlobalTensor<half> inputGm;   // FP16
    AscendC::GlobalTensor<half> meanGm;    // FP16
    AscendC::GlobalTensor<half> varianceGm; // FP16

    __aicore__ inline void ComputeMeanVariance(uint32_t channelIdx)
    {
        // FP16 累加 - 容易精度损失
        half channelSum = 0.0f;
        half channelSqSum = 0.0f;

        for (uint32_t spatialIdx = 0; spatialIdx < spatialSize; spatialIdx++) {
            AscendC::LocalTensor<half> dataLocal = inQueue.DeQue<half>();

            // 直接 FP16 累加 - 数值不稳定
            AscendC::ReduceSum(sumLocal, dataLocal, sumLocal, tileSize);
            channelSum += sumLocal.GetValue(0);  // FP16 累加

            // 计算平方和 - FP16 容易溢出
            AscendC::Mul(sqLocal, dataLocal, dataLocal, tileSize);
            AscendC::ReduceSum(sqSumLocal, sqLocal, sqSumLocal, tileSize);
            channelSqSum += sqSumLocal.GetValue(0);  // FP16 累加平方值

            inQueue.FreeTensor(dataLocal);
        }

        // FP16 计算均值和方差
        half mean = channelSum / spatialSize;
        half variance = (channelSqSum / spatialSize) - (mean * mean);

        // 存储 FP16 结果
        meanGm.SetValue(channelIdx, mean);
        varianceGm.SetValue(channelIdx, variance);
    }
};
```

**问题**：
1. **FP16 累加精度损失**：大量 FP16 数据累加，低位精度丢失严重
   - FP16 有效位数仅 10 位，累加 1000+ 个数时误差显著
2. **平方操作易溢出**：FP16 表示范围 ±65504，平方后容易超出范围
   - 如输入值 300，平方后 90000 已超出 FP16 范围
3. **方差计算数值不稳定**：E[x²] - (E[x])² 在 FP16 下容易出现灾难性抵消
   - 当数据方差较小时，两个接近的大数相减，精度损失严重
4. **无中间精度提升**：全程 FP16 计算，累积误差无法避免
5. **统计量存储精度低**：mean/variance 以 FP16 存储，后续使用时精度受限

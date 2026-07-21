# Base Code: 不进行升精度累加的低精度计算

来源：lingxi-code (adaptive_avg_pool3d - 推断)

```cpp
// 直接使用低精度类型（FP16/BF16）进行累加
__aicore__ inline void ComputeAccumulate()
{
    AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
    AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();

    // 问题：如果数据类型是 half 或 bfloat16_t，直接累加会损失精度
    AscendC::Add(accumLocal, accumLocal, inputLocal, C);

    inQueue.FreeTensor(inputLocal);
}

__aicore__ inline void ComputeAverage()
{
    AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();
    AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();

    // 低精度累加后的结果直接计算平均值
    AscendC::Muls(outputLocal, accumLocal, avg_scale, C);
    outQueue.EnQue(outputLocal);
}
```

**问题**：
1. 低精度类型（FP16/BF16）累加时，由于尾数位数有限，会产生累积误差
2. 大批量数据累加时，误差会显著放大（如累加 10000 个数值）
3. BF16 尾数仅 7 位，FP16 尾数 10 位，远小于 FP32 的 23 位
4. 数值范围差异大时（如最大值/最小值相差 1000 倍），小数值会被"吞没"
5. 最终结果的数值精度损失可能导致模型收敛问题

**典型问题场景**：
- Pooling 算子累加大量输入值计算平均值
- BatchNorm 计算均值和方差时累加大量样本
- Reduction 类算子对大量元素求和

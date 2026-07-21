# Base Code: 两趟算法计算均值和方差

来源：lingxi-code (batch_norm_v3 - 推断，基于传统实现)

```cpp
class KernelBatchNorm {
    __aicore__ inline void ComputeMean()
    {
        // 第一趟：计算均值
        AscendC::LocalTensor<float> sumLocal = sumBuf.Get<float>();
        AscendC::Duplicate(sumLocal, 0.0f, C);

        // 累加所有数据
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    AscendC::LocalTensor<float> xLocal = xQueue.DeQue<float>();
                    AscendC::Add(sumLocal, sumLocal, xLocal, C);
                    xQueue.FreeTensor(xLocal);
                }
            }
        }

        // 计算均值
        float count = static_cast<float>(N * H * W);
        AscendC::LocalTensor<float> meanLocal = meanBuf.Get<float>();
        AscendC::Muls(meanLocal, sumLocal, 1.0f / count, C);
    }

    __aicore__ inline void ComputeVariance()
    {
        // 第二趟：计算方差
        AscendC::LocalTensor<float> meanLocal = meanBuf.Get<float>();
        AscendC::LocalTensor<float> m2Local = m2Buf.Get<float>();
        AscendC::Duplicate(m2Local, 0.0f, C);

        // 重新遍历所有数据，计算 (x - mean)^2
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    AscendC::LocalTensor<float> xLocal = xQueue.DeQue<float>();
                    AscendC::LocalTensor<float> deltaLocal = deltaBuf.Get<float>();

                    // 计算 delta = x - mean
                    AscendC::Sub(deltaLocal, xLocal, meanLocal, C);

                    // 计算 delta^2
                    AscendC::Mul(deltaLocal, deltaLocal, deltaLocal, C);

                    // 累加到 m2
                    AscendC::Add(m2Local, m2Local, deltaLocal, C);

                    xQueue.FreeTensor(xLocal);
                }
            }
        }

        // 计算方差
        float count = static_cast<float>(N * H * W);
        AscendC::LocalTensor<float> varianceLocal = varianceBuf.Get<float>();
        AscendC::Muls(varianceLocal, m2Local, 1.0f / count, C);
    }

    __aicore__ inline void Process()
    {
        // 问题：需要两趟遍历数据
        ComputeMean();      // 第一趟
        ComputeVariance();  // 第二趟
        Normalize();
    }
};
```

**问题**：

1. **两趟数据遍历**
   - 第一趟：计算均值
   - 第二趟：计算方差（需要均值结果）
   - 数据需要从 GM 读取两次

2. **内存带宽浪费**
   - 输入数据 X 被读取两次，带宽翻倍
   - 对于大规模数据（如 N * H * W 很大），带宽成为瓶颈
   - UB 空间压力大（需要缓存所有数据或重复搬运）

3. **流水线效率低**
   - 第二趟必须等待第一趟完全结束
   - 无法充分利用流水线并行
   - 延迟增加一倍

4. **数值稳定性问题**
   - `(x - mean)^2` 计算可能产生较大中间值
   - 大数减小数的精度损失
   - 方差计算误差累积

**典型问题场景**：
- BatchNorm 输入 shape 很大时（如 [256, 512, 28, 28]）
- LayerNorm 序列长度很长时
- 带宽受限的平台（如低功耗设备）
- 需要实时推理的场景

# Base Code: 两趟算法计算 RMSNorm 梯度

来源：lingxi-code (rms_norm_grad)

```cpp
// 两趟算法：先计算 RMS，再计算梯度
class RmsNormGradKernel {
private:
    __aicore__ inline void ComputeGradient(uint32_t rowIdx)
    {
        // 第一趟：计算 RMS (root mean square)
        float sumX2 = 0.0f;
        uint32_t elementsProcessed = 0;

        while (elementsProcessed < this->cols) {
            uint32_t tileLength = min(maxTileLength, this->cols - elementsProcessed);

            // 加载 x
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[rowIdx * this->cols + elementsProcessed], tileLength);

            // x^2
            AscendC::Mul(x2Local, xLocal, xLocal, tileLength);

            // Reduce sum x^2
            AscendC::ReduceSum(sumTempLocal, x2Local, sumTempLocal, tileLength);
            float tileSumX2 = sumTempLocal.GetValue(0);
            sumX2 += tileSumX2;

            inQueueX.FreeTensor(xLocal);
            elementsProcessed += tileLength;
        }

        float meanX2 = sumX2 / this->cols;
        float rmsVal = sqrt(meanX2 + eps);
        float rstd = 1.0f / rmsVal;

        // 第二趟：计算梯度（需要再次加载所有数据）
        elementsProcessed = 0;
        while (elementsProcessed < this->cols) {
            uint32_t tileLength = min(maxTileLength, this->cols - elementsProcessed);

            // 再次加载 x 和 dy
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::LocalTensor<float> dyLocal = inQueueDY.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[rowIdx * this->cols + elementsProcessed], tileLength);
            AscendC::DataCopy(dyLocal, dyGm[rowIdx * this->cols + elementsProcessed], tileLength);

            // dx = dy * gamma * rstd - ...
            // 复杂的梯度计算

            outQueue.EnQue(dxLocal);
            elementsProcessed += tileLength;
        }
    }
};
```

**问题**：
1. **两趟遍历**：先计算 RMS，再计算梯度
2. **内存访问量大**：x 被读取两次，dy 被读取一次
3. **无法流水线**：第二趟必须等第一趟计算出 RMS
4. **数值不稳定**：直接计算 E[x²] 再开方，容易溢出
5. **大 hidden_size 性能差**：hidden_size > 4096 时两次遍历开销显著
6. **无 Welford 优化**：RMS 计算没有使用增量更新策略

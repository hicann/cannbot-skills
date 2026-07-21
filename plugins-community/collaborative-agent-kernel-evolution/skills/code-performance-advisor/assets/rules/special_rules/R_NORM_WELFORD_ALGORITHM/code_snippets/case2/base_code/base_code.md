# Base Code: 传统两趟算法计算 LayerNorm

来源：lingxi-code (layer_norm_v4)

```cpp
// 两趟算法：第一趟计算 mean，第二趟计算 variance
class LayerNormV4Kernel {
private:
    __aicore__ inline void ComputeRow(uint32_t rowIdx)
    {
        // 第一趟：计算均值
        float rowSum = 0.0f;
        uint32_t elementsProcessed = 0;

        while (elementsProcessed < this->cols) {
            uint32_t tileLength = min(maxTileLength, this->cols - elementsProcessed);

            // 加载数据
            AscendC::LocalTensor<float> inputLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(inputLocal, inputGm[rowIdx * this->cols + elementsProcessed], tileLength);
            inQueueX.EnQue(inputLocal);

            // Reduce sum
            inputLocal = inQueueX.DeQue<float>();
            AscendC::ReduceSum(sharedLocal, inputLocal, sharedLocal, tileLength);
            float tileSum = sharedLocal.GetValue(0);
            rowSum += tileSum;

            inQueueX.FreeTensor(inputLocal);
            elementsProcessed += tileLength;
        }

        float rowMean = rowSum / this->cols;

        // 第二趟：计算方差（需要再次遍历所有数据）
        float rowVarSum = 0.0f;
        elementsProcessed = 0;

        while (elementsProcessed < this->cols) {
            uint32_t tileLength = min(maxTileLength, this->cols - elementsProcessed);

            // 再次加载数据
            AscendC::LocalTensor<float> inputLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(inputLocal, inputGm[rowIdx * this->cols + elementsProcessed], tileLength);

            // (x - mean)^2
            AscendC::Adds(tempLocal, inputLocal, -rowMean, tileLength);
            AscendC::Mul(tempLocal, tempLocal, tempLocal, tileLength);

            // Reduce sum
            AscendC::ReduceSum(sharedLocal, tempLocal, sharedLocal, tileLength);
            float tileVar = sharedLocal.GetValue(0);
            rowVarSum += tileVar;

            inQueueX.FreeTensor(inputLocal);
            elementsProcessed += tileLength;
        }

        float rowVariance = rowVarSum / this->cols;

        // 第三趟：归一化（又一次遍历数据）
        float invStd = 1.0f / sqrt(rowVariance + this->eps);
        // ... 归一化处理
    }
};
```

**问题**：
1. **两趟遍历**：先计算 mean，再计算 variance，需要两次加载所有数据
2. **内存访问量翻倍**：每个元素被读取两次，内存带宽浪费
3. **三趟变两趟仍不够**：如果还要做归一化，可能需要第三趟
4. **数值稳定性问题**：使用 E[x²] - (E[x])² 公式时，当方差很小时容易出现灾难性抵消
5. **无法流水线**：第二趟必须等第一趟完全结束，无法重叠执行
6. **大数据量性能差**：对于 hidden_size > 4096 的场景，两次遍历开销巨大

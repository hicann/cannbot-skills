# Base Code: 逐元素处理 Padding 数据

来源：lingxi-code (layer_norm_v4)

```cpp
// 简单逐元素处理，无法跳过 Padding
class LayerNormV4Kernel {
private:
    __aicore__ inline void NormalizeRow(uint32_t rowIdx)
    {
        uint32_t elementsProcessed = 0;

        while (elementsProcessed < this->cols) {
            uint32_t tileLength = min(maxTileLength, this->cols - elementsProcessed);

            // 加载数据（包含 Padding）
            AscendC::LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[rowIdx * colsAlign + elementsProcessed], tileLength);

            // 归一化：(x - mean) / std
            // 问题：如果 tileLength 包含 Padding，需要逐元素判断
            for (uint32_t i = 0; i < tileLength; i++) {
                if (elementsProcessed + i < this->cols) {
                    // 真实数据
                    float val = xLocal.GetValue(i);
                    val = (val - rowMean) * invStd;
                    val = val * weight + bias;
                    yLocal.SetValue(i, val);
                } else {
                    // Padding 区域，设为 0 或保持不变
                    yLocal.SetValue(i, 0.0f);
                }
            }

            outQueue.EnQue(yLocal);
            elementsProcessed += tileLength;
        }
    }
};
```

**问题**：
1. **逐元素处理**：无法利用向量指令的并行能力
2. **条件分支**：循环内的 if 判断导致分支预测失败，性能下降
3. **无法跳过 Padding**：Padding 区域也参与计算，浪费计算资源
4. **向量指令效率低**：由于逐元素处理，向量单元利用率低
5. **大量标量操作**：GetValue/SetValue 是标量操作，慢于向量操作 10-100 倍
6. **Padding 污染统计量**：如果 Padding 不是 0，可能影响 mean/variance 计算

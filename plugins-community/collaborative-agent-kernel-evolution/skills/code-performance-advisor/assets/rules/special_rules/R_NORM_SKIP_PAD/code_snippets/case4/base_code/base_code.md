# Base Code: 逐元素处理,未利用 Repeat/Stride 优化

来源: lingxi-code (layer_norm_v3, 推断)

```cpp
template <typename T>
class LayerNormV3 {
    __aicore__ inline void Process() {
        // 问题1: 逐元素加载,未使用 Repeat 参数批量处理
        for (uint32_t n = 0; n < nDim; n++) {
            for (uint32_t h = 0; h < hDim; h++) {
                for (uint32_t w = 0; w < wDim; w++) {
                    uint32_t offset = (n * hDim * wDim + h * wDim + w) * cDim;

                    // 逐元素加载
                    LocalTensor<T> inputLocal = inputBuf.Get<T>();
                    for (uint32_t c = 0; c < cDim; c++) {
                        inputLocal.SetValue(c, inputGM[offset + c]);
                    }

                    // 计算均值和方差（标量循环）
                    T sum = 0, sumSq = 0;
                    for (uint32_t c = 0; c < cDim; c++) {
                        T val = inputLocal.GetValue(c);
                        sum += val;
                        sumSq += val * val;
                    }
                    T mean = sum / cDim;
                    T variance = sumSq / cDim - mean * mean;
                    T invStd = 1.0f / sqrt(variance + epsilon);

                    // 逐元素归一化和写回
                    for (uint32_t c = 0; c < cDim; c++) {
                        T normed = (inputLocal.GetValue(c) - mean) * invStd;
                        outputGM[offset + c] = normed;
                    }
                }
            }
        }
    }
};
```

**问题**: 标量循环,未利用 Vector 指令批量处理,性能低下。

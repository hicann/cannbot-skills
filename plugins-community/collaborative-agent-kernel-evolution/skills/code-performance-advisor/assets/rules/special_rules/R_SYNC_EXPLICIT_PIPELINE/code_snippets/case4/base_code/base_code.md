# Base Code: 多阶段计算缺少阶段间同步

来源:deep_norm (lingxi-code)

```cpp
class KernelDeepNorm {
    __aicore__ inline float ComputeMean(uint32_t rowIdx)
    {
        float sum = 0.0f;

        for (uint32_t tileId = 0; tileId < nTiles; tileId++) {
            uint32_t offset = rowIdx * normSize + tileId * tileLength;
            uint32_t length = (tileId == nTiles - 1) ? (normSize - tileId * tileLength) : tileLength;

            LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
            DataCopy(inputLocal, inputGm[offset], length);

            // 问题:DataCopy 后没有等待 MTE2 完成就使用
            for (uint32_t i = 0; i < length; i++) {
                sum += inputLocal.GetValue(i);  // Scalar 读取可能未完成的 DMA
            }

            inQueue.FreeTensor(inputLocal);
        }

        return sum / normSize;
    }

    __aicore__ inline float ComputeVariance(uint32_t rowIdx, float mean)
    {
        float variance = 0.0f;

        for (uint32_t tileId = 0; tileId < nTiles; tileId++) {
            uint32_t offset = rowIdx * normSize + tileId * tileLength;
            uint32_t length = (tileId == nTiles - 1) ? (normSize - tileId * tileLength) : tileLength;

            LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
            DataCopy(inputLocal, inputGm[offset], length);

            // 问题:重复的 Scalar 循环,效率低
            for (uint32_t i = 0; i < length; i++) {
                float diff = inputLocal.GetValue(i) - mean;
                variance += diff * diff;
            }

            inQueue.FreeTensor(inputLocal);
        }

        return variance / normSize;
    }

    __aicore__ inline void Normalize(uint32_t rowIdx, float mean, float variance)
    {
        float invStd = 1.0f / sqrt(variance + eps);

        for (uint32_t tileId = 0; tileId < nTiles; tileId++) {
            uint32_t offset = rowIdx * normSize + tileId * tileLength;
            uint32_t length = (tileId == nTiles - 1) ? (normSize - tileId * tileLength) : tileLength;

            LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
            LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();

            DataCopy(inputLocal, inputGm[offset], length);

            // 问题:Scalar 循环执行归一化,未使用 Vector Unit
            for (uint32_t i = 0; i < length; i++) {
                float val = inputLocal.GetValue(i);
                float normalized = (val - mean) * invStd;
                outputLocal.SetValue(i, normalized);  // Scalar 写 LocalTensor
            }

            // 问题:没有确保 Scalar 写入完成
            DataCopy(outputGm[offset], outputLocal, length);

            inQueue.FreeTensor(inputLocal);
            outQueue.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
        uint32_t startRow = GetBlockIdx() * rowsPerCore;
        uint32_t endRow = min(startRow + rowsPerCore, batchSize);

        for (uint32_t row = startRow; row < endRow; row++) {
            // 问题:三阶段串行执行,多次遍历数据
            float mean = ComputeMean(row);       // 第一遍
            float variance = ComputeVariance(row, mean);  // 第二遍
            Normalize(row, mean, variance);      // 第三遍
        }
    }
};
```

## 问题分析

### 1. 多次遍历数据,内存带宽浪费
**当前流程**:
```
Pass 1 (ComputeMean):     GM → UB → Scalar (sum)
Pass 2 (ComputeVariance): GM → UB → Scalar (variance)
Pass 3 (Normalize):       GM → UB → Scalar → Vector (可能) → UB → GM
```

**问题**:
- 每个 row 需要 **3 次完整的 GM 读取**
- 内存带宽利用率 = 1/3 理论值
- Cache 无法复用,每次都是 cold miss

### 2. Scalar 密集循环,未使用 Vector Unit
```cpp
for (uint32_t i = 0; i < length; i++) {
    sum += inputLocal.GetValue(i);  // Scalar 串行
}
```

**效率对比**:
| 操作 | Scalar 循环 | Vector 指令 | 加速比 |
|------|------------|-------------|--------|
| FP32 加法 | 1 op/cycle | 128 ops/cycle | 128x |
| FP16 加法 | 1 op/cycle | 256 ops/cycle | 256x |

### 3. Scalar-Vector 数据传递缺少同步
```cpp
outputLocal.SetValue(i, normalized);  // Scalar 写
DataCopy(outputGm[offset], outputLocal, length);  // MTE3 读
```

**风险**:
- Scalar `SetValue` 写入 LocalTensor 后,没有 `S_MTE3` 同步
- MTE3 可能读取到未完成的 Scalar 写入

### 4. 缺少 PipeBarrier 和硬件事件
- 没有 `PipeBarrier<PIPE_V>` 保证 Vector 操作完成
- 没有 `MTE2_S` 保证 DMA 搬入完成再 Scalar 读取
- 没有 `V_S` 保证 Vector 归约完成再 Scalar 读取结果

## 典型问题表现

- **性能远低于预期**: 3x 内存带宽浪费 + Scalar 瓶颈
- **随机数值错误**: Scalar-Vector/Scalar-DMA 数据竞争
- **大维度性能崩溃**: normSize > 4096 时,多次遍历开销巨大

## 性能影响

| 指标 | Base (多遍) | 理论最优 (单遍) |
|------|-----------|---------------|
| GM 读带宽利用 | 33% (3 遍) | 100% (1 遍) |
| 计算单元 | Scalar Only | Vector + Scalar |
| 总体性能 | 基准 | **8-15x** |

**实测 (batchSize=128, normSize=1024)**:
- Base: 2.5 ms
- 理论最优: ~0.2 ms

# Base Code: 多核归约缺少 AtomicAdd 处理边界

来源:rms_norm_grad (lingxi-code - 推断)

```cpp
template <typename T>
class KernelRmsNormGrad {
    __aicore__ inline void CopyOutDGamma(uint32_t colIdx)
    {
        LocalTensor<float> dgammaLocal = dgammaQueue.DeQue<float>();

        // 问题:多核并行归约到同一个全局位置,没有原子操作保护
        // 每个 Core 计算 dgamma 的部分结果,需要累加到全局
        DataCopy(dgammaGm[colIdx], dgammaLocal, blockSize);

        dgammaQueue.FreeTensor(dgammaLocal);
    }

    __aicore__ inline void ReduceDGamma()
    {
        // 每个 Core 处理部分行,但输出到相同的 dgamma 位置
        for (uint32_t rowIdx = startRow; rowIdx < endRow; rowIdx++) {
            // 计算当前行对 dgamma 的贡献
            LocalTensor<float> dyLocal = dyQueue.DeQue<float>();
            LocalTensor<float> xNormLocal = xNormQueue.DeQue<float>();
            LocalTensor<float> dgammaLocal = dgammaBuffer.Get<float>();

            // 逐元素乘法: dgamma += dy * xNorm
            Mul(dgammaLocal, dyLocal, xNormLocal, numCol);

            // 问题:多个 Core 写入同一个 dgammaGm 地址,产生竞争
            DataCopy(dgammaGm[0], dgammaLocal, numCol);

            dyQueue.FreeTensor(dyLocal);
            xNormQueue.FreeTensor(xNormLocal);
        }
    }

    __aicore__ inline void ProcessTailBlock()
    {
        // 最后一个 Core 处理尾块
        uint32_t tailLen = numCol % blockSize;
        if (tailLen == 0) return;

        uint32_t tailOffset = (numCol / blockSize) * blockSize;
        LocalTensor<float> dgammaLocal = dgammaBuffer.Get<float>();

        // 计算尾块的 dgamma
        for (uint32_t rowIdx = startRow; rowIdx < endRow; rowIdx++) {
            // ... 尾块计算 ...
        }

        // 问题:尾块写入时没有考虑跨 Core 边界
        // 可能与前一个 Core 的对齐块重叠
        DataCopy(dgammaGm[tailOffset], dgammaLocal, tailLen);
    }
};
```

## 问题分析

### 1. 多核写同一地址缺少原子保护
**场景**: BatchNorm / LayerNorm 的梯度归约
```
Core 0: 处理 row [0, 32)   → 累加到 dgamma[0:1024]
Core 1: 处理 row [32, 64)  → 累加到 dgamma[0:1024]
Core 2: 处理 row [64, 96)  → 累加到 dgamma[0:1024]
...
```

**问题**:
- 多个 Core 同时 `DataCopy` 到 `dgammaGm[0]`
- 后写入的数据覆盖先写入的数据
- 最终结果只保留最后一个 Core 的计算,其他 Core 的贡献丢失

### 2. 尾块与对齐块边界重叠
**场景**: numCol = 1020,blockSize = 32
```
Aligned blocks: [0, 32), [32, 64), ..., [992, 1024)
Tail block:     [1024, 1020)  ← 不存在,实际是 [1020, 1024) padding
```

但实际情况:
```
Block 31: [992, 1024)  ← 包含有效数据 [992, 1020) 和 padding [1020, 1024)
```

**问题**:
- 尾块写入时,末尾 4 个元素被 padding 覆盖
- 如果多个 Core 写入,可能导致部分有效数据丢失

### 3. 跨 Core 边界的非对齐写入
**场景**: Core N 负责 [0, 1000),Core N+1 负责 [1000, 2000)
```
Core N 最后写入:   [992, 1024)  ← 超出边界 [1020, 1024)
Core N+1 首次写入: [1000, 1032) ← 起始非对齐
```

**问题**:
- [1000, 1020) 区间被两个 Core 同时写入
- [1020, 1024) 区间被 Core N 的 padding 覆盖

### 4. 数据竞争导致非确定性结果
- **无同步机制**: 多个 Core 写入顺序不确定
- **覆盖而非累加**: 后写入覆盖先写入
- **随机性**: 不同运行结果不同,难以调试

## 典型问题表现

- **梯度错误**: dgamma 的值远小于预期(只保留了最后一个 Core 的贡献)
- **训练不收敛**: 梯度更新不正确,loss 震荡或发散
- **非确定性**: 多次运行结果不同
- **尾块数据异常**: 最后几个元素的梯度为 0 或异常值

## 性能影响

| 指标 | 影响 |
|------|------|
| **正确性** | 严重错误 (多核结果覆盖) |
| **数值精度** | 梯度丢失 > 90% |
| **训练收敛** | 无法收敛或严重延迟 |

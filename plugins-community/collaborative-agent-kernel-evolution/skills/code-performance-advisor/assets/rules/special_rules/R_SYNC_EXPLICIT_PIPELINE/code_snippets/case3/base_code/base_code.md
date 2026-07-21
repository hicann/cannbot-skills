# Base Code: 类型转换中缺少流水线同步

来源:add_rms_norm_cast (lingxi-code)

```cpp
template <typename T>
class KernelAddRmsNormCast {
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        // 搬入 x1 和 x2 数据
        LocalTensor<T> x1Local = inQueueX1.AllocTensor<T>();
        LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();

        // 问题:数据搬入后没有等待 MTE2 完成
        DataCopy(x1Local, x1Gm[offset], length);
        DataCopy(x2Local, x2Gm[offset], length);

        // 立即入队,没有确保 DMA 完成
        inQueueX1.EnQue(x1Local);
        inQueueX2.EnQue(x2Local);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<T> x1Local = inQueueX1.DeQue<T>();
        LocalTensor<T> x2Local = inQueueX2.DeQue<T>();

        // 问题:多步 Cast 和计算没有显式同步
        LocalTensor<float> x1Fp32 = castBuf.Get<float>();
        LocalTensor<float> x2Fp32 = tmpBuf.Get<float>();

        // Cast x1 from half/bf16 to fp32
        Cast(x1Fp32, x1Local, RoundMode::CAST_NONE, tileLength);

        // 问题:没有 PipeBarrier,直接进行下一步
        // Vector Unit 的 Cast 可能未完成

        // Cast x2 from half/bf16 to fp32
        Cast(x2Fp32, x2Local, RoundMode::CAST_NONE, tileLength);

        // 立即使用,可能读到未完成的 Cast 结果
        Add(x1Fp32, x1Fp32, x2Fp32, tileLength);

        // 问题:Add 后没有同步,直接 Cast 回
        Cast(x1Local, x1Fp32, RoundMode::CAST_RINT, tileLength);

        inQueueX1.FreeTensor(x1Local);
        inQueueX2.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> outputLocal = outQueue.DeQue<T>();

        // 问题:没有确保 Compute 的 Vector 操作完成
        // 就开始 MTE3 搬出
        DataCopy(yGm[offset], outputLocal, length);

        outQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void Process()
    {
        uint32_t nTiles = (hiddenSize + tileLength - 1) / tileLength;

        // 问题:简单的顺序执行,没有流水线重叠
        for (uint32_t i = 0; i < nTiles; i++) {
            CopyIn(i * tileLength, tileLength);
            Compute();
            CopyOut(i * tileLength, tileLength);
        }
    }
};
```

## 问题分析

### 1. Cast 指令链缺少 PipeBarrier
**问题链路**:
```
Cast(x1Fp32, x1Local) → [缺少同步] → Cast(x2Fp32, x2Local)
Cast(x2Fp32, x2Local) → [缺少同步] → Add(x1Fp32, x1Fp32, x2Fp32)
Add(...) → [缺少同步] → Cast(x1Local, x1Fp32)
```

**风险**:
- **读写冲突**: Add 可能在 Cast 完成前读取 x1Fp32/x2Fp32
- **缓冲区复用错误**: 下一个 Cast 可能覆盖未完成的上一个 Cast 的缓冲区
- **精度损失**: 混合精度计算的中间结果未稳定,导致数值错误

### 2. MTE2 → Vector Unit 缺少同步
- **搬入 (DataCopy)** 后立即 **EnQue** 并 **DeQue** 使用
- 没有 `MTE2_V` 事件同步,Vector Unit 可能读到部分传输的数据
- 特别是 BF16/FP16 数据,小尺寸传输更容易出现时序问题

### 3. Vector Unit → MTE3 缺少同步
- **Compute** 完成后立即 **CopyOut**
- 没有 `V_MTE3` 事件同步,MTE3 可能搬出未完成的计算结果
- 输出数据可能包含中间状态或垃圾值

### 4. 流水线深度不足
- 单缓冲设计 (queue depth = 1)
- CopyIn → Compute → CopyOut 串行执行
- 无法形成 Copy-Compute-Copy 的三级流水线

## 典型问题表现

- **FP16/BF16 精度异常**: Cast 链中断导致数值错误
- **输出随机错误**: 某些 tile 输出异常,其他正常
- **大 hidden size 失败**: 数据量大时,时序问题放大
- **性能未达预期**: 串行执行,无流水线重叠

## 性能影响

| 指标 | 当前实现 | 理论最优 |
|------|----------|---------|
| 流水线级数 | 1 (串行) | 3 (Copy-Compute-Copy) |
| MTE 利用率 | ~33% | ~90% |
| Vector Unit 利用率 | ~60% | ~95% |
| 总体吞吐 | 基准 | 1.8-2.5x |

# Base Code: 缺少显式流水线同步

来源：lingxi-code (adaptive_avg_pool3d - 推断)

```cpp
class KernelAdaptiveAvgPool3d {
    __aicore__ inline void ComputeAccumulate()
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();

        // 问题：没有显式同步，依赖隐式的 Queue 机制
        AscendC::Add(accumLocal, accumLocal, inputLocal, C);

        inQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void ComputeAverage()
    {
        AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();
        AscendC::LocalTensor<float> outputLocal = outQueue.AllocTensor<float>();

        // 问题：没有确保上一步的 Add 操作完成
        AscendC::Muls(outputLocal, accumLocal, avg_scale, C);

        outQueue.EnQue(outputLocal);
    }

    __aicore__ inline void Process()
    {
        // 简单的三阶段流水线
        for (uint32_t i = 0; i < num_iterations; i++) {
            CopyIn(i);
            ComputeAccumulate();
            CopyOut(i);
        }
    }
};
```

**问题**：

1. **数据依赖未显式保证**
   - Vector Unit 的计算结果何时可见？未明确
   - Scalar Unit 读取 Vector Unit 结果时可能读到旧值
   - 多个 Vector 操作之间的顺序未保证

2. **跨 Unit 数据竞争风险**
   - MTE2 (Memory Transfer Engine) 搬入数据后，Vector Unit 何时可以读？
   - Vector Unit 计算完成后，MTE3 何时可以搬出？
   - Scalar Unit 何时可以读取 Vector Unit 的计算结果？

3. **流水线深度受限**
   - 依赖隐式 Queue 机制，深度有限
   - 无法精确控制各阶段的并行度
   - 可能出现流水线停顿 (stall)

4. **调试困难**
   - 数据竞争问题随机出现，难以复现
   - 不同平台/编译器优化可能导致不同行为
   - 性能波动大，原因难以定位

**典型问题表现**：
- 计算结果不稳定，随机出错
- 性能时好时坏，无法预测
- 在某些输入 shape 下失败

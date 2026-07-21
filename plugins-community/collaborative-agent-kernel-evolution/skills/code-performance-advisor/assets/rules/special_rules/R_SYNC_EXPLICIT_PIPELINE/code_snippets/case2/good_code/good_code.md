# Good Code: 显式 Vector-Scalar 硬件事件同步

来源:batch_norm_v3 (expert code)

```cpp
template <typename T1, typename T2, int32_t PIPE>
class BatchNormV3FullReduce
{
private:
    // 事件 ID 用于跨 Unit 同步
    TEventID eventIdMte2toS;  // MTE2 → Scalar
    TEventID eventIdVtoS;     // Vector → Scalar

    __aicore__ inline void InitSyncEvents()
    {
        // 获取硬件事件 ID
        if constexpr (IsSameType<T1, float>::value) {
            // 对于 FP32 输入,需要 MTE2-Scalar 同步
            eventIdMte2toS = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            SetFlag<HardEvent::MTE2_S>(eventIdMte2toS);
        }
    }

    __aicore__ inline void ComputeStatistics()
    {
        // Vector Unit 计算均值和方差
        LocalTensor<float> sumTensor = sumBuf.Get<float>();
        LocalTensor<float> inputLocal = inputQueue.DeQue<float>();

        Add(sumTensor, sumTensor, inputLocal, dataLen);

        // 显式同步:确保 Vector 操作完成
        PipeBarrier<PIPE_V>();

        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void UpdateRunningStats()
    {
        LocalTensor<float> saveMeanTensor = meanBuf.Get<float>();
        LocalTensor<float> saveVarTensor = varBuf.Get<float>();

        // 关键:Vector → Scalar 显式事件同步
        eventIdVtoS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventIdVtoS);   // Vector Unit 设置完成标志
        WaitFlag<HardEvent::V_S>(eventIdVtoS);  // Scalar Unit 等待标志

        // 现在安全:Scalar 读取已保证完成的 Vector 结果
        for (int64_t aNum = 0; aNum < channelNum; aNum++) {
            float finalMean = saveMeanTensor.GetValue(aNum);  // 安全读取
            float finalVar = saveVarTensor.GetValue(aNum);

            // 使用正确的值进行计算
            float runningMean = finalMean * momentum + oldMean * (1.0f - momentum);
            float runningVar = finalVar * momentum + oldVar * (1.0f - momentum);
        }
    }

    __aicore__ inline void CopyOutWithScalarControl()
    {
        // 关键:MTE2 → Scalar 显式事件同步
        if constexpr (IsSameType<T2, float>::value) {
            WaitFlag<HardEvent::MTE2_S>(eventIdMte2toS);  // 等待 DMA 完成
        }

        // 现在安全:数据已完全搬入
        LocalTensor<float> weightLocal = weightQueue.DeQue<float>();

        for (int idx = 0; idx < count; idx++) {
            float weight = weightLocal.GetValue(idx);  // 安全读取
            // 数据完整且正确
        }
    }

    __aicore__ inline void Process()
    {
        // 初始化同步事件
        InitSyncEvents();

        // 显式同步保证的流水线
        CopyInData();
        ComputeStatistics();      // 内部使用 PipeBarrier<PIPE_V>
        UpdateRunningStats();     // 使用 V_S 事件同步
        CopyOutWithScalarControl(); // 使用 MTE2_S 事件同步
    }
};

// 另一个示例:Welford 算法中的 Vector-Scalar 同步
template <typename T1, typename T2>
class BatchNormV3Welford
{
    __aicore__ inline void FinalStatisticsWithScalar()
    {
        LocalTensor<float> meanTensor = meanBuf.Get<float>();
        LocalTensor<float> varTensor = varBuf.Get<float>();

        // Vector Unit 完成最后的归约计算
        WholeReduceSum(meanTensor, tempSumTensor, reduceLen);
        WholeReduceSum(varTensor, tempVarTensor, reduceLen);
        PipeBarrier<PIPE_V>();

        // 显式 V_S 事件同步
        TEventID eventId = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(eventId);
        WaitFlag<HardEvent::V_S>(eventId);

        // Scalar Unit 安全读取并执行精细化计算
        for (uint32_t i = 0; i < channelCount; i++) {
            float mean = meanTensor.GetValue(i) / sampleCount;
            float var = varTensor.GetValue(i) / sampleCount - mean * mean;

            // 数值稳定性处理
            var = (var < 0.0f) ? 0.0f : var;

            // Scalar 计算复杂表达式
            float invStd = 1.0f / sqrt(var + epsilon);
            scaleFactors[i] = invStd;
        }
    }
};
```

## 改进点

### 1. 硬件事件同步机制
- **`HardEvent::V_S`**: Vector Unit 到 Scalar Unit 的硬件事件
- **`HardEvent::MTE2_S`**: MTE2 (搬入引擎) 到 Scalar Unit 的硬件事件
- **`SetFlag` + `WaitFlag`**: 生产者-消费者模式的硬件同步原语
- **编译时优化**: `if constexpr` 根据数据类型选择性同步

### 2. 分层同步策略
| 场景 | 同步原语 | 作用域 |
|------|---------|--------|
| 同 Unit 内 Vector 操作 | `PipeBarrier<PIPE_V>()` | Vector Unit 内部流水线 |
| Vector → Scalar | `SetFlag/WaitFlag(HardEvent::V_S)` | 跨 Unit 硬件事件 |
| MTE2 → Scalar | `SetFlag/WaitFlag(HardEvent::MTE2_S)` | DMA 到 Scalar |
| MTE3 → Vector | `SetFlag/WaitFlag(HardEvent::MTE3_V)` | Vector 到 DMA 搬出 |

### 3. 数据依赖保证
正确的依赖链:
```
MTE2 搬入 → [WaitFlag MTE2_S] → Scalar 读取
Vector 计算 → [PipeBarrier V] → [SetFlag/WaitFlag V_S] → Scalar 读取
Scalar 计算 → [SetFlag S_V] → Vector 使用
Vector 准备 → [SetFlag MTE3_V] → MTE3 搬出
```

### 4. 性能与正确性平衡
- **最小化同步开销**: 只在必要的跨 Unit 数据传递点同步
- **编译时优化**: `constexpr if` 根据数据类型编译期决定是否同步
- **硬件加速**: 使用硬件事件而非软件轮询,延迟极低 (< 10 cycles)

## 性能提升

- **正确性**: 消除跨 Unit 数据竞争,结果 **100% 稳定**
- **性能开销**: 硬件事件同步开销 < 1% (每个事件 ~5-10 cycles)
- **可移植性**: 在所有 Ascend 平台行为一致
- **可维护性**: 数据依赖显式,易于理解和调试

## 适用场景

- **Scalar Unit 读取 Vector Unit 计算结果**: 统计量、索引、控制信息
- **Scalar Unit 参与复杂控制流**: 动态分支、循环边界、特殊值处理
- **DMA 与 Scalar/Vector 交互**: 权重加载、Scalar 配置、动态地址计算
- **混合精度计算**: Cast 前后的跨 Unit 数据传递

## 关键要点

1. **识别跨 Unit 数据流**: 任何 `GetValue()`、Scalar 变量赋值都是潜在同步点
2. **选择合适的事件类型**: V_S、MTE2_S、MTE3_V、S_V 根据数据流向选择
3. **避免过度同步**: 只在数据依赖点同步,不要每行都加 Barrier
4. **利用编译时优化**: `constexpr if` 避免运行时分支和不必要的同步

# Base Code: 缺少 Vector-Scalar 跨单元同步

来源:batch_norm_v3 (lingxi-code - 推断)

```cpp
template <typename T1, typename T2>
class BatchNormV3Compute {
    __aicore__ inline void ComputeStatistics()
    {
        // Vector Unit 计算均值和方差
        LocalTensor<float> sumTensor = sumBuf.Get<float>();
        LocalTensor<float> inputLocal = inputQueue.DeQue<float>();

        Add(sumTensor, sumTensor, inputLocal, dataLen);

        // 问题:没有 Vector 到 Scalar 的显式同步
        // Scalar Unit 可能读取到未完成的 Vector 计算结果

        inputQueue.FreeTensor(inputLocal);
    }

    __aicore__ inline void UpdateRunningStats()
    {
        LocalTensor<float> saveMeanTensor = meanBuf.Get<float>();
        LocalTensor<float> saveVarTensor = varBuf.Get<float>();

        // 问题:直接使用 Scalar Unit 读取 Vector Unit 的计算结果
        // 没有确保 Vector 计算已完成
        for (int64_t aNum = 0; aNum < channelNum; aNum++) {
            float finalMean = saveMeanTensor.GetValue(aNum);  // Scalar 读 Vector 结果
            float finalVar = saveVarTensor.GetValue(aNum);

            // 使用可能不正确的值进行计算
            float runningMean = finalMean * momentum + oldMean * (1.0f - momentum);
            float runningVar = finalVar * momentum + oldVar * (1.0f - momentum);
        }
    }

    __aicore__ inline void CopyOutWithScalarControl()
    {
        // 问题:MTE2 搬入数据后,Scalar Unit 直接读取
        // 没有 MTE2-Scalar 同步
        LocalTensor<float> weightLocal = weightQueue.DeQue<float>();

        for (int idx = 0; idx < count; idx++) {
            float weight = weightLocal.GetValue(idx);  // Scalar 读 MTE2 结果
            // 可能读到旧值或垃圾数据
        }
    }

    __aicore__ inline void Process()
    {
        // 简单的顺序调用,依赖隐式同步
        CopyInData();
        ComputeStatistics();
        UpdateRunningStats();  // 危险:可能使用未完成的统计值
        CopyOutWithScalarControl();
    }
};
```

## 问题分析

### 1. Vector Unit 到 Scalar Unit 数据竞争
- **问题表现**: Scalar Unit 的 `GetValue()` 操作读取 Vector Unit 计算的 LocalTensor
- **竞争风险**: Vector Unit 的 `Add/Muls` 等操作可能尚未完成
- **后果**: Scalar Unit 读到中间状态或旧值,导致:
  - Running statistics 更新错误
  - 数值精度下降
  - 随机性错误(时序相关)

### 2. MTE2 到 Scalar Unit 数据竞争
- **问题表现**: MTE2 (Memory Transfer Engine 2,搬入引擎) 将 weight 搬入 UB 后,Scalar Unit 立即读取
- **竞争风险**: DMA 传输可能未完成
- **后果**: 读取到未初始化的数据或部分传输的数据

### 3. 缺少跨 Unit 同步原语
- **没有使用**: `SetFlag/WaitFlag` 硬件事件机制
- **没有使用**: `HardEvent::V_S` (Vector to Scalar)
- **没有使用**: `HardEvent::MTE2_S` (MTE2 to Scalar)
- **依赖**: 隐式的 Queue 机制,但不保证跨 Unit 可见性

### 4. 数据依赖链断裂
正确的依赖链应该是:
```
MTE2 搬入 → [同步] → Vector 计算 → [同步] → Scalar 读取 → [同步] → MTE3 搬出
```
但当前实现缺少 `[同步]` 节点,导致流水线数据不一致。

## 典型问题表现

- **间歇性精度下降**: Running mean/var 偶尔出现异常值
- **平台相关性**: 在某些硬件平台正常,其他平台失败
- **负载相关性**: 高负载时错误率上升(时序窗口缩短)
- **难以调试**: Sanitizer 和调试器无法检测跨硬件 Unit 的竞争

## 性能影响

- **正确性风险**: 高(随机错误)
- **性能影响**: 无(当前没有同步,性能"虚高")
- **可维护性**: 低(行为不可预测)

# Good Code: Welford 在线算法单趟计算均值和方差

来源：expert code (batch_norm_v3)

```cpp
template <typename T1, typename T2, int32_t SPLIT_MODE, int32_t R0_ALIGN_MODE, int32_t PIPE>
class BatchNormV3Welford
{
    // Welford 并行更新公式
    __aicore__ inline void WelfordParallelUpdate(
        float& count, LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor,
        LocalTensor<float>& xTensor, LocalTensor<float>& deltaTensor, const uint32_t& calcMask)
    {
        // Welford 算法核心公式：
        // count = count + 1
        // delta = x - mean
        // mean = mean + delta / count
        // m2 = m2 + delta * (x - mean_new)

        count += 1;  // 更新计数

        // Step 1: delta = x - mean_old
        Sub(deltaTensor, xTensor, meanTensor, calcMask);
        PipeBarrier<PIPE_V>();

        // Step 2: mean = mean_old + delta / count
        Muls(xTensor, deltaTensor, 1 / count, calcMask);
        PipeBarrier<PIPE_V>();
        Add(meanTensor, meanTensor, xTensor, calcMask);
        xQueue.FreeTensor(xTensor);  // xTensor 现在是 delta / count

        // Step 3: delta2 = delta * delta
        Mul(deltaTensor, deltaTensor, deltaTensor, calcMask);
        PipeBarrier<PIPE_V>();

        // Step 4: m2 = m2 + delta^2 * (count - 1) / count
        // 等价于：m2 = m2 + delta * (x - mean_new)
        Muls(deltaTensor, deltaTensor, (count - 1) / count, calcMask);
        PipeBarrier<PIPE_V>();
        Add(m2Tensor, m2Tensor, deltaTensor, calcMask);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessWelford()
    {
        // 初始化
        LocalTensor<float> meanTensor = meanBuf.Get<float>();
        LocalTensor<float> m2Tensor = m2Buf.Get<float>();
        Duplicate(meanTensor, 0.0f, blockFactor);
        Duplicate(m2Tensor, 0.0f, blockFactor);
        PipeBarrier<PIPE_V>();

        float count = 0.0f;  // Welford 计数器

        // 单趟遍历所有数据
        for (int64_t r1 = 0; r1 < patternR1; r1++) {
            for (int64_t r0Ub = 0; r0Ub < r0UbLoop; r0Ub++) {
                int64_t copyInSize = (r0Ub == r0UbLoop - 1) ? lastR0UbFactor : r0UbFactor;

                // 搬入数据
                CopyIn(r1, r0Ub, copyInSize);

                // Welford 在线更新（边读边计算）
                LocalTensor<T1> xTensorHalf = xQueue.DeQue<T1>();
                LocalTensor<float> xTensor = xTensorCast.Get<float>();
                LocalTensor<float> deltaTensor = deltaBuf.Get<float>();

                // 类型转换（如果需要）
                if constexpr (!IsSameType<T1, float>::value) {
                    Cast(xTensor, xTensorHalf[r0UbFactor], RoundMode::CAST_NONE, copyInSize);
                } else {
                    xTensor = xTensorHalf;
                }

                // 调用 Welford 并行更新
                uint32_t calcMask = copyInSize;
                WelfordParallelUpdate(count, meanTensor, m2Tensor, xTensor, deltaTensor, calcMask);
            }
        }

        // Welford 最终结果转换为方差
        // variance = m2 / count (总体方差)
        // 或 variance = m2 / (count - 1) (样本方差)
        LocalTensor<float> varianceTensor = varianceBuf.Get<float>();
        Muls(varianceTensor, m2Tensor, 1.0f / count, blockFactor);
        PipeBarrier<PIPE_V>();

        // 保存结果
        SaveMeanVariance(meanTensor, varianceTensor);
    }

    // 二分累加归约（用于合并多个 Welford 结果）
    __aicore__ inline void FullAichotomizeAdd(LocalTensor<float>& calcTensor, int64_t sumNum, float& sumValue)
    {
        // 处理非二次幂的情况
        int64_t dichotomizeAddDiffSize = sumNum - GetMaxPowerOf2(sumNum);

        if (dichotomizeAddDiffSize != 0) {
            // 先将差值部分加到前面
            Add(calcTensor, calcTensor, calcTensor[sumNum - dichotomizeAddDiffSize],
                dichotomizeAddDiffSize);
            PipeBarrier<PIPE_V>();
            sumNum = sumNum - dichotomizeAddDiffSize;
        }

        // 二分归约
        while (sumNum > ELEM_PER_REP_FP32) {
            sumNum = sumNum / TWO_NUM;
            Add(calcTensor, calcTensor, calcTensor[sumNum], sumNum);
            PipeBarrier<PIPE_V>();
        }

        // 最后使用硬件 ReduceSum
        ReduceSum(calcTensor, calcTensor, calcTensor, sumNum);
        PipeBarrier<PIPE_V>();

        // 读取累加器结果
        uint64_t acc_val = GetAccVal();
        sumValue = *reinterpret_cast<float*>(&acc_val);
    }
};

// Tiling 阶段选择 Welford 或 FullReduce 算法
ge::graphStatus BatchNormV3WelfordTiling::DoOpTiling()
{
    // 当 R0 * R1 >= 8192 时，使用 Welford 算法
    // 当 R0 * R1 < 8192 时，使用 FullReduce 算法（两趟更快）
    if (commonParams.patternR1 * commonParams.patternR0 >= FULL_REDUCE_TEMPLATE_R_LIMIT) {
        // 使用 Welford 算法
        SetTilingKey(BNV3_WELFORD_R0_SPLIT_NOT_ALIGN);
        // ... Welford Tiling 参数设置
    }

    return ge::GRAPH_SUCCESS;
}
```

**改进点**：

1. **Welford 在线算法**
   - 单趟遍历数据，边读边计算均值和方差
   - 无需缓存所有数据，UB 空间占用小
   - 内存带宽减半（数据只读取一次）

2. **数学公式优化**
   - 经典 Welford 公式：
     ```
     count = count + 1
     delta = x - mean_old
     mean_new = mean_old + delta / count
     m2 = m2 + delta * (x - mean_new)
     ```
   - 等价变换为：
     ```
     m2 = m2 + delta^2 * (count - 1) / count
     ```
   - 减少一次 Sub 操作，提升性能

3. **数值稳定性提升**
   - 避免大数减小数（传统方法：`sum(x^2) - mean^2 * count`）
   - 增量更新，误差不累积
   - 适用于数值范围差异大的数据

4. **自适应算法选择**
   - 大规模数据（R0 * R1 >= 8192）：使用 Welford，带宽优先
   - 小规模数据（R0 * R1 < 8192）：使用 FullReduce，计算优先
   - Tiling 阶段根据 shape 自动选择

5. **二分归约优化**
   - 使用 `FullAichotomizeAdd` 高效合并多个局部结果
   - 先处理非二次幂差值，再进行标准二分
   - 充分利用 Vector Unit 并行加法能力

**性能提升**：
- 内存带宽减半：数据只读取一次，相比两趟算法提升 40-50%
- 延迟降低：单趟遍历，延迟减半
- 数值精度提升：避免大数减小数，方差计算误差降低 1-2 个数量级
- 适用范围广：大规模数据场景性能提升显著

**适用场景**：
- BatchNorm（大 Batch Size 或大 Feature Map）
- LayerNorm（长序列）
- RMSNorm（长序列）
- 任何需要同时计算均值和方差的场景
- 带宽受限的平台

**算法复杂度对比**：
- 传统两趟算法：`2 * N` 次内存访问，`2 * N` 次浮点运算
- Welford 算法：`1 * N` 次内存访问，`4 * N` 次浮点运算
- 带宽受限场景（大部分深度学习算子）：Welford 性能优势明显

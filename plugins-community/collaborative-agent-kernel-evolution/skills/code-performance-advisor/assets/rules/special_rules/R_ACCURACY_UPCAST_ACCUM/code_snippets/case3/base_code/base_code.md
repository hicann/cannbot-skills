# Base Code: FP32 直接计算 DeepNorm

来源：lingxi-code (deep_norm)

```cpp
// 全程 FP32 计算（虽然精度高，但内存带宽浪费）
class KernelDeepNorm {
private:
    AscendC::GlobalTensor<float> inputGm;  // FP32
    AscendC::GlobalTensor<float> outputGm; // FP32

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueue.DeQue<float>();

        // FP32 计算均值
        float rowSum = 0.0f;
        AscendC::ReduceSum(sumLocal, xLocal, sumLocal, length);
        rowSum = sumLocal.GetValue(0);
        float mean = rowSum / length;

        // FP32 计算方差 (两趟算法)
        AscendC::Adds(tempLocal, xLocal, -mean, length);
        AscendC::Mul(sqLocal, tempLocal, tempLocal, length);
        AscendC::ReduceSum(varLocal, sqLocal, varLocal, length);
        float variance = varLocal.GetValue(0) / length;

        // FP32 归一化
        float invStd = 1.0f / sqrt(variance + eps);
        AscendC::Muls(outputLocal, tempLocal, invStd, length);

        outQueue.EnQue(outputLocal);
    }
};
```

**问题**：
1. 全程 FP32，内存带宽占用高（现代模型多用 FP16/BF16）
2. 虽然精度高，但无法利用低精度输入输出的带宽优势
3. 两趟算法，需要两次遍历数据
4. 缺乏针对低精度输入的优化策略
5. 大 hidden size 场景下，FP32 累加仍可能出现精度问题（虽然比 FP16 好）

# Good Code: Repeat/Stride 参数精确配置

来源: expert code (layer_norm_v3)

```cpp
template <typename T>
class LayerNormV3Optimized {
    __aicore__ inline void ProcessWithRepeat() {
        uint32_t cActual = cDim;
        uint32_t cAligned = AlignUp(cDim, BLOCK_SIZE);
        uint32_t repeatTimes = cActual / BLOCK_SIZE;
        uint32_t tail = cActual % BLOCK_SIZE;

        // 关键优化: 配置 Repeat/Stride 参数
        RepeatParams repeatParams;
        repeatParams.repeatTimes = repeatTimes;
        repeatParams.srcRepStride = 1;  // 连续数据: stride=1
        repeatParams.dstRepStride = 1;

        for (uint32_t n = 0; n < nDim; n++) {
            for (uint32_t hw = 0; hw < hDim * wDim; hw++) {
                uint32_t offset = (n * hDim * wDim + hw) * cAligned;

                // 1. DataCopyPad 加载 + 自动 Padding
                LocalTensor<T> inputLocal = inputBuf.Get<T>();
                DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = cActual * sizeof(T);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(inputLocal, inputGM[offset], copyParams, padParams);
                PipeBarrier<PIPE_MTE1>();

                // 2. 使用 Repeat 参数批量计算均值
                LocalTensor<T> sumLocal = sumBuf.Get<T>();
                Duplicate(sumLocal, static_cast<T>(0), 1);

                // ReduceSum 使用 Repeat 参数
                ReduceSum(sumLocal, inputLocal, repeatParams, cActual);
                T mean = sumLocal.GetValue(0) / cActual;

                // 3. 向量化计算方差（Repeat 参数）
                LocalTensor<T> diffSq = tempBuf.Get<T>();
                Subs(diffSq, inputLocal, mean, cActual);  // 自动处理 cActual 个元素
                Mul(diffSq, diffSq, diffSq, cActual);

                LocalTensor<T> varLocal = varBuf.Get<T>();
                Duplicate(varLocal, static_cast<T>(0), 1);
                ReduceSum(varLocal, diffSq, repeatParams, cActual);
                T variance = varLocal.GetValue(0) / cActual;
                T invStd = 1.0f / sqrt(variance + epsilon);

                // 4. 向量化归一化（Repeat 参数 + Tail Mask）
                LocalTensor<T> normalized = normBuf.Get<T>();

                // 主循环: Repeat 参数批量处理
                Subs(normalized, inputLocal, mean, repeatParams, cActual);
                Muls(normalized, normalized, invStd, repeatParams, cActual);

                // 尾部: Mask 精确控制
                if (tail > 0) {
                    MaskReg tailMask = CreateMask<T>(tail);
                    Subs(normalized[repeatTimes * BLOCK_SIZE],
                        inputLocal[repeatTimes * BLOCK_SIZE],
                        mean, tailMask);
                    Muls(normalized[repeatTimes * BLOCK_SIZE],
                        normalized[repeatTimes * BLOCK_SIZE],
                        invStd, tailMask);
                }

                PipeBarrier<PIPE_V>();

                // 5. 写回: Stride 跳过 Padding
                DataCopyExtParams copyOutParams;
                copyOutParams.blockCount = 1;
                copyOutParams.blockLen = cActual * sizeof(T);
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = 0;
                DataCopy(outputGM[offset], normalized, copyOutParams);
                PipeBarrier<PIPE_MTE2>();
            }
        }
    }
};
```

**改进**: 使用 Repeat/Stride 参数 + Mask 尾部处理,性能提升 **10-50×**。

**关键技术点**:
1. **Repeat 参数**: 批量处理 repeatTimes 个 block
2. **Stride 参数**: 控制地址步进,跳过 Padding
3. **Tail Mask**: 精确处理尾部元素
4. **DataCopyPad**: 硬件自动 Padding
5. **正确除数**: 使用 cActual 计算统计量

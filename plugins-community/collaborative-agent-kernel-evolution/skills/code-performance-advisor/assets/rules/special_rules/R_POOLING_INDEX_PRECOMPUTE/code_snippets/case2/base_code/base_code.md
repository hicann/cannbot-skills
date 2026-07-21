# Base Code: 嵌套循环标量计算多维索引

来源: lingxi-code (max_pool_with_argmax_v3, 推断)

```cpp
template <typename T, typename T2>
class MaxPoolWithArgmaxV3 {
    __aicore__ inline void ComputePooling() {
        // 假设输出维度: [batchSize, outH, outW, channels]
        uint32_t outputCount = batchSize * outH * outW * channels;

        for (uint32_t idx = 0; idx < outputCount; idx++) {
            // 问题1: 运行时标量计算 4D 坐标
            uint32_t n = idx / (outH * outW * channels);
            uint32_t tmp = idx % (outH * outW * channels);
            uint32_t h = tmp / (outW * channels);
            tmp = tmp % (outW * channels);
            uint32_t w = tmp / channels;
            uint32_t c = tmp % channels;

            // 问题2: 每个输出点都要计算 Pooling 窗口的起始位置
            uint32_t hStart = h * strideH - padH;
            uint32_t wStart = w * strideW - padW;

            // 问题3: 嵌套循环遍历 Pooling 窗口
            float maxVal = -3.402823e+38f;  // 近似负无穷
            int32_t maxIdx = 0;

            for (uint32_t kh = 0; kh < kernelH; kh++) {
                for (uint32_t kw = 0; kw < kernelW; kw++) {
                    uint32_t inputH = hStart + kh * dilationH;
                    uint32_t inputW = wStart + kw * dilationW;

                    // 边界检查
                    if (inputH >= 0 && inputH < height && inputW >= 0 && inputW < width) {
                        // 计算输入索引: [n, inputH, inputW, c]
                        uint32_t inputIdx = n * (height * width * channels) +
                                            inputH * (width * channels) +
                                            inputW * channels + c;

                        // 读取数据并比较
                        T val = inputGM[inputIdx];
                        if (static_cast<float>(val) > maxVal) {
                            maxVal = static_cast<float>(val);
                            maxIdx = static_cast<int32_t>(inputH * width + inputW);
                        }
                    }
                }
            }

            // 写回结果
            outputGM[idx] = static_cast<T>(maxVal);
            argmaxGM[idx] = static_cast<T2>(maxIdx);
        }
    }
};
```

**问题分析**:

1. **标量索引计算开销大**
   - 每个输出点都要通过除法/取模计算 4D 坐标 (n, h, w, c)
   - 除法运算在 Scalar Unit 执行，延迟高（~10-20 周期）
   - 对于大输出 shape（如 [32, 64, 64, 128]），计算次数 = 32 × 64 × 64 × 128 = 16M+
   - 索引计算占总执行时间的 15-30%

2. **嵌套循环无法向量化**
   - 三重嵌套循环（输出点 → kernel H → kernel W）
   - 每次迭代只处理一个元素
   - Scalar Unit 成为瓶颈，Vector Unit 空闲
   - 无法利用 SIMD 并行能力

3. **指令级并行度低**
   - 索引计算和数据访问紧密耦合
   - 存在数据依赖（idx → n, h, w, c → inputIdx → val）
   - 流水线停顿频繁
   - CPU 前端解码和后端执行不平衡

4. **编译器优化受限**
   - 循环内存在复杂控制流（边界检查 if 语句）
   - 分支预测失败率高
   - 循环展开受限（依赖运行时变量）
   - 自动向量化几乎不可能

5. **Cache 局部性差**
   - 索引计算和数据访问交织
   - 频繁在 Scalar 和 Vector 操作之间切换
   - 指令 Cache 和数据 Cache 都易 Miss
   - 内存带宽利用率低

**性能瓶颈定位**:

- **Scalar Unit 饱和**: 索引计算集中在 Scalar Unit，利用率 > 90%
- **Vector Unit 空闲**: Vector Unit 利用率 < 30%（仅用于简单的 max/compare）
- **内存停顿**: 标量访存模式导致频繁的 Cache Miss
- **流水线效率低**: 指令级并行度 < 2，流水线深度未充分利用

**典型问题场景**:

- Max/Avg Pooling 算子（NHWC 格式）
- 输出 shape 较大：[32, 64, 64, 128] 或 [64, 32, 32, 256]
- Kernel 尺寸适中：3×3、5×5
- 需要计算 argmax 索引
- 索引计算占比 > 10% 的场景

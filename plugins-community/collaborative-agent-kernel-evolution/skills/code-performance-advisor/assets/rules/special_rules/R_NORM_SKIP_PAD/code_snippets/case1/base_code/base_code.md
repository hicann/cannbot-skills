# Base Code: 逐行处理 Padding 数据

来源：lingxi-code (batch_norm_v3 - 推断，基于传统实现)

```cpp
class KernelBatchNorm {
    __aicore__ inline void ProcessNormalization()
    {
        AscendC::LocalTensor<float> xLocal = xBuf.Get<float>();
        AscendC::LocalTensor<float> meanLocal = meanBuf.Get<float>();
        AscendC::LocalTensor<float> outputLocal = outputBuf.Get<float>();

        // 问题：逐元素或逐行处理 (x - mean)
        for (int64_t lineIdx = 0; lineIdx < lineNum; lineIdx++) {
            for (int64_t elemIdx = 0; elemIdx < patternR0; elemIdx++) {
                int64_t offset = lineIdx * patternR0Align + elemIdx;

                // 逐元素减去均值
                float x_val = xLocal.GetValue(offset);
                float mean_val = meanLocal.GetValue(0);
                float result = x_val - mean_val;
                outputLocal.SetValue(offset, result);
            }

            // Padding 区域（patternR0 到 patternR0Align）无需处理，但占用空间
        }
    }
};
```

**问题**：

1. **循环开销大**
   - 使用 `for` 循环逐行或逐元素处理
   - 无法利用 Vector Unit 的 SIMD 并行能力
   - 大量循环控制开销

2. **Padding 区域浪费指令**
   - 每行有效数据 `patternR0`，对齐到 `patternR0Align`
   - Padding 区域（`patternR0Align - patternR0`）需要额外处理或跳过
   - 无法高效跳过 Padding 区域

3. **指令数量多**
   - 每次迭代需要：地址计算 + Load + Sub + Store
   - 对于大规模数据（如 lineNum = 1000, patternR0 = 512），指令数量庞大
   - 指令发射成为瓶颈

4. **repeat 限制未处理**
   - Vector 指令的 `repeat` 参数有硬件限制（通常 <= 255）
   - 当 lineNum > 255 时，需要额外的外层循环
   - 代码复杂度增加

**典型问题场景**：
- BatchNorm 的 Normalize 阶段（x - mean）/ sqrt(var + eps)
- LayerNorm 的 Normalize 阶段
- 数据对齐后有大量 Padding（如 patternR0 = 513, patternR0Align = 520）
- 行数很多（lineNum > 255）

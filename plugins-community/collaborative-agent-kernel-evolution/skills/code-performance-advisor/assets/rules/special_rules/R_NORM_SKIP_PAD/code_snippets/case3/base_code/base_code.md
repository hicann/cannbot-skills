# Base Code: 显式 Padding 处理,性能低效

来源: lingxi-code (deep_norm, 推断)

```cpp
template <typename T>
class DeepNorm {
    __aicore__ inline void Process() {
        // 假设输入: [N, H, W, C] 但 C 不对齐
        uint32_t cActual = cDim;          // 实际 C 维度,如 127
        uint32_t cAligned = AlignUp(cDim, 32);  // 对齐到 32,变为 128

        // 问题1: 显式 Padding 处理
        LocalTensor<T> inputPadded = paddedBuf.Get<T>();

        for (uint32_t i = 0; i < nDim * hDim * wDim; i++) {
            // 加载实际数据
            for (uint32_t c = 0; c < cActual; c++) {
                inputPadded.SetValue(i * cAligned + c, inputGM[i * cActual + c]);
            }

            // 问题2: 显式填充 Padding 区域为 0
            for (uint32_t c = cActual; c < cAligned; c++) {
                inputPadded.SetValue(i * cAligned + c, static_cast<T>(0));
            }
        }

        // 问题3: Vector 指令处理 Padding 区域（浪费计算）
        LocalTensor<T> normalized = normBuf.Get<T>();
        for (uint32_t i = 0; i < nDim * hDim * wDim; i++) {
            // 计算均值（包含 Padding 的 0 值）
            T sum = 0;
            for (uint32_t c = 0; c < cAligned; c++) {
                sum += inputPadded.GetValue(i * cAligned + c);
            }
            T mean = sum / static_cast<T>(cAligned);  // 问题: 除数包含 Padding

            // 计算方差（包含 Padding）
            T variance = 0;
            for (uint32_t c = 0; c < cAligned; c++) {
                T diff = inputPadded.GetValue(i * cAligned + c) - mean;
                variance += diff * diff;
            }
            variance = variance / static_cast<T>(cAligned);  // 问题: 除数错误

            // 归一化（包含 Padding 区域）
            T invStd = 1.0f / sqrt(variance + epsilon);
            for (uint32_t c = 0; c < cAligned; c++) {
                T normed = (inputPadded.GetValue(i * cAligned + c) - mean) * invStd;
                normalized.SetValue(i * cAligned + c, normed);
            }
        }

        // 问题4: 写回时需要再次跳过 Padding
        for (uint32_t i = 0; i < nDim * hDim * wDim; i++) {
            for (uint32_t c = 0; c < cActual; c++) {
                outputGM[i * cActual + c] = normalized.GetValue(i * cAligned + c);
            }
        }
    }
};
```

**问题分析**:

1. **显式 Padding 填充开销**
   - 需要逐元素填充 0 到 Padding 区域
   - Padding 比例 = (cAligned - cActual) / cAligned
   - 典型: C=127 → cAligned=128, Padding 占 0.8%
   - 极端: C=97 → cAligned=128, Padding 占 24%
   - 填充操作纯开销,无计算价值

2. **Padding 参与计算浪费**
   - Padding 的 0 值参与均值/方差计算
   - 虽然 Padding 为 0,但仍需加载/计算
   - Vector Unit 处理 Padding 区域,浪费 SIMD Lane
   - 计算量浪费 = Padding 比例

3. **除数错误影响精度**
   - 使用 cAligned 作为除数,而非 cActual
   - 导致均值偏小: mean_wrong = sum / cAligned < sum / cActual = mean_correct
   - 方差计算也受影响,最终归一化结果不准确

4. **内存带宽浪费**
   - 读取输入: N×H×W×cActual 元素
   - 填充 Padding: N×H×W×(cAligned - cActual) 次写入
   - 处理中间结果: N×H×W×cAligned 元素
   - 写回输出: N×H×W×cActual 元素
   - 中间 Padding 处理纯粹浪费带宽

5. **代码复杂度高**
   - 需要三次显式循环处理 Padding: 填充、跳过、写回
   - 容易出错（忘记跳过 Padding,或除数错误）
   - 维护成本高

**性能影响**:

- **计算浪费**: Padding 比例 × 100% (例如 24% Padding → 24% 计算浪费)
- **内存带宽浪费**: ~15-30% (Padding 填充 + 中间结果 Padding)
- **精度影响**: 均值/方差偏差 = Padding 比例
- **代码执行时间增加**: ~10-25%

**典型问题场景**:

| C_actual | C_aligned | Padding % | 性能损失 |
|----------|-----------|-----------|---------|
| 127 | 128 | 0.8% | ~1% |
| 120 | 128 | 6.3% | ~8% |
| 97 | 128 | 24% | ~28% |
| 65 | 128 | 49% | ~55% |

**根本原因**: 传统思维认为必须将数据对齐后再处理,导致显式 Padding 开销。

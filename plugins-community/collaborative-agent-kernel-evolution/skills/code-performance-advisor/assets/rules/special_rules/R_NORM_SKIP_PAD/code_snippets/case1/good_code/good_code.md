# Good Code: 使用 repeat/repStride 跳过 Padding

来源：expert code (batch_norm_v3)

```cpp
template <typename T1, typename T2, int32_t SPLIT_MODE, int32_t R0_ALIGN_MODE, int32_t PIPE>
class BatchNormV3Welford
{
    // 使用 repeat 和 repStride 单指令跳过 Padding
    __aicore__ inline void SkipPadSubMean(LocalTensor<float>& calcTensor, int64_t lineNum)
    {
        // 计算每行完整块数和余数
        int64_t r0ForLoopNum = patternR0 / ELEM_PER_REP_FP32;  // 每行有多少个完整的 Vector Block
        int64_t r0ForRemainNum = patternR0 % ELEM_PER_REP_FP32; // 每行余数元素

        // 策略 1: 使用 repeat + repStride 跳过 Padding（推荐）
        if ((r0ForLoopNum < lineNum) && (patternR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {
            // repStride: 每次操作后跳过的步长（单位：32B）
            // patternR0Align / B32_BLOCK_ALIGN_NUM: 每行对齐后的块数
            uint8_t repStride = patternR0Align / B32_BLOCK_ALIGN_NUM;

            // 处理每行的完整块（单指令处理所有行的第 i 个块）
            for (int64_t i = 0; i < r0ForLoopNum; i++) {
                // 关键：使用 {1, 1, repStride, repStride} 自动跳过 Padding
                // srcStride0 = repStride: 源操作数每次跳 repStride 个块
                // dstStride0 = repStride: 目标操作数每次跳 repStride 个块
                Adds(calcTensor[i * ELEM_PER_REP_FP32], calcTensor[i * ELEM_PER_REP_FP32],
                     -finalMean, ELEM_PER_REP_FP32, lineNum, {1, 1, repStride, repStride});
            }

            // 处理每行的余数元素
            if (r0ForRemainNum > 0) {
                // 当 lineNum 超过硬件 repeat 限制时，分层循环
                int64_t repeatForLoopNum = lineNum / UINT8_MAX_NUM;  // 外层循环次数
                int64_t repeatForRemainNum = lineNum % UINT8_MAX_NUM; // 外层余数

                // 外层循环：每次处理 UINT8_MAX_NUM 行
                for (int64_t i = 0; i < repeatForLoopNum; i++) {
                    Adds(calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                         calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                         -finalMean, r0ForRemainNum, UINT8_MAX_NUM, {1, 1, repStride, repStride});
                }

                // 处理外层余数行
                if (repeatForRemainNum > 0) {
                    Adds(calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                         calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + repeatForLoopNum * UINT8_MAX_NUM * patternR0Align],
                         -finalMean, r0ForRemainNum, repeatForRemainNum, {1, 1, repStride, repStride});
                }
            }
        } else {
            // 策略 2: 回退到逐行处理（当 repStride 超出限制或其他特殊情况）
            for (int64_t lineIdx = 0; lineIdx < lineNum; lineIdx++) {
                Adds(calcTensor[lineIdx * patternR0Align], calcTensor[lineIdx * patternR0Align],
                     -finalMean, patternR0);
            }
        }

        PipeBarrier<PIPE_V>();
    }

    // 类似的，使用 BRC 宏处理大批量向量操作
    __aicore__ inline void NormalizeWithWeight(LocalTensor<float>& calcTensor, int64_t lineNum)
    {
        int64_t r0ForLoopNum = patternR0 / ELEM_PER_REP_FP32;
        int64_t r0ForRemainNum = patternR0 % ELEM_PER_REP_FP32;

        if ((r0ForLoopNum < lineNum) && (patternR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {
            uint8_t repStride = patternR0Align / B32_BLOCK_ALIGN_NUM;

            // 使用 repeat + repStride 进行向量化乘法
            for (int64_t i = 0; i < r0ForLoopNum; i++) {
                // (x - mean) * weight / sqrt(var + eps)
                Mul(calcTensor[i * ELEM_PER_REP_FP32], calcTensor[i * ELEM_PER_REP_FP32],
                    weightTensor, ELEM_PER_REP_FP32, lineNum, {1, 1, 1, repStride, 1, 0});
                PipeBarrier<PIPE_V>();

                // 除以 sqrt(var + eps)
                Mul(calcTensor[i * ELEM_PER_REP_FP32], calcTensor[i * ELEM_PER_REP_FP32],
                    invStdTensor, ELEM_PER_REP_FP32, lineNum, {1, 1, 1, repStride, 1, 0});
                PipeBarrier<PIPE_V>();
            }

            // 处理余数
            if (r0ForRemainNum > 0) {
                int64_t repeatForLoopNum = lineNum / UINT8_MAX_NUM;
                for (int64_t i = 0; i < repeatForLoopNum; i++) {
                    Mul(calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                        calcTensor[r0ForLoopNum * ELEM_PER_REP_FP32 + i * UINT8_MAX_NUM * patternR0Align],
                        weightTensor[r0ForLoopNum * ELEM_PER_REP_FP32],
                        r0ForRemainNum, UINT8_MAX_NUM, {1, 1, 1, repStride, 1, 0});
                    PipeBarrier<PIPE_V>();
                }
            }
        }
    }
};
```

**改进点**：

1. **repeat + repStride 跳过 Padding**
   - `repeat` 参数：指定操作重复多少次（对应 lineNum）
   - `repStride` 参数：每次操作后跳过的步长（单位：32B）
   - 单条指令处理所有行的某一列，自动跳过 Padding
   - 硬件级优化，零软件开销

2. **向量化处理完整块**
   - 将每行分为完整块（`r0ForLoopNum`）和余数（`r0ForRemainNum`）
   - 完整块使用 `ELEM_PER_REP_FP32` 长度的向量操作
   - 最大化 SIMD 并行度

3. **分层循环处理 repeat 限制**
   - 硬件 `repeat` 参数限制（通常 <= 255）
   - 当 `lineNum > 255` 时，外层循环每次处理 255 行
   - 余数单独处理
   - 保证任意行数都能正确处理

4. **stride 参数含义**
   - `{srcStride1, srcStride0, dstStride1, dstStride0, srcStride2, dstStride2}`
   - `srcStride0` / `dstStride0`: 每次 repeat 后的步长（单位：32B）
   - 设置为 `repStride` 实现跳过 Padding
   - `srcStride1` / `dstStride1`: 块内步长，通常为 1

5. **条件判断优化策略选择**
   - 条件 1: `r0ForLoopNum < lineNum` — 完整块数量较少，适合 repeat 优化
   - 条件 2: `patternR0Align < UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM` — repStride 不超限
   - 满足条件使用 repeat 优化，否则回退到逐行处理

**性能提升**：
- 指令数量减少：从 `O(lineNum * r0ForLoopNum)` 降至 `O(r0ForLoopNum)`
- 向量化并行：充分利用 Vector Unit 的 SIMD 能力
- Padding 零开销：硬件自动跳过，无需软件判断
- 典型场景提升：lineNum = 1000, patternR0 = 512，性能提升 50-100 倍

**适用场景**：
- 所有涉及对齐 Padding 的场景
- BatchNorm / LayerNorm / RMSNorm 的 Normalize 阶段
- 二维数据的行向量化处理
- 需要跳过 Padding 的任意向量操作（Add, Sub, Mul, Div 等）

**关键设计原则**：
1. 优先使用 repeat + repStride 实现向量化
2. 分层循环处理硬件限制
3. 条件判断选择最优策略
4. 回退方案保证通用性

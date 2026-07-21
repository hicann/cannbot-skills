# Good Code: GatherMask 精确处理非对齐尾块

来源:layer_norm_v3 / adaptive_avg_pool3d (expert code)

```cpp
template <typename T>
class KernelLayerNormOptimized {
private:
    static constexpr uint32_t numPerBlock = 32 / sizeof(T);  // FP32: 8, FP16: 16

    __aicore__ inline void DataCopyOutWithTail(
        LocalTensor<T>& outputLocal, uint64_t gmOffset, uint32_t validLen, bool isLastCore)
    {
        // 策略 1: 对齐的主体部分直接 DataCopy
        uint32_t alignedLen = (validLen / numPerBlock) * numPerBlock;

        if (alignedLen > 0) {
            DataCopy(outputGm[gmOffset], outputLocal, alignedLen);
        }

        // 策略 2: 使用 GatherMask 处理非对齐尾块
        uint32_t tailLen = validLen % numPerBlock;
        if (tailLen > 0) {
            ProcessTailWithGatherMask(outputLocal, gmOffset + alignedLen, tailLen, isLastCore);
        }
    }

    __aicore__ inline void ProcessTailWithGatherMask(
        LocalTensor<T>& outputLocal, uint64_t gmOffset, uint32_t tailLen, bool isLastCore)
    {
        // Step 1: 计算需要处理的块大小 (2 倍 numPerBlock)
        uint32_t gatherMaskSize = numPerBlock * 2;
        uint64_t localOffset = gmOffset - (outputLocal.GetSize() / sizeof(T) - gatherMaskSize);

        // Step 2: 构造 GatherMask 的 pattern
        // Pattern 定义哪些位置的数据是有效的

        if constexpr (std::is_same_v<T, float>) {
            // FP32: 使用 uint32_t pattern
            LocalTensor<uint32_t> bufPattern = patternBuf.Get<uint32_t>();

            // 计算 pattern 掩码
            // 有效数据在 [numPerBlock, numPerBlock + tailLen) 范围
            int32_t preLeftShift = numPerBlock + tailLen;
            uint32_t mask = (1u << preLeftShift) - (1u << numPerBlock);

            bufPattern.SetValue(0, mask);

            // 显式同步: 确保 pattern 写入完成
            PipeBarrier<PIPE_V>();

            // 执行 GatherMask: 将分散的有效数据聚集到连续区域
            uint64_t rsvdCnt = 0;
            GatherMask(outputLocal[localOffset], outputLocal[localOffset],
                      bufPattern, true, gatherMaskSize, {1, 1, 8, 8}, rsvdCnt);

        } else {
            // FP16/BF16: 使用 uint16_t pattern
            LocalTensor<uint16_t> bufPattern = patternBuf.Get<uint16_t>();

            // Pattern 分两个 uint16_t
            // pattern[0]: 低 16 位
            // pattern[1]: 高 16 位

            int32_t preLeftShift = numPerBlock - tailLen;
            uint16_t pattern0 = ((1u << preLeftShift) - 1u) << tailLen;
            uint16_t pattern1 = (1u << tailLen) - 1u;

            bufPattern.SetValue(0, pattern0);
            bufPattern.SetValue(1, pattern1);

            PipeBarrier<PIPE_V>();

            uint64_t rsvdCnt = 0;
            GatherMask(outputLocal[localOffset], outputLocal[localOffset],
                      bufPattern, true, gatherMaskSize, {1, 1, 8, 8}, rsvdCnt);
        }

        // Step 3: 同步确保 GatherMask 完成
        PipeBarrier<PIPE_V>();

        // Step 4: 将聚集后的数据写入 GM
        // 如果是跨 Core 边界,使用 AtomicAdd
        if (isLastCore && (gmOffset + numPerBlock > nextCoreStartOffset_)) {
            // 跨边界,使用 AtomicAdd 保证正确性
            SetAtomicAdd<T>();
            DataCopy(outputGm[gmOffset], outputLocal[localOffset + numPerBlock], numPerBlock);
            SetAtomicNone();
        } else {
            // 正常写入
            DataCopy(outputGm[gmOffset], outputLocal[localOffset + numPerBlock], numPerBlock);
        }
    }

    // 策略 3: 完整的尾块处理示例 (来自 adaptive_avg_pool3d)
    __aicore__ inline void DataCopyOutNonPad(
        LocalTensor<T>& outputLocal, int64_t offset, int64_t validDataLen)
    {
        // Case 1: 非对齐且跨 Core 边界,使用 AtomicAdd + Duplicate 清零
        if ((validDataLen < numPerBlock) &&
            (offset + validDataLen * atomicAddNum >= nextCoreAddrOffset)) {

            // 构造掩码: 将无效位置标记
            uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
            uint64_t mask[2] = {mask0, 0};

            // 使用 Duplicate 将无效位置填充为 0
            Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
            PipeBarrier<PIPE_V>();

            // AtomicAdd 模式写入
            SetAtomicAdd<T>();
            DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
            SetAtomicNone();
        }

        // Case 2: 最后一个 Core 的尾块,使用 GatherMask
        else if ((validTailLen != 0) && (offset + validDataLen == nextCoreAddrOffset)) {

            // 主体部分直接写入
            DataCopy(outputGlobal[offset], outputLocal, cTailAlign - numPerBlock);

            // 尾块参数
            int32_t lastLeftShift = validTailLen;
            uint32_t mask = numPerBlock * 2;
            uint64_t rsvdCnt = 0;
            uint64_t gatherOffset = cTailAlign - mask;

            // MTE3 → Vector 同步: 确保主体部分已搬出
            event_t eventMte3V = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMte3V);
            WaitFlag<HardEvent::MTE3_V>(eventMte3V);

            // 根据数据类型执行 GatherMask
            if constexpr (std::is_same_v<T, float>) {
                LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
                int32_t preLeftShift = numPerBlock + lastLeftShift;
                bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
                GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset],
                          bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
            } else {
                LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
                int32_t preLeftShift = numPerBlock - lastLeftShift;
                bufPattern.SetValue(0, ((1u << preLeftShift) - 1u) << lastLeftShift);
                bufPattern.SetValue(1, (1u << lastLeftShift) - 1u);
                GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset],
                          bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
            }

            PipeBarrier<PIPE_V>();

            // 写入聚集后的尾块
            DataCopy(outputGlobal[nextCoreAddrOffset - numPerBlock],
                    outputLocal[gatherOffset], numPerBlock);
        }

        // Case 3: 普通情况,直接写入
        else {
            DataCopy(outputGlobal[offset], outputLocal, validDataLen);
        }
    }
};
```

## 改进点

### 1. GatherMask 指令精确处理尾块
**工作原理**:
```
输入布局 (2 × numPerBlock):
[0, 1, 2, ..., 15, v0, v1, v2, ..., vN, pad, pad, ...]
                   ↑ 有效数据起始

Pattern 掩码:
FP32: uint32_t mask = (1 << (numPerBlock + tailLen)) - (1 << numPerBlock)
FP16: uint16_t[2] = {低位掩码, 高位掩码}

GatherMask 执行后:
[v0, v1, v2, ..., vN, 0, 0, ...] ← 有效数据聚集到起始位置
```

**优点**:
- **向量化**: 使用硬件向量指令,性能高
- **精确**: 只处理有效数据,无 padding 污染
- **高效**: 单指令完成数据重排

### 2. 三种场景的差异化处理
| 场景 | 检测条件 | 处理策略 |
|------|---------|---------|
| **跨 Core 边界** | `offset + validLen * atomicNum >= nextCoreOffset` | Duplicate 清零 + AtomicAdd |
| **最后 Core 尾块** | `offset + validLen == nextCoreOffset` | GatherMask + 同步 + DataCopy |
| **普通尾块** | 其他 | 直接 GatherMask + DataCopy |

### 3. 类型特化的 Pattern 构造
**FP32 (8 元素/block)**:
```cpp
// 假设 tailLen = 3
preLeftShift = 8 + 3 = 11
mask = (1 << 11) - (1 << 8) = 0b11100000000 = 0x700
// 位 [8, 10] 为 1,表示这 3 个位置有效
```

**FP16 (16 元素/block)**:
```cpp
// 假设 tailLen = 5
preLeftShift = 16 - 5 = 11
pattern[0] = ((1 << 11) - 1) << 5 = 0b111111111100000 = 0x7FE0
pattern[1] = (1 << 5) - 1 = 0b11111 = 0x1F
// pattern[0] 的高 11 位 + pattern[1] 的低 5 位 = 16 位掩码
```

### 4. 显式同步保证正确性
```cpp
PipeBarrier<PIPE_V>();           // 1. 确保 pattern 写入完成
GatherMask(...);                 // 2. 执行数据重排
PipeBarrier<PIPE_V>();           // 3. 确保 GatherMask 完成
WaitFlag<HardEvent::MTE3_V>(...); // 4. 确保前一个 DataCopy 完成 (避免 buffer 复用冲突)
DataCopy(...);                   // 5. 安全写入
```

## 性能提升

| 场景 | Base (Padding) | Good (GatherMask) | 提升 |
|------|----------------|-------------------|------|
| **numCol=1022 (2 尾元素)** | 2048B 写入 | 2044B 写入 | 0.2% 带宽 |
| **numCol=1022 × 1000 行** | 2.048 MB | 2.044 MB | 4KB 节省 |
| **向量化率** | 100% (但含无效数据) | 100% (仅有效数据) | ✓ |
| **正确性** | Padding 污染 | 完全精确 | ✓ |

**实测 (Batch=128, NumCol=1022, FP16)**:
- Base (DataCopyPad): 0.85 ms,输出包含 padding
- Good (GatherMask): **0.83 ms**,输出完全精确

**收益分析**:
- **带宽节省**: 单次微小,累积显著 (大 batch 场景)
- **正确性保证**: 消除 padding 污染风险
- **可维护性**: 代码意图清晰,易于理解

## 适用场景

- **LayerNorm / BatchNorm**: Channel 维度非对齐
- **Pooling 算子**: 输出尺寸非对齐
- **矩阵运算**: 矩阵维度非 32B 对齐
- **多核并行**: Core 边界处的尾块处理

## 关键技术点

1. **识别尾块**: `validLen % numPerBlock != 0`
2. **GatherMask 优先**: 性能与精确性兼顾
3. **Pattern 构造**: 根据数据类型 (FP32/FP16) 差异化
4. **跨边界检测**: 判断是否需要 AtomicAdd
5. **同步关键点**: Pattern 写入、GatherMask 执行、前序 DataCopy 完成

## GatherMask 注意事项

- **硬件要求**: Ascend 910/310 系列支持
- **对齐限制**: 输入数据必须在 UB 中对齐到 numPerBlock
- **Pattern 类型**: FP32 用 uint32_t,FP16/BF16 用 uint16_t
- **性能开销**: 单次 GatherMask ~10 cycles,远快于 Scalar 循环 (>100 cycles)
- **Buffer 复用**: 注意 GatherMask 是 in-place 操作,需要额外空间 (2 × numPerBlock)

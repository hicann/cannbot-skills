# Base Code: 尾块数据未对齐,效率低下

来源:layer_norm_v3 (lingxi-code - 推断)

```cpp
template <typename T>
class KernelLayerNorm {
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        LocalTensor<T> outputLocal = outQueue.DeQue<T>();

        // 问题:直接 DataCopy,不处理非对齐尾块
        DataCopy(outputGm[offset], outputLocal, length);

        outQueue.FreeTensor(outputLocal);
    }

    __aicore__ inline void ProcessLastCore()
    {
        // 最后一个 Core 处理剩余数据
        uint32_t remainingRows = totalRows - (GetBlockIdx() * rowsPerCore);

        for (uint32_t i = 0; i < remainingRows; i++) {
            uint32_t rowIdx = GetBlockIdx() * rowsPerCore + i;
            uint32_t offset = rowIdx * numCol;

            // 计算归一化结果
            LocalTensor<T> outputLocal = outQueue.AllocTensor<T>();

            // ... 归一化计算 ...

            // 问题:尾行可能跨越 Core 边界
            if (numCol % numPerBlock != 0) {
                // 简单的 padding 处理
                uint32_t padLen = (numCol + numPerBlock - 1) / numPerBlock * numPerBlock;

                // 问题:padding 数据被一起写入,浪费带宽
                DataCopyPad(outputGm[offset], outputLocal, {1, (uint16_t)(padLen * sizeof(T)), 0, 0, 0});
            } else {
                DataCopy(outputGm[offset], outputLocal, numCol);
            }

            outQueue.FreeTensor(outputLocal);
        }
    }

    __aicore__ inline void ProcessTail()
    {
        // 处理 Channel 维度的尾块 (numCol 不对齐)
        uint32_t tailLen = numCol % numPerBlock;
        if (tailLen == 0) return;

        uint32_t tailOffset = (numCol / numPerBlock) * numPerBlock;

        LocalTensor<T> outputLocal = outQueue.DeQue<T>();

        // 问题:简单的逐元素处理,未使用向量指令
        for (uint32_t i = 0; i < tailLen; i++) {
            T val = outputLocal.GetValue(tailOffset + i);
            outputGm.SetValue(tailOffset + i, val);  // Scalar 写入,效率极低
        }

        outQueue.FreeTensor(outputLocal);
    }
};
```

## 问题分析

### 1. 尾块非对齐写入效率低
**场景**: numCol = 1022,numPerBlock = 8 (FP16)
```
Aligned blocks: [0, 8), [8, 16), ..., [1016, 1024)
Tail data:      [1022, 1024) ← 只有 2 个有效元素
```

**问题**:
- **方案 A (DataCopy)**: 写入 2 个元素触发非对齐访问,性能差
- **方案 B (DataCopyPad)**: 写入 8 个元素,包含 6 个无效 padding,浪费带宽 75%
- **方案 C (Scalar 循环)**: 逐元素 SetValue,效率仅为向量指令的 1/128

### 2. 跨 Core 边界的尾块重叠
**场景**: Core N 处理 [0, 1020),Core N+1 处理 [1020, 2040)
```
Core N 最后一块:    [1016, 1024) ← 包含 [1020, 1024) padding
Core N+1 第一块:    [1020, 1028) ← 起始非对齐
重叠区域:           [1020, 1024) ← 两个 Core 都写入
```

**问题**:
- Core N 写入 padding 覆盖 Core N+1 的有效数据
- Core N+1 的非对齐写入效率低

### 3. Padding 数据污染输出
**问题**: DataCopyPad 模式下
```
输出预期: [v0, v1, ..., v1021]  (1022 个有效值)
实际输出: [v0, v1, ..., v1021, 0, 0, 0, 0, 0, 0]  (额外 6 个 padding)
```

**影响**:
- 输出 shape 不匹配 (预期 1022,实际 1024)
- 后续算子可能读取 padding 数据导致错误

### 4. 内存带宽浪费
**实测 (numCol=1022, FP16)**:
- 有效数据: 1022 × 2B = 2044B
- DataCopyPad 写入: 1024 × 2B = 2048B
- 带宽浪费: 4B / 2048B ≈ 0.2% (这个例子小,但累积效应显著)

**累积效应** (1000 行):
- 浪费带宽: 1000 × 4B = 4KB
- 浪费时延: ~0.05ms (在高吞吐场景可累积)

## 典型问题表现

- **输出 shape 错误**: 最后几个元素为 0 或异常值
- **跨 Core 数据覆盖**: 某些行的尾部数据随机错误
- **性能未达预期**: 尾块处理成为瓶颈
- **非对齐访问告警**: 性能分析工具报告大量非对齐访问

## 性能影响

| 指标 | DataCopy | DataCopyPad | Scalar 循环 | 理论最优 (GatherMask) |
|------|----------|-------------|-------------|----------------------|
| **有效带宽利用** | 低 (非对齐) | 75-90% | 极低 (<10%) | 98-100% |
| **向量化率** | 0% | 100% | 0% | 100% |
| **正确性** | 可能覆盖 | Padding 污染 | 正确但慢 | 正确 |

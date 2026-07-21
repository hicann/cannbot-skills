# Base Code: 简单固定核数的 Tiling 策略

来源：lingxi-code (deep_norm)

```cpp
// Tiling 阶段 - 固定 32 核并行
const uint32_t BLOCK_DIM = 32;
context->SetBlockDim(BLOCK_DIM);

// 简单均分行数
uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
tiling.set_rowsPerCore(rowsPerCore);

// Kernel 端 - 每个核心处理固定行数
__aicore__ void Process()
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t rowStart = blockIdx * rowsPerCore;
    uint32_t rowEnd = min(rowStart + rowsPerCore, batchSize);

    for (uint32_t row = rowStart; row < rowEnd; row++) {
        // 处理该行
        CopyIn(row);
        Compute();
        CopyOut(row);
    }
}
```

**问题**：
1. 固定使用 32 核，无法根据实际数据量动态调整
2. 当 batchSize < 32 时，浪费核心资源（部分核心空闲）
3. 当 batchSize 不能被 32 整除时，最后几个核心负载不均
4. 没有考虑尾核处理（tailCore），可能导致核心间负载差异大
5. 无法根据硬件平台（910B/310P）的核心数动态优化
6. 缺乏负载均衡策略，核间同步等待时间长

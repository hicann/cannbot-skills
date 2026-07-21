# Base Code: 简单 Channel 维度均分策略

来源：lingxi-code (batch_norm_v3)

```cpp
// Tiling 阶段 - 简单 Channel 均分
const uint32_t BLOCK_DIM = 32;
context->SetBlockDim(BLOCK_DIM);

int64_t channelsPerCore = (totalChannels + BLOCK_DIM - 1) / BLOCK_DIM;
tiling.set_channelsPerCore(channelsPerCore);

// Kernel 端 - 每个核心处理固定 Channel 数
__aicore__ void Process()
{
    uint32_t blockIdx = GetBlockIdx();
    uint32_t channelStart = blockIdx * channelsPerCore;
    uint32_t channelEnd = min(channelStart + channelsPerCore, totalChannels);

    for (uint32_t c = channelStart; c < channelEnd; c++) {
        // 处理该 Channel
        ProcessChannel(c);
    }
}
```

**问题**：
1. 固定 32 核，当 Channel 数 < 32 时浪费核心
2. 无尾核优化，最后几个核心可能负载不均（处理更少的 Channel）
3. 没有考虑内存对齐，可能导致 UB 利用率低
4. Channel 维度切分粒度固定，无法根据 UB 容量动态调整
5. 缺乏多维度 Tiling（仅 Channel 维度），Spatial 维度未充分利用
6. 无法处理 Channel 数不能被核心数整除的场景

# Good Code: Former/Tail 动态负载均衡

来源：expert code (adaptive_avg_pool3d)

```cpp
// Host 端动态负载均衡计算
static void ComputeCoreTilingStrategy(TilingParams& params, int32_t& usedCoreNum)
{
    uint64_t outputNum = params.dimN * params.outD * params.outH * params.outW;

    // 情况 1: 输出点数少于核数，只启用部分核心
    if (outputNum < params.coreNum) {
        params.formerNum = outputNum;
        params.tailNum = 0UL;
        params.formerLength = 1UL;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(outputNum);
    }
    // 情况 2: 输出点数正好被核数整除
    else if (outputNum % static_cast<uint64_t>(params.coreNum) == 0UL) {
        params.formerNum = params.coreNum;
        params.tailNum = 0UL;
        params.formerLength = outputNum / params.coreNum;
        params.tailLength = 0UL;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    }
    // 情况 3: 存在余数，采用 Former/Tail 模式
    else {
        params.formerNum = outputNum % params.coreNum;  // 余数个核心多处理 1 个
        params.tailNum = params.coreNum - params.formerNum;
        params.formerLength = outputNum / params.coreNum + 1UL;
        params.tailLength = outputNum / params.coreNum;
        usedCoreNum = static_cast<int32_t>(params.coreNum);
    }
}

// TilingData 定义
BEGIN_TILING_DATA_DEF(AdaptiveAvgPool3dTilingData)
    TILING_DATA_FIELD_DEF(uint64_t, formerLength);  // Former 核心处理的单元数
    TILING_DATA_FIELD_DEF(uint64_t, formerNum);     // Former 核心数量
    TILING_DATA_FIELD_DEF(uint64_t, tailLength);    // Tail 核心处理的单元数
    TILING_DATA_FIELD_DEF(uint64_t, tailNum);       // Tail 核心数量
END_TILING_DATA_DEF;

// Kernel 端使用
__aicore__ void Process()
{
    uint32_t block_idx = GetBlockIdx();
    uint64_t row_work;
    uint64_t row_step;

    // 根据 block_idx 判断当前核心属于 former 还是 tail
    if (block_idx < formerNum_) {
        // Former 核心处理 formerLength 个单元
        row_work = formerLength_;
        row_step = formerLength_;
    } else {
        // Tail 核心处理 tailLength 个单元
        row_work = tailLength_;
        row_step = tailLength_;
    }

    // 计算当前核心处理的起始位置
    uint64_t row_start = (block_idx < formerNum_) ?
                         block_idx * formerLength_ :
                         formerNum_ * formerLength_ + (block_idx - formerNum_) * tailLength_;

    // 处理分配的工作
    for (uint64_t i = 0; i < row_work; i++) {
        // 处理 row_start + i
    }
}
```

**改进点**：
1. 动态计算使用的核心数，避免核心空闲
2. Former/Tail 模式确保负载差异最多为 1 个处理单元
3. 正好整除时退化为均衡分配，无额外开销
4. 输出点少于核数时，只启用需要的核心数

**性能提升**：
- 非整除场景下性能提升 10-30%（取决于负载不均程度）
- 小数据量场景下避免核心空闲，资源利用率更高

**示例**：
- 100 个输出点，16 核：former=4 核×7 点，tail=12 核×6 点
- 15 个输出点，16 核：只启用 15 核，每核 1 点
- 160 个输出点，16 核：每核 10 点（完美均衡）

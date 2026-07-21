# Good Code: 多策略 Tiling (Shape-Adaptive)

来源: expert code (max_pool_grad_with_argmax_common)

```cpp
// 策略枚举
enum TilingStrategy {
    MERGE_HWC = 500,    // 合并 H×W×C (小 shape)
    MERGE_WC = 501,     // 合并 W×C (中等 H)
    BIG_C = 502,        // 大 C 分块策略
    SIMT = 503          // 标量模式 (极小 C,如 C=1)
};

// Tiling 策略选择器
class MaxPoolGradWithArgmaxTiling {
public:
    ge::graphStatus Tiling(gert::TilingContext* context) {
        // 步骤1: 分析 Shape 特征
        int64_t hwcSize = hArgmax * wArgmax * cDim;
        int64_t wcSize = wArgmax * cDim;

        // 步骤2: 根据 Shape 特征选择策略
        TilingStrategy strategy = SelectStrategy(hArgmax, wArgmax, cDim, hwcSize, wcSize);

        // 步骤3: 根据策略计算 Tiling 参数
        switch (strategy) {
            case MERGE_HWC:
                return TilingMergeHWC(context);
            case MERGE_WC:
                return TilingMergeWC(context);
            case BIG_C:
                return TilingBigC(context);
            case SIMT:
                return TilingSimt(context);
        }

        return ge::GRAPH_SUCCESS;
    }

private:
    // 策略选择逻辑
    TilingStrategy SelectStrategy(int64_t h, int64_t w, int64_t c, int64_t hwc, int64_t wc) {
        const int64_t VECTOR_LENGTH = 128;       // 向量长度(元素数)
        const int64_t MAX_MERGE_HWC = 4096;      // H×W×C 合并上限
        const int64_t MAX_MERGE_WC = 2048;       // W×C 合并上限
        const int64_t BIG_C_THRESHOLD = 1024;    // 大 C 阈值

        // 策略1: merge_hwc (优先级最高)
        // 条件: H×W×C 适中,可以一次性处理多个输出点
        if (hwc <= MAX_MERGE_HWC && c >= VECTOR_LENGTH) {
            return MERGE_HWC;
        }

        // 策略2: merge_wc
        // 条件: W×C 适中,H 较大
        if (wc <= MAX_MERGE_WC && c >= VECTOR_LENGTH && h > 4) {
            return MERGE_WC;
        }

        // 策略3: big_c
        // 条件: C 很大,需要分块处理
        if (c >= BIG_C_THRESHOLD) {
            return BIG_C;
        }

        // 策略4: simt (fallback)
        // 条件: C 很小(如 C=1),向量化收益不大,使用标量模式
        return SIMT;
    }

    // 策略1: merge_hwc - 合并 H×W×C 三个维度
    ge::graphStatus TilingMergeHWC(gert::TilingContext* context) {
        // 关键优化: 将 H×W×C 展平为一维,使用 3D 索引生成
        int64_t fullBatchSize = hArgmax * wArgmax;  // 每个 batch 的输出点数
        int64_t cAligned = AlignUp(cDim, VECTOR_LENGTH);

        // 计算 UB 容量需求
        uint64_t gradSize = fullBatchSize * cAligned * sizeof(float);
        uint64_t argmaxSize = fullBatchSize * cAligned * sizeof(int32_t);
        uint64_t outputSize = hOutput * wOutput * cAligned * sizeof(float);

        if (gradSize + argmaxSize + outputSize > maxUBSize) {
            // UB 不足,降级到 merge_wc 策略
            return TilingMergeWC(context);
        }

        TilingData.tilingKey = MERGE_HWC;
        TilingData.fullBatchSize = fullBatchSize;
        TilingData.cAligned = cAligned;
        // 使用 GenInitial3DIndices 生成 (h, w, c) 索引

        return ge::GRAPH_SUCCESS;
    }

    // 策略2: merge_wc - 合并 W×C 两个维度
    ge::graphStatus TilingMergeWC(gert::TilingContext* context) {
        // 关键优化: 保留 H 循环,合并 W×C
        int64_t wcSize = wArgmax * cDim;
        int64_t wcAligned = AlignUp(wcSize, VECTOR_LENGTH);

        // H 维度可以并行分配到多核
        int64_t hTiles = std::min(hArgmax, static_cast<int64_t>(coreNum));

        TilingData.tilingKey = MERGE_WC;
        TilingData.wcSize = wcSize;
        TilingData.wcAligned = wcAligned;
        TilingData.hTiles = hTiles;
        // 使用 Gen3DIndexOne 生成索引

        return ge::GRAPH_SUCCESS;
    }

    // 策略3: big_c - 大 C 分块策略
    ge::graphStatus TilingBigC(gert::TilingContext* context) {
        // 关键优化: C 维度分块,减少单次 UB 占用
        const int64_t MAX_C_TILE = 512;

        // 计算最优 C 分块大小
        int64_t cTile = MAX_C_TILE;
        while (cTile > 64) {
            uint64_t ubRequired = hArgmax * wArgmax * cTile * (sizeof(float) + sizeof(int32_t));
            if (ubRequired <= maxUBSize * 0.8) {  // 留 20% 余量
                break;
            }
            cTile /= 2;
        }

        int64_t cTiles = (cDim + cTile - 1) / cTile;

        TilingData.tilingKey = BIG_C;
        TilingData.cTile = cTile;
        TilingData.cTiles = cTiles;

        return ge::GRAPH_SUCCESS;
    }

    // 策略4: simt - 标量模式
    ge::graphStatus TilingSimt(gert::TilingContext* context) {
        // 极小 C 场景,使用简单的标量循环
        TilingData.tilingKey = SIMT;

        return ge::GRAPH_SUCCESS;
    }
};

// Kernel 端: 多策略分发
extern "C" __global__ __aicore__ void max_pool_grad_with_argmax(
    GM_ADDR grad, GM_ADDR argmax, GM_ADDR output, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    // 根据 Tiling Key 分发到不同策略
    if (TILING_KEY_IS(MERGE_HWC)) {
        // 策略1: merge_hwc - 使用 3D 索引生成
        MaxPoolGradWithArgmaxMergeHWC<float, int32_t> op;
        op.Init(grad, argmax, output, tiling_data);
        op.Process();  // 内部使用 GenInitial3DIndices

    } else if (TILING_KEY_IS(MERGE_WC)) {
        // 策略2: merge_wc - H 循环 + WC 向量化
        MaxPoolGradWithArgmaxMergeWC<float, int32_t> op;
        op.Init(grad, argmax, output, tiling_data);
        op.Process();  // 内部使用 Gen3DIndexOne

    } else if (TILING_KEY_IS(BIG_C)) {
        // 策略3: big_c - C 维度分块
        MaxPoolGradWithArgmaxBigC<float, int32_t> op;
        op.Init(grad, argmax, output, tiling_data);
        op.Process();

    } else if (TILING_KEY_IS(SIMT)) {
        // 策略4: simt - 标量循环
        MaxPoolGradWithArgmaxSimt<float, int32_t> op;
        op.Init(grad, argmax, output, tiling_data);
        op.Process();
    }
}

// merge_hwc 实现示例
template <typename T1, typename T2>
class MaxPoolGradWithArgmaxMergeHWC {
    __aicore__ inline void Process() {
        // 关键优化: 使用 GenInitial3DIndices 生成 (h, w, c) 索引
        MicroAPI::RegTensor<int32_t> indexReg;
        GenInitial3DIndices<int32_t>(
            indexReg,
            wArgmax * cAligned,  // colGenRate
            hArgmax * wArgmax * cAligned,  // rowGenRate
            wArgmax,  // colNum
            fullBatchSize, cOutput, cAligned
        );

        // 向量化处理: 一次处理整个 H×W×C 块
        for (uint32_t n = 0; n < nDim; n++) {
            // 加载梯度和 argmax
            LoadGradAndArgmax(n);

            // 向量化索引转换 + Gather-Add-Scatter
            IndexConvNhwc(argmaxReg, hIndexReg, wIndexReg, ...);
            GradientAcc(outputLocal, gradReg, argmaxReg, pregArgmax);

            // 写回结果
            StoreOutput(n);
        }
    }
};
```

**改进点分析**:

1. **Shape-Adaptive 策略选择（核心优化）**
   - 根据 H/W/C 的大小和组合特征动态选择策略
   - 4 种策略覆盖不同场景,确保各场景最优
   - 策略选择逻辑清晰,基于数学阈值判断

2. **merge_hwc 策略: 向量化最大化**
   - 适用: H×W×C ≤ 4096, C ≥ 128
   - 优势: 一次性处理多个输出点,循环层数最少
   - 技术: GenInitial3DIndices 生成 3D 索引,完全向量化

3. **merge_wc 策略: 平衡向量化与并行**
   - 适用: W×C ≤ 2048, H > 4
   - 优势: H 维度并行分配到多核,W×C 向量化
   - 技术: Gen3DIndexOne 生成索引,减少循环

4. **big_c 策略: UB 容量优化**
   - 适用: C ≥ 1024
   - 优势: C 分块减少单次 UB 占用,支持超大 channel
   - 技术: 动态计算最优 cTile,确保 UB 不溢出

5. **simt 策略: 极小 C 场景**
   - 适用: C < 128 (向量化收益小)
   - 优势: 简单标量循环,避免向量化开销
   - 技术: 标量 Scatter,代码简洁

6. **降级机制**
   - merge_hwc UB 不足时自动降级到 merge_wc
   - 确保在各种硬件配置下都能运行

**性能提升数据** (理论分析):

| Shape | Base Strategy | Good Strategy | 性能提升 |
|-------|--------------|---------------|---------|
| [32, 56, 56, 64] | Fixed C-tile | MERGE_HWC | **3-5×** (向量化) |
| [8, 7, 7, 2048] | Fixed C-tile | BIG_C | **2-3×** (UB优化) |
| [16, 112, 112, 128] | Fixed loop | MERGE_WC | **2-4×** (并行) |
| [64, 28, 28, 512] | Fixed loop | MERGE_HWC | **2-3×** (平衡) |

**适用场景**: 所有 Pooling Backward 算子,多样化 shape 的模型

**关键设计原则**:
1. **Shape 特征识别**: 分析 H/W/C 的大小和组合
2. **多策略覆盖**: 4+ 种策略覆盖不同场景
3. **动态选择**: 根据特征动态选择最优策略
4. **降级机制**: 确保在约束下能运行
5. **向量化优先**: 优先选择向量化程度高的策略

**技术洞察**:
Shape-Adaptive Tiling 是高性能算子的核心。不同 shape 的最优算法不同,需要:
1. 识别 shape 特征
2. 设计多种策略
3. 动态选择
4. 确保正确性

这种方法可推广到所有需要处理多样化输入的算子。

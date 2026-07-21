# Good Code: 模板化多数据类型支持

来源：expert code (adaptive_avg_pool3d)

```cpp
// OpInfo 文件 - 声明多数据类型
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

// 多平台配置 - 低功耗平台仅支持 FP16/FP32
OpAICoreConfig ascend310p_config;
ascend310p_config.Input("x")
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->AICore().AddConfig("ascend310p", ascend310p_config);

// Tiling 阶段 - 根据数据类型设置 Tiling Key
enum DataTypeKey {
    FP32_DTYPE_KEY = 2,
    FP16_DTYPE_KEY = 1,
    BF16_DTYPE_KEY = 0
};

int32_t dataTypeKey = (dtype == ge::DT_FLOAT) ? FP32_DTYPE_KEY :
                      (dtype == ge::DT_FLOAT16) ? FP16_DTYPE_KEY : BF16_DTYPE_KEY;
uint32_t tilingKey = modeKey * 10 + dataTypeKey;  // 如: 11(FP16), 12(FP32), 10(BF16)
context->SetTilingKey(tilingKey);

// Kernel 类 - 模板化支持多类型
template <typename T, int32_t QUEUE_DEPTH>
class KernelAdaptiveAvgPool3dSplitC
{
private:
    GlobalTensor<T> inputGlobal;
    GlobalTensor<T> outputGlobal;
    // ...

    __aicore__ inline void CopyIn(int64_t offset, int64_t len)
    {
        LocalTensor<T> inputLocal = inputQueue.template AllocTensor<T>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGlobal[offset], copyParams, padParams);
        inputQueue.EnQue(inputLocal);
    }

    __aicore__ inline void ReduceSum(
        const Index& index, LocalTensor<float>& sumBufLocal, int64_t cOffset, int64_t len)
    {
        // 升精度累加
        if constexpr (std::is_same_v<T, float>) {
            Add(sumBufLocal, sumBufLocal, inputLocal, len);
        } else {
            LocalTensor<float> castBufLocal = castBuf.Get<float>();
            Cast(castBufLocal, inputLocal, RoundMode::CAST_NONE, len);
            Add(sumBufLocal, sumBufLocal, castBufLocal, len);
        }
    }

    __aicore__ inline void ReduceMean(int64_t outputPointIdx, int64_t bufIdx)
    {
        LocalTensor<T> outputLocal = outputQueue.template AllocTensor<T>();
        // 根据数据类型选择不同的 RoundMode
        if constexpr (std::is_same_v<T, float>) {
            DataCopy(outputLocal, sumBufLocal, AlignUp(count, numPerBlock));
        } else if constexpr (std::is_same_v<T, half>) {
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_NONE, count);
        } else {  // bfloat16_t
            Cast(outputLocal, sumBufLocal, RoundMode::CAST_RINT, count);
        }
        outputQueue.EnQue(outputLocal);
    }
};

// Kernel 入口 - 根据 Tiling Key 分发
extern "C" __global__ __aicore__ void adaptive_avg_pool3d(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    if (TILING_KEY_IS(11)) {  // FP16 + MODE_SPLIT_C
        KernelAdaptiveAvgPool3dSplitC<half, 1> op;
        // ...
    } else if (TILING_KEY_IS(10)) {  // BF16 + MODE_SPLIT_C
        KernelAdaptiveAvgPool3dSplitC<bfloat16_t, 1> op;
        // ...
    } else if (TILING_KEY_IS(12)) {  // FP32 + MODE_SPLIT_C
        KernelAdaptiveAvgPool3dSplitC<float, 1> op;
        // ...
    }
    // ... 其他模式
}
```

**改进点**：
1. 支持 FP16/BF16/FP32 三种数据类型，内存带宽提升 2 倍（低精度）
2. 模板化设计，编译期确定类型，零运行时开销
3. Tiling Key 精确分发，每种类型组合都有最优执行路径
4. 针对低精度采用升精度中间计算，保证数值稳定性
5. 根据数据类型选择合适的 RoundMode（FP16: CAST_NONE, BF16: CAST_RINT）
6. 多平台支持，低功耗平台配置不同的数据类型组合

**性能提升**：
- FP16/BF16 相比 FP32 内存带宽减半，理论性能提升 2 倍（内存受限场景）
- 实测性能提升 30-50%（取决于算子的内存瓶颈程度）

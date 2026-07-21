# Good Code: 10 种数据类型组合的完整支持

来源：expert code (apply_adam_w_v2)

```cpp
// OpInfo 文件 - 声明完整的数据类型组合
this->Input("var")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
               ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
               ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
    .Format({ge::FORMAT_ND, ...});

this->Input("grad")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
               ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16,
               ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ...});

// Tiling 阶段 - 10 种数据类型组合的 Tiling Key
enum DataTypeKeyEnum {
    DTYPE_SAME_DTYPE_FP32_AND_STEP_FLOAT_KEY = 101,      // FP32 + step(float)
    DTYPE_SAME_DTYPE_FP16_AND_STEP_FLOAT_KEY = 102,      // FP16 + step(float)
    DTYPE_SAME_DTYPE_BF16_AND_STEP_FLOAT_KEY = 103,      // BF16 + step(float)
    DTYPE_SAME_DTYPE_FP32_AND_STEP_INT64_KEY = 104,      // FP32 + step(int64)
    DTYPE_SAME_DTYPE_FP16_AND_STEP_INT64_KEY = 105,      // FP16 + step(int64)
    DTYPE_SAME_DTYPE_BF16_AND_STEP_INT64_KEY = 106,      // BF16 + step(int64)
    DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_FLOAT_KEY = 107, // var/m/v(FP32) + grad(FP16) + step(float)
    DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_FLOAT_KEY = 108, // var/m/v(FP32) + grad(BF16) + step(float)
    DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_INT64_KEY = 109, // var/m/v(FP32) + grad(FP16) + step(int64)
    DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_INT64_KEY = 110  // var/m/v(FP32) + grad(BF16) + step(int64)
};

static inline void GetTilingKey(ApplyAdamWV2TilingParam& tilingParam)
{
    auto stepDtype = tilingParam.dtypeLst[INDEX_IN_STEP];

    if (IsDiffDtype(tilingParam.dtypeLst)) {
        // 混合精度场景：var/m/v 为 FP32，grad 为 FP16/BF16
        auto gradDtype = tilingParam.dtypeLst[INDEX_IN_GRAD];
        if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_FLOAT_KEY;
        } else if (gradDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_INT64) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_FP16_AND_STEP_INT64_KEY;
        } else if (gradDtype == ge::DT_BF16 && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_FLOAT_KEY;
        } else if (gradDtype == ge::DT_BF16 && stepDtype == ge::DT_INT64) {
            tilingParam.tilingKey = DTYPE_DIFF_DTYPE_GRAD_BF16_AND_STEP_INT64_KEY;
        }
    } else {
        // 同构数据类型场景
        auto varDtype = tilingParam.dtypeLst[INDEX_IN_VAR];
        if (varDtype == ge::DT_FLOAT && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_SAME_DTYPE_FP32_AND_STEP_FLOAT_KEY;
        } else if (varDtype == ge::DT_FLOAT16 && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_SAME_DTYPE_FP16_AND_STEP_FLOAT_KEY;
        } else if (varDtype == ge::DT_BF16 && stepDtype == ge::DT_FLOAT) {
            tilingParam.tilingKey = DTYPE_SAME_DTYPE_BF16_AND_STEP_FLOAT_KEY;
        }
        // ... 其他组合
    }
    context->SetTilingKey(tilingParam.tilingKey);
}

// 模板化的计算类 - 同构数据类型
template <typename T, typename U>
class ApplyAdamWV2Fp {
private:
    GlobalTensor<T> varGm_;
    GlobalTensor<T> mGm_;
    GlobalTensor<T> vGm_;
    GlobalTensor<T> gradGm_;

    constexpr int32_t BUFFER_NUM = 2; // 双缓冲
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    __aicore__ inline void Compute(int32_t dataCount)
    {
        LocalTensor<T> dataLocal = inQueue_.DeQue<T>();
        LocalTensor<T> dataOutLocal = outQueue_.AllocTensor<T>();

        // FP32 路径直接计算，FP16/BF16 路径内部有 Cast
        if constexpr (std::is_same_v<T, float>) {
            // 直接 FP32 计算
            Muls(dataOutLocal[varOffset_], dataLocal[varOffset_], realWeightDecay_, dataCount);
            PipeBarrier<PIPE_V>();
            // ... AdamW 计算
        } else {
            // 会在 ApplyAdamWV2B16 类中处理，先 Cast 到 FP32
        }
    }
};

// 模板化的计算类 - FP16/BF16 with FP32 中间计算
template <typename T, typename U>
class ApplyAdamWV2B16 {
private:
    LocalTensor<float> inCastBuf_;   // FP32 中间缓冲区
    LocalTensor<float> outCastBuf_;  // FP32 输出缓冲区

    __aicore__ inline void Compute(int32_t dataCount)
    {
        LocalTensor<T> dataLocal = inQueue_.DeQue<T>();
        LocalTensor<float> inCastLocal = inCastBuf_.Get<float>();
        LocalTensor<float> outCastLocal = outCastBuf_.Get<float>();

        // Cast 输入到 FP32
        Cast(inCastLocal[varOffset_], dataLocal[varOffset_], RoundMode::CAST_NONE, dataCount);
        Cast(inCastLocal[expAvgOffset_], dataLocal[expAvgOffset_], RoundMode::CAST_NONE, dataCount);
        Cast(inCastLocal[expAvgSqOffset_], dataLocal[expAvgSqOffset_], RoundMode::CAST_NONE, dataCount);
        Cast(inCastLocal[gradOffset_], dataLocal[gradOffset_], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();

        // FP32 精度计算
        Muls(outCastLocal[varOffset_], inCastLocal[varOffset_], realWeightDecay_, dataCount);
        PipeBarrier<PIPE_V>();
        // ... 完整 AdamW 计算

        // Cast 回输出精度
        LocalTensor<T> dataOutLocal = outQueue_.AllocTensor<T>();
        if (isBfloat16_) {
            Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_ROUND, dataCount);
        } else {
            Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_RINT, dataCount);
        }
    }
};

// 模板化的计算类 - 混合数据类型（var/m/v: FP32, grad: FP16/BF16）
template <typename T, typename U, typename Z>
class ApplyAdamWV2MixType {
private:
    GlobalTensor<T> varGm_;   // FP32
    GlobalTensor<T> mGm_;     // FP32
    GlobalTensor<T> vGm_;     // FP32
    GlobalTensor<U> gradGm_;  // FP16 or BF16

    // 双独立队列管理
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTypeT_;  // for var/m/v (FP32)
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueTypeU_;  // for grad (FP16/BF16)
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueTypeT_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueTypeU_;

    __aicore__ inline void CopyIn(int64_t loopIdx, int32_t dataCount)
    {
        // Type T 数据（FP32）
        LocalTensor<T> dataLocalT = inQueueTypeT_.AllocTensor<T>();
        DataCopy(dataLocalT[varOffset_], varGm_[offset], dataCount);
        DataCopy(dataLocalT[expAvgOffset_], mGm_[offset], dataCount);
        DataCopy(dataLocalT[expAvgSqOffset_], vGm_[offset], dataCount);
        inQueueTypeT_.EnQue(dataLocalT);

        // Type U 数据（FP16/BF16）
        LocalTensor<U> dataLocalU = inQueueTypeU_.AllocTensor<U>();
        DataCopy(dataLocalU[gradOffset_], gradGm_[offset], dataCount);
        inQueueTypeU_.EnQue(dataLocalU);
    }

    __aicore__ inline void Compute(int32_t dataCount)
    {
        LocalTensor<T> dataLocalT = inQueueTypeT_.DeQue<T>();
        LocalTensor<U> dataLocalU = inQueueTypeU_.DeQue<U>();
        LocalTensor<T> dataOutLocalT = outQueueTypeT_.AllocTensor<T>();

        // grad Cast 到 FP32
        LocalTensor<float> gradCastLocal = castBuf_.Get<float>();
        Cast(gradCastLocal, dataLocalU[gradOffset_], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();

        // 全 FP32 计算
        Muls(dataOutLocalT[varOffset_], dataLocalT[varOffset_], realWeightDecay_, dataCount);
        PipeBarrier<PIPE_V>();
        // ... AdamW 计算使用 gradCastLocal

        outQueueTypeT_.EnQue(dataOutLocalT);
        inQueueTypeT_.FreeTensor(dataLocalT);
        inQueueTypeU_.FreeTensor(dataLocalU);
    }
};

// Kernel 入口 - 根据 Tiling Key 分发到不同模板
extern "C" __global__ __aicore__ void apply_adam_w_v2(
    GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR grad,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(101)) {  // FP32 + step(float)
        ApplyAdamWV2Fp<float, float> op;
        op.Init(var, m, v, grad, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(102)) {  // FP16 + step(float)
        ApplyAdamWV2B16<half, float> op;
        op.Init(var, m, v, grad, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(103)) {  // BF16 + step(float)
        ApplyAdamWV2B16<bfloat16_t, float> op;
        op.Init(var, m, v, grad, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(107)) {  // Mixed: var/m/v(FP32) + grad(FP16)
        ApplyAdamWV2MixType<float, half, float> op;
        op.Init(var, m, v, grad, workspace, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(108)) {  // Mixed: var/m/v(FP32) + grad(BF16)
        ApplyAdamWV2MixType<float, bfloat16_t, float> op;
        op.Init(var, m, v, grad, workspace, tilingData);
        op.Process();
    }
    // ... 其他 6 种组合
}
```

**改进点**：
1. **完整的混合精度支持**：10 种数据类型组合覆盖所有实际训练场景
2. **编译期类型分发**：Tiling Key 101-110 精确映射到不同模板实例，零运行时开销
3. **三类模板特化**：
   - `ApplyAdamWV2Fp`：FP32 全精度计算
   - `ApplyAdamWV2B16`：FP16/BF16 输入输出，FP32 中间计算
   - `ApplyAdamWV2MixType`：混合精度（优化器状态 FP32 + 梯度 FP16/BF16）
4. **独立队列管理**：混合类型场景使用双独立队列，清晰分离不同类型数据
5. **精确的舍入模式**：BF16 使用 CAST_ROUND，FP16 使用 CAST_RINT
6. **双缓冲优化**：所有模板均使用 BUFFER_NUM=2

**性能提升**：
- 混合精度模式（Tiling Key 107-110）：grad 使用 FP16/BF16 减少 50% 内存带宽，优化器状态保持 FP32 精度
- 纯低精度模式（Tiling Key 102-106）：内存带宽减半，理论性能提升 2 倍（内存受限场景）
- 实测性能提升：大模型训练场景 30-60%（取决于优化器状态占比）

**最佳实践**：
- 大模型训练推荐使用 Tiling Key 107/108（var/m/v FP32 + grad FP16/BF16）
- 小模型或推理场景可使用纯低精度模式（Tiling Key 102/103）
- step 使用 int64 还是 float 取决于训练步数规模

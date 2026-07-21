# Good Code: 模板化 FP16/FP32/BF16 全支持

来源：expert code (deep_norm)

```cpp
// OpInfo 文件 - 声明多数据类型支持
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Input("gx")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Input("beta")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

// 多平台配置 - 低功耗平台仅支持 FP16/FP32
OpAICoreConfig ascend310p_config;
ascend310p_config.Input("x")
    .DataType({ge::DT_FLOAT, ge::DT_FLOAT16})
    .Format({ge::FORMAT_ND, ge::FORMAT_ND});
this->AICore().AddConfig("ascend310p", ascend310p_config);

// Tiling 阶段 - 根据数据类型和维度大小设置 Tiling Key
// Tiling Key 编码：数据类型 + 维度范围
// fp16: 1, 5, 9, 13, 17     (对应不同维度范围)
// fp32: 2, 6, 10, 14, 18
// bf16: 3, 7, 11, 15

uint32_t isFP32 = (dtype == ge::DT_FLOAT) ? TILING_ISFP32_OFFSET : 0;
uint32_t isFP16 = (dtype == ge::DT_FLOAT16) ? TILING_ISFP16_OFFSET : 0;
uint32_t isBF16 = (dtype == ge::DT_BF16) ? TILING_ISBF16_OFFSET : 0;

// 根据维度大小选择不同处理模式
uint32_t isShort = (D <= 500) ? TILING_ISSHORT_OFFSET : 0;
uint32_t upperLimit = (D > 500 && D <= 4096) ? TILING_UPPER_LIMIT_OFFSET : 0;
uint32_t beyondLimit = (D > 4096 && D <= 8192) ? TILING_BEYOND_LIMIT_OFFSET : 0;

uint32_t dtypeKey = isShort + upperLimit + beyondLimit + isFP32 + isFP16 + isBF16;
context->SetTilingKey(dtypeKey);

// 模板化的 Kernel 类 - 支持多数据类型
template <typename T>
class KernelDeepNorm {
public:
    GlobalTensor<T> x_gm;
    GlobalTensor<T> gx_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> z_gm;

    // FP16 处理路径：先转 FP32 计算，再转回
    __aicore__ inline void ProcessFp16LELimit()
    {
        LocalTensor<T> x_local = x_queue.DeQue<T>();
        LocalTensor<T> gx_local = gx_queue.DeQue<T>();

        // 关键：FP16 输入先转 FP32 计算
        LocalTensor<float> local_y_fp32 = workspace_fp32.Get<float>();
        LocalTensor<float> local_x_fp32 = workspace_fp32_2.Get<float>();

        Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();
        Cast(local_x_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();

        // FP32 精度计算 DeepNorm
        Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();

        // 计算 mean (FP32 精度)
        float mean_local_temp = ReduceSumCustom(local_y_fp32[offset], num_last_dim);
        mean_local_temp = mean_local_temp * avgFactor_;

        // 计算 variance (FP32 精度，避免溢出)
        Muls(local_y_fp32[offset], local_y_fp32[offset], meanNum, num_last_dim);
        PipeBarrier<PIPE_V>();
        Mul(local_x_fp32, local_y_fp32, local_y_fp32, stepSize);
        PipeBarrier<PIPE_V>();
        float var_local_temp = ReduceSumCustom(local_x_fp32[offset], num_last_dim) * meanNum;

        // 归一化计算 (FP32 精度)
        float inv_std = 1.0f / sqrt(var_local_temp + eps);
        Muls(local_y_fp32, local_y_fp32, inv_std, stepSize);
        PipeBarrier<PIPE_V>();

        // 转回 FP16 输出
        LocalTensor<T> z_local = z_queue.AllocTensor<T>();
        Cast(z_local, local_y_fp32, RoundMode::CAST_NONE, stepSize);
        PipeBarrier<PIPE_V>();

        z_queue.EnQue(z_local);
    }

    // FP32 处理路径：直接计算
    __aicore__ inline void ProcessFp32LELimit()
    {
        LocalTensor<float> x_local = x_queue.DeQue<float>();
        LocalTensor<float> gx_local = gx_queue.DeQue<float>();
        LocalTensor<float> z_local = z_queue.AllocTensor<float>();

        // 直接 FP32 计算，无需 Cast
        Axpy(gx_local, x_local, alphaVal, stepSize);
        PipeBarrier<PIPE_V>();

        float mean_val = ReduceSumCustom(x_local[offset], num_last_dim) * avgFactor_;
        // ... FP32 精度计算

        z_queue.EnQue(z_local);
    }

    // BF16 处理路径：类似 FP16 但使用不同舍入模式
    __aicore__ inline void ProcessBf16LELimit()
    {
        // 类似 FP16 路径，但输出时使用 CAST_RINT
        LocalTensor<bfloat16_t> x_local = x_queue.DeQue<bfloat16_t>();
        LocalTensor<float> local_y_fp32 = workspace_fp32.Get<float>();

        Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
        // ... FP32 计算

        // BF16 特殊舍入模式
        LocalTensor<bfloat16_t> z_local = z_queue.AllocTensor<bfloat16_t>();
        Cast(z_local, local_y_fp32, RoundMode::CAST_RINT, stepSize);
    }
};

// Kernel 入口 - 根据 Tiling Key 分发
extern "C" __global__ __aicore__ void deep_norm(
    GM_ADDR x, GM_ADDR gx, GM_ADDR beta, GM_ADDR gamma,
    GM_ADDR mean, GM_ADDR rstd, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(1)) {  // FP16 + Short (D <= 500)
        KernelDeepNorm<half> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessFp16Short();
    } else if (TILING_KEY_IS(5)) {  // FP16 + LELimit (500 < D <= 4096)
        KernelDeepNorm<half> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessFp16LELimit();
    } else if (TILING_KEY_IS(2)) {  // FP32 + Short
        KernelDeepNorm<float> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessFp32Short();
    } else if (TILING_KEY_IS(6)) {  // FP32 + LELimit
        KernelDeepNorm<float> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessFp32LELimit();
    } else if (TILING_KEY_IS(3)) {  // BF16 + Short
        KernelDeepNorm<bfloat16_t> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessBf16Short();
    } else if (TILING_KEY_IS(7)) {  // BF16 + LELimit
        KernelDeepNorm<bfloat16_t> op;
        op.Init(x, gx, beta, gamma, mean, rstd, z, tilingData);
        op.ProcessBf16LELimit();
    }
    // ... 其他维度范围的 Tiling Key
}
```

**改进点**：
1. **完整的数据类型支持**：FP16/BF16/FP32 三种类型，覆盖所有 Ascend 平台
2. **模板化设计**：`template <typename T>` 实现编译期类型多态，单套代码支持多类型
3. **多平台自适应**：通过 OpAICoreConfig 为不同平台配置支持的数据类型组合
4. **FP32 中间计算**：低精度输入（FP16/BF16）在计算过程中提升为 FP32，保证数值稳定性
5. **精确的舍入控制**：
   - FP16 输出使用 `RoundMode::CAST_NONE`
   - BF16 输出使用 `RoundMode::CAST_RINT`（更适合 BF16 的表示范围）
6. **多维度 Tiling**：根据隐藏层维度 D 选择不同处理策略（Short/LELimit/GTLimit/Common）
7. **参数精确传输**：epsilon/alpha 通过 memcpy + reinterpret_cast 确保位级精确性

**性能提升**：
- FP16/BF16 输入输出：内存带宽减半，内存受限场景性能提升 50-80%
- FP32 中间计算：避免低精度累加误差，统计计算精度提升 10-100x（取决于数据分布）
- 多维度 Tiling：不同 hidden size 自动选择最优路径，性能稳定

**最佳实践**：
- Transformer 模型推荐使用 FP16（训练）或 BF16（推理）
- 隐藏层维度 D <= 500 使用 Short 路径（Tiling Key 1/2/3）
- 隐藏层维度 500 < D <= 4096 使用 LELimit 路径（Tiling Key 5/6/7）
- 大模型（D > 8192）使用 GTLimit/Common 路径（Tiling Key 13-18）
- 低功耗平台（310P）自动降级为 FP16/FP32 组合

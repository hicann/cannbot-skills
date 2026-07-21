# Base Code: 仅支持 FP32 的 DeepNorm 实现

来源：lingxi-code (deep_norm)

```cpp
// OpInfo 文件 - 仅声明 FP32
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Input("gx")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Input("beta")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Output("z")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

// Tiling 阶段 - 单一 Tiling Key
context->SetTilingKey(1);

// Kernel 类 - 固定 float 类型
class KernelDeepNorm {
private:
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> gxGm;
    AscendC::GlobalTensor<float> betaGm;
    AscendC::GlobalTensor<float> gammaGm;
    AscendC::GlobalTensor<float> outputGm;

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> gxLocal = inQueueGx.AllocTensor<float>();

        AscendC::DataCopy(xLocal, inputGm[offset], length);
        AscendC::DataCopy(gxLocal, gxGm[offset], length);

        inQueueX.EnQue(xLocal);
        inQueueGx.EnQue(gxLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> gxLocal = inQueueGx.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueue.AllocTensor<float>();

        // DeepNorm 计算: z = alpha * x + gx
        // 然后计算 mean 和 variance
        float rowMean = ComputeMean(xLocal, length);
        float rowStd = ComputeStd(xLocal, rowMean, length);

        // 归一化
        AscendC::Adds(zLocal, xLocal, -rowMean, length);
        AscendC::Muls(zLocal, zLocal, 1.0f / rowStd, length);

        outQueue.EnQue(zLocal);
    }
};
```

**问题**：
1. 仅支持 FP32，无法利用 FP16/BF16 的内存带宽优势
2. 现代 Transformer 模型大量使用 FP16/BF16，需要额外类型转换
3. 无法支持不同平台的数据类型组合（如 310P 仅支持 FP16/FP32）
4. 统计计算（mean/variance）在 FP32 下可能出现数值溢出（大 batch）
5. 缺乏针对不同数据类型的优化策略

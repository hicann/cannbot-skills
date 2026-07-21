# Base Code: 仅支持 FP32 的单一类型实现

来源：lingxi-code (adaptive_avg_pool3d)

```cpp
// OpInfo 文件 - 仅声明 FP32
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});

this->Output("y")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND})
    .UnknownShapeFormat({ge::FORMAT_ND});

// Kernel 类 - 固定 float 类型
class KernelAdaptiveAvgPool3d {
private:
    AscendC::GlobalTensor<float> inputGm;
    AscendC::GlobalTensor<float> outputGm;
    // ...

    __aicore__ inline void CopyIn(uint32_t in_offset)
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.AllocTensor<float>();
        AscendC::DataCopyPad(inputLocal, inputGm[in_offset],
                             {1, static_cast<uint16_t>(C * sizeof(float)), 0, 0},
                             {false, 0, 0, 0});
        inQueue.EnQue(inputLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> inputLocal = inQueue.DeQue<float>();
        AscendC::LocalTensor<float> accumLocal = accumBuf.Get<float>();
        AscendC::Add(accumLocal, accumLocal, inputLocal, C);
        inQueue.FreeTensor(inputLocal);
    }
};

// Tiling 阶段 - 无类型分发
context->SetTilingKey(1);  // 单一 Tiling Key
```

**问题**：
1. 仅支持 FP32，无法利用 FP16/BF16 的内存带宽优势
2. 模型使用 FP16/BF16 时需要额外的类型转换
3. 无法针对不同数据类型优化（如低精度的升精度计算）

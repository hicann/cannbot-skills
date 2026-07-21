# Base Code: 仅支持 FP32 的 ApplyAdamW 实现

来源：lingxi-code (apply_adam_w_v2)

```cpp
// OpInfo 文件 - 仅声明 FP32
this->Input("var")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Input("m")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Input("v")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

this->Input("grad")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT})
    .Format({ge::FORMAT_ND});

// Tiling 阶段 - 单一 Tiling Key
context->SetTilingKey(1);

// Kernel 类 - 固定 float 类型
class ApplyAdamWV2Kernel {
private:
    GlobalTensor<float> varGm;
    GlobalTensor<float> mGm;
    GlobalTensor<float> vGm;
    GlobalTensor<float> gradGm;

    constexpr int32_t BUFFER_NUM = 1; // 单缓冲
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue_;

    __aicore__ inline void CopyIn(int64_t offset, int64_t count)
    {
        LocalTensor<float> dataLocal = inQueue_.AllocTensor<float>();
        DataCopy(dataLocal[0], varGm[offset], count);
        DataCopy(dataLocal[count], mGm[offset], count);
        DataCopy(dataLocal[count * 2], vGm[offset], count);
        DataCopy(dataLocal[count * 3], gradGm[offset], count);
        inQueue_.EnQue(dataLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        LocalTensor<float> dataLocal = inQueue_.DeQue<float>();
        LocalTensor<float> varLocal = dataLocal[0];
        LocalTensor<float> mLocal = dataLocal[count];
        LocalTensor<float> vLocal = dataLocal[count * 2];
        LocalTensor<float> gradLocal = dataLocal[count * 3];

        // AdamW 计算（全 FP32）
        Muls(varLocal, varLocal, decayFactor, count);
        Muls(mLocal, mLocal, beta1_, count);
        Muls(gradLocal, gradLocal, (1.0f - beta1_), count);
        Add(mLocal, mLocal, gradLocal, count);
        // ... 更多计算

        inQueue_.FreeTensor(dataLocal);
    }
};
```

**问题**：
1. 仅支持 FP32，无法利用混合精度训练的优势
2. 模型使用 FP16/BF16 时，grad 输入需要额外的类型转换开销
3. 内存带宽受限场景下性能不佳（FP32 占用双倍带宽）
4. 无法支持现代大模型训练中的混合精度优化器状态（var/m/v 用 FP32，grad 用 FP16）
5. 缺乏针对不同数据类型的优化路径

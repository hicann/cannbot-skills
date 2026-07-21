# Base Code: 仅支持 FP32 的 BatchNorm 实现

来源：推断（batch_norm_v3 专家实现对比）

```cpp
// OpInfo 文件 - 仅声明 FP32
this->Input("x")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT});

this->Input("weight")
    .ParamType(REQUIRED)
    .DataType({ge::DT_FLOAT});

// 数据类型检查 - 固定 FP32
if (xDtype != ge::DT_FLOAT || weightDtype != ge::DT_FLOAT) {
    OP_LOGE("Only FP32 is supported");
    return false;
}

// Kernel 类 - 固定 float 类型
class KernelBatchNormV3 {
private:
    GlobalTensor<float> xGm;
    GlobalTensor<float> weightGm;

    __aicore__ void ComputeMean() {
        LocalTensor<float> xLocal = xQueue.DeQue<float>();
        LocalTensor<float> meanLocal = meanBuf.Get<float>();
        // 直接 FP32 计算
        ReduceSum(meanLocal, xLocal, patternR0);
    }
};
```

**问题**：
1. 仅支持 FP32，内存带宽利用率低
2. 无法支持 FP16/BF16 混合精度训练
3. 模型使用低精度时需要额外转换
